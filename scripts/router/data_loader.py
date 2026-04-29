"""
PERCEIVE Router — data loading and feature extraction.

Loads benchmark samples, eval results, routing labels, and API results,
then builds feature matrices for training/evaluation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from scripts.router.config import (
    COMPLEXITY_TIER_TO_MODEL_TIER,
    CONFIG_IDX,
    CONFIG_LIST,
    DOC_TYPES_TOP,
    KEYWORD_FLAGS,
    MODEL_IDX,
    MODEL_LIST,
    MODELS,
    N_CONFIGS,
    N_MODELS,
    QUESTION_PREFIXES,
    TASK_TYPES,
    TIER_CHEAPEST_CONFIG_IDX,
)

# ── Root directory ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]

# ── Data file paths ───────────────────────────────────────────────────────────

BENCHMARK_PATH = ROOT / "data" / "benchmark" / "benchmark_5000.jsonl"
EVAL_RESULTS_PATH = ROOT / "data" / "model_eval_results" / "final_eval_correct.jsonl"
ROUTING_LABELS_PATH = ROOT / "data" / "routing_labels" / "routing_labels.jsonl"
API_RESULTS_DIR = ROOT / "data" / "model_eval_results"
# Anchor/validation API results on HuggingFace live under anchor_evaluations/
ANCHOR_EVAL_DIR = ROOT / "data" / "anchor_evaluations"
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
ENCODER_PATHS = {
    "clip": EMBEDDINGS_DIR / "clip_vitb32.npz",
    "mobilenet": EMBEDDINGS_DIR / "mobilenetv3.npz",
}


# ── Dataset container ─────────────────────────────────────────────────────────

@dataclass
class RouterDataset:
    """Container for router training/evaluation data."""

    X: np.ndarray              # (n_samples, n_features) float32
    y_model: np.ndarray        # (n_samples,) int64 — index into MODEL_LIST, -1 if not routable
    y_config: np.ndarray       # (n_samples,) int64 — index into CONFIG_LIST, -1 if not routable
    y_vds: np.ndarray          # (n_samples,) float32 — vds_probe_avg
    y_rds: np.ndarray          # (n_samples,) float32 — rds_probe_avg
    y_ses: np.ndarray          # (n_samples,) float32 — ses_probe_avg
    eval_correct: np.ndarray   # (n_samples, 24) bool — correctness for each config
    sample_ids: list[str]
    is_routable: np.ndarray    # (n_samples,) bool
    feature_names: list[str]
    complexity_tiers: np.ndarray  # (n_samples,) int — tier_final (1/2/3)
    sample_weights: np.ndarray  # (n_samples,) float32 — 1.0 for anchor, 0.7 for remaining
    is_anchor: np.ndarray       # (n_samples,) bool — True for anchor split samples
    # Budget regression targets (for decomposed model+budget architecture)
    budget_targets: np.ndarray  # (n_samples, 7) float32 — log(reasoning_tokens+1) per model
    model_solvable: np.ndarray  # (n_samples, 7) bool — True if model has a correct budget
    # Observation mask (for IPS-weighted training on partially observed data)
    model_observed: np.ndarray  # (n_samples, 7) bool — True if any config of model was evaluated


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_features(sample: dict) -> tuple[np.ndarray, list[str]]:
    """Extract feature vector from a single benchmark sample."""
    features = []
    names = []

    features.extend([
        sample["vds_probe_avg"],
        sample["rds_probe_avg"],
        sample["ses_probe_avg"],
    ])
    names.extend(["vds_probe_avg", "rds_probe_avg", "ses_probe_avg"])

    features.append(sample["composite_est"])
    names.append("composite_est")

    task = sample["task_type"]
    for tt in TASK_TYPES:
        features.append(1.0 if task == tt else 0.0)
        names.append(f"task_{tt}")

    doc_type = sample["doc_type"]
    for dt in DOC_TYPES_TOP:
        features.append(1.0 if doc_type == dt else 0.0)
        names.append(f"doc_{dt}")
    features.append(0.0 if doc_type in DOC_TYPES_TOP else 1.0)
    names.append("doc_other")

    features.append(np.log(sample["num_pages"] + 1))
    names.append("num_pages_log")

    query = sample["query"]
    query_lower = query.lower()

    features.append(np.log(len(query) + 1))
    names.append("query_length_log")

    features.append(np.log(len(query.split()) + 1))
    names.append("query_word_count_log")

    for kw in KEYWORD_FLAGS:
        features.append(1.0 if kw in query_lower else 0.0)
        names.append(f"kw_{kw.replace(' ', '_')}")

    for prefix in QUESTION_PREFIXES:
        features.append(1.0 if query_lower.startswith(prefix) else 0.0)
        names.append(f"qprefix_{prefix}")

    features.extend([
        1.0 if sample["has_table_detected"] else 0.0,
        1.0 if sample["has_chart_detected"] else 0.0,
        1.0 if sample["has_figure_detected"] else 0.0,
        1.0 if sample["has_handwriting_detected"] else 0.0,
    ])
    names.extend([
        "has_table_detected", "has_chart_detected",
        "has_figure_detected", "has_handwriting_detected",
    ])

    features.append(np.log(sample["visual_element_count"] + 1))
    names.append("visual_element_count_log")

    features.append(np.log(sample["image_bytes_size"] + 1))
    names.append("image_bytes_size_log")

    features.append(float(sample["tier_final"]))
    names.append("tier_final")

    features.extend([
        1.0 if sample["probe_gpt52_correct"] else 0.0,
        1.0 if sample["probe_flash_correct"] else 0.0,
    ])
    names.extend(["probe_gpt52_correct", "probe_flash_correct"])

    probe_agr = sample["probe_agreement"]
    features.extend([
        1.0 if probe_agr == "both_correct" else 0.0,
        1.0 if probe_agr == "both_wrong" else 0.0,
    ])
    names.extend(["probe_agreement_both_correct", "probe_agreement_both_wrong"])

    return np.array(features, dtype=np.float32), names


# ── API results loading (for reasoning token targets) ─────────────────────────

def _load_api_results(split: str) -> dict[tuple[str, str, str], dict]:
    """Load API results for a split, returning {(sample_id, model, budget): {reasoning_tokens, total_cost_usd}}.

    Searches API_RESULTS_DIR first, then ANCHOR_EVAL_DIR as fallback (HuggingFace
    stores anchor/validation results under data/anchor_evaluations/).
    """
    results = {}
    split_files = {
        "anchor": ["api_results_anchor.jsonl"],
        "validation": ["api_results_validation.jsonl"],
        "remaining": ["api_results_remaining.jsonl"],
    }
    for fname in split_files.get(split, []):
        # Try primary location first, then anchor_evaluations/ fallback
        candidates = [API_RESULTS_DIR / fname, ANCHOR_EVAL_DIR / fname]
        fpath = next((p for p in candidates if p.exists()), None)
        if fpath is None:
            continue
        with open(fpath) as f:
            for line in f:
                r = json.loads(line)
                key = (r["sample_id"], r["yaml_key"], r["budget_level"])
                results[key] = {
                    "reasoning_tokens": r.get("reasoning_tokens", 0),
                    "total_cost_usd": r.get("total_cost_usd", 0.0),
                }
    return results


def _build_budget_targets(
    sample_ids: list[str],
    eval_correct_matrix: np.ndarray,
    api_results: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-model budget regression targets.

    For each (sample, model): find the cheapest correct budget,
    get its actual reasoning_tokens, return log(tokens + 1).

    Returns:
        budget_targets: (n_samples, 7) float32
        model_solvable: (n_samples, 7) bool
    """
    n_samples = len(sample_ids)
    budget_targets = np.zeros((n_samples, N_MODELS), dtype=np.float32)
    model_solvable = np.zeros((n_samples, N_MODELS), dtype=bool)

    budget_order = {"B0": 0, "B1": 1, "B2": 2, "B3": 3}

    for i, sid in enumerate(sample_ids):
        for m_idx, yk in enumerate(MODEL_LIST):
            valid_budgets = sorted(MODELS[yk].valid_budgets, key=lambda b: budget_order[b])
            for bl in valid_budgets:
                config_idx = CONFIG_IDX.get((yk, bl))
                if config_idx is not None and eval_correct_matrix[i, config_idx]:
                    api_data = api_results.get((sid, yk, bl), {})
                    rt = api_data.get("reasoning_tokens", 0)
                    budget_targets[i, m_idx] = np.log(rt + 1)
                    model_solvable[i, m_idx] = True
                    break

    return budget_targets, model_solvable


def _build_observation_mask(
    sample_ids: list[str],
    api_results: dict,
    eval_results: dict | None = None,
) -> np.ndarray:
    """Build per-model observation mask: True if any config of that model was evaluated.

    Uses api_results (explicit call logs) when available. Falls back to eval_results
    (final_eval_correct.jsonl entries) for samples with no api_results coverage — this
    handles remaining-split samples when api_results_remaining.jsonl is absent, deriving
    observation from the fact that an entry exists in the eval matrix at all.
    """
    n_samples = len(sample_ids)
    model_observed = np.zeros((n_samples, N_MODELS), dtype=bool)
    for i, sid in enumerate(sample_ids):
        for m_idx, yk in enumerate(MODEL_LIST):
            for bl in MODELS[yk].valid_budgets:
                if (sid, yk, bl) in api_results:
                    model_observed[i, m_idx] = True
                    break
                elif (
                    eval_results is not None
                    and sid in eval_results
                    and (yk, bl) in eval_results[sid]
                ):
                    # Entry exists in final_eval_correct.jsonl → config was called
                    model_observed[i, m_idx] = True
                    break
    return model_observed


# ── Image embedding loading ──────────────────────────────────────────────────

def _load_image_embeddings(encoder: str | None) -> dict[str, np.ndarray] | None:
    """Load image embeddings for a given encoder. Returns {sample_id: (dim,) float32} or None."""
    if encoder is None:
        return None
    path = ENCODER_PATHS.get(encoder)
    if path is None:
        raise ValueError(f"Unknown encoder '{encoder}'. Available: {list(ENCODER_PATHS)}")
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {path}\n"
            f"Download from HuggingFace:\n"
            f"  python -c \"from huggingface_hub import hf_hub_download; hf_hub_download("
            f"'quantiphi-routing/perceive-benchmark', 'data/embeddings/{path.name}', repo_type='dataset')\"\n"
            f"To use text-only features instead (expect ~6pp accuracy drop), pass encoder=None."
        )
    data = np.load(path, allow_pickle=False)
    sample_ids = data["sample_ids"]
    embeddings = data["embeddings"]
    return {sid: embeddings[i] for i, sid in enumerate(sample_ids)}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(split: str, encoder: str | None = None) -> RouterDataset:
    """Load and featurize a split. Supports 'anchor', 'validation', 'remaining', or combined like 'anchor+remaining'.

    Args:
        split: Data split(s) to load, e.g. 'anchor', 'anchor+remaining'.
        encoder: Image encoder embeddings to concatenate ('clip', 'mobilenet', or None for text-only).
            Default is None (text-only, 48 features). Controlled ablation shows CLIP/MobileNet
            reduce accuracy by ~2pp at the 1,500-sample anchor scale due to redundancy with
            probe features + overfitting. The paper-reported 61.6% uses text-only features.
            Embeddings at data/embeddings/ are provided for reproducibility research.
    """
    sub_splits = [s.strip() for s in split.split("+")]
    benchmark = {}
    with open(BENCHMARK_PATH) as f:
        for line in f:
            sample = json.loads(line)
            benchmark[sample["sample_id"]] = sample

    eval_results: dict[str, dict[tuple[str, str], bool]] = {}
    with open(EVAL_RESULTS_PATH) as f:
        for line in f:
            rec = json.loads(line)
            sid = rec["sample_id"]
            config = (rec["yaml_key"], rec["budget_level"])
            if sid not in eval_results:
                eval_results[sid] = {}
            eval_results[sid][config] = rec["eval_correct"]

    routing_labels = {}
    with open(ROUTING_LABELS_PATH) as f:
        for line in f:
            rec = json.loads(line)
            routing_labels[rec["sample_id"]] = rec

    sample_ids = sorted(
        sid for sid, rec in routing_labels.items() if rec["split"] in sub_splits
    )

    X_list = []
    y_model_list = []
    y_config_list = []
    y_vds_list = []
    y_rds_list = []
    y_ses_list = []
    eval_correct_list = []
    is_routable_list = []
    complexity_tier_list = []
    sample_weight_list = []
    is_anchor_list = []
    feature_names = None

    for sid in sample_ids:
        sample = benchmark[sid]
        label = routing_labels[sid]
        is_anchor_sample = label["split"] == "anchor"
        sample_weight_list.append(1.0 if is_anchor_sample else 0.7)
        is_anchor_list.append(is_anchor_sample)

        features, names = _extract_features(sample)
        if feature_names is None:
            feature_names = names
        X_list.append(features)

        y_vds_list.append(sample["vds_probe_avg"])
        y_rds_list.append(sample["rds_probe_avg"])
        y_ses_list.append(sample["ses_probe_avg"])

        is_routable = label["is_routable"]
        is_routable_list.append(is_routable)

        tier_final = sample.get("tier_final", 1)
        complexity_tier_list.append(tier_final)

        if is_routable:
            cheapest_model = label["cheapest_correct_model"]
            cheapest_budget = label["cheapest_correct_budget"]
            y_model_list.append(MODEL_IDX[cheapest_model])
            y_config_list.append(CONFIG_IDX[(cheapest_model, cheapest_budget)])
        else:
            model_tier = COMPLEXITY_TIER_TO_MODEL_TIER.get(tier_final, "A")
            tier_config_idx = TIER_CHEAPEST_CONFIG_IDX[model_tier]
            tier_model_key = CONFIG_LIST[tier_config_idx][0]
            y_model_list.append(MODEL_IDX[tier_model_key])
            y_config_list.append(tier_config_idx)

        eval_row = []
        for config in CONFIG_LIST:
            correct = eval_results.get(sid, {}).get(config, False)
            eval_row.append(correct)
        eval_correct_list.append(eval_row)

    X = np.array(X_list, dtype=np.float32)
    y_model = np.array(y_model_list, dtype=np.int64)
    y_config = np.array(y_config_list, dtype=np.int64)
    y_vds = np.array(y_vds_list, dtype=np.float32)
    y_rds = np.array(y_rds_list, dtype=np.float32)
    y_ses = np.array(y_ses_list, dtype=np.float32)
    eval_correct = np.array(eval_correct_list, dtype=bool)
    is_routable = np.array(is_routable_list, dtype=bool)
    complexity_tiers = np.array(complexity_tier_list, dtype=np.int64)
    sample_weights = np.array(sample_weight_list, dtype=np.float32)
    is_anchor = np.array(is_anchor_list, dtype=bool)

    # Budget regression targets (merge API results from all sub-splits)
    api_results: dict = {}
    for s in sub_splits:
        api_results.update(_load_api_results(s))
    budget_targets, model_solvable = _build_budget_targets(
        sample_ids, eval_correct, api_results
    )
    model_observed = _build_observation_mask(sample_ids, api_results, eval_results)

    # Image embedding concatenation (if encoder specified and file exists)
    image_embeddings = _load_image_embeddings(encoder)
    if image_embeddings is not None:
        emb_dim = next(iter(image_embeddings.values())).shape[0]
        emb_matrix = np.zeros((len(sample_ids), emb_dim), dtype=np.float32)
        for i, sid in enumerate(sample_ids):
            if sid in image_embeddings:
                emb_matrix[i] = image_embeddings[sid]
        X = np.concatenate([X, emb_matrix], axis=1)
        feature_names = feature_names + [f"{encoder}_{j}" for j in range(emb_dim)]

    return RouterDataset(
        X=X,
        y_model=y_model,
        y_config=y_config,
        y_vds=y_vds,
        y_rds=y_rds,
        y_ses=y_ses,
        eval_correct=eval_correct,
        sample_ids=sample_ids,
        is_routable=is_routable,
        feature_names=feature_names,
        complexity_tiers=complexity_tiers,
        sample_weights=sample_weights,
        is_anchor=is_anchor,
        budget_targets=budget_targets,
        model_solvable=model_solvable,
        model_observed=model_observed,
    )
