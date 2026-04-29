#!/usr/bin/env python3
"""
DocRouteBench Phase 2 — Stratified Sampler

Selects the final 5,000 benchmark samples from a ~25K candidate pool using
greedy coverage with a confidence × diversity priority score.

Inputs:
    data/processed/difficulty_scores.jsonl   — scored candidates
    data/processed/samples_with_prior.jsonl  — full sample metadata

Outputs:
    data/benchmark/benchmark_5000_ids.json   — list of selected sample_ids
    data/benchmark/benchmark_5000.jsonl      — full sample dicts for selected samples
    data/benchmark/sampling_report.json      — distribution statistics

Usage:
    python -m src.sampling.stratified_sampler              # full run (5000 samples)
    python -m src.sampling.stratified_sampler --water-test # smoke test (30 samples)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ── Project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.schema import load_jsonl, TASK_TYPES

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("stratified_sampler")

# ── Constants ─────────────────────────────────────────────────────────────────

ANSWER_TYPES = ["numeric", "extractive", "boolean", "abstractive"]
MIN_DATASET_SAMPLES = 10       # hard minimum per dataset (relaxed during water-test)
DIVERSITY_FEATURE_KEYS = [     # categorical features for diversity scoring
    "task_type",
    "source_dataset",
    "doc_type_detected",            # probe-detected per-sample (replaces hardcoded doc_type)
    "has_table_detected",           # probe-detected (replaces hardcoded has_table)
    "has_chart_detected",           # probe-detected (replaces hardcoded has_chart)
    "has_figure_detected",          # probe-detected (replaces hardcoded has_figure)
    "has_handwriting_detected",     # probe-detected (replaces hardcoded has_handwriting)
    "visual_element_count_bin",     # binned: 0, 1, 2, 3+
    "answer_type_inferred",         # numeric, extractive, boolean, abstractive
    "probe_complexity_bin",         # low / medium / high
    "probe_agreement",              # both_correct, both_wrong, disagree, missing
]


# ── Answer type inference ─────────────────────────────────────────────────────

def _infer_answer_type(sample: dict) -> str:
    """
    Infer answer type from sample fields when not explicitly annotated.

    Heuristic priority:
      1. Use explicit 'answer_type' field if present.
      2. Use correctness_metric as a proxy.
      3. Use gt_answer content (yes/no → boolean, numeric string, etc.).
    """
    if sample.get("answer_type"):
        return sample["answer_type"]

    metric = sample.get("correctness_metric", "")
    if metric in ("exact_match", "denotation"):
        gt = str(sample.get("gt_answer", "")).strip().lower()
        if gt in ("yes", "no", "true", "false"):
            return "boolean"
        try:
            float(gt.replace(",", "").replace("%", ""))
            return "numeric"
        except ValueError:
            pass

    if metric in ("anls", "vqa_accuracy", "slidevqa_em", "rouge_cider"):
        gt = str(sample.get("gt_answer", "")).strip().lower()
        if gt in ("yes", "no", "true", "false"):
            return "boolean"
        try:
            float(gt.replace(",", "").replace("%", ""))
            return "numeric"
        except ValueError:
            pass
        if len(gt.split()) > 5:
            return "abstractive"
        return "extractive"

    if metric in ("field_f1", "teds"):
        return "extractive"

    return "extractive"


def _prepare_enriched_features(sample: dict) -> dict:
    """
    Compute binned/derived features for the enriched diversity vector.
    Falls back to adapter flags if probe-detected fields are missing.
    """
    s = dict(sample)

    # Visual element count binning: 0, 1, 2, 3+
    vec = s.get("visual_element_count", 0)
    s["visual_element_count_bin"] = min(int(vec), 3)

    # Answer type
    s["answer_type_inferred"] = _infer_answer_type(s)

    # Probe complexity bin from probe VDS/RDS/SES averages
    vds = s.get("vds_probe_avg", s.get("vds_est", 2))
    rds = s.get("rds_probe_avg", s.get("rds_est", 2))
    ses = s.get("ses_probe_avg", s.get("ses_est", 2))
    composite = 0.30 * float(vds) + 0.45 * float(rds) + 0.25 * float(ses)
    if composite < 2.0:
        s["probe_complexity_bin"] = "low"
    elif composite < 3.0:
        s["probe_complexity_bin"] = "medium"
    else:
        s["probe_complexity_bin"] = "high"

    # Probe agreement (may already be set by difficulty estimator)
    if "probe_agreement" not in s:
        g = s.get("probe_gpt52_correct")
        f = s.get("probe_flash_correct")
        if g is None or f is None:
            s["probe_agreement"] = "missing"
        elif g == f:
            s["probe_agreement"] = "both_correct" if g else "both_wrong"
        else:
            s["probe_agreement"] = "disagree"

    # Fallback: if probe-detected flags are missing, use adapter flags
    for flag in ("has_table", "has_chart", "has_figure", "has_handwriting"):
        detected_key = f"{flag}_detected"
        if detected_key not in s or s[detected_key] is None:
            s[detected_key] = s.get(flag, False)

    # Fallback for doc_type_detected
    if "doc_type_detected" not in s or not s["doc_type_detected"]:
        s["doc_type_detected"] = s.get("doc_type", "other")

    return s


# ── Feature vector for diversity ──────────────────────────────────────────────

def _feature_vector(sample: dict) -> tuple:
    """
    Build a hashable feature tuple for diversity scoring.
    Prepares enriched features first, falling back to adapter flags.
    """
    s = _prepare_enriched_features(sample)
    return tuple(
        int(s.get(k, False)) if isinstance(s.get(k), bool)
        else (s.get(k) or "")
        for k in DIVERSITY_FEATURE_KEYS
    )


def _hamming_distance(a: tuple, b: tuple) -> int:
    """Count positions where feature vectors differ."""
    return sum(x != y for x, y in zip(a, b))


def _build_feature_matrix(vecs: list[tuple]) -> "np.ndarray":
    """Convert list of feature tuples to numpy int8 matrix for fast ops."""
    import numpy as np
    # Map categorical string features to int (by hash mod 256)
    rows = []
    for v in vecs:
        row = []
        for x in v:
            if isinstance(x, bool):
                row.append(int(x))
            elif isinstance(x, (int, float)):
                row.append(int(x) % 256)
            else:
                row.append(hash(x) % 256)
        rows.append(row)
    return np.array(rows, dtype=np.int16)


def _greedy_diverse_select_numpy(
    candidates: list[dict],
    vecs: dict,
    priority: dict,
    quota: int,
    seeded_ids: set,
) -> list[dict]:
    """
    Fast numpy-based greedy diversity selection.
    O(quota × n_candidates) with numpy broadcasting instead of Python loops.
    """
    import numpy as np

    remaining = [s for s in candidates if s["sample_id"] not in seeded_ids]
    if not remaining:
        return []

    ids = [s["sample_id"] for s in remaining]
    sample_map = {s["sample_id"]: s for s in remaining}

    all_vecs = [vecs[sid] for sid in ids]
    mat = _build_feature_matrix(all_vecs)          # (n_remaining, n_features)
    n_features = mat.shape[1] if mat.ndim > 1 else 1
    n_remaining = len(ids)

    selected_idx = []
    # min_dist[i] = min Hamming distance from candidate i to any selected candidate
    # Initialized to n_features (max diversity = fully diverse from empty set)
    min_dist = np.full(n_remaining, n_features, dtype=np.float32)

    quota = min(quota, n_remaining)

    for _ in range(quota):
        if not selected_idx:
            # First pick: use priority only
            pri_arr = np.array([priority.get(ids[i], 0.5) for i in range(n_remaining)], dtype=np.float32)
            best = int(np.argmax(pri_arr))
        else:
            # Update min_dist with distances from last selected
            last_vec = mat[selected_idx[-1]]                  # (n_features,)
            dists = (mat != last_vec).sum(axis=1).astype(np.float32)  # (n_remaining,)
            np.minimum(min_dist, dists, out=min_dist)

            # Score = priority × (normalized min distance)
            pri_arr = np.array([priority.get(ids[i], 0.5) for i in range(n_remaining)], dtype=np.float32)
            norm_div = min_dist / n_features                  # in [0, 1]
            scores = pri_arr * (0.5 + 0.5 * norm_div)        # blend priority + diversity

            # Zero out already selected
            for idx in selected_idx:
                scores[idx] = -1.0
            best = int(np.argmax(scores))

        selected_idx.append(best)

    return [sample_map[ids[i]] for i in selected_idx]


# ── Diversity score against already-selected set ──────────────────────────────

def _diversity_score(
    candidate_vec: tuple,
    selected_vecs: list[tuple],
    n_features: int,
) -> float:
    """
    Diversity of a candidate relative to already-selected samples.
    Uses mean normalised Hamming distance across the selected pool.
    """
    if not selected_vecs:
        return 1.0
    if n_features == 0:
        return 0.0
    total_dist = sum(_hamming_distance(candidate_vec, sv) for sv in selected_vecs)
    mean_dist = total_dist / len(selected_vecs)
    return min(1.0, mean_dist / n_features)


# ── Coverage constraints check ────────────────────────────────────────────────

def _check_coverage(
    selected: list[dict],
    tier_target: dict[int, int],
    min_dataset_samples: int,
) -> dict:
    """
    Return coverage statistics for the current selection.
    """
    task_counts: dict[str, int] = defaultdict(int)
    dataset_counts: dict[str, int] = defaultdict(int)
    tier_counts: dict[int, int] = defaultdict(int)
    answer_type_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for s in selected:
        tt = s.get("task_type", "?")
        ds = s.get("source_dataset", "?")
        tier = s.get("tier_final", s.get("tier_prior", 2))
        at = _infer_answer_type(s)

        task_counts[tt] += 1
        dataset_counts[ds] += 1
        tier_counts[int(tier)] += 1
        answer_type_counts[tt][at] += 1

    return {
        "task_counts":        dict(task_counts),
        "dataset_counts":     dict(dataset_counts),
        "tier_counts":        dict(tier_counts),
        "answer_type_counts": {k: dict(v) for k, v in answer_type_counts.items()},
        "task_types_covered": sorted(task_counts.keys()),
        "datasets_covered":   sorted(dataset_counts.keys()),
        "min_dataset_ok":     all(
            v >= min_dataset_samples for v in dataset_counts.values()
        ),
    }


# ── Greedy selection within a single tier ─────────────────────────────────────

def _select_tier(
    candidates: list[dict],
    quota: int,
    task_types: list[str],
    min_dataset_samples: int,
    max_greedy_pool: int = 5000,
    rng_seed: int = 42,
) -> list[dict]:
    """
    Greedy selection of `quota` samples from `candidates` (all same tier).

    Priority score = confidence × diversity_score.
    Diversity is computed against already-selected samples in this tier.

    Performance: if len(candidates) > max_greedy_pool, pre-subsample to
    max_greedy_pool using stratified random sampling before running greedy.
    This reduces O(n^2 * k) to O(max_greedy_pool^2 * k) with negligible
    quality loss when max_greedy_pool >> quota.

    Guarantees:
      - All task_types in `task_types` that exist in candidates are represented.
      - Each dataset represented by at least min_dataset_samples (best-effort;
        downgraded to 1 if quota is too small).

    Returns list of selected sample dicts.
    """
    import random as _random
    if not candidates:
        return []

    quota = min(quota, len(candidates))
    n_features = len(DIVERSITY_FEATURE_KEYS)

    # Pre-subsample for performance: greedy is O(pool² × quota), so cap pool.
    # Keep at least 4× the quota to maintain diversity quality.
    effective_pool_size = max(max_greedy_pool, quota * 4)
    if len(candidates) > effective_pool_size:
        rng = _random.Random(rng_seed)
        # Stratified subsample: preserve dataset and task_type distribution
        by_group: dict = {}
        for s in candidates:
            key = (s.get("source_dataset", ""), s.get("task_type", ""))
            by_group.setdefault(key, []).append(s)
        subsampled: list[dict] = []
        ratio = effective_pool_size / len(candidates)
        for group in by_group.values():
            n = max(1, round(len(group) * ratio))
            subsampled.extend(rng.sample(group, min(n, len(group))))
        # Top-up if rounding left us short
        remaining_ids = {s["sample_id"] for s in subsampled}
        extras = [s for s in candidates if s["sample_id"] not in remaining_ids]
        if len(subsampled) < effective_pool_size and extras:
            fill = rng.sample(extras, min(effective_pool_size - len(subsampled), len(extras)))
            subsampled.extend(fill)
        candidates = subsampled
        log.info(
            "Pre-subsampled %d → %d candidates for greedy (quota=%d).",
            len(candidates) + len(extras), len(candidates), quota,
        )

    # Precompute feature vectors
    vecs = {s["sample_id"]: _feature_vector(s) for s in candidates}

    selected: list[dict] = []
    selected_vecs: list[tuple] = []
    selected_ids: set[str] = set()

    # Phase 1: Seed — ensure each task_type gets at least one sample
    # (pick highest-confidence sample per task_type first)
    tasks_present = {s.get("task_type") for s in candidates}
    for tt in task_types:
        if tt not in tasks_present:
            continue
        if len(selected) >= quota:
            break
        tt_cands = [s for s in candidates if s.get("task_type") == tt]
        if not tt_cands:
            continue
        # Best seed = highest confidence
        best = max(tt_cands, key=lambda s: (s.get("confidence", 0.0), s.get("difficulty_score", 0.0)))
        selected.append(best)
        selected_vecs.append(vecs[best["sample_id"]])
        selected_ids.add(best["sample_id"])

    # Phase 2: Fast numpy greedy fill to quota
    still_needed = quota - len(selected)
    if still_needed > 0:
        priority = {
            s["sample_id"]: s.get("confidence", 0.5) * (0.5 + 0.5 * s.get("difficulty_score", 0.5))
            for s in candidates
        }
        numpy_selected = _greedy_diverse_select_numpy(
            candidates, vecs, priority, still_needed, selected_ids
        )
        selected.extend(numpy_selected)
        selected_ids.update(s["sample_id"] for s in numpy_selected)

    return selected


# ── Build sampling report ─────────────────────────────────────────────────────

def _build_report(
    selected: list[dict],
    total_target: int,
    tier_targets: dict[int, int],
    min_dataset_samples: int,
) -> dict:
    """Build the sampling_report.json content."""
    coverage = _check_coverage(selected, tier_targets, min_dataset_samples)
    tier_counts = coverage["tier_counts"]
    total_selected = len(selected)

    # Serialise tier keys as strings so they survive JSON round-trip consistently.
    tier_counts_str  = {str(k): v for k, v in tier_counts.items()}
    tier_targets_str = {str(k): v for k, v in tier_targets.items()}

    return {
        "total_selected": total_selected,
        "total_target":   total_target,
        "tier_targets":   tier_targets_str,
        "tier_counts":    tier_counts_str,
        "tier_pct": {
            str(t): round(tier_counts_str.get(str(t), 0) / total_selected, 4) if total_selected else 0.0
            for t in [1, 2, 3]
        },
        "task_counts":       coverage["task_counts"],
        "task_types_covered": coverage["task_types_covered"],
        "dataset_counts":    coverage["dataset_counts"],
        "datasets_covered":  coverage["datasets_covered"],
        "n_datasets":        len(coverage["datasets_covered"]),
        "min_dataset_ok":    coverage["min_dataset_ok"],
        "answer_type_counts": coverage["answer_type_counts"],
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_stratified_sampling(
    scores_path: str = "data/processed/difficulty_scores.jsonl",
    all_samples_path: str = "data/processed/samples_with_prior.jsonl",
    output_dir: str = "data/benchmark",
    checkpoint_dir: str = "data/phase2_checkpoints",
    total_samples: int = 5000,
    tier_weights: tuple = (0.25, 0.50, 0.25),
    overwrite: bool = False,
) -> str:
    """
    Select benchmark samples using greedy coverage with priority scoring.

    Args:
        scores_path:      JSONL with difficulty_score and tier_final per sample.
        all_samples_path: JSONL with full sample metadata (used if scores are partial).
        output_dir:       Directory for output files.
        checkpoint_dir:   Directory for checkpoint file.
        total_samples:    Total number of samples to select (default 5000).
        tier_weights:     Fractional target for (T1, T2, T3); must sum to 1.0.
        overwrite:        If False, skip if checkpoint already exists.

    Returns:
        Absolute path to benchmark_5000.jsonl (or the appropriately named file).
    """
    # ── Path resolution ───────────────────────────────────────────────────────
    def _abs(p: str) -> str:
        path = Path(p)
        return str(path if path.is_absolute() else _PROJECT_ROOT / path)

    scores_path      = _abs(scores_path)
    all_samples_path = _abs(all_samples_path)
    output_dir_path  = Path(_abs(output_dir))
    checkpoint_dir_p = Path(_abs(checkpoint_dir))

    output_ids_path    = output_dir_path / "benchmark_5000_ids.json"
    output_jsonl_path  = output_dir_path / "benchmark_5000.jsonl"
    output_report_path = output_dir_path / "sampling_report.json"
    checkpoint_file    = checkpoint_dir_p / "stratified_sampler.done"

    # ── Checkpoint guard ──────────────────────────────────────────────────────
    if checkpoint_file.exists() and not overwrite:
        log.info(
            "Checkpoint found: %s — skipping (use --overwrite to re-run).",
            checkpoint_file,
        )
        return str(output_jsonl_path)

    # ── Tier quotas ───────────────────────────────────────────────────────────
    assert abs(sum(tier_weights) - 1.0) < 1e-6, "tier_weights must sum to 1.0"
    tier_targets: dict[int, int] = {}
    allocated = 0
    for i, w in enumerate(tier_weights):
        tier = i + 1
        q = int(total_samples * w)
        tier_targets[tier] = q
        allocated += q
    # Assign remainder to T2 (middle tier)
    tier_targets[2] += total_samples - allocated

    log.info(
        "Tier targets: T1=%d, T2=%d, T3=%d (total=%d)",
        tier_targets[1], tier_targets[2], tier_targets[3], total_samples,
    )

    # ── Load scored candidates ────────────────────────────────────────────────
    log.info("Loading difficulty scores from %s …", scores_path)
    try:
        scored = load_jsonl(scores_path)
    except FileNotFoundError:
        log.error(
            "Difficulty scores file not found: %s\n"
            "Run difficulty_estimator.py first.",
            scores_path,
        )
        raise

    # Filter out non-sample rows and build index.
    # Only include probed samples (probe_agreement != "missing") to ensure
    # all benchmark entries have empirical difficulty validation.
    candidates_by_id: dict[str, dict] = {}
    skipped_unprobed = 0
    for row in scored:
        sid = row.get("sample_id")
        if not sid:
            continue
        if row.get("probe_agreement") == "missing":
            skipped_unprobed += 1
            continue
        candidates_by_id[sid] = row

    log.info(
        "Loaded %d probed candidates (skipped %d unprobed).",
        len(candidates_by_id), skipped_unprobed,
    )

    # ── Optionally merge with full metadata ───────────────────────────────────
    # If scores don't carry all Sample fields, merge from all_samples_path.
    has_full_metadata = any(
        "source_dataset" in v for v in list(candidates_by_id.values())[:5]
    )
    if not has_full_metadata:
        log.info(
            "Scores lack full metadata — merging from %s …", all_samples_path
        )
        try:
            for row in load_jsonl(all_samples_path):
                sid = row.get("sample_id")
                if sid and sid in candidates_by_id:
                    merged = dict(row)
                    merged.update(candidates_by_id[sid])
                    candidates_by_id[sid] = merged
        except FileNotFoundError:
            log.warning("all_samples_path not found; proceeding with scores only.")

    candidates = list(candidates_by_id.values())

    # ── Determine effective min_dataset_samples ───────────────────────────────
    n_datasets = len({s.get("source_dataset", "") for s in candidates})
    min_ds = min(MIN_DATASET_SAMPLES, max(1, total_samples // max(n_datasets, 1)))
    log.info(
        "Effective min_dataset_samples=%d for %d datasets.", min_ds, n_datasets
    )

    # ── Split candidates by tier ──────────────────────────────────────────────
    by_tier: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    for s in candidates:
        tier = int(s.get("tier_final", s.get("tier_prior", 2)))
        if tier not in by_tier:
            tier = 2
        by_tier[tier].append(s)

    for t in [1, 2, 3]:
        log.info("  Tier %d: %d candidates available.", t, len(by_tier[t]))

    # ── Load task-type budgets from datasets.yaml for proportional allocation ──
    import yaml
    datasets_yaml = _PROJECT_ROOT / "configs" / "datasets.yaml"
    task_type_budgets: dict[str, int] = defaultdict(int)
    if datasets_yaml.exists():
        with open(datasets_yaml) as _f:
            _cfg = yaml.safe_load(_f)
        for _ds_meta in (_cfg.get("datasets") or {}).values():
            _tt = _ds_meta.get("task_type", "")
            _budget = _ds_meta.get("sample_budget", 0)
            if "+" in _tt:
                # Split budget (e.g. "T3+T6" with task_type_split)
                _split = _ds_meta.get("task_type_split", {})
                for _sub_tt, _sub_budget in _split.items():
                    task_type_budgets[_sub_tt] += int(_sub_budget)
            else:
                task_type_budgets[_tt] += int(_budget)
        log.info("Task-type budgets from datasets.yaml: %s", dict(sorted(task_type_budgets.items())))

    # ── Greedy selection per tier with task-type-aware allocation ──────────────
    all_selected: list[dict] = []
    for tier in [1, 2, 3]:
        tier_quota = tier_targets[tier]
        pool = by_tier[tier]

        if not pool:
            log.warning("Tier %d has no candidates — skipping.", tier)
            continue

        if len(pool) < tier_quota:
            log.warning(
                "Tier %d: only %d candidates available for quota=%d; taking all.",
                tier, len(pool), tier_quota,
            )
            tier_quota = len(pool)

        # Split pool by task type
        pool_by_task: dict[str, list[dict]] = defaultdict(list)
        for s in pool:
            pool_by_task[s.get("task_type", "T4")].append(s)

        # Allocate per-task quotas proportional to dataset budgets,
        # with a minimum floor of min(10, available) per task type
        task_types_present = sorted(pool_by_task.keys())
        total_budget = sum(task_type_budgets.get(tt, 100) for tt in task_types_present)
        task_quotas: dict[str, int] = {}
        allocated_in_tier = 0

        for i, tt in enumerate(task_types_present):
            available = len(pool_by_task[tt])
            if i == len(task_types_present) - 1:
                # Last task type gets remainder
                raw_quota = tier_quota - allocated_in_tier
            else:
                budget_share = task_type_budgets.get(tt, 100) / max(total_budget, 1)
                raw_quota = round(tier_quota * budget_share)
            # Enforce minimum floor but don't exceed available
            floor = min(10, available)
            quota = max(floor, min(raw_quota, available))
            task_quotas[tt] = quota
            allocated_in_tier += quota

        # If over-allocated (due to floors), scale down proportionally
        if allocated_in_tier > tier_quota:
            scale = tier_quota / allocated_in_tier
            adjusted = {}
            remaining = tier_quota
            for i, (tt, q) in enumerate(sorted(task_quotas.items())):
                if i == len(task_quotas) - 1:
                    adjusted[tt] = remaining
                else:
                    adj = max(1, round(q * scale))
                    adjusted[tt] = min(adj, len(pool_by_task[tt]))
                    remaining -= adjusted[tt]
            task_quotas = adjusted

        log.info(
            "Tier %d: task-type quotas: %s (tier total=%d)",
            tier, dict(sorted(task_quotas.items())), tier_quota,
        )

        # Select within each task type using greedy diversity
        selected_tier: list[dict] = []
        for tt in task_types_present:
            tt_pool = pool_by_task[tt]
            tt_quota = task_quotas.get(tt, 0)
            if tt_quota <= 0 or not tt_pool:
                continue
            tt_selected = _select_tier(tt_pool, tt_quota, [tt], min_ds)
            selected_tier.extend(tt_selected)
            log.info(
                "    Tier %d / %s: selected %d / %d (pool=%d)",
                tier, tt, len(tt_selected), tt_quota, len(tt_pool),
            )

        log.info("  Selected %d samples from Tier %d.", len(selected_tier), tier)
        all_selected.extend(selected_tier)

    log.info("Total selected: %d / %d target.", len(all_selected), total_samples)

    # ── Per-dataset minimum enforcement ──────────────────────────────────────
    # Ensure every dataset in the candidate pool has at least
    # MIN_PER_DATASET samples in the benchmark. If a dataset is below the
    # floor, swap in its candidates by replacing excess samples from the
    # most overrepresented dataset (largest surplus above its proportional
    # share). This preserves total_samples exactly.
    MIN_PER_DATASET = 50

    ds_counts: dict[str, int] = defaultdict(int)
    for s in all_selected:
        ds_counts[s.get("source_dataset", "?")] += 1

    # All datasets in the full candidate pool
    all_candidate_ds = {s.get("source_dataset", "?") for s in candidates}
    selected_ids = {s["sample_id"] for s in all_selected}

    deficit_datasets = []
    for ds in sorted(all_candidate_ds):
        current = ds_counts.get(ds, 0)
        available = sum(1 for s in candidates if s.get("source_dataset") == ds and s["sample_id"] not in selected_ids)
        if current < MIN_PER_DATASET and (current + available) > 0:
            need = min(MIN_PER_DATASET - current, available)
            if need > 0:
                deficit_datasets.append((ds, current, need))

    if deficit_datasets:
        log.info("Per-dataset minimum enforcement (floor=%d):", MIN_PER_DATASET)
        total_need = sum(need for _, _, need in deficit_datasets)

        # Identify donor datasets: those with the most surplus above proportional share
        total_budget = sum(task_type_budgets.values()) or total_samples
        for ds, current, need in deficit_datasets:
            # Collect unselected candidates from this dataset, sorted by confidence
            ds_candidates = [
                s for s in candidates
                if s.get("source_dataset") == ds and s["sample_id"] not in selected_ids
            ]
            ds_candidates.sort(key=lambda s: s.get("confidence", 0.5), reverse=True)
            additions = ds_candidates[:need]

            # Find donor: dataset with most samples in current selection
            donor_ds = max(ds_counts, key=ds_counts.get)
            # Remove `need` samples from donor (lowest confidence first)
            donor_in_selection = [
                (i, s) for i, s in enumerate(all_selected)
                if s.get("source_dataset") == donor_ds
            ]
            donor_in_selection.sort(key=lambda x: x[1].get("confidence", 0.5))
            remove_indices = [idx for idx, _ in donor_in_selection[:need]]

            # Swap
            for idx in sorted(remove_indices, reverse=True):
                removed = all_selected.pop(idx)
                ds_counts[removed.get("source_dataset", "?")] -= 1

            for s in additions:
                all_selected.append(s)
                selected_ids.add(s["sample_id"])
                ds_counts[s.get("source_dataset", "?")] += 1

            log.info(
                "    %s: %d → %d (added %d, donor=%s now %d)",
                ds, current, ds_counts[ds], len(additions), donor_ds, ds_counts[donor_ds],
            )

        log.info("After enforcement: %d total samples.", len(all_selected))

    # Log final per-dataset counts
    final_ds_counts = defaultdict(int)
    for s in all_selected:
        final_ds_counts[s.get("source_dataset", "?")] += 1
    log.info("Final dataset distribution: %s", dict(sorted(final_ds_counts.items())))

    # ── Write outputs ─────────────────────────────────────────────────────────
    output_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir_p.mkdir(parents=True, exist_ok=True)

    # benchmark_5000_ids.json
    selected_ids = [s["sample_id"] for s in all_selected]
    output_ids_path.write_text(json.dumps(selected_ids, indent=2) + "\n")
    log.info("Wrote %d sample IDs to %s", len(selected_ids), output_ids_path)

    # benchmark_5000.jsonl
    with open(output_jsonl_path, "w") as f:
        for s in all_selected:
            f.write(json.dumps(s, default=str) + "\n")
    log.info("Wrote benchmark JSONL to %s", output_jsonl_path)

    # sampling_report.json
    report = _build_report(all_selected, total_samples, tier_targets, min_ds)
    output_report_path.write_text(json.dumps(report, indent=2) + "\n")
    log.info("Wrote sampling report to %s", output_report_path)

    # Print summary
    _print_report(report)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    checkpoint_file.write_text(
        json.dumps(
            {
                "total_selected": len(all_selected),
                "total_target":   total_samples,
                "output_jsonl":   str(output_jsonl_path),
                "output_ids":     str(output_ids_path),
                "output_report":  str(output_report_path),
            },
            indent=2,
        )
        + "\n"
    )
    log.info("Checkpoint written: %s", checkpoint_file)

    return str(output_jsonl_path)


def _print_report(report: dict) -> None:
    """Pretty-print sampling report to stdout."""
    print()
    print("=" * 60)
    print("SAMPLING REPORT")
    print("=" * 60)
    print(f"  Selected: {report['total_selected']} / {report['total_target']}")
    print()
    print("  Tier distribution:")
    for t in [1, 2, 3]:
        # Tier keys are stored as strings in the report JSON
        n = report["tier_counts"].get(str(t), report["tier_counts"].get(t, 0))
        pct = report["tier_pct"].get(str(t), 0.0)
        target = report["tier_targets"].get(str(t), report["tier_targets"].get(t, 0))
        print(f"    T{t}: {n:>5d}  ({pct:.1%})  target={target}")
    print()
    print(f"  Task types covered: {sorted(report['task_counts'].keys())}")
    for tt, n in sorted(report["task_counts"].items()):
        print(f"    {tt}: {n}")
    print()
    print(f"  Datasets covered: {report['n_datasets']}")
    for ds, n in sorted(report["dataset_counts"].items()):
        print(f"    {ds}: {n}")
    print()
    print(f"  Min dataset coverage OK: {report['min_dataset_ok']}")
    print("=" * 60)
    print()


# ── Water-test smoke test ─────────────────────────────────────────────────────

def run_water_test() -> None:
    """
    Smoke test: run stratified sampling on available water-test samples
    with a target of 30 samples. Verifies tier distribution and task coverage.
    """
    import random
    import tempfile
    import shutil

    log.info("=" * 60)
    log.info("WATER-TEST MODE — stratified_sampler (target=30)")
    log.info("=" * 60)

    # ── Generate synthetic difficulty scores from normalized files ────────────
    normalized_dir = _PROJECT_ROOT / "data" / "processed"
    samples: list[dict] = []
    for fpath in sorted(normalized_dir.glob("*_normalized.jsonl")):
        try:
            for row in load_jsonl(str(fpath)):
                if row.get("sample_id"):
                    samples.append(row)
        except Exception as exc:
            log.warning("Skipping %s: %s", fpath.name, exc)

    if not samples:
        log.error("No samples found for water-test.")
        return

    log.info("Water-test: %d raw samples available.", len(samples))

    # Inject synthetic difficulty fields
    _task_to_tier: dict[str, int] = {
        "T1": 1, "T2": 2, "T3": 3, "T4": 2, "T5": 2, "T6": 3,
    }
    _tier_to_soft = {
        1: [0.70, 0.20, 0.10],
        2: [0.15, 0.65, 0.20],
        3: [0.10, 0.20, 0.70],
    }
    rng = random.Random(42)
    scored_samples: list[dict] = []
    for s in samples:
        tier = _task_to_tier.get(s.get("task_type", "T1"), 2)
        soft = _tier_to_soft[tier]
        conf = rng.uniform(0.45, 0.95)
        scored = dict(s)
        scored.update(
            {
                "tier_prior":         tier,
                "tier_prior_soft":    soft,
                "probe_gpt52_correct": rng.random() > 0.3,
                "probe_flash_correct": rng.random() > 0.4,
                "tier_posterior_soft": soft,
                "tier_final":         tier,
                "difficulty_score":   float(tier) + rng.uniform(-0.3, 0.3),
                "confidence":         conf,
            }
        )
        scored_samples.append(scored)

    # Write to temp files
    tmp_dir = Path(tempfile.mkdtemp(prefix="drb_sampler_water_"))
    scores_path   = str(tmp_dir / "difficulty_scores.jsonl")
    output_dir    = str(tmp_dir / "benchmark")
    checkpoint_dir = str(tmp_dir / "checkpoints")

    with open(scores_path, "w") as f:
        for s in scored_samples:
            f.write(json.dumps(s) + "\n")

    # Run sampler
    out_path = run_stratified_sampling(
        scores_path      = scores_path,
        all_samples_path = scores_path,  # self-contained for water-test
        output_dir       = output_dir,
        checkpoint_dir   = checkpoint_dir,
        total_samples    = 30,
        tier_weights     = (0.25, 0.50, 0.25),
        overwrite        = True,
    )

    # ── Validate ──────────────────────────────────────────────────────────────
    selected = load_jsonl(out_path)
    n = len(selected)
    assert n > 0, "No samples selected!"
    log.info("Selected %d samples.", n)

    # Check IDs file
    ids_path = Path(output_dir) / "benchmark_5000_ids.json"
    assert ids_path.exists(), "benchmark_5000_ids.json not written"
    ids = json.loads(ids_path.read_text())
    assert len(ids) == n, f"IDs file length mismatch: {len(ids)} vs {n}"

    # Check report
    report_path = Path(output_dir) / "sampling_report.json"
    assert report_path.exists(), "sampling_report.json not written"
    report = json.loads(report_path.read_text())

    # Task type coverage
    covered_task_types = set(report["task_types_covered"])
    available_task_types = {s.get("task_type") for s in scored_samples}
    missing = available_task_types - covered_task_types
    assert not missing, f"Missing task types in selection: {missing}"
    log.info("Task types covered: %s", sorted(covered_task_types))

    # Tier distribution — for small N (30) allow wide bounds
    # Note: JSON round-trip converts int keys to strings ("1", "2", "3")
    tier_counts = report["tier_counts"]
    log.info("Tier distribution: %s", tier_counts)
    total_selected = sum(tier_counts.values())
    t2_pct = tier_counts.get("2", tier_counts.get(2, 0)) / total_selected if total_selected else 0
    assert 0.20 <= t2_pct <= 0.80, (
        f"Tier 2 fraction {t2_pct:.1%} is outside expected range for small sample"
    )

    # No duplicate IDs
    all_ids = [s["sample_id"] for s in selected]
    assert len(all_ids) == len(set(all_ids)), "Duplicate sample_ids in selection!"

    # Required fields in output
    required_fields = {
        "sample_id", "task_type", "tier_final", "difficulty_score", "confidence",
    }
    for s in selected:
        missing_f = required_fields - s.keys()
        assert not missing_f, f"Missing fields: {missing_f} in {s.get('sample_id')}"

    # Checkpoint exists
    assert (Path(checkpoint_dir) / "stratified_sampler.done").exists(), (
        "Checkpoint file not written"
    )

    log.info("=" * 60)
    log.info("WATER-TEST PASSED — stratified_sampler")
    log.info("=" * 60)

    shutil.rmtree(tmp_dir, ignore_errors=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DocRouteBench Phase 2 — Stratified Sampler",
    )
    parser.add_argument(
        "--scores",
        default="data/processed/difficulty_scores.jsonl",
        help="Path to difficulty scores JSONL",
    )
    parser.add_argument(
        "--samples",
        default="data/processed/samples_with_prior.jsonl",
        help="Path to full samples JSONL (for metadata merge)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/benchmark",
        help="Directory for output files",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="data/phase2_checkpoints",
        help="Directory for checkpoint file",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=5000,
        help="Total samples to select (default: 5000)",
    )
    parser.add_argument(
        "--tier-weights",
        nargs=3,
        type=float,
        default=[0.25, 0.50, 0.25],
        metavar=("W1", "W2", "W3"),
        help="Tier weights for T1/T2/T3 (default: 0.25 0.50 0.25)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs and checkpoint",
    )
    parser.add_argument(
        "--water-test",
        action="store_true",
        help="Smoke test on water-test samples with total=30",
    )
    args = parser.parse_args()

    if args.water_test:
        run_water_test()
        return

    run_stratified_sampling(
        scores_path      = args.scores,
        all_samples_path = args.samples,
        output_dir       = args.output_dir,
        checkpoint_dir   = args.checkpoint_dir,
        total_samples    = args.total,
        tier_weights     = tuple(args.tier_weights),
        overwrite        = args.overwrite,
    )


if __name__ == "__main__":
    main()
