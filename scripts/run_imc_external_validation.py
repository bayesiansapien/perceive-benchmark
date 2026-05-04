#!/usr/bin/env python3
"""
IMC External Validation: Experiment 1.1a (NeurIPS gap fix)

Validates that IMC generalises to genuinely new architecture families not present
in the original 7-model pool. Uses Qwen3-VL-Plus (Alibaba) and Llama 4 Scout (Meta)
accessed via OpenRouter.

Protocol:
  Phase A: K=25 anchor seed: run new model on 25 stratified anchor samples.
             Sanity-checks pipeline before the expensive Phase B run.
  Phase B: Ground truth: run new model on remaining ~1,475 anchor samples.
  Phase C: IMC prediction: train IMC on existing 7-model anchor matrix,
             predict new model performance using only its config features + K=25 seed,
             compute AUC against Phase B ground truth.

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  # Phase A only (sanity check, ~100 calls for Qwen, ~25 for Llama):
  python scripts/run_imc_external_validation.py --phase A

  # Full run (background):
  nohup python scripts/run_imc_external_validation.py --phase AB > logs/imc_ext_val.log 2>&1 &

  # IMC analysis only (after Phases A+B complete):
  python scripts/run_imc_external_validation.py --phase C

Output:
  data/imc_external_validation/
    phase_a_results.jsonl      , K=25 seed evaluations
    phase_b_results.jsonl      , remaining 1,475 ground truth evaluations
    imc_report.json            , AUC, routing regret, per-tier breakdown
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.schema import load_jsonl, append_jsonl
from src.sampling.api_probe import load_image_b64
from src.model_eval.answer_extractor import extract_answer
from src.model_eval.cost_calculator import compute_cost
from src.model_eval.model_adapters.openrouter_adapter import OpenRouterAdapter
from src.scoring.unified import is_correct as unified_is_correct

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("imc_ext_val")

# ── Paths ─────────────────────────────────────────────────────────────────────
BENCHMARK_PATH   = _PROJECT_ROOT / "data/benchmark/benchmark_5000.jsonl"
MODEL_POOL_PATH  = _PROJECT_ROOT / "configs/model_pool.yaml"
EVAL_CORRECT_PATH = _PROJECT_ROOT / "data/model_eval_results/final_eval_correct.jsonl"
OUT_DIR          = _PROJECT_ROOT / "data/imc_external_validation"

PHASE_A_PATH = OUT_DIR / "phase_a_results.jsonl"
PHASE_B_PATH = OUT_DIR / "phase_b_results.jsonl"
REPORT_PATH  = OUT_DIR / "imc_report.json"

# ── Experiment config ─────────────────────────────────────────────────────────
K_SEED = 25          # anchor seed samples per model per budget
MAX_WORKERS = 4
RANDOM_SEED = 42

NEW_MODEL_KEYS = ["ext_qwen3vl", "ext_llama4scout"]

_BUDGET_TOKENS = {"B0": 0, "B1": 1024, "B2": 4096, "B3": 16384}
_write_lock = threading.Lock()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_anchor_samples() -> list[dict]:
    all_samples = load_jsonl(str(BENCHMARK_PATH))
    return [s for s in all_samples if s.get("in_anchor_set")]


def load_model_pool() -> dict:
    with open(MODEL_POOL_PATH) as f:
        return yaml.safe_load(f)["models"]


def load_existing_eval_matrix(anchor_ids: set[str]) -> dict[tuple[str, str, str], bool]:
    """Load existing eval results as {(sample_id, yaml_key, budget_level): is_correct}."""
    matrix: dict[tuple[str, str, str], bool] = {}
    for row in load_jsonl(str(EVAL_CORRECT_PATH)):
        sid = row["sample_id"]
        if sid not in anchor_ids:
            continue
        matrix[(sid, row["yaml_key"], row["budget_level"])] = bool(row["eval_correct"])
    return matrix


def load_phase_results(path: Path) -> dict[tuple[str, str, str], dict]:
    """Load phase A or B results as {(sample_id, yaml_key, budget_level): row}."""
    results = {}
    if not path.exists():
        return results
    for row in load_jsonl(str(path)):
        key = (row["sample_id"], row["yaml_key"], row["budget_level"])
        results[key] = row
    return results


# ── K=25 stratified sample selection ─────────────────────────────────────────

def select_seed_samples(anchor_samples: list[dict], k: int = K_SEED) -> list[dict]:
    """Stratified K-sample selection across tier × task_type."""
    random.seed(RANDOM_SEED)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in anchor_samples:
        key = (s.get("tier_final", 2), s.get("task_type", "T1"))
        groups[key].append(s)

    # Proportional allocation, minimum 1 per group
    total = len(anchor_samples)
    allocated: dict[tuple, int] = {}
    remaining = k
    for gkey, gsamples in sorted(groups.items()):
        alloc = max(1, round(k * len(gsamples) / total))
        allocated[gkey] = alloc
        remaining -= alloc

    # Fix rounding to hit exactly k
    keys = sorted(allocated)
    i = 0
    while remaining > 0:
        allocated[keys[i % len(keys)]] += 1
        remaining -= 1
        i += 1
    while remaining < 0:
        if allocated[keys[i % len(keys)]] > 1:
            allocated[keys[i % len(keys)]] -= 1
            remaining += 1
        i += 1

    seed = []
    for gkey, alloc in allocated.items():
        samples = groups[gkey]
        random.shuffle(samples)
        seed.extend(samples[:alloc])

    random.shuffle(seed)
    return seed[:k]


# ── Inference ─────────────────────────────────────────────────────────────────

def _eval_one(
    sample: dict,
    yaml_key: str,
    model_cfg: dict,
    budget_level: str,
    out_path: Path,
    done_keys: set[tuple],
) -> Optional[dict]:
    key = (sample["sample_id"], yaml_key, budget_level)
    if key in done_keys:
        return None

    try:
        image_b64 = load_image_b64(sample.get("image_path", ""))
    except FileNotFoundError:
        log.warning("Image not found: %s", sample["sample_id"])
        return None

    adapter = OpenRouterAdapter(yaml_key, model_cfg, budget_level)

    raw = {}
    error_str = None
    try:
        raw = adapter.call(image_b64, sample.get("query", ""))
    except Exception as exc:
        error_str = str(exc)[:200]
        log.warning("Inference failed %s/%s/%s: %s", sample["sample_id"], yaml_key, budget_level, error_str)

    raw_answer = raw.get("answer", "")
    predicted = extract_answer(raw_answer, model_cfg.get("model_id", ""))

    correct = False
    if error_str is None and predicted:
        gt = sample.get("gt_answer", "")
        aliases = sample.get("gt_answer_aliases", [])
        all_gt = [gt] + aliases if gt else aliases
        metric = sample.get("correctness_metric", "anls")
        try:
            correct = unified_is_correct(
                predicted=predicted,
                ground_truth=all_gt if all_gt else [""],
                metric=metric,
                dataset=sample.get("source_dataset", ""),
            )
        except Exception:
            pass

    cost = compute_cost(
        model_cfg,
        raw.get("input_tokens", 0),
        raw.get("output_tokens", 0),
        raw.get("reasoning_tokens", 0),
    )

    record = {
        "sample_id":        sample["sample_id"],
        "yaml_key":         yaml_key,
        "budget_level":     budget_level,
        "budget_tokens":    _BUDGET_TOKENS.get(budget_level, 0),
        "is_correct":       bool(correct),
        "predicted_answer": predicted,
        "raw_answer":       raw_answer,
        "reasoning_content": raw.get("reasoning_content", ""),
        "input_tokens":     raw.get("input_tokens", 0),
        "output_tokens":    raw.get("output_tokens", 0),
        "reasoning_tokens": raw.get("reasoning_tokens", 0),
        "total_cost_usd":   cost,
        "latency_ms":       raw.get("latency_ms", 0),
        "error":            error_str,
        "tier_final":       sample.get("tier_final", 0),
        "task_type":        sample.get("task_type", ""),
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }
    with _write_lock:
        append_jsonl(str(out_path), record)
    return record


def run_phase(
    samples: list[dict],
    phase_label: str,
    out_path: Path,
    model_pool: dict,
) -> None:
    """Run inference on samples for all new models, writing to out_path."""
    existing = load_phase_results(out_path)
    done_keys = set(existing.keys())

    work_items = []
    for yaml_key in NEW_MODEL_KEYS:
        cfg = model_pool[yaml_key]
        for budget in cfg.get("budgets", ["B0"]):
            for s in samples:
                key = (s["sample_id"], yaml_key, budget)
                if key not in done_keys:
                    work_items.append((s, yaml_key, cfg, budget))

    log.info("Phase %s: %d work items (%d already done)", phase_label, len(work_items), len(done_keys))
    if not work_items:
        log.info("Phase %s already complete.", phase_label)
        return

    total_cost = 0.0
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_eval_one, s, yk, cfg, bl, out_path, done_keys): (s["sample_id"], yk, bl)
            for s, yk, cfg, bl in work_items
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    total_cost += result.get("total_cost_usd", 0)
                    completed += 1
                    if completed % 50 == 0:
                        log.info("Phase %s: %d/%d done | $%.2f", phase_label, completed, len(work_items), total_cost)
            except Exception as exc:
                log.error("Worker error: %s", exc)

    log.info("Phase %s complete: %d calls | $%.2f total", phase_label, completed, total_cost)


# ── IMC implementation ────────────────────────────────────────────────────────

def _sample_features(sample: dict) -> np.ndarray:
    """Build x_i ∈ R^11 sample feature vector (matches paper §5.5 description)."""
    tier = sample.get("tier_final", 2)
    task = sample.get("task_type", "T1")
    task_idx = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "T5": 4, "T6": 5}.get(task, 0)
    task_oh = [0.0] * 6
    task_oh[task_idx] = 1.0

    features = [
        float(tier) / 3.0,                                  # normalised tier
        float(sample.get("has_table_detected", sample.get("has_table", False))),
        float(sample.get("has_chart_detected", sample.get("has_chart", False))),
        float(sample.get("has_figure_detected", sample.get("has_figure", False))),
        float(sample.get("has_handwriting_detected", sample.get("has_handwriting", False))),
        *task_oh,                                             # T1-T6 one-hot
    ]
    return np.array(features, dtype=np.float32)  # R^11


def _config_features(yaml_key: str, cfg: dict, budget_level: str) -> np.ndarray:
    """Build c_j ∈ R^7 config feature vector."""
    tier_char = yaml_key[0].upper() if yaml_key[0].upper() in "ABC" else "B"
    tier_oh = [float(tier_char == t) for t in "ABC"]  # 3 dims

    cost = cfg.get("cost_per_1M_input", 0.04)
    budget_tokens = _BUDGET_TOKENS.get(budget_level, 0)

    provider = cfg.get("provider", "openai")
    # For external models, use model_family to determine provider group
    family = cfg.get("model_family", "")
    provider_group = {
        "openai": 0, "anthropic": 1, "google": 2,
        "qwen": 3, "meta": 4,
    }.get(family or provider, 3)

    features = [
        *tier_oh,
        math.log1p(cost),
        math.log1p(budget_tokens),
        float(provider_group) / 4.0,   # normalised provider index
        float(budget_tokens > 0),      # has_thinking flag
    ]
    return np.array(features, dtype=np.float32)  # R^7


def _outer_product(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.outer(x, c).ravel()  # R^{11*7} = R^77


def run_imc_analysis(
    anchor_samples: list[dict],
    model_pool: dict,
    eval_matrix: dict,
) -> dict:
    """Train IMC on existing 7-model matrix, predict new models, report AUC."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError:
        log.error("scikit-learn not available. Run: pip install scikit-learn")
        sys.exit(1)

    anchor_ids = {s["sample_id"] for s in anchor_samples}
    sample_map = {s["sample_id"]: s for s in anchor_samples}

    # Core 7-model yaml keys (exclude imc_validation_only models)
    core_keys = [k for k, v in model_pool.items() if not v.get("imc_validation_only")]

    # ── Build training data from existing 7-model anchor matrix ──────────────
    X_train, y_train = [], []
    for yaml_key in core_keys:
        cfg = model_pool[yaml_key]
        for budget in cfg.get("budgets", ["B0"]):
            c_j = _config_features(yaml_key, cfg, budget)
            for sid in anchor_ids:
                label = eval_matrix.get((sid, yaml_key, budget))
                if label is None:
                    continue
                x_i = _sample_features(sample_map[sid])
                X_train.append(_outer_product(x_i, c_j))
                y_train.append(int(label))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    log.info("IMC training: %d observations, %d features", len(y_train), X_train.shape[1])

    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)
    log.info("IMC model trained. Training accuracy: %.3f", clf.score(X_train, y_train))

    # ── Load seed (phase A) and ground truth (phase B) results ───────────────
    phase_a = load_phase_results(PHASE_A_PATH)
    phase_b = load_phase_results(PHASE_B_PATH)

    # Merge phase A + B into full ground truth
    ground_truth: dict[tuple, bool] = {}
    for key, row in {**phase_a, **phase_b}.items():
        ground_truth[key] = bool(row.get("is_correct", False))

    seed_ids = {key[0] for key in phase_a.keys()}
    log.info("Seed samples (phase A): %d unique sample IDs", len(seed_ids))
    log.info("Ground truth entries (phase A+B): %d", len(ground_truth))

    # ── Predict + evaluate for each new model ────────────────────────────────
    results_by_model = {}

    for yaml_key in NEW_MODEL_KEYS:
        cfg = model_pool[yaml_key]
        model_results = {}

        for budget in cfg.get("budgets", ["B0"]):
            # Predict on non-seed anchor samples
            y_pred_prob, y_true = [], []
            for s in anchor_samples:
                sid = s["sample_id"]
                if sid in seed_ids:
                    continue  # exclude seed from evaluation
                key = (sid, yaml_key, budget)
                gt = ground_truth.get(key)
                if gt is None:
                    continue
                x_i = _sample_features(s)
                c_j = _config_features(yaml_key, cfg, budget)
                feat = _outer_product(x_i, c_j).reshape(1, -1)
                prob = clf.predict_proba(feat)[0][1]
                y_pred_prob.append(prob)
                y_true.append(int(gt))

            if len(y_true) < 10:
                log.warning("Too few ground truth entries for %s/%s (%d). Run Phase B first.", yaml_key, budget, len(y_true))
                model_results[budget] = {"n": len(y_true), "auc": None, "routing_regret": None}
                continue

            auc = roc_auc_score(y_true, y_pred_prob)

            # Routing regret: fraction where IMC's top-1 prediction matches GT
            y_pred_bin = [1 if p >= 0.5 else 0 for p in y_pred_prob]
            agreement = sum(p == g for p, g in zip(y_pred_bin, y_true)) / len(y_true)

            model_results[budget] = {
                "n": len(y_true),
                "auc": round(auc, 4),
                "label_agreement": round(agreement, 4),
                "mean_gt_correct": round(sum(y_true) / len(y_true), 4),
            }
            log.info("%s / %s: n=%d AUC=%.4f agreement=%.4f", yaml_key, budget, len(y_true), auc, agreement)

        # Per-tier breakdown using phase B ground truth
        tier_breakdown = defaultdict(lambda: {"n": 0, "tp": 0, "gt_correct": 0})
        for s in anchor_samples:
            sid = s["sample_id"]
            if sid in seed_ids:
                continue
            budget = cfg.get("budgets", ["B0"])[0]  # use first budget for tier breakdown
            key = (sid, yaml_key, budget)
            gt = ground_truth.get(key)
            if gt is None:
                continue
            tier = s.get("tier_final", 2)
            x_i = _sample_features(s)
            c_j = _config_features(yaml_key, cfg, budget)
            feat = _outer_product(x_i, c_j).reshape(1, -1)
            pred = clf.predict(feat)[0]
            tier_breakdown[tier]["n"] += 1
            tier_breakdown[tier]["gt_correct"] += int(gt)
            if pred == int(gt):
                tier_breakdown[tier]["tp"] += 1

        tier_acc = {
            tier: {
                "n": v["n"],
                "accuracy": round(v["tp"] / v["n"], 4) if v["n"] > 0 else None,
                "gt_correct_rate": round(v["gt_correct"] / v["n"], 4) if v["n"] > 0 else None,
            }
            for tier, v in tier_breakdown.items()
        }

        results_by_model[yaml_key] = {
            "model_name": cfg.get("name", yaml_key),
            "provider": cfg.get("provider", ""),
            "model_family": cfg.get("model_family", ""),
            "budgets": cfg.get("budgets", ["B0"]),
            "budget_results": model_results,
            "tier_breakdown": tier_acc,
        }

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "k_seed": K_SEED,
        "n_anchor": len(anchor_samples),
        "n_seed": len(seed_ids),
        "n_eval": len(anchor_samples) - len(seed_ids),
        "imc_training_obs": len(y_train),
        "results": results_by_model,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log.info("IMC report written to %s", REPORT_PATH)

    # Print summary
    print("\n" + "=" * 60)
    print("IMC EXTERNAL VALIDATION SUMMARY")
    print("=" * 60)
    for mk, mr in results_by_model.items():
        print(f"\n{mr['model_name']} ({mr['model_family']} family)")
        for budget, br in mr["budget_results"].items():
            if br["auc"] is not None:
                print(f"  {budget}: AUC={br['auc']:.4f}  label_agreement={br['label_agreement']:.4f}  n={br['n']}")
            else:
                print(f"  {budget}: insufficient ground truth (n={br['n']}), run Phase B")
    print("=" * 60)

    return report


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IMC External Validation")
    parser.add_argument(
        "--phase", default="AB",
        choices=["A", "B", "AB", "C", "ABC"],
        help="A=seed only, B=full ground truth, C=analysis only, AB=seed+ground truth, ABC=all",
    )
    parser.add_argument("--seed-k", type=int, default=K_SEED, help=f"Seed samples (default {K_SEED})")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading benchmark and model pool...")
    anchor_samples = load_anchor_samples()
    model_pool = load_model_pool()
    log.info("Anchor samples: %d", len(anchor_samples))

    seed_samples = select_seed_samples(anchor_samples, k=args.seed_k)
    anchor_ids = {s["sample_id"] for s in anchor_samples}
    seed_ids = {s["sample_id"] for s in seed_samples}
    remaining_samples = [s for s in anchor_samples if s["sample_id"] not in seed_ids]

    log.info("Seed: %d | Remaining: %d", len(seed_samples), len(remaining_samples))

    if "A" in args.phase:
        log.info("=== PHASE A: K=%d seed evaluations ===", args.seed_k)
        run_phase(seed_samples, "A", PHASE_A_PATH, model_pool)

    if "B" in args.phase:
        log.info("=== PHASE B: %d ground truth evaluations ===", len(remaining_samples))
        run_phase(remaining_samples, "B", PHASE_B_PATH, model_pool)

    if "C" in args.phase:
        log.info("=== PHASE C: IMC analysis ===")
        eval_matrix = load_existing_eval_matrix(anchor_ids)
        log.info("Existing eval matrix entries: %d", len(eval_matrix))
        run_imc_analysis(anchor_samples, model_pool, eval_matrix)


if __name__ == "__main__":
    main()
