"""
IMC K-Anchor Row Extension: Experiment 1.1c (NeurIPS gap fix)

Symmetric complement to the column extension (1.1a). Whereas 1.1a showed that
IMC generalises to new MODEL columns with K=25 anchor evaluations, this experiment
shows it also generalises to new DOCUMENT DOMAINS (rows) with the same protocol.

Protocol:
  - Hold out all anchor samples from SlideVQA, HierText, TabFact (137 samples).
  - Train IMC on remaining 1363 anchor samples (13 base datasets).
  - For K in {0, 5, 10, 25, 50}: add K samples from the held-out pool to training,
    test on remaining held-out samples. Run 5 random seeds per K.
  - K=0 reproduces the existing AUC=0.60 baseline (zero anchors).
  - Report AUC vs K curve. Expected: strong improvement by K=25, mirroring 1.1a.

No new API calls required: all eval results already exist in
data/model_eval_results/final_eval_correct.jsonl.

Usage:
    python scripts/run_imc_row_extension.py

Output:
    data/imc_row_extension/row_extension_report.json
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.schema import load_jsonl

# ── Config ────────────────────────────────────────────────────────────────────

BENCHMARK_PATH    = _PROJECT_ROOT / "data/benchmark/benchmark_5000.jsonl"
EVAL_CORRECT_PATH = _PROJECT_ROOT / "data/model_eval_results/final_eval_correct.jsonl"
OUT_DIR           = _PROJECT_ROOT / "data/imc_row_extension"
REPORT_PATH       = OUT_DIR / "row_extension_report.json"

HELD_OUT_DATASETS = {"SlideVQA", "HierText", "TabFact"}
K_VALUES          = [0, 5, 10, 25, 50, 100, 200]
SEEDS             = [42, 123, 456, 789, 2024]


# ── Feature construction (mirrors run_imc_external_validation.py) ─────────────

def _sample_features(sample: dict) -> np.ndarray:
    tier = sample.get("tier_final", 2)
    task = sample.get("task_type", "T1")
    task_idx = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "T5": 4, "T6": 5}.get(task, 0)
    task_oh = [0.0] * 6
    task_oh[task_idx] = 1.0
    features = [
        float(tier) / 3.0,
        float(sample.get("has_table_detected", sample.get("has_table", False))),
        float(sample.get("has_chart_detected", sample.get("has_chart", False))),
        float(sample.get("has_figure_detected", sample.get("has_figure", False))),
        float(sample.get("has_handwriting_detected", sample.get("has_handwriting", False))),
        *task_oh,
    ]
    return np.array(features, dtype=np.float32)  # R^11


def _config_features(yaml_key: str, budget_level: str, cost_map: dict) -> np.ndarray:
    tier_char = yaml_key[0].upper() if yaml_key[0].upper() in "ABC" else "B"
    tier_oh = [float(tier_char == t) for t in "ABC"]
    cost = cost_map.get(yaml_key, 0.04)
    budget_tokens = {"B0": 0, "B1": 1024, "B2": 4096, "B3": 16384}.get(budget_level, 0)
    features = [
        *tier_oh,
        math.log1p(cost),
        math.log1p(budget_tokens),
        float(budget_tokens > 0),
    ]
    return np.array(features, dtype=np.float32)  # R^7


def _outer(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.outer(x, c).ravel()  # R^77


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    all_samples = load_jsonl(str(BENCHMARK_PATH))
    # Anchor samples (base training pool): anchor set from non-held-out datasets
    # Held-out pool: ALL samples from held-out datasets (not just anchor subset)
    anchor_samples = [s for s in all_samples if s.get("in_anchor_set")]
    held_out_all = [s for s in all_samples
                    if s.get("source_dataset") in HELD_OUT_DATASETS]
    # Sample map covers both anchor and held-out (all have eval results)
    sample_map = {s["sample_id"]: s for s in anchor_samples + held_out_all}

    # Build eval matrix: {(sample_id, yaml_key, budget_level): is_correct}
    eval_matrix: dict[tuple, bool] = {}
    cost_map: dict[str, float] = {}
    relevant_ids = set(sample_map.keys())
    for row in load_jsonl(str(EVAL_CORRECT_PATH)):
        sid = row["sample_id"]
        if sid not in relevant_ids:
            continue
        key = (sid, row["yaml_key"], row["budget_level"])
        eval_matrix[key] = bool(row.get("eval_correct", False))
        if row["yaml_key"] not in cost_map:
            tier = row["yaml_key"][0].upper()
            cost_map[row["yaml_key"]] = {"A": 0.04, "B": 0.40, "C": 4.0}.get(tier, 0.40)

    return anchor_samples, held_out_all, sample_map, eval_matrix, cost_map


def build_observations(
    sample_ids: list[str],
    sample_map: dict,
    eval_matrix: dict,
    cost_map: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) feature matrix for a set of sample IDs."""
    # Collect all unique (yaml_key, budget_level) pairs
    config_keys = set()
    for (sid, yk, bl) in eval_matrix:
        if sid in {s for s in sample_ids}:
            config_keys.add((yk, bl))

    sample_id_set = set(sample_ids)
    X, y = [], []
    for sid in sample_ids:
        s = sample_map[sid]
        x_i = _sample_features(s)
        for (yk, bl) in config_keys:
            label = eval_matrix.get((sid, yk, bl))
            if label is None:
                continue
            c_j = _config_features(yk, bl, cost_map)
            X.append(_outer(x_i, c_j))
            y.append(int(label))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ── Single K-seed experiment ──────────────────────────────────────────────────

def run_one(
    base_ids: list[str],
    held_out_ids: list[str],
    held_out_ds: dict[str, list[str]],  # dataset -> sample_ids
    sample_map: dict,
    eval_matrix: dict,
    cost_map: dict,
    k: int,
    seed: int,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = random.Random(seed)

    # Select K anchors proportionally across held-out datasets
    anchor_ids: list[str] = []
    if k > 0:
        # Proportional allocation
        total_held = len(held_out_ids)
        for ds, ds_ids in held_out_ds.items():
            alloc = max(1, round(k * len(ds_ids) / total_held)) if k >= len(held_out_ds) else 1
            picked = rng.sample(ds_ids, min(alloc, len(ds_ids)))
            anchor_ids.extend(picked)
        # Trim to exactly k if overallocated
        rng.shuffle(anchor_ids)
        anchor_ids = anchor_ids[:k]

    anchor_set = set(anchor_ids)
    test_ids = [sid for sid in held_out_ids if sid not in anchor_set]

    if len(test_ids) < 5:
        return {"k": k, "seed": seed, "auc": None, "n_test": len(test_ids), "note": "too few test samples"}

    # Build training observations
    train_ids = base_ids + anchor_ids
    X_train, y_train = build_observations(train_ids, sample_map, eval_matrix, cost_map)

    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)

    # Build test observations: predict per (sample, config) pair
    config_keys = set()
    for (sid, yk, bl) in eval_matrix:
        if sid in set(test_ids):
            config_keys.add((yk, bl))

    y_pred_prob, y_true = [], []
    for sid in test_ids:
        s = sample_map[sid]
        x_i = _sample_features(s)
        for (yk, bl) in config_keys:
            label = eval_matrix.get((sid, yk, bl))
            if label is None:
                continue
            c_j = _config_features(yk, bl, cost_map)
            feat = _outer(x_i, c_j).reshape(1, -1)
            prob = clf.predict_proba(feat)[0][1]
            y_pred_prob.append(prob)
            y_true.append(int(label))

    if len(set(y_true)) < 2:
        return {"k": k, "seed": seed, "auc": None, "n_test": len(test_ids), "note": "single class in test"}

    auc = roc_auc_score(y_true, y_pred_prob)
    return {
        "k": k,
        "seed": seed,
        "auc": round(float(auc), 4),
        "n_train_obs": len(y_train),
        "n_test_obs": len(y_pred_prob),
        "n_anchor_ids": len(anchor_ids),
        "n_test_ids": len(test_ids),
    }


# ── Per-dataset breakdown for a given K ──────────────────────────────────────

def run_per_dataset(
    base_ids: list[str],
    held_out_ds: dict[str, list[str]],
    sample_map: dict,
    eval_matrix: dict,
    cost_map: dict,
    k: int,
    seed: int,
) -> dict[str, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = random.Random(seed)
    held_out_ids = [sid for ids in held_out_ds.values() for sid in ids]
    total_held = len(held_out_ids)

    anchor_ids: list[str] = []
    if k > 0:
        for ds, ds_ids in held_out_ds.items():
            alloc = max(1, round(k * len(ds_ids) / total_held)) if k >= len(held_out_ds) else 1
            picked = rng.sample(ds_ids, min(alloc, len(ds_ids)))
            anchor_ids.extend(picked)
        rng.shuffle(anchor_ids)
        anchor_ids = anchor_ids[:k]

    anchor_set = set(anchor_ids)
    train_ids = base_ids + anchor_ids
    X_train, y_train = build_observations(train_ids, sample_map, eval_matrix, cost_map)
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)

    results = {}
    for ds, ds_ids in held_out_ds.items():
        test_ids = [sid for sid in ds_ids if sid not in anchor_set]
        if len(test_ids) < 3:
            results[ds] = {"n_test": len(test_ids), "auc": None}
            continue

        config_keys = set()
        for (sid, yk, bl) in eval_matrix:
            if sid in set(test_ids):
                config_keys.add((yk, bl))

        y_pred_prob, y_true = [], []
        for sid in test_ids:
            s = sample_map[sid]
            x_i = _sample_features(s)
            for (yk, bl) in config_keys:
                label = eval_matrix.get((sid, yk, bl))
                if label is None:
                    continue
                c_j = _config_features(yk, bl, cost_map)
                feat = _outer(x_i, c_j).reshape(1, -1)
                prob = clf.predict_proba(feat)[0][1]
                y_pred_prob.append(prob)
                y_true.append(int(label))

        if len(set(y_true)) < 2:
            results[ds] = {"n_test": len(test_ids), "auc": None}
            continue

        auc = roc_auc_score(y_true, y_pred_prob)
        results[ds] = {"n_test": len(test_ids), "auc": round(float(auc), 4)}

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    anchor_samples, held_out_all, sample_map, eval_matrix, cost_map = load_data()
    print(f"  Anchor samples (base training pool): {len(anchor_samples)}")
    print(f"  Eval matrix entries: {len(eval_matrix)}")

    # Base training: anchor samples from 13 non-held-out datasets
    base_ids = [s["sample_id"] for s in anchor_samples
                if s.get("source_dataset") not in HELD_OUT_DATASETS]

    # Held-out pool: ALL samples from held-out datasets
    held_out_ds: dict[str, list[str]] = defaultdict(list)
    for s in held_out_all:
        held_out_ds[s.get("source_dataset", "")].append(s["sample_id"])
    held_out_ids = [sid for ids in held_out_ds.values() for sid in ids]

    print(f"  Base samples (13 datasets): {len(base_ids)}")
    print(f"  Held-out pool (all 3 datasets): {len(held_out_ids)}")
    for ds, ids in held_out_ds.items():
        print(f"    {ds}: {len(ids)}")

    # ── K sweep ──────────────────────────────────────────────────────────────
    all_results = []
    summary_by_k = {}

    for k in K_VALUES:
        seed_results = []
        for seed in SEEDS:
            r = run_one(base_ids, held_out_ids, dict(held_out_ds),
                        sample_map, eval_matrix, cost_map, k, seed)
            seed_results.append(r)
            auc_str = f"{r['auc']:.4f}" if r.get("auc") is not None else "N/A"
            print(f"  K={k:3d}  seed={seed}  AUC={auc_str}")
        all_results.extend(seed_results)

        valid_aucs = [r["auc"] for r in seed_results if r.get("auc") is not None]
        if valid_aucs:
            summary_by_k[k] = {
                "auc_mean": round(float(np.mean(valid_aucs)), 4),
                "auc_std":  round(float(np.std(valid_aucs)), 4),
                "n_seeds":  len(valid_aucs),
            }
            print(f"  K={k:3d}  AUC={summary_by_k[k]['auc_mean']:.4f} ± {summary_by_k[k]['auc_std']:.4f}")

    # ── Per-dataset breakdown at K=25 ─────────────────────────────────────────
    print("\nPer-dataset breakdown at K=25 (seed=42)...")
    per_dataset_k25 = run_per_dataset(
        base_ids, dict(held_out_ds), sample_map, eval_matrix, cost_map, k=25, seed=42
    )
    for ds, r in per_dataset_k25.items():
        auc_str = f"{r['auc']:.4f}" if r.get("auc") is not None else "N/A"
        print(f"  {ds}: n={r['n_test']}  AUC={auc_str}")

    # ── Report ────────────────────────────────────────────────────────────────
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "1.1c_imc_row_extension",
        "held_out_datasets": list(HELD_OUT_DATASETS),
        "n_base_samples": len(base_ids),
        "n_held_out_samples": len(held_out_ids),
        "held_out_per_dataset": {ds: len(ids) for ds, ids in held_out_ds.items()},
        "k_values": K_VALUES,
        "n_seeds": len(SEEDS),
        "baseline_auc_k0": summary_by_k.get(0, {}).get("auc_mean"),
        "summary_by_k": summary_by_k,
        "per_dataset_k25": per_dataset_k25,
        "per_seed_results": all_results,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {REPORT_PATH}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("IMC ROW EXTENSION SUMMARY")
    print("=" * 50)
    print(f"{'K':>6}  {'AUC mean':>10}  {'AUC std':>9}  {'vs baseline':>12}")
    baseline = summary_by_k.get(0, {}).get("auc_mean", 0.60)
    for k, s in summary_by_k.items():
        delta = s["auc_mean"] - baseline if k > 0 else 0.0
        print(f"{k:>6}  {s['auc_mean']:>10.4f}  {s['auc_std']:>9.4f}  {delta:>+12.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
