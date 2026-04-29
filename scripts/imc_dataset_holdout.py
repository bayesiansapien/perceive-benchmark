#!/usr/bin/env python3
"""
IMC Dataset Hold-out Validation — Experiment 1.1b (NeurIPS gap fix)

Validates that IMC generalises to document types not seen during training.
Holds out three diverse datasets from IMC training, trains on anchor-set data
from the remaining 13+ datasets, and predicts model correctness for held-out
samples using only parse-time features (no LLM calls).

Claim under test: "new document types require zero LLM calls (AUC 0.88)"

Output: data/imc_dataset_holdout_report.json
"""
from __future__ import annotations

import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("imc_holdout")

# ── Paths ─────────────────────────────────────────────────────────────────────
BENCHMARK_PATH    = _PROJECT_ROOT / "data/benchmark/benchmark_5000.jsonl"
MODEL_POOL_PATH   = _PROJECT_ROOT / "configs/model_pool.yaml"
EVAL_CORRECT_PATH = _PROJECT_ROOT / "data/model_eval_results/final_eval_correct.jsonl"
REPORT_PATH       = _PROJECT_ROOT / "data/imc_dataset_holdout_report.json"

# ── Experiment config ─────────────────────────────────────────────────────────
HELD_OUT_DATASETS = ["SlideVQA", "HierText", "TabFact"]

_BUDGET_TOKENS = {"B0": 0, "B1": 1024, "B2": 4096, "B3": 16384}

_PROVIDER_GROUP = {
    "openai": 0,
    "anthropic": 1,
    "google": 2,
    "qwen": 3,
    "meta": 4,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_benchmark() -> list[dict]:
    return load_jsonl(str(BENCHMARK_PATH))


def load_model_pool() -> dict:
    with open(MODEL_POOL_PATH) as f:
        return yaml.safe_load(f)["models"]


def load_eval_matrix() -> dict[tuple[str, str, str], bool]:
    """Load all eval results as {(sample_id, yaml_key, budget_level): is_correct}."""
    matrix: dict[tuple[str, str, str], bool] = {}
    for row in load_jsonl(str(EVAL_CORRECT_PATH)):
        key = (row["sample_id"], row["yaml_key"], row["budget_level"])
        matrix[key] = bool(row["eval_correct"])
    return matrix


# ── Feature engineering (mirrors run_imc_external_validation.py) ──────────────

def _sample_features(sample: dict) -> np.ndarray:
    """Build x_i in R^11: normalised tier, 4 visual flags, T1-T6 one-hot."""
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


def _config_features(yaml_key: str, cfg: dict, budget_level: str) -> np.ndarray:
    """Build c_j in R^7: tier one-hot ABC, log1p cost, log1p budget_tokens,
    normalised provider group, has_thinking flag."""
    tier_char = yaml_key[0].upper() if yaml_key[0].upper() in "ABC" else "B"
    tier_oh = [float(tier_char == t) for t in "ABC"]  # 3 dims

    cost = cfg.get("cost_per_1M_input", 0.04)
    budget_tokens = _BUDGET_TOKENS.get(budget_level, 0)

    provider = cfg.get("provider", "openai")
    family = cfg.get("model_family", "")
    provider_group = _PROVIDER_GROUP.get(family or provider, 3)

    features = [
        *tier_oh,
        math.log1p(cost),
        math.log1p(budget_tokens),
        float(provider_group) / 4.0,
        float(budget_tokens > 0),  # has_thinking flag
    ]
    return np.array(features, dtype=np.float32)  # R^7


def _outer_product(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Outer product vectorisation: R^{11*7} = R^77."""
    return np.outer(x, c).ravel()


# ── Main experiment ───────────────────────────────────────────────────────────

def main() -> None:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError:
        log.error("scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(1)

    log.info("Loading benchmark, model pool, and eval matrix...")
    all_samples = load_benchmark()
    model_pool = load_model_pool()
    eval_matrix = load_eval_matrix()

    log.info("Total samples: %d", len(all_samples))
    log.info("Total eval_correct records: %d", len(eval_matrix))

    # ── Core models only (exclude imc_validation_only) ───────────────────────
    core_keys = [k for k, v in model_pool.items() if not v.get("imc_validation_only")]
    log.info("Core yaml_keys: %s", core_keys)

    # ── Dataset split ─────────────────────────────────────────────────────────
    held_out_set = set(HELD_OUT_DATASETS)

    # Training samples: anchor set, NOT from held-out datasets
    train_samples = [
        s for s in all_samples
        if s.get("in_anchor_set") and s.get("source_dataset") not in held_out_set
    ]

    # Test samples: any split, FROM held-out datasets (ground truth available for all splits)
    test_samples = [
        s for s in all_samples
        if s.get("source_dataset") in held_out_set
    ]

    train_ids = {s["sample_id"] for s in train_samples}
    test_ids  = {s["sample_id"] for s in test_samples}

    log.info(
        "Dataset split — Train (anchor, non-held-out): %d samples | Test (held-out, any split): %d samples",
        len(train_samples), len(test_samples),
    )

    # Distribution of held-out datasets in test
    from collections import Counter
    test_ds_counts = Counter(s.get("source_dataset") for s in test_samples)
    log.info("Held-out dataset counts: %s", dict(test_ds_counts))

    # ── Build training data ───────────────────────────────────────────────────
    X_train: list[np.ndarray] = []
    y_train: list[int] = []

    for yaml_key in core_keys:
        cfg = model_pool[yaml_key]
        for budget in cfg.get("budgets", ["B0"]):
            c_j = _config_features(yaml_key, cfg, budget)
            for s in train_samples:
                label = eval_matrix.get((s["sample_id"], yaml_key, budget))
                if label is None:
                    continue
                x_i = _sample_features(s)
                X_train.append(_outer_product(x_i, c_j))
                y_train.append(int(label))

    X_train_arr = np.array(X_train)
    y_train_arr = np.array(y_train)

    log.info(
        "IMC training matrix: %d observations, %d features, %.1f%% positive",
        len(y_train_arr), X_train_arr.shape[1], 100.0 * y_train_arr.mean(),
    )

    # ── Train IMC model ───────────────────────────────────────────────────────
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    clf.fit(X_train_arr, y_train_arr)
    train_acc = clf.score(X_train_arr, y_train_arr)
    log.info("IMC trained. Training accuracy: %.4f", train_acc)

    # ── Predict on held-out test samples ─────────────────────────────────────
    # Per-model AUC
    per_model_results: dict[str, dict] = {}
    all_y_true: list[int] = []
    all_y_prob: list[float] = []

    # Pre-build sample feature cache for test samples
    test_sample_map = {s["sample_id"]: s for s in test_samples}

    for yaml_key in core_keys:
        cfg = model_pool[yaml_key]
        model_y_true: list[int] = []
        model_y_prob: list[float] = []

        for budget in cfg.get("budgets", ["B0"]):
            c_j = _config_features(yaml_key, cfg, budget)
            for s in test_samples:
                label = eval_matrix.get((s["sample_id"], yaml_key, budget))
                if label is None:
                    continue
                x_i = _sample_features(s)
                feat = _outer_product(x_i, c_j).reshape(1, -1)
                prob = clf.predict_proba(feat)[0][1]
                model_y_true.append(int(label))
                model_y_prob.append(prob)
                all_y_true.append(int(label))
                all_y_prob.append(prob)

        if len(model_y_true) < 10:
            log.warning("%s: too few test observations (%d), skipping AUC", yaml_key, len(model_y_true))
            per_model_results[yaml_key] = {
                "model_name": cfg.get("name", yaml_key),
                "n": len(model_y_true),
                "auc": None,
                "label_agreement": None,
            }
            continue

        model_auc = roc_auc_score(model_y_true, model_y_prob)
        model_preds = [1 if p >= 0.5 else 0 for p in model_y_prob]
        model_agree = sum(p == g for p, g in zip(model_preds, model_y_true)) / len(model_y_true)

        log.info(
            "%s: n=%d  AUC=%.4f  label_agreement=%.4f  pos_rate=%.3f",
            yaml_key, len(model_y_true), model_auc, model_agree,
            sum(model_y_true) / len(model_y_true),
        )

        per_model_results[yaml_key] = {
            "model_name": cfg.get("name", yaml_key),
            "n": len(model_y_true),
            "auc": round(model_auc, 4),
            "label_agreement": round(model_agree, 4),
            "pos_rate": round(sum(model_y_true) / len(model_y_true), 4),
        }

    # ── Aggregate AUC (all models x budgets pooled) ───────────────────────────
    if len(all_y_true) >= 10:
        aggregate_auc = roc_auc_score(all_y_true, all_y_prob)
        agg_preds = [1 if p >= 0.5 else 0 for p in all_y_prob]
        agg_agree = sum(p == g for p, g in zip(agg_preds, all_y_true)) / len(all_y_true)
        log.info(
            "AGGREGATE: n=%d  AUC=%.4f  label_agreement=%.4f",
            len(all_y_true), aggregate_auc, agg_agree,
        )
    else:
        aggregate_auc = None
        agg_agree = None
        log.warning("Insufficient test observations for aggregate AUC")

    # ── Cheapest-correct routing agreement ────────────────────────────────────
    # For each test sample, find cheapest config (lowest cost_per_1M_input) where
    # predicted_correct=True vs cheapest config where actual eval_correct=True.
    # Report fraction of samples where they agree (or both have no correct config).

    # Build cost-ordered config list for routing decisions
    config_cost_list: list[tuple[float, str, str]] = []
    for yaml_key in core_keys:
        cfg = model_pool[yaml_key]
        cost = cfg.get("cost_per_1M_input", 999.0)
        for budget in cfg.get("budgets", ["B0"]):
            config_cost_list.append((cost, yaml_key, budget))
    config_cost_list.sort()  # ascending by cost

    routing_agreements = 0
    routing_total = 0

    for s in test_samples:
        sid = s["sample_id"]

        # Ground truth: cheapest config where eval_correct=True
        actual_cheapest: tuple[float, str, str] | None = None
        for cost, yaml_key, budget in config_cost_list:
            label = eval_matrix.get((sid, yaml_key, budget))
            if label is True:
                actual_cheapest = (cost, yaml_key, budget)
                break

        # Predicted: cheapest config where predicted_correct=True (prob >= 0.5)
        predicted_cheapest: tuple[float, str, str] | None = None
        for cost, yaml_key, budget in config_cost_list:
            cfg = model_pool[yaml_key]
            label = eval_matrix.get((sid, yaml_key, budget))
            if label is None:
                continue  # skip if no ground truth exists to pair with
            x_i = _sample_features(s)
            c_j = _config_features(yaml_key, cfg, budget)
            feat = _outer_product(x_i, c_j).reshape(1, -1)
            prob = clf.predict_proba(feat)[0][1]
            if prob >= 0.5:
                predicted_cheapest = (cost, yaml_key, budget)
                break

        # Only count samples where at least one side has a determination
        if actual_cheapest is None and predicted_cheapest is None:
            continue  # no ground truth at all — skip

        routing_total += 1
        # Agreement: both pick same yaml_key+budget OR both find nothing
        actual_route   = (actual_cheapest[1],    actual_cheapest[2])    if actual_cheapest    else None
        predicted_route = (predicted_cheapest[1], predicted_cheapest[2]) if predicted_cheapest else None

        if actual_route == predicted_route:
            routing_agreements += 1

    routing_agreement_rate = (
        routing_agreements / routing_total if routing_total > 0 else None
    )
    log.info(
        "Cheapest-correct routing agreement: %d/%d = %.4f",
        routing_agreements, routing_total,
        routing_agreement_rate if routing_agreement_rate is not None else float("nan"),
    )

    # ── Per-dataset AUC breakdown ─────────────────────────────────────────────
    per_dataset_results: dict[str, dict] = {}
    for ds in HELD_OUT_DATASETS:
        ds_samples = [s for s in test_samples if s.get("source_dataset") == ds]
        ds_y_true: list[int] = []
        ds_y_prob: list[float] = []

        for yaml_key in core_keys:
            cfg = model_pool[yaml_key]
            for budget in cfg.get("budgets", ["B0"]):
                c_j = _config_features(yaml_key, cfg, budget)
                for s in ds_samples:
                    label = eval_matrix.get((s["sample_id"], yaml_key, budget))
                    if label is None:
                        continue
                    x_i = _sample_features(s)
                    feat = _outer_product(x_i, c_j).reshape(1, -1)
                    prob = clf.predict_proba(feat)[0][1]
                    ds_y_true.append(int(label))
                    ds_y_prob.append(prob)

        if len(ds_y_true) >= 10:
            ds_auc = roc_auc_score(ds_y_true, ds_y_prob)
            per_dataset_results[ds] = {
                "n_samples": len(ds_samples),
                "n_observations": len(ds_y_true),
                "auc": round(ds_auc, 4),
                "pos_rate": round(sum(ds_y_true) / len(ds_y_true), 4),
            }
            log.info("  %s: n_obs=%d  AUC=%.4f", ds, len(ds_y_true), ds_auc)
        else:
            per_dataset_results[ds] = {
                "n_samples": len(ds_samples),
                "n_observations": len(ds_y_true),
                "auc": None,
            }

    # ── Write report ──────────────────────────────────────────────────────────
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "1.1b_imc_dataset_holdout",
        "held_out_datasets": HELD_OUT_DATASETS,
        "n_train_samples": len(train_samples),
        "n_test_samples": len(test_samples),
        "n_train_observations": len(y_train_arr),
        "n_test_observations": len(all_y_true),
        "imc_train_accuracy": round(train_acc, 4),
        "aggregate_auc": round(aggregate_auc, 4) if aggregate_auc is not None else None,
        "aggregate_label_agreement": round(agg_agree, 4) if agg_agree is not None else None,
        "cheapest_correct_routing_agreement": round(routing_agreement_rate, 4) if routing_agreement_rate is not None else None,
        "cheapest_correct_routing_n": routing_total,
        "per_model_results": per_model_results,
        "per_dataset_results": per_dataset_results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report written to %s", REPORT_PATH)

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("IMC DATASET HOLD-OUT VALIDATION SUMMARY (Experiment 1.1b)")
    print("=" * 65)
    print(f"Held-out datasets : {', '.join(HELD_OUT_DATASETS)}")
    print(f"Train samples     : {len(train_samples):,}  (anchor set, non-held-out)")
    print(f"Test samples      : {len(test_samples):,}  (all splits, held-out datasets)")
    print(f"Train observations: {len(y_train_arr):,}")
    print(f"Test observations : {len(all_y_true):,}")
    print()
    print(f"AGGREGATE AUC     : {aggregate_auc:.4f}" if aggregate_auc is not None else "AGGREGATE AUC     : N/A")
    print(f"Label agreement   : {agg_agree:.4f}" if agg_agree is not None else "Label agreement   : N/A")
    print(f"Routing agreement : {routing_agreement_rate:.4f}  (cheapest-correct, n={routing_total})" if routing_agreement_rate is not None else "Routing agreement : N/A")
    print()
    print("Per-model AUC:")
    for yaml_key, res in per_model_results.items():
        auc_str = f"{res['auc']:.4f}" if res["auc"] is not None else "  N/A  "
        agree_str = f"{res['label_agreement']:.4f}" if res["label_agreement"] is not None else "  N/A  "
        print(f"  {yaml_key:<20s}  AUC={auc_str}  agreement={agree_str}  n={res['n']}")
    print()
    print("Per-dataset AUC (held-out):")
    for ds, res in per_dataset_results.items():
        auc_str = f"{res['auc']:.4f}" if res["auc"] is not None else "  N/A  "
        print(f"  {ds:<15s}  AUC={auc_str}  n_obs={res['n_observations']}  n_samples={res['n_samples']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
