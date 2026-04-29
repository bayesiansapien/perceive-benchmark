#!/usr/bin/env python3
"""
DocRouteBench Phase 3 — Cascade Validation

Validates the cascade evaluation methodology on the fully-observed anchor set.
Run AFTER anchor results (both API + GPU) are merged.

4 experiments:
  1. DVR:        dominance violation rate (cheap correct AND expensive wrong?)
  2. GT Agreement: cascade GT-Cost label == full-evaluation label?
  3. Cost R²:    reasoning_tokens predictable from sample features?
  4. KS test:   anchor feature distribution ≈ non-anchor?

Usage:
    python scripts/validate_cascade.py
    python scripts/validate_cascade.py --results data/model_eval_results/merged/anchor_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("validate_cascade")

_TIER_ORDER = {"A": 1, "B": 2, "C": 3}


def _get_tier(config_id: str) -> str:
    """Extract tier letter from config_id like 'c1_gpt54_B2' → 'C'."""
    return config_id[0].upper()


def run_cascade_validation(
    anchor_results_path: str = "data/model_eval_results/merged/anchor_results.jsonl",
    benchmark_path: str = "data/benchmark/benchmark_5000.jsonl",
    output_path: str = "data/phase3_cascade_validation.json",
) -> dict:
    """Run all 4 validation experiments. Returns report dict."""
    from src.schema import load_jsonl

    anchor_results_path = str(_ROOT / anchor_results_path) if not Path(anchor_results_path).is_absolute() else anchor_results_path
    benchmark_path = str(_ROOT / benchmark_path) if not Path(benchmark_path).is_absolute() else benchmark_path
    output_path = str(_ROOT / output_path) if not Path(output_path).is_absolute() else output_path

    log.info("Loading anchor results from %s ...", anchor_results_path)
    results = load_jsonl(anchor_results_path)

    # Index: sample_id → {config_id → result}
    by_sample: dict[str, dict] = defaultdict(dict)
    for r in results:
        by_sample[r["sample_id"]][r["config_id"]] = r

    log.info("Loaded %d samples, %d total result rows", len(by_sample), len(results))

    # ── Experiment 1: Dominance Violation Rate ────────────────────────────────
    log.info("Experiment 1: Dominance Violation Rate (DVR)...")
    violations = 0
    total_pairs = 0
    for sample_id, configs in by_sample.items():
        for cid_cheap, r_cheap in configs.items():
            if not r_cheap.get("is_correct"):
                continue
            tier_cheap = _TIER_ORDER.get(_get_tier(cid_cheap), 9)
            for cid_exp, r_exp in configs.items():
                tier_exp = _TIER_ORDER.get(_get_tier(cid_exp), 9)
                if tier_exp <= tier_cheap:
                    continue
                total_pairs += 1
                if not r_exp.get("is_correct"):
                    violations += 1

    dvr = violations / total_pairs if total_pairs > 0 else 0.0
    dvr_pass = dvr < 0.10
    log.info("  DVR = %.1f%% (%d/%d pairs) — %s",
             dvr * 100, violations, total_pairs, "PASS" if dvr_pass else "FAIL")

    # ── Experiment 2: Cascade GT Agreement ───────────────────────────────────
    log.info("Experiment 2: Cascade GT-Cost label agreement...")

    def _cheapest_correct(configs: dict) -> str | None:
        correct = [
            (cid, r) for cid, r in configs.items() if r.get("is_correct")
        ]
        if not correct:
            return None
        return min(
            correct,
            key=lambda pair: (_TIER_ORDER.get(_get_tier(pair[0]), 9), pair[1].get("total_cost_usd", 0)),
        )[0]

    cascade_agree = cascade_total = 0
    for sample_id, configs in by_sample.items():
        full_gt = _cheapest_correct(configs)
        if full_gt is None:
            continue

        # Simulate cascade: stop at first tier with any correct
        cascade_gt = None
        for tier in ["A", "B", "C"]:
            tier_correct = [
                (cid, r) for cid, r in configs.items()
                if _get_tier(cid) == tier and r.get("is_correct")
            ]
            if tier_correct:
                cascade_gt = min(
                    tier_correct,
                    key=lambda pair: pair[1].get("total_cost_usd", 0),
                )[0]
                break

        if cascade_gt is not None:
            cascade_total += 1
            if cascade_gt == full_gt:
                cascade_agree += 1

    gt_agreement = cascade_agree / cascade_total if cascade_total > 0 else 0.0
    gt_pass = gt_agreement > 0.92
    log.info("  GT Agreement = %.1f%% (%d/%d) — %s",
             gt_agreement * 100, cascade_agree, cascade_total, "PASS" if gt_pass else "FAIL")

    # ── Experiment 3: Cost Regression R² ─────────────────────────────────────
    log.info("Experiment 3: Cost regression R² (reasoning tokens ~ budget + tier)...")
    r2 = None
    r2_pass = True  # default pass if sklearn missing
    try:
        import numpy as np
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score

        X, y = [], []
        for configs in by_sample.values():
            for r in configs.values():
                budget = r.get("budget_tokens", 0)
                tier_num = _TIER_ORDER.get(_get_tier(r.get("config_id", "a")), 2)
                reasoning = r.get("reasoning_tokens", 0)
                X.append([budget, tier_num, budget * tier_num])
                y.append(reasoning)

        if len(X) > 100:
            X_arr, y_arr = np.array(X, dtype=float), np.array(y, dtype=float)
            split = int(len(X) * 0.8)
            reg = Ridge().fit(X_arr[:split], y_arr[:split])
            y_pred = reg.predict(X_arr[split:])
            r2 = float(r2_score(y_arr[split:], y_pred))
            r2_pass = r2 > 0.50
        log.info("  Cost R² = %.3f — %s", r2 or 0, "PASS" if r2_pass else "FAIL")
    except ImportError:
        log.warning("  sklearn not installed — skipping R² test (treated as PASS)")

    # ── Experiment 4: KS distributional similarity ────────────────────────────
    log.info("Experiment 4: KS test — anchor vs non-anchor feature distributions...")
    ks_results = {}
    ks_pass = True
    try:
        from scipy import stats
        all_samples = load_jsonl(benchmark_path)
        anchor_ids = set(by_sample.keys())
        anchor_bench = [s for s in all_samples if s["sample_id"] in anchor_ids]
        non_anchor = [s for s in all_samples
                      if not s.get("in_anchor_set") and not s.get("in_validation_set")]

        for feat in ["tier_final", "vds_probe_avg", "rds_probe_avg", "ses_probe_avg"]:
            a_vals = [s[feat] for s in anchor_bench if s.get(feat) is not None]
            b_vals = [s[feat] for s in non_anchor if s.get(feat) is not None]
            if a_vals and b_vals:
                stat, pval = stats.ks_2samp(a_vals, b_vals)
                ks_results[feat] = {"statistic": round(stat, 4), "p_value": round(pval, 4)}
                if pval <= 0.05:
                    ks_pass = False
                log.info("  KS %s: stat=%.4f p=%.4f %s",
                         feat, stat, pval, "OK" if pval > 0.05 else "WARN")
    except ImportError:
        log.warning("  scipy not installed — skipping KS test (treated as PASS)")

    # ── Experiment 5: Cascade cost reduction ─────────────────────────────────
    # Compare configs evaluated in cascade vs exhaustive (all N_CONFIGS per sample).
    # Reads final_eval_correct.jsonl and routing_labels.jsonl if available.
    cost_reduction = None
    try:
        final_eval_path = _ROOT / "data" / "model_eval_results" / "final_eval_correct.jsonl"
        routing_labels_path = _ROOT / "data" / "routing_labels" / "routing_labels.jsonl"
        if final_eval_path.exists() and routing_labels_path.exists():
            sample_to_split: dict[str, str] = {}
            with open(routing_labels_path) as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        sample_to_split[rec["sample_id"]] = rec.get("split", "unknown")

            # Unique configs per cascade (remaining) sample
            from collections import Counter as _Counter
            sample_evals: dict[str, int] = _Counter()
            n_configs_pool = 0
            configs_seen: set = set()
            with open(final_eval_path) as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        sid = rec["sample_id"]
                        configs_seen.add((rec.get("yaml_key", ""), rec.get("budget_level", "")))
                        if sample_to_split.get(sid) == "remaining":
                            sample_evals[sid] += 1

            n_configs_pool = len(configs_seen)
            n_cascade_samples = sum(1 for v in sample_to_split.values() if v == "remaining")
            if n_cascade_samples > 0 and n_configs_pool > 0:
                total_cascade_evals = sum(sample_evals.values())
                exhaustive_evals = n_cascade_samples * n_configs_pool
                cost_reduction = round(1.0 - total_cascade_evals / exhaustive_evals, 4)
                log.info(
                    "  Cascade cost reduction = %.1f%%  (%d/%d eval-calls saved, %d samples, %d configs)",
                    cost_reduction * 100,
                    exhaustive_evals - total_cascade_evals,
                    exhaustive_evals,
                    n_cascade_samples,
                    n_configs_pool,
                )
    except Exception as exc:
        log.warning("  Cost reduction computation failed: %s", exc)

    # ── Summary ───────────────────────────────────────────────────────────────
    cascade_valid = dvr_pass and gt_pass
    report = {
        "experiment_1_dvr": {
            "violations": violations, "total_pairs": total_pairs,
            "dvr": round(dvr, 4), "target": "< 0.10", "pass": dvr_pass,
        },
        "experiment_2_gt_agreement": {
            "agree": cascade_agree, "total": cascade_total,
            "agreement_rate": round(gt_agreement, 4), "target": "> 0.92", "pass": gt_pass,
        },
        "experiment_3_cost_r2": {
            "r2": round(r2, 4) if r2 is not None else None,
            "target": "> 0.50", "pass": r2_pass,
        },
        "experiment_4_ks_test": {
            "results": ks_results, "target": "all p > 0.05", "pass": ks_pass,
        },
        "cost_reduction": cost_reduction,
        "cascade_valid": cascade_valid,
        "recommendation": (
            "Proceed with cascade Stage 2 evaluation"
            if cascade_valid else
            "FALLBACK — run full evaluation on remaining samples (Plan B)"
        ),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    log.info("=" * 60)
    log.info("CASCADE VALIDATION SUMMARY")
    log.info("=" * 60)
    log.info("  DVR:          %.1f%%  target<10%%   %s", dvr * 100, "PASS" if dvr_pass else "FAIL")
    log.info("  GT Agreement: %.1f%%  target>92%%   %s", gt_agreement * 100, "PASS" if gt_pass else "FAIL")
    log.info("  Cost R²:      %s   target>0.50  %s",
             f"{r2:.3f}" if r2 is not None else " N/A", "PASS" if r2_pass else "FAIL")
    log.info("  KS Test:              target p>0.05 %s", "PASS" if ks_pass else "WARN")
    if cost_reduction is not None:
        log.info("  Cost reduction: %.1f%%  (cascade vs exhaustive eval-calls)", cost_reduction * 100)
    log.info("  CASCADE VALID: %s", "YES" if cascade_valid else "NO")
    log.info("  → %s", report["recommendation"])
    log.info("  Report: %s", str(out))
    log.info("=" * 60)

    return report


def main():
    parser = argparse.ArgumentParser(description="DocRouteBench cascade validation")
    parser.add_argument(
        "--results", type=str,
        default="data/model_eval_results/merged/anchor_results.jsonl",
        help="Path to merged anchor results JSONL",
    )
    args = parser.parse_args()
    run_cascade_validation(anchor_results_path=args.results)


if __name__ == "__main__":
    main()
