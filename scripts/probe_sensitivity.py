#!/usr/bin/env python3
"""
DocRouteBench — Probe Sensitivity Analysis

Validates that the Phase 2 sampling pipeline is robust to probe model choice.
Four experiments using existing probe data (zero additional API cost).

Usage:
    python scripts/probe_sensitivity.py
    python scripts/probe_sensitivity.py --probe-path data/processed/probe_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.schema import load_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("probe_sensitivity")


def _load_paired_probes(probe_path: str) -> dict[str, dict[str, dict]]:
    """Load probe results indexed by sample_id -> {model_key: record}."""
    _MODEL_MAP = {
        "gpt52": "gpt52",
        "gemini25flash": "gemini_flash",
        "gemini_flash": "gemini_flash",
    }
    index: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in load_jsonl(probe_path):
        sid = r.get("sample_id", "")
        raw_mid = r.get("model_id", "")
        if not sid or not raw_mid or r.get("error"):
            continue
        key = _MODEL_MAP.get(raw_mid)
        if key:
            index[sid][key] = r
    return dict(index)


def _tier_from_probes(sample_probes: dict[str, dict], drop_model: str | None = None) -> int | None:
    """Compute tier from probe results, optionally dropping one model."""
    probes = {k: v for k, v in sample_probes.items() if k != drop_model}
    if not probes:
        return None

    correct_values = [m.get("is_correct") for m in probes.values() if m.get("is_correct") is not None]
    if not correct_values:
        return None

    vds_scores = [m.get("vds_probe", 2) for m in probes.values()]
    rds_scores = [m.get("rds_probe", 2) for m in probes.values()]
    ses_scores = [m.get("ses_probe", 2) for m in probes.values()]

    vds_avg = sum(vds_scores) / len(vds_scores)
    rds_avg = sum(rds_scores) / len(rds_scores)
    ses_avg = sum(ses_scores) / len(ses_scores)
    probe_composite = 0.30 * vds_avg + 0.45 * rds_avg + 0.25 * ses_avg

    both_correct = all(correct_values)
    both_wrong = not any(correct_values)

    if both_correct:
        return 1
    elif both_wrong:
        return 3 if probe_composite >= 2.8 else 2
    else:
        return 2 if probe_composite < 2.5 else 3


def exp1_single_model_ablation(probes: dict, model_to_drop: str) -> dict:
    """Recompute tiers using only one probe model. Measure tier stability."""
    baseline_tiers = {}
    ablated_tiers = {}

    for sid, models in probes.items():
        if len(models) < 2:
            continue
        baseline = _tier_from_probes(models)
        ablated = _tier_from_probes(models, drop_model=model_to_drop)
        if baseline is not None and ablated is not None:
            baseline_tiers[sid] = baseline
            ablated_tiers[sid] = ablated

    n = len(baseline_tiers)
    if n == 0:
        return {"model_dropped": model_to_drop, "n_samples": 0, "stability": 0.0}

    same = sum(1 for sid in baseline_tiers if baseline_tiers[sid] == ablated_tiers[sid])
    return {
        "model_dropped": model_to_drop,
        "n_samples": n,
        "same_tier": same,
        "stability": round(same / n, 4),
        "tier_shift_distribution": dict(
            sorted(
                defaultdict(int, {
                    f"{baseline_tiers[sid]}->{ablated_tiers[sid]}": 0
                    for sid in baseline_tiers
                }).items()
            )
        ) if False else _compute_shifts(baseline_tiers, ablated_tiers),
    }


def _compute_shifts(baseline: dict, ablated: dict) -> dict:
    shifts = defaultdict(int)
    for sid in baseline:
        shifts[f"T{baseline[sid]}->T{ablated[sid]}"] += 1
    return dict(sorted(shifts.items()))


def exp3_random_perturbation(probes: dict, flip_rate: float = 0.10, n_trials: int = 100) -> dict:
    """Randomly flip is_correct for flip_rate of samples, measure tier stability."""
    rng = random.Random(42)
    stabilities = []

    for trial in range(n_trials):
        perturbed_same = 0
        perturbed_total = 0

        for sid, models in probes.items():
            if len(models) < 2:
                continue

            baseline = _tier_from_probes(models)
            if baseline is None:
                continue

            # Create perturbed copy
            perturbed = {}
            for k, m in models.items():
                pm = dict(m)
                if rng.random() < flip_rate:
                    pm["is_correct"] = not pm.get("is_correct", False)
                perturbed[k] = pm

            perturbed_tier = _tier_from_probes(perturbed)
            if perturbed_tier is not None:
                perturbed_total += 1
                if baseline == perturbed_tier:
                    perturbed_same += 1

        if perturbed_total > 0:
            stabilities.append(perturbed_same / perturbed_total)

    import statistics
    return {
        "flip_rate": flip_rate,
        "n_trials": n_trials,
        "stability_mean": round(statistics.mean(stabilities), 4) if stabilities else 0,
        "stability_std": round(statistics.stdev(stabilities), 4) if len(stabilities) > 1 else 0,
        "stability_min": round(min(stabilities), 4) if stabilities else 0,
        "stability_max": round(max(stabilities), 4) if stabilities else 0,
    }


def exp4_vds_rds_ses_agreement(probes: dict) -> dict:
    """Compare VDS/RDS/SES ratings between the two probe models."""
    agreements = {"VDS": 0, "RDS": 0, "SES": 0}
    totals = {"VDS": 0, "RDS": 0, "SES": 0}

    for sid, models in probes.items():
        if "gpt52" not in models or "gemini_flash" not in models:
            continue

        g = models["gpt52"]
        f = models["gemini_flash"]

        for axis, field in [("VDS", "vds_label"), ("RDS", "rds_label"), ("SES", "ses_label")]:
            g_val = g.get(field, "")
            f_val = f.get(field, "")
            if g_val and f_val:
                totals[axis] += 1
                if g_val == f_val:
                    agreements[axis] += 1

    result = {}
    for axis in ["VDS", "RDS", "SES"]:
        n = totals[axis]
        agree = agreements[axis]
        result[axis] = {
            "n_compared": n,
            "exact_match": agree,
            "agreement_rate": round(agree / n, 4) if n > 0 else 0,
        }

    return result


def run_sensitivity_analysis(probe_path: str, output_path: str) -> dict:
    """Run all 4 experiments and save report."""
    log.info("Loading probe results from %s ...", probe_path)
    probes = _load_paired_probes(probe_path)
    paired = {sid: m for sid, m in probes.items() if len(m) >= 2}
    log.info("Loaded %d total, %d paired (both models) samples.", len(probes), len(paired))

    report = {}

    # Exp 1: Drop Gemini Flash
    log.info("Experiment 1: Single-model ablation (drop gemini_flash)...")
    report["exp1_drop_gemini"] = exp1_single_model_ablation(paired, "gemini_flash")
    log.info("  Stability: %.1f%%", report["exp1_drop_gemini"]["stability"] * 100)

    # Exp 2: Drop GPT-5.2
    log.info("Experiment 2: Single-model ablation (drop gpt52)...")
    report["exp2_drop_gpt52"] = exp1_single_model_ablation(paired, "gpt52")
    log.info("  Stability: %.1f%%", report["exp2_drop_gpt52"]["stability"] * 100)

    # Exp 3: Random perturbation
    log.info("Experiment 3: Random perturbation (10%% flip, 100 trials)...")
    report["exp3_perturbation"] = exp3_random_perturbation(paired)
    log.info("  Stability: %.1f%% +/- %.1f%%",
             report["exp3_perturbation"]["stability_mean"] * 100,
             report["exp3_perturbation"]["stability_std"] * 100)

    # Exp 4: VDS/RDS/SES agreement
    log.info("Experiment 4: VDS/RDS/SES inter-model agreement...")
    report["exp4_complexity_agreement"] = exp4_vds_rds_ses_agreement(paired)
    for axis, data in report["exp4_complexity_agreement"].items():
        log.info("  %s agreement: %.1f%% (%d/%d)",
                 axis, data["agreement_rate"] * 100, data["exact_match"], data["n_compared"])

    # Summary
    stabilities = [
        report["exp1_drop_gemini"]["stability"],
        report["exp2_drop_gpt52"]["stability"],
        report["exp3_perturbation"]["stability_mean"],
    ]
    report["summary"] = {
        "min_stability": round(min(stabilities), 4),
        "all_above_85_pct": all(s >= 0.85 for s in stabilities),
        "conclusion": "Pipeline is robust to probe model choice" if all(s >= 0.85 for s in stabilities) else "Pipeline shows sensitivity to probe model choice — investigate",
    }

    log.info("\n=== SUMMARY ===")
    log.info("  Min stability: %.1f%%", report["summary"]["min_stability"] * 100)
    log.info("  Conclusion: %s", report["summary"]["conclusion"])

    # Save report
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    log.info("Report saved to %s", output_path)

    return report


def main():
    parser = argparse.ArgumentParser(description="DocRouteBench Probe Sensitivity Analysis")
    parser.add_argument("--probe-path", default="data/processed/probe_results.jsonl")
    parser.add_argument("--output", default="data/phase2_sensitivity_report.json")
    args = parser.parse_args()

    abs_probe = str(_PROJECT_ROOT / args.probe_path) if not Path(args.probe_path).is_absolute() else args.probe_path
    abs_output = str(_PROJECT_ROOT / args.output) if not Path(args.output).is_absolute() else args.output

    run_sensitivity_analysis(abs_probe, abs_output)


if __name__ == "__main__":
    main()
