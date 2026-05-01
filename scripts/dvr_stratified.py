"""
DVR stratified analysis for the camera-ready Tier 2 numbers.

Computes three quantities the paper currently does not surface:
  (1) DVR restricted to the GT-validated subset (same population as the 100%
      per-query agreement number), reconciling per-check and per-query metrics.
  (2) DVR by task family (T1..T6).
  (3) Cost regret on the GT-validated subset.

DVR follows the paper's formal definition (Appendix Cascade Validation):
  For each fully-evaluated sample q with cascade-selected cheapest-correct
  configuration ĉ_q at cost c(ĉ_q), every config m' with c(m') < c(ĉ_q) is
  a "candidate-cheaper check"; m' is a violation if eval_correct(m', q).
  DVR = violations / total candidate-cheaper checks.

The cascade simulator follows the paper's evaluation order (Algorithm 1):
  evaluate Tier-A configs in cost order, stop at first correct; if all
  Tier-A wrong, evaluate Tier-B in cost order, stop at first correct;
  else Tier-C. The cascade-selected configuration is the first correct
  config encountered.

Usage:
    python scripts/dvr_stratified.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "data" / "benchmark" / "benchmark_5000.jsonl"
LABELS_PATH = ROOT / "data" / "routing_labels" / "routing_labels.jsonl"
EVAL_PATH = ROOT / "data" / "model_eval_results" / "final_eval_correct.jsonl"
OUT_PATH = ROOT / "data" / "paper_notes" / "dvr_stratified_results.json"


# Cost table (per-call USD), copied from scripts/router/config.py to keep
# this script self-contained.
ACTUAL_COSTS: dict[tuple[str, str], float] = {
    ("a2_flashlite", "B1"): 0.00014246,
    ("a2_flashlite", "B2"): 0.00009332,
    ("a2_flashlite", "B3"): 0.00043329,
    ("a4_gpt54nano", "B0"): 0.00004894,
    ("b1_gpt54mini", "B0"): 0.00022614,
    ("b1_gpt54mini", "B1"): 0.00043393,
    ("b1_gpt54mini", "B2"): 0.00103158,
    ("b1_gpt54mini", "B3"): 0.00192311,
    ("b3_sonnet", "B0"): 0.00328967,
    ("b3_sonnet", "B1"): 0.00390005,
    ("b3_sonnet", "B2"): 0.00483883,
    ("b3_sonnet", "B3"): 0.00526220,
    ("c1_gpt54", "B0"): 0.00316043,
    ("c1_gpt54", "B1"): 0.00643326,
    ("c1_gpt54", "B2"): 0.01486339,
    ("c1_gpt54", "B3"): 0.03047738,
    ("c2_opus", "B0"): 0.01557899,
    ("c2_opus", "B1"): 0.02181283,
    ("c2_opus", "B2"): 0.02325841,
    ("c2_opus", "B3"): 0.02409178,
    ("c3_gemini_pro", "B0"): 0.00310000,
    ("c3_gemini_pro", "B1"): 0.00498000,
    ("c3_gemini_pro", "B2"): 0.01275000,
    ("c3_gemini_pro", "B3"): 0.02380000,
}

MODEL_TIER = {
    "a2_flashlite": "A",
    "a4_gpt54nano": "A",
    "b1_gpt54mini": "B",
    "b3_sonnet": "B",
    "c1_gpt54": "C",
    "c2_opus": "C",
    "c3_gemini_pro": "C",
}

TIER_ORDER = {"A": 1, "B": 2, "C": 3}


def load_data():
    """Load benchmark, routing labels, and per-cell eval correctness."""
    bench = {}
    with open(BENCH_PATH) as f:
        for line in f:
            r = json.loads(line)
            bench[r["sample_id"]] = r

    labels = {}
    with open(LABELS_PATH) as f:
        for line in f:
            r = json.loads(line)
            labels[r["sample_id"]] = r

    eval_matrix: dict[str, dict[tuple[str, str], bool]] = defaultdict(dict)
    with open(EVAL_PATH) as f:
        for line in f:
            r = json.loads(line)
            key = (r["yaml_key"], r["budget_level"])
            eval_matrix[r["sample_id"]][key] = bool(r["eval_correct"])

    return bench, labels, eval_matrix


def simulate_cascade(
    eval_row: dict[tuple[str, str], bool],
    tier_final: int,
) -> tuple[tuple | None, str | None]:
    """Simulate the paper's Algorithm 1 cascade.

    Returns (cascade-selected config, stopping-tier letter). Stopping tier
    is the tier at which the cascade halted. Returns (None, None) if the
    cascade returns unroutable.

    Algorithm 1: evaluate all Tier-A configs in parallel. If any correct,
    select cheapest Tier-A correct. Else if tier_final == Easy, unroutable.
    Else evaluate all Tier-B. If any correct, select cheapest Tier-B
    correct. Else if tier_final == Medium, unroutable. Else evaluate all
    Tier-C and select cheapest correct overall in C, or unroutable.
    """
    by_tier = defaultdict(list)
    for cfg, ok in eval_row.items():
        cost = ACTUAL_COSTS.get(cfg)
        if cost is None:
            continue
        tier = MODEL_TIER[cfg[0]]
        by_tier[tier].append((cost, cfg, ok))

    # Tier A
    correct_a = [(c, cfg) for c, cfg, ok in by_tier["A"] if ok]
    if correct_a:
        correct_a.sort()
        return correct_a[0][1], "A"
    if tier_final == 1:
        return None, None

    # Tier B
    correct_b = [(c, cfg) for c, cfg, ok in by_tier["B"] if ok]
    if correct_b:
        correct_b.sort()
        return correct_b[0][1], "B"
    if tier_final == 2:
        return None, None

    # Tier C
    correct_c = [(c, cfg) for c, cfg, ok in by_tier["C"] if ok]
    if correct_c:
        correct_c.sort()
        return correct_c[0][1], "C"
    return None, None


def actual_cheapest_correct(eval_row: dict[tuple[str, str], bool]) -> tuple | None:
    """Cheapest correct config across all 24 (no tier preference)."""
    correct = [
        (ACTUAL_COSTS[cfg], cfg)
        for cfg, ok in eval_row.items()
        if ok and cfg in ACTUAL_COSTS
    ]
    if not correct:
        return None
    correct.sort()
    return correct[0][1]


def wilson_ci(violations: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = violations / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def compute_dvr(
    sample_ids: list[str],
    eval_matrix: dict[str, dict[tuple[str, str], bool]],
    bench: dict,
    skipped_only: bool = False,
) -> dict:
    """Compute DVR per the paper's definition over a population.

    The paper's |P| count is consistent with the inclusive interpretation:
    candidate-cheaper checks include every config m' with c(m') < c(ĉ_q),
    regardless of whether the cascade evaluated m'. A violation requires
    that m' was independently evaluated AND eval_correct(m', q) is True.

    skipped_only=True restricts to configs the cascade never evaluated
    (cross-tier overlap only). This is the stricter "label-could-be-
    dominated-by-an-unevaluated-cheaper-correct" reading.
    """
    violations = 0
    total = 0
    samples_with_violation = 0
    samples_audited = 0

    for sid in sample_ids:
        eval_row = eval_matrix.get(sid)
        if not eval_row or len(eval_row) < 23:
            continue
        tier_final = bench.get(sid, {}).get("tier_final", 3)
        cascade_pick, stop_tier = simulate_cascade(eval_row, tier_final)
        if cascade_pick is None:
            continue
        c_chat = ACTUAL_COSTS[cascade_pick]
        stop_tier_idx = TIER_ORDER[stop_tier]
        samples_audited += 1
        sample_violations = 0
        for cfg, ok in eval_row.items():
            cost = ACTUAL_COSTS.get(cfg)
            if cost is None or cost >= c_chat:
                continue
            cfg_tier_idx = TIER_ORDER[MODEL_TIER[cfg[0]]]
            if skipped_only and cfg_tier_idx <= stop_tier_idx:
                continue
            total += 1
            if ok:
                violations += 1
                sample_violations += 1
        if sample_violations > 0:
            samples_with_violation += 1

    rate = violations / total if total else 0.0
    lo, hi = wilson_ci(violations, total)
    return {
        "samples_audited": samples_audited,
        "candidate_cheaper_checks": total,
        "violations": violations,
        "dvr": round(rate, 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "samples_with_at_least_one_violation": samples_with_violation,
    }


def cost_regret(
    sample_ids: list[str],
    eval_matrix: dict[str, dict[tuple[str, str], bool]],
    bench: dict,
) -> dict:
    """Mean and max cost difference between cascade-selected and true
    cheapest-correct, in micro-USD per query, on routable samples."""
    regrets = []
    for sid in sample_ids:
        eval_row = eval_matrix.get(sid)
        if not eval_row or len(eval_row) < 23:
            continue
        tier_final = bench.get(sid, {}).get("tier_final", 3)
        cascade_pick, _ = simulate_cascade(eval_row, tier_final)
        true_pick = actual_cheapest_correct(eval_row)
        if cascade_pick is None or true_pick is None:
            continue
        diff = ACTUAL_COSTS[cascade_pick] - ACTUAL_COSTS[true_pick]
        regrets.append(diff * 1e6)
    if not regrets:
        return {"n": 0}
    arr = np.array(regrets)
    return {
        "n": len(arr),
        "mean_regret_uUSD": round(float(arr.mean()), 3),
        "max_regret_uUSD": round(float(arr.max()), 3),
        "p95_regret_uUSD": round(float(np.percentile(arr, 95)), 3),
        "n_zero_regret": int((arr == 0).sum()),
        "frac_zero_regret": round(float((arr == 0).mean()), 4),
    }


def main():
    bench, labels, eval_matrix = load_data()

    # Population A: full cascade-evaluated population
    # (anchor + validation splits, all 2,250 samples with 24-config eval)
    full_pop_ids = [
        sid for sid, r in labels.items() if r["split"] in ("anchor", "validation")
    ]

    # Population B: GT-validated subset (routable samples in full-eval pop)
    gt_validated_ids = [
        sid for sid in full_pop_ids if labels[sid]["is_routable"]
    ]

    print(f"Full cascade-evaluated population (anchor + validation): {len(full_pop_ids)}")
    print(f"GT-validated routable subset: {len(gt_validated_ids)}")
    print()

    # ── DVR on full population ───────────────────────────────────────────────
    dvr_full = compute_dvr(full_pop_ids, eval_matrix, bench)
    print("[DVR] Full cascade-evaluated population (anchor + validation):")
    print(f"  audited: {dvr_full['samples_audited']}")
    print(f"  candidate-cheaper checks: {dvr_full['candidate_cheaper_checks']:,}")
    print(f"  violations: {dvr_full['violations']:,}")
    print(f"  DVR: {dvr_full['dvr'] * 100:.2f}% (95% CI {dvr_full['ci_lo']*100:.2f}--{dvr_full['ci_hi']*100:.2f})")
    print()

    # ── DVR on GT-validated subset (same population as 100% agreement) ──────
    dvr_gt = compute_dvr(gt_validated_ids, eval_matrix, bench)
    print("[DVR] GT-validated routable subset (reconciliation with 100% per-query agreement):")
    print(f"  audited: {dvr_gt['samples_audited']}")
    print(f"  candidate-cheaper checks: {dvr_gt['candidate_cheaper_checks']:,}")
    print(f"  violations: {dvr_gt['violations']:,}")
    print(f"  DVR: {dvr_gt['dvr'] * 100:.2f}% (95% CI {dvr_gt['ci_lo']*100:.2f}--{dvr_gt['ci_hi']*100:.2f})")
    print(f"  samples with at least one violation: {dvr_gt['samples_with_at_least_one_violation']}")
    print()

    # ── DVR by task family ───────────────────────────────────────────────────
    task_rename = {"element_localization": "T6"}
    by_task = defaultdict(list)
    for sid in full_pop_ids:
        task = bench.get(sid, {}).get("task_type")
        if task:
            task = task_rename.get(task, task)
            by_task[task].append(sid)
    print("[DVR] By task family (full cascade-evaluated population):")
    per_task = {}
    for task in sorted(by_task.keys()):
        d = compute_dvr(by_task[task], eval_matrix, bench)
        per_task[task] = d
        print(
            f"  {task}: n={d['samples_audited']}  "
            f"checks={d['candidate_cheaper_checks']:,}  "
            f"DVR={d['dvr']*100:.2f}% (CI {d['ci_lo']*100:.2f}--{d['ci_hi']*100:.2f})"
        )
    print()

    # ── Cost regret on GT-validated subset ──────────────────────────────────
    regret = cost_regret(gt_validated_ids, eval_matrix, bench)
    print("[Cost regret] GT-validated routable subset:")
    print(f"  n: {regret['n']}")
    print(f"  mean: {regret['mean_regret_uUSD']:.3f} uUSD/query")
    print(f"  max:  {regret['max_regret_uUSD']:.3f} uUSD/query")
    print(f"  p95:  {regret['p95_regret_uUSD']:.3f} uUSD/query")
    print(f"  zero-regret samples: {regret['n_zero_regret']} ({regret['frac_zero_regret']*100:.2f}%)")
    print()

    out = {
        "dvr_full_cascade_population": dvr_full,
        "dvr_gt_validated_subset": dvr_gt,
        "dvr_by_task": per_task,
        "cost_regret_gt_validated_subset": regret,
        "population_sizes": {
            "full_cascade_evaluated": len(full_pop_ids),
            "gt_validated_routable": len(gt_validated_ids),
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
