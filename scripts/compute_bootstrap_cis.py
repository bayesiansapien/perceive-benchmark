#!/usr/bin/env python3
"""
Bootstrap 95% CIs for two cascade validity metrics:
  - DVR (Dominance Violation Rate): computed on cascade validation set
  - GT label agreement: computed on anchor set

Bootstrap unit: sample. 1,000 resamples, seed=42.
Falls back to Wilson/Clopper-Pearson if per-sample DVR status unavailable.
"""
from __future__ import annotations
import json, random, math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_FILE = ROOT / "data/model_eval_results/merged/anchor_results.jsonl"
VALIDATION_FILE = ROOT / "data/model_eval_results/api_results_validation.jsonl"
PHASE3_FILE = ROOT / "data/phase3_cascade_validation.json"

N_BOOTSTRAP = 1_000
SEED = 42
TIER_RANK = {"A": 1, "B": 2, "C": 3}

# ── data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_sample_index(rows: list[dict]) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        idx[r["sample_id"]].append(r)
    return dict(idx)


# ── metric helpers ────────────────────────────────────────────────────────────

def cheapest_correct(configs: list[dict]) -> dict | None:
    """Return the lowest-cost correct config, or None if all wrong."""
    correct = [r for r in configs if r.get("is_correct")]
    if not correct:
        return None
    return min(correct, key=lambda r: (TIER_RANK.get(r.get("tier", "C"), 9),
                                        r.get("total_cost_usd", 0.0)))


def cascade_label(configs: list[dict]) -> dict | None:
    """
    Simulate cascade: stop at first tier that has any correct config,
    return cheapest-correct in that tier.
    """
    for tier in ("A", "B", "C"):
        tier_configs = [r for r in configs if r.get("tier") == tier]
        correct = [r for r in tier_configs if r.get("is_correct")]
        if correct:
            return min(correct, key=lambda r: r.get("total_cost_usd", 0.0))
    return None  # unroutable


def sample_dvr_status(configs: list[dict]) -> bool | None:
    """
    True = dominance violation (cascade stopped too deep; exhaustive
    cheapest-correct is cheaper than cascade label).
    None = unroutable (skip).
    """
    exh = cheapest_correct(configs)
    if exh is None:
        return None  # unroutable, skip
    cas = cascade_label(configs)
    if cas is None:
        return None
    exh_tier = TIER_RANK.get(exh.get("tier", "C"), 9)
    cas_tier = TIER_RANK.get(cas.get("tier", "C"), 9)
    return cas_tier > exh_tier  # True = violation


def sample_gt_agree(configs: list[dict]) -> bool | None:
    """True = cascade label matches exhaustive label (same config_id)."""
    exh = cheapest_correct(configs)
    if exh is None:
        return None
    cas = cascade_label(configs)
    if cas is None:
        return None
    return cas.get("config_id") == exh.get("config_id")


# ── bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_proportion(flags: list[bool], n_boot: int = N_BOOTSTRAP, seed: int = SEED):
    rng = random.Random(seed)
    n = len(flags)
    pt = sum(flags) / n
    boots = []
    for _ in range(n_boot):
        sample = [rng.choice(flags) for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(n_boot * 0.025)]
    hi = boots[int(n_boot * 0.975) - 1]
    return pt, lo, hi


def wilson_ci(k: int, n: int, z: float = 1.96):
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0.0, centre - spread), min(1.0, centre + spread)


def clopper_pearson_lower(n: int, alpha: float = 0.05) -> float:
    """One-sided lower 95% CP bound for k=n (all successes)."""
    return alpha ** (1.0 / n)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # ── GT agreement on anchor set ─────────────────────────────────────────
    print(f"Loading anchor set: {ANCHOR_FILE}")
    anchor_rows = load_jsonl(ANCHOR_FILE)
    anchor_idx = build_sample_index(anchor_rows)
    n_anchor = len(anchor_idx)
    print(f"  {len(anchor_rows):,} rows, {n_anchor:,} samples")

    gt_flags = []
    for sid, cfgs in anchor_idx.items():
        status = sample_gt_agree(cfgs)
        if status is not None:
            gt_flags.append(status)

    gt_pt, gt_lo, gt_hi = bootstrap_proportion(gt_flags)
    print(f"\nGT label agreement (anchor, n={len(gt_flags):,}):")
    print(f"  Point: {gt_pt*100:.2f}%")
    print(f"  95% bootstrap CI: [{gt_lo*100:.1f}%, {gt_hi*100:.1f}%]")

    if gt_pt == 1.0:
        cp_lo = clopper_pearson_lower(len(gt_flags)) * 100
        print(f"  Clopper-Pearson lower bound (100% case): [{cp_lo:.1f}%, 100%]")

    # ── DVR on validation / anchor set ────────────────────────────────────
    dvr_source = None
    if VALIDATION_FILE.exists():
        print(f"\nLoading cascade validation set: {VALIDATION_FILE}")
        val_rows = load_jsonl(VALIDATION_FILE)
        val_idx = build_sample_index(val_rows)
        dvr_source = ("validation", val_idx)
    else:
        print(f"\nValidation file not found, computing DVR on anchor set as fallback")
        dvr_source = ("anchor", anchor_idx)

    label, idx_dvr = dvr_source
    dvr_flags = []
    for sid, cfgs in idx_dvr.items():
        status = sample_dvr_status(cfgs)
        if status is not None:
            dvr_flags.append(status)

    dvr_pt, dvr_lo, dvr_hi = bootstrap_proportion(dvr_flags)
    print(f"\nDVR ({label} set, n={len(dvr_flags):,}):")
    print(f"  Point: {dvr_pt*100:.2f}%")
    print(f"  95% bootstrap CI: [{dvr_lo*100:.1f}%, {dvr_hi*100:.1f}%]")

    # Wilson for comparison
    k_dvr = sum(dvr_flags)
    _, w_lo, w_hi = wilson_ci(k_dvr, len(dvr_flags))
    print(f"  Wilson CI (for comparison): [{w_lo*100:.1f}%, {w_hi*100:.1f}%]")

    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"DVR:  {dvr_pt*100:.1f}% (95% CI [{dvr_lo*100:.1f}%, {dvr_hi*100:.1f}%]) on {label} set, n={len(dvr_flags):,}")
    print(f"GT:   {gt_pt*100:.1f}% (95% CI [{gt_lo*100:.1f}%, {gt_hi*100:.1f}%]) on anchor set, n={len(gt_flags):,}")


if __name__ == "__main__":
    main()
