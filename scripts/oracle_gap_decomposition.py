#!/usr/bin/env python3
"""
Oracle gap decomposition (task 1.4).

Decomposes the 17.7pp gap between PERCEIVE-IPS (61.6%) and oracle (79.3%)
into structural causes using rule-based routing policies on the validation set.
Fit on anchor set, evaluate on validation set. No ML training needed.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS_FILE = ROOT / "data/routing_labels/routing_labels.jsonl"
OUT_FILE    = ROOT / "data/oracle_gap_decomposition.json"

# PERCEIVE-IPS accuracy: read from trained router results if available,
# otherwise fall back to the paper-reported figure.
_IPS_RESULTS = ROOT / "results" / "router" / "perceive_ips_results.json"
if _IPS_RESULTS.exists():
    with open(_IPS_RESULTS) as _f:
        _ips_data = json.load(_f)
    PERCEIVE_IPS_ACC = _ips_data["mean_accuracy"]
else:
    PERCEIVE_IPS_ACC = 0.616   # paper-reported figure (run train_perceive_router.py to refresh)
TIER_NAMES = {1: "TierA", 2: "TierB", 3: "TierC"}

# ── load ─────────────────────────────────────────────────────────────────────

def load_labels():
    anchor, val = [], []
    with open(LABELS_FILE) as fh:
        for line in fh:
            r = json.loads(line)
            if r["split"] == "anchor":
                anchor.append(r)
            elif r["split"] == "validation":
                val.append(r)
    return anchor, val

# ── fit rule-based mappings on anchor set ────────────────────────────────────

def fit_mode_map(anchor: list[dict], keys: list[str]) -> dict:
    """For each unique key combo, find the mode cheapest_correct_tier.
    Unroutable samples (cheapest_correct_tier=None) are excluded from fit."""
    from collections import Counter
    counts: dict[tuple, Counter] = defaultdict(Counter)
    for r in anchor:
        if not r["is_routable"]:
            continue
        k = tuple(r[key] for key in keys)
        counts[k][r["cheapest_correct_tier"]] += 1
    return {k: c.most_common(1)[0][0] for k, c in counts.items()}

def predict(val: list[dict], mode_map: dict, keys: list[str], fallback: int = 2) -> list[dict]:
    preds = []
    for r in val:
        k = tuple(r[key] for key in keys)
        pred_tier = mode_map.get(k, fallback)
        correct = r["is_routable"] and (pred_tier == r["cheapest_correct_tier"])
        preds.append({"sample_id": r["sample_id"], "pred_tier": pred_tier,
                      "true_tier": r["cheapest_correct_tier"],
                      "is_routable": r["is_routable"], "correct": correct})
    return preds

def accuracy(preds: list[dict]) -> float:
    return sum(p["correct"] for p in preds) / len(preds)

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    anchor, val = load_labels()
    n_val = len(val)
    print(f"Anchor: {len(anchor)}, Validation: {n_val}")

    # Oracle accuracy: fraction of routable samples on validation set
    oracle_acc = sum(r["is_routable"] for r in val) / n_val
    unroutable_frac = 1.0 - oracle_acc
    print(f"\nOracle accuracy (validation):  {oracle_acc*100:.1f}%")
    print(f"Unroutable fraction:           {unroutable_frac*100:.1f}%")

    # Policy A: tier_final → cheapest_correct_tier (mode map from anchor)
    map_tier = fit_mode_map(anchor, ["tier_final"])
    preds_tier = predict(val, map_tier, ["tier_final"])
    acc_tier = accuracy(preds_tier)

    # Policy B: (tier_final, task_type) → cheapest_correct_tier
    map_tier_task = fit_mode_map(anchor, ["tier_final", "task_type"])
    preds_tier_task = predict(val, map_tier_task, ["tier_final", "task_type"])
    acc_tier_task = accuracy(preds_tier_task)

    # Policy C: oracle cheat (use true label), should equal oracle_acc
    preds_oracle = [{"correct": r["is_routable"]} for r in val]
    acc_oracle_check = sum(p["correct"] for p in preds_oracle) / n_val

    print(f"\nPolicy A (tier_final only):    {acc_tier*100:.1f}%")
    print(f"Policy B (tier + task_type):   {acc_tier_task*100:.1f}%")
    print(f"Policy C (oracle cheat check): {acc_oracle_check*100:.1f}%")
    print(f"PERCEIVE-IPS (paper figure):   {PERCEIVE_IPS_ACC*100:.1f}%")

    # ── gap decomposition ───────────────────────────────────────────────────
    gap_total         = oracle_acc - PERCEIVE_IPS_ACC
    gap_unroutable    = unroutable_frac                        # absolute ceiling
    gap_tier_signal   = oracle_acc - acc_tier                  # gap from tier_final alone
    gap_task_marginal = acc_tier - acc_tier_task               # marginal gain from task_type
    gap_residual      = acc_tier_task - PERCEIVE_IPS_ACC       # trained-router deficit

    print(f"""
═══ Oracle Gap Decomposition ═══════════════════════════════════════════
  Oracle accuracy:               {oracle_acc*100:.1f}%
  PERCEIVE-IPS:                  {PERCEIVE_IPS_ACC*100:.1f}%
  Total gap:                     {gap_total*100:.1f}pp

  Unroutable ceiling (no policy can help):  {gap_unroutable*100:.1f}%
  Gap with tier_final only:                 {gap_tier_signal*100:.1f}pp  (oracle − tier-only)
  Marginal gain from task_type:             {gap_task_marginal*100:.1f}pp (tier-only − tier+task)
  Trained-router residual:                  {gap_residual*100:.1f}pp  (tier+task − PERCEIVE-IPS)

Interpretation:
  {gap_unroutable*100:.0f}% of queries have no correct model (irreducible ceiling).
  A tier_final heuristic alone closes {(gap_total - gap_tier_signal)/gap_total*100:.0f}% of the gap.
  Adding task_type adds {gap_task_marginal*100:.1f}pp.
  The residual {gap_residual*100:.1f}pp reflects what the trained router under-captures
  relative to the joint (tier, task) heuristic on the validation set.
════════════════════════════════════════════════════════════════════════""")

    # Per-tier_final accuracy on validation set (diagnostic)
    print("\nPer tier_final routing accuracy (validation):")
    for tf in [1, 2, 3]:
        subset = [r for r in val if r["tier_final"] == tf]
        if not subset:
            continue
        routable = [r for r in subset if r["is_routable"]]
        mode_pred = map_tier.get((tf,), 2)
        correct = sum(1 for r in subset if r["is_routable"] and r["cheapest_correct_tier"] == mode_pred)
        print(f"  tier_final={tf}: n={len(subset)}, routable={len(routable)} "
              f"({len(routable)/len(subset)*100:.0f}%), "
              f"tier_only_acc={correct/len(subset)*100:.1f}%, mode_pred→Tier{mode_pred}")

    # Per-task_type breakdown
    print("\nOracle accuracy by task_type (validation):")
    task_stats = defaultdict(lambda: {"n": 0, "routable": 0})
    for r in val:
        task_stats[r["task_type"]]["n"] += 1
        task_stats[r["task_type"]]["routable"] += int(r["is_routable"])
    for tt in sorted(task_stats):
        s = task_stats[tt]
        print(f"  {tt}: n={s['n']}, oracle_acc={s['routable']/s['n']*100:.1f}%")

    # Save results
    results = {
        "oracle_acc": oracle_acc,
        "perceive_ips_acc": PERCEIVE_IPS_ACC,
        "acc_tier_only": acc_tier,
        "acc_tier_task": acc_tier_task,
        "gap_total_pp": round(gap_total * 100, 2),
        "gap_unroutable_pct": round(unroutable_frac * 100, 2),
        "gap_tier_signal_pp": round(gap_tier_signal * 100, 2),
        "gap_task_marginal_pp": round(gap_task_marginal * 100, 2),
        "gap_residual_pp": round(gap_residual * 100, 2),
        "n_val": n_val,
    }
    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {OUT_FILE}")


if __name__ == "__main__":
    main()
