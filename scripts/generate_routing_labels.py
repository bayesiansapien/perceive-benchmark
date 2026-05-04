#!/usr/bin/env python3
"""
DocRouteBench: Routing Label Generator

Combines all evaluation signals (rule-based, neural judge, oracle) to produce:
  1. data/model_eval_results/final_eval_correct.jsonl
       One record per (sample_id, yaml_key, budget_level).
       Fields: sample_id, yaml_key, budget_level, eval_correct, eval_signal
  2. data/routing_labels/routing_labels.jsonl
       One record per sample_id with cheapest-correct routing label.

Signal priority:  oracle  >  neural_judge  >  rule_only

Usage:
    python scripts/generate_routing_labels.py
    python scripts/generate_routing_labels.py --dry-run   # summary only, no files written
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

RESULTS_DIR  = _ROOT / "data" / "model_eval_results"
LABELS_DIR   = _ROOT / "data" / "routing_labels"
BENCHMARK    = _ROOT / "data" / "benchmark" / "benchmark_5000.jsonl"

# ── Input paths ──────────────────────────────────────────────────────────────

JV2_PATH      = RESULTS_DIR / "all_models_judgments_v2.jsonl"
JUDGMENTS_PATH = RESULTS_DIR / "all_models_judgments.jsonl"   # fallback (nano_correct)
ORACLE_PATH   = RESULTS_DIR / "oracle_verdicts.jsonl"
ANCHOR_PATH     = RESULTS_DIR / "api_results_anchor.jsonl"
VALIDATION_PATH = RESULTS_DIR / "api_results_validation.jsonl"
REMAINING_PATH  = RESULTS_DIR / "api_results_remaining.jsonl"

# ── Output paths ─────────────────────────────────────────────────────────────

FINAL_EVAL_PATH   = RESULTS_DIR / "final_eval_correct.jsonl"
ROUTING_OUT_PATH  = LABELS_DIR / "routing_labels.jsonl"

# ── Cost table (input cost per 1M tokens) ────────────────────────────────────
# Used only for tie-breaking within the same tier level.

COST = {
    "a2_flashlite":  0.05,
    "a4_gpt54nano":  0.04,
    "b1_gpt54mini":  0.20,
    "b3_sonnet":     3.00,
    "c1_gpt54":      3.00,
    "c2_opus":      15.00,
    "c3_gemini_pro": 1.25,
}

TIER_ORDER = {"a": 1, "b": 2, "c": 3}


def _tier(yaml_key: str) -> int:
    """Return numeric tier for a yaml_key (a→1, b→2, c→3)."""
    return TIER_ORDER.get(yaml_key[0].lower(), 9)


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_benchmark() -> dict[str, dict]:
    bench: dict[str, dict] = {}
    for rec in _load_jsonl(BENCHMARK):
        bench[rec["sample_id"]] = rec
    return bench


def _sample_split(sample: dict) -> str:
    if sample.get("in_anchor_set"):
        return "anchor"
    if sample.get("in_validation_set"):
        return "validation"
    return "remaining"


# ── Signal resolution ─────────────────────────────────────────────────────────

def _load_oracle_verdicts() -> dict[tuple, dict]:
    """key: (sample_id, yaml_key, budget_level) -> record"""
    oracle: dict[tuple, dict] = {}
    for r in _load_jsonl(ORACLE_PATH):
        key = (r["sample_id"], r["yaml_key"], r.get("budget_level", "?"))
        oracle[key] = r
    return oracle


def _load_judgments_v2() -> dict[tuple, dict]:
    """
    Preferred: all_models_judgments_v2.jsonl (has eval_correct + needs_oracle).
    Fallback:  all_models_judgments.jsonl    (has nano_correct → treat as neural).
    key: (sample_id, yaml_key, budget_level) -> record
    """
    # Try v2 first
    records = _load_jsonl(JV2_PATH)
    if records:
        print(f"  Loaded {len(records):,} records from judgments_v2", file=sys.stderr)
        out: dict[tuple, dict] = {}
        for r in records:
            key = (r["sample_id"], r["yaml_key"], r.get("budget_level", "?"))
            out[key] = r
        return out

    # Fallback: original judgments with nano_correct field
    records = _load_jsonl(JUDGMENTS_PATH)
    if records:
        print(
            f"  judgments_v2 not found, falling back to all_models_judgments.jsonl "
            f"({len(records):,} records, using nano_correct as neural signal)",
            file=sys.stderr,
        )
        out = {}
        for r in records:
            key = (r["sample_id"], r["yaml_key"], r.get("budget_level", "?"))
            # Synthesise eval_correct from nano_correct if available, else rule
            nano_c = r.get("nano_correct")
            rule_c = r.get("rule_correct", False)
            if nano_c is None:
                eval_correct = rule_c
            else:
                eval_correct = bool(nano_c)
            # Emit a normalised record
            out[key] = {
                **r,
                "neural_correct": nano_c,
                "eval_correct": eval_correct,
                "needs_oracle": False,
            }
        return out

    print("  WARNING: no judgments file found, using rule-only fallback", file=sys.stderr)
    return {}


def _load_raw_results() -> dict[tuple, dict]:
    """
    Load rule-based is_correct from raw API result files.
    key: (sample_id, yaml_key, budget_level) -> record
    Used as last-resort fallback when no neural judgment exists.
    """
    raw: dict[tuple, dict] = {}
    for path in [ANCHOR_PATH, VALIDATION_PATH, REMAINING_PATH]:
        for r in _load_jsonl(path):
            key = (r["sample_id"], r["yaml_key"], r.get("budget_level", "?"))
            raw[key] = r
    return raw


def get_final_eval_correct(
    key: tuple,
    judgments_v2: dict[tuple, dict],
    oracle_verdicts: dict[tuple, dict],
    raw_results: dict[tuple, dict],
) -> tuple[bool, str]:
    """
    Returns (eval_correct: bool, signal: str).
    signal is one of: 'oracle', 'neural', 'rule'
    Priority: oracle > neural_judge > rule_only
    """
    # 1. Oracle (highest authority)
    if key in oracle_verdicts:
        ov = oracle_verdicts[key]
        return bool(ov.get("eval_correct", ov.get("oracle_correct", False))), "oracle"

    # 2. Neural judge
    if key in judgments_v2:
        j = judgments_v2[key]
        return bool(j.get("eval_correct", False)), "neural"

    # 3. Rule-only fallback from raw results
    if key in raw_results:
        return bool(raw_results[key].get("is_correct", False)), "rule"

    # 4. Unknown: treat as incorrect
    return False, "rule"


# ── Cheapest-correct routing ──────────────────────────────────────────────────

def cheapest_correct(
    configs: list[tuple[str, str, bool]],   # (yaml_key, budget_level, eval_correct)
) -> tuple[str | None, str | None]:
    """
    From all (yaml_key, budget_level, eval_correct) for a sample:
    - Keep only correct ones
    - Return (yaml_key, budget_level) for the cheapest by (tier, cost_per_M)
    - Returns (None, None) if nothing is correct (unroutable)
    """
    correct = [
        (yk, bl, _tier(yk), COST.get(yk, 999.0))
        for yk, bl, ec in configs
        if ec
    ]
    if not correct:
        return None, None
    best = min(correct, key=lambda x: (x[2], x[3]))
    return best[0], best[1]   # yaml_key, budget_level


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    print("Loading data...", file=sys.stderr)
    bench        = _load_benchmark()
    oracle       = _load_oracle_verdicts()
    judgments_v2 = _load_judgments_v2()
    raw_results  = _load_raw_results()

    print(
        f"  Benchmark: {len(bench):,} samples | "
        f"Oracle: {len(oracle):,} | "
        f"Judgments: {len(judgments_v2):,} | "
        f"Raw results: {len(raw_results):,}",
        file=sys.stderr,
    )

    # Collect ALL (sample_id, yaml_key, budget_level) keys across every source
    all_keys: set[tuple] = set()
    all_keys.update(oracle.keys())
    all_keys.update(judgments_v2.keys())
    all_keys.update(raw_results.keys())

    # Resolve eval_correct for every key
    final_records: list[dict] = []
    for key in sorted(all_keys):
        sample_id, yaml_key, budget_level = key
        eval_correct, signal = get_final_eval_correct(
            key, judgments_v2, oracle, raw_results
        )
        sample = bench.get(sample_id, {})
        final_records.append({
            "sample_id":     sample_id,
            "yaml_key":      yaml_key,
            "budget_level":  budget_level,
            "eval_correct":  eval_correct,
            "eval_signal":   signal,
            "source_dataset": sample.get("source_dataset", ""),
            "task_type":     sample.get("task_type", ""),
            "tier_final":    sample.get("tier_final", 0),
        })

    print(f"  Resolved {len(final_records):,} (sample, config) pairs", file=sys.stderr)

    # Group by sample_id for routing label generation
    by_sample: dict[str, list[tuple[str, str, bool]]] = defaultdict(list)
    signal_by_sample: dict[str, str] = {}    # highest-priority signal seen per sample
    _signal_priority = {"oracle": 3, "neural": 2, "rule": 1}

    for r in final_records:
        sid = r["sample_id"]
        by_sample[sid].append((r["yaml_key"], r["budget_level"], r["eval_correct"]))
        prev = signal_by_sample.get(sid, "rule")
        if _signal_priority[r["eval_signal"]] > _signal_priority[prev]:
            signal_by_sample[sid] = r["eval_signal"]

    # Build routing labels
    routing_records: list[dict] = []
    for sample_id, configs in sorted(by_sample.items()):
        sample = bench.get(sample_id, {})
        best_yk, best_bl = cheapest_correct(configs)
        n_correct = sum(1 for _, _, ec in configs if ec)

        best_tier: int | None = None
        if best_yk is not None:
            best_tier = _tier(best_yk)

        routing_records.append({
            "sample_id":               sample_id,
            "task_type":               sample.get("task_type", ""),
            "tier_final":              sample.get("tier_final", 0),
            "source_dataset":          sample.get("source_dataset", ""),
            "cheapest_correct_model":  best_yk,
            "cheapest_correct_budget": best_bl,
            "cheapest_correct_tier":   best_tier,
            "is_routable":             best_yk is not None,
            "n_correct_configs":       n_correct,
            "split":                   _sample_split(sample),
            "eval_signal":             signal_by_sample.get(sample_id, "rule"),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(routing_records)
    routable = sum(1 for r in routing_records if r["is_routable"])
    unroutable = total - routable

    tier_counts: dict[int | None, int] = defaultdict(int)
    for r in routing_records:
        tier_counts[r["cheapest_correct_tier"]] += 1

    print()
    print("=" * 60)
    print("ROUTING LABEL SUMMARY")
    print("=" * 60)
    print(f"  Total samples:   {total:,}")
    print(f"  Routable:        {routable:,}  ({routable/total*100:.1f}%)")
    print(f"  Unroutable:      {unroutable:,}  ({unroutable/total*100:.1f}%)")
    print()
    print("  Cheapest correct tier distribution (routable only):")
    for tier_label, tier_num in [("Tier A (cheapest)", 1), ("Tier B", 2), ("Tier C (frontier)", 3)]:
        n = tier_counts.get(tier_num, 0)
        pct = n / routable * 100 if routable else 0
        print(f"    {tier_label:<28} {n:>5}  ({pct:.1f}%)")
    print()

    # Per-model accuracy using eval_correct
    model_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in final_records:
        yk = r["yaml_key"]
        model_stats[yk]["total"] += 1
        if r["eval_correct"]:
            model_stats[yk]["correct"] += 1

    print(f"  {'Model':<25} {'N':>7}  {'Acc':>8}")
    print("  " + "-" * 45)
    for yk in sorted(model_stats, key=lambda k: (_tier(k), COST.get(k, 999))):
        s = model_stats[yk]
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"  {yk:<25} {s['total']:>7,}  {acc:>7.1f}%")
    print("=" * 60)

    if dry_run:
        print("  [DRY RUN] No files written.")
        return

    # ── Write outputs ─────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(FINAL_EVAL_PATH, "w") as f:
        for r in final_records:
            f.write(json.dumps(r) + "\n")
    print(f"\n  Written: {FINAL_EVAL_PATH}  ({len(final_records):,} records)")

    with open(ROUTING_OUT_PATH, "w") as f:
        for r in routing_records:
            f.write(json.dumps(r) + "\n")
    print(f"  Written: {ROUTING_OUT_PATH}  ({len(routing_records):,} records)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate final eval_correct and routing labels from all evaluation signals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary only; do not write output files.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(dry_run=args.dry_run)
