"""
DocRouteBench: Phase 2 Structural Prior
=========================================
Assigns rule-based VDS / RDS / SES estimates and a P(Tier1/2/3) prior to
every sample in the deduped JSONL, producing an enriched output file.

Added fields per sample:
  vds_est           int   1-4   Visual Dependency Score estimate
  rds_est           int   1-4   Reasoning Depth Score estimate
  ses_est           int   1-4   Spatial Extent Score estimate
  composite_est     float       0.30*VDS + 0.45*RDS + 0.25*SES
  tier_prior        int   1-3   hard tier assignment
  tier_prior_soft   list[float] [P(T1), P(T2), P(T3)]

Rules follow neurips-plan.md §6 annotation rubric plus dataset-level hard priors.

Checkpoint: data/phase2_checkpoints/structural_prior.done
Input:      data/processed/all_samples_deduped.jsonl
Output:     data/processed/samples_with_prior.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── Keyword sets ───────────────────────────────────────────────────────────────

_SPATIAL_KEYWORDS = frozenset([
    "where", "position", "location", "top", "bottom", "left", "right",
    "corner", "region", "area",
    "locate", "coordinates", "bbox", "bounding box",
])

_CALC_COMPARISON_KEYWORDS = frozenset([
    "compare", "calculate", "sum", "average", "difference", "ratio",
    "trend", "total", "how many", "increase", "decrease", "percent",
    "highest", "lowest", "smallest", "largest", "minimum", "maximum",
    "greater than", "less than", "more than",
])

_MULTI_DOC_KEYWORDS = frozenset([
    "based on", "according to both", "considering",
    "cross-reference", "across pages", "between page",
    "which page", "what page", "page number",
])

_SPATIAL_SPREAD_KEYWORDS = frozenset([
    "across", "throughout", "span", "spread", "entire", "whole document",
    "multiple sections", "each section",
])


# ── Tier boundary constants ────────────────────────────────────────────────────
_TIER1_MAX = 2.2   # composite < 2.2  → Tier 1
_TIER2_MAX = 3.0   # composite < 3.0  → Tier 2; else Tier 3 (lowered from 3.4 for balance)

# Base soft probabilities per tier
_BASE_SOFT: dict[int, List[float]] = {
    1: [0.70, 0.25, 0.05],
    2: [0.15, 0.65, 0.20],
    3: [0.05, 0.25, 0.70],
}


# ── Helper: keyword presence ──────────────────────────────────────────────────

def _query_lower(query: str) -> str:
    return query.lower()


def _has_any(query_lower: str, keywords: frozenset) -> bool:
    """True if the lowercased query contains any keyword (substring match)."""
    return any(kw in query_lower for kw in keywords)


def _word_count(query: str) -> int:
    return len(query.split())


# ── VDS estimation ────────────────────────────────────────────────────────────

def _estimate_vds(record: dict) -> int:
    """
    Visual Dependency Score (1-4).
      4, has_chart AND has_figure (cross-modal fusion)
      3, has_chart OR has_figure
      2, spatial keywords present in query
      1, otherwise
    """
    has_chart = record.get("has_chart", False)
    has_figure = record.get("has_figure", False)
    if has_chart and has_figure:
        return 4
    if has_chart or has_figure:
        return 3
    ql = _query_lower(record.get("query", ""))
    if _has_any(ql, _SPATIAL_KEYWORDS):
        return 2
    return 1


# ── RDS estimation ────────────────────────────────────────────────────────────

def _estimate_rds(record: dict) -> int:
    """
    Reasoning Depth Score (1-4).
      4, cross-document / multi-page reference keywords
      3, calculation or comparison keywords
      1, short query (<8 words) with no calc/comparison keywords
      2, otherwise
    """
    query = record.get("query", "")
    ql = _query_lower(query)

    if _has_any(ql, _MULTI_DOC_KEYWORDS):
        return 4
    if _has_any(ql, _CALC_COMPARISON_KEYWORDS):
        return 3
    if _word_count(query) < 8 and not _has_any(ql, _CALC_COMPARISON_KEYWORDS):
        return 1
    return 2


# ── SES estimation ────────────────────────────────────────────────────────────

def _estimate_ses(record: dict) -> int:
    """
    Spatial Extent Score (1-4).
      4, num_pages > 1
      3, (has_table AND has_figure) OR spatial-spread keywords in query
      2, has_table OR has_figure
      1, otherwise
    """
    num_pages = record.get("num_pages", 1)
    if num_pages > 1:
        return 4
    has_table = record.get("has_table", False)
    has_figure = record.get("has_figure", False)
    ql = _query_lower(record.get("query", ""))
    if (has_table and has_figure) or _has_any(ql, _SPATIAL_SPREAD_KEYWORDS):
        return 3
    if has_table or has_figure:
        return 2
    return 1


# ── Composite + Tier ──────────────────────────────────────────────────────────

def _compute_composite(vds: int, rds: int, ses: int) -> float:
    return round(0.30 * vds + 0.45 * rds + 0.25 * ses, 4)


def _composite_to_tier(composite: float) -> int:
    if composite < _TIER1_MAX:
        return 1
    if composite < _TIER2_MAX:
        return 2
    return 3


# ── Soft probability conversion ───────────────────────────────────────────────

def _tier_to_soft(tier: int, composite: float) -> List[float]:
    """
    Convert hard tier to soft [P(T1), P(T2), P(T3)].
    Applies ±0.10 variance based on how far the composite is from tier
    boundaries, then normalises to sum to 1.0.
    """
    base = list(_BASE_SOFT[tier])   # copy

    if tier == 1:
        # Distance from upper boundary _TIER1_MAX
        # Further below boundary → more confident in T1
        dist = _TIER1_MAX - composite          # positive: well inside T1
        delta = min(0.10, max(-0.10, dist * 0.10))
        base[0] = min(0.95, max(0.50, base[0] + delta))
        # Redistribute remainder to T2/T3 proportionally
        remainder = 1.0 - base[0]
        t2_share = _BASE_SOFT[1][1] / (_BASE_SOFT[1][1] + _BASE_SOFT[1][2])
        base[1] = round(remainder * t2_share, 4)
        base[2] = round(remainder * (1 - t2_share), 4)

    elif tier == 2:
        # Distance from both boundaries
        dist_lower = composite - _TIER1_MAX    # positive: distance from T1 boundary
        dist_upper = _TIER2_MAX - composite    # positive: distance from T3 boundary
        # If close to T1 boundary, shift some probability to T1
        if dist_lower < 0.5:
            shift = min(0.10, (0.5 - dist_lower) * 0.20)
            base[0] = min(0.40, base[0] + shift)
            base[1] = max(0.40, base[1] - shift)
        # If close to T3 boundary, shift some probability to T3
        if dist_upper < 0.5:
            shift = min(0.10, (0.5 - dist_upper) * 0.20)
            base[2] = min(0.40, base[2] + shift)
            base[1] = max(0.40, base[1] - shift)

    else:  # tier == 3
        # Distance from lower boundary _TIER2_MAX
        dist = composite - _TIER2_MAX          # positive: well inside T3
        delta = min(0.10, max(-0.10, dist * 0.10))
        base[2] = min(0.95, max(0.50, base[2] + delta))
        remainder = 1.0 - base[2]
        t1_share = _BASE_SOFT[3][0] / (_BASE_SOFT[3][0] + _BASE_SOFT[3][1])
        base[0] = round(remainder * t1_share, 4)
        base[1] = round(remainder * (1 - t1_share), 4)

    # Normalise (guard against floating-point drift)
    total = sum(base)
    normalised = [round(p / total, 4) for p in base]
    # Force exact sum to 1.0 by adjusting the dominant component
    diff = 1.0 - sum(normalised)
    dominant = normalised.index(max(normalised))
    normalised[dominant] = round(normalised[dominant] + diff, 4)
    return normalised


# ── Dataset-level hard priors ─────────────────────────────────────────────────

def _apply_dataset_priors(
    record: dict,
    vds: int,
    rds: int,
    ses: int,
    tier: int,
) -> Tuple[int, int, int, int]:
    """
    Apply dataset-level overrides.  Returns (vds, rds, ses, tier).
    Composite is re-derived by caller after this function.
    """
    src = record.get("source_dataset", "").lower().replace("-", "").replace(" ", "")

    # rvlcdip, classification is always easy
    if src in ("rvlcdip", "rvl-cdip", "rvlcdip"):
        tier = 1

    # MP-DocVQA: multi-page → ses always 4, weight toward tier 2-3
    if src in ("mpdocvqa", "mp-docvqa"):
        ses = 4
        if tier == 1:
            tier = 2   # can't be easy if it's multi-page

    # SlideVQA: arithmetic subtypes → tier 2; single-slide → tier 1
    if src == "slidevqa":
        query = record.get("query", "").lower()
        is_arithmetic = _has_any(query, _CALC_COMPARISON_KEYWORDS)
        if is_arithmetic:
            tier = max(tier, 2)
        else:
            tier = min(tier, 2)   # single-slide at most tier 2

    # CORD / SROIE: receipts; at most tier 2
    if src in ("cord", "sroie"):
        tier = min(tier, 2)

    # TabFact: always requires table reading, rds >= 2
    if src == "tabfact":
        rds = max(rds, 2)

    # WikiTableQuestions: multi-step table reasoning, rds >= 3
    if src in ("wtq", "wikitablequestions"):
        rds = max(rds, 3)

    return vds, rds, ses, tier


# ── Per-sample prior computation ──────────────────────────────────────────────

def _compute_prior(record: dict) -> dict:
    """
    Compute all prior fields for a single sample record.
    Returns a dict of new fields to merge into the record.
    """
    vds = _estimate_vds(record)
    rds = _estimate_rds(record)
    ses = _estimate_ses(record)

    composite = _compute_composite(vds, rds, ses)
    tier = _composite_to_tier(composite)

    # Dataset-level overrides
    vds, rds, ses, tier = _apply_dataset_priors(record, vds, rds, ses, tier)

    # Recompute composite after overrides
    composite = _compute_composite(vds, rds, ses)

    # If dataset overrides changed the tier, respect it; otherwise recompute
    # (dataset priors may set tier directly, which may not match composite).
    # We keep the dataset-overridden tier for tier_prior but still store the
    # rule-derived composite for transparency.
    tier_for_soft = tier  # use overridden tier for soft probs

    soft = _tier_to_soft(tier_for_soft, composite)

    return {
        "vds_est": vds,
        "rds_est": rds,
        "ses_est": ses,
        "composite_est": composite,
        "tier_prior": tier,
        "tier_prior_soft": soft,
    }


# ── Main pipeline function ────────────────────────────────────────────────────

def run_structural_prior(
    input_path: str = "data/processed/all_samples_deduped.jsonl",
    output_path: str = "data/processed/samples_with_prior.jsonl",
    checkpoint_dir: str = "data/phase2_checkpoints",
) -> str:
    """
    Annotate every sample with structural prior fields.

    Returns the path to the enriched output file.
    """
    project_root = str(Path(__file__).resolve().parents[2])

    abs_input = os.path.join(project_root, input_path) \
        if not os.path.isabs(input_path) else input_path
    abs_output = os.path.join(project_root, output_path) \
        if not os.path.isabs(output_path) else output_path
    abs_checkpoint_dir = os.path.join(project_root, checkpoint_dir) \
        if not os.path.isabs(checkpoint_dir) else checkpoint_dir
    checkpoint_file = os.path.join(abs_checkpoint_dir, "structural_prior.done")

    # ── Checkpoint check ────────────────────────────────────────────────────
    if os.path.exists(checkpoint_file):
        log.info("Checkpoint found: skipping structural prior. Output: %s", abs_output)
        return abs_output

    # ── Create dirs ─────────────────────────────────────────────────────────
    os.makedirs(abs_checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    if not os.path.exists(abs_input):
        raise FileNotFoundError(
            f"Input file not found: {abs_input}\n"
            "Run dedup.py first to generate all_samples_deduped.jsonl"
        )

    # ── Load input ───────────────────────────────────────────────────────────
    log.info("Loading samples from %s", abs_input)
    records: list[dict] = []
    with open(abs_input, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    log.warning("Skipping malformed line: %s", exc)

    total = len(records)
    log.info("Loaded %d samples", total)

    # ── Annotate and write ───────────────────────────────────────────────────
    tier_counts = {1: 0, 2: 0, 3: 0}

    with open(abs_output, "w") as out_fh:
        for idx, record in enumerate(records):
            if idx > 0 and idx % 5_000 == 0:
                log.info("  Progress: %d / %d annotated", idx, total)

            prior = _compute_prior(record)
            enriched = {**record, **prior}
            out_fh.write(json.dumps(enriched, default=str) + "\n")

            tier_counts[prior["tier_prior"]] += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("── Structural prior complete ────────────────────────────────────")
    log.info("  Total annotated: %d", total)
    for t in (1, 2, 3):
        cnt = tier_counts[t]
        pct = 100.0 * cnt / total if total else 0.0
        log.info("  Tier %d: %4d samples  (%.1f%%)", t, cnt, pct)
    log.info("Output: %s", abs_output)

    # ── Checkpoint ───────────────────────────────────────────────────────────
    checkpoint_payload = {
        "input_path": abs_input,
        "output_path": abs_output,
        "total_annotated": total,
        "tier_counts": tier_counts,
    }
    with open(checkpoint_file, "w") as fh:
        fh.write(json.dumps(checkpoint_payload, indent=2))
    log.info("Checkpoint written: %s", checkpoint_file)

    return abs_output


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from collections import Counter

    print("=" * 60)
    print("DocRouteBench Phase 2, Structural Prior Smoke Test")
    print("=" * 60)

    force_rerun = "--force" in sys.argv

    project_root = str(Path(__file__).resolve().parents[2])
    checkpoint_file = os.path.join(
        project_root, "data/phase2_checkpoints/structural_prior.done"
    )
    dedup_checkpoint = os.path.join(
        project_root, "data/phase2_checkpoints/dedup.done"
    )

    if force_rerun and os.path.exists(checkpoint_file):
        print(f"[--force] Removing checkpoint: {checkpoint_file}")
        os.remove(checkpoint_file)

    # If dedup output doesn't exist yet, run dedup first
    dedup_output = os.path.join(project_root, "data/processed/all_samples_deduped.jsonl")
    if not os.path.exists(dedup_output):
        print("Deduped file not found: running deduplication first...")
        # Add src to path so we can import dedup
        src_path = str(Path(__file__).resolve().parents[1])
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from ingestion.dedup import run_deduplication
        run_deduplication(force_rerun=False)  # won't rerun if checkpoint exists

    output_path = run_structural_prior(
        input_path="data/processed/all_samples_deduped.jsonl",
        output_path="data/processed/samples_with_prior.jsonl",
        checkpoint_dir="data/phase2_checkpoints",
    )

    abs_output = os.path.join(project_root, output_path) \
        if not os.path.isabs(output_path) else output_path

    if not os.path.exists(abs_output):
        print(f"ERROR: output file not found at {abs_output}")
        sys.exit(1)

    records: list[dict] = []
    with open(abs_output) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"\nOutput file: {abs_output}")
    print(f"Total samples annotated: {len(records)}")

    # ── Tier distribution ──────────────────────────────────────────────────
    tier_counter = Counter(r["tier_prior"] for r in records)
    print("\nTier distribution:")
    for tier in (1, 2, 3):
        cnt = tier_counter[tier]
        pct = 100.0 * cnt / len(records) if records else 0.0
        print(f"  Tier {tier}: {cnt:>4d} samples  ({pct:5.1f}%)")

    # ── Per-dataset breakdown ─────────────────────────────────────────────
    print("\nTier distribution per dataset:")
    ds_tiers: dict[str, Counter] = {}
    for r in records:
        ds = r.get("source_dataset", "?")
        if ds not in ds_tiers:
            ds_tiers[ds] = Counter()
        ds_tiers[ds][r["tier_prior"]] += 1

    header = f"  {'Dataset':<30s}  T1    T2    T3    composite_avg"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for ds in sorted(ds_tiers):
        c = ds_tiers[ds]
        composites = [r["composite_est"] for r in records if r.get("source_dataset") == ds]
        avg_c = sum(composites) / len(composites) if composites else 0.0
        print(f"  {ds:<30s}  {c[1]:>3d}   {c[2]:>3d}   {c[3]:>3d}   {avg_c:.3f}")

    # ── Sample of soft probabilities ──────────────────────────────────────
    print("\nFirst 5 samples, prior fields:")
    for r in records[:5]:
        print(
            f"  {r['sample_id']:<35s}  "
            f"VDS={r['vds_est']}  RDS={r['rds_est']}  SES={r['ses_est']}  "
            f"comp={r['composite_est']:.2f}  "
            f"tier={r['tier_prior']}  "
            f"soft={[round(p, 3) for p in r['tier_prior_soft']]}"
        )

    print("\nSmoke test PASSED.")
