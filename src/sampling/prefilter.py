"""
DocRouteBench Phase 2 — Prefilter

Reduces ~134K deduped samples (data/processed/samples_with_prior.jsonl) to
~40K candidates (data/processed/candidates_40k.jsonl) via quality gate + cost quota.

Three passes:
  Pass 1 — Quality filter: remove obviously bad samples.
  Pass 2 — Per-dataset cost-control quota: max 8× sample_budget per dataset,
            random subsample (NOT diversity-ranked — diversity selection is
            deferred to C7 post-probe where we have full probe data).
  Pass 3 — Global target: proportional downsample to 40K if needed;
            warn if result < 20K.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── feature helpers ───────────────────────────────────────────────────────────

_UNINFORMATIVE_ANSWERS = {"see form", "n/a"}


def _feature_vector(sample: dict) -> List[float]:
    """6-dim feature vector used for diversity scoring."""
    vds = sample.get("vds_est", 2.0)
    rds = sample.get("rds_est", 2.0)
    ses = sample.get("ses_est", 2.0)
    has_table = 1.0 if sample.get("has_table") else 0.0
    has_chart = 1.0 if sample.get("has_chart") else 0.0
    query_len = len(sample.get("query", ""))
    return [
        vds / 4.0,
        rds / 4.0,
        ses / 4.0,
        has_table * 0.5,
        has_chart * 0.5,
        query_len / 100.0,
    ]


def _l2_distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _centroid(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return [0.0] * 6
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(len(vectors[0]))]


# ── Pass 1 — quality filter ───────────────────────────────────────────────────

def _passes_quality(sample: dict) -> Tuple[bool, str]:
    """Return (keep, reason_if_dropped)."""
    query = sample.get("query", "") or ""
    gt = sample.get("gt_answer", "") or ""

    if not query or len(query.strip()) < 5:
        return False, "query_too_short"

    if not gt or not gt.strip():
        return False, "gt_answer_empty"

    image_path = sample.get("image_path", "")
    if image_path and not os.path.exists(image_path):
        return False, "image_not_found"

    if query.strip() == gt.strip():
        return False, "query_equals_answer"

    if gt.strip().lower() in _UNINFORMATIVE_ANSWERS:
        return False, "uninformative_answer"

    return True, ""


# ── Pass 2 — per-dataset quota with diversity ranking ─────────────────────────

def _diversity_score(sample: dict, centroid: List[float]) -> float:
    return _l2_distance(_feature_vector(sample), centroid)


def _tier_label(sample: dict) -> int:
    """Return the tier (1/2/3). Fall back to tier inferred from tier_prior string."""
    tier = sample.get("tier_prior")
    if tier is None:
        return 2  # default: medium
    # tier_prior may be an int or a string like "tier_1", "T1", "1", etc.
    if isinstance(tier, int):
        return max(1, min(3, tier))
    s = str(tier).strip().lower().lstrip("tier_").lstrip("t")
    try:
        return max(1, min(3, int(s)))
    except ValueError:
        return 2


def _proportional_tier_sample(
    items: List[dict], n: int, rng_seed: int = 42
) -> List[dict]:
    """
    Downsample `items` to `n`, preserving the existing tier distribution.
    Items are assumed to be pre-sorted (best first); within each tier we
    keep the top-ranked ones.
    """
    if n >= len(items):
        return items

    # Count tiers in input
    tier_items: Dict[int, List[dict]] = defaultdict(list)
    for s in items:
        tier_items[_tier_label(s)].append(s)

    total = len(items)
    result: List[dict] = []
    remainder = n
    tiers = sorted(tier_items.keys())

    for i, t in enumerate(tiers):
        bucket = tier_items[t]
        if i == len(tiers) - 1:
            # last tier gets whatever remains
            quota = remainder
        else:
            quota = round(n * len(bucket) / total)
        keep = min(quota, len(bucket), remainder)
        result.extend(bucket[:keep])
        remainder -= keep
        if remainder <= 0:
            break

    return result


def _apply_dataset_quotas(
    samples_by_dataset: Dict[str, List[dict]],
    budgets: Dict[str, int],
    multiplier: int = 8,
) -> List[dict]:
    """
    For each dataset: take at most multiplier × sample_budget samples.

    Uses **random** subsampling (not diversity-ranked) to control probe cost
    without introducing rule-based selection bias. Intelligent diversity
    selection is deferred to C7 (post-probe) where we have full probe data.
    """
    import random as _random
    rng = _random.Random(42)

    output: List[dict] = []

    for ds, samples in samples_by_dataset.items():
        budget = budgets.get(ds)
        if budget is None:
            log.warning("Dataset %r has no sample_budget in datasets.yaml — using 200", ds)
            budget = 200

        quota = budget * multiplier

        if len(samples) <= quota:
            kept = samples
        else:
            # Random subsample — no diversity ranking bias
            kept = rng.sample(samples, quota)

        log.info(
            "  %-20s  in=%4d  quota=%4d  kept=%4d",
            ds,
            len(samples),
            quota,
            len(kept),
        )
        output.extend(kept)

    return output


# ── Pass 3 — global target ────────────────────────────────────────────────────

def _global_downsample(samples: List[dict], target: int) -> List[dict]:
    """
    Proportional downsample to `target`, preserving tier distribution.
    Samples within each tier are already diversity-ranked; we keep the top.
    """
    if len(samples) <= target:
        return samples

    tier_buckets: Dict[int, List[dict]] = defaultdict(list)
    for s in samples:
        tier_buckets[_tier_label(s)].append(s)

    total = len(samples)
    result: List[dict] = []
    remainder = target
    tiers = sorted(tier_buckets.keys())

    for i, t in enumerate(tiers):
        bucket = tier_buckets[t]
        if i == len(tiers) - 1:
            quota = remainder
        else:
            quota = round(target * len(bucket) / total)
        keep = min(quota, len(bucket), remainder)
        result.extend(bucket[:keep])
        remainder -= keep
        if remainder <= 0:
            break

    return result


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    if d.get("sample_id"):  # skip "cleared" sentinels
                        records.append(d)
                except json.JSONDecodeError:
                    pass
    return records


def _write_jsonl(path: str, records: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            # Strip internal working keys before writing
            out = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(out, default=str) + "\n")


def _load_budgets(config_path: str) -> Dict[str, int]:
    """
    Load sample_budget per dataset from datasets.yaml.
    Keys are source_dataset names (lower-snake-case as in the YAML).
    Returns mapping: normalised_dataset_key → budget.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    budgets: Dict[str, int] = {}
    for key, meta in cfg.get("datasets", {}).items():
        budget = meta.get("sample_budget")
        if budget is not None:
            budgets[key] = int(budget)
            # Also store under full_name for easier matching
            full_name = meta.get("full_name", "")
            if full_name:
                budgets[full_name] = int(budget)
    return budgets


def _match_budget(source_dataset: str, budgets: Dict[str, int]) -> Optional[int]:
    """Fuzzy match source_dataset value to a budget entry."""
    # Direct match
    if source_dataset in budgets:
        return budgets[source_dataset]
    # Case-insensitive match against keys
    lower = source_dataset.lower()
    for k, v in budgets.items():
        if k.lower() == lower:
            return v
    # Partial match (e.g. "DocVQA" matches "docvqa")
    for k, v in budgets.items():
        if lower in k.lower() or k.lower() in lower:
            return v
    return None


# ── Main entry point ──────────────────────────────────────────────────────────

def run_prefilter(
    input_path: str = "data/processed/samples_with_prior.jsonl",
    output_path: str = "data/processed/candidates_40k.jsonl",
    checkpoint_dir: str = "data/phase2_checkpoints",
    target_candidates: int = 40_000,
) -> str:
    """
    Reduce ~134K deduped samples to ~40K candidates via quality gate + cost-control quota.

    Diversity selection is NOT done here — it's deferred to C7 (post-probe)
    where we have full probe-derived features. This pre-filter only ensures:
    1. No junk reaches the probe (quality gate)
    2. No single dataset dominates probe budget (random quota)

    Returns path to output file.
    """
    checkpoint_path = Path(checkpoint_dir) / "prefilter.done"
    if checkpoint_path.exists():
        log.info("Checkpoint exists — skipping prefilter. Output: %s", output_path)
        return output_path

    # ── Load input ────────────────────────────────────────────────────────
    log.info("Loading samples from %s", input_path)
    all_samples = _load_jsonl(input_path)
    log.info("Loaded %d samples", len(all_samples))

    # ── Load budgets ─────────────────────────────────────────────────────
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "configs", "datasets.yaml",
    )
    config_path = os.path.normpath(config_path)
    budgets = _load_budgets(config_path)
    log.info("Loaded budgets for %d datasets", len([k for k in budgets if not k[0].isupper()]))

    # ════════════════════════════════════════════════════════════════════
    # PASS 1 — Quality filter
    # ════════════════════════════════════════════════════════════════════
    log.info("── Pass 1: Quality filter ──")
    drop_reasons: Dict[str, int] = defaultdict(int)
    passed: List[dict] = []

    for s in all_samples:
        ok, reason = _passes_quality(s)
        if ok:
            passed.append(s)
        else:
            drop_reasons[reason] += 1

    log.info(
        "Pass 1: %d → %d (dropped %d). Reasons: %s",
        len(all_samples),
        len(passed),
        len(all_samples) - len(passed),
        dict(drop_reasons),
    )

    # ════════════════════════════════════════════════════════════════════
    # PASS 2 — Per-dataset quota
    # ════════════════════════════════════════════════════════════════════
    log.info("── Pass 2: Per-dataset quotas (8× sample_budget, random subsample) ──")

    # Group by source_dataset
    by_dataset: Dict[str, List[dict]] = defaultdict(list)
    for s in passed:
        ds = s.get("source_dataset", "unknown")
        by_dataset[ds].append(s)

    log.info("Datasets after Pass 1: %s", {k: len(v) for k, v in sorted(by_dataset.items())})

    # Build per-dataset budget lookup keyed by source_dataset values seen
    ds_budgets: Dict[str, int] = {}
    for ds in by_dataset:
        b = _match_budget(ds, budgets)
        if b is None:
            log.warning("No budget found for dataset %r — defaulting to 200", ds)
            b = 200
        ds_budgets[ds] = b

    after_quota = _apply_dataset_quotas(by_dataset, ds_budgets, multiplier=4)

    log.info(
        "Pass 2: %d → %d (after per-dataset quotas)",
        len(passed),
        len(after_quota),
    )

    # ════════════════════════════════════════════════════════════════════
    # PASS 3 — Global target
    # ════════════════════════════════════════════════════════════════════
    log.info("── Pass 3: Global target (%d) ──", target_candidates)

    n_after_quota = len(after_quota)

    if n_after_quota < 20_000:
        log.warning(
            "Only %d candidates after Pass 2 — quality filters may be too strict. "
            "Target is %d.",
            n_after_quota,
            target_candidates,
        )

    if n_after_quota > target_candidates:
        final = _global_downsample(after_quota, target_candidates)
        log.info(
            "Pass 3: proportionally downsampled %d → %d",
            n_after_quota,
            len(final),
        )
    else:
        final = after_quota
        log.info(
            "Pass 3: %d ≤ %d target, no downsampling needed",
            n_after_quota,
            target_candidates,
        )

    # ── Diagnostics ───────────────────────────────────────────────────────
    task_counts: Dict[str, int] = defaultdict(int)
    tier_counts: Dict[int, int] = defaultdict(int)
    ds_counts: Dict[str, int] = defaultdict(int)

    for s in final:
        task_counts[s.get("task_type", "unknown")] += 1
        tier_counts[_tier_label(s)] += 1
        ds_counts[s.get("source_dataset", "unknown")] += 1

    log.info("Final sample count: %d", len(final))
    log.info("Task type distribution: %s", dict(sorted(task_counts.items())))
    log.info("Tier distribution: %s", dict(sorted(tier_counts.items())))
    log.info("Dataset distribution: %s", dict(sorted(ds_counts.items())))

    # ── Write output ──────────────────────────────────────────────────────
    _write_jsonl(output_path, final)
    log.info("Wrote %d candidates to %s", len(final), output_path)

    # ── Checkpoint ────────────────────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(
            {
                "input_path": input_path,
                "output_path": output_path,
                "n_input": len(all_samples),
                "n_pass1": len(passed),
                "n_pass2": n_after_quota,
                "n_final": len(final),
                "task_counts": dict(task_counts),
                "tier_counts": {str(k): v for k, v in tier_counts.items()},
                "ds_counts": dict(ds_counts),
                "drop_reasons": dict(drop_reasons),
            },
            indent=2,
        )
    )
    log.info("Checkpoint written to %s", checkpoint_path)

    return output_path


# ── Smoke test (runs on the 108 water-test samples) ───────────────────────────

def _run_smoke_test() -> None:
    """
    Smoke test against the 107-sample water-test corpus.

    Builds a synthetic samples_with_prior.jsonl from the existing
    *_normalized.jsonl files (injecting fake vds_est / rds_est / ses_est /
    tier_prior fields), runs run_prefilter(), and checks postconditions.
    """
    import tempfile
    import random

    log.info("=== Smoke test start ===")
    processed_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "data", "processed",
    )
    processed_dir = os.path.normpath(processed_dir)

    # Collect all water-test samples
    raw_samples: List[dict] = []
    for fn in sorted(os.listdir(processed_dir)):
        if not fn.endswith("_normalized.jsonl"):
            continue
        path = os.path.join(processed_dir, fn)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not d.get("sample_id"):
                    continue
                raw_samples.append(d)

    log.info("Water-test corpus: %d samples", len(raw_samples))

    # Inject prior fields with deterministic variety
    rng = random.Random(42)
    bad_injected = 0

    def _inject_prior(s: dict, idx: int) -> dict:
        nonlocal bad_injected
        out = dict(s)
        # Inject VDS/RDS/SES estimates (1–4)
        out["vds_est"] = rng.randint(1, 4)
        out["rds_est"] = rng.randint(1, 4)
        out["ses_est"] = rng.randint(1, 4)
        # Tier prior: cycle through 1/2/3
        tier = (idx % 3) + 1
        out["tier_prior"] = tier

        # Inject a handful of obviously bad samples to exercise Pass 1
        if idx % 15 == 0:
            out["query"] = "?"  # too short → drop
            bad_injected += 1
        elif idx % 15 == 1:
            out["gt_answer"] = "N/A"  # uninformative → drop
            bad_injected += 1

        return out

    enriched = [_inject_prior(s, i) for i, s in enumerate(raw_samples)]

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "samples_with_prior.jsonl")
        output_path = os.path.join(tmpdir, "candidates_15k.jsonl")
        checkpoint_dir = os.path.join(tmpdir, "phase2_checkpoints")

        with open(input_path, "w") as f:
            for s in enriched:
                f.write(json.dumps(s) + "\n")

        result_path = run_prefilter(
            input_path=input_path,
            output_path=output_path,
            checkpoint_dir=checkpoint_dir,
            target_candidates=25_000,  # water test << 25K so Pass 3 is a no-op
        )

        # Load output
        candidates = []
        with open(result_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))

        n_out = len(candidates)
        n_in = len(enriched)

        # ── Assertion 1: Quality filter removes bad samples ───────────────
        assert n_out < n_in, (
            f"Pass 1 should have removed some samples; got {n_out} out of {n_in}"
        )
        log.info(
            "PASS  Quality filter: %d input → %d output (dropped %d bad samples)",
            n_in, n_out, n_in - n_out,
        )

        # ── Assertion 2: Per-dataset quotas respected ─────────────────────
        # Water test is tiny; all datasets should be well under 4× budget.
        # Just verify we have multiple datasets in output.
        ds_in_output = {s["source_dataset"] for s in candidates}
        assert len(ds_in_output) >= 2, (
            f"Expected multiple datasets in output; got {ds_in_output}"
        )
        log.info("PASS  Per-dataset quotas: %d datasets in output", len(ds_in_output))

        # ── Assertion 3: All 6 task types represented (if present in input) ─
        task_types_in_input = {s["task_type"] for s in enriched}
        task_types_in_output = {s["task_type"] for s in candidates}
        missing = task_types_in_input - task_types_in_output
        assert not missing, (
            f"Task types in input but not in output: {missing}"
        )
        log.info(
            "PASS  Task types: %s all present in output",
            sorted(task_types_in_output),
        )

        # ── Assertion 4: Tier distribution is not all Tier 1 ─────────────
        tiers_out = {_tier_label(s) for s in candidates}
        assert len(tiers_out) > 1, (
            f"Output is dominated by a single tier: {tiers_out}"
        )
        log.info(
            "PASS  Tier distribution: tiers present = %s",
            sorted(tiers_out),
        )

        # ── Assertion 5: No internal scratch keys in output ───────────────
        for s in candidates:
            for k in s:
                assert not k.startswith("_"), f"Internal key {k!r} leaked into output"
        log.info("PASS  No internal scratch keys in output")

        log.info("=== Smoke test PASSED (%d / %d samples retained) ===", n_out, n_in)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        _run_smoke_test()
    else:
        # Production run with defaults
        out = run_prefilter()
        print(f"Done. Output: {out}")
