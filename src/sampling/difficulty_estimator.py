#!/usr/bin/env python3
"""
DocRouteBench Phase 2 — Difficulty Estimator

Combines structural prior with enriched probe results (visual elements,
VDS/RDS/SES assessments, correctness) into probe-driven difficulty estimates.

Input:
    data/processed/samples_with_prior.jsonl   — samples with tier_prior_soft field
    data/processed/probe_results.jsonl        — enriched probe records (correctness + visual + VDS/RDS/SES)

Output:
    data/processed/difficulty_scores.jsonl    — one line per sample, posterior fields

Usage:
    python -m src.sampling.difficulty_estimator              # full run
    python -m src.sampling.difficulty_estimator --water-test # smoke test (107 samples)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ── Project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.schema import load_jsonl, append_jsonl

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("difficulty_estimator")

# ── Accuracy tables (pilot data) ──────────────────────────────────────────────
# P(model_correct | Tier) by task_type
# Tier 1 = easy, Tier 2 = medium, Tier 3 = hard
MODEL_ACCURACY_BY_TIER: dict[str, dict[str, dict[int, float]]] = {
    "gpt52": {
        "T1": {1: 0.95, 2: 0.82, 3: 0.45},
        "T2": {1: 0.92, 2: 0.78, 3: 0.40},
        "T3": {1: 0.88, 2: 0.70, 3: 0.35},
        "T4": {1: 0.90, 2: 0.75, 3: 0.38},
        "T5": {1: 0.72, 2: 0.55, 3: 0.25},
        "T6": {1: 0.65, 2: 0.45, 3: 0.20},
    },
    "gemini_flash": {
        "T1": {1: 0.92, 2: 0.78, 3: 0.40},
        "T2": {1: 0.89, 2: 0.72, 3: 0.35},
        "T3": {1: 0.85, 2: 0.65, 3: 0.30},
        "T4": {1: 0.87, 2: 0.70, 3: 0.32},
        "T5": {1: 0.68, 2: 0.50, 3: 0.22},
        "T6": {1: 0.60, 2: 0.42, 3: 0.18},
    },
}

# Probe model IDs as used in probe_results.jsonl
# Map from api_probe model_id → accuracy table key
# api_probe uses "gpt52" and "gemini25flash" as model_id values
_PROBE_MODEL_MAP: dict[str, str] = {
    "gpt52":        "gpt52",
    "gemini25flash": "gemini_flash",
    # also accept the accuracy table key directly
    "gemini_flash": "gemini_flash",
}

TIERS = [1, 2, 3]
PROBE_MODELS_ORDERED = ["gpt52", "gemini_flash"]  # canonical output order


def compute_likelihood(model_id: str, task_type: str, is_correct: bool, tier: int) -> float:
    """
    P(probe_result | Tier) for a single model.

    Args:
        model_id:   Key in MODEL_ACCURACY_BY_TIER ("gpt52" or "gemini_flash").
        task_type:  One of T1-T6.
        is_correct: Whether the model answered correctly.
        tier:       1, 2, or 3.

    Returns:
        Likelihood value in (0, 1).
    """
    acc = MODEL_ACCURACY_BY_TIER[model_id][task_type][tier]
    return acc if is_correct else (1.0 - acc)


def _bayesian_update(
    prior: list[float],
    task_type: str,
    probe_results: dict[str, Optional[bool]],
) -> list[float]:
    """
    Update a 3-tier prior using available probe results.

    probe_results maps model_id (accuracy table key) to is_correct (bool) or
    None if that probe is missing.  Only non-None entries are used in the update.

    Returns:
        Normalised posterior [P(T1|data), P(T2|data), P(T3|data)].
    """
    posterior = list(prior)  # copy

    for model_id, is_correct in probe_results.items():
        if is_correct is None:
            continue  # missing probe — use prior only
        for i, tier in enumerate(TIERS):
            posterior[i] *= compute_likelihood(model_id, task_type, is_correct, tier)

    total = sum(posterior)
    if total <= 0.0:
        # Degenerate case: fall back to uniform
        return [1.0 / len(TIERS)] * len(TIERS)

    return [p / total for p in posterior]


def _tier_from_soft(soft: list[float]) -> int:
    """Argmax of soft probabilities → 1-indexed tier."""
    return int(max(range(len(soft)), key=lambda i: soft[i])) + 1


def _difficulty_score(soft: list[float]) -> float:
    """
    Continuous difficulty in [1.0, 3.0].
    Weighted sum: score = 1*P(T1) + 2*P(T2) + 3*P(T3).
    """
    return sum((i + 1) * p for i, p in enumerate(soft))


# ── Index probe results by sample_id ─────────────────────────────────────────

def _load_probe_index(probe_path: str) -> dict[str, dict[str, dict]]:
    """
    Load probe_results.jsonl and index by sample_id → {model_key: {full record}}.

    Enriched records include: is_correct, visual_elements, doc_type_detected,
    vds_probe, rds_probe, ses_probe, vds_label, rds_label, ses_label,
    vds_evidence, rds_evidence, ses_evidence, answer_word_count, predicted_answer, etc.
    """
    index: dict[str, dict[str, dict]] = defaultdict(dict)
    try:
        records = load_jsonl(probe_path)
    except FileNotFoundError:
        log.warning("Probe results file not found: %s — treating all as missing.", probe_path)
        return {}

    for r in records:
        sid = r.get("sample_id", "")
        raw_mid = r.get("model_id", "")

        if not sid or not raw_mid:
            continue

        if r.get("error"):
            continue

        key = _PROBE_MODEL_MAP.get(raw_mid)
        if key is None:
            log.debug("Unknown model_id in probe results: %r — skipping.", raw_mid)
            continue

        index[sid][key] = r  # store full record

    log.info("Probe index: %d unique samples with probe data.", len(index))
    return dict(index)


# ── Default prior fallback ────────────────────────────────────────────────────

_UNIFORM_PRIOR = [1 / 3, 1 / 3, 1 / 3]
_TIER_TO_SOFT: dict[int, list[float]] = {
    1: [0.70, 0.20, 0.10],
    2: [0.15, 0.65, 0.20],
    3: [0.10, 0.20, 0.70],
}


def _resolve_prior(sample: dict) -> tuple[int, list[float]]:
    """
    Extract tier_prior (int) and tier_prior_soft ([float×3]) from a sample dict.
    Falls back gracefully if fields are missing.
    """
    soft = sample.get("tier_prior_soft")
    tier = sample.get("tier_prior")

    if soft is None and tier is not None:
        soft = _TIER_TO_SOFT.get(int(tier), _UNIFORM_PRIOR)
    elif soft is None:
        soft = _UNIFORM_PRIOR

    if tier is None:
        tier = _tier_from_soft(soft)

    # Normalise in case of floating-point drift
    total = sum(soft)
    if total > 0:
        soft = [p / total for p in soft]
    else:
        soft = _UNIFORM_PRIOR

    return int(tier), soft


def _aggregate_probe_features(sample_probes: dict[str, dict]) -> dict:
    """
    Merge visual + complexity signals from both probe models into sample-level features.
    """
    if not sample_probes:
        return {}

    all_visual = set()
    vds_scores, rds_scores, ses_scores = [], [], []
    doc_types, answers = [], []

    for model_data in sample_probes.values():
        all_visual.update(model_data.get("visual_elements", []))
        vds_scores.append(model_data.get("vds_probe", 2))
        rds_scores.append(model_data.get("rds_probe", 2))
        ses_scores.append(model_data.get("ses_probe", 2))
        doc_types.append(model_data.get("doc_type_detected", "other"))
        answers.append(model_data.get("predicted_answer", ""))

    return {
        "has_table_detected": "table" in all_visual,
        "has_chart_detected": bool(all_visual & {"bar_chart", "line_chart", "pie_chart", "scatter_plot", "chart"}),
        "has_figure_detected": bool(all_visual & {"figure", "photograph", "diagram", "map", "flowchart", "illustration"}),
        "has_handwriting_detected": bool(all_visual & {"handwriting", "signature"}),
        "visual_element_count": len(all_visual),
        "visual_elements_union": sorted(all_visual),
        "vds_probe_avg": round(sum(vds_scores) / len(vds_scores), 2),
        "rds_probe_avg": round(sum(rds_scores) / len(rds_scores), 2),
        "ses_probe_avg": round(sum(ses_scores) / len(ses_scores), 2),
        "doc_type_detected": max(set(doc_types), key=doc_types.count) if doc_types else "other",
        "probe_answers_agree": len(set(a.strip().lower() for a in answers if a)) <= 1,
        "avg_answer_words": round(sum(len(a.split()) for a in answers) / max(len(answers), 1), 1),
    }


# ── Probe complexity likelihood ──────────────────────────────────────────────

# Expected composite score per tier (centre of Gaussian likelihood)
_TIER_COMPOSITE_MEAN = {1: 1.5, 2: 2.5, 3: 3.5}
_TIER_COMPOSITE_SIGMA = 0.6  # std dev — controls how sharply complexity maps to tier


def _complexity_likelihood(probe_composite: float, tier: int) -> float:
    """
    P(probe_composite | tier) modelled as Gaussian.

    A probe composite of 3.5 is very likely under Tier 3 and unlikely under
    Tier 1.  This provides an independent signal from binary correctness —
    if both models get a complex question right, the complexity likelihood
    still pushes toward higher tiers, preventing naive Tier 1 assignment.
    """
    import math
    mu = _TIER_COMPOSITE_MEAN[tier]
    sigma = _TIER_COMPOSITE_SIGMA
    return math.exp(-0.5 * ((probe_composite - mu) / sigma) ** 2)


# ── Core estimation logic ─────────────────────────────────────────────────────

def estimate_difficulty(
    sample: dict,
    probe_index: dict[str, dict[str, dict]],
) -> dict:
    """
    Compute difficulty fields for a single sample using Bayesian posterior
    combining three signals:
      1. Structural prior  (rule-based VDS/RDS/SES → soft tier probabilities)
      2. Binary correctness (did mid-tier probe models answer correctly?)
      3. Probe complexity   (VDS/RDS/SES rated by the model that saw the image)

    posterior ∝ prior × P(correct₁|tier) × P(correct₂|tier) × P(probe_composite|tier)
    """
    sid = sample["sample_id"]
    task_type = sample.get("task_type", "T1")

    tier_prior, prior_soft = _resolve_prior(sample)

    # Collect enriched probe results for this sample
    sample_probes = probe_index.get(sid, {})

    # Aggregate probe features
    agg = _aggregate_probe_features(sample_probes)

    # VDS/RDS/SES: prefer probe-derived, fall back to structural prior
    vds_avg = agg.get("vds_probe_avg", sample.get("vds_est", 2))
    rds_avg = agg.get("rds_probe_avg", sample.get("rds_est", 2))
    ses_avg = agg.get("ses_probe_avg", sample.get("ses_est", 2))

    probe_composite = round(0.30 * vds_avg + 0.45 * rds_avg + 0.25 * ses_avg, 4)

    # ── Bayesian posterior with three likelihood terms ────────────────────
    posterior = list(prior_soft)  # start with prior

    # Term 1 + 2: Binary correctness likelihood (per model)
    for model_key in PROBE_MODELS_ORDERED:
        model_data = sample_probes.get(model_key, {})
        is_correct = model_data.get("is_correct")
        if is_correct is None:
            continue  # missing probe — skip this likelihood term
        for i, tier in enumerate(TIERS):
            posterior[i] *= compute_likelihood(model_key, task_type, is_correct, tier)

    # Term 3: Probe complexity likelihood (Gaussian around tier mean)
    if agg:  # only if we have probe-derived complexity
        for i, tier in enumerate(TIERS):
            posterior[i] *= _complexity_likelihood(probe_composite, tier)

    # Normalize
    total = sum(posterior)
    if total <= 0.0:
        posterior_soft = [1.0 / len(TIERS)] * len(TIERS)
    else:
        posterior_soft = [p / total for p in posterior]

    tier_final = _tier_from_soft(posterior_soft)
    confidence = round(max(posterior_soft), 6)
    diff_score = round(_difficulty_score(posterior_soft), 6)

    # Probe agreement label
    probe_gpt52 = sample_probes.get("gpt52", {}).get("is_correct")
    probe_flash = sample_probes.get("gemini_flash", {}).get("is_correct")

    if probe_gpt52 is None and probe_flash is None:
        probe_agreement = "missing"
    elif probe_gpt52 == probe_flash:
        probe_agreement = "both_correct" if probe_gpt52 else "both_wrong"
    else:
        probe_agreement = "disagree"

    # Build output record
    out = dict(sample)
    out.update({
        "tier_prior":          tier_prior,
        "tier_prior_soft":     [round(p, 6) for p in prior_soft],
        "probe_gpt52_correct": probe_gpt52,
        "probe_flash_correct": probe_flash,
        "probe_agreement":     probe_agreement,
        "probe_composite":     probe_composite,
        "tier_posterior_soft":  [round(p, 6) for p in posterior_soft],
        "tier_final":          tier_final,
        "difficulty_score":    diff_score,
        "confidence":          confidence,
    })

    # Add aggregated probe features (for downstream sampler)
    out.update(agg)

    return out


# ── Public API ────────────────────────────────────────────────────────────────

def run_difficulty_estimator(
    samples_path: str = "data/processed/samples_with_prior.jsonl",
    probe_path: str = "data/processed/probe_results.jsonl",
    output_path: str = "data/processed/difficulty_scores.jsonl",
    checkpoint_dir: str = "data/phase2_checkpoints",
    overwrite: bool = False,
) -> str:
    """
    Run the difficulty estimator over all samples and write difficulty_scores.jsonl.

    Args:
        samples_path:   JSONL with structural prior fields (tier_prior_soft).
        probe_path:     JSONL with probe (sample_id, model_id, is_correct) rows.
        output_path:    Destination JSONL for difficulty scores.
        checkpoint_dir: Directory for the .done checkpoint file.
        overwrite:      If False, skip if output already exists.

    Returns:
        Absolute path to the output file.
    """
    # Resolve paths relative to project root
    def _abs(p: str) -> str:
        path = Path(p)
        return str(path if path.is_absolute() else _PROJECT_ROOT / path)

    samples_path  = _abs(samples_path)
    probe_path    = _abs(probe_path)
    output_path   = _abs(output_path)
    checkpoint_dir = _abs(checkpoint_dir)

    checkpoint_file = Path(checkpoint_dir) / "difficulty_estimator.done"

    # ── Check checkpoint ──────────────────────────────────────────────────────
    if checkpoint_file.exists() and not overwrite:
        log.info("Checkpoint found: %s — skipping (use --overwrite to re-run).", checkpoint_file)
        return output_path

    if Path(output_path).exists() and not overwrite:
        log.warning(
            "Output file %s already exists. Use --overwrite to regenerate.", output_path
        )
        return output_path

    # ── Load inputs ───────────────────────────────────────────────────────────
    log.info("Loading samples from %s …", samples_path)
    try:
        raw_samples = load_jsonl(samples_path)
    except FileNotFoundError:
        log.error("Samples file not found: %s", samples_path)
        raise

    # Filter out sentinel rows (e.g. {"cleared": true})
    samples = [s for s in raw_samples if s.get("sample_id")]
    log.info("Loaded %d samples (skipped %d non-sample rows).", len(samples), len(raw_samples) - len(samples))

    log.info("Loading probe results from %s …", probe_path)
    probe_index = _load_probe_index(probe_path)

    # ── Process ───────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Truncate output file before writing
    with open(output_path, "w") as _:
        pass

    n_no_probes = 0
    n_one_probe = 0
    n_both_probes = 0

    for sample in samples:
        sid = sample["sample_id"]
        probes = probe_index.get(sid, {})
        n_present = sum(1 for v in probes.values() if v is not None)

        if n_present == 0:
            n_no_probes += 1
        elif n_present == 1:
            n_one_probe += 1
        else:
            n_both_probes += 1

        record = estimate_difficulty(sample, probe_index)
        append_jsonl(output_path, record)

    # ── Stats ─────────────────────────────────────────────────────────────────
    log.info(
        "Difficulty estimation complete: %d samples total.", len(samples)
    )
    log.info(
        "  Both probes present: %d | One probe: %d | No probes: %d",
        n_both_probes, n_one_probe, n_no_probes,
    )

    # Write checkpoint
    checkpoint_file.write_text(
        json.dumps(
            {
                "n_samples":    len(samples),
                "n_both_probes": n_both_probes,
                "n_one_probe":  n_one_probe,
                "n_no_probes":  n_no_probes,
                "output_path":  output_path,
            },
            indent=2,
        )
        + "\n"
    )
    log.info("Checkpoint written: %s", checkpoint_file)
    return output_path


# ── Water-test smoke test ─────────────────────────────────────────────────────

def run_water_test() -> None:
    """
    Smoke test on the available normalized samples (water-test pool).
    Generates synthetic prior fields from task_type and runs the estimator
    on whatever samples are available in data/processed/*_normalized.jsonl.
    """
    import random

    log.info("=" * 60)
    log.info("WATER-TEST MODE — difficulty_estimator")
    log.info("=" * 60)

    # Collect all available samples from normalized files
    normalized_dir = _PROJECT_ROOT / "data" / "processed"
    samples: list[dict] = []
    for fpath in sorted(normalized_dir.glob("*_normalized.jsonl")):
        try:
            for row in load_jsonl(str(fpath)):
                if row.get("sample_id"):
                    samples.append(row)
        except Exception as exc:
            log.warning("Skipping %s: %s", fpath.name, exc)

    if not samples:
        log.error("No samples found for water-test. Run ingestion first.")
        return

    log.info("Water-test: %d samples available.", len(samples))

    # Inject synthetic tier_prior_soft based on task_type heuristic
    _task_to_tier = {"T1": 1, "T2": 2, "T3": 3, "T4": 2, "T5": 2, "T6": 3}
    for s in samples:
        tier = _task_to_tier.get(s.get("task_type", "T1"), 2)
        s.setdefault("tier_prior", tier)
        s.setdefault("tier_prior_soft", _TIER_TO_SOFT[tier])

    # Write synthetic samples_with_prior.jsonl to a temp location
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="drb_water_"))
    samples_path   = str(tmp_dir / "samples_with_prior.jsonl")
    probe_path     = str(tmp_dir / "probe_results_empty.jsonl")  # intentionally empty
    output_path    = str(tmp_dir / "difficulty_scores.jsonl")
    checkpoint_dir = str(tmp_dir / "checkpoints")

    with open(samples_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # Empty probe file — test the "both probes missing" path
    Path(probe_path).write_text("")

    # Also inject synthetic probe results for half the samples (test update path)
    synthetic_probes: list[dict] = []
    rng = random.Random(42)
    for s in samples[: len(samples) // 2]:
        for mid_key, mid_label in [("gpt52", "gpt52"), ("gemini_flash", "gemini25flash")]:
            task_type = s.get("task_type", "T1")
            tier = s.get("tier_prior", 2)
            acc = MODEL_ACCURACY_BY_TIER[mid_key][task_type][tier]
            is_correct = rng.random() < acc
            synthetic_probes.append(
                {
                    "sample_id":  s["sample_id"],
                    "model_id":   mid_label,
                    "is_correct": is_correct,
                }
            )

    probe_with_data = str(tmp_dir / "probe_results_with_data.jsonl")
    with open(probe_with_data, "w") as f:
        for p in synthetic_probes:
            f.write(json.dumps(p) + "\n")

    # Run with empty probes first
    log.info("--- Pass 1: empty probe file (all samples should use prior) ---")
    out1 = run_difficulty_estimator(
        samples_path=samples_path,
        probe_path=probe_path,
        output_path=output_path,
        checkpoint_dir=checkpoint_dir,
        overwrite=True,
    )
    results1 = [r for r in load_jsonl(out1) if r.get("sample_id")]
    assert len(results1) == len(samples), f"Expected {len(samples)}, got {len(results1)}"
    no_probe_results = [r for r in results1 if r["confidence"] == 0.0]
    assert len(no_probe_results) == len(samples), (
        f"All should have confidence=0.0 with empty probes; got {len(no_probe_results)}"
    )
    log.info("Pass 1 PASSED: %d samples, all confidence=0.0", len(results1))

    # Run with synthetic probes
    log.info("--- Pass 2: synthetic probe data for %d/%d samples ---", len(samples) // 2, len(samples))
    out2 = run_difficulty_estimator(
        samples_path=samples_path,
        probe_path=probe_with_data,
        output_path=output_path,
        checkpoint_dir=checkpoint_dir,
        overwrite=True,
    )
    results2 = [r for r in load_jsonl(out2) if r.get("sample_id")]
    assert len(results2) == len(samples)

    with_probe = [r for r in results2 if r["confidence"] > 0.0]
    without_probe = [r for r in results2 if r["confidence"] == 0.0]
    log.info(
        "Pass 2: %d with probe updates (confidence>0), %d prior-only",
        len(with_probe), len(without_probe),
    )

    # Check required fields
    required = {
        "sample_id", "tier_prior", "tier_prior_soft",
        "probe_gpt52_correct", "probe_flash_correct",
        "tier_posterior_soft", "tier_final", "difficulty_score", "confidence",
    }
    for r in results2[:5]:
        missing = required - r.keys()
        assert not missing, f"Missing fields in output: {missing}"

    # Check score range
    for r in results2:
        assert 1.0 <= r["difficulty_score"] <= 3.0, (
            f"difficulty_score out of range: {r['difficulty_score']}"
        )
        assert 1 <= r["tier_final"] <= 3, f"tier_final invalid: {r['tier_final']}"
        assert 0.0 <= r["confidence"] <= 1.0, f"confidence out of range: {r['confidence']}"

    # Check tier distribution
    tier_counts = {1: 0, 2: 0, 3: 0}
    for r in results2:
        tier_counts[r["tier_final"]] += 1
    log.info("Tier distribution: %s", tier_counts)

    log.info("=" * 60)
    log.info("WATER-TEST PASSED — difficulty_estimator")
    log.info("=" * 60)

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DocRouteBench Phase 2 — Difficulty Estimator",
    )
    parser.add_argument(
        "--samples",
        default="data/processed/samples_with_prior.jsonl",
        help="Path to samples JSONL with tier_prior_soft field",
    )
    parser.add_argument(
        "--probes",
        default="data/processed/probe_results.jsonl",
        help="Path to probe results JSONL",
    )
    parser.add_argument(
        "--output",
        default="data/processed/difficulty_scores.jsonl",
        help="Path to write difficulty scores JSONL",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="data/phase2_checkpoints",
        help="Directory for checkpoint file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output and checkpoint",
    )
    parser.add_argument(
        "--water-test",
        action="store_true",
        help="Run smoke test on available water-test samples",
    )
    args = parser.parse_args()

    if args.water_test:
        run_water_test()
        return

    run_difficulty_estimator(
        samples_path   = args.samples,
        probe_path     = args.probes,
        output_path    = args.output,
        checkpoint_dir = args.checkpoint_dir,
        overwrite      = args.overwrite,
    )


if __name__ == "__main__":
    main()
