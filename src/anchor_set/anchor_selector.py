#!/usr/bin/env python3
"""
DocRouteBench Phase 2, Anchor Set Selector
============================================
Selects 1,500 anchor samples and 750 validation samples from the 5,000
benchmark via submodular facility-location greedy optimization.

Algorithm: Submodular Facility Location (Nemhauser et al. 1978):
  Objective: F(A) = sum_s max_{a in A} S(s, a)
  Greedy:    iteratively add the sample that maximises marginal gain
  Guarantee: achieves (1 - 1/e) ≈ 63.2% of optimal F

Feature space (30-dim):
  6  task-type one-hot  (T1–T6)
  3  tier one-hot       (1–3)
  16 dataset one-hot    (16 datasets)
  3  complexity scalars (VDS, RDS, SES)
  2  structural flags   (has_table, has_chart)
  ─────────────────────
  30 total

Similarity kernel: RBF  S(a, b) = exp(-||a-b||² / (2σ²))
  σ = median pairwise distance (auto-calibrated from a random subsample).

Hard constraints (seeded before greedy):
  per_task_min:    T1≥100, T2≥170, T3≥100, T4≥220, T5≥150, T6≥110
  per_tier_min:    tier1≥200, tier2≥480, tier3≥200
  per_dataset_min: ≥15 per dataset

Validation set: stratified sample from remaining 2,750 (non-overlapping).
  Tier weights: {1: 0.20, 2: 0.55, 3: 0.25}

Outputs:
  data/anchor_set/anchor_ids.json
  data/validation_set/validation_ids.json
  data/anchor_set/selection_report.json

Checkpoint: data/phase2_checkpoints/anchor_selector.done
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.schema import load_jsonl, TASK_TYPES

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("anchor_selector")

# ── Constants ─────────────────────────────────────────────────────────────────

# All 16 active datasets (DocBank and DeepForm were dropped).
ALL_DATASETS: List[str] = [
    "RVL-CDIP",
    "FUNSD",
    "CORD",
    "SROIE",
    "TextVQA",
    "ST-VQA",
    "HierText",
    "PubLayNet",
    "DocVQA",
    "InfographicVQA",
    "ChartQA",
    "TabFact",
    "WikiTableQuestions",
    "VisualMRC",
    "MP-DocVQA",
    "SlideVQA",
]

CONSTRAINTS: Dict = {
    "per_task_min": {
        "T1": 100,
        "T2": 170,
        "T3": 100,
        "T4": 220,
        "T5": 150,
        "T6": 110,
    },
    "per_tier_min": {1: 200, 2: 480, 3: 200},
    "per_dataset_min": 15,
}

VALIDATION_TIER_WEIGHTS: Dict[int, float] = {1: 0.20, 2: 0.55, 3: 0.25}

# For sigma calibration: subsample at most this many points to estimate median dist.
SIGMA_SUBSAMPLE = 2000
RANDOM_SEED = 42


# ── Feature embedding ─────────────────────────────────────────────────────────

def _one_hot(value: str, categories: List[str]) -> List[float]:
    """Return a one-hot list for *value* over *categories*. Unknown → all-zero."""
    return [1.0 if value == cat else 0.0 for cat in categories]


def embed_sample(sample: dict) -> np.ndarray:
    """
    Embed a benchmark sample into a 30-dimensional feature vector.

    Dimensions:
      0–5   : task_type one-hot (T1–T6)
      6–8   : tier one-hot (1, 2, 3)
      9–24  : source_dataset one-hot (16 datasets)
      25–27 : complexity scalars (vds_est/4, rds_est/4, ses_est/4)
      28–29 : structural flags (has_table, has_chart)
    """
    task_onehot = _one_hot(sample["task_type"], TASK_TYPES)                           # 6
    tier_onehot = _one_hot(str(sample.get("tier_final", 2)), ["1", "2", "3"])         # 3
    dataset_onehot = _one_hot(sample["source_dataset"], ALL_DATASETS)                  # 16
    complexity = [
        sample.get("vds_est", 2) / 4.0,
        sample.get("rds_est", 2) / 4.0,
        sample.get("ses_est", 1) / 4.0,
    ]                                                                                   # 3
    structural = [
        float(sample.get("has_table", False)),
        float(sample.get("has_chart", False)),
    ]                                                                                   # 2
    vec = task_onehot + tier_onehot + dataset_onehot + complexity + structural         # 30
    return np.array(vec, dtype=np.float32)


# ── RBF kernel helpers ────────────────────────────────────────────────────────

def _calibrate_sigma(X: np.ndarray, subsample: int = SIGMA_SUBSAMPLE, seed: int = RANDOM_SEED) -> float:
    """
    Estimate σ as the median pairwise Euclidean distance on a random subsample.
    Falls back to 1.0 if the matrix is trivially small.
    """
    n = X.shape[0]
    if n <= 1:
        return 1.0

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(subsample, n), replace=False)
    Xs = X[idx]

    # Pairwise squared distances via broadcasting (memory-efficient for ≤2K points)
    diff = Xs[:, np.newaxis, :] - Xs[np.newaxis, :, :]   # (m, m, d)
    sq_dists = (diff ** 2).sum(axis=-1)                   # (m, m)

    # Take upper triangle (exclude diagonal zeros)
    upper = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    median_sq = float(np.median(upper))
    sigma = float(np.sqrt(median_sq)) if median_sq > 0 else 1.0
    log.info("Calibrated σ = %.4f (from %d sample subsample)", sigma, len(idx))
    return sigma


def _rbf_kernel(X: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute full RBF kernel matrix S[i,j] = exp(-||xi-xj||² / (2σ²)).
    Uses ||xi-xj||² = ||xi||² + ||xj||² - 2 xi·xj for efficiency.
    Returns (n, n) float32 array.
    """
    sq_norms = (X ** 2).sum(axis=1)                          # (n,)
    sq_dists = (
        sq_norms[:, np.newaxis]
        + sq_norms[np.newaxis, :]
        - 2.0 * (X @ X.T)
    )
    sq_dists = np.maximum(sq_dists, 0.0)                     # numerical safety
    gamma = 1.0 / (2.0 * sigma ** 2)
    return np.exp(-gamma * sq_dists).astype(np.float32)


# ── Constraint seeding ────────────────────────────────────────────────────────

def _seed_constrained(
    samples: List[dict],
    constraints: Dict,
    anchor_size: int,
    rng: random.Random,
) -> List[int]:
    """
    Greedily populate the initial anchor set to satisfy hard constraints.

    Returns a list of integer indices (into *samples*) that form the seed.
    If constraints cannot be satisfied within anchor_size, logs a warning and
    returns the best-effort seed.
    """
    # Build lookup maps
    by_task: Dict[str, List[int]] = defaultdict(list)
    by_tier: Dict[int, List[int]] = defaultdict(list)
    by_dataset: Dict[str, List[int]] = defaultdict(list)

    for i, s in enumerate(samples):
        by_task[s["task_type"]].append(i)
        tier = int(s.get("tier_final", 2))
        by_tier[tier].append(i)
        by_dataset[s["source_dataset"]].append(i)

    # Shuffle each bucket for random tie-breaking
    for lst in by_task.values():
        rng.shuffle(lst)
    for lst in by_tier.values():
        rng.shuffle(lst)
    for lst in by_dataset.values():
        rng.shuffle(lst)

    selected: set = set()
    cursors: Dict = {
        "task": {k: 0 for k in by_task},
        "tier": {k: 0 for k in by_tier},
        "dataset": {k: 0 for k in by_dataset},
    }

    def _pick_from(bucket: List[int], cursor_key, cursor_dict: Dict) -> Optional[int]:
        """Pick next unselected item from bucket."""
        c = cursor_dict[cursor_key]
        while c < len(bucket):
            idx = bucket[c]
            c += 1
            cursor_dict[cursor_key] = c
            if idx not in selected:
                return idx
        cursor_dict[cursor_key] = c
        return None

    def _fill_to(bucket: List[int], cursor_key, cursor_dict: Dict, need: int) -> int:
        added = 0
        while added < need and len(selected) < anchor_size:
            idx = _pick_from(bucket, cursor_key, cursor_dict)
            if idx is None:
                break
            selected.add(idx)
            added += 1
        return added

    # 1. Fill per-task minimums, capped to what's actually available
    per_task_min = constraints["per_task_min"]
    for task, min_count in per_task_min.items():
        available = len(by_task.get(task, []))
        effective_min = min(min_count, available)  # never demand more than exists
        if effective_min == 0:
            log.warning("Task type %s has no candidates in benchmark, skipping constraint.", task)
            continue
        if available < min_count:
            log.warning("Task type %s: only %d available (constraint wants %d), using all.", task, available, min_count)
        have = sum(1 for i in selected if samples[i]["task_type"] == task)
        need = max(0, effective_min - have)
        added = _fill_to(by_task.get(task, []), task, cursors["task"], need)
        if added < need:
            log.warning(
                "Task %s: needed %d more but only %d available after existing selection.",
                task, need, added,
            )

    # 2. Fill per-tier minimums
    per_tier_min = constraints["per_tier_min"]
    for tier, min_count in per_tier_min.items():
        have = sum(1 for i in selected if int(samples[i].get("tier_final", 2)) == tier)
        need = max(0, min_count - have)
        added = _fill_to(by_tier.get(tier, []), tier, cursors["tier"], need)
        if added < need:
            log.warning(
                "Tier %d: needed %d more but only %d available.",
                tier, need, added,
            )

    # 3. Fill per-dataset minimums
    per_dataset_min = constraints["per_dataset_min"]
    for dataset, idxs in by_dataset.items():
        have = sum(1 for i in selected if samples[i]["source_dataset"] == dataset)
        need = max(0, per_dataset_min - have)
        added = _fill_to(idxs, dataset, cursors["dataset"], need)
        if added < need:
            log.warning(
                "Dataset %s: needed %d more but only %d available.",
                dataset, need, added,
            )

    log.info("Constraint seeding complete: %d samples in initial anchor set.", len(selected))
    return list(selected)


# ── Greedy submodular maximisation ────────────────────────────────────────────

def _greedy_facility_location(
    S: np.ndarray,
    seed_indices: List[int],
    target_size: int,
) -> List[int]:
    """
    Greedy (1-1/e)-approximate facility location.

    F(A) = sum_i max_{j in A} S[i, j]

    Starting from *seed_indices*, iteratively add the sample that maximises
    the marginal gain until |A| = target_size.

    Args:
        S:            (n, n) similarity matrix.
        seed_indices: Indices already in A (constraint seeds).
        target_size:  Desired |A|.

    Returns:
        List of indices in A (seed + greedily added).
    """
    n = S.shape[0]
    anchor_set = list(seed_indices)
    in_anchor = set(anchor_set)

    # contribution[i] = max_{j in A} S[i, j]  for all i
    if anchor_set:
        contribution = S[:, anchor_set].max(axis=1)          # (n,)
    else:
        contribution = np.zeros(n, dtype=np.float32)

    remaining = target_size - len(anchor_set)
    log.info(
        "Greedy FL: starting from %d seeds, adding %d more (total target %d).",
        len(anchor_set), remaining, target_size,
    )

    report_interval = max(1, remaining // 10)

    for step in range(remaining):
        if step % report_interval == 0 or step == remaining - 1:
            log.info("  Greedy step %d / %d …", step + 1, remaining)

        # Marginal gain for each candidate = sum_i max(contribution[i], S[i, c]) - contribution[i]
        # = sum_i max(0, S[i, c] - contribution[i])
        gains = np.maximum(0.0, S - contribution[:, np.newaxis]).sum(axis=0)  # (n,)

        # Zero out already-selected candidates
        for idx in in_anchor:
            gains[idx] = -1.0

        best = int(np.argmax(gains))
        in_anchor.add(best)
        anchor_set.append(best)

        # Update contributions
        contribution = np.maximum(contribution, S[:, best])

    log.info("Greedy FL complete. Anchor set size: %d", len(anchor_set))
    return anchor_set


# ── Validation set sampling ───────────────────────────────────────────────────

def _sample_validation(
    samples: List[dict],
    anchor_indices: set,
    validation_size: int,
    tier_weights: Dict[int, float],
    rng: random.Random,
) -> List[int]:
    """
    Stratified sample of *validation_size* from samples NOT in *anchor_indices*.

    Samples are grouped by tier and drawn proportional to *tier_weights*.
    Any leftover quota is redistributed to the largest available pool.
    """
    remaining_by_tier: Dict[int, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        if i not in anchor_indices:
            tier = int(s.get("tier_final", 2))
            remaining_by_tier[tier].append(i)

    # Shuffle each tier bucket
    for lst in remaining_by_tier.values():
        rng.shuffle(lst)

    # Compute per-tier quotas
    total_weight = sum(tier_weights.get(t, 0) for t in remaining_by_tier)
    quotas: Dict[int, int] = {}
    allocated = 0
    tiers_sorted = sorted(remaining_by_tier.keys())

    for i, tier in enumerate(tiers_sorted):
        w = tier_weights.get(tier, 0) / total_weight if total_weight > 0 else 1 / len(tiers_sorted)
        if i == len(tiers_sorted) - 1:
            # Last tier gets the remainder to avoid rounding drift
            quotas[tier] = validation_size - allocated
        else:
            quotas[tier] = round(validation_size * w)
        allocated += quotas[tier]

    val_indices: List[int] = []
    for tier in tiers_sorted:
        pool = remaining_by_tier[tier]
        take = min(quotas.get(tier, 0), len(pool))
        val_indices.extend(pool[:take])

    # If we're short (due to small pools), fill from any remaining
    if len(val_indices) < validation_size:
        all_remaining = set(i for idxs in remaining_by_tier.values() for i in idxs)
        all_remaining -= set(val_indices)
        extras = list(all_remaining)
        rng.shuffle(extras)
        need = validation_size - len(val_indices)
        val_indices.extend(extras[:need])
        if len(val_indices) < validation_size:
            log.warning(
                "Validation set is short: requested %d, got %d (pool exhausted).",
                validation_size, len(val_indices),
            )

    return val_indices[:validation_size]


# ── Reporting ─────────────────────────────────────────────────────────────────

def _build_report(
    samples: List[dict],
    anchor_indices: List[int],
    val_indices: List[int],
    sigma: float,
    elapsed_s: float,
) -> dict:
    """Build a structured selection report dict."""

    def _distribution(idxs: List[int], key: str) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for i in idxs:
            counts[str(samples[i].get(key, "unknown"))] += 1
        return dict(sorted(counts.items()))

    def _tier_key(i: int) -> str:
        return str(samples[i].get("tier_final", 2))

    anchor_task_dist = _distribution(anchor_indices, "task_type")
    anchor_tier_dist = _distribution(anchor_indices, "tier_final")
    anchor_dataset_dist = _distribution(anchor_indices, "source_dataset")
    val_tier_dist = _distribution(val_indices, "tier_final")
    val_task_dist = _distribution(val_indices, "task_type")

    return {
        "anchor_size": len(anchor_indices),
        "validation_size": len(val_indices),
        "total_benchmark_size": len(samples),
        "sigma": round(sigma, 6),
        "elapsed_seconds": round(elapsed_s, 2),
        "constraints": CONSTRAINTS,
        "anchor_distribution": {
            "by_task": anchor_task_dist,
            "by_tier": anchor_tier_dist,
            "by_dataset": anchor_dataset_dist,
        },
        "validation_distribution": {
            "by_tier": val_tier_dist,
            "by_task": val_task_dist,
        },
        "constraint_checks": {
            "per_task_min": {
                task: {
                    "required": req,
                    "actual": anchor_task_dist.get(task, 0),
                    "satisfied": anchor_task_dist.get(task, 0) >= req,
                }
                for task, req in CONSTRAINTS["per_task_min"].items()
            },
            "per_tier_min": {
                str(tier): {
                    "required": req,
                    "actual": anchor_tier_dist.get(str(tier), 0),
                    "satisfied": anchor_tier_dist.get(str(tier), 0) >= req,
                }
                for tier, req in CONSTRAINTS["per_tier_min"].items()
            },
            "per_dataset_min": {
                ds: {
                    "required": CONSTRAINTS["per_dataset_min"],
                    "actual": anchor_dataset_dist.get(ds, 0),
                    "satisfied": anchor_dataset_dist.get(ds, 0) >= CONSTRAINTS["per_dataset_min"],
                }
                for ds in ALL_DATASETS
                if anchor_dataset_dist.get(ds, 0) > 0 or ds in [s["source_dataset"] for s in samples]
            },
        },
    }


def _log_report_summary(report: dict) -> None:
    log.info("=" * 60)
    log.info("ANCHOR SELECTION REPORT")
    log.info("=" * 60)
    log.info("  Anchor set size : %d", report["anchor_size"])
    log.info("  Validation size : %d", report["validation_size"])
    log.info("  Sigma (RBF)     : %.4f", report["sigma"])
    log.info("  Elapsed         : %.1f s", report["elapsed_seconds"])
    log.info("")
    log.info("  Anchor by task  : %s", report["anchor_distribution"]["by_task"])
    log.info("  Anchor by tier  : %s", report["anchor_distribution"]["by_tier"])
    log.info("")

    checks = report["constraint_checks"]
    all_ok = True
    for task, info in checks["per_task_min"].items():
        if not info["satisfied"]:
            log.warning("  CONSTRAINT FAIL: task %s: need %d, got %d", task, info["required"], info["actual"])
            all_ok = False
    for tier, info in checks["per_tier_min"].items():
        if not info["satisfied"]:
            log.warning("  CONSTRAINT FAIL: tier %s: need %d, got %d", tier, info["required"], info["actual"])
            all_ok = False
    for ds, info in checks.get("per_dataset_min", {}).items():
        if not info["satisfied"]:
            log.warning("  CONSTRAINT FAIL: dataset %s: need %d, got %d", ds, info["required"], info["actual"])
            all_ok = False

    if all_ok:
        log.info("  All hard constraints satisfied.")
    log.info("=" * 60)


# ── Public API ────────────────────────────────────────────────────────────────

def run_anchor_selection(
    benchmark_path: str = "data/benchmark/benchmark_5000.jsonl",
    anchor_output: str = "data/anchor_set/anchor_ids.json",
    validation_output: str = "data/validation_set/validation_ids.json",
    checkpoint_dir: str = "data/phase2_checkpoints",
    anchor_size: int = 1500,
    validation_size: int = 750,
) -> Tuple[str, str]:
    """
    Select the anchor set (1,500 samples) and validation set (750 samples)
    from the 5,000-sample benchmark using submodular facility location.

    Args:
        benchmark_path:   Path to benchmark_5000.jsonl (absolute or relative to project root).
        anchor_output:    Output path for anchor_ids.json.
        validation_output: Output path for validation_ids.json.
        checkpoint_dir:   Directory for the .done checkpoint file.
        anchor_size:      Number of samples in the anchor set.
        validation_size:  Number of samples in the validation set.

    Returns:
        Tuple of (anchor_output_path, validation_output_path).

    Raises:
        FileNotFoundError: if benchmark_path does not exist.
        ValueError: if the benchmark has fewer samples than anchor_size + validation_size.
    """
    # ── Resolve paths ─────────────────────────────────────────────────────────
    def _resolve(p: str) -> str:
        pp = Path(p)
        return str(pp) if pp.is_absolute() else str(_PROJECT_ROOT / p)

    benchmark_path = _resolve(benchmark_path)
    anchor_output = _resolve(anchor_output)
    validation_output = _resolve(validation_output)
    checkpoint_dir = _resolve(checkpoint_dir)

    checkpoint_file = Path(checkpoint_dir) / "anchor_selector.done"
    report_path = str(Path(anchor_output).parent / "selection_report.json")

    # ── Checkpoint check ──────────────────────────────────────────────────────
    if checkpoint_file.exists():
        log.info("Checkpoint found: anchor selection already complete. Skipping.")
        return anchor_output, validation_output

    # ── Load benchmark ────────────────────────────────────────────────────────
    log.info("Loading benchmark from %s …", benchmark_path)
    samples = load_jsonl(benchmark_path)
    n = len(samples)
    log.info("Loaded %d benchmark samples.", n)

    if n < anchor_size + validation_size:
        raise ValueError(
            f"Benchmark has only {n} samples, but anchor_size ({anchor_size}) + "
            f"validation_size ({validation_size}) = {anchor_size + validation_size}."
        )

    # ── Feature matrix ────────────────────────────────────────────────────────
    log.info("Computing 30-dim feature vectors …")
    X = np.stack([embed_sample(s) for s in samples], axis=0)  # (n, 30)
    log.info("Feature matrix shape: %s", X.shape)

    # ── Sigma calibration ─────────────────────────────────────────────────────
    t0 = time.monotonic()
    sigma = _calibrate_sigma(X, subsample=SIGMA_SUBSAMPLE, seed=RANDOM_SEED)

    # ── RBF similarity matrix ─────────────────────────────────────────────────
    log.info("Computing %d × %d RBF kernel matrix (sigma=%.4f) …", n, n, sigma)
    S = _rbf_kernel(X, sigma)
    log.info("Kernel matrix computed (%.1f MB).", S.nbytes / 1e6)

    # ── Constraint-based seed ─────────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)
    log.info("Seeding anchor set to satisfy hard constraints …")
    seed_indices = _seed_constrained(samples, CONSTRAINTS, anchor_size, rng)

    # ── Greedy facility location ──────────────────────────────────────────────
    anchor_indices = _greedy_facility_location(S, seed_indices, target_size=anchor_size)

    # ── Validation set ────────────────────────────────────────────────────────
    log.info("Sampling validation set (n=%d, tier_weights=%s) …", validation_size, VALIDATION_TIER_WEIGHTS)
    val_indices = _sample_validation(
        samples,
        anchor_indices=set(anchor_indices),
        validation_size=validation_size,
        tier_weights=VALIDATION_TIER_WEIGHTS,
        rng=rng,
    )

    elapsed = time.monotonic() - t0

    # ── Build outputs ─────────────────────────────────────────────────────────
    anchor_ids = [samples[i]["sample_id"] for i in anchor_indices]
    val_ids = [samples[i]["sample_id"] for i in val_indices]

    # Validate no overlap
    overlap = set(anchor_ids) & set(val_ids)
    if overlap:
        log.error("BUG: %d sample_ids appear in both anchor and validation sets!", len(overlap))
        raise RuntimeError(f"Anchor/validation overlap detected: {list(overlap)[:5]}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    Path(anchor_output).parent.mkdir(parents=True, exist_ok=True)
    Path(validation_output).parent.mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    with open(anchor_output, "w") as f:
        json.dump(anchor_ids, f, indent=2)
    log.info("Wrote %d anchor IDs → %s", len(anchor_ids), anchor_output)

    with open(validation_output, "w") as f:
        json.dump(val_ids, f, indent=2)
    log.info("Wrote %d validation IDs → %s", len(val_ids), validation_output)

    # ── Selection report ──────────────────────────────────────────────────────
    report = _build_report(samples, anchor_indices, val_indices, sigma, elapsed)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Wrote selection report → %s", report_path)

    _log_report_summary(report)

    # ── Write checkpoint ──────────────────────────────────────────────────────
    checkpoint_file.write_text("done\n")
    log.info("Checkpoint written → %s", checkpoint_file)

    return anchor_output, validation_output


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="DocRouteBench: Anchor Set Selector (submodular facility location)"
    )
    parser.add_argument(
        "--benchmark",
        default="data/benchmark/benchmark_5000.jsonl",
        help="Path to benchmark_5000.jsonl",
    )
    parser.add_argument(
        "--anchor-output",
        default="data/anchor_set/anchor_ids.json",
        help="Output path for anchor_ids.json",
    )
    parser.add_argument(
        "--validation-output",
        default="data/validation_set/validation_ids.json",
        help="Output path for validation_ids.json",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="data/phase2_checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--anchor-size",
        type=int,
        default=1500,
        help="Number of anchor samples (default: 1500)",
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=750,
        help="Number of validation samples (default: 750)",
    )
    args = parser.parse_args()

    run_anchor_selection(
        benchmark_path=args.benchmark,
        anchor_output=args.anchor_output,
        validation_output=args.validation_output,
        checkpoint_dir=args.checkpoint_dir,
        anchor_size=args.anchor_size,
        validation_size=args.validation_size,
    )


if __name__ == "__main__":
    main()
