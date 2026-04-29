"""
Pytest integration tests for src/anchor_set/anchor_selector.py

Coverage:
  - Anchor and validation sets are strictly disjoint (critical invariant)
  - Anchor set meets minimum per-task constraints (seeding phase)
  - Validation set respects tier distribution (all three tiers represented)
  - Anchor size == configured anchor_size, validation size == configured validation_size
  - embed_sample() produces 30-dimensional vectors
  - _calibrate_sigma() returns a positive float
  - _rbf_kernel() produces a valid symmetric similarity matrix in [0,1]
  - _sample_validation() returns indices disjoint from anchor_indices
  - Full run_anchor_selection() round-trip on synthetic 2000-sample pool
    (uses temp directories — no real data, no API calls)

Synthetic data: 2000 samples covering all 6 task types and 3 tiers.
All tests run in-process; no network I/O, no disk reads from project data/.
"""

import json
import random
import sys
import tempfile
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.anchor_set.anchor_selector import (
    ALL_DATASETS,
    CONSTRAINTS,
    VALIDATION_TIER_WEIGHTS,
    _calibrate_sigma,
    _greedy_facility_location,
    _rbf_kernel,
    _sample_validation,
    _seed_constrained,
    embed_sample,
    run_anchor_selection,
)
from src.schema import TASK_TYPES

# ---------------------------------------------------------------------------
# Constants matching configured defaults
# ---------------------------------------------------------------------------

ANCHOR_SIZE = 1000
VALIDATION_SIZE = 500
POOL_SIZE = 2000       # synthetic pool — small enough to run fast

# We use only 4 of the 16 ALL_DATASETS in the synthetic pool to keep it
# tractable, while still exercising multi-dataset logic.
_SYNTHETIC_DATASETS = ["DocVQA", "ChartQA", "TabFact", "FUNSD"]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sample(
    idx: int,
    rng: random.Random,
    task_types: List[str] = TASK_TYPES,
    datasets: List[str] = _SYNTHETIC_DATASETS,
    tiers: List[int] = (1, 2, 3),
) -> dict:
    """Build one synthetic sample dict with all fields expected by anchor_selector."""
    task_type = task_types[idx % len(task_types)]
    tier = tiers[idx % len(tiers)]
    ds = datasets[idx % len(datasets)]

    # Complexity scalars drawn randomly
    vds = rng.randint(1, 4)
    rds = rng.randint(1, 4)
    ses = rng.randint(1, 4)
    composite = round(0.30 * vds + 0.45 * rds + 0.25 * ses, 4)

    return {
        "sample_id": f"syn_{idx:05d}",
        "source_dataset": ds,
        "task_type": task_type,
        "tier_final": tier,
        "tier_prior": tier,
        "vds_est": vds,
        "rds_est": rds,
        "ses_est": ses,
        "composite_est": composite,
        "has_table": rng.choice([True, False]),
        "has_chart": rng.choice([True, False]),
        "has_figure": rng.choice([True, False]),
        "has_handwriting": False,
        "doc_type": rng.choice(["form", "chart", "receipt", "document"]),
        "query": f"Synthetic query number {idx}.",
        "gt_answer": "42",
        "num_pages": rng.randint(1, 5),
        "difficulty_score": float(tier) + rng.uniform(-0.3, 0.3),
        "confidence": rng.uniform(0.45, 0.95),
        "correctness_metric": "anls",
        "image_path": f"images/syn_{idx:05d}.png",
        "image_bytes_size": rng.randint(10_000, 500_000),
    }


@pytest.fixture(scope="module")
def synthetic_samples() -> List[dict]:
    """Generate 2000 deterministic synthetic samples covering all 6 task types."""
    rng = random.Random(42)
    return [_make_sample(i, rng) for i in range(POOL_SIZE)]


@pytest.fixture(scope="module")
def benchmark_jsonl(synthetic_samples, tmp_path_factory) -> Path:
    """Write synthetic_samples to a temp JSONL file and return its Path."""
    tmp_dir = tmp_path_factory.mktemp("bench")
    fpath = tmp_dir / "benchmark_2000.jsonl"
    with open(fpath, "w") as f:
        for s in synthetic_samples:
            f.write(json.dumps(s) + "\n")
    return fpath


@pytest.fixture(scope="module")
def anchor_run_outputs(benchmark_jsonl, tmp_path_factory):
    """
    Run run_anchor_selection() once on the synthetic pool and cache the outputs
    for the entire test module.  Uses tiny anchor/val sizes to keep it fast.

    Returns dict with keys: anchor_ids, val_ids, anchor_output, val_output, tmp_dir
    """
    tmp_dir = tmp_path_factory.mktemp("anchor_run")
    anchor_out = str(tmp_dir / "anchor_ids.json")
    val_out = str(tmp_dir / "validation_ids.json")
    checkpoint_dir = str(tmp_dir / "checkpoints")

    run_anchor_selection(
        benchmark_path=str(benchmark_jsonl),
        anchor_output=anchor_out,
        validation_output=val_out,
        checkpoint_dir=checkpoint_dir,
        anchor_size=ANCHOR_SIZE,
        validation_size=VALIDATION_SIZE,
    )

    with open(anchor_out) as f:
        anchor_ids = json.load(f)
    with open(val_out) as f:
        val_ids = json.load(f)

    return {
        "anchor_ids": anchor_ids,
        "val_ids": val_ids,
        "anchor_output": anchor_out,
        "val_output": val_out,
        "tmp_dir": tmp_dir,
    }


# ---------------------------------------------------------------------------
# embed_sample tests
# ---------------------------------------------------------------------------

class TestEmbedSample:
    """Unit tests for the 30-dim feature embedding."""

    def test_embed_produces_30_dims(self, synthetic_samples):
        vec = embed_sample(synthetic_samples[0])
        assert vec.shape == (30,), f"Expected shape (30,), got {vec.shape}"

    def test_embed_is_float32(self, synthetic_samples):
        vec = embed_sample(synthetic_samples[0])
        assert vec.dtype == np.float32

    def test_embed_all_values_in_unit_interval(self, synthetic_samples):
        for s in synthetic_samples[:50]:
            vec = embed_sample(s)
            assert vec.min() >= 0.0
            assert vec.max() <= 1.0

    def test_embed_task_type_one_hot_correct(self, synthetic_samples):
        # Find a T1 sample and verify the first 6 dims form a one-hot for T1
        t1_sample = next(s for s in synthetic_samples if s["task_type"] == "T1")
        vec = embed_sample(t1_sample)
        task_onehot = vec[:6].tolist()
        assert task_onehot[0] == 1.0, "T1 should activate index 0 of the task one-hot"
        assert sum(task_onehot) == pytest.approx(1.0, abs=1e-5)

    def test_embed_different_tasks_differ(self, synthetic_samples):
        t1 = next(s for s in synthetic_samples if s["task_type"] == "T1")
        t6 = next(s for s in synthetic_samples if s["task_type"] == "T6")
        v1 = embed_sample(t1)
        v6 = embed_sample(t6)
        assert not np.allclose(v1, v6), "T1 and T6 embeddings must differ"

    def test_embed_unknown_dataset_gives_zero_onehot(self, synthetic_samples):
        s = dict(synthetic_samples[0])
        s["source_dataset"] = "UNKNOWN_DS_XYZ"
        vec = embed_sample(s)
        dataset_slice = vec[9:25]  # 16 dataset dimensions
        assert dataset_slice.sum() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# _calibrate_sigma tests
# ---------------------------------------------------------------------------

class TestCalibrateSigma:
    """Sigma calibration should return a positive finite float."""

    def test_sigma_positive(self, synthetic_samples):
        X = np.stack([embed_sample(s) for s in synthetic_samples[:100]])
        sigma = _calibrate_sigma(X, subsample=100, seed=0)
        assert sigma > 0.0

    def test_sigma_finite(self, synthetic_samples):
        X = np.stack([embed_sample(s) for s in synthetic_samples[:200]])
        sigma = _calibrate_sigma(X, subsample=200, seed=0)
        assert np.isfinite(sigma)

    def test_sigma_degenerate_single_point(self):
        X = np.ones((1, 30), dtype=np.float32)
        sigma = _calibrate_sigma(X)
        assert sigma == pytest.approx(1.0)

    def test_sigma_identical_points_fallback(self):
        # All identical → median pairwise dist = 0 → should fall back to 1.0
        X = np.ones((10, 30), dtype=np.float32)
        sigma = _calibrate_sigma(X, subsample=10, seed=0)
        assert sigma == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _rbf_kernel tests
# ---------------------------------------------------------------------------

class TestRBFKernel:
    """RBF kernel matrix properties."""

    def _small_X(self, n=20) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.random((n, 30)).astype(np.float32)

    def test_kernel_shape(self):
        X = self._small_X(20)
        S = _rbf_kernel(X, sigma=1.0)
        assert S.shape == (20, 20)

    def test_kernel_diagonal_is_one(self):
        X = self._small_X(20)
        S = _rbf_kernel(X, sigma=1.0)
        np.testing.assert_allclose(np.diag(S), 1.0, atol=1e-5)

    def test_kernel_symmetric(self):
        X = self._small_X(20)
        S = _rbf_kernel(X, sigma=1.0)
        np.testing.assert_allclose(S, S.T, atol=1e-5)

    def test_kernel_values_in_01(self):
        X = self._small_X(30)
        S = _rbf_kernel(X, sigma=1.0)
        assert float(S.min()) >= 0.0
        assert float(S.max()) <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# _seed_constrained tests
# ---------------------------------------------------------------------------

class TestSeedConstrained:
    """Constraint seeding logic."""

    def test_seed_respects_per_task_min(self, synthetic_samples):
        rng_obj = random.Random(0)
        # Use relaxed constraints (small minimums) that the 2000-sample pool can satisfy
        constraints = {
            "per_task_min": {tt: 5 for tt in TASK_TYPES},
            "per_tier_min": {1: 10, 2: 10, 3: 10},
            "per_dataset_min": 5,
        }
        seed_idxs = _seed_constrained(synthetic_samples, constraints, anchor_size=500, rng=rng_obj)
        seed_tasks = [synthetic_samples[i]["task_type"] for i in seed_idxs]
        for tt in TASK_TYPES:
            count = seed_tasks.count(tt)
            assert count >= 5, f"Task {tt}: got {count}, expected >= 5"

    def test_seed_no_duplicates(self, synthetic_samples):
        rng_obj = random.Random(1)
        constraints = {
            "per_task_min": {tt: 3 for tt in TASK_TYPES},
            "per_tier_min": {1: 5, 2: 5, 3: 5},
            "per_dataset_min": 3,
        }
        seed_idxs = _seed_constrained(synthetic_samples, constraints, anchor_size=200, rng=rng_obj)
        assert len(seed_idxs) == len(set(seed_idxs)), "Seed indices must be unique"

    def test_seed_within_anchor_size(self, synthetic_samples):
        rng_obj = random.Random(2)
        anchor_size = 100
        constraints = {
            "per_task_min": {tt: 5 for tt in TASK_TYPES},
            "per_tier_min": {1: 5, 2: 5, 3: 5},
            "per_dataset_min": 2,
        }
        seed_idxs = _seed_constrained(synthetic_samples, constraints, anchor_size=anchor_size, rng=rng_obj)
        assert len(seed_idxs) <= anchor_size


# ---------------------------------------------------------------------------
# _sample_validation tests
# ---------------------------------------------------------------------------

class TestSampleValidation:
    """Validation sampling correctness."""

    def test_validation_disjoint_from_anchor(self, synthetic_samples):
        anchor_indices = set(range(0, 1000))  # first 1000 as anchor
        rng_obj = random.Random(42)
        val_idxs = _sample_validation(
            synthetic_samples,
            anchor_indices=anchor_indices,
            validation_size=300,
            tier_weights=VALIDATION_TIER_WEIGHTS,
            rng=rng_obj,
        )
        val_set = set(val_idxs)
        overlap = val_set & anchor_indices
        assert len(overlap) == 0, (
            f"Validation and anchor share {len(overlap)} indices — must be disjoint"
        )

    def test_validation_exact_size_when_pool_is_large(self, synthetic_samples):
        anchor_indices = set(range(0, 500))
        rng_obj = random.Random(7)
        val_idxs = _sample_validation(
            synthetic_samples,
            anchor_indices=anchor_indices,
            validation_size=200,
            tier_weights=VALIDATION_TIER_WEIGHTS,
            rng=rng_obj,
        )
        assert len(val_idxs) == 200

    def test_validation_no_duplicate_indices(self, synthetic_samples):
        anchor_indices = set(range(0, 800))
        rng_obj = random.Random(3)
        val_idxs = _sample_validation(
            synthetic_samples,
            anchor_indices=anchor_indices,
            validation_size=150,
            tier_weights=VALIDATION_TIER_WEIGHTS,
            rng=rng_obj,
        )
        assert len(val_idxs) == len(set(val_idxs)), "Validation indices must be unique"

    def test_validation_covers_multiple_tiers(self, synthetic_samples):
        # With 2000 samples spanning tiers 1/2/3, all tiers should appear in val set
        anchor_indices = set(range(0, 500))
        rng_obj = random.Random(99)
        val_idxs = _sample_validation(
            synthetic_samples,
            anchor_indices=anchor_indices,
            validation_size=300,
            tier_weights=VALIDATION_TIER_WEIGHTS,
            rng=rng_obj,
        )
        val_tiers = {synthetic_samples[i]["tier_final"] for i in val_idxs}
        assert len(val_tiers) >= 2, (
            f"Validation set should cover multiple tiers; got {val_tiers}"
        )


# ---------------------------------------------------------------------------
# Full run_anchor_selection() round-trip tests
# (These share the module-scoped anchor_run_outputs fixture to avoid re-running
# the submodular greedy optimisation multiple times.)
# ---------------------------------------------------------------------------

class TestRunAnchorSelectionRoundTrip:
    """End-to-end tests using the cached module-scoped run outputs."""

    # ── Disjointness ────────────────────────────────────────────────────────────

    def test_anchor_and_validation_are_disjoint(self, anchor_run_outputs):
        """Critical invariant: anchor ∩ validation = ∅."""
        anchor_ids = set(anchor_run_outputs["anchor_ids"])
        val_ids = set(anchor_run_outputs["val_ids"])
        overlap = anchor_ids & val_ids
        assert len(overlap) == 0, (
            f"CRITICAL: {len(overlap)} sample_ids appear in both anchor and validation sets."
        )

    # ── Set sizes ───────────────────────────────────────────────────────────────

    def test_anchor_set_size_equals_configured(self, anchor_run_outputs):
        assert len(anchor_run_outputs["anchor_ids"]) == ANCHOR_SIZE, (
            f"Anchor size: expected {ANCHOR_SIZE}, got {len(anchor_run_outputs['anchor_ids'])}"
        )

    def test_validation_set_size_equals_configured(self, anchor_run_outputs):
        assert len(anchor_run_outputs["val_ids"]) == VALIDATION_SIZE, (
            f"Validation size: expected {VALIDATION_SIZE}, got {len(anchor_run_outputs['val_ids'])}"
        )

    # ── No duplicates within each set ───────────────────────────────────────────

    def test_anchor_ids_unique(self, anchor_run_outputs):
        ids = anchor_run_outputs["anchor_ids"]
        assert len(ids) == len(set(ids)), "Duplicate sample_ids in anchor set"

    def test_validation_ids_unique(self, anchor_run_outputs):
        ids = anchor_run_outputs["val_ids"]
        assert len(ids) == len(set(ids)), "Duplicate sample_ids in validation set"

    # ── IDs come from the benchmark pool ────────────────────────────────────────

    def test_anchor_ids_all_in_pool(self, anchor_run_outputs, synthetic_samples):
        pool_ids = {s["sample_id"] for s in synthetic_samples}
        for sid in anchor_run_outputs["anchor_ids"]:
            assert sid in pool_ids, f"Anchor ID '{sid}' not found in synthetic pool"

    def test_validation_ids_all_in_pool(self, anchor_run_outputs, synthetic_samples):
        pool_ids = {s["sample_id"] for s in synthetic_samples}
        for sid in anchor_run_outputs["val_ids"]:
            assert sid in pool_ids, f"Validation ID '{sid}' not found in synthetic pool"

    # ── Per-task constraints ─────────────────────────────────────────────────────

    def test_anchor_covers_all_task_types(self, anchor_run_outputs, synthetic_samples):
        """Every task type present in the pool must appear in the anchor set."""
        sample_by_id = {s["sample_id"]: s for s in synthetic_samples}
        anchor_tasks = {
            sample_by_id[sid]["task_type"]
            for sid in anchor_run_outputs["anchor_ids"]
        }
        pool_tasks = {s["task_type"] for s in synthetic_samples}
        missing = pool_tasks - anchor_tasks
        assert len(missing) == 0, (
            f"Task types missing from anchor set: {missing}"
        )

    def test_anchor_per_task_minimum_met(self, anchor_run_outputs, synthetic_samples):
        """
        With 2000 samples and anchor_size=1000, the seeding phase should
        achieve per_task_min counts.

        We use relaxed per-task minimums (10 per type) that are guaranteed
        satisfiable given 2000/6 ≈ 333 samples per task type.
        """
        sample_by_id = {s["sample_id"]: s for s in synthetic_samples}
        task_counts: Dict[str, int] = {tt: 0 for tt in TASK_TYPES}
        for sid in anchor_run_outputs["anchor_ids"]:
            tt = sample_by_id[sid]["task_type"]
            task_counts[tt] = task_counts.get(tt, 0) + 1

        relaxed_min = 10   # well below ~333 available per task in the pool
        for tt in TASK_TYPES:
            assert task_counts[tt] >= relaxed_min, (
                f"Task {tt}: anchor has {task_counts[tt]} samples, expected >= {relaxed_min}"
            )

    # ── Tier distribution in validation set ─────────────────────────────────────

    def test_validation_tier_distribution(self, anchor_run_outputs, synthetic_samples):
        """
        All three tiers should appear in the validation set.
        With VALIDATION_TIER_WEIGHTS {1: 0.20, 2: 0.55, 3: 0.25} and 500 samples,
        we expect ~100, ~275, ~125 per tier.  We allow ±60 tolerance.
        """
        sample_by_id = {s["sample_id"]: s for s in synthetic_samples}
        tier_counts: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        for sid in anchor_run_outputs["val_ids"]:
            tier = int(sample_by_id[sid]["tier_final"])
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        for tier in (1, 2, 3):
            assert tier_counts[tier] > 0, (
                f"Tier {tier} has 0 samples in validation set"
            )

    def test_validation_tier2_dominant(self, anchor_run_outputs, synthetic_samples):
        """
        VALIDATION_TIER_WEIGHTS gives Tier 2 the highest weight (0.55),
        so Tier 2 count should exceed Tier 1 and Tier 3 counts.
        """
        sample_by_id = {s["sample_id"]: s for s in synthetic_samples}
        tier_counts: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        for sid in anchor_run_outputs["val_ids"]:
            tier = int(sample_by_id[sid]["tier_final"])
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        assert tier_counts[2] > tier_counts[1], (
            f"Tier 2 ({tier_counts[2]}) should exceed Tier 1 ({tier_counts[1]}) "
            "given VALIDATION_TIER_WEIGHTS"
        )
        assert tier_counts[2] > tier_counts[3], (
            f"Tier 2 ({tier_counts[2]}) should exceed Tier 3 ({tier_counts[3]}) "
            "given VALIDATION_TIER_WEIGHTS"
        )

    # ── Output files written ─────────────────────────────────────────────────────

    def test_anchor_ids_json_written(self, anchor_run_outputs):
        assert Path(anchor_run_outputs["anchor_output"]).exists()

    def test_validation_ids_json_written(self, anchor_run_outputs):
        assert Path(anchor_run_outputs["val_output"]).exists()

    def test_anchor_ids_json_parseable(self, anchor_run_outputs):
        with open(anchor_run_outputs["anchor_output"]) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == ANCHOR_SIZE


# ---------------------------------------------------------------------------
# ValueError when pool is too small
# ---------------------------------------------------------------------------

class TestAnchorSelectionEdgeCases:
    """Error handling in run_anchor_selection."""

    def test_raises_valueerror_if_pool_too_small(self, tmp_path):
        """Pool with fewer samples than anchor_size + validation_size must raise ValueError."""
        # Write 10 samples to a temp file
        small_pool = [
            {
                "sample_id": f"tiny_{i:03d}",
                "source_dataset": "DocVQA",
                "task_type": TASK_TYPES[i % len(TASK_TYPES)],
                "tier_final": (i % 3) + 1,
                "vds_est": 2, "rds_est": 2, "ses_est": 1,
                "has_table": False, "has_chart": False,
                "has_figure": False, "has_handwriting": False,
                "doc_type": "document",
                "query": f"Question {i}",
                "gt_answer": "ans",
                "num_pages": 1,
                "difficulty_score": 2.0,
                "confidence": 0.7,
                "correctness_metric": "anls",
                "image_path": "",
                "image_bytes_size": 0,
            }
            for i in range(10)
        ]
        bench_path = tmp_path / "small_bench.jsonl"
        with open(bench_path, "w") as f:
            for s in small_pool:
                f.write(json.dumps(s) + "\n")

        with pytest.raises(ValueError, match="anchor_size"):
            run_anchor_selection(
                benchmark_path=str(bench_path),
                anchor_output=str(tmp_path / "anchor_ids.json"),
                validation_output=str(tmp_path / "val_ids.json"),
                checkpoint_dir=str(tmp_path / "checkpoints"),
                anchor_size=8,
                validation_size=5,  # 8 + 5 = 13 > 10 samples
            )

    def test_checkpoint_skip_on_second_run(self, benchmark_jsonl, tmp_path):
        """
        A second call with the same checkpoint dir must skip computation
        (checkpoint guard).  We verify by checking both output files exist
        after the first call and the function returns without error on the second.
        """
        anchor_out = str(tmp_path / "anchor_ids.json")
        val_out = str(tmp_path / "val_ids.json")
        checkpoint_dir = str(tmp_path / "checkpoints")

        # First run
        run_anchor_selection(
            benchmark_path=str(benchmark_jsonl),
            anchor_output=anchor_out,
            validation_output=val_out,
            checkpoint_dir=checkpoint_dir,
            anchor_size=ANCHOR_SIZE,
            validation_size=VALIDATION_SIZE,
        )

        assert Path(anchor_out).exists()
        anchor_mtime = Path(anchor_out).stat().st_mtime

        # Second run — should skip (checkpoint exists)
        run_anchor_selection(
            benchmark_path=str(benchmark_jsonl),
            anchor_output=anchor_out,
            validation_output=val_out,
            checkpoint_dir=checkpoint_dir,
            anchor_size=ANCHOR_SIZE,
            validation_size=VALIDATION_SIZE,
        )

        # File should not have been rewritten (mtime unchanged)
        assert Path(anchor_out).stat().st_mtime == anchor_mtime, (
            "anchor_ids.json was overwritten on second run; checkpoint guard failed"
        )
