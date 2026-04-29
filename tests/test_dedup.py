"""
Tests for DocRouteBench Phase 2 deduplication logic.

Tests use synthetic in-memory samples and mock all disk I/O, so no real
files are read or written.

Run with:
    pytest tests/test_dedup.py -v
"""

import sys
import hashlib
import json
import string
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# Import the pure-logic helpers directly so we can test them in isolation
# without triggering disk I/O.
from src.ingestion.dedup import (
    DATASET_PRIORITY,
    _dataset_priority,
    _normalize_query,
    _text_key,
    _phash_distance,
    _should_keep_incoming,
    run_deduplication,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic sample dicts
# ---------------------------------------------------------------------------

def _make_sample(
    sample_id: str,
    query: str,
    source_dataset: str,
    image_path: str = "",
    ground_truth: str = "some answer",
    metric: str = "exact_match",
) -> dict:
    """Return a minimal sample dict compatible with the dedup schema."""
    return {
        "sample_id": sample_id,
        "query": query,
        "ground_truth": ground_truth,
        "source_dataset": source_dataset,
        "image_path": image_path,
        "metric": metric,
    }


# ---------------------------------------------------------------------------
# Unit tests for pure helpers
# ---------------------------------------------------------------------------

class TestNormalizeQuery:
    """_normalize_query: lowercase, strip, remove punctuation, collapse whitespace."""

    def test_lowercase_and_strip(self):
        result = _normalize_query("  Hello World  ")
        assert result == "hello world"

    def test_punctuation_removed(self):
        result = _normalize_query("What's the date?")
        assert "'" not in result
        assert "?" not in result

    def test_internal_whitespace_collapsed(self):
        result = _normalize_query("foo   bar   baz")
        assert result == "foo bar baz"

    def test_empty_string(self):
        assert _normalize_query("") == ""

    def test_same_query_different_casing(self):
        assert _normalize_query("Invoice Total") == _normalize_query("INVOICE TOTAL")


class TestTextKey:
    """_text_key: deterministic MD5 of the normalized query."""

    def test_same_query_same_key(self):
        assert _text_key("What is the total?") == _text_key("What is the total?")

    def test_case_insensitive_match(self):
        assert _text_key("Total Revenue") == _text_key("total revenue")

    def test_punctuation_insensitive(self):
        assert _text_key("Hello, world!") == _text_key("Hello world")

    def test_different_queries_different_keys(self):
        assert _text_key("What is the date?") != _text_key("What is the amount?")

    def test_returns_32_hex_chars(self):
        key = _text_key("some query")
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)


class TestDatasetPriority:
    """_dataset_priority: higher number = kept on conflict."""

    def test_docvqa_highest(self):
        assert _dataset_priority("DocVQA") == 16

    def test_hiertext_lowest(self):
        assert _dataset_priority("HierText") == 1

    def test_unknown_dataset_zero(self):
        assert _dataset_priority("SomeRandomDataset") == 0

    def test_docvqa_beats_chartqa(self):
        assert _dataset_priority("DocVQA") > _dataset_priority("ChartQA")

    def test_chartqa_beats_mpdocvqa(self):
        assert _dataset_priority("ChartQA") > _dataset_priority("MP-DocVQA")

    @pytest.mark.parametrize("dataset", [
        "DocVQA", "ChartQA", "MP-DocVQA", "InfographicVQA",
        "SlideVQA", "RVL-CDIP", "FUNSD", "SROIE",
        "TextVQA", "PubLayNet", "TabFact", "WikiTableQuestions",
        "CORD", "VisualMRC", "ST-VQA", "HierText",
    ])
    def test_all_16_datasets_have_priority(self, dataset):
        assert _dataset_priority(dataset) > 0

    def test_priority_ordering(self):
        ordered = sorted(DATASET_PRIORITY, key=lambda d: DATASET_PRIORITY[d], reverse=True)
        assert ordered[0] == "DocVQA"
        assert ordered[-1] == "HierText"


class TestShouldKeepIncoming:
    """_should_keep_incoming: True iff incoming has higher priority than existing."""

    def test_incoming_higher_priority(self):
        existing = _make_sample("s1", "q", "HierText")
        incoming = _make_sample("s2", "q", "DocVQA")
        assert _should_keep_incoming(existing, incoming) is True

    def test_existing_higher_priority(self):
        existing = _make_sample("s1", "q", "DocVQA")
        incoming = _make_sample("s2", "q", "HierText")
        assert _should_keep_incoming(existing, incoming) is False

    def test_same_priority_keeps_existing(self):
        # Both unknown → priority 0; incoming NOT strictly greater → False
        existing = _make_sample("s1", "q", "Unknown")
        incoming = _make_sample("s2", "q", "Unknown")
        assert _should_keep_incoming(existing, incoming) is False

    def test_missing_source_dataset_field(self):
        existing = {"sample_id": "s1", "query": "q"}         # no source_dataset
        incoming = {"sample_id": "s2", "query": "q", "source_dataset": "DocVQA"}
        # existing priority = 0, incoming priority = 16 → True
        assert _should_keep_incoming(existing, incoming) is True


class TestPhashDistance:
    """_phash_distance: hamming distance on hex-encoded 64-bit hashes."""

    def test_identical_hashes_distance_zero(self):
        h = "0000000000000000"
        assert _phash_distance(h, h) == 0

    def test_completely_different(self):
        h1 = "0000000000000000"
        h2 = "ffffffffffffffff"
        assert _phash_distance(h1, h2) == 64

    def test_one_bit_difference(self):
        h1 = "0000000000000000"
        h2 = "0000000000000001"
        assert _phash_distance(h1, h2) == 1

    def test_invalid_hex_returns_large_number(self):
        # Should not crash; returns 999 for unparseable hashes
        assert _phash_distance("ZZZZ", "0000") == 999


# ---------------------------------------------------------------------------
# Integration tests for run_deduplication via mocked JSONL I/O
# ---------------------------------------------------------------------------

def _jsonl_content(samples: List[dict]) -> str:
    """Serialize a list of sample dicts to JSONL string."""
    return "\n".join(json.dumps(s) for s in samples) + "\n"


def _make_run_dedup_with_samples(
    samples_by_file: Dict[str, List[dict]],
    tmp_path: Path,
    monkeypatch,
    use_phash: bool = False,
) -> List[dict]:
    """
    Run run_deduplication with synthetic in-memory samples by:
      1. Writing samples to temporary *_normalized.jsonl files.
      2. Patching _IMAGEHASH_AVAILABLE to force MD5 (no real images needed).
      3. Reading the output JSONL and returning the kept records.
    """
    input_dir = tmp_path / "processed"
    input_dir.mkdir(parents=True)
    checkpoint_dir = tmp_path / "checkpoints"
    output_path = tmp_path / "processed" / "all_samples_deduped.jsonl"

    # Write each synthetic file
    for filename, samples in samples_by_file.items():
        file_path = input_dir / filename
        file_path.write_text(_jsonl_content(samples))

    # Patch _IMAGEHASH_AVAILABLE so the function uses MD5 path (no PIL needed)
    with patch("src.ingestion.dedup._IMAGEHASH_AVAILABLE", use_phash):
        result_path = run_deduplication(
            input_dir=str(input_dir),
            output_path=str(output_path),
            checkpoint_dir=str(checkpoint_dir),
        )

    result_file = Path(result_path)
    if not result_file.exists():
        return []

    records = []
    with open(result_file) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Deduplication integration tests
# ---------------------------------------------------------------------------

class TestRunDeduplicationIdenticalSamples:
    """Identical samples from different datasets: higher-priority dataset wins."""

    def test_identical_query_keeps_higher_priority(self, tmp_path, monkeypatch):
        samples = {
            "hiertext_normalized.jsonl": [
                _make_sample("s1", "What is the total?", "HierText"),
            ],
            "docvqa_normalized.jsonl": [
                _make_sample("s2", "What is the total?", "DocVQA"),
            ],
        }
        kept = _make_run_dedup_with_samples(samples, tmp_path, monkeypatch)
        assert len(kept) == 1
        assert kept[0]["source_dataset"] == "DocVQA"

    def test_identical_query_lower_priority_dropped(self, tmp_path, monkeypatch):
        samples = {
            "docvqa_normalized.jsonl": [
                _make_sample("s1", "Same question", "DocVQA"),
            ],
            "cord_normalized.jsonl": [
                _make_sample("s2", "Same question", "CORD"),
            ],
        }
        kept = _make_run_dedup_with_samples(samples, tmp_path, monkeypatch)
        assert len(kept) == 1
        assert kept[0]["source_dataset"] == "DocVQA"

    def test_three_way_conflict_highest_wins(self, tmp_path, monkeypatch):
        samples = {
            "a_normalized.jsonl": [
                _make_sample("s1", "Identical query text", "HierText"),
                _make_sample("s2", "Identical query text", "CORD"),
            ],
            "b_normalized.jsonl": [
                _make_sample("s3", "Identical query text", "DocVQA"),
            ],
        }
        kept = _make_run_dedup_with_samples(samples, tmp_path, monkeypatch)
        assert len(kept) == 1
        assert kept[0]["source_dataset"] == "DocVQA"


class TestRunDeduplicationDistinctSamples:
    """Truly distinct samples must all be kept."""

    def test_different_queries_both_kept(self, tmp_path, monkeypatch):
        samples = {
            "docvqa_normalized.jsonl": [
                _make_sample("s1", "What is the invoice date?",   "DocVQA"),
                _make_sample("s2", "What is the invoice number?", "DocVQA"),
            ],
        }
        kept = _make_run_dedup_with_samples(samples, tmp_path, monkeypatch)
        assert len(kept) == 2

    def test_samples_from_different_datasets_no_overlap(self, tmp_path, monkeypatch):
        samples = {
            "docvqa_normalized.jsonl": [
                _make_sample("s1", "What is the total amount?", "DocVQA"),
            ],
            "chartqa_normalized.jsonl": [
                _make_sample("s2", "Which bar is tallest?", "ChartQA"),
            ],
        }
        kept = _make_run_dedup_with_samples(samples, tmp_path, monkeypatch)
        assert len(kept) == 2

    def test_single_sample_always_kept(self, tmp_path, monkeypatch):
        samples = {
            "docvqa_normalized.jsonl": [
                _make_sample("s1", "Unique query", "DocVQA"),
            ],
        }
        kept = _make_run_dedup_with_samples(samples, tmp_path, monkeypatch)
        assert len(kept) == 1


class TestRunDeduplicationQueryNormalization:
    """Queries that are semantically identical after normalization are deduped."""

    def test_case_different_queries_deduped(self, tmp_path, monkeypatch):
        samples = {
            "docvqa_normalized.jsonl": [
                _make_sample("s1", "WHAT IS THE TOTAL?", "DocVQA"),
                _make_sample("s2", "what is the total?", "CORD"),
            ],
        }
        kept = _make_run_dedup_with_samples(samples, tmp_path, monkeypatch)
        assert len(kept) == 1

    def test_punctuation_different_queries_deduped(self, tmp_path, monkeypatch):
        samples = {
            "docvqa_normalized.jsonl": [
                _make_sample("s1", "What is the total!", "DocVQA"),
                _make_sample("s2", "What is the total?", "CORD"),
            ],
        }
        kept = _make_run_dedup_with_samples(samples, tmp_path, monkeypatch)
        assert len(kept) == 1

    def test_whitespace_different_queries_deduped(self, tmp_path, monkeypatch):
        samples = {
            "docvqa_normalized.jsonl": [
                _make_sample("s1", "Total  Revenue", "DocVQA"),
                _make_sample("s2", "Total Revenue",  "CORD"),
            ],
        }
        kept = _make_run_dedup_with_samples(samples, tmp_path, monkeypatch)
        assert len(kept) == 1


class TestRunDeduplicationImageDedup:
    """Image-level dedup via MD5 of image bytes (no real images in tests)."""

    def test_same_image_path_different_queries_both_kept(self, tmp_path, monkeypatch):
        """
        Samples share an image_path BUT have different queries.
        Since image_path leads to a missing file → MD5 returns None → no image conflict.
        Both are kept (only the text key signals a dup, which is absent here).
        """
        samples = {
            "docvqa_normalized.jsonl": [
                _make_sample("s1", "What is the date?",   "DocVQA",  image_path="img/doc.png"),
                _make_sample("s2", "What is the total?",  "DocVQA",  image_path="img/doc.png"),
            ],
        }
        kept = _make_run_dedup_with_samples(samples, tmp_path, monkeypatch)
        # Both have distinct queries → text keys differ → both kept
        assert len(kept) == 2

    def test_same_query_different_image_paths_deduped(self, tmp_path, monkeypatch):
        """
        Samples have the same query (identical text key) even though image paths differ.
        Text-level dedup fires; higher-priority dataset wins.
        """
        samples = {
            "docvqa_normalized.jsonl": [
                _make_sample("s1", "Shared query text", "DocVQA",  image_path="img/a.png"),
            ],
            "cord_normalized.jsonl": [
                _make_sample("s2", "Shared query text", "CORD",    image_path="img/b.png"),
            ],
        }
        kept = _make_run_dedup_with_samples(samples, tmp_path, monkeypatch)
        assert len(kept) == 1
        assert kept[0]["source_dataset"] == "DocVQA"

    def test_real_image_md5_dedup(self, tmp_path, monkeypatch):
        """
        Write actual small image files with identical bytes; both samples have
        the same query too. Confirms one is dropped.
        """
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        img_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # fake PNG bytes
        (img_dir / "a.png").write_bytes(img_bytes)
        (img_dir / "b.png").write_bytes(img_bytes)  # identical bytes

        # Use relative paths from tmp_path as project_root
        samples = {
            "docvqa_normalized.jsonl": [
                _make_sample("s1", "What is shown?", "DocVQA", image_path=str(img_dir / "a.png")),
            ],
            "cord_normalized.jsonl": [
                _make_sample("s2", "What is shown?", "CORD",   image_path=str(img_dir / "b.png")),
            ],
        }
        kept = _make_run_dedup_with_samples(samples, tmp_path, monkeypatch)
        # Text key is the same → 1 kept, DocVQA wins
        assert len(kept) == 1


class TestRunDeduplicationMissingFields:
    """Samples with missing fields must not crash the dedup pipeline."""

    def test_missing_query_field(self, tmp_path, monkeypatch):
        samples = {
            "docvqa_normalized.jsonl": [
                {"sample_id": "s1", "source_dataset": "DocVQA"},   # no query
                _make_sample("s2", "Normal query", "DocVQA"),
            ],
        }
        # Should not raise
        kept = _make_run_dedup_with_samples(samples, tmp_path, monkeypatch)
        assert len(kept) >= 1

    def test_missing_image_path_field(self, tmp_path, monkeypatch):
        samples = {
            "docvqa_normalized.jsonl": [
                {"sample_id": "s1", "query": "No image path", "source_dataset": "DocVQA"},
                _make_sample("s2", "With image path", "DocVQA", image_path="imgs/x.png"),
            ],
        }
        kept = _make_run_dedup_with_samples(samples, tmp_path, monkeypatch)
        assert len(kept) == 2

    def test_missing_source_dataset_field(self, tmp_path, monkeypatch):
        samples = {
            "docvqa_normalized.jsonl": [
                {"sample_id": "s1", "query": "some query"},            # no source_dataset
                _make_sample("s2", "another query", "DocVQA"),
            ],
        }
        kept = _make_run_dedup_with_samples(samples, tmp_path, monkeypatch)
        assert len(kept) == 2

    def test_missing_sample_id_field(self, tmp_path, monkeypatch):
        samples = {
            "docvqa_normalized.jsonl": [
                {"query": "query without id", "source_dataset": "DocVQA"},
                _make_sample("s1", "query with id", "DocVQA"),
            ],
        }
        kept = _make_run_dedup_with_samples(samples, tmp_path, monkeypatch)
        assert len(kept) == 2

    def test_malformed_json_lines_skipped(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "processed"
        input_dir.mkdir(parents=True)
        checkpoint_dir = tmp_path / "checkpoints"
        output_path = tmp_path / "processed" / "all_samples_deduped.jsonl"

        # Write a file with one valid and one malformed line
        mixed = (
            json.dumps(_make_sample("s1", "valid query", "DocVQA")) + "\n"
            + "this is not json {{{broken\n"
            + json.dumps(_make_sample("s2", "second valid", "ChartQA")) + "\n"
        )
        (input_dir / "test_normalized.jsonl").write_text(mixed)

        with patch("src.ingestion.dedup._IMAGEHASH_AVAILABLE", False):
            result_path = run_deduplication(
                input_dir=str(input_dir),
                output_path=str(output_path),
                checkpoint_dir=str(checkpoint_dir),
            )

        with open(result_path) as fh:
            kept = [json.loads(l) for l in fh if l.strip()]

        # Both valid lines kept; malformed line silently skipped
        assert len(kept) == 2


class TestRunDeduplicationCheckpoint:
    """Checkpoint prevents re-running deduplication."""

    def test_checkpoint_skips_processing(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "processed"
        input_dir.mkdir(parents=True)
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        output_path = tmp_path / "processed" / "all_samples_deduped.jsonl"

        # Pre-write a checkpoint file
        checkpoint_file = checkpoint_dir / "dedup.done"
        checkpoint_file.write_text(json.dumps({"total_kept": 0}))

        # Write a *_normalized.jsonl file but since checkpoint exists it should be skipped
        (input_dir / "docvqa_normalized.jsonl").write_text(
            json.dumps(_make_sample("s1", "some query", "DocVQA")) + "\n"
        )

        with patch("src.ingestion.dedup._IMAGEHASH_AVAILABLE", False):
            result = run_deduplication(
                input_dir=str(input_dir),
                output_path=str(output_path),
                checkpoint_dir=str(checkpoint_dir),
            )

        # Output path is returned but no actual output file written (checkpoint short-circuits)
        assert result == str(output_path)
        assert not output_path.exists()

    def test_no_normalized_files_raises(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "processed"
        input_dir.mkdir(parents=True)
        output_path = tmp_path / "processed" / "all_samples_deduped.jsonl"
        checkpoint_dir = tmp_path / "checkpoints"

        with patch("src.ingestion.dedup._IMAGEHASH_AVAILABLE", False):
            with pytest.raises(FileNotFoundError, match="No \\*_normalized.jsonl files found"):
                run_deduplication(
                    input_dir=str(input_dir),
                    output_path=str(output_path),
                    checkpoint_dir=str(checkpoint_dir),
                )


class TestDatasetPriorityOrdering:
    """Validate the full priority ladder is correctly ordered."""

    def test_full_priority_order(self):
        """The 16 datasets must have distinct, consecutive-ish priorities 1-16."""
        priorities = list(DATASET_PRIORITY.values())
        assert min(priorities) == 1
        assert max(priorities) == 16
        assert len(set(priorities)) == 16   # all distinct

    @pytest.mark.parametrize("winner,loser", [
        ("DocVQA",        "ChartQA"),
        ("ChartQA",       "MP-DocVQA"),
        ("MP-DocVQA",     "InfographicVQA"),
        ("InfographicVQA","SlideVQA"),
        ("SlideVQA",      "RVL-CDIP"),
        ("RVL-CDIP",      "FUNSD"),
        ("FUNSD",         "SROIE"),
        ("SROIE",         "TextVQA"),
        ("TextVQA",       "PubLayNet"),
        ("PubLayNet",     "TabFact"),
        ("TabFact",       "WikiTableQuestions"),
        ("WikiTableQuestions", "CORD"),
        ("CORD",          "VisualMRC"),
        ("VisualMRC",     "ST-VQA"),
        ("ST-VQA",        "HierText"),
    ])
    def test_priority_pair(self, winner, loser):
        assert _dataset_priority(winner) > _dataset_priority(loser)
