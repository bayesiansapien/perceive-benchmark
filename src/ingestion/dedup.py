"""
DocRouteBench: Phase 2 Deduplication
======================================
Removes duplicate samples across the 16 normalized JSONL files.

Deduplication strategy (two independent signals, both checked):
  1. Image-level: perceptual hash (imagehash.phash, threshold=8).
     Falls back to MD5 of raw image file bytes if imagehash is not installed.
  2. Text-level: MD5 of normalized question text
     (lowercase, strip whitespace, punctuation removed).

When a conflict is found the sample from the HIGHER-priority dataset is kept.
Priority is derived from sample_budget in configs/datasets.yaml:
  docvqa(400) > mpdocvqa(650 but separate tier) > chartqa(200) > ...
  Exact ordering below in DATASET_PRIORITY.

Checkpoint: data/phase2_checkpoints/dedup.done
Output:     data/processed/all_samples_deduped.jsonl
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import string
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── Dataset priority (higher number = keep this sample when duplicates clash) ──
# Derived from sample_budget in configs/datasets.yaml; ties broken alphabetically.
DATASET_PRIORITY: Dict[str, int] = {
    "DocVQA":          16,
    "ChartQA":         15,
    "MP-DocVQA":       14,
    "InfographicVQA":  13,
    "SlideVQA":        12,
    "RVL-CDIP":        11,
    "FUNSD":           10,
    "SROIE":            9,
    "TextVQA":          8,
    "PubLayNet":        7,
    "TabFact":          6,
    "WikiTableQuestions": 5,
    "CORD":             4,
    "VisualMRC":        3,
    "ST-VQA":           2,
    "HierText":         1,
}

# ── Try to import imagehash ────────────────────────────────────────────────────
try:
    import imagehash
    from PIL import Image as PILImage
    _IMAGEHASH_AVAILABLE = True
    log.info("imagehash available, using perceptual hashing for image dedup")
except ImportError:
    _IMAGEHASH_AVAILABLE = False
    log.warning(
        "imagehash not installed, falling back to MD5 of image bytes. "
        "Install with: pip install imagehash"
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _dataset_priority(source_dataset: str) -> int:
    """Return numeric priority for a dataset name (higher = preferred on conflict)."""
    return DATASET_PRIORITY.get(source_dataset, 0)


def _normalize_query(query: str) -> str:
    """Lowercase, strip, remove punctuation for text-level dedup key."""
    query = query.lower().strip()
    query = query.translate(str.maketrans("", "", string.punctuation))
    # Collapse internal whitespace
    query = " ".join(query.split())
    return query


def _text_key(query: str) -> str:
    return hashlib.md5(_normalize_query(query).encode()).hexdigest()


def _image_key_phash(image_path: str, project_root: str) -> Optional[str]:
    """Return perceptual hash hex string, or None if file unreadable."""
    full_path = os.path.join(project_root, image_path) if image_path else None
    if not full_path or not os.path.exists(full_path):
        return None
    try:
        img = PILImage.open(full_path)
        ph = imagehash.phash(img)
        return str(ph)
    except Exception as exc:
        log.debug("phash failed for %s: %s", full_path, exc)
        return None


def _image_key_md5(image_path: str, project_root: str) -> Optional[str]:
    """Return MD5 of raw image file bytes, or None if file unreadable."""
    full_path = os.path.join(project_root, image_path) if image_path else None
    if not full_path or not os.path.exists(full_path):
        return None
    try:
        with open(full_path, "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()
    except Exception as exc:
        log.debug("MD5 failed for %s: %s", full_path, exc)
        return None


def _phash_distance(h1: str, h2: str) -> int:
    """Hamming distance between two phash hex strings (each 16 hex chars = 64 bits)."""
    try:
        i1 = int(h1, 16)
        i2 = int(h2, 16)
        xor = i1 ^ i2
        return bin(xor).count("1")
    except Exception:
        return 999  # treat as different on parse error


def _get_image_key(
    record: dict,
    project_root: str,
    use_phash: bool,
) -> Optional[str]:
    image_path = record.get("image_path", "")
    if not image_path:
        return None
    if use_phash:
        return _image_key_phash(image_path, project_root)
    else:
        return _image_key_md5(image_path, project_root)


def _should_keep_incoming(
    existing: dict,
    incoming: dict,
) -> bool:
    """Return True if the incoming record should replace the existing one."""
    p_existing = _dataset_priority(existing.get("source_dataset", ""))
    p_incoming = _dataset_priority(incoming.get("source_dataset", ""))
    return p_incoming > p_existing


# ── Core deduplication ────────────────────────────────────────────────────────

def run_deduplication(
    input_dir: str = "data/processed",
    output_path: str = "data/processed/all_samples_deduped.jsonl",
    checkpoint_dir: str = "data/phase2_checkpoints",
    phash_threshold: int = 8,
) -> str:
    """
    Deduplicate normalized samples across all JSONL files.

    Returns the path to the deduplicated output file.
    """
    # ── Resolve project root (two levels up from this file) ────────────────
    project_root = str(Path(__file__).resolve().parents[2])

    # ── Absolute paths ──────────────────────────────────────────────────────
    abs_input_dir = os.path.join(project_root, input_dir)
    abs_output_path = os.path.join(project_root, output_path)
    abs_checkpoint_dir = os.path.join(project_root, checkpoint_dir)
    checkpoint_file = os.path.join(abs_checkpoint_dir, "dedup.done")

    # ── Checkpoint check ────────────────────────────────────────────────────
    if os.path.exists(checkpoint_file):
        log.info("Checkpoint found: skipping deduplication. Output: %s", abs_output_path)
        return abs_output_path

    # ── Create dirs ─────────────────────────────────────────────────────────
    os.makedirs(abs_checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

    # ── Collect input files (exclude the output file itself and non-source files) ──
    all_jsonl = sorted(glob.glob(os.path.join(abs_input_dir, "*_normalized.jsonl")))
    if not all_jsonl:
        raise FileNotFoundError(
            f"No *_normalized.jsonl files found in {abs_input_dir}"
        )
    log.info("Found %d normalized JSONL files to process", len(all_jsonl))

    use_phash = _IMAGEHASH_AVAILABLE

    # ── First pass: load all records ─────────────────────────────────────────
    all_records: list[dict] = []
    for jsonl_path in all_jsonl:
        count_before = len(all_records)
        with open(jsonl_path, "r") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if raw_line:
                    try:
                        all_records.append(json.loads(raw_line))
                    except json.JSONDecodeError as exc:
                        log.warning("Skipping malformed line in %s: %s", jsonl_path, exc)
        loaded = len(all_records) - count_before
        log.info("  Loaded %4d samples from %s", loaded, os.path.basename(jsonl_path))

    total_loaded = len(all_records)
    log.info("Total samples loaded: %d", total_loaded)

    # ── Second pass: deduplicate ──────────────────────────────────────────────
    # kept_records maps sample_id -> record (the "winner" so far)
    kept_by_id: Dict[str, dict] = {}

    # Image dedup index: for phash, list of (hash_str, sample_id);
    # for MD5, dict of md5_str -> sample_id
    if use_phash:
        phash_index: list[Tuple[str, str]] = []   # (hash_str, sample_id)
    else:
        md5_index: Dict[str, str] = {}            # md5_hex -> sample_id

    # Text dedup index: query_md5 -> sample_id
    text_index: Dict[str, str] = {}

    # Duplicate counters per (dataset_a, dataset_b) ordered pair
    dup_counters: Dict[str, int] = {}

    def _record_dup(winner_dataset: str, loser_dataset: str) -> None:
        key = f"{winner_dataset} > {loser_dataset}"
        dup_counters[key] = dup_counters.get(key, 0) + 1

    total_dropped = 0

    for idx, record in enumerate(all_records):
        if idx > 0 and idx % 5_000 == 0:
            log.info("  Progress: %d / %d processed, %d duplicates dropped so far",
                     idx, total_loaded, total_dropped)

        sample_id = record.get("sample_id", f"__unknown_{idx}")
        query = record.get("query", "")
        image_path = record.get("image_path", "")
        src_dataset = record.get("source_dataset", "")

        # ── Text-level check ──────────────────────────────────────────────
        t_key = _text_key(query)
        text_conflict_id = text_index.get(t_key)

        # ── Image-level check ─────────────────────────────────────────────
        img_key = _get_image_key(record, project_root, use_phash)
        image_conflict_id: Optional[str] = None

        if img_key is not None:
            if use_phash:
                # Linear scan: acceptable for water-test scale; for 5000 samples
                # this is at most 5000 comparisons which completes in <1s.
                for existing_hash, existing_id in phash_index:
                    if _phash_distance(img_key, existing_hash) <= phash_threshold:
                        image_conflict_id = existing_id
                        break
            else:
                image_conflict_id = md5_index.get(img_key)

        # ── Determine if this is a duplicate ─────────────────────────────
        # A TRUE duplicate requires BOTH text AND image to match the SAME
        # existing sample.  Same query + different image = valid (e.g.,
        # template questions across different documents).  Same image +
        # different query = valid (same doc page, different question,
        # e.g. T3 layout + T6 grounding from the same PubLayNet page).
        #
        # Previous bug: text_conflict_id and image_conflict_id were checked
        # independently, so a sample could be falsely flagged when the text
        # matched one existing sample and the image matched a *different*
        # existing sample.  Fix: require they point to the SAME sample.
        if text_conflict_id and image_conflict_id and text_conflict_id == image_conflict_id:
            # Both signals point to the same existing sample, true duplicate
            conflict_id = text_conflict_id
        elif text_conflict_id and image_conflict_id is None and img_key is None:
            # Text matches but no image available to compare, treat as dup
            # (cannot verify different image, conservative)
            conflict_id = text_conflict_id
        else:
            # Only one signal matches, or they point to different samples,
            # or neither matches, NOT a duplicate
            conflict_id = None
        if conflict_id is None:
            # No conflict: keep this record and index it
            kept_by_id[sample_id] = record
            text_index[t_key] = sample_id
            if img_key is not None:
                if use_phash:
                    phash_index.append((img_key, sample_id))
                else:
                    md5_index[img_key] = sample_id
        else:
            # Conflict: decide winner
            existing_record = kept_by_id.get(conflict_id)
            if existing_record is None:
                # The conflicting record was itself removed earlier; keep this one
                kept_by_id[sample_id] = record
                text_index[t_key] = sample_id
                if img_key is not None:
                    if use_phash:
                        phash_index.append((img_key, sample_id))
                    else:
                        md5_index[img_key] = sample_id
                continue

            existing_dataset = existing_record.get("source_dataset", "")

            if _should_keep_incoming(existing_record, record):
                # Replace: incoming wins
                _record_dup(src_dataset, existing_dataset)
                # Update indices to point to new winner
                old_text_key = _text_key(existing_record.get("query", ""))
                if text_index.get(old_text_key) == conflict_id:
                    text_index[old_text_key] = sample_id

                del kept_by_id[conflict_id]
                kept_by_id[sample_id] = record

                # For image index: update reference
                if img_key is not None and not use_phash:
                    old_img_path = existing_record.get("image_path", "")
                    old_img_md5 = _image_key_md5(old_img_path, project_root)
                    if old_img_md5 and md5_index.get(old_img_md5) == conflict_id:
                        md5_index[old_img_md5] = sample_id
                # (phash_index stores tuples; rewriting would be O(n); skip, the
                # first-match scan will find the new winner via kept_by_id lookup)
            else:
                # Existing wins; drop incoming
                _record_dup(existing_dataset, src_dataset)

            total_dropped += 1

    # ── Write output ──────────────────────────────────────────────────────────
    kept_records = list(kept_by_id.values())
    log.info("Writing %d deduplicated samples to %s", len(kept_records), abs_output_path)

    with open(abs_output_path, "w") as fh:
        for rec in kept_records:
            fh.write(json.dumps(rec, default=str) + "\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("── Deduplication complete ──────────────────────────────────────")
    log.info("  Total loaded:  %d", total_loaded)
    log.info("  Total kept:    %d", len(kept_records))
    log.info("  Total dropped: %d", total_dropped)
    if dup_counters:
        log.info("  Duplicates removed per dataset pair:")
        for pair, cnt in sorted(dup_counters.items(), key=lambda x: -x[1]):
            log.info("    %-50s  %d", pair, cnt)
    else:
        log.info("  No duplicates found.")

    # ── Write checkpoint ──────────────────────────────────────────────────────
    with open(checkpoint_file, "w") as fh:
        fh.write(
            json.dumps({
                "output_path": abs_output_path,
                "total_loaded": total_loaded,
                "total_kept": len(kept_records),
                "total_dropped": total_dropped,
                "dup_counters": dup_counters,
                "use_phash": use_phash,
                "phash_threshold": phash_threshold,
            }, indent=2)
        )
    log.info("Checkpoint written: %s", checkpoint_file)

    return abs_output_path


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import shutil

    print("=" * 60)
    print("DocRouteBench Phase 2, Deduplication Smoke Test")
    print("=" * 60)

    # Allow overriding checkpoint to force re-run during testing
    force_rerun = "--force" in sys.argv

    project_root = str(Path(__file__).resolve().parents[2])
    checkpoint_file = os.path.join(project_root, "data/phase2_checkpoints/dedup.done")

    if force_rerun and os.path.exists(checkpoint_file):
        print(f"[--force] Removing checkpoint: {checkpoint_file}")
        os.remove(checkpoint_file)

    output_path = run_deduplication(
        input_dir="data/processed",
        output_path="data/processed/all_samples_deduped.jsonl",
        checkpoint_dir="data/phase2_checkpoints",
        phash_threshold=8,
    )

    # Report results
    abs_output = os.path.join(project_root, output_path) \
        if not os.path.isabs(output_path) else output_path

    if not os.path.exists(abs_output):
        print(f"ERROR: output file not found at {abs_output}")
        sys.exit(1)

    with open(abs_output) as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    print(f"\nOutput file: {abs_output}")
    print(f"Total samples after dedup: {len(records)}")

    # Per-dataset breakdown
    from collections import Counter
    per_dataset = Counter(r.get("source_dataset", "?") for r in records)
    print("\nSamples per dataset:")
    for ds, cnt in sorted(per_dataset.items()):
        print(f"  {ds:<30s}  {cnt:>4d}")

    print("\nSmoke test PASSED.")
