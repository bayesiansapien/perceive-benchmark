"""
DocRouteBench — SROIE Dataset Adapter

Dataset : darentang/sroie  (fallback: jinhybr/OCR-SROIE-2019)
Task    : T2 — Structured Extraction
Metric  : field_f1
Split   : test

SROIE (Scanned Receipts OCR and Information Extraction) contains receipt
images annotated with four key fields: company, date, address, and total.
The query is fixed and the ground truth is a JSON object with those four keys.
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

from src.ingestion.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

# HuggingFace dataset identifiers tried in order
# priyank-m/SROIE_2019_text_recognition: each sample is a cropped text region
# from a receipt image, with the transcribed text as ground truth.
# Task becomes T2 OCR recognition: "What text is written in this region?"
_HF_IDS = [
    "priyank-m/SROIE_2019_text_recognition",
    "darentang/sroie",
    "jinhybr/OCR-SROIE-2019",
]

_FIXED_QUERY = "What text is written in this region of the receipt?"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_sroie_dataset():
    """
    Try each known HuggingFace ID in turn.  Returns (dataset, hf_id) on
    success.  Raises RuntimeError with a helpful message if all attempts fail.
    """
    from datasets import load_dataset

    last_exc: Exception | None = None
    for hf_id in _HF_IDS:
        try:
            logger.info("[sroie] Trying HuggingFace ID: %s …", hf_id)
            ds = load_dataset(hf_id, split="test", )
            logger.info("[sroie] Loaded '%s' — %d samples", hf_id, len(ds))
            return ds, hf_id
        except Exception as exc:
            logger.warning("[sroie] '%s' failed: %s", hf_id, exc)
            last_exc = exc

    raise RuntimeError(
        f"[sroie] Could not load SROIE from any known HuggingFace ID.\n"
        f"Last error: {last_exc}\n"
        f"{_DOWNLOAD_HINT}"
    )


def _extract_entity_field(row: dict, field: str) -> str:
    """
    Pull a SROIE entity field from a dataset row, trying multiple common
    column layouts.  Returns an empty string if the field is absent/null.

    Layouts handled:
      1. row["entities"] is a dict  → row["entities"][field]
      2. row["entities"] is a JSON string → parse then index
      3. Direct column: row[field] or row[field.upper()]
    """
    # Layout 1 & 2 — nested under an "entities" column
    entities_raw = row.get("entities") or row.get("annotation")
    if entities_raw is not None:
        if isinstance(entities_raw, str):
            try:
                entities_raw = json.loads(entities_raw)
            except Exception:
                entities_raw = None
        if isinstance(entities_raw, dict):
            val = (
                entities_raw.get(field)
                or entities_raw.get(field.upper())
                or ""
            )
            return str(val).strip() if val else ""

    # Layout 3 — flat columns
    for key in (field, field.upper(), field.capitalize()):
        val = row.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()

    return ""


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class SROIEAdapter(BaseAdapter):
    """Adapter for the SROIE receipt information extraction benchmark."""

    dataset_name = "sroie"
    task_type    = "T2"
    metric       = "field_f1"

    def iter_samples(self) -> Iterator[dict]:
        try:
            ds, _ = _load_sroie_dataset()
        except RuntimeError as exc:
            # Print a clear, user-facing error and bail gracefully
            print("\n" + "!"*60)
            print("ERROR: SROIE dataset could not be loaded.")
            print("-"*60)
            print(str(exc))
            print("!"*60 + "\n")
            logger.error("[sroie] %s", exc)
            return

        for idx, row in enumerate(ds):
            image = row.get("image")
            if image is None:
                logger.warning("[sroie] Sample %d has no image — skipping", idx)
                continue

            # priyank-m/SROIE_2019_text_recognition: each sample is a cropped
            # text region. gt_answer is the transcribed text for that region.
            # Falls back to field extraction layout for older HF mirrors.
            if "text" in row and row["text"]:
                gt_answer = str(row["text"]).strip()
                metric = "anls"   # OCR recognition uses ANLS
            else:
                gt_dict = {
                    field: _extract_entity_field(row, field)
                    for field in ["company", "date", "address", "total"]
                }
                if not any(gt_dict.values()):
                    logger.warning("[sroie] Sample %d has no GT — skipping", idx)
                    continue
                gt_answer = json.dumps(gt_dict, ensure_ascii=False)
                metric = self.metric

            if not gt_answer:
                continue

            yield {
                "sample_id"         : f"sroie_test_{idx:06d}",
                "query"             : _FIXED_QUERY,
                "gt_answer"         : gt_answer,
                "gt_answer_aliases" : [gt_answer],
                "image"             : image,
                "task_type"         : self.task_type,
                "correctness_metric": metric,
                "doc_type"          : "receipt",
                "source_split"      : "test",
                "has_table"         : False,
            }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    adapter = SROIEAdapter(max_samples=3)
    print(f"\n{'='*60}")
    print("SROIE Adapter — smoke test (3 samples)")
    print("="*60)

    for i, sample in enumerate(adapter.iter_samples()):
        print(f"\n--- Sample {i+1} ---")
        print(f"  sample_id : {sample['sample_id']}")
        print(f"  doc_type  : {sample['doc_type']}")
        print(f"  task_type : {sample['task_type']}")
        print(f"  metric    : {sample['correctness_metric']}")
        print(f"  query     : {sample['query']}")
        print(f"  gt_answer : {sample['gt_answer']}")
        print(f"  image     : {type(sample['image']).__name__} {getattr(sample['image'], 'size', '')}")
        if i >= 2:
            break

    print("\nSmoke test complete — run adapter.run() to write full JSONL output.")
    sys.exit(0)
