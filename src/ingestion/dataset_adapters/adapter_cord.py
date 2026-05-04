"""
DocRouteBench: CORD v2 Dataset Adapter

Dataset : naver-clova-ix/cord-v2
Task    : T2, Structured Extraction
Metric  : field_f1
Split   : test (100 samples, all used)

CORD contains scanned receipt images paired with deeply nested JSON ground
truth (menu items, sub-totals, totals, etc.).  We flatten the entire GT tree
into a single-level key:value mapping, build a query that asks the model to
extract exactly those fields, and serialise the flattened dict as the answer.
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

from datasets import load_dataset

from src.ingestion.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(obj, parent_key: str = "", sep: str = ".") -> dict:
    """
    Recursively flatten a nested dict/list structure into dot-separated keys.

    Lists are expanded with a numeric index suffix, e.g. "menu.0.nm".
    Leaf values are cast to str; None / empty-string leaves are dropped.
    """
    items: dict = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(_flatten(v, new_key, sep))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(_flatten(v, new_key, sep))

    else:
        # Leaf node
        val = str(obj).strip() if obj is not None else ""
        if val:
            items[parent_key] = val

    return items


def _parse_gt_json(gt_parse) -> dict:
    """
    Convert the 'ground_truth' field (string or dict) into a flat key:value
    mapping.  Returns an empty dict if parsing fails.
    """
    try:
        if isinstance(gt_parse, str):
            gt_parse = json.loads(gt_parse)

        # cord-v2 wraps the payload in {"gt_parse": {...}}
        if isinstance(gt_parse, dict) and "gt_parse" in gt_parse:
            gt_parse = gt_parse["gt_parse"]

        if not isinstance(gt_parse, dict):
            return {}

        return _flatten(gt_parse)
    except Exception as exc:
        logger.debug("GT parse error: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class CORDAdapter(BaseAdapter):
    """Adapter for the CORD-v2 receipt understanding benchmark."""

    dataset_name = "cord"
    task_type    = "T2"
    metric       = "field_f1"

    def iter_samples(self) -> Iterator[dict]:
        logger.info("[cord] Loading naver-clova-ix/cord-v2 (test split) …")
        try:
            ds = load_dataset("naver-clova-ix/cord-v2", split="test", )
        except Exception as exc:
            logger.error(
                "[cord] Failed to load dataset from HuggingFace: %s\n"
                "       You can download it manually from:\n"
                "       https://huggingface.co/datasets/naver-clova-ix/cord-v2",
                exc,
            )
            return

        for idx, row in enumerate(ds):
            # ----------------------------------------------------------------
            # Parse ground truth
            # ----------------------------------------------------------------
            gt_parse = row.get("ground_truth", {})
            flat_fields = _parse_gt_json(gt_parse)

            if not flat_fields:
                logger.warning("[cord] Sample %d has empty GT: skipping", idx)
                continue

            # ----------------------------------------------------------------
            # Build query
            # ----------------------------------------------------------------
            field_list = ", ".join(flat_fields.keys())
            query = (
                f"Extract the following fields from this receipt: {field_list}"
            )

            # ----------------------------------------------------------------
            # Build gt_answer (JSON string of flattened key:value pairs)
            # ----------------------------------------------------------------
            gt_answer = json.dumps(flat_fields, ensure_ascii=False)

            # ----------------------------------------------------------------
            # Image
            # ----------------------------------------------------------------
            image = row.get("image")
            if image is None:
                logger.warning("[cord] Sample %d has no image, skipping", idx)
                continue

            yield {
                "sample_id"         : f"cord_test_{idx:06d}",
                "query"             : query,
                "gt_answer"         : gt_answer,
                "gt_answer_aliases" : [],
                "image"             : image,
                "task_type"         : self.task_type,
                "correctness_metric": self.metric,
                "doc_type"          : "receipt",
                "source_split"      : "test",
                "has_table"         : True,
            }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    adapter = CORDAdapter(max_samples=3)
    print(f"\n{'='*60}")
    print("CORD Adapter: smoke test (3 samples)")
    print("="*60)

    for i, sample in enumerate(adapter.iter_samples()):
        print(f"\n--- Sample {i+1} ---")
        print(f"  sample_id : {sample['sample_id']}")
        print(f"  doc_type  : {sample['doc_type']}")
        print(f"  task_type : {sample['task_type']}")
        print(f"  metric    : {sample['correctness_metric']}")
        print(f"  query     : {sample['query'][:120]}{'…' if len(sample['query']) > 120 else ''}")
        gt_preview = sample["gt_answer"][:200]
        print(f"  gt_answer : {gt_preview}{'…' if len(sample['gt_answer']) > 200 else ''}")
        print(f"  image     : {type(sample['image']).__name__} {getattr(sample['image'], 'size', '')}")
        if i >= 2:
            break

    print("\nSmoke test complete, run adapter.run() to write full JSONL output.")
    sys.exit(0)
