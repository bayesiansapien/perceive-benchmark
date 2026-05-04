"""
DocRouteBench: ChartQA Dataset Adapter

HuggingFace ID : lmms-lab/ChartQA
Split          : test
Task type      : T4 (Chart QA)
Metric         : relaxed_accuracy

ChartQA has two subsets:
  - "human"     : human-authored questions (harder)
  - "augmented" : machine-augmented questions (easier)

The subset label is stored in the "type" (or "source") field of each row
and surfaced here via a note field for downstream analysis.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from datasets import load_dataset

from src.ingestion.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

HF_DATASET_ID = "lmms-lab/ChartQA"
SPLIT = "test"

# Field names to probe for the subset label (dataset releases vary slightly)
_SUBSET_FIELD_CANDIDATES = ("type", "source")


def _get_subset(row: dict) -> str:
    """Return the subset label ('human' or 'augmented'), or '' if not present."""
    for field in _SUBSET_FIELD_CANDIDATES:
        if field in row and row[field] is not None:
            return str(row[field])
    return ""


class ChartQAAdapter(BaseAdapter):
    dataset_name = "chartqa"
    task_type = "T4"
    metric = "relaxed_accuracy"

    def __init__(self, max_samples: Optional[int] = None, seed: int = 42):
        super().__init__(max_samples=max_samples, seed=seed)

    def iter_samples(self) -> Iterator[dict]:
        logger.info(f"[{self.dataset_name}] Loading '{HF_DATASET_ID}' split='{SPLIT}' from HuggingFace")
        ds = load_dataset(HF_DATASET_ID, split=SPLIT)

        for idx, row in enumerate(ds):
            if self.max_samples is not None and idx >= self.max_samples:
                break

            gt_answer = str(row["answer"])
            subset = _get_subset(row)

            # Surface subset as a note so downstream tools can stratify results
            note = f"subset:{subset}" if subset else ""

            yield {
                "sample_id": f"chartqa_test_{idx:06d}",
                "source_split": SPLIT,
                "task_type": self.task_type,
                "correctness_metric": self.metric,
                "query": row["question"],
                "gt_answer": gt_answer,
                "gt_answer_aliases": [gt_answer],
                "image": row["image"],
                "num_pages": 1,
                "has_table": False,
                "has_chart": True,
                "has_figure": False,
                "has_handwriting": False,
                "doc_type": "chart",
                # note is not a standard BaseAdapter field but iter_samples() is free
                # to include extra keys; run() will ignore unknown keys gracefully.
                "note": note,
            }


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=== ChartQA Adapter: smoke test (5 samples) ===")
    adapter = ChartQAAdapter(max_samples=5)

    for i, sample in enumerate(adapter.iter_samples()):
        display = {k: v for k, v in sample.items() if k != "image"}
        display["image"] = f"<PIL.Image size={sample['image'].size}>"
        print(f"\n--- Sample {i} ---")
        print(json.dumps(display, indent=2, default=str))

    print("\nSmoke test passed.")
