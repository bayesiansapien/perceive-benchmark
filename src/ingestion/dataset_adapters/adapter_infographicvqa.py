"""
DocRouteBench — InfographicVQA Dataset Adapter

Task type : T4 (Visual Compositional QA)
Metric    : anls

HF IDs tried in order:
  1. lmms-lab/InfographicVQA
  2. docvqa/InfographicVQA
  3. ayoubkirouane/infographic-VQA

For each ID, splits are tried in order: validation, test, train.
A source is accepted only if its rows contain a 'question' field.

If no source loads successfully a RuntimeError is raised with instructions
for obtaining the dataset manually from https://www.docvqa.org/datasets/infographicvqa
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from src.ingestion.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

_HF_IDS = [
    "Ryoo72/InfographicsVQA",           # test split, keys: questionId/question/answers/image
    "ayoubkirouane/infographic-VQA",    # train split, fallback
    "lmms-lab/InfographicVQA",          # may become available later
    "docvqa/InfographicVQA",
]
_PREFERRED_SPLITS = ["test", "validation", "train"]


class InfographicVQAAdapter(BaseAdapter):
    dataset_name = "infographicvqa"
    task_type = "T4"
    metric = "anls"

    def __init__(self, max_samples: Optional[int] = None, seed: int = 42):
        super().__init__(max_samples=max_samples, seed=seed)
        self._source_used: str = ""
        self._split_used: str = ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_hf_dataset(self):
        """
        Try each (hf_id, split) combination in order.
        Returns (dataset, hf_id, split_name).
        Raises RuntimeError if nothing works.
        """
        from datasets import load_dataset

        errors: list[str] = []

        for hf_id in _HF_IDS:
            for split in _PREFERRED_SPLITS:
                try:
                    logger.info(
                        f"[infovqa] Trying '{hf_id}' split='{split}' ..."
                    )
                    ds = load_dataset(hf_id, split=split, streaming=True)

                    # Peek at first row to confirm QA structure
                    first_row = next(iter(ds))
                    if "question" not in first_row:
                        msg = (
                            f"'{hf_id}' split='{split}' has no 'question' field "
                            f"(keys: {list(first_row.keys())})"
                        )
                        logger.warning(f"[infovqa] {msg}")
                        errors.append(msg)
                        continue

                    logger.info(
                        f"[infovqa] Accepted '{hf_id}' split='{split}'"
                    )
                    return ds, hf_id, split

                except Exception as exc:
                    msg = f"'{hf_id}' split='{split}': {exc}"
                    logger.warning(f"[infovqa] Failed to load {msg}")
                    errors.append(msg)

        error_detail = "\n  ".join(errors)
        raise RuntimeError(
            f"[infovqa] Could not load InfographicVQA from any known source.\n"
            f"Tried:\n  {error_detail}\n\n"
            f"To fix: download InfographicVQA from "
            f"https://www.docvqa.org/datasets/infographicvqa "
            f"and place it where a supported HF dataset ID can access it, "
            f"or update _HF_IDS in adapter_infographicvqa.py with the correct ID."
        )

    # ------------------------------------------------------------------
    # Required abstract method
    # ------------------------------------------------------------------

    def iter_samples(self) -> Iterator[dict]:
        ds, hf_id, split = self._load_hf_dataset()
        self._source_used = hf_id
        self._split_used = split

        for idx, row in enumerate(ds):
            if self.max_samples is not None and idx >= self.max_samples:
                break

            # Normalise answers — some versions use 'answers' (list), others 'answer' (str)
            if isinstance(row.get("answers"), list):
                answers: list[str] = row["answers"]
                gt_answer = answers[0] if answers else ""
            else:
                gt_answer = str(row.get("answer", ""))
                answers = [gt_answer]

            yield {
                "sample_id": f"infovqa_{split}_{idx:06d}",
                "source_split": split,
                "task_type": self.task_type,
                "correctness_metric": self.metric,
                "query": row["question"],
                "gt_answer": gt_answer,
                "gt_answer_aliases": answers,
                "image": row["image"],
                "num_pages": 1,
                "has_table": False,
                "has_chart": True,
                "has_figure": True,
                "has_handwriting": False,
                "doc_type": "infographic",
            }


if __name__ == "__main__":
    import json
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=== InfographicVQA Adapter — smoke test (5 samples) ===")
    adapter = InfographicVQAAdapter(max_samples=5)

    for i, sample in enumerate(adapter.iter_samples()):
        display = {k: v for k, v in sample.items() if k != "image"}
        display["image"] = f"<PIL.Image size={sample['image'].size}>"
        print(f"\n--- Sample {i} ---")
        print(json.dumps(display, indent=2, default=str))

    print(f"\nSource: {adapter._source_used}  split: {adapter._split_used}")
    print("Smoke test passed.")
