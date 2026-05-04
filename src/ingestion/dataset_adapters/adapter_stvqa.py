"""
DocRouteBench: ST-VQA Dataset Adapter

Task: T2 (OCR-grounded QA)
HF ID: vikhyatk/st-vqa
Split: "train" (val/test GT not publicly available).
Metric: anls (Average Normalized Levenshtein Similarity)

vikhyatk/st-vqa schema:
  keys: ['image', 'qas']
  qas: [{'question': '...', 'answers': ['...', ...]}, ...]

Each row may contain multiple QA pairs. iter_samples() expands them into one
sample per QA pair.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from src.ingestion.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

_HF_ID = "vikhyatk/st-vqa"
_SPLIT = "train"


class STVQAAdapter(BaseAdapter):
    dataset_name = "stvqa"
    task_type = "T2"
    metric = "anls"

    def __init__(self, max_samples: Optional[int] = None, seed: int = 42):
        super().__init__(max_samples=max_samples, seed=seed)
        self._split_used: str = _SPLIT

    # ------------------------------------------------------------------
    # Required abstract method
    # ------------------------------------------------------------------

    def iter_samples(self) -> Iterator[dict]:
        from datasets import load_dataset

        logger.info(f"[stvqa] Loading '{_HF_ID}' split='{_SPLIT}' (streaming) ...")
        ds = load_dataset(_HF_ID, split=_SPLIT, streaming=True)

        sample_count = 0
        for row_idx, row in enumerate(ds):
            if self.max_samples is not None and sample_count >= self.max_samples:
                break

            image = row.get("image")
            if image is None:
                logger.warning(f"[stvqa] Row {row_idx} has no image field, skipping")
                continue

            qas = row.get("qas", [])
            if not qas:
                logger.warning(f"[stvqa] Row {row_idx} has no qas, skipping")
                continue

            for qa_idx, qa in enumerate(qas):
                if self.max_samples is not None and sample_count >= self.max_samples:
                    break

                question = qa.get("question", "")
                answers = qa.get("answers", [])

                if not question or not answers:
                    continue

                gt_answer = answers[0] if answers else ""

                yield {
                    "sample_id": f"stvqa_train_{row_idx:06d}_{qa_idx:04d}",
                    "query": question,
                    "gt_answer": gt_answer,
                    "gt_answer_aliases": answers,
                    "image": image,
                    "task_type": self.task_type,
                    "correctness_metric": self.metric,
                    "source_split": _SPLIT,
                    "doc_type": "scene_text",
                    "num_pages": 1,
                    "has_table": False,
                    "has_chart": False,
                    "has_figure": False,
                    "has_handwriting": False,
                }
                sample_count += 1


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    max_s = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    adapter = STVQAAdapter(max_samples=max_s)

    print(f"Running STVQAAdapter smoke test (max_samples={max_s}) ...")
    samples = list(adapter.iter_samples())
    print(f"  iter_samples() yielded {len(samples)} sample(s)")
    if samples:
        s = samples[0]
        print(f"  sample_id        : {s['sample_id']}")
        print(f"  query            : {s['query']}")
        print(f"  gt_answer        : {s['gt_answer']}")
        print(f"  gt_answer_aliases: {s['gt_answer_aliases']}")
        print(f"  doc_type         : {s['doc_type']}")
        print(f"  image type       : {type(s['image'])}")
        print(f"  split used       : {adapter._split_used}")

    written = adapter.run()
    print(f"  adapter.run() wrote {written} record(s) to {adapter.output_path}")
    print("Smoke test PASSED.")
