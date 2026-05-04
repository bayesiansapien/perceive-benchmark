"""
DocRouteBench: TextVQA Dataset Adapter

Task: T2 (OCR-grounded QA)
HF ID: lmms-lab/textvqa  (fallback: textvqa)
Split: validation
Metric: vqa_accuracy

VQA accuracy requires all 10 annotator answers. They are stored in
gt_answer_aliases so the scorer can apply the standard VQA formula:
  min(count(answer) / 3, 1)  averaged over the 10 answer slots.
gt_answer holds the mode (most common) answer as a human-readable label.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Iterator, Optional

from src.ingestion.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

_HF_IDS = ["lmms-lab/textvqa", "textvqa"]
_SPLIT = "validation"


class TextVQAAdapter(BaseAdapter):
    dataset_name = "textvqa"
    task_type = "T2"
    metric = "vqa_accuracy"

    def __init__(self, max_samples: Optional[int] = None, seed: int = 42):
        super().__init__(max_samples=max_samples, seed=seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_hf_dataset(self):
        """Try each HF ID in order; return the first that succeeds."""
        from datasets import load_dataset

        for hf_id in _HF_IDS:
            try:
                logger.info(f"[textvqa] Trying to load '{hf_id}' split='{_SPLIT}' ...")
                ds = load_dataset(hf_id, split=_SPLIT, )
                logger.info(f"[textvqa] Loaded {len(ds)} rows from '{hf_id}'")
                return ds
            except Exception as exc:
                logger.warning(f"[textvqa] Failed to load '{hf_id}': {exc}")

        raise RuntimeError(
            f"[textvqa] Could not load dataset from any of: {_HF_IDS}"
        )

    # ------------------------------------------------------------------
    # Required abstract method
    # ------------------------------------------------------------------

    def iter_samples(self) -> Iterator[dict]:
        ds = self._load_hf_dataset()

        for idx, row in enumerate(ds):
            if self.max_samples and idx >= self.max_samples:
                break

            # ---- answers ----
            # HF TextVQA stores answers as a list of 10 strings under
            # the key "answers". Fall back gracefully if the schema differs.
            answers: list[str] = []
            if "answers" in row and isinstance(row["answers"], list):
                answers = [str(a) for a in row["answers"]]
            elif "answer" in row:
                # Some versions expose a single answer string
                answers = [str(row["answer"])]

            if not answers:
                logger.warning(f"[textvqa] Row {idx} has no answers, skipping")
                continue

            # Mode answer as the canonical gt_answer
            mode_answer = Counter(answers).most_common(1)[0][0]

            yield {
                "sample_id": f"textvqa_val_{idx:06d}",
                "query": str(row["question"]),
                "gt_answer": mode_answer,
                "gt_answer_aliases": answers,   # all 10 for VQA accuracy scoring
                "image": row["image"],           # PIL Image from HF
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


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    max_s = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    adapter = TextVQAAdapter(max_samples=max_s)

    print(f"Running TextVQAAdapter smoke test (max_samples={max_s}) ...")
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

    written = adapter.run()
    print(f"  adapter.run() wrote {written} record(s) to {adapter.output_path}")
    print("Smoke test PASSED.")
