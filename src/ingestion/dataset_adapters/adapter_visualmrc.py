"""
DocRouteBench: VisualMRC Dataset Adapter

Dataset : jeepliu/VisualMRC
Split   : test
Task    : T4 (Semantic Comprehension)
Metric  : rouge_cider

VisualMRC contains webpage screenshot images paired with abstractive
question-answer pairs. Each sample has a single image, a question, and a
free-form answer string.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from datasets import load_dataset

from ..base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

# Try the canonical HF repo ID first; fall back to a known mirror if needed.
_HF_IDS = [
    "jeepliu/VisualMRC",
    "jeepliu/VisualMRC",
]


class VisualMRCAdapter(BaseAdapter):
    dataset_name = "visualmrc"
    task_type = "T4"
    metric = "rouge_cider"

    def __init__(self, max_samples: Optional[int] = None, seed: int = 42):
        super().__init__(max_samples=max_samples, seed=seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_hf_dataset(self):
        """Attempt each known HF repo ID until one succeeds."""
        last_exc: Exception | None = None
        for hf_id in _HF_IDS:
            try:
                logger.info(f"[{self.dataset_name}] Trying HF dataset ID: {hf_id}")
                ds = load_dataset(hf_id, split="test", )
                logger.info(
                    f"[{self.dataset_name}] Loaded {len(ds)} samples from '{hf_id}'"
                )
                return ds
            except Exception as exc:
                logger.warning(
                    f"[{self.dataset_name}] Failed to load '{hf_id}': {exc}"
                )
                last_exc = exc
        raise RuntimeError(
            f"[{self.dataset_name}] Could not load dataset from any known HF ID. "
            f"Last error: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def iter_samples(self) -> Iterator[dict]:
        """
        Yield one unified sample dict per row.

        Expected row fields
        -------------------
        question : str
        answer   : str
        image    : PIL.Image (webpage screenshot)
        """
        ds = self._load_hf_dataset()

        for idx, row in enumerate(ds):
            if self.max_samples is not None and idx >= self.max_samples:
                break

            question: str = row["question"]
            answer: str = row["answer"]
            image = row["image"]  # PIL.Image from HF datasets

            sample_id = f"visualmrc_test_{idx:06d}"

            yield {
                "sample_id": sample_id,
                "source_split": "test",
                "task_type": self.task_type,
                "correctness_metric": self.metric,
                # Query / answer
                "query": question,
                "gt_answer": answer,
                "gt_answer_aliases": [answer],
                # Image
                "image": image,
                # Document metadata
                "doc_type": "webpage",
                "num_pages": 1,
                "has_figure": True,   # webpages routinely contain mixed media
                "has_table": False,
                "has_chart": False,
                "has_handwriting": False,
            }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=== VisualMRC smoke test (max_samples=3) ===")
    adapter = VisualMRCAdapter(max_samples=3)
    for i, sample in enumerate(adapter.iter_samples()):
        print(
            f"[{i}] id={sample['sample_id']} | "
            f"query={sample['query'][:60]!r} | "
            f"gt_answer={sample['gt_answer'][:60]!r} | "
            f"image={sample['image']}"
        )
    print("Smoke test complete: calling adapter.run() to write JSONL")
    adapter2 = VisualMRCAdapter(max_samples=3)
    n = adapter2.run()
    print(f"Written {n} samples.")
