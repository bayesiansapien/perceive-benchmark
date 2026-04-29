"""
DocRouteBench — DocVQA Dataset Adapter

Dataset: lmms-lab/DocVQA  (config="DocVQA")
Task: T4 (Semantic & Compositional QA)
Metric: anls  (Average Normalised Levenshtein Similarity)

DocVQA contains natural-language questions over scanned document images
(forms, letters, tables, invoices, etc.).  Ground-truth answers are provided
as a list of acceptable strings; we store all of them as aliases.

We use the "validation" split because the "test" split has no GT answers.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from datasets import load_dataset

from ..base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

# Keywords in questions that suggest tabular content
TABLE_KEYWORDS = {"table", "column", "row", "rows", "columns", "cell"}

HF_ID = "lmms-lab/DocVQA"
HF_CONFIG = "DocVQA"
HF_SPLIT = "validation"


class DocVQAAdapter(BaseAdapter):
    """Adapter for the DocVQA document visual question-answering dataset."""

    dataset_name = "docvqa"
    task_type = "T4"
    metric = "anls"

    def __init__(self, max_samples: Optional[int] = None, seed: int = 42):
        super().__init__(max_samples=max_samples, seed=seed)

    def _load_hf_dataset(self):
        """Load DocVQA validation split from HuggingFace."""
        logger.info(f"[docvqa] Loading {HF_ID} config={HF_CONFIG} split={HF_SPLIT}")
        ds = load_dataset(HF_ID, name=HF_CONFIG, split=HF_SPLIT, )
        logger.info(f"[docvqa] Loaded {len(ds)} samples")
        return ds

    def _resolve_field(self, row: dict, candidates: list[str]):
        """Return the value of the first candidate key present in row, or None."""
        for key in candidates:
            if key in row and row[key] is not None:
                return row[key]
        return None

    def _has_table_mention(self, question: str) -> bool:
        """Return True if the question text references tabular structure."""
        q_lower = question.lower()
        return any(kw in q_lower for kw in TABLE_KEYWORDS)

    def iter_samples(self) -> Iterator[dict]:
        """
        Yield one sample per DocVQA validation example.

        Expected HuggingFace fields:
          question  — natural-language question string
          image     — PIL Image (or bytes)
          answers   — list of acceptable answer strings
          (questionId / docId may also be present but are not required)

        We handle alternative field names defensively.
        """
        ds = self._load_hf_dataset()

        count = 0
        for idx, row in enumerate(ds):
            # --- Question ---
            question = self._resolve_field(row, ["question", "Question", "query"])
            if not question:
                logger.warning(f"[docvqa] Row {idx}: no question field, skipping")
                continue
            question = str(question).strip()

            # --- Answers ---
            answers_raw = self._resolve_field(row, ["answers", "answer", "Answers", "Answer"])
            if not answers_raw:
                logger.warning(f"[docvqa] Row {idx}: no answers field, skipping")
                continue

            # Normalise to list of strings
            if isinstance(answers_raw, str):
                answers = [answers_raw]
            else:
                answers = [str(a) for a in answers_raw if a is not None]

            if not answers:
                logger.warning(f"[docvqa] Row {idx}: empty answers list, skipping")
                continue

            # --- Image ---
            image = self._resolve_field(row, ["image", "img", "page_image", "document_image"])
            if image is None:
                logger.warning(f"[docvqa] Row {idx}: no image field, skipping")
                continue

            sample_id = f"docvqa_val_{idx:06d}"
            has_table = self._has_table_mention(question)

            yield {
                "sample_id": sample_id,
                "query": question,
                "gt_answer": answers[0],
                "gt_answer_aliases": answers,
                "image": image,
                "task_type": self.task_type,
                "correctness_metric": self.metric,
                "source_split": HF_SPLIT,
                "doc_type": "document",
                "has_table": has_table,
                "has_chart": False,
                "has_figure": False,
                "has_handwriting": False,
                "num_pages": 1,
            }

            count += 1
            if self.max_samples and count >= self.max_samples:
                break


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)
    print("=== DocVQA Adapter Smoke Test (3 samples) ===\n")

    adapter = DocVQAAdapter(max_samples=3)
    for i, sample in enumerate(adapter.iter_samples()):
        print(f"--- Sample {i + 1} ---")
        display = {k: v for k, v in sample.items() if k != "image"}
        print(json.dumps(display, indent=2))
        img = sample["image"]
        print(f"  image type : {type(img).__name__}")
        if hasattr(img, "size"):
            print(f"  image size : {img.size}")
        print()
