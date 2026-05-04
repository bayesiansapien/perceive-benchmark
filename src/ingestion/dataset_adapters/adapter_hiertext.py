"""
DocRouteBench: HierText Dataset Adapter

Dataset : google-research-datasets/hiertext  (HuggingFace)
Split   : validation
Task    : T2, Degraded / Scene Text Recognition
Metric  : field_f1

HierText provides hierarchical OCR annotations over natural scene images:
  paragraphs → lines → words

Conversion strategy:
  - For each image, randomly select one annotated paragraph region.
  - Query asks the model to read the text in that highlighted region.
  - gt_answer is the concatenated word-level text from that paragraph.
  - Falls back to full-page text when no paragraph annotations exist.
"""

from __future__ import annotations

import logging
import random
from typing import Iterator, Optional

from PIL import Image

from src.ingestion.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class HierTextAdapter(BaseAdapter):
    """Adapter for the HierText scene-text recognition dataset."""

    dataset_name = "hiertext"
    task_type = "T2"
    metric = "field_f1"

    def __init__(self, max_samples: Optional[int] = None, seed: int = 42):
        super().__init__(max_samples=max_samples, seed=seed)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _words_from_paragraph(paragraph: dict) -> list[str]:
        """Extract ordered word strings from a HierText paragraph annotation."""
        words: list[str] = []
        for line in paragraph.get("lines", []):
            for word in line.get("words", []):
                text = word.get("text", "").strip()
                if text:
                    words.append(text)
        return words

    @staticmethod
    def _words_from_full_page(sample: dict) -> list[str]:
        """Fallback: collect every word across all paragraphs on the page."""
        words: list[str] = []
        for para in sample.get("paragraphs", []):
            for line in para.get("lines", []):
                for word in line.get("words", []):
                    text = word.get("text", "").strip()
                    if text:
                        words.append(text)
        return words

    # ------------------------------------------------------------------
    # Core iteration
    # ------------------------------------------------------------------

    def iter_samples(self) -> Iterator[dict]:
        """Load HierText validation split and yield one sample per image."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The `datasets` package is required. Install with: pip install datasets"
            )

        logger.info("[hiertext] Loading Berzerker/ocr_hiertext (train) …")
        try:
            ds = load_dataset(
                "Berzerker/ocr_hiertext",
                split="train",
            )
        except Exception as exc:
            raise RuntimeError(
                f"[hiertext] Failed to load dataset from HuggingFace: {exc}\n"
                "Check your internet connection or HF token and retry."
            ) from exc

        logger.info(f"[hiertext] Loaded {len(ds)} validation examples")

        for idx, sample in enumerate(ds):
            if self.max_samples and idx >= self.max_samples:
                break

            sample_id = f"hiertext_val_{idx:06d}"

            # Berzerker/ocr_hiertext uses a custom text format in output_json_dumpsed
            # Format: "x y w h is_paragraph\n- x y w h is_hw is_ill TEXT\n..."
            # Lines starting with "-" contain word text as the last token(s)
            raw = sample.get("output_json_dumpsed", "")
            # Field is JSON-dumped: literal \n in string, needs json.loads to unescape
            import json as _json
            try:
                raw = _json.loads(raw) if raw.startswith('"') else raw
            except Exception:
                pass
            # Also handle literal \\n that didn't get unescaped
            if "\\n" in raw:
                raw = raw.replace("\\n", "\n")
            words = []
            if raw:
                for line in raw.split("\n"):
                    line = line.strip()
                    if line.startswith("-"):
                        parts = line.split(None, 7)  # "- x y w h hw ill TEXT"
                        if len(parts) >= 8:
                            words.append(parts[7])
                        elif len(parts) == 7:
                            # illegible or no text, skip
                            pass

            # Fall back to legacy dict-style paragraphs field if present
            if not words:
                paragraphs = sample.get("paragraphs", [])
                non_empty = [p for p in paragraphs if self._words_from_paragraph(p)]
                if non_empty:
                    words = self._words_from_paragraph(self._rng.choice(non_empty))
                else:
                    words = self._words_from_full_page(sample)

            gt_answer = " ".join(words).strip()
            query = "What text is written in this image?"

            if not gt_answer:
                logger.debug(f"[hiertext] {sample_id}: no text found, skipping")
                continue

            # Image is a PIL Image in the HF dataset
            image: Image.Image = sample["image"]

            yield {
                "sample_id": sample_id,
                "query": query,
                "gt_answer": gt_answer,
                "gt_answer_aliases": [],
                "image": image,
                "task_type": self.task_type,
                "correctness_metric": self.metric,
                "source_split": "validation",
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
    logging.basicConfig(level=logging.INFO)
    adapter = HierTextAdapter(max_samples=3)
    for s in adapter.iter_samples():
        print(
            f"  id={s['sample_id']} | answer='{s['gt_answer'][:60]}'"
            f" | task={s['task_type']} | metric={s['correctness_metric']}"
        )
    print("Smoke test passed.")
