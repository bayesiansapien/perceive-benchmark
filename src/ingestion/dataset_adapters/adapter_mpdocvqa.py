from __future__ import annotations
import json
import ast as _ast
"""
DocRouteBench: MP-DocVQA Dataset Adapter

Dataset : lmms-lab/MP-DocVQA
Split   : validation
Task    : T5 (Multi-Page QA)
Metric  : anls

MP-DocVQA contains multi-page document images paired with extractive
question-answer sets. The image field may be a single representative page
image or a list of page images. When multiple page images are present they
are concatenated vertically into a single PIL image before being saved.
"""


import logging
from typing import Iterator, Optional

from PIL import Image
from datasets import load_dataset

from ..base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

_HF_ID = "lmms-lab/MP-DocVQA"

# Keywords that suggest the question involves a table
_TABLE_KEYWORDS = frozenset(
    [
        "table",
        "row",
        "column",
        "cell",
        "entry",
        "entries",
        "record",
        "records",
        "grid",
        "tabular",
        "total",
        "subtotal",
    ]
)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def concat_pages_vertically(images: list) -> Image.Image:
    """
    Stack a list of PIL images vertically (top to bottom).

    All images are first scaled to the same width (the minimum width found
    across all pages) before being stacked, which preserves aspect ratios and
    avoids artificially widening narrow pages.

    Parameters
    ----------
    images:
        Non-empty list of PIL.Image objects (any mode).

    Returns
    -------
    PIL.Image in RGB mode containing all pages stacked top-to-bottom.
    """
    if not images:
        raise ValueError("concat_pages_vertically received an empty list")

    pil_images: list[Image.Image] = []
    for img in images:
        if not isinstance(img, Image.Image):
            raise TypeError(
                f"Expected PIL.Image, got {type(img).__name__}"
            )
        pil_images.append(img.convert("RGB"))

    if len(pil_images) == 1:
        return pil_images[0]

    target_width = min(img.width for img in pil_images)

    resized: list[Image.Image] = []
    for img in pil_images:
        if img.width != target_width:
            scale = target_width / img.width
            new_height = max(1, int(img.height * scale))
            img = img.resize((target_width, new_height), Image.LANCZOS)
        resized.append(img)

    total_height = sum(img.height for img in resized)
    canvas = Image.new("RGB", (target_width, total_height), color=(255, 255, 255))

    y_offset = 0
    for img in resized:
        canvas.paste(img, (0, y_offset))
        y_offset += img.height

    return canvas


def _extract_image(row: dict) -> tuple[Image.Image, int]:
    """
    Return (composite_image, num_pages) from a dataset row.

    Handles three possible formats for the image field:
    1. A single PIL.Image                    -> use as-is, num_pages=1
    2. A list of PIL.Image objects           -> concat vertically
    3. An evidence-page index is provided    -> use that page only
    """
    raw_image = row.get("image") or row.get("images")

    # MP-DocVQA stores pages as image_1..image_20 fields
    if raw_image is None:
        page_images = [row[f"image_{i}"] for i in range(1, 21) if row.get(f"image_{i}") is not None]
        if page_images:
            raw_image = page_images
        else:
            raise ValueError("Row contains neither 'image' nor 'images' nor 'image_N' fields")

    if isinstance(raw_image, list):
        # Possibly a list of page images
        pil_list = [img for img in raw_image if isinstance(img, Image.Image)]
        if not pil_list:
            raise ValueError("'image' list contains no PIL.Image objects")
        num_pages = len(pil_list)

        # If an evidence page index is provided, prefer that single page
        evidence_page = row.get("evidence_page_no") or row.get("page_idx")
        if evidence_page is not None:
            try:
                ep = int(evidence_page)
                if 0 <= ep < len(pil_list):
                    return pil_list[ep].convert("RGB"), num_pages
            except (TypeError, ValueError):
                pass  # fall through to concatenation

        return concat_pages_vertically(pil_list), num_pages

    # Single image
    if isinstance(raw_image, Image.Image):
        return raw_image.convert("RGB"), row.get("num_pages", 1) or 1

    raise TypeError(f"Unexpected image type: {type(raw_image).__name__}")


def _has_table_question(question: str) -> bool:
    """Return True if the question text contains table-related keywords."""
    lower = question.lower()
    return any(kw in lower for kw in _TABLE_KEYWORDS)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class MPDocVQAAdapter(BaseAdapter):
    dataset_name = "mpdocvqa"
    task_type = "T5"
    metric = "anls"

    def __init__(self, max_samples: Optional[int] = None, seed: int = 42):
        super().__init__(max_samples=max_samples, seed=seed)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def iter_samples(self) -> Iterator[dict]:
        """
        Yield one unified sample dict per row.

        Expected row fields
        -------------------
        question    : str
        answers     : list[str]
        image       : PIL.Image | list[PIL.Image]
        num_pages   : int  (optional)
        evidence_page_no / page_idx : int (optional, 0-indexed evidence page)
        """
        logger.info(f"[{self.dataset_name}] Loading {_HF_ID} split=validation")
        ds = load_dataset(_HF_ID, split="val")
        logger.info(f"[{self.dataset_name}] Loaded {len(ds)} samples")

        for idx, row in enumerate(ds):
            if self.max_samples is not None and idx >= self.max_samples:
                break

            question: str = row["question"]
            raw_answers = row["answers"]
            answers: list[str] = _ast.literal_eval(raw_answers) if isinstance(raw_answers, str) else raw_answers
            gt_answer: str = answers[0] if answers else ""

            try:
                composite_image, num_pages = _extract_image(row)
            except Exception as exc:
                logger.warning(
                    f"[{self.dataset_name}] Skipping idx={idx}, image error: {exc}"
                )
                continue

            # num_pages from the row takes priority; fall back to page count
            num_pages_final = row.get("num_pages") or num_pages or 1

            sample_id = f"mpdocvqa_val_{idx:06d}"

            yield {
                "sample_id": sample_id,
                "source_split": "validation",
                "task_type": self.task_type,
                "correctness_metric": self.metric,
                # Query / answer
                "query": question,
                "gt_answer": gt_answer,
                "gt_answer_aliases": answers,
                # Image (single composite)
                "image": composite_image,
                # Document metadata
                "doc_type": "document",
                "num_pages": num_pages_final,
                "has_table": _has_table_question(question),
                "has_chart": False,
                "has_figure": False,
                "has_handwriting": False,
            }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=== MP-DocVQA smoke test (max_samples=3) ===")
    adapter = MPDocVQAAdapter(max_samples=3)
    for i, sample in enumerate(adapter.iter_samples()):
        img = sample["image"]
        print(
            f"[{i}] id={sample['sample_id']} | "
            f"num_pages={sample['num_pages']} | "
            f"has_table={sample['has_table']} | "
            f"image_size={img.size} | "
            f"query={sample['query'][:60]!r} | "
            f"gt_answer={sample['gt_answer'][:60]!r}"
        )
    print("Smoke test complete: calling adapter.run() to write JSONL")
    adapter2 = MPDocVQAAdapter(max_samples=3)
    n = adapter2.run()
    print(f"Written {n} samples.")
