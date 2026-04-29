"""
DocRouteBench — SlideVQA Dataset Adapter

Task:   T5 (Multi-Page / Slide Deck Reasoning)
Metric: slidevqa_em (exact match)
Source: nttmdlab-nlp/SlideVQA  (HuggingFace)
Split:  test

Each sample is a question about a slide deck (~20 slides). A 2x2 grid of
the first four slide images is used as the representative document image.
"""

from __future__ import annotations

import logging
import sys
from typing import Iterator, Optional

from PIL import Image

# Allow running from repo root without installing the package
import sys
from pathlib import Path
_SRC = Path(__file__).parent.parent.parent  # src/
_ROOT = _SRC.parent                          # project root
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ingestion.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

SLIDE_W, SLIDE_H = 400, 300   # per-cell size in the composite grid
GRID_COLS, GRID_ROWS = 2, 2   # 2 × 2 layout → 800 × 600 output
MAX_SLIDES = GRID_COLS * GRID_ROWS  # 4


def make_slide_grid(slides: list, max_slides: int = MAX_SLIDES) -> Image.Image:
    """
    Arrange the first *max_slides* slide images in a 2-column grid.

    Parameters
    ----------
    slides:     list of PIL.Image objects (or objects convertible to PIL.Image)
    max_slides: cap on how many slides to include (default 4)

    Returns
    -------
    A single PIL.Image with dimensions (GRID_COLS * SLIDE_W) × (GRID_ROWS * SLIDE_H),
    i.e. 800 × 600 for the default settings.
    """
    selected = slides[: min(len(slides), max_slides)]
    n = len(selected)

    if n == 0:
        # Return a blank placeholder if no slides at all
        return Image.new("RGB", (SLIDE_W, SLIDE_H), color=(200, 200, 200))

    # For a single slide just resize and return — no grid needed
    if n == 1:
        return selected[0].convert("RGB").resize((SLIDE_W, SLIDE_H), Image.LANCZOS)

    cols = min(n, GRID_COLS)
    rows = (n + cols - 1) // cols  # ceil division

    canvas_w = cols * SLIDE_W
    canvas_h = rows * SLIDE_H
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    for i, slide in enumerate(selected):
        cell = slide.convert("RGB").resize((SLIDE_W, SLIDE_H), Image.LANCZOS)
        col = i % cols
        row = i // cols
        canvas.paste(cell, (col * SLIDE_W, row * SLIDE_H))

    return canvas


class SlideVQAAdapter(BaseAdapter):
    """Adapter for the SlideVQA benchmark (slide-deck visual question answering)."""

    dataset_name = "slidevqa"
    task_type    = "T5"
    metric       = "slidevqa_em"

    # Candidate HuggingFace dataset IDs — tried in order
    _HF_IDS = [
        "Ahren09/SlideVQA",
        "NTT-hil-insight/SlideVQA",
        "nttmdlab-nlp/SlideVQA",
    ]

    def __init__(self, max_samples: Optional[int] = None, seed: int = 42):
        super().__init__(max_samples=max_samples, seed=seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_hf_dataset(self):
        """Try each candidate HF ID and return the first that loads."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required. Install it with:\n"
                "  pip install datasets"
            ) from exc

        last_err: Optional[Exception] = None
        for hf_id in self._HF_IDS:
            splits_to_try = ["test", "train", "validation"]
            for split in splits_to_try:
                try:
                    logger.info(f"[slidevqa] Trying to load '{hf_id}' split='{split}' (streaming) …")
                    ds = load_dataset(hf_id, split=split, streaming=True)
                    # Peek at first row to verify it loaded
                    first = next(iter(ds))
                    logger.info(f"[slidevqa] Loaded '{hf_id}' split='{split}' (streaming). Columns: {list(first.keys())}")
                    # Return a chain of the peeked row + the rest
                    import itertools
                    return itertools.chain([first], ds)
                except Exception as exc:
                    logger.warning(f"[slidevqa] Failed to load '{hf_id}' split='{split}': {exc}")
                    last_err = exc

        raise RuntimeError(
            "Could not load SlideVQA from any known HuggingFace ID.\n"
            f"Last error: {last_err}\n"
            "Suggestions:\n"
            "  1. Search https://huggingface.co/datasets?search=slidevqa for the current ID.\n"
            "  2. Download manually and pass a local path.\n"
            "  3. Accept gated-dataset access on the HF website if required."
        )

    @staticmethod
    def _extract_slides(row: dict) -> list:
        """
        Return a list of PIL.Image objects from a row.

        SlideVQA stores slide images under various field names depending on
        the dataset version:
          - Ahren09/SlideVQA: page_1, page_2, ..., page_20 (individual columns)
          - Other versions: "images", "slides", "slide_images", or "image" (list or single)
        """
        # Try page_1..page_20 format (Ahren09/SlideVQA)
        page_slides = []
        for i in range(1, 21):
            val = row.get(f"page_{i}")
            if val is None:
                continue
            if isinstance(val, Image.Image):
                page_slides.append(val)
            elif isinstance(val, dict) and "bytes" in val:
                import io
                page_slides.append(Image.open(io.BytesIO(val["bytes"])))
            elif isinstance(val, bytes):
                import io
                page_slides.append(Image.open(io.BytesIO(val)))
        if page_slides:
            return page_slides

        # Try list-based formats
        for field in ("images", "slides", "slide_images", "image"):
            val = row.get(field)
            if val is None:
                continue
            if isinstance(val, list) and len(val) > 0:
                pil_slides = []
                for item in val:
                    if isinstance(item, Image.Image):
                        pil_slides.append(item)
                    elif isinstance(item, dict) and "bytes" in item:
                        import io
                        pil_slides.append(Image.open(io.BytesIO(item["bytes"])))
                    elif isinstance(item, bytes):
                        import io
                        pil_slides.append(Image.open(io.BytesIO(item)))
                if pil_slides:
                    return pil_slides
            # Single image — wrap in list
            if isinstance(val, Image.Image):
                return [val]
            if isinstance(val, dict) and "bytes" in val:
                import io
                return [Image.open(io.BytesIO(val["bytes"]))]
        return []

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def iter_samples(self) -> Iterator[dict]:
        """Yield one normalized sample dict per SlideVQA test row."""
        ds = self._load_hf_dataset()

        for idx, row in enumerate(ds):
            if self.max_samples is not None and idx >= self.max_samples:
                break

            sample_id = f"slidevqa_test_{idx:06d}"

            # --- query / answer ---
            query = str(row.get("question", row.get("query", "")))
            answer_raw = row.get("answer", row.get("answers", ""))
            # answers can be a list in some versions
            if isinstance(answer_raw, list):
                gt_answer = str(answer_raw[0]) if answer_raw else ""
                gt_answer_aliases = [str(a) for a in answer_raw]
            else:
                gt_answer = str(answer_raw)
                gt_answer_aliases = [gt_answer]

            # --- slides ---
            slides = self._extract_slides(row)

            if not slides:
                logger.warning(
                    f"[slidevqa] {sample_id}: no slide images found — skipping"
                )
                continue

            num_pages = len(slides)

            # Build composite image: 2×2 grid of first 4 slides
            representative_image = make_slide_grid(slides, max_slides=MAX_SLIDES)

            yield {
                "sample_id":          sample_id,
                "query":              query,
                "gt_answer":          gt_answer,
                "gt_answer_aliases":  gt_answer_aliases,
                "image":              representative_image,
                "task_type":          self.task_type,
                "correctness_metric": self.metric,
                "source_split":       "test",
                # document metadata
                "num_pages":          num_pages,
                "doc_type":           "presentation",
                "has_figure":         True,   # slides are inherently visual
                "has_chart":          True,   # slides frequently contain charts
                "has_table":          False,
                "has_handwriting":    False,
            }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print("=== SlideVQA Adapter — smoke test ===")
    adapter = SlideVQAAdapter(max_samples=3)

    samples = []
    try:
        for s in adapter.iter_samples():
            samples.append(s)
            print(f"\n  sample_id : {s['sample_id']}")
            print(f"  query     : {s['query'][:80]}")
            print(f"  gt_answer : {s['gt_answer']}")
            print(f"  num_pages : {s['num_pages']}")
            img = s["image"]
            print(f"  image     : {img.size} mode={img.mode}")
    except Exception as exc:
        print(f"\n[FAIL] iter_samples raised: {exc}")
        sys.exit(1)

    if not samples:
        print("\n[WARN] No samples yielded — dataset may not be accessible.")
        sys.exit(0)

    print(f"\n=== {len(samples)} sample(s) OK — running full pipeline on 2 samples ===")
    adapter2 = SlideVQAAdapter(max_samples=2)
    n = adapter2.run()
    print(f"Written {n} record(s) to {adapter2.output_path}")
    print("=== DONE ===")
