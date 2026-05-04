"""
DocRouteBench: PubLayNet Dataset Adapter

Dataset : JamieSJS/publaynet-samples  (primary HF ID)
          ds4sd/DocLayNet             (fallback HF ID)
          Synthetic fallback          (if both HF loads fail)
Split   : validation  (train used as proxy when validation absent)
Tasks   : T3, Layout & Spatial Reasoning
          T6, Visual Grounding
Metrics : exact_match (T3)  |  iou (T6)

PubLayNet is a document-layout detection benchmark.  Each image carries
bounding-box annotations for five element types:
  text, title, list, figure, table

Conversion strategy (1-2 QA pairs per image):
  T3 , pick an element whose bbox centroid falls in a cardinal region
        (top / bottom / left / right); ask what element type is there.
  T6 , pick a random element; ask the model to locate it and return
        its bbox normalised to [0, 1].

Bbox format in the source: [x1, y1, width, height] (COCO style).
Normalised output format:  "[x1, y1, x2, y2]" (fractional, 0-1).
"""

from __future__ import annotations

import logging
import random
from typing import Iterator, Optional

from PIL import Image, ImageDraw

from src.ingestion.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# Category id → human-readable name (PubLayNet / COCO-style numbering)
_LABEL_MAP: dict[int, str] = {
    1: "text",
    2: "title",
    3: "list",
    4: "table",
    5: "figure",
}

# DocLayNet uses string category names directly
_DOCLAYNET_LABEL_MAP: dict[str, str] = {
    "Text": "text",
    "Section-header": "title",
    "Title": "title",
    "List-item": "list",
    "Table": "table",
    "Figure": "figure",
    "Caption": "text",
    "Footnote": "text",
    "Formula": "text",
    "Page-header": "title",
    "Page-footer": "text",
    "Picture": "figure",
    "Checkbox-selected": "text",
    "Checkbox-unselected": "text",
    "Code": "text",
    "Form": "text",
    "Key-value region": "text",
}

_CARDINAL_REGIONS = ("top", "bottom", "left", "right")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _coco_to_xyxy_norm(bbox_coco: list[float], img_w: int, img_h: int) -> list[float]:
    """Convert COCO [x1, y1, w, h] to normalised [x1, y1, x2, y2]."""
    x1, y1, bw, bh = bbox_coco
    x2, y2 = x1 + bw, y1 + bh
    return [
        round(x1 / img_w, 4),
        round(y1 / img_h, 4),
        round(x2 / img_w, 4),
        round(y2 / img_h, 4),
    ]


def _xyxy_to_norm(bbox_xyxy: list[float], img_w: int, img_h: int) -> list[float]:
    """Normalise an absolute [x1, y1, x2, y2] bbox."""
    x1, y1, x2, y2 = bbox_xyxy
    return [
        round(x1 / img_w, 4),
        round(y1 / img_h, 4),
        round(x2 / img_w, 4),
        round(y2 / img_h, 4),
    ]


def _centroid(norm_bbox: list[float]) -> tuple[float, float]:
    cx = (norm_bbox[0] + norm_bbox[2]) / 2.0
    cy = (norm_bbox[1] + norm_bbox[3]) / 2.0
    return cx, cy


def _cardinal_region(cx: float, cy: float) -> str:
    """Return the dominant cardinal region for a centroid (cx, cy) in [0,1]."""
    if cy < 0.33:
        return "top"
    if cy > 0.67:
        return "bottom"
    if cx < 0.4:
        return "left"
    return "right"


# --------------------------------------------------------------------------
# Synthetic fallback generator
# --------------------------------------------------------------------------

def _make_synthetic_samples(
    n_images: int = 50,
    rng: random.Random | None = None,
) -> list[dict]:
    """
    Generate minimal synthetic PubLayNet-like records when HF loading fails.
    Each record mimics the processed internal format used by iter_samples.
    """
    if rng is None:
        rng = random.Random(42)

    element_types = list(_LABEL_MAP.values())
    samples = []
    for i in range(n_images):
        W, H = 800, 1100
        img = Image.new("RGB", (W, H), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        annotations = []
        for _ in range(rng.randint(3, 7)):
            etype = rng.choice(element_types)
            x1 = rng.randint(20, W // 2)
            y1 = rng.randint(20, H // 2)
            x2 = rng.randint(x1 + 40, min(x1 + 300, W - 20))
            y2 = rng.randint(y1 + 20, min(y1 + 200, H - 20))
            draw.rectangle([x1, y1, x2, y2], outline=(180, 180, 180), width=1)
            annotations.append({
                "label": etype,
                "norm_bbox": _xyxy_to_norm([x1, y1, x2, y2], W, H),
            })

        samples.append({"image": img, "annotations": annotations, "img_w": W, "img_h": H})

    logger.info(f"[publaynet] Generated {len(samples)} synthetic fallback samples")
    return samples


# --------------------------------------------------------------------------
# Main adapter
# --------------------------------------------------------------------------

class PubLayNetAdapter(BaseAdapter):
    """
    Adapter for PubLayNet (layout detection → QA).

    Yields 1-2 samples per image:
      - One T3 (layout region classification) sample
      - One T6 (visual grounding / bbox prediction) sample  [when possible]
    """

    dataset_name = "publaynet"
    # task_type and metric are sample-level; set class defaults for T3
    task_type = "T3"
    metric = "exact_match"

    _PRIMARY_HF_ID = "jordanparker6/publaynet"
    _FALLBACK_HF_ID = "JamieSJS/publaynet-samples"

    def __init__(self, max_samples: Optional[int] = None, seed: int = 42):
        super().__init__(max_samples=max_samples, seed=seed)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # HF loading helpers
    # ------------------------------------------------------------------

    def _load_hf_primary(self):
        from datasets import load_dataset
        logger.info(f"[publaynet] Trying primary HF dataset: {self._PRIMARY_HF_ID} …")
        ds = load_dataset(self._PRIMARY_HF_ID, split="validation", streaming=True)
        return ds, "primary"

    def _load_hf_fallback(self):
        from datasets import load_dataset
        logger.info(f"[publaynet] Trying fallback HF dataset: {self._FALLBACK_HF_ID} …")
        ds = load_dataset(self._FALLBACK_HF_ID, split="validation", streaming=True)
        return ds, "doclaynet"

    # ------------------------------------------------------------------
    # Annotation normalisation
    # ------------------------------------------------------------------

    def _parse_primary(self, sample: dict) -> dict | None:
        """
        Parse a JamieSJS/publaynet-samples record into internal format:
          { image, annotations: [{label, norm_bbox}], img_w, img_h }

        The exact schema depends on the HF dataset card; we handle the most
        common variants defensively.
        """
        image: Image.Image = sample.get("image")
        if image is None:
            return None
        img_w, img_h = image.size

        annotations = []

        # Variant A: 'objects' column with COCO-style bboxes
        objects = sample.get("objects") or sample.get("annotations") or []
        for obj in objects:
            # category may be int id or string name
            cat = obj.get("category_id") or obj.get("label") or obj.get("category")
            if isinstance(cat, int):
                label = _LABEL_MAP.get(cat)
            elif isinstance(cat, str):
                label = cat.lower()
            else:
                continue

            if not label:
                continue

            bbox = obj.get("bbox") or obj.get("bounding_box")
            if bbox is None:
                continue

            # COCO [x,y,w,h] vs xyxy, detect by checking if x2>x1 makes sense
            if len(bbox) == 4:
                x1, y1, v3, v4 = bbox
                # heuristic: if v3 < img_w and v4 < img_h and v3 > x1, likely xyxy
                if v3 > x1 and v4 > y1:
                    norm = _xyxy_to_norm([x1, y1, v3, v4], img_w, img_h)
                else:
                    norm = _coco_to_xyxy_norm([x1, y1, v3, v4], img_w, img_h)
            else:
                continue

            annotations.append({"label": label, "norm_bbox": norm})

        if not annotations:
            return None
        return {"image": image, "annotations": annotations, "img_w": img_w, "img_h": img_h}

    def _parse_doclaynet(self, sample: dict) -> dict | None:
        """Parse a ds4sd/DocLayNet record."""
        image: Image.Image = sample.get("image")
        if image is None:
            return None
        img_w, img_h = image.size

        annotations = []
        bboxes = sample.get("bboxes_block") or sample.get("bboxes") or []
        labels = sample.get("labels") or []

        for bbox, label_raw in zip(bboxes, labels):
            if isinstance(label_raw, int):
                # DocLayNet sometimes ships category ids
                label = _LABEL_MAP.get(label_raw)
            elif isinstance(label_raw, str):
                label = _DOCLAYNET_LABEL_MAP.get(label_raw, label_raw.lower())
            else:
                continue

            if not label or not bbox or len(bbox) != 4:
                continue

            # DocLayNet bboxes are normalised [x1,y1,x2,y2] in [0,1000] range
            # or already in pixel coords, normalise defensively
            x1, y1, x2, y2 = bbox
            if max(x1, y1, x2, y2) > 1:
                # Pixel coordinates
                norm = _xyxy_to_norm([x1, y1, x2, y2], img_w, img_h)
            else:
                norm = [round(v, 4) for v in [x1, y1, x2, y2]]

            annotations.append({"label": label, "norm_bbox": norm})

        if not annotations:
            return None
        return {"image": image, "annotations": annotations, "img_w": img_w, "img_h": img_h}

    # ------------------------------------------------------------------
    # QA pair generation
    # ------------------------------------------------------------------

    def _make_t3_sample(
        self,
        parsed: dict,
        sample_id: str,
        has_figure: bool,
        has_table: bool,
    ) -> dict | None:
        """Build a T3 layout-classification QA sample."""
        ann = parsed["annotations"]
        # Try to find an annotation whose centroid falls in a cardinal region
        self._rng.shuffle(ann)
        chosen = None
        for a in ann:
            region = _cardinal_region(*_centroid(a["norm_bbox"]))
            chosen = (a, region)
            break

        if chosen is None:
            return None

        element, region = chosen
        query = (
            f"What type of layout element appears at the {region} of this page? "
            "Choose from: text, title, list, table, figure."
        )
        return {
            "sample_id": sample_id,
            "query": query,
            "gt_answer": element["label"],
            "gt_answer_aliases": [],
            "image": parsed["image"],
            "task_type": "T3",
            "correctness_metric": "exact_match",
            "source_split": "validation",
            "doc_type": "academic_paper",
            "num_pages": 1,
            "has_table": has_table,
            "has_chart": False,
            "has_figure": has_figure,
            "has_handwriting": False,
        }

    def _make_t6_sample(
        self,
        parsed: dict,
        sample_id: str,
        has_figure: bool,
        has_table: bool,
    ) -> dict | None:
        """Build a T6 visual-grounding QA sample."""
        ann = parsed["annotations"]
        if not ann:
            return None

        element = self._rng.choice(ann)
        label = element["label"]
        nb = element["norm_bbox"]
        gt_str = f"[{nb[0]}, {nb[1]}, {nb[2]}, {nb[3]}]"

        query = (
            f"Locate the {label} on this page and provide its bounding box "
            f"coordinates as [x1, y1, x2, y2] normalized to [0, 1]."
        )
        return {
            "sample_id": sample_id,
            "query": query,
            "gt_answer": gt_str,
            "gt_answer_aliases": [],
            "image": parsed["image"],
            "task_type": "T6",
            "correctness_metric": "iou",
            "source_split": "validation",
            "doc_type": "academic_paper",
            "num_pages": 1,
            "has_table": has_table,
            "has_chart": False,
            "has_figure": has_figure,
            "has_handwriting": False,
        }

    # ------------------------------------------------------------------
    # iter_samples
    # ------------------------------------------------------------------

    def iter_samples(self) -> Iterator[dict]:
        """
        Load PubLayNet (or fallback) and yield 1-2 QA samples per image:
          qa_idx=0  →  T3 layout classification
          qa_idx=1  →  T6 visual grounding
        """
        # --- attempt HF loading ---
        raw_ds = None
        parse_fn = None

        try:
            raw_ds, source = self._load_hf_primary()
            parse_fn = self._parse_primary
        except Exception as e1:
            logger.warning(f"[publaynet] Primary HF load failed: {e1}")
            try:
                raw_ds, source = self._load_hf_fallback()
                parse_fn = self._parse_doclaynet
            except Exception as e2:
                logger.warning(f"[publaynet] Fallback HF load failed: {e2}")
                logger.warning(
                    "[publaynet] Both HF datasets unavailable: using synthetic fallback. "
                    "Install datasets and ensure network access for real data."
                )
                source = "synthetic"

        # ----------------------------------------------------------------
        # Iterate
        # ----------------------------------------------------------------
        qa_count = 0
        img_idx = 0

        if source == "synthetic":
            records = _make_synthetic_samples(n_images=200, rng=self._rng)
            iterable = records
            use_synthetic = True
        else:
            iterable = raw_ds
            use_synthetic = False

        for img_idx, raw_sample in enumerate(iterable):
            if self.max_samples and qa_count >= self.max_samples:
                break

            # Parse into internal format
            if use_synthetic:
                parsed = raw_sample  # already in internal format
            else:
                parsed = parse_fn(raw_sample)  # type: ignore[operator]
                if parsed is None:
                    continue

            ann_labels = {a["label"] for a in parsed["annotations"]}
            has_figure = "figure" in ann_labels
            has_table = "table" in ann_labels

            # T3 sample (qa_idx=0)
            if not self.max_samples or qa_count < self.max_samples:
                t3 = self._make_t3_sample(
                    parsed,
                    sample_id=f"publaynet_val_{img_idx:06d}_0",
                    has_figure=has_figure,
                    has_table=has_table,
                )
                if t3 is not None:
                    yield t3
                    qa_count += 1

            # T6 sample (qa_idx=1)
            if not self.max_samples or qa_count < self.max_samples:
                t6 = self._make_t6_sample(
                    parsed,
                    sample_id=f"publaynet_val_{img_idx:06d}_1",
                    has_figure=has_figure,
                    has_table=has_table,
                )
                if t6 is not None:
                    yield t6
                    qa_count += 1

        logger.info(
            f"[publaynet] iter_samples complete, {qa_count} QA pairs from {img_idx + 1} images"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    adapter = PubLayNetAdapter(max_samples=4)
    for s in adapter.iter_samples():
        print(
            f"  id={s['sample_id']} | task={s['task_type']}"
            f" | metric={s['correctness_metric']}"
            f" | q='{s['query'][:60]}'"
            f" | a='{s['gt_answer']}'"
        )
    print("Smoke test passed.")
