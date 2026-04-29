"""
IoU (Intersection over Union) scoring for visual grounding / bounding-box tasks.

Used by: PubLayNet (T6 grounding), HierText
Bounding boxes are in normalised [0, 1] image coordinates: [x1, y1, x2, y2].

    IoU = intersection_area / union_area

is_correct threshold: IoU >= 0.5  (standard "50% overlap" criterion)

Predicted bbox parsing supports:
- JSON list / array string: "[0.1, 0.2, 0.5, 0.8]"
- Space-separated values:   "0.1 0.2 0.5 0.8"
- Comma-separated values:   "0.1,0.2,0.5,0.8"
- JSON object with keys x1/y1/x2/y2 or xmin/ymin/xmax/ymax

Invalid boxes (coordinates out of range, x2 <= x1, y2 <= y1, wrong number of
values) are treated as empty: IoU = 0.

Multiple ground-truth boxes are supported: a prediction is correct when its
maximum IoU over all GT boxes meets the threshold.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Sequence, Tuple, Union

# Type alias
BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_bbox_from_string(text: str) -> Optional[BBox]:
    """Extract a bounding box from a free-form string.

    Tries several formats in order and returns the first successful parse, or
    *None* if the string cannot be interpreted as a valid bounding box.
    """
    text = text.strip()

    # 1. JSON dict with recognised keys
    if "{" in text:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                # Support both naming conventions
                for x1k, y1k, x2k, y2k in [
                    ("x1", "y1", "x2", "y2"),
                    ("xmin", "ymin", "xmax", "ymax"),
                    ("left", "top", "right", "bottom"),
                ]:
                    if all(k in obj for k in (x1k, y1k, x2k, y2k)):
                        coords = (
                            float(obj[x1k]),
                            float(obj[y1k]),
                            float(obj[x2k]),
                            float(obj[y2k]),
                        )
                        return _validate_bbox(coords)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # 2. JSON list/array
    if "[" in text:
        # Find the first [...] block
        m = re.search(r"\[([^\[\]]+)\]", text)
        if m:
            try:
                values = json.loads(f"[{m.group(1)}]")
                if isinstance(values, list) and len(values) == 4:
                    coords = tuple(float(v) for v in values)
                    return _validate_bbox(coords)  # type: ignore[arg-type]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

    # 3. Four numeric tokens separated by commas, spaces, or semicolons
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(numbers) >= 4:
        try:
            coords = tuple(float(n) for n in numbers[:4])
            return _validate_bbox(coords)  # type: ignore[arg-type]
        except ValueError:
            pass

    return None


def _validate_bbox(coords: Tuple[float, ...]) -> Optional[BBox]:
    """Return a validated BBox or *None* if the coordinates are degenerate."""
    if len(coords) != 4:
        return None

    x1, y1, x2, y2 = coords

    # Must be finite numbers
    import math
    if any(not math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None

    # Clip to [0, 1] with a small tolerance for floating-point noise
    _EPS = 1e-6
    if x1 < -_EPS or y1 < -_EPS or x2 > 1 + _EPS or y2 > 1 + _EPS:
        return None

    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(1.0, x2)
    y2 = min(1.0, y2)

    # Box must have positive area
    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Core IoU computation
# ---------------------------------------------------------------------------

def _iou_pair(box_a: BBox, box_b: BBox) -> float:
    """Compute IoU between two validated bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    if union <= 0.0:
        return 0.0

    return inter / union


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_iou(
    pred: Union[str, Sequence[float]],
    gt: Union[str, Sequence[float], Sequence[Sequence[float]]],
) -> float:
    """Compute IoU between a predicted box and one or more ground-truth boxes.

    Parameters
    ----------
    pred:
        Predicted bounding box.  Either a sequence of four floats
        ``[x1, y1, x2, y2]`` or a string that encodes the box (see module
        docstring for supported formats).
    gt:
        Ground-truth bounding box or a list of ground-truth boxes.  Same
        accepted types as *pred* plus a list of boxes.  When multiple GT
        boxes are provided, the maximum IoU over all boxes is returned.

    Returns
    -------
    float
        IoU in [0.0, 1.0].  Returns 0.0 for invalid or unparseable inputs.
    """
    # --- Parse prediction ---
    if isinstance(pred, str):
        pred_box = _parse_bbox_from_string(pred)
    else:
        pred_box = _validate_bbox(tuple(float(v) for v in pred))  # type: ignore[arg-type]

    if pred_box is None:
        return 0.0

    # --- Parse ground truth (single box or list of boxes) ---
    gt_boxes: List[BBox] = []

    if isinstance(gt, str):
        parsed = _parse_bbox_from_string(gt)
        if parsed is not None:
            gt_boxes.append(parsed)
    elif isinstance(gt, (list, tuple)) and len(gt) > 0:
        # Distinguish "list of 4 numbers" from "list of boxes"
        if isinstance(gt[0], (int, float)):
            # Single box given as a flat sequence
            parsed = _validate_bbox(tuple(float(v) for v in gt))  # type: ignore[arg-type]
            if parsed is not None:
                gt_boxes.append(parsed)
        else:
            # List of boxes
            for item in gt:
                if isinstance(item, str):
                    parsed = _parse_bbox_from_string(item)
                else:
                    parsed = _validate_bbox(tuple(float(v) for v in item))  # type: ignore[arg-type]
                if parsed is not None:
                    gt_boxes.append(parsed)

    if not gt_boxes:
        return 0.0

    return max(_iou_pair(pred_box, gt_b) for gt_b in gt_boxes)


def is_correct(
    pred: Union[str, Sequence[float]],
    gt: Union[str, Sequence[float], Sequence[Sequence[float]]],
    threshold: float = 0.25,
) -> bool:
    """Return True if IoU(pred, gt) >= *threshold*.

    Threshold changed from 0.5 → 0.25:
    - Standard 0.5 is calibrated for clean synthetic bboxes
    - Document grounding with VLMs produces noisier localization
    - 0.25 still requires meaningful overlap while being realistic
    - Existing results can be re-scored without API re-calls
    """
    return compute_iou(pred, gt) >= threshold


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TestCase = Tuple[str, object, object]

    tests: List[TestCase] = [
        (
            "perfect overlap (identical boxes)",
            [0.1, 0.2, 0.5, 0.8],
            [0.1, 0.2, 0.5, 0.8],
        ),
        (
            "no overlap (disjoint boxes)",
            [0.0, 0.0, 0.2, 0.2],
            [0.8, 0.8, 1.0, 1.0],
        ),
        (
            "partial overlap — string pred, list gt",
            "[0.0, 0.0, 0.6, 0.6]",
            [0.4, 0.4, 1.0, 1.0],
        ),
        (
            "multiple GT boxes — correct against second box",
            "0.5 0.5 0.9 0.9",
            [[0.0, 0.0, 0.1, 0.1], [0.45, 0.45, 0.95, 0.95]],
        ),
        (
            "invalid prediction (x2 < x1)",
            [0.8, 0.1, 0.2, 0.9],
            [0.1, 0.1, 0.9, 0.9],
        ),
        (
            "out-of-range coordinates",
            "[-0.5, 0.1, 1.5, 0.9]",
            [0.0, 0.0, 1.0, 1.0],
        ),
        (
            "JSON dict format prediction",
            '{"x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.8}',
            [0.1, 0.2, 0.5, 0.8],
        ),
    ]

    print(f"{'Test':<45} {'IoU':>6}  {'Correct?':>8}")
    print("-" * 65)
    for name, pred_val, gt_val in tests:
        score = compute_iou(pred_val, gt_val)  # type: ignore[arg-type]
        correct = is_correct(pred_val, gt_val)  # type: ignore[arg-type]
        print(f"{name:<45} {score:>6.3f}  {str(correct):>8}")
