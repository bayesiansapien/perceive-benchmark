"""
Relaxed accuracy scoring metric for numeric and free-text answers.

Used by:
    - ChartQA (primary metric; numeric tolerance of 5%)
    - PlotQA
    - FigureQA (partial)

The metric first attempts a relaxed numeric comparison (allowing up to 5%
relative error after stripping currency symbols, commas, and percent signs).
If the predicted or ground-truth value is non-numeric, it falls back to a
case-insensitive substring match in either direction.

Reference:
    Masry et al., "ChartQA: A Benchmark for Question Answering about Charts
    with Visual and Logical Reasoning", ACL Findings 2022.
"""

import re
from typing import Optional, Union


# Characters to strip before attempting numeric parsing
_STRIP_PATTERN = re.compile(r"[\$€£,\s]")
_PERCENT_PATTERN = re.compile(r"%\s*$")


def _clean_for_numeric(text: str) -> str:
    """
    Remove currency symbols, commas, whitespace, and trailing percent sign
    so that strings like '$1,234.56' or '42%' can be parsed as floats.
    """
    text = _PERCENT_PATTERN.sub("", text)
    text = _STRIP_PATTERN.sub("", text)
    return text


def _try_parse_float(text: str) -> Optional[float]:
    """
    Attempt to parse *text* as a float after cleaning. Returns None on failure.
    """
    cleaned = _clean_for_numeric(text)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _is_na(text: str) -> bool:
    """Return True if text represents a not-applicable / missing sentinel."""
    return text.strip().lower() in {"n/a", "na", "none", "null", "-", ""}


def _numeric_match(pred_val: float, gt_val: float, tolerance: float = 0.05) -> bool:
    """
    Return True if |pred - gt| / max(|gt|, 1e-9) < tolerance.

    The denominator floor of 1e-9 prevents division-by-zero when gt is 0,
    in which case the comparison becomes effectively an absolute check.
    """
    denominator = max(abs(gt_val), 1e-9)
    return abs(pred_val - gt_val) / denominator < tolerance


def _string_match(prediction: str, ground_truth: str) -> bool:
    """
    Case-insensitive substring match in either direction:
    True if pred in gt OR gt in pred.
    """
    pred_lower = prediction.lower().strip()
    gt_lower = ground_truth.lower().strip()
    return pred_lower in gt_lower or gt_lower in pred_lower


def compute_relaxed_accuracy(
    prediction: Optional[str],
    ground_truths: Union[str, list[str]],
    numeric_tolerance: float = 0.05,
) -> float:
    """
    Compute the relaxed accuracy score for a single question–answer pair.

    The function returns 1.0 (correct) or 0.0 (incorrect). It checks every
    ground-truth alias and returns 1.0 as soon as any alias matches.

    Evaluation order for each (prediction, gt) pair:
        1. If prediction is None, empty, or "N/A"-like → 0.0 immediately.
        2. If both prediction and gt parse as floats → numeric comparison.
        3. Otherwise → case-insensitive substring match.

    Args:
        prediction:       The model's predicted answer string, or None.
        ground_truths:    A single GT string or a list of GT alias strings.
        numeric_tolerance: Relative tolerance for numeric comparison (default 0.05).

    Returns:
        1.0 if any alias is matched, 0.0 otherwise.
    """
    # Handle missing / null predictions
    if prediction is None:
        return 0.0
    pred_str = str(prediction).strip()
    if _is_na(pred_str):
        return 0.0

    # Normalize ground_truths to a list
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    if not ground_truths:
        return 0.0

    pred_float = _try_parse_float(pred_str)

    for gt in ground_truths:
        if gt is None:
            gt_str = ""
        else:
            gt_str = str(gt).strip()

        gt_float = _try_parse_float(gt_str)

        # Numeric path: both sides parsed successfully
        if pred_float is not None and gt_float is not None:
            if _numeric_match(pred_float, gt_float, tolerance=numeric_tolerance):
                return 1.0
        else:
            # String path: fall back to substring match
            if _string_match(pred_str, gt_str):
                return 1.0

    return 0.0


def is_correct_relaxed_accuracy(
    prediction: Optional[str],
    ground_truths: Union[str, list[str]],
    numeric_tolerance: float = 0.05,
) -> bool:
    """
    Return True if the relaxed accuracy score is 1.0.

    Args:
        prediction:        The model's predicted answer string, or None.
        ground_truths:     A single GT string or a list of GT alias strings.
        numeric_tolerance: Relative tolerance for numeric comparison (default 0.05).

    Returns:
        True if the prediction matches any ground-truth alias, False otherwise.
    """
    return compute_relaxed_accuracy(
        prediction, ground_truths, numeric_tolerance=numeric_tolerance
    ) == 1.0


if __name__ == "__main__":
    test_cases = [
        # (description, prediction, ground_truths, expected_correct)
        (
            "Exact numeric match",
            "42",
            ["42"],
            True,
        ),
        (
            "Numeric within 5% tolerance ($1,000 vs 1020 = 2% error)",
            "$1,020",
            ["$1,000"],
            True,
        ),
        (
            "Numeric outside 5% tolerance (1,200 vs 1,000 = 20% error)",
            "1200",
            ["1000"],
            False,
        ),
        (
            "Percent sign and currency stripped before comparison (95% vs 95)",
            "95%",
            ["95"],
            True,
        ),
        (
            "None prediction returns False",
            None,
            ["some answer"],
            False,
        ),
        (
            "String substring match — gt contained in prediction",
            "North America region",
            ["North America"],
            True,
        ),
        (
            "N/A prediction returns False regardless of GT",
            "N/A",
            ["42"],
            False,
        ),
    ]

    print("Relaxed Accuracy scoring tests")
    print("=" * 60)
    all_passed = True
    for desc, pred, gts, expected in test_cases:
        score = compute_relaxed_accuracy(pred, gts)
        correct = is_correct_relaxed_accuracy(pred, gts)
        status = "PASS" if correct == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"[{status}] {desc}")
        print(f"       pred={pred!r}  gt={gts}")
        print(f"       score={score:.1f}  is_correct={correct}  expected_correct={expected}")
        print()

    print("=" * 60)
    print("All tests passed." if all_passed else "Some tests FAILED.")
