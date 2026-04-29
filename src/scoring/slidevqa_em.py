"""
SlideVQA scoring: exact match with relaxed numeric accuracy for arithmetic questions.

Used by:
    - SlideVQA: a multi-page slide document VQA dataset containing both
      factual/extractive questions and arithmetic questions (e.g. totals, sums,
      averages, counts).

Two evaluation modes:
    1. Arithmetic questions  → relaxed numeric accuracy (5 % relative tolerance).
       Detected when the question contains trigger words ("total", "sum",
       "average", "mean", "how many") OR when both predicted and GT answers
       parse as numbers.
    2. Non-arithmetic questions → case-insensitive exact match after
       normalisation (strip whitespace, lowercase).

is_correct: em_or_relaxed(predicted, gt, question) returns True/False.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RELAXED_TOLERANCE: float = 0.05  # 5 % relative tolerance

_ARITHMETIC_KEYWORDS = re.compile(
    r"\b(total|sum|average|mean|how\s+many|count|add|combined|altogether)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _normalize_string(text: str) -> str:
    """Lowercase and strip leading/trailing whitespace."""
    return text.lower().strip()


def _try_parse_number(text: str) -> Optional[float]:
    """
    Attempt to parse *text* as a float after removing common formatting.

    Returns the float value or None if parsing fails.
    """
    cleaned = (
        text.strip()
        .replace(",", "")       # thousands separator
        .replace("$", "")       # currency
        .replace("€", "")
        .replace("£", "")
        .replace("%", "")       # percent
        .replace("~", "")       # approximation prefix
    )
    try:
        return float(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Arithmetic question detection
# ---------------------------------------------------------------------------


def _is_arithmetic_question(question: str, predicted: str, gt: str) -> bool:
    """
    Return True if the question-answer pair should be evaluated with relaxed
    numeric accuracy instead of exact match.

    A pair is arithmetic when ANY of the following hold:
    - The question contains one of the arithmetic trigger keywords.
    - Both *predicted* and *gt* parse successfully as numbers.
    """
    if _ARITHMETIC_KEYWORDS.search(question):
        return True

    if _try_parse_number(predicted) is not None and _try_parse_number(gt) is not None:
        return True

    return False


# ---------------------------------------------------------------------------
# Relaxed numeric comparison
# ---------------------------------------------------------------------------


def _relaxed_numeric_match(predicted: str, gt: str) -> bool:
    """
    Return True if *predicted* and *gt* represent numbers within RELAXED_TOLERANCE.

    Relative tolerance formula (same as numpy.isclose with atol=0):
        |pred - gt| <= tolerance * |gt|

    Special case: if gt == 0, use absolute tolerance == tolerance.
    """
    pred_val = _try_parse_number(predicted)
    gt_val = _try_parse_number(gt)

    if pred_val is None or gt_val is None:
        # Cannot parse one or both — fall back to string exact match
        return _normalize_string(predicted) == _normalize_string(gt)

    if gt_val == 0.0:
        return abs(pred_val) <= RELAXED_TOLERANCE

    return abs(pred_val - gt_val) <= RELAXED_TOLERANCE * abs(gt_val)


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------


def _exact_match(predicted: str, gt: str) -> bool:
    """Case-insensitive exact match after stripping whitespace."""
    return _normalize_string(predicted) == _normalize_string(gt)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def em_or_relaxed(predicted: str, gt: str, question: str) -> float:
    """
    Compute the SlideVQA score for a single prediction.

    Parameters
    ----------
    predicted:
        The model's predicted answer string.
    gt:
        The single ground-truth answer string.
    question:
        The original question text (used to detect arithmetic questions).

    Returns
    -------
    float
        1.0 if correct, 0.0 otherwise.
    """
    if _is_arithmetic_question(question, predicted, gt):
        return 1.0 if _relaxed_numeric_match(predicted, gt) else 0.0
    return 1.0 if _exact_match(predicted, gt) else 0.0


def is_correct(predicted: str, gt: str, question: str) -> bool:
    """
    Return True if the prediction is correct under the appropriate metric.

    Uses relaxed numeric accuracy (5 % tolerance) for arithmetic questions and
    case-insensitive exact match for all other questions.

    Parameters
    ----------
    predicted:
        The model's predicted answer.
    gt:
        The ground-truth answer.
    question:
        The original question string (used for arithmetic detection).

    Returns
    -------
    bool
    """
    return em_or_relaxed(predicted, gt, question) == 1.0


# ---------------------------------------------------------------------------
# Self-contained tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_CASES = [
        # (predicted, gt, question, expected_is_correct, description)
        (
            "Paris",
            "Paris",
            "What city is shown on the title slide?",
            True,
            "Exact match: identical strings",
        ),
        (
            "paris",
            "Paris",
            "What city is shown on the title slide?",
            True,
            "Exact match: case-insensitive",
        ),
        (
            "Berlin",
            "Paris",
            "What city is shown on the title slide?",
            False,
            "Exact match: wrong answer",
        ),
        (
            "1050",
            "1000",
            "What is the total revenue across all slides?",
            True,
            "Arithmetic (keyword 'total'): 5% tolerance — 1050 within 5% of 1000",
        ),
        (
            "1100",
            "1000",
            "What is the total revenue across all slides?",
            False,
            "Arithmetic: 10% off — outside 5% tolerance",
        ),
        (
            "42",
            "44",
            "How many data points are shown in the chart?",
            True,
            "Arithmetic (keyword 'how many'): ~4.5% off, within tolerance",
        ),
    ]

    print("=" * 70)
    print("SlideVQA EM + relaxed numeric tests")
    print("=" * 70)
    all_passed = True
    for pred, gt, question, expected, desc in TEST_CASES:
        score = em_or_relaxed(pred, gt, question)
        result = is_correct(pred, gt, question)
        is_arith = _is_arithmetic_question(question, pred, gt)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"[{status}] {desc}")
        print(f"       pred      : {pred!r}")
        print(f"       gt        : {gt!r}")
        print(f"       question  : {question!r}")
        print(f"       arithmetic: {is_arith}  |  score={score}  |  is_correct={result}  (expected={expected})")
        print()

    print("All tests passed." if all_passed else "Some tests FAILED.")
