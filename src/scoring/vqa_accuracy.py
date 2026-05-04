"""
VQA accuracy scoring for TextVQA (multi-annotator).

Used by:
    - TextVQA: a dataset where each question has 10 human-annotated answers.
      The standard VQA accuracy metric rewards a prediction that matches
      at least 3 of the 10 annotators.

Scoring formula (per VQA v2 / TextVQA convention):
    score = min(count_matching_annotators / 3, 1.0)

A prediction "matches" an annotator answer via normalised case-insensitive
exact match after:
    - Lowercasing
    - Stripping leading/trailing whitespace
    - Removing articles: "a", "an", "the"
    - Normalising numbers (e.g. "two" stays as-is; "2.0" → "2")

is_correct: vqa_accuracy_score(predicted, gt_list_of_10) >= 1.0
  i.e. the prediction matches at least 3 of the 10 annotators.
"""

from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}

# Punctuation to strip (VQA standard: remove most punctuation)
_PUNCT_RE = re.compile(r"[^\w\s]")

# Trailing / leading whitespace collapser
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_number(token: str) -> str:
    """
    If *token* is a float with no meaningful decimal, convert to int string.

    "2.0" → "2", "3.14" → "3.14", "seven" → "seven"
    """
    try:
        val = float(token)
        if val == int(val):
            return str(int(val))
        return f"{val:g}"
    except ValueError:
        return token


def _normalize(text: str) -> str:
    """
    Apply VQA-standard normalisation to an answer string.

    Steps:
    1. Lowercase.
    2. Tokenise on whitespace.
    3. For each token: attempt number normalisation first (preserving the
       decimal point), then strip remaining punctuation from non-numeric tokens.
    4. Remove articles (a, an, the).
    5. Re-join with single spaces.

    Number normalisation is applied before punctuation stripping so that
    "2.0" → "2" rather than being fragmented into "2 0" by punct removal.
    """
    text = text.lower().strip()
    raw_tokens = _WHITESPACE_RE.split(text)
    processed: list[str] = []
    for tok in raw_tokens:
        if not tok:
            continue
        # Try numeric normalisation first (keeps decimal point intact)
        num = _normalize_number(tok)
        try:
            float(num)  # if it still parses as a number, use it as-is
            processed.append(num)
        except ValueError:
            # Not a number: strip punctuation then keep
            cleaned = _PUNCT_RE.sub(" ", tok).strip()
            sub_tokens = _WHITESPACE_RE.split(cleaned)
            processed.extend(t for t in sub_tokens if t)
    tokens = [t for t in processed if t and t not in _ARTICLES]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Match predicate
# ---------------------------------------------------------------------------

def _match(predicted: str, annotator_answer: str) -> bool:
    """Return True if normalised *predicted* equals normalised *annotator_answer*."""
    return _normalize(predicted) == _normalize(annotator_answer)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

VQA_CORRECT_THRESHOLD: float = 1.0   # score must be >= 1.0 (i.e. >= 3 matches)
_MIN_AGREEING_ANNOTATORS: int = 3


def vqa_accuracy_score(
    predicted: str,
    gt_answers: List[str],
) -> float:
    """
    Compute VQA accuracy score for a single prediction.

    Parameters
    ----------
    predicted:
        The model's predicted answer string.
    gt_answers:
        List of ground-truth annotator answers.  Typically 10 answers for
        TextVQA, but the function is robust to other lengths.

    Returns
    -------
    float
        min(number_of_matching_annotators / 3, 1.0) in [0.0, 1.0].

    Raises
    ------
    ValueError
        If gt_answers is empty.
    """
    if not gt_answers:
        raise ValueError("gt_answers must contain at least one annotator answer.")

    count = sum(1 for ann in gt_answers if _match(predicted, ann))
    return min(count / _MIN_AGREEING_ANNOTATORS, 1.0)


def is_correct(
    predicted: str,
    gt_answers: List[str],
    threshold: float = VQA_CORRECT_THRESHOLD,
) -> bool:
    """
    Return True if vqa_accuracy_score >= threshold (default 1.0, i.e. >= 3 matches).

    Parameters
    ----------
    predicted:
        The model's predicted answer.
    gt_answers:
        List of annotator ground-truth answers (typically 10 for TextVQA).
    threshold:
        Minimum score to be considered correct (default 1.0).

    Returns
    -------
    bool
    """
    return vqa_accuracy_score(predicted, gt_answers) >= threshold


# ---------------------------------------------------------------------------
# Self-contained tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build test cases using 10-element annotator lists
    def _make_gt(dominant: str, count: int, filler: str = "other") -> List[str]:
        """Return 10 annotator answers with *count* copies of *dominant*."""
        answers = [dominant] * count + [filler] * (10 - count)
        return answers

    TEST_CASES = [
        # (predicted, gt_answers, expected_is_correct, description)
        (
            "cat",
            _make_gt("cat", 10),
            True,
            "All 10 annotators agree → score = 1.0",
        ),
        (
            "cat",
            _make_gt("cat", 3),
            True,
            "Exactly 3 annotators agree → score = 1.0 (boundary)",
        ),
        (
            "cat",
            _make_gt("cat", 2),
            False,
            "Only 2 annotators agree → score = 0.67, not correct",
        ),
        (
            "The Cat",
            _make_gt("cat", 5),
            True,
            "Article removal + case-fold: 'The Cat' matches 'cat'",
        ),
        (
            "2",
            _make_gt("2.0", 4),
            True,
            "Number normalisation: '2' matches '2.0'",
        ),
        (
            "paris",
            _make_gt("London", 10),
            False,
            "Wrong answer: zero annotator matches",
        ),
    ]

    print("=" * 70)
    print("VQA accuracy tests (TextVQA)")
    print("=" * 70)
    all_passed = True
    for pred, gts, expected, desc in TEST_CASES:
        score = vqa_accuracy_score(pred, gts)
        result = is_correct(pred, gts)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        matching = sum(1 for a in gts if _match(pred, a))
        print(f"[{status}] {desc}")
        print(f"       pred    : {pred!r}  (normalised: {_normalize(pred)!r})")
        print(f"       matches : {matching}/10  →  score={score:.4f}  |  is_correct={result}  (expected={expected})")
        print()

    print("All tests passed." if all_passed else "Some tests FAILED.")
