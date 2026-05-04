"""
ROUGE-L scoring for abstractive VQA tasks.

Used by:
    - VisualMRC: abstractive visual machine reading comprehension where
      predicted answers are compared against reference answers using
      longest-common-subsequence-based ROUGE-L.

ROUGE-L measures the longest common subsequence (LCS) between the
predicted string and each ground-truth answer, rewarding fluency and
recall without requiring contiguous matches.

Threshold for is_correct: ROUGE-L >= 0.5
"""

from __future__ import annotations

from typing import List, Union

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and strip leading/trailing whitespace."""
    return text.lower().strip()


# ---------------------------------------------------------------------------
# LCS-based ROUGE-L (manual fallback / primary implementation)
# ---------------------------------------------------------------------------

def _lcs_length(a: List[str], b: List[str]) -> int:
    """Return the length of the longest common subsequence of token lists a and b."""
    m, n = len(a), len(b)
    # Use two-row DP to keep memory O(min(m, n))
    if m < n:
        a, b = b, a
        m, n = n, m
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _rouge_l_score(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 between a single prediction and a single reference.

    Tokenisation: whitespace split on normalised strings.
    Returns a float in [0.0, 1.0].
    """
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0.0:
        return 0.0

    # F1 with beta=1
    f1 = (2.0 * precision * recall) / (precision + recall)
    return f1


# ---------------------------------------------------------------------------
# Try to use rouge_score library; fall back to manual implementation
# ---------------------------------------------------------------------------

try:
    from rouge_score import rouge_scorer as _rouge_scorer  # type: ignore

    _SCORER = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def _rouge_l_single(prediction: str, reference: str) -> float:
        """ROUGE-L F1 via the rouge_score library."""
        score = _SCORER.score(
            _normalize(reference),
            _normalize(prediction),
        )
        return score["rougeL"].fmeasure

except ImportError:  # pragma: no cover
    _rouge_l_single = _rouge_l_score  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ROUGE_L_THRESHOLD: float = 0.5


def compute_rouge_l(
    predicted: str,
    gt_answers: Union[str, List[str]],
) -> float:
    """
    Compute ROUGE-L between *predicted* and one or more ground-truth answers.

    When multiple GT answers are provided the score is the **maximum** ROUGE-L
    over all references (lenient evaluation, consistent with VisualMRC practice).

    Parameters
    ----------
    predicted:
        The model's predicted answer string.
    gt_answers:
        A single ground-truth string or a list of ground-truth strings.

    Returns
    -------
    float
        ROUGE-L F1 in [0.0, 1.0].
    """
    if isinstance(gt_answers, str):
        gt_answers = [gt_answers]

    if not gt_answers:
        raise ValueError("gt_answers must contain at least one reference string.")

    return max(_rouge_l_single(predicted, ref) for ref in gt_answers)


def is_correct(
    predicted: str,
    gt_answers: Union[str, List[str]],
    threshold: float = ROUGE_L_THRESHOLD,
) -> bool:
    """
    Return True if compute_rouge_l(predicted, gt_answers) >= threshold.

    Parameters
    ----------
    predicted:
        The model's predicted answer.
    gt_answers:
        One or more ground-truth answers.
    threshold:
        Minimum ROUGE-L score to be considered correct (default 0.5).

    Returns
    -------
    bool
    """
    return compute_rouge_l(predicted, gt_answers) >= threshold


# ---------------------------------------------------------------------------
# Self-contained tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_CASES = [
        # (predicted, gt_answers, expected_is_correct, description)
        (
            "the cat sat on the mat",
            ["the cat sat on the mat"],
            True,
            "Exact match → ROUGE-L = 1.0",
        ),
        (
            "the cat sat",
            ["the cat sat on the mat"],
            True,
            "Partial match: high overlap, should be >= 0.5",
        ),
        (
            "dog ran across field",
            ["the cat sat on the mat"],
            False,
            "Low overlap: should be < 0.5",
        ),
        (
            "Paris is the capital of France",
            ["Paris is the capital of France", "The capital city is Paris"],
            True,
            "Multiple GT answers: max taken; pred matches first GT exactly",
        ),
        (
            "",
            ["non-empty reference"],
            False,
            "Empty prediction → ROUGE-L = 0.0",
        ),
        (
            "revenue increased significantly in Q3",
            ["revenue increased in Q3", "Q3 saw a significant revenue increase"],
            True,
            "Multiple GT answers; good overlap with both",
        ),
    ]

    print("=" * 70)
    print("ROUGE-L scoring tests")
    print("=" * 70)
    all_passed = True
    for pred, gts, expected, desc in TEST_CASES:
        score = compute_rouge_l(pred, gts)
        result = is_correct(pred, gts)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"[{status}] {desc}")
        print(f"       pred : {pred!r}")
        print(f"       gts  : {gts}")
        print(f"       score: {score:.4f}  |  is_correct={result}  (expected={expected})")
        print()

    print("All tests passed." if all_passed else "Some tests FAILED.")
