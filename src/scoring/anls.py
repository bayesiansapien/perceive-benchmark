"""
Average Normalized Levenshtein Similarity (ANLS) scoring metric.

Used by:
    - DocVQA (Single Page Document VQA)
    - MP-DocVQA (Multi-Page Document VQA)
    - InfographicVQA
    - ST-VQA (Scene Text Visual Question Answering)

Reference:
    Biten et al., "Scene Text Visual Question Answering", ICCV 2019.
    Mathew et al., "DocVQA: A Dataset for VQA on Document Images", WACV 2021.

The metric handles multiple ground-truth answer aliases by taking the maximum
ANLS score across all aliases, rewarding any correct interpretation of the answer.
"""

import unicodedata
from typing import Optional, Union


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    # Use two-row DP to save memory
    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)

    for i in range(1, len1 + 1):
        curr[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, curr = curr, prev

    return prev[len2]


def _normalize(text: str) -> str:
    """Lowercase and normalize whitespace; apply Unicode NFC normalization."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Collapse internal whitespace and strip leading/trailing
    text = " ".join(text.split())
    return text


def _anls_single(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
    """
    Compute ANLS between a single prediction and a single ground-truth string.

    Args:
        prediction:   The model's predicted answer (already normalized).
        ground_truth: One ground-truth answer string (already normalized).
        threshold:    Similarity threshold τ; scores below this are set to 0.

    Returns:
        ANLS score in [0, 1].
    """
    max_len = max(len(prediction), len(ground_truth))
    if max_len == 0:
        # Both strings are empty — treat as a perfect match
        return 1.0

    edit_dist = _levenshtein_distance(prediction, ground_truth)
    normalized_edit_dist = edit_dist / max_len

    if normalized_edit_dist >= threshold:
        return 0.0
    return 1.0 - normalized_edit_dist


def compute_anls(
    prediction: Optional[str],
    ground_truths: Union[str, list[str]],
    threshold: float = 0.5,
) -> float:
    """
    Compute the ANLS score for a single question–answer pair.

    The score is the maximum ANLS over all provided ground-truth aliases,
    following the standard DocVQA evaluation protocol.

    Args:
        prediction:    The model's predicted answer string, or None.
        ground_truths: A single GT string or a list of GT alias strings.
        threshold:     Similarity threshold τ (default 0.5 per DocVQA standard).

    Returns:
        ANLS score in [0, 1].
    """
    # Normalize prediction
    if prediction is None:
        pred_norm = ""
    else:
        pred_norm = _normalize(str(prediction))

    # Ensure ground_truths is a list
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    if not ground_truths:
        # No ground-truth aliases available — cannot score
        return 0.0

    best = 0.0
    for gt in ground_truths:
        gt_norm = _normalize(str(gt)) if gt is not None else ""
        score = _anls_single(pred_norm, gt_norm, threshold=threshold)
        if score > best:
            best = score

    return best


def is_correct_anls(
    prediction: Optional[str],
    ground_truths: Union[str, list[str]],
    threshold: float = 0.5,
    correctness_threshold: float = 0.5,
) -> bool:
    """
    Return True if the ANLS score meets or exceeds the correctness threshold.

    Args:
        prediction:            The model's predicted answer string, or None.
        ground_truths:         A single GT string or a list of GT alias strings.
        threshold:             ANLS similarity threshold τ (default 0.5).
        correctness_threshold: Minimum ANLS score to count as correct (default 0.5).

    Returns:
        True if ANLS >= correctness_threshold, False otherwise.
    """
    return compute_anls(prediction, ground_truths, threshold=threshold) >= correctness_threshold


if __name__ == "__main__":
    test_cases = [
        # (description, prediction, ground_truths, expected_correct)
        (
            "Exact match",
            "quarterly report",
            ["quarterly report"],
            True,
        ),
        (
            "Minor typo within threshold (1 edit on 8 chars = 0.125 < 0.5)",
            "quartely report",
            ["quarterly report"],
            True,
        ),
        (
            "None prediction treated as empty string",
            None,
            ["some answer"],
            False,
        ),
        (
            "Multiple aliases — correct match on second alias",
            "12,345",
            ["twelve thousand three hundred forty-five", "12,345", "12345"],
            True,
        ),
        (
            "Case and whitespace normalization",
            "  TOTAL REVENUE  ",
            ["total revenue"],
            True,
        ),
        (
            "Completely wrong answer — large edit distance exceeds threshold",
            "elephant",
            ["quarterly report"],
            False,
        ),
    ]

    print("ANLS scoring tests")
    print("=" * 60)
    all_passed = True
    for desc, pred, gts, expected in test_cases:
        score = compute_anls(pred, gts)
        correct = is_correct_anls(pred, gts)
        status = "PASS" if correct == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"[{status}] {desc}")
        print(f"       pred={pred!r}  gt={gts}")
        print(f"       ANLS={score:.4f}  is_correct={correct}  expected_correct={expected}")
        print()

    print("=" * 60)
    print("All tests passed." if all_passed else "Some tests FAILED.")
