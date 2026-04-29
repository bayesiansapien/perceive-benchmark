"""
exact_match.py - Exact string match (case-insensitive, normalized) for classification tasks.

Used by: RVL-CDIP, TabFact, AI2D, DocBank, PubLayNet.

Normalization pipeline:
  1. Lowercase.
  2. Strip leading/trailing whitespace.
  3. Collapse multiple internal spaces to a single space.
  4. Dataset-specific alias resolution (TabFact boolean labels, RVL-CDIP int->label).

A prediction is considered correct when normalized(predicted) == normalized(ground_truth).
"""

import re
from typing import Union

# ---------------------------------------------------------------------------
# Dataset-specific alias maps
# ---------------------------------------------------------------------------

# TabFact: map numeric strings and common textual variants to canonical labels.
_TABFACT_ALIASES: dict[str, str] = {
    "1": "entailed",
    "true": "entailed",
    "yes": "entailed",
    "support": "entailed",
    "supports": "entailed",
    "0": "refuted",
    "false": "refuted",
    "no": "refuted",
    "refute": "refuted",
    "contradicts": "refuted",
    "contradict": "refuted",
}

# RVL-CDIP: map integer class indices (as strings) to human-readable label strings.
_RVL_CDIP_LABELS: dict[str, str] = {
    "0": "letter",
    "1": "form",
    "2": "email",
    "3": "handwritten",
    "4": "advertisement",
    "5": "scientific report",
    "6": "scientific publication",
    "7": "specification",
    "8": "file folder",
    "9": "news article",
    "10": "budget",
    "11": "invoice",
    "12": "presentation",
    "13": "questionnaire",
    "14": "resume",
    "15": "memo",
}

_WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _base_normalize(text: str) -> str:
    """Apply the common normalization pipeline to *text*."""
    text = text.lower().strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def _resolve_tabfact(text: str) -> str:
    """Map TabFact aliases to canonical 'entailed' / 'refuted' labels."""
    return _TABFACT_ALIASES.get(text, text)


def _resolve_rvl_cdip(text: str) -> str:
    """Map RVL-CDIP integer class indices to human-readable label strings."""
    return _RVL_CDIP_LABELS.get(text, text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize(
    text: Union[str, int, float, None],
    dataset: str = "",
) -> str:
    """
    Normalize *text* for exact-match comparison.

    Parameters
    ----------
    text:
        Raw predicted or ground-truth string.  Non-string scalars are
        converted to strings before normalization.
    dataset:
        Optional dataset hint for alias resolution.  Recognized values:
        ``"tabfact"`` and ``"rvl-cdip"`` (case-insensitive).

    Returns
    -------
    str
        Normalized string.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    normalized = _base_normalize(text)

    dataset_key = dataset.lower().replace("_", "-")
    if dataset_key == "tabfact":
        normalized = _resolve_tabfact(normalized)
    elif dataset_key in ("rvl-cdip", "rvlcdip"):
        normalized = _resolve_rvl_cdip(normalized)

    return normalized


def exact_match(
    predicted: Union[str, int, float, None],
    ground_truth: Union[str, int, float, None],
    dataset: str = "",
) -> float:
    """
    Compute exact-match score between *predicted* and *ground_truth*.

    Parameters
    ----------
    predicted:
        Model output.
    ground_truth:
        Reference answer.
    dataset:
        Optional dataset hint passed to :func:`normalize`.

    Returns
    -------
    float
        1.0 if the normalized strings are equal, 0.0 otherwise.
    """
    return 1.0 if normalize(predicted, dataset) == normalize(ground_truth, dataset) else 0.0


def is_correct(
    predicted: Union[str, int, float, None],
    ground_truth: Union[str, int, float, None],
    dataset: str = "",
) -> bool:
    """
    Return True when *predicted* exactly matches *ground_truth* after normalization.

    Parameters
    ----------
    predicted:
        Model output.
    ground_truth:
        Reference answer.
    dataset:
        Optional dataset hint passed to :func:`normalize`.

    Returns
    -------
    bool
    """
    return exact_match(predicted, ground_truth, dataset) == 1.0


def exact_match_any(
    predicted: Union[str, int, float, None],
    gt_list: list[Union[str, int, float, None]],
    dataset: str = "",
) -> float:
    """
    Return 1.0 if *predicted* exactly matches any element in *gt_list*.

    Useful when a question has multiple acceptable reference answers.

    Parameters
    ----------
    predicted:
        Model output.
    gt_list:
        List of acceptable reference answers.
    dataset:
        Optional dataset hint passed to :func:`normalize`.

    Returns
    -------
    float
        1.0 if any match is found, 0.0 otherwise.
    """
    pred_norm = normalize(predicted, dataset)
    return 1.0 if any(pred_norm == normalize(gt, dataset) for gt in gt_list) else 0.0


# ---------------------------------------------------------------------------
# Self-contained tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _PASS = "\033[92mPASS\033[0m"
    _FAIL = "\033[91mFAIL\033[0m"

    def _check(label: str, got, expected) -> None:
        ok = got == expected
        status = _PASS if ok else _FAIL
        print(f"[{status}] {label}: got={got!r}, expected={expected!r}")

    print("=== exact_match.py test cases ===\n")

    # 1. Basic case-insensitive match.
    _check(
        "1. Case-insensitive match",
        is_correct("Invoice", "invoice"),
        True,
    )

    # 2. Whitespace normalization.
    _check(
        "2. Extra internal whitespace",
        is_correct("scientific  publication", "scientific publication"),
        True,
    )

    # 3. TabFact: numeric label '1' -> 'entailed'.
    _check(
        "3. TabFact '1' == 'entailed'",
        is_correct("1", "entailed", dataset="tabfact"),
        True,
    )

    # 4. TabFact: 'false' -> 'refuted'.
    _check(
        "4. TabFact 'false' == 'refuted'",
        is_correct("false", "refuted", dataset="tabfact"),
        True,
    )

    # 5. RVL-CDIP: integer index 11 -> 'invoice'.
    _check(
        "5. RVL-CDIP int 11 -> 'invoice'",
        is_correct(11, "invoice", dataset="rvl-cdip"),
        True,
    )

    # 6. RVL-CDIP: string index '6' -> 'scientific publication'.
    _check(
        "6. RVL-CDIP '6' -> 'scientific publication'",
        is_correct("6", "scientific publication", dataset="rvl-cdip"),
        True,
    )

    # 7. exact_match_any: predicted matches second element of list.
    score = exact_match_any("Yes", ["no", "entailed", "yes"], dataset="")
    _check(
        "7. exact_match_any hits second acceptable answer",
        score,
        1.0,
    )

    # 8. Mismatch returns 0.0.
    score = exact_match("form", "invoice")
    _check("8. Mismatch -> 0.0", score, 0.0)

    # 9. None predicted -> empty string normalization.
    _check(
        "9. None predicted does not match non-empty gt",
        is_correct(None, "letter"),
        False,
    )

    print("\nDone.")
