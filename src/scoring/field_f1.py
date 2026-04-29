"""
field_f1.py - Token-level F1 scoring for structured extraction tasks.

Used by: FUNSD, CORD, SROIE, DeepForm, HierText, SlideVQA (partially).

Algorithm:
  1. Normalize input: parse JSON if needed, concatenate all string values.
  2. Tokenize by whitespace (lowercase, strip punctuation and currency symbols).
  3. Compute precision and recall over token multisets.
  4. F1 = 2*P*R / (P+R), with special handling for empty inputs.

A prediction is considered correct when F1 >= 0.5.
"""

import json
import re
import string
from collections import Counter
from typing import Union


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CURRENCY_RE = re.compile(r"[$€£¥₹]")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _extract_text(value: Union[str, dict, list, int, float, None]) -> str:
    """Recursively extract all leaf string values from a parsed JSON structure."""
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_extract_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_extract_text(item) for item in value)
    return str(value)


def _normalize(text: str) -> str:
    """Normalize a raw string: strip currency, collapse whitespace, lowercase, remove punctuation."""
    text = _CURRENCY_RE.sub("", text)
    text = text.replace(",", "")
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _to_normalized_string(value: Union[str, dict, list, None]) -> str:
    """
    Convert a predicted or ground-truth value to a single normalized string.

    Handles:
      - Plain string (possibly a JSON-encoded string).
      - dict / list (already parsed).
      - None -> empty string.
    """
    if value is None:
        return ""

    # If it is already a dict or list, extract text directly.
    if isinstance(value, (dict, list)):
        return _normalize(_extract_text(value))

    # It must be a string at this point.
    text: str = value

    # Attempt JSON parse so that JSON-encoded dicts/lists are handled.
    stripped = text.strip()
    if stripped and stripped[0] in ("{", "["):
        try:
            parsed = json.loads(stripped)
            return _normalize(_extract_text(parsed))
        except (json.JSONDecodeError, ValueError):
            pass  # treat as plain string

    return _normalize(text)


def _tokenize(text: str) -> list[str]:
    """Split normalized text into tokens; return empty list for empty strings."""
    if not text:
        return []
    return text.split()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_f1(
    predicted: Union[str, dict, list, None],
    ground_truth: Union[str, dict, list, None],
) -> float:
    """
    Compute token-level F1 between *predicted* and *ground_truth*.

    Parameters
    ----------
    predicted:
        The model's output.  May be a plain string, a JSON-encoded string,
        a dict, or a list.
    ground_truth:
        The reference answer.  Same type flexibility as *predicted*.

    Returns
    -------
    float
        F1 score in [0.0, 1.0].
        Returns 1.0 when both inputs are effectively empty.
        Returns 0.0 when exactly one input is empty.
    """
    pred_norm = _to_normalized_string(predicted)
    gt_norm = _to_normalized_string(ground_truth)

    pred_tokens = _tokenize(pred_norm)
    gt_tokens = _tokenize(gt_norm)

    # Both empty -> perfect match.
    if not pred_tokens and not gt_tokens:
        return 1.0

    # One empty -> no overlap possible.
    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    # Token overlap (multiset intersection).
    common: Counter = pred_counter & gt_counter
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / sum(pred_counter.values())
    recall = num_common / sum(gt_counter.values())
    f1 = 2.0 * precision * recall / (precision + recall)
    return f1


def is_correct(
    predicted: Union[str, dict, list, None],
    ground_truth: Union[str, dict, list, None],
    threshold: float = 0.5,
) -> bool:
    """
    Return True when the token-level F1 meets or exceeds *threshold*.

    Parameters
    ----------
    predicted:
        Model output.
    ground_truth:
        Reference answer.
    threshold:
        Minimum F1 required to count as correct (default 0.5).

    Returns
    -------
    bool
    """
    return compute_f1(predicted, ground_truth) >= threshold


# ---------------------------------------------------------------------------
# Self-contained tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _PASS = "\033[92mPASS\033[0m"
    _FAIL = "\033[91mFAIL\033[0m"

    def _check(label: str, got, expected) -> None:
        ok = abs(got - expected) < 1e-6 if isinstance(expected, float) else got == expected
        status = _PASS if ok else _FAIL
        print(f"[{status}] {label}: got={got!r}, expected={expected!r}")

    print("=== field_f1.py test cases ===\n")

    # 1. Identical plain strings.
    f1 = compute_f1("Invoice Total", "Invoice Total")
    _check("1. Identical strings -> F1=1.0", f1, 1.0)

    # 2. Completely different strings.
    f1 = compute_f1("hello world", "foo bar baz")
    _check("2. No overlap -> F1=0.0", f1, 0.0)

    # 3. Partial overlap (precision=1/2, recall=1/3, F1=2/(2+3)=0.4).
    f1 = compute_f1("foo bar", "foo baz qux")
    expected_f1 = round(2 * (1 / 2) * (1 / 3) / (1 / 2 + 1 / 3), 6)
    _check(f"3. Partial overlap -> F1={expected_f1}", round(f1, 6), expected_f1)

    # 4. Both empty strings -> correct (F1=1.0).
    f1 = compute_f1("", "")
    _check("4. Both empty -> F1=1.0", f1, 1.0)
    _check("4b. Both empty -> is_correct=True", is_correct("", ""), True)

    # 5. JSON-encoded dict as predicted, plain string as ground truth.
    pred_json = '{"company": "Acme Corp", "total": "$1,234.56"}'
    gt_plain = "acme corp 123456"
    f1 = compute_f1(pred_json, gt_plain)
    _check("5. JSON dict predicted, partial match -> F1>0", f1 > 0.0, True)
    print(f"     (actual F1={f1:.4f})")

    # 6. Currency and comma normalization.
    f1 = compute_f1("$1,234.56", "1234.56")
    _check("6. Currency/comma stripping -> F1=1.0", f1, 1.0)

    # 7. is_correct threshold check.
    # "the quick brown fox" vs "the slow brown fox": 3/4 overlap
    correct = is_correct("the quick brown fox", "the slow brown fox", threshold=0.5)
    _check("7. is_correct with 3/4 overlap and threshold=0.5 -> True", correct, True)

    print("\nDone.")
