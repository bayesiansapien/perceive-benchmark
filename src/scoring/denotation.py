"""
Denotation accuracy scoring for WikiTableQuestions (WTQ).

Used by:
    - WTQ (WikiTableQuestions): open-domain table QA where answers may be
      numbers, dates, strings, or lists thereof.  Two answers are considered
      correct when they *denote* the same value under semantic normalisation,
      not merely when they share surface form.

Normalisation pipeline (self-contained, no official WTQ repo required):
    1. Numbers  : strip commas/currency symbols, unify int/float repr.
    2. Dates    : parse common date formats → ISO-8601 (YYYY-MM-DD).
    3. Lists    : split on commas/semicolons, normalise each element, sort.
    4. Strings  : case-fold, strip whitespace.

is_correct: denotation_match(predicted, gt_list) where gt_list contains one
or more acceptable answer strings.
"""

from __future__ import annotations

import re
from typing import List, Optional, Union

# ---------------------------------------------------------------------------
# Number normalisation
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(
    r"^[+\-]?"            # optional sign
    r"[$€£¥]?"            # optional currency symbol
    r"([0-9,]+)"          # integer part (with optional thousands commas)
    r"(\.[0-9]*)?"        # optional decimal part
    r"\s*%?$"             # optional percentage / trailing space
)


def _try_parse_number(text: str) -> Optional[float]:
    """Return a float if *text* represents a number, else None."""
    t = text.strip().replace(",", "").replace("$", "").replace("€", "").replace("£", "").replace("¥", "")
    try:
        return float(t)
    except ValueError:
        return None


def _normalize_number(text: str) -> str:
    """
    Canonicalise a numeric string.

    "1,234"  → "1234"
    "1234.0" → "1234"
    "3.14"   → "3.14"
    """
    val = _try_parse_number(text)
    if val is None:
        return text
    # Drop unnecessary trailing zeros / decimal point
    if val == int(val):
        return str(int(val))
    # Keep significant decimal digits but strip trailing zeros
    return f"{val:g}"


# ---------------------------------------------------------------------------
# Date normalisation
# ---------------------------------------------------------------------------

# (regex, named groups) pairs, tried in order
_DATE_PATTERNS: List[tuple] = [
    # YYYY-MM-DD or YYYY/MM/DD
    (
        re.compile(r"^(?P<y>\d{4})[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})$"),
        ("y", "m", "d"),
    ),
    # MM/DD/YYYY or MM-DD-YYYY
    (
        re.compile(r"^(?P<m>\d{1,2})[-/](?P<d>\d{1,2})[-/](?P<y>\d{4})$"),
        ("y", "m", "d"),
    ),
    # Month name variants: "January 1, 2020" or "1 January 2020"
    (
        re.compile(
            r"^(?:(?P<d1>\d{1,2})\s+)?(?P<mon>[A-Za-z]+)\.?\s*(?P<d2>\d{1,2})?,?\s*(?P<y>\d{4})$"
        ),
        None,  # handled specially
    ),
]

_MONTH_MAP = {
    name: f"{i:02d}"
    for i, name in enumerate(
        [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ],
        start=1,
    )
}
# Short names
_MONTH_MAP.update(
    {name[:3]: num for name, num in _MONTH_MAP.items()}
)


def _try_parse_date(text: str) -> Optional[str]:
    """
    Return ISO-8601 string "YYYY-MM-DD" if *text* looks like a date, else None.
    Partial dates (year only) are returned as "YYYY".
    """
    t = text.strip()

    # Year only
    if re.fullmatch(r"\d{4}", t):
        return t

    for pattern, _ in _DATE_PATTERNS[:2]:
        m = pattern.match(t)
        if m:
            y = m.group("y")
            mo = m.group("m").zfill(2)
            d = m.group("d").zfill(2)
            return f"{y}-{mo}-{d}"

    # Month name pattern
    m = _DATE_PATTERNS[2][0].match(t)
    if m:
        mon_str = m.group("mon").lower().rstrip(".")
        mon_num = _MONTH_MAP.get(mon_str) or _MONTH_MAP.get(mon_str[:3])
        if mon_num:
            y = m.group("y")
            d_raw = m.group("d1") or m.group("d2") or "1"
            d = d_raw.strip(",").zfill(2)
            return f"{y}-{mon_num}-{d}"

    return None


# ---------------------------------------------------------------------------
# List detection / splitting
# ---------------------------------------------------------------------------

_LIST_SEP_RE = re.compile(r"[;,]\s*")


def _is_list_answer(text: str) -> bool:
    """Heuristic: answer is a list if it contains ';' or a comma not inside a number."""
    if ";" in text:
        return True
    # Comma inside a number: "1,234", don't treat as list
    cleaned = re.sub(r"\d,\d", "", text)
    return "," in cleaned


# ---------------------------------------------------------------------------
# Core normalisation
# ---------------------------------------------------------------------------

def _normalize_single(text: str) -> str:
    """
    Normalise a single (non-list) answer token for denotation comparison.

    Priority: number → date → string (case-folded, stripped).
    """
    t = text.strip()

    num = _try_parse_number(t)
    if num is not None:
        return _normalize_number(t)

    date = _try_parse_date(t)
    if date is not None:
        return date

    return t.lower()


def _normalize_answer(text: str) -> Union[str, List[str]]:
    """
    Full normalisation of an answer string.

    Returns a sorted list of normalised tokens if the answer appears to be a
    list, otherwise returns a single normalised string.

    Date strings (e.g. "January 5, 2020") are parsed first to avoid the
    comma inside them being mis-treated as a list separator.
    """
    t = text.strip()

    # Attempt date parsing before list detection, a date like "January 5, 2020"
    # contains a comma but is not a list.
    date = _try_parse_date(t)
    if date is not None:
        return date

    if _is_list_answer(t):
        parts = _LIST_SEP_RE.split(t)
        normalised = sorted(_normalize_single(p) for p in parts if p.strip())
        return normalised

    return _normalize_single(t)


# ---------------------------------------------------------------------------
# Denotation match
# ---------------------------------------------------------------------------

def denotation_match(predicted: str, gt_answers: Union[str, List[str]]) -> bool:
    """
    Return True if *predicted* denotation-matches any answer in *gt_answers*.

    Parameters
    ----------
    predicted:
        The model's raw predicted answer string.
    gt_answers:
        A single GT string or a list of acceptable GT strings (WTQ sometimes
        has multiple valid answers).

    Returns
    -------
    bool
    """
    if isinstance(gt_answers, str):
        gt_answers = [gt_answers]

    norm_pred = _normalize_answer(predicted)

    for gt in gt_answers:
        norm_gt = _normalize_answer(gt)
        if norm_pred == norm_gt:
            return True

    return False


def compute_denotation_accuracy(
    predicted: str,
    gt_answers: Union[str, List[str]],
) -> float:
    """
    Return 1.0 if denotation_match succeeds, else 0.0.

    Provided for API consistency with other scoring modules.
    """
    return 1.0 if denotation_match(predicted, gt_answers) else 0.0


def is_correct(predicted: str, gt_answers: Union[str, List[str]]) -> bool:
    """Alias for denotation_match; included for API consistency."""
    return denotation_match(predicted, gt_answers)


# ---------------------------------------------------------------------------
# Self-contained tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_CASES = [
        # (predicted, gt_answers, expected_match, description)
        (
            "1,234",
            ["1234"],
            True,
            "Number: comma-formatted == plain integer",
        ),
        (
            "1234.0",
            ["1234"],
            True,
            "Number: float with trailing zero == integer",
        ),
        (
            "January 5, 2020",
            ["2020-01-05"],
            True,
            "Date: written month name → ISO-8601",
        ),
        (
            "03/15/2019",
            ["2019-03-15"],
            True,
            "Date: MM/DD/YYYY → ISO-8601",
        ),
        (
            "France, Germany, Italy",
            ["Germany, Italy, France"],
            True,
            "List: order-independent comparison after sort",
        ),
        (
            "Paris",
            ["london", "berlin"],
            False,
            "String: no match in GT list",
        ),
    ]

    print("=" * 70)
    print("Denotation accuracy tests (WTQ)")
    print("=" * 70)
    all_passed = True
    for pred, gts, expected, desc in TEST_CASES:
        result = is_correct(pred, gts)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        norm_pred = _normalize_answer(pred)
        norm_gts = [_normalize_answer(g) for g in gts]
        print(f"[{status}] {desc}")
        print(f"       pred (raw)  : {pred!r}")
        print(f"       pred (norm) : {norm_pred!r}")
        print(f"       gts  (norm) : {norm_gts}")
        print(f"       match={result}  (expected={expected})")
        print()

    print("All tests passed." if all_passed else "Some tests FAILED.")
