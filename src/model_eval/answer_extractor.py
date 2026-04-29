"""
DocRouteBench — Answer Extractor

Clean raw model output into a scorable answer string.
Critical: wrong extraction = wrong GT labels.
"""
from __future__ import annotations

import json
import re

# Common prefixes models add before the actual answer
_PREFIX_PATTERNS = [
    re.compile(r"^(?:the answer is|answer is|answer:|a:|ans:|result is|result:)\s*", re.I),
    re.compile(r"^(?:based on (?:the )?(?:image|document|chart|table),?\s*(?:the answer is)?)\s*", re.I),
    re.compile(r"^(?:looking at (?:the )?(?:image|document|chart),?\s*(?:the answer is)?)\s*", re.I),
    re.compile(r"^(?:according to (?:the )?(?:image|document|chart|table),?\s*(?:the answer is)?)\s*", re.I),
]

# Conjunctions/continuations that signal an explanation follows
_EXPLANATION_STARTERS = re.compile(
    r"^[,.\s]+(?:which|that|because|as\s|since|so\s|but\s|and\s|or\s|it\s|the\s|this\s|"
    r"there\s|a\s|an\s|indicating|showing|according|however|although|while|where|when|"
    r"specifically|namely|meaning|i\.e\.|e\.g\.)",
    re.I,
)


def _truncate_at_explanation(text: str) -> str:
    """
    If the answer starts with a short answer phrase followed by an explanation,
    extract just the short answer.

    Examples:
      "Yes, it is part of the SEMIOTEXT(E) series"  → "Yes"
      "No, half dollar."                             → "No"
      "13-15 years, approximately"                  → "13-15 years"
      "Paris, the capital of France"                → "Paris"
      "The document is a form, showing fields"      → kept as-is (too long before comma)
    """
    # Split on first comma or period
    m = re.match(r"^([^,\.;!?]+)[,\.;](.*)$", text, re.DOTALL)
    if not m:
        return text

    before = m.group(1).strip()
    after  = m.group(2).strip()

    # Only truncate for single-word binary/boolean answers followed by explanation.
    # "Yes, it is part of..." → "Yes"
    # "No, the document shows..." → "No"
    # Do NOT truncate numeric or multi-word answers — commas may be part of the answer
    # (e.g. "table, figure", "1, 2, 3", "[0.08, 0.82]")
    _BINARY_ANSWERS = {"yes", "no", "true", "false", "entailed", "refuted"}
    if before.lower().strip() in _BINARY_ANSWERS:
        return before

    return text


def extract_answer(raw_output: str, model_id: str = "") -> str:
    """
    Extract clean answer from raw model output.
    Returns empty string on failure (will score as False).

    Handles:
    - <think>...</think> blocks from Qwen/reasoning models
    - Common answer prefixes ("The answer is:", "Answer:", etc.)
    - JSON outputs ({"answer": "..."})
    - Multi-line: take first non-empty line
    - Surrounding quotes
    - "Short answer, long explanation" → extract short answer only
    """
    if not raw_output or not raw_output.strip():
        return ""

    text = raw_output.strip()

    # Strip any remaining <think>...</think> blocks (safety net)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown bold/italic formatting (**answer** or *answer*)
    text = re.sub(r"\*{1,3}([^*\n]+)\*{1,3}", r"\1", text).strip()

    if not text:
        return ""

    # Try JSON parse (some models return {"answer": "..."})
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            for key in ("answer", "result", "value", "output", "text"):
                if key in parsed:
                    text = str(parsed[key])
                    break
        except (json.JSONDecodeError, TypeError):
            pass

    # Take first non-empty line
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return ""
    text = lines[0]

    # Strip common prefixes
    for pattern in _PREFIX_PATTERNS:
        text = pattern.sub("", text).strip()

    # Truncate "short answer, long explanation" → keep short answer only
    text = _truncate_at_explanation(text)

    # Strip surrounding quotes
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()

    return text
