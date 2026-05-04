"""
TEDS (Tree Edit Distance Similarity) scoring for table structure evaluation.

Used by: FinTabNet
Compares a predicted HTML table string against a ground-truth HTML table string
by converting each into a tree structure and computing normalized tree edit
distance.

    TEDS = 1 - TED(pred_tree, gt_tree) / max(|pred_tree|, |gt_tree|)

is_correct threshold: TEDS >= 0.7

Fallback behaviour:
- If the model outputs a Markdown table it is converted to HTML first.
- If HTML parsing fails entirely, token-level F1 on the raw string is returned.
- If beautifulsoup4 is not installed, a simple row/column heuristic is used.
"""

from __future__ import annotations

import re
import string
from typing import Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Optional dependency: beautifulsoup4
# ---------------------------------------------------------------------------
try:
    from bs4 import BeautifulSoup, Tag

    _BS4_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BS4_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tree representation
# ---------------------------------------------------------------------------

class _Node:
    """Minimal tree node used for edit-distance computation."""

    def __init__(self, label: str, children: Optional[List["_Node"]] = None) -> None:
        self.label = label
        self.children: List[_Node] = children if children is not None else []

    def __repr__(self) -> str:  # helpful when debugging
        return f"_Node({self.label!r}, {self.children!r})"

    def size(self) -> int:
        """Return total number of nodes in this subtree (inclusive)."""
        return 1 + sum(c.size() for c in self.children)


# ---------------------------------------------------------------------------
# Markdown → HTML conversion
# ---------------------------------------------------------------------------

def _markdown_table_to_html(text: str) -> str:
    """Convert a GitHub-Flavoured Markdown table to a minimal HTML table.

    Returns the original *text* unchanged if no Markdown table is detected.
    """
    lines = [l.rstrip() for l in text.strip().splitlines()]
    # A Markdown table must have at least a header row and a separator row.
    table_lines: List[str] = []
    in_table = False
    for line in lines:
        if re.match(r"^\s*\|", line) or re.match(r"^[^\|]*\|", line):
            table_lines.append(line)
            in_table = True
        elif in_table:
            break  # table ended

    if len(table_lines) < 2:
        return text  # not a recognisable Markdown table

    # Separator row looks like |---|---|
    sep_idx: Optional[int] = None
    for i, l in enumerate(table_lines):
        if re.match(r"^\s*\|?\s*[-:]+[-| :]*$", l):
            sep_idx = i
            break

    if sep_idx is None:
        return text

    def _split_row(row: str) -> List[str]:
        parts = row.strip().strip("|").split("|")
        return [p.strip() for p in parts]

    html_parts = ["<table>"]

    # Header
    header_rows = table_lines[:sep_idx]
    if header_rows:
        html_parts.append("<thead>")
        for row in header_rows:
            cells = _split_row(row)
            html_parts.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
        html_parts.append("</thead>")

    # Body
    body_rows = table_lines[sep_idx + 1 :]
    if body_rows:
        html_parts.append("<tbody>")
        for row in body_rows:
            cells = _split_row(row)
            html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        html_parts.append("</tbody>")

    html_parts.append("</table>")
    return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# HTML → tree (bs4 path)
# ---------------------------------------------------------------------------

def _bs4_to_tree(tag: Any) -> _Node:
    """Recursively convert a bs4 Tag/NavigableString into a _Node tree."""
    from bs4 import NavigableString, Tag as BS4Tag  # type: ignore[attr-defined]

    if isinstance(tag, NavigableString):
        text = str(tag).strip()
        return _Node(text) if text else _Node("")

    # It is a Tag
    children: List[_Node] = []
    for child in tag.children:
        child_node = _bs4_to_tree(child)
        # Drop empty leaf nodes that add no information
        if child_node.label != "" or child_node.children:
            children.append(child_node)

    return _Node(tag.name, children)


def _html_to_tree_bs4(html: str) -> Optional[_Node]:
    """Parse *html* with bs4 and return the <table> subtree, or None on failure."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if table is None:
            # Try treating the whole fragment as a tree
            body = soup.find("body") or soup
            return _bs4_to_tree(body)
        return _bs4_to_tree(table)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Tree Edit Distance (Zhang-Shasha, simplified O(n²) approximation)
# ---------------------------------------------------------------------------

def _ted(node1: _Node, node2: _Node) -> int:
    """Compute tree edit distance between two trees (insert/delete/relabel cost 1).

    Uses a straightforward recursive DP that is correct for small tables
    (which is the typical use-case here).  For very large trees the memoised
    recursion keeps memory bounded.
    """
    memo: dict[Tuple[int, int], int] = {}

    def _dist(a: Optional[_Node], b: Optional[_Node]) -> int:
        if a is None and b is None:
            return 0
        if a is None:
            # Insert all nodes in b's subtree
            assert b is not None
            return b.size()
        if b is None:
            # Delete all nodes in a's subtree
            return a.size()

        key = (id(a), id(b))
        if key in memo:
            return memo[key]

        # Cost of relabelling root
        relabel_cost = 0 if a.label == b.label else 1

        # Align children greedily (DP on sequence pairs)
        ac, bc = a.children, b.children
        m, n = len(ac), len(bc)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + ac[i - 1].size()
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + bc[j - 1].size()
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = min(
                    dp[i - 1][j] + ac[i - 1].size(),   # delete subtree ac[i-1]
                    dp[i][j - 1] + bc[j - 1].size(),   # insert subtree bc[j-1]
                    dp[i - 1][j - 1] + _dist(ac[i - 1], bc[j - 1]),
                )

        result = relabel_cost + dp[m][n]
        memo[key] = result
        return result

    return _dist(node1, node2)


# ---------------------------------------------------------------------------
# Fallback: token F1 on raw strings
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lower-case, strip punctuation, split on whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def _token_f1(pred: str, gt: str) -> float:
    """Compute token-level F1 between two strings."""
    pred_tokens = _tokenize(pred)
    gt_tokens = _tokenize(gt)
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    gt_counts: dict[str, int] = {}
    for t in gt_tokens:
        gt_counts[t] = gt_counts.get(t, 0) + 1

    common = sum(min(pred_counts.get(t, 0), gt_counts[t]) for t in gt_counts)
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Heuristic fallback when bs4 is unavailable
# ---------------------------------------------------------------------------

def _heuristic_teds(pred: str, gt: str) -> float:
    """Very rough TEDS approximation based on row/column counts.

    Only used when beautifulsoup4 is not installed.
    """
    def _count_rows(html: str) -> int:
        return len(re.findall(r"<tr[\s>]", html, re.IGNORECASE))

    def _count_cells(html: str) -> int:
        return len(re.findall(r"<t[dh][\s>]", html, re.IGNORECASE))

    pred_rows = max(_count_rows(pred), 1)
    gt_rows = max(_count_rows(gt), 1)
    pred_cells = max(_count_cells(pred), 1)
    gt_cells = max(_count_cells(gt), 1)

    row_sim = min(pred_rows, gt_rows) / max(pred_rows, gt_rows)
    cell_sim = min(pred_cells, gt_cells) / max(pred_cells, gt_cells)
    return 0.5 * row_sim + 0.5 * cell_sim


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_teds(pred: str, gt: str) -> float:
    """Compute TEDS between a predicted and ground-truth table string.

    Parameters
    ----------
    pred:
        Model output: may be an HTML table, a Markdown table, or plain text.
    gt:
        Ground-truth HTML table string.

    Returns
    -------
    float
        TEDS score in [0.0, 1.0].  Higher is better.
    """
    # Normalise whitespace
    pred = pred.strip()
    gt = gt.strip()

    if pred == gt:
        return 1.0

    # Convert Markdown to HTML if needed
    if "|" in pred and "<table" not in pred.lower():
        pred = _markdown_table_to_html(pred)

    # ---- bs4 path ----
    if _BS4_AVAILABLE:
        pred_tree = _html_to_tree_bs4(pred)
        gt_tree = _html_to_tree_bs4(gt)

        if pred_tree is not None and gt_tree is not None:
            pred_size = pred_tree.size()
            gt_size = gt_tree.size()
            denom = max(pred_size, gt_size)
            if denom == 0:
                return 1.0
            distance = _ted(pred_tree, gt_tree)
            return max(0.0, 1.0 - distance / denom)

        # HTML parsing failed: fall back to token F1
        return _token_f1(pred, gt)

    # ---- no bs4: try heuristic if HTML-like, else token F1 ----
    if "<table" in pred.lower() or "<tr" in pred.lower():
        return _heuristic_teds(pred, gt)

    return _token_f1(pred, gt)


def is_correct(pred: str, gt: str, threshold: float = 0.7) -> bool:
    """Return True if TEDS(pred, gt) >= *threshold* (default 0.7)."""
    return compute_teds(pred, gt) >= threshold


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests: List[Tuple[str, str, str]] = [
        (
            "identical tables",
            "<table><tr><td>A</td><td>B</td></tr></table>",
            "<table><tr><td>A</td><td>B</td></tr></table>",
        ),
        (
            "one cell different",
            "<table><tr><td>A</td><td>X</td></tr></table>",
            "<table><tr><td>A</td><td>B</td></tr></table>",
        ),
        (
            "extra row in prediction",
            (
                "<table><tr><td>A</td><td>B</td></tr>"
                "<tr><td>C</td><td>D</td></tr>"
                "<tr><td>E</td><td>F</td></tr></table>"
            ),
            "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>",
        ),
        (
            "markdown table input",
            "| Name | Score |\n|------|-------|\n| Alice | 95 |\n| Bob | 87 |",
            "<table><thead><tr><th>Name</th><th>Score</th></tr></thead>"
            "<tbody><tr><td>Alice</td><td>95</td></tr>"
            "<tr><td>Bob</td><td>87</td></tr></tbody></table>",
        ),
        (
            "completely wrong prediction (plain text)",
            "There is no table here.",
            "<table><tr><td>A</td><td>B</td></tr></table>",
        ),
        (
            "empty prediction vs non-empty gt",
            "",
            "<table><tr><td>A</td></tr></table>",
        ),
    ]

    print(f"{'Test':<35} {'TEDS':>6}  {'Correct?':>8}")
    print("-" * 55)
    for name, pred_html, gt_html in tests:
        score = compute_teds(pred_html, gt_html)
        correct = is_correct(pred_html, gt_html)
        print(f"{name:<35} {score:>6.3f}  {str(correct):>8}")
