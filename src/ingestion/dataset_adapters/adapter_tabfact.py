"""
DocRouteBench: TabFact Dataset Adapter

Task: T4, Table Fact Verification
Metric: exact_match
Source: table-benchmark/tabfact (HuggingFace)

TabFact is a text-based dataset (tables stored as HTML/list-of-lists).
We render each table as a PIL image using a simple grid layout before
passing it through the standard BaseAdapter image pipeline.
"""

from __future__ import annotations

import io
import logging
from typing import Iterator, Optional

from PIL import Image, ImageDraw, ImageFont

from src.ingestion.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

# ── Rendering constants ────────────────────────────────────────────────────────
CELL_PAD_X = 8          # horizontal padding inside each cell (pixels)
CELL_PAD_Y = 5          # vertical padding inside each cell (pixels)
FONT_SIZE = 13          # target font size
ROW_LIMIT = 10          # max rows to render (header + 9 data rows)
MIN_COL_WIDTH = 60      # minimum column width in pixels
MAX_COL_WIDTH = 220     # cap individual column width to keep image readable
BORDER_COLOR = (80, 80, 80)
HEADER_BG = (52, 101, 164)      # blue header background
HEADER_FG = (255, 255, 255)
ROW_BG_EVEN = (245, 248, 255)
ROW_BG_ODD = (255, 255, 255)
TEXT_COLOR = (30, 30, 30)


# ── Table parsing helpers ──────────────────────────────────────────────────────

def _parse_html_table(html: str) -> list[list[str]]:
    """Very lightweight HTML table parser: no external deps."""
    from html.parser import HTMLParser

    class _TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows: list[list[str]] = []
            self._current_row: list[str] = []
            self._current_cell: list[str] = []
            self._in_cell = False

        def handle_starttag(self, tag, attrs):
            if tag in ("tr",):
                self._current_row = []
            elif tag in ("td", "th"):
                self._current_cell = []
                self._in_cell = True

        def handle_endtag(self, tag):
            if tag in ("td", "th"):
                self._current_cell_text = "".join(self._current_cell).strip()
                self._current_row.append(self._current_cell_text)
                self._in_cell = False
            elif tag == "tr":
                if self._current_row:
                    self.rows.append(self._current_row)

        def handle_data(self, data):
            if self._in_cell:
                self._current_cell.append(data)

    parser = _TableParser()
    parser.feed(html)
    return parser.rows


def _parse_csv_table(csv_str: str) -> list[list[str]]:
    import csv, io as _io
    reader = csv.reader(_io.StringIO(csv_str))
    return [row for row in reader if row]


def _normalize_table(table_data) -> list[list[str]]:
    """
    Accept multiple table representations and return list-of-lists of strings.
    Supported inputs:
      - list[list[str|int|float]] , already parsed
      - str starting with '<'     , HTML
      - str with commas           , CSV
      - dict with 'header'/'rows' , some HF formats
    """
    if isinstance(table_data, list):
        # Already list-of-lists; stringify every cell
        return [[str(cell) for cell in row] for row in table_data]

    if isinstance(table_data, dict):
        # Some HF datasets expose {"header": [...], "rows": [[...], ...]}
        header = table_data.get("header", table_data.get("columns", []))
        rows = table_data.get("rows", table_data.get("data", []))
        result = [[str(c) for c in header]] if header else []
        result += [[str(c) for c in row] for row in rows]
        return result

    if isinstance(table_data, str):
        stripped = table_data.strip()
        if stripped.startswith("<"):
            return _parse_html_table(stripped)
        return _parse_csv_table(stripped)

    return []


# ── Image renderer ─────────────────────────────────────────────────────────────

def render_table_as_image(table_data) -> Image.Image:
    """
    Render a table (any supported format) as a PIL Image.

    The first row is treated as the header and given a coloured background.
    Tables with more than ROW_LIMIT rows are truncated; a note is added.
    """
    rows = _normalize_table(table_data)
    if not rows:
        # Fallback: blank image with error text
        img = Image.new("RGB", (400, 60), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 20), "[Table data unavailable]", fill=(180, 0, 0))
        return img

    truncated = False
    if len(rows) > ROW_LIMIT + 1:          # +1 for header
        rows = rows[: ROW_LIMIT + 1]
        truncated = True

    # ── Compute column widths ───────────────────────────────────────────────
    # Load a font; fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONT_SIZE)
        font_bold = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE
        )
    except OSError:
        font = ImageFont.load_default()
        font_bold = font

    # Use a temporary draw surface to measure text
    _probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def _text_w(text: str, bold: bool = False) -> int:
        f = font_bold if bold else font
        bbox = _probe.textbbox((0, 0), text, font=f)
        return bbox[2] - bbox[0]

    def _text_h(f) -> int:
        bbox = _probe.textbbox((0, 0), "Ag", font=f)
        return bbox[3] - bbox[1]

    n_cols = max(len(r) for r in rows)
    col_widths = [MIN_COL_WIDTH] * n_cols
    for r_idx, row in enumerate(rows):
        bold = r_idx == 0
        for c_idx, cell in enumerate(row):
            w = _text_w(str(cell), bold=bold) + 2 * CELL_PAD_X
            col_widths[c_idx] = min(MAX_COL_WIDTH, max(col_widths[c_idx], w))

    row_h = _text_h(font_bold) + 2 * CELL_PAD_Y
    total_w = sum(col_widths) + (n_cols + 1)        # +1 for borders
    footnote_h = (row_h + 4) if truncated else 0
    total_h = len(rows) * (row_h + 1) + 1 + footnote_h  # +1 per border

    img = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = 0
    for r_idx, row in enumerate(rows):
        is_header = r_idx == 0
        bg = HEADER_BG if is_header else (ROW_BG_EVEN if r_idx % 2 == 0 else ROW_BG_ODD)
        fg = HEADER_FG if is_header else TEXT_COLOR
        f = font_bold if is_header else font

        x = 0
        for c_idx in range(n_cols):
            cw = col_widths[c_idx]
            # Cell background
            draw.rectangle([x, y, x + cw, y + row_h], fill=bg)
            # Cell border (right + bottom)
            draw.line([(x + cw, y), (x + cw, y + row_h)], fill=BORDER_COLOR, width=1)
            # Cell text (truncate if too wide)
            cell_text = str(row[c_idx]) if c_idx < len(row) else ""
            # Truncate text to fit column
            while cell_text and _text_w(cell_text, bold=is_header) > cw - 2 * CELL_PAD_X:
                cell_text = cell_text[:-1]
            if len(cell_text) < len(str(row[c_idx]) if c_idx < len(row) else ""):
                cell_text = cell_text[:-1] + "…"
            draw.text((x + CELL_PAD_X, y + CELL_PAD_Y), cell_text, fill=fg, font=f)
            x += cw + 1  # +1 for border pixel

        # Horizontal border below row
        draw.line([(0, y + row_h), (total_w, y + row_h)], fill=BORDER_COLOR, width=1)
        y += row_h + 1

    # Left border
    draw.line([(0, 0), (0, y)], fill=BORDER_COLOR, width=1)

    if truncated:
        note = f"[Table truncated: showing first {ROW_LIMIT} data rows]"
        draw.text((CELL_PAD_X, y + 2), note, fill=(120, 80, 0), font=font)

    return img


# ── Adapter ────────────────────────────────────────────────────────────────────

class TabFactAdapter(BaseAdapter):
    """Adapter for the TabFact table fact-verification dataset."""

    dataset_name = "tabfact"
    task_type = "T4"
    metric = "exact_match"

    def __init__(self, max_samples: Optional[int] = None, seed: int = 42):
        super().__init__(max_samples=max_samples, seed=seed)

    def iter_samples(self) -> Iterator[dict]:
        from datasets import load_dataset  # lazy import

        logger.info("[tabfact] Loading dataset from HuggingFace…")
        # Try multiple known HF identifiers in order of preference
        ds = None
        for hf_id in ("table-benchmark/tabfact", "table-benchmark/tabfact", "EleutherAI/tab-fact"):
            try:
                ds = load_dataset(hf_id, split="test", )
                logger.info(f"[tabfact] Loaded '{hf_id}', {len(ds)} samples")
                break
            except Exception as exc:
                logger.debug(f"[tabfact] Could not load '{hf_id}': {exc}")

        if ds is None:
            raise RuntimeError(
                "[tabfact] Failed to load dataset. "
                "Tried: table-benchmark/tabfact, tab_fact, EleutherAI/tab-fact"
            )

        for idx, row in enumerate(ds):
            if self.max_samples and idx >= self.max_samples:
                break

            # ── Ground-truth answer ───────────────────────────────────────
            # table-benchmark/tabfact uses `answer` field with string values
            # "Entailed" / "Refuted"; fall back to integer `label` (1/0) for
            # other forks that store the answer as an int.
            raw_answer = row.get("answer") or row.get("label")
            if isinstance(raw_answer, str):
                gt_answer = raw_answer.strip().lower()  # "entailed" or "refuted"
            else:
                # Legacy integer encoding: 1 = entailed, 0 = refuted
                gt_answer = "entailed" if int(raw_answer or 0) == 1 else "refuted"
            gt_answer_aliases = [gt_answer]  # only the canonical form

            # ── Statement / claim ─────────────────────────────────────────
            # table-benchmark/tabfact stores the claim in the `text` field;
            # other forks may use `statement`, `claim`, or `sent`.
            statement: str = (
                row.get("question")
                or row.get("text")
                or row.get("statement")
                or row.get("claim")
                or row.get("sent")
                or ""
            ).strip()

            # ── Table data ────────────────────────────────────────────────
            table_raw = (
                row.get("table")
                or row.get("table_text")
                or row.get("context")
                or ""
            )

            # ── Render table → PIL Image ──────────────────────────────────
            try:
                pil_image = render_table_as_image(table_raw)
            except Exception as exc:
                logger.warning(f"[tabfact] Render failed for idx={idx}: {exc}")
                pil_image = Image.new("RGB", (400, 60), color=(255, 255, 255))

            sample_id = f"tabfact_test_{idx:06d}"
            query = (
                f"Does the following table support or refute this claim: "
                f"'{statement}'? Answer with 'entailed' or 'refuted'."
            )

            yield {
                "sample_id": sample_id,
                "query": query,
                "gt_answer": gt_answer,
                "gt_answer_aliases": gt_answer_aliases,
                "image": pil_image,
                "task_type": self.task_type,
                "correctness_metric": self.metric,
                "source_split": "test",
                "doc_type": "table",
                "has_table": True,
                "has_chart": False,
                "has_figure": False,
                "has_handwriting": False,
                "num_pages": 1,
            }


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"TabFact smoke test: loading {n} samples")

    adapter = TabFactAdapter(max_samples=n)

    for sample in adapter.iter_samples():
        img: Image.Image = sample["image"]
        print(
            f"  {sample['sample_id']} | "
            f"gt={sample['gt_answer']!r} | "
            f"img_size={img.size} | "
            f"query={sample['query'][:80]!r}…"
        )

    print("\nRunning full pipeline (save images + write JSONL)…")
    count = adapter.run()
    print(f"Done: {count} samples written.")
