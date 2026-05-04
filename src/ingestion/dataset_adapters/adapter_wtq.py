"""
DocRouteBench: WikiTableQuestions (WTQ) Dataset Adapter

Task: T4, Compositional Table Reasoning
Metric: denotation
Source: TableSenseAI/WikiTableQuestions (HuggingFace)

WTQ is a text-based dataset (tables as HTML/CSV).
We render each table as a PIL image using a simple grid layout before
passing it through the standard BaseAdapter image pipeline.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from PIL import Image, ImageDraw, ImageFont

from src.ingestion.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

# ── Rendering constants ────────────────────────────────────────────────────────
CELL_PAD_X = 8
CELL_PAD_Y = 5
FONT_SIZE = 13
ROW_LIMIT = 10          # header + up to ROW_LIMIT data rows
MIN_COL_WIDTH = 60
MAX_COL_WIDTH = 220
BORDER_COLOR = (80, 80, 80)
HEADER_BG = (34, 119, 75)       # green header to distinguish from TabFact
HEADER_FG = (255, 255, 255)
ROW_BG_EVEN = (240, 255, 245)
ROW_BG_ODD = (255, 255, 255)
TEXT_COLOR = (30, 30, 30)


# ── Table parsing helpers ──────────────────────────────────────────────────────

def _parse_html_table(html: str) -> list[list[str]]:
    """Lightweight HTML table parser with no external dependencies."""
    from html.parser import HTMLParser

    class _TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows: list[list[str]] = []
            self._current_row: list[str] = []
            self._current_cell: list[str] = []
            self._in_cell = False

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self._current_row = []
            elif tag in ("td", "th"):
                self._current_cell = []
                self._in_cell = True

        def handle_endtag(self, tag):
            if tag in ("td", "th"):
                self._current_row.append("".join(self._current_cell).strip())
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
      - str containing commas     , CSV
      - dict with 'header'/'rows' , common HF dict format
    """
    if isinstance(table_data, list):
        return [[str(cell) for cell in row] for row in table_data]

    if isinstance(table_data, dict):
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

    The first row is treated as the header and rendered with a coloured
    background. Tables exceeding ROW_LIMIT rows are truncated and annotated.
    """
    rows = _normalize_table(table_data)
    if not rows:
        img = Image.new("RGB", (400, 60), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 20), "[Table data unavailable]", fill=(180, 0, 0))
        return img

    truncated = False
    if len(rows) > ROW_LIMIT + 1:      # +1 for header row
        rows = rows[: ROW_LIMIT + 1]
        truncated = True

    # ── Font loading ────────────────────────────────────────────────────────
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONT_SIZE
        )
        font_bold = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE
        )
    except OSError:
        font = ImageFont.load_default()
        font_bold = font

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
    total_w = sum(col_widths) + (n_cols + 1)
    footnote_h = (row_h + 4) if truncated else 0
    total_h = len(rows) * (row_h + 1) + 1 + footnote_h

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
            draw.rectangle([x, y, x + cw, y + row_h], fill=bg)
            draw.line([(x + cw, y), (x + cw, y + row_h)], fill=BORDER_COLOR, width=1)

            cell_text = str(row[c_idx]) if c_idx < len(row) else ""
            # Truncate text to fit within column width
            original = cell_text
            while cell_text and _text_w(cell_text, bold=is_header) > cw - 2 * CELL_PAD_X:
                cell_text = cell_text[:-1]
            if len(cell_text) < len(original):
                cell_text = cell_text[:-1] + "…"

            draw.text((x + CELL_PAD_X, y + CELL_PAD_Y), cell_text, fill=fg, font=f)
            x += cw + 1

        draw.line([(0, y + row_h), (total_w, y + row_h)], fill=BORDER_COLOR, width=1)
        y += row_h + 1

    # Left border
    draw.line([(0, 0), (0, y)], fill=BORDER_COLOR, width=1)

    if truncated:
        note = f"[Table truncated: showing first {ROW_LIMIT} data rows]"
        draw.text((CELL_PAD_X, y + 2), note, fill=(120, 80, 0), font=font)

    return img


# ── Answer normalisation ───────────────────────────────────────────────────────

def _join_answers(answers: list) -> str:
    """
    Flatten a (possibly nested) answers structure to a single string.
    WTQ answers are typically a list of strings.
    Multiple answers are joined with ' | ' to preserve all alternatives.
    """
    flat: list[str] = []
    for a in answers:
        if isinstance(a, (list, tuple)):
            flat.extend(str(x).strip() for x in a if str(x).strip())
        else:
            s = str(a).strip()
            if s:
                flat.append(s)
    if not flat:
        return ""
    return flat[0] if len(flat) == 1 else " | ".join(flat)


def _answer_aliases(answers: list) -> list[str]:
    """Return a flat list of all answer strings for alias matching."""
    flat: list[str] = []
    for a in answers:
        if isinstance(a, (list, tuple)):
            flat.extend(str(x).strip() for x in a if str(x).strip())
        else:
            s = str(a).strip()
            if s:
                flat.append(s)
    return flat


# ── Adapter ────────────────────────────────────────────────────────────────────

class WTQAdapter(BaseAdapter):
    """Adapter for the WikiTableQuestions (WTQ) compositional QA dataset."""

    dataset_name = "wtq"
    task_type = "T4"
    metric = "denotation"

    def __init__(self, max_samples: Optional[int] = None, seed: int = 42):
        super().__init__(max_samples=max_samples, seed=seed)

    def iter_samples(self) -> Iterator[dict]:
        from datasets import load_dataset  # lazy import

        logger.info("[wtq] Loading dataset from HuggingFace…")
        ds = None
        for hf_id in (
            "TableSenseAI/WikiTableQuestions",
            "danwakeem/wikitablequestions-wtq",
            "msr_wiki_tables",
        ):
            try:
                ds = load_dataset(hf_id, split="test", )
                logger.info(f"[wtq] Loaded '{hf_id}', {len(ds)} samples")
                break
            except Exception as exc:
                logger.debug(f"[wtq] Could not load '{hf_id}': {exc}")

        if ds is None:
            raise RuntimeError(
                "[wtq] Failed to load dataset. "
                "Tried: TableSenseAI/WikiTableQuestions, wikitablequestions"
            )

        for idx, row in enumerate(ds):
            if self.max_samples and idx >= self.max_samples:
                break

            # ── Question ──────────────────────────────────────────────────
            question: str = (
                row.get("question")
                or row.get("utterance")
                or ""
            ).strip()

            # ── Answers ───────────────────────────────────────────────────
            raw_answers = (
                row.get("answers")
                or row.get("answer")
                or row.get("target_value")
                or []
            )
            # Normalise scalar → list
            if isinstance(raw_answers, str):
                raw_answers = [raw_answers]
            elif not isinstance(raw_answers, list):
                raw_answers = [str(raw_answers)]

            gt_answer = _join_answers(raw_answers)
            gt_answer_aliases = _answer_aliases(raw_answers)

            # ── Table ─────────────────────────────────────────────────────
            table_raw = (
                row.get("table")
                or row.get("table_text")
                or row.get("context")
                or row.get("table_html")
                or ""
            )

            # ── Render table → PIL Image ──────────────────────────────────
            try:
                pil_image = render_table_as_image(table_raw)
            except Exception as exc:
                logger.warning(f"[wtq] Render failed for idx={idx}: {exc}")
                pil_image = Image.new("RGB", (400, 60), color=(255, 255, 255))

            sample_id = f"wtq_test_{idx:06d}"

            yield {
                "sample_id": sample_id,
                "query": question,
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
    print(f"WTQ smoke test: loading {n} samples")

    adapter = WTQAdapter(max_samples=n)

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
