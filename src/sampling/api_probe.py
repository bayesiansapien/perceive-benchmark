#!/usr/bin/env python3
"""
DocRouteBench Phase 2 — API Probe Runner

Runs GPT-5.2 (OpenAI) and Gemini 2.5 Flash (Vertex AI) in parallel on ~15K
candidate samples to produce a difficulty signal for dataset curation.

Cost: ~$23 for the full run. Resume-safe: never redoes completed work.

Usage:
    python -m src.sampling.api_probe                    # full run
    python -m src.sampling.api_probe --water-test       # 20-sample smoke test
    python -m src.sampling.api_probe --daily-limit 30   # lower spend cap
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── Project imports ───────────────────────────────────────────────────────────
# Allow running as a module from the project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.schema import load_jsonl, append_jsonl
from src.scoring.unified import is_correct as unified_is_correct

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("api_probe")

# ── Constants ─────────────────────────────────────────────────────────────────

GCP_PROJECT   = "speedy-aurora-193605"
GEMINI_LOCATION = "us-central1"

# Probe model definitions — instruct mode only (B0), no reasoning budget.
PROBE_MODELS: list[dict] = [
    {
        "model_id": "gpt52",
        "api_model_id": "gpt-5.2",
        "provider": "openai",
        # Per-token pricing in USD (from pilot_results/run_collection.py)
        "cost_per_1M_input":  1.00,   # $1.00 / 1M input tokens
        "cost_per_1M_output": 4.00,   # $4.00 / 1M output tokens
    },
    {
        "model_id": "gemini25flash",
        "api_model_id": "gemini-2.5-flash",
        "provider": "google",
        "cost_per_1M_input":  0.15,   # $0.15 / 1M input tokens
        "cost_per_1M_output": 0.60,   # $0.60 / 1M output tokens
    },
]

SYSTEM_PROMPT = """\
You are a document understanding expert. Answer the question about the document image, then analyze its visual composition and question complexity.

RESPOND IN THIS EXACT FORMAT (one item per line):

ANSWER: <concise — just the number, name, phrase, or short text>
DOC_TYPE: <form|receipt|invoice|academic_paper|report|letter|memo|slide|webpage|table_render|infographic|certificate|handwritten_note|scene_photo|other>
VISUAL: <comma-separated elements visible in the image from: table, bar_chart, line_chart, pie_chart, scatter_plot, diagram, flowchart, equation, figure, photograph, map, handwriting, signature, stamp, logo, form_field, checkbox, dense_text, multi_column, header_footer, bullet_list, none>
VDS: <level> | <evidence>
RDS: <level> | <evidence>
SES: <level> | <evidence>

VDS — Visual Dependency (does answering require understanding the image beyond raw text?):
  TEXT_ONLY         answer derivable from OCR text alone
  LAYOUT_DEPENDENT  spatial position, reading order, or alignment of text matters
  VISUAL_ELEMENT    must interpret a chart, figure, diagram, or table structure
  CROSS_MODAL       must fuse textual content with visual elements together

RDS — Reasoning Depth (how many cognitive steps from observation to answer?):
  DIRECT_LOOKUP     answer literally visible (a label, title, header, cell value)
  SINGLE_STEP       one extraction, match, or comparison operation
  MULTI_STEP        2+ chained inferences, simple calculation, or cross-referencing
  COMPLEX           arithmetic, domain knowledge, logical deduction, or synthesis

SES — Spatial Extent (how much of the document must be examined?):
  SINGLE_FIELD      one cell, word, or labeled value
  ONE_REGION        one paragraph, table, or figure
  FULL_PAGE         evidence scattered across the page
  MULTI_PAGE        evidence spans multiple pages or entire document"""

# ── VDS/RDS/SES label → numeric mappings ─────────────────────────────────────
_VDS_MAP = {"TEXT_ONLY": 1, "LAYOUT_DEPENDENT": 2, "VISUAL_ELEMENT": 3, "CROSS_MODAL": 4}
_RDS_MAP = {"DIRECT_LOOKUP": 1, "SINGLE_STEP": 2, "MULTI_STEP": 3, "COMPLEX": 4}
_SES_MAP = {"SINGLE_FIELD": 1, "ONE_REGION": 2, "FULL_PAGE": 3, "MULTI_PAGE": 4}

MAX_RETRIES   = 3
BASE_BACKOFF  = 2.0   # seconds; doubles on each retry
MAX_OUTPUT_TOKENS = 512  # increased for enriched response (~50-60 output tokens)
LOG_INTERVAL  = 100   # log progress every N completions

# ── Lazy API clients (initialised once per process) ───────────────────────────

_openai_client  = None
_google_client  = None
_client_lock    = threading.Lock()


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        with _client_lock:
            if _openai_client is None:
                import openai
                _openai_client = openai.OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
    return _openai_client


def _get_google_client():
    global _google_client
    if _google_client is None:
        with _client_lock:
            if _google_client is None:
                from google import genai
                _google_client = genai.Client(
                    vertexai=True,
                    project=GCP_PROJECT,
                    location=GEMINI_LOCATION,
                )
    return _google_client


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image_b64(image_path: str, project_root: Path = _PROJECT_ROOT) -> str:
    """
    Load an image from disk and return a base64-encoded PNG string.

    Tries the path as-is first (absolute), then relative to project root.
    Raises FileNotFoundError if neither works.
    """
    p = Path(image_path)
    if not p.is_absolute():
        p = project_root / image_path

    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path} (resolved: {p})")

    with open(p, "rb") as fh:
        raw = fh.read()

    # If already PNG, encode directly; otherwise convert via Pillow.
    if raw[:4] == b"\x89PNG":
        return base64.b64encode(raw).decode("ascii")

    try:
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        # Pillow not installed — just encode as-is and hope the API handles it.
        return base64.b64encode(raw).decode("ascii")


# ── Enriched response parsing ────────────────────────────────────────────────

def _parse_level_evidence(raw: str) -> tuple:
    """Parse 'LEVEL_NAME | evidence_tag' into (label, evidence)."""
    parts = raw.strip().split("|", 1)
    label = parts[0].strip().upper().replace(" ", "_")
    evidence = parts[1].strip() if len(parts) > 1 else ""
    return label, evidence


def _parse_enriched_response(raw_text: str) -> dict:
    """
    Parse the structured enriched probe response into fields.
    Falls back gracefully if the model doesn't follow the format exactly.
    """
    result = {
        "answer": raw_text.strip(),  # fallback: entire response is the answer
        "doc_type_detected": "other",
        "visual_elements": [],
        "vds_probe": 2, "rds_probe": 2, "ses_probe": 2,
        "vds_label": "", "rds_label": "", "ses_label": "",
        "vds_evidence": "", "rds_evidence": "", "ses_evidence": "",
    }

    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            if line.upper().startswith("ANSWER:"):
                result["answer"] = line[7:].strip()
            elif line.upper().startswith("DOC_TYPE:"):
                result["doc_type_detected"] = line[9:].strip().lower().replace(" ", "_")
            elif line.upper().startswith("VISUAL:"):
                elements = [e.strip().lower().replace(" ", "_") for e in line[7:].split(",")]
                result["visual_elements"] = [e for e in elements if e and e != "none"]
            elif line.upper().startswith("VDS:"):
                label, evidence = _parse_level_evidence(line[4:])
                result["vds_label"] = label
                result["vds_evidence"] = evidence
                result["vds_probe"] = _VDS_MAP.get(label, 2)
            elif line.upper().startswith("RDS:"):
                label, evidence = _parse_level_evidence(line[4:])
                result["rds_label"] = label
                result["rds_evidence"] = evidence
                result["rds_probe"] = _RDS_MAP.get(label, 2)
            elif line.upper().startswith("SES:"):
                label, evidence = _parse_level_evidence(line[4:])
                result["ses_label"] = label
                result["ses_evidence"] = evidence
                result["ses_probe"] = _SES_MAP.get(label, 2)
        except Exception:
            continue  # skip malformed lines gracefully

    # Derive boolean flags from visual elements
    ve = set(result["visual_elements"])
    result["has_table_detected"] = bool(ve & {"table"})
    result["has_chart_detected"] = bool(ve & {"bar_chart", "line_chart", "pie_chart", "scatter_plot", "chart"})
    result["has_figure_detected"] = bool(ve & {"figure", "photograph", "diagram", "map", "flowchart", "illustration"})
    result["has_handwriting_detected"] = bool(ve & {"handwriting", "signature"})
    result["visual_element_count"] = len(ve)
    result["answer_word_count"] = len(result["answer"].split())

    return result


# ── Per-model API callers ─────────────────────────────────────────────────────

def _call_openai(api_model_id: str, image_b64: str, query: str) -> dict:
    """
    Call an OpenAI vision model in instruct mode (no reasoning budget).
    Returns raw response dict with keys: answer, input_tokens, output_tokens, latency_ms.
    Raises on hard errors; caller handles retries.
    """
    client = _get_openai_client()

    t0 = time.monotonic()
    response = client.chat.completions.create(
        model=api_model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        # instruct mode: no reasoning_effort / temperature kwarg for gpt-5.2
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    answer = (response.choices[0].message.content or "").strip()
    usage  = response.usage
    return {
        "answer":        answer,
        "input_tokens":  usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "latency_ms":    latency_ms,
    }


def _call_google(api_model_id: str, image_b64: str, query: str) -> dict:
    """
    Call a Gemini model via Vertex AI in instruct mode (no thinking budget).
    Returns raw response dict.
    """
    from google.genai import types as genai_types

    client = _get_google_client()

    img_bytes = base64.b64decode(image_b64)
    contents  = [
        genai_types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        genai_types.Part.from_text(text=query),
    ]
    gen_config = genai_types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.0,
        # No thinking_config → instruct mode
    )

    t0 = time.monotonic()
    response = client.models.generate_content(
        model=api_model_id,
        contents=contents,
        config=gen_config,
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    # Extract only the non-thought parts as the answer.
    answer = ""
    for part in response.candidates[0].content.parts:
        if not (hasattr(part, "thought") and part.thought):
            answer += part.text or ""
    answer = answer.strip()

    usage         = response.usage_metadata
    input_tokens  = getattr(usage, "prompt_token_count",     0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0

    return {
        "answer":        answer,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "latency_ms":    latency_ms,
    }


# ── Retry wrapper ─────────────────────────────────────────────────────────────

def _call_with_retry(
    model_cfg: dict,
    image_b64: str,
    query: str,
) -> Optional[dict]:
    """
    Dispatch to the right provider with exponential backoff on 429 / transient
    errors. Returns None after MAX_RETRIES failures (caller marks as skipped).
    """
    provider     = model_cfg["provider"]
    api_model_id = model_cfg["api_model_id"]

    caller = {
        "openai": _call_openai,
        "google": _call_google,
    }[provider]

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return caller(api_model_id, image_b64, query)

        except Exception as exc:
            last_exc = exc
            exc_str  = str(exc)

            # Detect rate-limit signals from either provider.
            is_rate_limit = (
                "429" in exc_str
                or "rate" in exc_str.lower()
                or "quota" in exc_str.lower()
                or "resource_exhausted" in exc_str.lower()
            )

            backoff = BASE_BACKOFF * (2 ** (attempt - 1))
            if is_rate_limit:
                log.warning(
                    "Rate-limited (%s attempt %d/%d). Sleeping %.1fs.",
                    api_model_id, attempt, MAX_RETRIES, backoff,
                )
            else:
                log.warning(
                    "API error (%s attempt %d/%d): %s. Sleeping %.1fs.",
                    api_model_id, attempt, MAX_RETRIES, exc_str[:120], backoff,
                )
            time.sleep(backoff)

    log.error(
        "Giving up on %s after %d attempts. Last error: %s",
        api_model_id, MAX_RETRIES, str(last_exc)[:200],
    )
    return None


# ── Cost calculation ──────────────────────────────────────────────────────────

def _compute_cost(model_cfg: dict, input_tokens: int, output_tokens: int) -> float:
    cost = (
        input_tokens  / 1_000_000 * model_cfg["cost_per_1M_input"]
        + output_tokens / 1_000_000 * model_cfg["cost_per_1M_output"]
    )
    return round(cost, 8)


# ── Resume support ────────────────────────────────────────────────────────────

def _load_completed_keys(results_path: str) -> set[tuple[str, str]]:
    """
    Return set of (sample_id, model_id) pairs already written to the results
    file. Key uses model_id (short name like 'gpt52'), not the API model id.
    """
    completed: set[tuple[str, str]] = set()
    try:
        for record in load_jsonl(results_path):
            sid = record.get("sample_id", "")
            mid = record.get("model_id", "")
            if sid and mid:
                completed.add((sid, mid))
    except FileNotFoundError:
        pass
    return completed


# ── Atomic write with file lock ───────────────────────────────────────────────

_write_lock = threading.Lock()


def _atomic_append(results_path: str, record: dict) -> None:
    """Thread-safe append of one JSONL record."""
    with _write_lock:
        append_jsonl(results_path, record)


# ── Shared cost tracker ───────────────────────────────────────────────────────

class _CostTracker:
    """Thread-safe running cost + completion counter."""

    def __init__(self, daily_limit: float):
        self._lock         = threading.Lock()
        self._total_cost   = 0.0
        self._daily_limit  = daily_limit
        self._completed    = 0   # successful probe calls (both models)
        self._aborted      = False

    def add(self, cost: float) -> bool:
        """
        Add cost and increment counter.
        Returns True if we're still under budget, False if limit exceeded.
        """
        with self._lock:
            if self._aborted:
                return False
            self._total_cost += cost
            self._completed  += 1
            if self._total_cost >= self._daily_limit:
                self._aborted = True
                log.error(
                    "COST GUARD: running cost $%.4f reached daily limit $%.2f. "
                    "Aborting remaining calls.",
                    self._total_cost, self._daily_limit,
                )
                return False
            return True

    @property
    def total_cost(self) -> float:
        with self._lock:
            return self._total_cost

    @property
    def completed(self) -> int:
        with self._lock:
            return self._completed

    @property
    def aborted(self) -> bool:
        with self._lock:
            return self._aborted


# ── Core worker ───────────────────────────────────────────────────────────────

def _probe_one(
    *,
    sample: dict,
    model_cfg: dict,
    results_path: str,
    tracker: _CostTracker,
) -> Optional[dict]:
    """
    Run one probe call for a single (sample, model) pair.

    Returns the probe result dict on success, None on skip/error.
    Side-effect: atomically appends to results_path.
    """
    if tracker.aborted:
        return None

    sample_id = sample["sample_id"]
    model_id  = model_cfg["model_id"]

    # Load image
    image_path = sample.get("image_path", "")
    try:
        image_b64 = load_image_b64(image_path)
    except FileNotFoundError:
        log.warning("Image not found for %s (%s). Skipping.", sample_id, image_path)
        return None

    query = sample.get("query", sample.get("question", ""))

    # API call with retries
    raw = _call_with_retry(model_cfg, image_b64, query)
    if raw is None:
        # All retries exhausted — record a failure row so we don't retry forever
        # on a persistently broken sample.  is_correct=False, cost=0.
        result = {
            "sample_id":        sample_id,
            "model_id":         model_id,
            "is_correct":       False,
            "predicted_answer": "",
            "doc_type_detected": "other",
            "visual_elements": [],
            "vds_probe": 2, "vds_label": "", "vds_evidence": "",
            "rds_probe": 2, "rds_label": "", "rds_evidence": "",
            "ses_probe": 2, "ses_label": "", "ses_evidence": "",
            "has_table_detected": False, "has_chart_detected": False,
            "has_figure_detected": False, "has_handwriting_detected": False,
            "visual_element_count": 0, "answer_word_count": 0,
            "input_tokens":     0,
            "output_tokens":    0,
            "cost_usd":         0.0,
            "latency_ms":       0,
            "error":            "max_retries_exceeded",
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        }
        _atomic_append(results_path, result)
        return result

    predicted_answer = raw["answer"]

    # Parse enriched response
    enriched = _parse_enriched_response(predicted_answer)
    # Use the parsed answer (first line only) for scoring
    predicted_answer = enriched["answer"]

    # Score
    gt_answer = sample.get("gt_answer", "")
    gt_aliases: list[str] = sample.get("gt_answer_aliases", [])
    all_gt = [gt_answer] + gt_aliases if gt_answer else gt_aliases
    metric  = sample.get("correctness_metric", "anls")

    try:
        correct = unified_is_correct(
            predicted  = predicted_answer if predicted_answer else None,
            ground_truth = all_gt if all_gt else [""],
            metric     = metric,
            dataset    = sample.get("source_dataset", ""),
        )
    except Exception as exc:
        log.warning(
            "Scorer error for %s / %s: %s. Marking is_correct=False.",
            sample_id, metric, exc,
        )
        correct = False

    # Cost
    cost_usd = _compute_cost(model_cfg, raw["input_tokens"], raw["output_tokens"])

    result = {
        "sample_id":        sample_id,
        "model_id":         model_id,
        "is_correct":       bool(correct),
        "predicted_answer": predicted_answer,
        # Enriched probe fields
        "doc_type_detected":      enriched["doc_type_detected"],
        "visual_elements":        enriched["visual_elements"],
        "vds_probe":              enriched["vds_probe"],
        "vds_label":              enriched["vds_label"],
        "vds_evidence":           enriched["vds_evidence"],
        "rds_probe":              enriched["rds_probe"],
        "rds_label":              enriched["rds_label"],
        "rds_evidence":           enriched["rds_evidence"],
        "ses_probe":              enriched["ses_probe"],
        "ses_label":              enriched["ses_label"],
        "ses_evidence":           enriched["ses_evidence"],
        "has_table_detected":     enriched["has_table_detected"],
        "has_chart_detected":     enriched["has_chart_detected"],
        "has_figure_detected":    enriched["has_figure_detected"],
        "has_handwriting_detected": enriched["has_handwriting_detected"],
        "visual_element_count":   enriched["visual_element_count"],
        "answer_word_count":      enriched["answer_word_count"],
        # Token/cost/timing
        "input_tokens":     raw["input_tokens"],
        "output_tokens":    raw["output_tokens"],
        "cost_usd":         cost_usd,
        "latency_ms":       raw["latency_ms"],
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }

    _atomic_append(results_path, result)

    # Update tracker; this may flip aborted flag.
    tracker.add(cost_usd)

    return result


# ── ETA helper ────────────────────────────────────────────────────────────────

class _ETATracker:
    def __init__(self, total: int):
        self._total     = total
        self._start     = time.monotonic()
        self._lock      = threading.Lock()
        self._done      = 0

    def tick(self) -> str:
        with self._lock:
            self._done += 1
            done = self._done
        elapsed = time.monotonic() - self._start
        if done == 0:
            return "ETA: unknown"
        rate = done / elapsed          # completions per second
        remaining = max(0, self._total - done)
        eta_s = remaining / rate if rate > 0 else float("inf")
        if eta_s < 60:
            eta_str = f"{eta_s:.0f}s"
        elif eta_s < 3600:
            eta_str = f"{eta_s/60:.1f}min"
        else:
            eta_str = f"{eta_s/3600:.1f}h"
        return f"ETA: {eta_str}"


# ── Public API ────────────────────────────────────────────────────────────────

def run_api_probe(
    candidates_path: str = "data/processed/candidates_40k.jsonl",
    results_path:    str = "data/processed/probe_results.jsonl",
    checkpoint_dir:  str = "data/phase2_checkpoints",
    daily_spend_limit: float = 50.0,
    max_workers:     int = 4,
    n_samples:       Optional[int] = None,   # set to 20 for water-test
) -> dict:
    """
    Run both probe models (GPT-5.2 and Gemini 2.5 Flash) in parallel over the
    candidate sample pool.

    Args:
        candidates_path:   Path to the 15K candidate JSONL file.
        results_path:      Output JSONL (append-only). One line per (sample, model).
        checkpoint_dir:    Directory for any auxiliary checkpoint data.
        daily_spend_limit: Abort if running cost exceeds this value (USD).
        max_workers:       Number of parallel API calls *per model*.
        n_samples:         If set, only process the first N samples (water-test).

    Returns:
        Summary dict: {model_id: {"completed": int, "correct": int, "cost": float}}
    """
    candidates_path = str(_PROJECT_ROOT / candidates_path) if not Path(candidates_path).is_absolute() else candidates_path
    results_path    = str(_PROJECT_ROOT / results_path)    if not Path(results_path).is_absolute()    else results_path
    checkpoint_dir  = str(_PROJECT_ROOT / checkpoint_dir)  if not Path(checkpoint_dir).is_absolute()  else checkpoint_dir

    # Ensure output directories exist
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── Load candidates ───────────────────────────────────────────────────────
    log.info("Loading candidates from %s …", candidates_path)
    try:
        candidates = load_jsonl(candidates_path)
    except FileNotFoundError:
        log.error("Candidates file not found: %s", candidates_path)
        raise

    if n_samples is not None:
        candidates = candidates[:n_samples]
        log.info("Water-test mode: using first %d samples.", len(candidates))

    total_samples = len(candidates)
    log.info("Loaded %d candidate samples.", total_samples)

    # ── Resume: load already-completed (sample_id, model_id) pairs ───────────
    completed_keys = _load_completed_keys(results_path)
    log.info("Already completed: %d probe calls (from prior runs).", len(completed_keys))

    # Build work list: (sample, model_cfg) pairs not yet done
    work_items: list[tuple[dict, dict]] = []
    for sample in candidates:
        for model_cfg in PROBE_MODELS:
            key = (sample["sample_id"], model_cfg["model_id"])
            if key not in completed_keys:
                work_items.append((sample, model_cfg))

    total_work = len(work_items)
    already_done = total_samples * len(PROBE_MODELS) - total_work
    log.info(
        "Work queue: %d calls to run, %d already done (skipping).",
        total_work, already_done,
    )

    if total_work == 0:
        log.info("Nothing left to do — all samples already probed.")
        return get_probe_summary(results_path)

    # ── Shared state ──────────────────────────────────────────────────────────
    tracker     = _CostTracker(daily_spend_limit)
    eta_tracker = _ETATracker(total_work)

    # Per-model accumulators (for return summary, not persisted)
    model_stats: dict[str, dict] = {
        m["model_id"]: {"completed": 0, "correct": 0, "cost": 0.0}
        for m in PROBE_MODELS
    }
    stats_lock = threading.Lock()

    def _update_stats(model_id: str, result: Optional[dict]) -> None:
        if result is None:
            return
        with stats_lock:
            s = model_stats[model_id]
            s["completed"] += 1
            if result.get("is_correct"):
                s["correct"] += 1
            s["cost"] = round(s["cost"] + result.get("cost_usd", 0.0), 8)

    # ── Progress logger ───────────────────────────────────────────────────────
    progress_counter = [0]  # mutable int via list
    progress_lock    = threading.Lock()

    def _maybe_log_progress() -> None:
        with progress_lock:
            progress_counter[0] += 1
            count = progress_counter[0]
        if count % LOG_INTERVAL == 0 or count == total_work:
            eta_str = eta_tracker.tick()
            log.info(
                "Progress: %d / %d calls complete | cost so far $%.4f | %s",
                already_done + count,
                total_samples * len(PROBE_MODELS),
                tracker.total_cost,
                eta_str,
            )
        else:
            eta_tracker.tick()  # keep ETA numerics fresh even when not logging

    # ── Thread pool execution ─────────────────────────────────────────────────
    # We run both models fully in parallel.  max_workers applies per model,
    # so total concurrency is max_workers × num_models.  For typical API
    # rate limits, max_workers=4 (default) gives 8 concurrent requests which
    # is well within both OpenAI and Vertex AI quotas.

    total_concurrent = max_workers * len(PROBE_MODELS)

    log.info(
        "Starting parallel probe: %d workers × %d models = %d concurrent calls.",
        max_workers, len(PROBE_MODELS), total_concurrent,
    )
    log.info("Daily spend limit: $%.2f", daily_spend_limit)

    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=total_concurrent) as executor:
        futures = {
            executor.submit(
                _probe_one,
                sample=sample,
                model_cfg=model_cfg,
                results_path=results_path,
                tracker=tracker,
            ): (sample["sample_id"], model_cfg["model_id"])
            for sample, model_cfg in work_items
        }

        for future in as_completed(futures):
            sample_id, model_id = futures[future]
            try:
                result = future.result()
                _update_stats(model_id, result)
            except Exception as exc:
                log.error(
                    "Unhandled exception for %s / %s: %s",
                    sample_id, model_id, exc, exc_info=True,
                )

            _maybe_log_progress()

            if tracker.aborted:
                log.warning(
                    "Cost guard triggered. Cancelling remaining futures."
                )
                # Cancel any not-yet-started futures.
                for f in futures:
                    f.cancel()
                break

    elapsed = time.monotonic() - start_time
    log.info(
        "Probe run complete in %.1f min. Total cost: $%.4f",
        elapsed / 60,
        tracker.total_cost,
    )

    # Build and return summary
    summary = get_probe_summary(results_path)
    return summary


# ── Summary loader ────────────────────────────────────────────────────────────

def get_probe_summary(results_path: str) -> dict:
    """
    Load probe_results.jsonl and return completion statistics.

    Returns:
        {
            model_id: {
                "completed":  int,   # total rows (including error rows)
                "correct":    int,
                "accuracy":   float, # correct / completed
                "cost":       float, # USD
                "error_rows": int,   # rows where error field is set
            },
            "_total": {
                "completed": int,
                "cost":      float,
                "unique_samples": int,
            }
        }
    """
    results_path = str(_PROJECT_ROOT / results_path) if not Path(results_path).is_absolute() else results_path

    stats: dict[str, dict] = {}
    unique_samples: set[str] = set()

    try:
        records = load_jsonl(results_path)
    except FileNotFoundError:
        log.warning("Results file not found: %s", results_path)
        return {}

    for r in records:
        mid = r.get("model_id", "unknown")
        if mid not in stats:
            stats[mid] = {
                "completed":  0,
                "correct":    0,
                "cost":       0.0,
                "error_rows": 0,
            }
        s = stats[mid]
        s["completed"] += 1
        if r.get("is_correct"):
            s["correct"] += 1
        s["cost"] = round(s["cost"] + r.get("cost_usd", 0.0), 6)
        if r.get("error"):
            s["error_rows"] += 1
        unique_samples.add(r.get("sample_id", ""))

    for mid, s in stats.items():
        s["accuracy"] = round(s["correct"] / s["completed"], 4) if s["completed"] else 0.0

    total_cost = sum(s["cost"] for s in stats.values())
    stats["_total"] = {
        "completed":      sum(s["completed"] for s in stats.values()),
        "cost":           round(total_cost, 6),
        "unique_samples": len(unique_samples),
    }

    return stats


# ── Water-test ────────────────────────────────────────────────────────────────

def _validate_water_test(summary: dict) -> None:
    """Basic sanity checks after a water-test run."""
    errors = []

    for model_cfg in PROBE_MODELS:
        mid = model_cfg["model_id"]
        if mid not in summary:
            errors.append(f"Model '{mid}' missing from results.")
            continue
        s = summary[mid]
        if s["completed"] == 0:
            errors.append(f"Model '{mid}' has 0 completed calls.")
        if not isinstance(s.get("accuracy"), float):
            errors.append(f"Model '{mid}' accuracy is not a float: {s.get('accuracy')!r}")
        if s["cost"] < 0:
            errors.append(f"Model '{mid}' cost is negative.")

    if errors:
        log.error("Water-test FAILED validation:")
        for e in errors:
            log.error("  • %s", e)
    else:
        log.info("Water-test PASSED — both models returned results with valid schema.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DocRouteBench Phase 2 — API Probe Runner",
    )
    parser.add_argument(
        "--candidates",
        default="data/processed/candidates_40k.jsonl",
        help="Path to candidate samples JSONL (default: data/processed/candidates_40k.jsonl)",
    )
    parser.add_argument(
        "--results",
        default="data/processed/probe_results.jsonl",
        help="Path to output results JSONL (default: data/processed/probe_results.jsonl)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="data/phase2_checkpoints",
        help="Directory for auxiliary checkpoints",
    )
    parser.add_argument(
        "--daily-limit",
        type=float,
        default=50.0,
        help="Abort if running cost exceeds this USD value (default: 50.0)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel API calls per model (default: 4)",
    )
    parser.add_argument(
        "--water-test",
        action="store_true",
        help="Run on 20 samples only (smoke test, ~$0.01, ~2 min)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summary of existing results and exit (no API calls)",
    )
    args = parser.parse_args()

    if args.summary_only:
        summary = get_probe_summary(args.results)
        _print_summary(summary)
        return

    n_samples = 20 if args.water_test else None
    if args.water_test:
        log.info("=" * 60)
        log.info("WATER-TEST MODE: 20 samples × 2 models = 40 API calls")
        log.info("Expected cost: ~$0.01  |  Expected time: ~2 min")
        log.info("=" * 60)

    summary = run_api_probe(
        candidates_path   = args.candidates,
        results_path      = args.results,
        checkpoint_dir    = args.checkpoint_dir,
        daily_spend_limit = args.daily_limit,
        max_workers       = args.max_workers,
        n_samples         = n_samples,
    )

    _print_summary(summary)

    if args.water_test:
        _validate_water_test(summary)


def _print_summary(summary: dict) -> None:
    """Pretty-print the probe summary to stdout."""
    if not summary:
        print("No results found.")
        return

    print()
    print("=" * 60)
    print("PROBE SUMMARY")
    print("=" * 60)

    for key, stats in sorted(summary.items()):
        if key == "_total":
            continue
        print(
            f"  {key:<20s}  completed={stats['completed']:>6d}  "
            f"accuracy={stats['accuracy']:.1%}  "
            f"cost=${stats['cost']:.4f}  "
            f"errors={stats.get('error_rows', 0)}"
        )

    totals = summary.get("_total", {})
    print("-" * 60)
    print(
        f"  {'TOTAL':<20s}  completed={totals.get('completed', 0):>6d}  "
        f"unique_samples={totals.get('unique_samples', 0)}  "
        f"total_cost=${totals.get('cost', 0.0):.4f}"
    )
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
