#!/usr/bin/env python3
"""
DocRouteBench Phase 3 — API Evaluation Harness

Evaluates all API-based model configs on benchmark samples.
Mirrors the structure of src/sampling/api_probe.py.

Usage (via run_phase3.py):
    run_evaluation(benchmark_path, model_pool_path, output_path, split="anchor")
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.schema import load_jsonl, append_jsonl
from src.scoring.unified import is_correct as unified_is_correct
from src.sampling.api_probe import load_image_b64
from src.model_eval.answer_extractor import extract_answer
from src.model_eval.cost_calculator import compute_cost
from src.model_eval.model_adapters.openai_adapter import OpenAIAdapter
from src.model_eval.model_adapters.anthropic_adapter import AnthropicAdapter
from src.model_eval.model_adapters.google_adapter import GoogleAdapter
from src.model_eval.model_adapters.openrouter_adapter import OpenRouterAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("eval_harness")

LOG_INTERVAL = 100
_write_lock = threading.Lock()

PROVIDER_TO_ADAPTER = {
    "openai":      OpenAIAdapter,
    "anthropic":   AnthropicAdapter,
    "google":      GoogleAdapter,
    "openrouter":  OpenRouterAdapter,
}

_BUDGET_TOKENS = {"B0": 0, "B1": 1024, "B2": 4096, "B3": 16384}


# ── Cost tracker ──────────────────────────────────────────────────────────────

class _CostTracker:
    def __init__(self, daily_limit: float):
        self._lock = threading.Lock()
        self._total = 0.0
        self._limit = daily_limit
        self._count = 0
        self._aborted = False

    def add(self, cost: float) -> bool:
        with self._lock:
            self._total += cost
            self._count += 1
            if self._total >= self._limit:
                self._aborted = True
            return not self._aborted

    @property
    def total(self) -> float:
        return self._total

    @property
    def count(self) -> int:
        return self._count

    @property
    def aborted(self) -> bool:
        return self._aborted


# ── ETA tracker ───────────────────────────────────────────────────────────────

class _ETATracker:
    def __init__(self, total: int):
        self._total = total
        self._done = 0
        self._start = time.monotonic()
        self._lock = threading.Lock()

    @property
    def total(self) -> int:
        return self._total

    def tick(self) -> Tuple[int, str]:
        with self._lock:
            self._done += 1
            elapsed = time.monotonic() - self._start
            rate = self._done / elapsed if elapsed > 0 else 0
            remaining = (self._total - self._done) / rate if rate > 0 else float("inf")
            eta = f"{remaining/60:.1f}min" if remaining < 3600 else f"{remaining/3600:.1f}h"
            return self._done, eta


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_completed_keys(results_path: str) -> Set[Tuple[str, str]]:
    """Return set of (sample_id, config_id) already written."""
    completed: Set[Tuple[str, str]] = set()
    try:
        for r in load_jsonl(results_path):
            sid = r.get("sample_id", "")
            cid = r.get("config_id", "")
            if sid and cid:
                completed.add((sid, cid))
    except FileNotFoundError:
        pass
    return completed


def _atomic_append(results_path: str, record: dict) -> None:
    with _write_lock:
        append_jsonl(results_path, record)


# ── Sample / config loading ───────────────────────────────────────────────────

def _filter_samples(samples: List[dict], split: str) -> List[dict]:
    if split == "anchor":
        return [s for s in samples if s.get("in_anchor_set")]
    elif split == "validation":
        return [s for s in samples if s.get("in_validation_set")]
    elif split == "remaining":
        return [s for s in samples
                if not s.get("in_anchor_set") and not s.get("in_validation_set")]
    elif split == "all":
        return samples
    raise ValueError(f"Unknown split: {split!r}. Use anchor/validation/remaining/all")


def _load_api_configs(
    model_pool_path: str,
    tier_filter: Optional[List[str]] = None,
) -> List[Tuple[str, dict, str]]:
    """Return list of (yaml_key, model_cfg, budget_level) for all API configs."""
    with open(model_pool_path) as f:
        cfg = yaml.safe_load(f)
    configs = []
    for yaml_key, m in cfg["models"].items():
        if m.get("provider") == "self_hosted":
            continue
        tier = yaml_key[0].upper()
        if tier_filter and tier not in tier_filter:
            continue
        for budget in m.get("budgets", []):
            configs.append((yaml_key, m, budget))
    return configs


# ── Per-item worker ───────────────────────────────────────────────────────────

def _eval_one(
    sample: dict,
    yaml_key: str,
    model_cfg: dict,
    budget_level: str,
    tracker: _CostTracker,
    eta: _ETATracker,
    results_path: str,
) -> Optional[dict]:
    """Evaluate one (sample, config) pair. Returns result dict or None on skip."""
    if tracker.aborted:
        return None

    config_id = f"{yaml_key}_{budget_level}"
    sample_id = sample["sample_id"]

    # Load image
    try:
        image_b64 = load_image_b64(sample.get("image_path", ""))
    except FileNotFoundError:
        log.warning("Image not found for %s — skipping.", sample_id)
        return None

    # Get adapter
    provider = model_cfg.get("provider", "")
    AdapterClass = PROVIDER_TO_ADAPTER.get(provider)
    if AdapterClass is None:
        log.warning("No adapter for provider %r — skipping %s", provider, config_id)
        return None

    adapter = AdapterClass(yaml_key, model_cfg, budget_level)

    # Call model with retry handled inside adapter
    raw_answer = ""
    raw = {}
    error_str = None
    try:
        raw = adapter.call(image_b64, sample.get("query", ""))
        raw_answer = raw.get("answer", "")
    except Exception as exc:
        error_str = str(exc)[:200]
        log.warning("Inference failed %s / %s: %s", sample_id, config_id, error_str)

    # Score
    predicted = extract_answer(raw_answer, model_cfg.get("model_id", ""))
    correct = False
    if error_str is None and predicted:
        gt = sample.get("gt_answer", "")
        aliases = sample.get("gt_answer_aliases", [])
        all_gt = [gt] + aliases if gt else aliases
        metric = sample.get("correctness_metric", "anls")
        try:
            correct = unified_is_correct(
                predicted=predicted,
                ground_truth=all_gt if all_gt else [""],
                metric=metric,
                dataset=sample.get("source_dataset", ""),
            )
        except Exception as exc:
            log.debug("Scorer error %s: %s", sample_id, exc)

    # Cost
    cost = compute_cost(
        model_cfg,
        raw.get("input_tokens", 0),
        raw.get("output_tokens", 0),
        raw.get("reasoning_tokens", 0),
    )

    result = {
        "sample_id":       sample_id,
        "config_id":       config_id,
        "yaml_key":        yaml_key,
        "model_name":      model_cfg.get("name", ""),
        "provider":        provider,
        "tier":            yaml_key[0].upper(),
        "budget_level":    budget_level,
        "budget_tokens":   _BUDGET_TOKENS.get(budget_level, 0),
        "is_correct":      bool(correct),
        "predicted_answer": predicted,
        "raw_answer":      raw_answer,
        "input_tokens":    raw.get("input_tokens", 0),
        "output_tokens":   raw.get("output_tokens", 0),
        "reasoning_tokens": raw.get("reasoning_tokens", 0),
        "total_cost_usd":  cost,
        "latency_ms":      raw.get("latency_ms", 0),
        "error":           error_str,
        "track":           "api",
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }

    _atomic_append(results_path, result)
    tracker.add(cost)

    done, eta_str = eta.tick()
    if done % LOG_INTERVAL == 0:
        log.info(
            "Progress: %d/%d | cost=$%.2f | ETA=%s",
            done, eta.total, tracker.total, eta_str,
        )

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def run_evaluation(
    benchmark_path: str = "data/benchmark/benchmark_5000.jsonl",
    model_pool_path: str = "configs/model_pool.yaml",
    output_path: str = "data/model_eval_results/api_results_anchor.jsonl",
    split: str = "anchor",
    tier_filter: Optional[List[str]] = None,
    max_workers: int = 6,
    daily_spend_limit: float = 150.0,
) -> str:
    """
    Evaluate all API model configs on the specified benchmark split.

    Args:
        split:             "anchor" | "validation" | "remaining" | "all"
        tier_filter:       ["A","B","C"] or subset (None = all tiers)
        max_workers:       Concurrent API calls
        daily_spend_limit: Abort if cost exceeds this

    Returns:
        Absolute path to output results file.
    """
    def _abs(p: str) -> str:
        path = Path(p)
        return str(path if path.is_absolute() else _PROJECT_ROOT / path)

    benchmark_path = _abs(benchmark_path)
    model_pool_path = _abs(model_pool_path)
    output_path = _abs(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading benchmark from %s ...", benchmark_path)
    all_samples = load_jsonl(benchmark_path)
    samples = _filter_samples(all_samples, split)
    log.info("Split '%s': %d samples", split, len(samples))

    api_configs = _load_api_configs(model_pool_path, tier_filter)
    log.info("API configs: %d (tier_filter=%s)", len(api_configs), tier_filter)

    # Build + filter work items
    completed = _load_completed_keys(output_path)
    pending = [
        (s, yk, mc, bl)
        for s in samples
        for yk, mc, bl in api_configs
        if (s["sample_id"], f"{yk}_{bl}") not in completed
    ]
    log.info("Total work items: %d | Already done: %d | Pending: %d",
             len(samples) * len(api_configs), len(completed), len(pending))

    if not pending:
        log.info("All done — nothing to evaluate.")
        return output_path

    tracker = _CostTracker(daily_spend_limit)
    eta = _ETATracker(len(pending))

    log.info("Starting evaluation: %d workers, spend limit=$%.0f", max_workers, daily_spend_limit)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_eval_one, s, yk, mc, bl, tracker, eta, output_path): (s["sample_id"], f"{yk}_{bl}")
            for s, yk, mc, bl in pending
        }
        for future in as_completed(futures):
            if tracker.aborted:
                log.warning("Daily spend limit $%.2f reached — aborting.", daily_spend_limit)
                pool.shutdown(wait=False, cancel_futures=True)
                break
            try:
                future.result()
            except Exception as exc:
                log.error("Unexpected worker error: %s", exc)

    log.info(
        "Evaluation complete: %d calls | $%.2f total cost",
        tracker.count, tracker.total,
    )
    return output_path
