#!/usr/bin/env python3
"""
DocRouteBench Phase 3 — GPU Evaluation Harness (DGX)

Evaluates GPU-based model configs on benchmark samples using vLLM.
Designed to run on DGX server. Results appended to gpu_results.jsonl,
then pushed to git for sync to this VM.

Usage:
    from src.model_eval.eval_harness_gpu import run_gpu_evaluation
    run_gpu_evaluation(yaml_key="a1_qwen35vl4b", budget_level="B0", split="anchor")
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from src.schema import load_jsonl, append_jsonl
from src.scoring.unified import is_correct as unified_is_correct
from src.sampling.api_probe import load_image_b64
from src.model_eval.answer_extractor import extract_answer
from src.model_eval.model_adapters.gpu_adapter import GPUAdapter

log = logging.getLogger("eval_harness_gpu")

LOG_INTERVAL = 50
_BUDGET_TOKENS = {"B0": 0, "B1": 1024, "B2": 4096, "B3": 16384}


def _load_completed_keys(results_path: str) -> set:
    completed = set()
    try:
        for r in load_jsonl(results_path):
            sid = r.get("sample_id", "")
            cid = r.get("config_id", "")
            if sid and cid:
                completed.add((sid, cid))
    except FileNotFoundError:
        pass
    return completed


def _filter_samples(samples: list, split: str) -> list:
    if split == "anchor":
        return [s for s in samples if s.get("in_anchor_set")]
    elif split == "validation":
        return [s for s in samples if s.get("in_validation_set")]
    elif split == "remaining":
        return [s for s in samples
                if not s.get("in_anchor_set") and not s.get("in_validation_set")]
    elif split == "all":
        return samples
    raise ValueError(f"Unknown split: {split!r}")


def run_gpu_evaluation(
    yaml_key: str,
    budget_level: str = "B0",
    benchmark_path: str = "data/benchmark/benchmark_5000.jsonl",
    model_pool_path: str = "configs/model_pool.yaml",
    output_path: str = "data/model_eval_results/gpu_results.jsonl",
    split: str = "anchor",
    batch_size: int = 8,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 1,
) -> str:
    """
    Evaluate one GPU model config on the specified benchmark split.
    Appends to output_path (resume-safe via completed key check).

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

    # Load model config
    with open(model_pool_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["models"].get(yaml_key)
    if model_cfg is None:
        raise ValueError(f"Model key not found in pool: {yaml_key!r}")
    if model_cfg.get("provider") != "self_hosted":
        raise ValueError(f"{yaml_key} is not a self-hosted GPU model")
    if budget_level not in model_cfg.get("budgets", []):
        raise ValueError(f"{yaml_key} does not support budget {budget_level}")

    config_id = f"{yaml_key}_{budget_level}"
    log.info("=" * 60)
    log.info("Evaluating: %s | split=%s", config_id, split)
    log.info("=" * 60)

    # Load and filter samples
    all_samples = load_jsonl(benchmark_path)
    samples = _filter_samples(all_samples, split)
    log.info("Split '%s': %d samples", split, len(samples))

    # Resume: skip already completed
    completed = _load_completed_keys(output_path)
    pending = [s for s in samples if (s["sample_id"], config_id) not in completed]
    log.info("Pending: %d | Already done: %d", len(pending), len(completed))

    if not pending:
        log.info("Nothing to do for %s.", config_id)
        return output_path

    # Load model via vLLM (one model at a time on DGX)
    log.info("Loading model %s via vLLM ...", yaml_key)
    adapter = GPUAdapter.load_model(
        yaml_key=yaml_key,
        model_cfg=model_cfg,
        budget_level=budget_level,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )

    correct_count = 0
    total = len(pending)
    t_start = time.monotonic()

    for i, sample in enumerate(pending):
        sample_id = sample["sample_id"]

        # Load image
        try:
            image_b64 = load_image_b64(sample.get("image_path", ""))
        except FileNotFoundError:
            log.warning("Image not found: %s — skipping.", sample_id)
            continue

        # Inference
        raw = {}
        error_str = None
        raw_answer = ""
        try:
            raw = adapter.call(image_b64, sample.get("query", ""))
            raw_answer = raw.get("answer", "")
        except Exception as exc:
            error_str = str(exc)[:200]
            log.warning("Inference failed %s: %s", sample_id, error_str)

        # Score
        predicted = extract_answer(raw_answer, model_cfg.get("name", ""))
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
            except Exception:
                pass

        if correct:
            correct_count += 1

        result = {
            "sample_id":        sample_id,
            "config_id":        config_id,
            "yaml_key":         yaml_key,
            "model_name":       model_cfg.get("name", ""),
            "provider":         "self_hosted",
            "tier":             yaml_key[0].upper(),
            "budget_level":     budget_level,
            "budget_tokens":    _BUDGET_TOKENS.get(budget_level, 0),
            "is_correct":       bool(correct),
            "predicted_answer": predicted,
            "raw_answer":       raw_answer,
            "input_tokens":     raw.get("input_tokens", 0),
            "output_tokens":    raw.get("output_tokens", 0),
            "reasoning_tokens": raw.get("reasoning_tokens", 0),
            "total_cost_usd":   0.0,
            "latency_ms":       raw.get("latency_ms", 0),
            "error":            error_str,
            "track":            "gpu",
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(output_path, result)

        if (i + 1) % LOG_INTERVAL == 0:
            elapsed = time.monotonic() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 1
            eta_min = (total - i - 1) / rate / 60
            acc = correct_count / (i + 1) * 100
            log.info(
                "[%s] %d/%d | acc=%.1f%% | %.2fs/sample | ETA=%.1fmin",
                config_id, i + 1, total, acc, 1 / rate, eta_min,
            )

    elapsed = time.monotonic() - t_start
    acc = correct_count / total * 100 if total else 0
    log.info(
        "Done %s: %d samples in %.1fs | accuracy=%.1f%%",
        config_id, total, elapsed, acc,
    )
    return output_path
