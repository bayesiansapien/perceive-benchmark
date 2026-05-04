#!/usr/bin/env python3
"""
DocRouteBench Phase 3, API Evaluation Orchestrator

Runs the staged Phase 3 evaluation on this Vertex AI VM (API models).
GPU models run separately on DGX via run_phase3_gpu.py.

Usage:
    python scripts/run_phase3.py --water-test              # 10 samples x all configs (~$1)
    python scripts/run_phase3.py --stage 1a               # Anchor full eval (~$383)
    python scripts/run_phase3.py --stage 1b               # Validation eval (~$67)
    python scripts/run_phase3.py --merge --split anchor    # Merge API + GPU results
    python scripts/run_phase3.py --cascade-validate        # 4 validation experiments
    python scripts/run_phase3.py --stage 2                 # Cascade remaining (~$194)
    python scripts/run_phase3.py --stage 2 --full          # Full remaining (Plan B)
    python scripts/run_phase3.py --status                  # Show progress

Staged approach:
  Stage 1a: Anchor (1,500) x 21 API configs = 31,500 calls  ~$383
  Stage 1b: Validation (750) x 21 API, Tier C on Tier3 only  ~$67
  [GPU track on DGX runs simultaneously: push results when done]
  Merge:    Combine API + GPU results per split
  Validate: 4 cascade experiments on merged anchor
  Stage 2:  Cascade remaining 2,551 (or full fallback)        ~$194
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("run_phase3")

RESULTS_DIR    = _ROOT / "data" / "model_eval_results"
BENCHMARK_PATH = str(_ROOT / "data" / "benchmark" / "benchmark_5000.jsonl")
MODEL_POOL     = str(_ROOT / "configs" / "model_pool.yaml")


def _count(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return sum(1 for _ in open(p))


# ── Status ────────────────────────────────────────────────────────────────────

def show_status():
    log.info("=" * 70)
    log.info("PHASE 3 STATUS")
    log.info("=" * 70)
    files = {
        "API anchor":       RESULTS_DIR / "api_results_anchor.jsonl",
        "API validation":   RESULTS_DIR / "api_results_validation.jsonl",
        "API remaining":    RESULTS_DIR / "api_results_remaining.jsonl",
        "GPU results":      RESULTS_DIR / "gpu_results.jsonl",
        "Merged anchor":    RESULTS_DIR / "merged" / "anchor_results.jsonl",
        "Merged validation":RESULTS_DIR / "merged" / "validation_results.jsonl",
        "Cascade report":   _ROOT / "data" / "phase3_cascade_validation.json",
    }
    for label, path in files.items():
        n = _count(str(path))
        status = f"{n:,} records" if n > 0 else "not started"
        log.info("  %-26s  %s", label, status)
    log.info("")
    log.info("  Expected anchor (21 API x 1500):         31,500")
    log.info("  Expected validation (21 API x 750 opt):  ~10,000")
    log.info("  Expected GPU all splits (6 cfg x 2250):  13,500")
    log.info("=" * 70)


# ── Water test ────────────────────────────────────────────────────────────────

def run_water_test():
    """10 anchor samples x all 21 API configs. Verifies everything works."""
    log.info("=" * 60)
    log.info("WATER TEST: 10 anchor samples x all API configs")
    log.info("=" * 60)

    import json, tempfile
    from src.schema import load_jsonl
    from src.model_eval.eval_harness import run_evaluation

    samples = load_jsonl(BENCHMARK_PATH)
    anchor10 = [s for s in samples if s.get("in_anchor_set")][:10]
    if len(anchor10) < 10:
        log.warning("Only %d anchor samples found!", len(anchor10))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = str(RESULTS_DIR / "water_test_results.jsonl")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for s in anchor10:
            f.write(json.dumps(s) + "\n")
        tmp = f.name

    try:
        run_evaluation(
            benchmark_path=tmp,
            model_pool_path=MODEL_POOL,
            output_path=out,
            split="all",
            max_workers=4,
            daily_spend_limit=5.0,
        )
    finally:
        import os
        os.unlink(tmp)

    n = _count(out)
    log.info("Water test done: %d results → %s", n, out)
    log.info("Review results, then run --stage 1a to start full evaluation.")


# ── Stage 1a. Anchor ─────────────────────────────────────────────────────────

def run_stage_1a():
    """Full eval on anchor (1,500 x all 21 API configs)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    from src.model_eval.eval_harness import run_evaluation
    log.info("Stage 1A: Full anchor evaluation (~$383, ~6h parallel)")
    run_evaluation(
        benchmark_path=BENCHMARK_PATH,
        model_pool_path=MODEL_POOL,
        output_path=str(RESULTS_DIR / "api_results_anchor.jsonl"),
        split="anchor",
        max_workers=6,
        daily_spend_limit=450.0,
    )
    log.info("Stage 1A done.")


# ── Stage 1b. Validation ─────────────────────────────────────────────────────

def run_stage_1b():
    """
    Validation eval (750 samples).
    Tier A+B on all validation. Tier C on Tier 3 only (~195 samples).
    """
    import json, tempfile, os
    from src.schema import load_jsonl
    from src.model_eval.eval_harness import run_evaluation

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_samples = load_jsonl(BENCHMARK_PATH)
    val_all   = [s for s in all_samples if s.get("in_validation_set")]
    val_tier3 = [s for s in val_all if s.get("tier_final", 2) == 3]
    log.info("Stage 1B Validation: %d total, %d Tier 3", len(val_all), len(val_tier3))

    out = str(RESULTS_DIR / "api_results_validation.jsonl")

    # Tier A + B on all validation
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for s in val_all:
            f.write(json.dumps(s) + "\n")
        tmp_ab = f.name
    try:
        run_evaluation(
            benchmark_path=tmp_ab, model_pool_path=MODEL_POOL, output_path=out,
            split="all", tier_filter=["A", "B"], max_workers=6, daily_spend_limit=80.0,
        )
    finally:
        os.unlink(tmp_ab)

    # Tier C on Tier 3 validation only
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for s in val_tier3:
            f.write(json.dumps(s) + "\n")
        tmp_c = f.name
    try:
        run_evaluation(
            benchmark_path=tmp_c, model_pool_path=MODEL_POOL, output_path=out,
            split="all", tier_filter=["C"], max_workers=4, daily_spend_limit=100.0,
        )
    finally:
        os.unlink(tmp_c)

    log.info("Stage 1B done. %d results in %s", _count(out), out)


# ── Merge ─────────────────────────────────────────────────────────────────────

def run_merge(split: str):
    """Merge API + GPU results for a split."""
    from src.model_eval.result_merger import merge_results
    api_file  = str(RESULTS_DIR / f"api_results_{split}.jsonl")
    gpu_file  = str(RESULTS_DIR / "gpu_results.jsonl")
    merged_dir = RESULTS_DIR / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    out = str(merged_dir / f"{split}_results.jsonl")
    n = merge_results(api_file, gpu_file, out)
    log.info("Merged %d records → %s", n, out)


# ── Cascade validate ──────────────────────────────────────────────────────────

def run_cascade_validate():
    """4 validation experiments on merged anchor results."""
    from scripts.validate_cascade import run_cascade_validation
    merged_path = str(RESULTS_DIR / "merged" / "anchor_results.jsonl")
    if not Path(merged_path).exists():
        log.error("Merged anchor results not found: %s", merged_path)
        log.error("Run --merge --split anchor first.")
        sys.exit(1)
    run_cascade_validation(anchor_results_path=merged_path)


# ── Stage 2. Remaining ───────────────────────────────────────────────────────

def run_stage_2(full: bool = False):
    """Cascade (or full) eval on remaining 2,551 samples."""
    from src.model_eval.eval_harness import run_evaluation
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = str(RESULTS_DIR / "api_results_remaining.jsonl")

    if full:
        log.info("Stage 2 FULL: all 21 configs on remaining 2,551 (~$250)")
        run_evaluation(
            benchmark_path=BENCHMARK_PATH, model_pool_path=MODEL_POOL,
            output_path=out, split="remaining", max_workers=6, daily_spend_limit=280.0,
        )
    else:
        log.info("Stage 2 CASCADE: Tier A all → B on A-failures → C on AB-failures (~$194)")
        for tier, limit in [("A", 12.0), ("B", 110.0), ("C", 120.0)]:
            log.info("  Running Tier %s ...", tier)
            run_evaluation(
                benchmark_path=BENCHMARK_PATH, model_pool_path=MODEL_POOL,
                output_path=out, split="remaining",
                tier_filter=[tier], max_workers=6, daily_spend_limit=limit,
            )

    log.info("Stage 2 done. %d results in %s", _count(out), out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DocRouteBench Phase 3 Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--water-test",       action="store_true")
    parser.add_argument("--stage",            type=str, choices=["1a", "1b", "2"])
    parser.add_argument("--full",             action="store_true", help="Stage 2 full (Plan B)")
    parser.add_argument("--merge",            action="store_true")
    parser.add_argument("--split",            type=str, default="anchor",
                        choices=["anchor", "validation", "remaining"])
    parser.add_argument("--cascade-validate", action="store_true")
    parser.add_argument("--status",           action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.status:
        show_status()
    elif args.water_test:
        run_water_test()
    elif args.stage == "1a":
        run_stage_1a()
    elif args.stage == "1b":
        run_stage_1b()
    elif args.merge:
        run_merge(args.split)
    elif args.cascade_validate:
        run_cascade_validate()
    elif args.stage == "2":
        run_stage_2(full=args.full)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
