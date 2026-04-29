#!/usr/bin/env python3
"""
DocRouteBench Phase 3 — GPU Evaluation Script (run on DGX)

Pull the repo, run this script for each GPU model, push results.

Usage:
    python scripts/run_phase3_gpu.py --model a1_qwen35vl4b --split anchor
    python scripts/run_phase3_gpu.py --model b4_qwen35b_moe --split anchor --budget B3
    python scripts/run_phase3_gpu.py --all --split anchor      # sequential, all GPU models

With 2 A100s (run simultaneously in separate terminals):
    CUDA_VISIBLE_DEVICES=0 python scripts/run_phase3_gpu.py --model b4_qwen35b_moe --split anchor
    CUDA_VISIBLE_DEVICES=1 python scripts/run_phase3_gpu.py --model a1_qwen35vl4b --split anchor

After completion:
    git add data/model_eval_results/gpu_results.jsonl
    git commit -m "GPU eval: <model> <split> results"
    git push origin benchmark/dataset-pipeline
"""
import argparse
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
log = logging.getLogger("run_phase3_gpu")

# All GPU model yaml_keys → supported budget levels
GPU_MODELS = {
    "a1_qwen35vl4b":  ["B0"],
    "a3_phi4vision":   ["B0"],
    "b2_internvl3":    ["B0"],
    "b4_qwen35b_moe":  ["B0", "B1", "B3"],
}


def main():
    parser = argparse.ArgumentParser(
        description="DocRouteBench Phase 3 GPU Evaluation (run on DGX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", type=str,
        help=f"YAML key of model to run. One of: {list(GPU_MODELS.keys())}",
    )
    parser.add_argument(
        "--budget", type=str, default=None,
        help="Budget level (B0/B1/B3). Default: all supported budgets for this model.",
    )
    parser.add_argument(
        "--split", type=str, default="anchor",
        choices=["anchor", "validation", "remaining", "all"],
        help="Benchmark split to evaluate (default: anchor)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all GPU models sequentially (ignores --model/--budget)",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/model_eval_results/gpu_results.jsonl",
        help="Output JSONL file (appended, resume-safe)",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size",   type=int,   default=1)
    parser.add_argument("--batch-size",              type=int,   default=8)
    args = parser.parse_args()

    from src.model_eval.eval_harness_gpu import run_gpu_evaluation

    if args.all:
        models_to_run = [
            (k, b) for k, budgets in GPU_MODELS.items() for b in budgets
        ]
    elif args.model:
        if args.model not in GPU_MODELS:
            log.error("Unknown model: %r. Valid keys: %s", args.model, list(GPU_MODELS.keys()))
            sys.exit(1)
        budgets = [args.budget] if args.budget else GPU_MODELS[args.model]
        models_to_run = [(args.model, b) for b in budgets]
    else:
        parser.print_help()
        sys.exit(1)

    log.info("GPU evaluation plan: %d runs", len(models_to_run))
    for yaml_key, budget_level in models_to_run:
        log.info("  %s_%s on split=%s", yaml_key, budget_level, args.split)

    for yaml_key, budget_level in models_to_run:
        log.info("=" * 70)
        log.info("RUNNING: %s  budget=%s  split=%s", yaml_key, budget_level, args.split)
        log.info("=" * 70)
        try:
            out = run_gpu_evaluation(
                yaml_key=yaml_key,
                budget_level=budget_level,
                split=args.split,
                output_path=args.output,
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=args.tensor_parallel_size,
                batch_size=args.batch_size,
            )
            log.info("Results appended to: %s", out)
        except Exception as exc:
            log.error("FAILED %s/%s: %s", yaml_key, budget_level, exc)
            raise

    log.info("")
    log.info("All GPU evaluation done. Sync results:")
    log.info("  git add %s", args.output)
    log.info("  git commit -m 'GPU eval: %s %s results'",
             args.model or "all", args.split)
    log.info("  git push origin benchmark/dataset-pipeline")


if __name__ == "__main__":
    main()
