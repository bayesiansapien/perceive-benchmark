#!/usr/bin/env python3
"""
DocRouteBench Phase 3, Merge API + GPU Results

Merges API track (this VM) and GPU track (DGX) result files for a given split.
Run after pushing GPU results from DGX.

Usage:
    python scripts/merge_results.py --split anchor
    python scripts/merge_results.py --split validation
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
)


def main():
    parser = argparse.ArgumentParser(description="Merge API + GPU Phase 3 results")
    parser.add_argument("--split", type=str, default="anchor",
                        choices=["anchor", "validation", "remaining"])
    args = parser.parse_args()

    results_dir = _ROOT / "data" / "model_eval_results"
    api_file  = str(results_dir / f"api_results_{args.split}.jsonl")
    gpu_file  = str(results_dir / "gpu_results.jsonl")
    merged_dir = results_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    out = str(merged_dir / f"{args.split}_results.jsonl")

    from src.model_eval.result_merger import merge_results
    n = merge_results(api_file, gpu_file, out)
    print(f"Merged {n:,} records → {out}")


if __name__ == "__main__":
    main()
