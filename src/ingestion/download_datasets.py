"""
DocRouteBench — Dataset Download & Normalization Orchestrator

Runs all 18 dataset adapters, optionally in parallel.
Produces: data/processed/{dataset}_normalized.jsonl for each dataset.

Usage:
  # Run all datasets sequentially
  python -m src.ingestion.download_datasets

  # Run specific datasets
  python -m src.ingestion.download_datasets --datasets docvqa chartqa

  # Run with sample limit (for testing)
  python -m src.ingestion.download_datasets --max-samples 50

  # Run N datasets in parallel (uses ProcessPoolExecutor)
  python -m src.ingestion.download_datasets --workers 4
"""

import argparse
import logging
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Registry: dataset_name → adapter class import path
ADAPTER_REGISTRY = {
    "rvlcdip":        "src.ingestion.dataset_adapters.adapter_rvlcdip.RvlCdipAdapter",
    "funsd":          "src.ingestion.dataset_adapters.adapter_funsd.FunsdAdapter",
    "cord":           "src.ingestion.dataset_adapters.adapter_cord.CORDAdapter",
    "sroie":          "src.ingestion.dataset_adapters.adapter_sroie.SROIEAdapter",
    "textvqa":        "src.ingestion.dataset_adapters.adapter_textvqa.TextVQAAdapter",
    "stvqa":          "src.ingestion.dataset_adapters.adapter_stvqa.STVQAAdapter",
    "hiertext":       "src.ingestion.dataset_adapters.adapter_hiertext.HierTextAdapter",
    "publaynet":      "src.ingestion.dataset_adapters.adapter_publaynet.PubLayNetAdapter",
    # docbank: DROPPED — 47GB manual download, HF mirror broken, redistribution prohibited
    # deepform: DROPPED — requires multi-day PDF download from DocumentCloud
    "docvqa":         "src.ingestion.dataset_adapters.adapter_docvqa.DocVQAAdapter",
    "infographicvqa": "src.ingestion.dataset_adapters.adapter_infographicvqa.InfographicVQAAdapter",
    "chartqa":        "src.ingestion.dataset_adapters.adapter_chartqa.ChartQAAdapter",
    "tabfact":        "src.ingestion.dataset_adapters.adapter_tabfact.TabFactAdapter",
    "wtq":            "src.ingestion.dataset_adapters.adapter_wtq.WTQAdapter",
    "visualmrc":      "src.ingestion.dataset_adapters.adapter_visualmrc.VisualMRCAdapter",
    "mpdocvqa":       "src.ingestion.dataset_adapters.adapter_mpdocvqa.MPDocVQAAdapter",
    "slidevqa":       "src.ingestion.dataset_adapters.adapter_slidevqa.SlideVQAAdapter",
}


def _import_adapter(class_path: str):
    """Dynamically import an adapter class from dotted path."""
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def run_adapter(dataset_name: str, max_samples: int = None) -> dict:
    """Run a single adapter. Returns result dict. Designed to run in subprocess."""
    start = time.time()
    result = {"dataset": dataset_name, "status": "unknown", "count": 0, "elapsed_s": 0}
    try:
        AdapterClass = _import_adapter(ADAPTER_REGISTRY[dataset_name])
        adapter = AdapterClass(max_samples=max_samples)
        count = adapter.run()
        result.update({"status": "success", "count": count, "elapsed_s": round(time.time() - start, 1)})
        logger.info(f"[{dataset_name}] ✓ {count} samples in {result['elapsed_s']}s")
    except Exception as e:
        result.update({"status": "error", "error": str(e), "elapsed_s": round(time.time() - start, 1)})
        logger.error(f"[{dataset_name}] ✗ {e}")
    return result


def run_all(datasets: list, max_samples: int = None, workers: int = 1) -> list:
    """Run all adapters, optionally in parallel."""
    results = []
    if workers == 1:
        for ds in datasets:
            results.append(run_adapter(ds, max_samples))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_adapter, ds, max_samples): ds for ds in datasets}
            for future in as_completed(futures):
                results.append(future.result())
    return results


def print_summary(results: list):
    success = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]
    total = sum(r["count"] for r in success)

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    for r in sorted(results, key=lambda x: x["dataset"]):
        if r["status"] == "success":
            print(f"  ✓ {r['dataset']:<18} {r['count']:>5} samples  {r['elapsed_s']}s")
        else:
            print(f"  ✗ {r['dataset']:<18} ERROR: {r.get('error','')[:50]}")
    print(f"\n  Total: {total} samples across {len(success)}/{len(results)} datasets")
    if errors:
        print(f"  Failed: {[r['dataset'] for r in errors]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(ADAPTER_REGISTRY), default=list(ADAPTER_REGISTRY))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (1=sequential)")
    args = parser.parse_args()

    logger.info(f"Running {len(args.datasets)} datasets with {args.workers} worker(s)")
    results = run_all(args.datasets, max_samples=args.max_samples, workers=args.workers)
    print_summary(results)

    # Save run report
    report_path = Path("data/processed/ingestion_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Report saved to {report_path}")
