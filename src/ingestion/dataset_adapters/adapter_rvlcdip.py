"""
DocRouteBench: RVL-CDIP Dataset Adapter

Task: T1 (Document Classification)
HuggingFace ID: aharley/rvl_cdip
Split: test
Metric: exact_match

Each sample is a single document image labelled with one of 16 document
type classes. The adapter emits one Sample per image.
"""

from __future__ import annotations
import logging
from typing import Iterator, Optional

from ..base_adapter import BaseAdapter, make_sample_id

logger = logging.getLogger(__name__)

LABEL_MAP = {
    0:  "letter",
    1:  "form",
    2:  "email",
    3:  "handwritten",
    4:  "advertisement",
    5:  "scientific report",
    6:  "scientific publication",
    7:  "specification",
    8:  "file folder",
    9:  "news article",
    10: "budget",
    11: "invoice",
    12: "presentation",
    13: "questionnaire",
    14: "resume",
    15: "memo",
}

QUERY = (
    "What type of document is this? "
    "Choose from: letter, form, email, handwritten, advertisement, "
    "scientific report, scientific publication, specification, file folder, "
    "news article, budget, invoice, presentation, questionnaire, resume, memo."
)


class RvlCdipAdapter(BaseAdapter):
    dataset_name = "rvlcdip"
    task_type = "T1"
    metric = "exact_match"

    def __init__(self, max_samples: Optional[int] = None, seed: int = 42):
        super().__init__(max_samples=max_samples, seed=seed)

    def iter_samples(self) -> Iterator[dict]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required. Install it with: pip install datasets"
            ) from exc

        # aharley/rvl_cdip uses a legacy loading script no longer supported.
        # chainyo/rvl-cdip is a Parquet-based mirror with identical content.
        # Use streaming=True to avoid downloading all 400K+ images upfront.
        # chainyo/rvl-cdip is a Parquet-based mirror of aharley/rvl_cdip.
        HF_IDS = ["chainyo/rvl-cdip", "aharley/rvl_cdip"]
        ds = None
        for hf_id in HF_IDS:
            try:
                logger.info(f"[rvlcdip] Trying {hf_id} (streaming) ...")
                ds = load_dataset(hf_id, split="test", streaming=True)
                logger.info(f"[rvlcdip] Loaded from {hf_id} in streaming mode")
                break
            except Exception as exc:
                logger.warning(f"[rvlcdip] {hf_id} failed: {exc}")
        if ds is None:
            raise RuntimeError("Could not load RVL-CDIP from any known HF ID.")

        logger.info("[rvlcdip] Streaming: samples will download on demand")

        for idx, row in enumerate(ds):
            label_int = row["label"]
            label_str = LABEL_MAP.get(label_int)
            if label_str is None:
                logger.warning(
                    f"[rvlcdip] Unknown label int {label_int} at index {idx}, skipping"
                )
                continue

            sample_id = f"rvlcdip_test_{idx:06d}"
            image = row["image"]  # PIL Image

            yield {
                "sample_id": sample_id,
                "query": QUERY,
                "gt_answer": label_str,
                "gt_answer_aliases": [],
                "image": image,
                "task_type": self.task_type,
                "correctness_metric": self.metric,
                "source_split": "test",
                "doc_type": label_str,
                "num_pages": 1,
                "has_table": False,
                "has_chart": False,
                "has_figure": False,
                "has_handwriting": label_str == "handwritten",
            }


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=== RVL-CDIP Adapter Smoke Test (5 samples) ===")
    adapter = RvlCdipAdapter(max_samples=5)

    samples = []
    for raw in adapter.iter_samples():
        samples.append(raw)
        if len(samples) >= 5:
            break

    if not samples:
        print("ERROR: No samples yielded.")
        sys.exit(1)

    for i, s in enumerate(samples):
        img = s["image"]
        print(
            f"  [{i}] id={s['sample_id']} | label={s['gt_answer']!r} "
            f"| img_size={img.size} | doc_type={s['doc_type']!r}"
        )
        # Verify required keys
        for key in ("sample_id", "query", "gt_answer", "gt_answer_aliases", "image",
                    "task_type", "correctness_metric"):
            assert key in s, f"Missing key: {key}"
        assert s["task_type"] == "T1"
        assert s["correctness_metric"] == "exact_match"
        assert s["gt_answer"] in LABEL_MAP.values(), f"Unexpected label: {s['gt_answer']}"
        assert s["gt_answer_aliases"] == []

    print(f"\nAll {len(samples)} samples passed verification.")

    # Quick end-to-end run (writes images + JSONL for 5 samples)
    print("\nRunning adapter.run() for 5 samples ...")
    n = adapter.run()
    print(f"adapter.run() wrote {n} samples to {adapter.output_path}")
    print("Smoke test PASSED.")
