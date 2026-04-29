"""
DocRouteBench — Base Dataset Adapter

All 18 dataset adapters inherit from BaseAdapter.
Each adapter:
  1. Loads dataset from HuggingFace (or disk)
  2. Converts each sample to the unified Sample schema
  3. Saves images to data/raw/{dataset_name}/images/
  4. Writes normalized JSONL to data/processed/{dataset_name}_normalized.jsonl
"""

from __future__ import annotations
import io
import os
import json
import hashlib
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from PIL import Image

# Project root is 2 levels up from this file
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_IMAGE_DIM = 2048  # resize longest side if larger


class BaseAdapter(ABC):
    """Abstract base class for all DocRouteBench dataset adapters."""

    dataset_name: str       # override in subclass, e.g. "docvqa"
    task_type: str          # "T1" through "T6"
    metric: str             # correctness metric name

    def __init__(self, max_samples: Optional[int] = None, seed: int = 42):
        self.max_samples = max_samples
        self.seed = seed
        self.image_dir = DATA_RAW / self.dataset_name / "images"
        self.output_path = DATA_PROCESSED / f"{self.dataset_name}_normalized.jsonl"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def iter_samples(self) -> Iterator[dict]:
        """
        Yield raw sample dicts ready to be converted to Sample schema.
        Each dict must have at minimum:
          - sample_id: str
          - query: str
          - gt_answer: str
          - gt_answer_aliases: list[str]
          - image: PIL.Image or bytes or str (path)
          - task_type: str
          - correctness_metric: str
        Optional:
          - num_pages, has_table, has_chart, has_figure, doc_type, etc.
        """
        ...

    def save_image(self, image, sample_id: str) -> tuple[str, int]:
        """
        Save a PIL Image or bytes to disk. Returns (relative_path, bytes_size).
        Resizes if longest side > MAX_IMAGE_DIM.
        """
        # Convert to PIL if needed
        if isinstance(image, bytes):
            img = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, str):
            # It's a path
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        # Resize if too large
        w, h = img.size
        if max(w, h) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # Save
        filename = f"{sample_id}.jpg"
        abs_path = self.image_dir / filename
        img.save(abs_path, "JPEG", quality=85)
        rel_path = str(abs_path.relative_to(PROJECT_ROOT))
        return rel_path, abs_path.stat().st_size

    def run(self) -> int:
        """
        Run the full adapter pipeline. Returns count of samples written.
        Skips already-processed samples if output file exists.
        """
        # Load already-processed sample IDs for resume support
        existing_ids = set()
        if self.output_path.exists():
            with open(self.output_path) as f:
                for line in f:
                    try:
                        existing_ids.add(json.loads(line)["sample_id"])
                    except Exception:
                        pass
            logger.info(f"[{self.dataset_name}] Resuming — {len(existing_ids)} already done")

        count = 0
        with open(self.output_path, "a") as out_f:
            for raw in self.iter_samples():
                sid = raw["sample_id"]
                if sid in existing_ids:
                    continue

                # Save image
                try:
                    img_path, img_size = self.save_image(raw["image"], sid)
                except Exception as e:
                    logger.warning(f"[{self.dataset_name}] Image save failed for {sid}: {e}")
                    continue

                # Build Sample dict (matches src/schema.py)
                record = {
                    "sample_id": sid,
                    "source_dataset": self.dataset_name.upper()
                        .replace("_", "")
                        .replace("RVLCDIP", "RVL-CDIP")
                        .replace("MPDOCVQA", "MP-DocVQA")
                        .replace("SLIDEVQA", "SlideVQA")
                        .replace("CHARTQA", "ChartQA")
                        .replace("DOCVQA", "DocVQA")
                        .replace("DOCBANK", "DocBank")
                        .replace("PUBLAYNET", "PubLayNet")
                        .replace("HIERTEXT", "HierText")
                        .replace("FINTABNET", "FinTabNet")
                        .replace("TABFACT", "TabFact")
                        .replace("TEXTVQA", "TextVQA")
                        .replace("STVQA", "ST-VQA")
                        .replace("DEEPFORM", "DeepForm")
                        .replace("VISUALMRC", "VisualMRC")
                        .replace("INFOGRAPHICVQA", "InfographicVQA")
                        .replace("FUNSD", "FUNSD")
                        .replace("CORD", "CORD")
                        .replace("SROIE", "SROIE")
                        .replace("WTQ", "WTQ"),
                    "source_split": raw.get("source_split", "test"),
                    "task_type": raw.get("task_type", self.task_type),
                    "query": raw["query"],
                    "gt_answer": raw["gt_answer"],
                    "gt_answer_aliases": raw.get("gt_answer_aliases", []),
                    "correctness_metric": raw.get("correctness_metric", self.metric),
                    "image_path": img_path,
                    "num_pages": raw.get("num_pages", 1),
                    "has_table": raw.get("has_table", False),
                    "has_chart": raw.get("has_chart", False),
                    "has_figure": raw.get("has_figure", False),
                    "has_handwriting": raw.get("has_handwriting", False),
                    "doc_type": raw.get("doc_type", "document"),
                    "image_bytes_size": img_size,
                    "in_anchor_set": False,
                    "in_validation_set": False,
                }

                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                count += 1
                existing_ids.add(sid)

                if count % 100 == 0:
                    logger.info(f"[{self.dataset_name}] {count} samples written")

                if self.max_samples and count >= self.max_samples:
                    break

        logger.info(f"[{self.dataset_name}] Done — {count} new samples written to {self.output_path}")
        return count


def make_sample_id(dataset_name: str, split: str, idx: int) -> str:
    """Stable, reproducible sample ID."""
    return f"{dataset_name}_{split}_{idx:06d}"
