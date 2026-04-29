"""
DocRouteBench — DocBank Dataset Adapter

Dataset: ds4sd/DocBank (HuggingFace)
Task: T3 (Layout & Spatial Reasoning)
Metric: exact_match

DocBank provides token-level structural labels for academic paper pages.
Each sample contains text tokens and their associated layout element labels
(abstract, author, caption, date, equation, figure, footer, list,
paragraph, reference, section, table, title).

We convert to QA: given a text excerpt, predict its structural element type.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from datasets import load_dataset

from ..base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

# All structural element types in DocBank
DOCBANK_LABELS = {
    "abstract", "author", "caption", "date", "equation",
    "figure", "footer", "list", "paragraph", "reference",
    "section", "table", "title",
}

# HuggingFace dataset ID candidates (try in order)
# NOTE: DocBank requires manual download — 47GB images from Azure Blob Storage.
# HF mirror (liminghao1630/DocBank) has broken zip files.
# Official download: https://github.com/doc-analysis/DocBank
# Redistribution prohibited — users must download themselves.
# This adapter falls back to synthetic samples if HF is unavailable.
HF_IDS = ["liminghao1630/DocBank", "ds4sd/DocBank", "ds4sd/docbank"]


class DocBankAdapter(BaseAdapter):
    """Adapter for the DocBank layout understanding dataset."""

    dataset_name = "docbank"
    task_type = "T3"
    metric = "exact_match"

    def __init__(self, max_samples: Optional[int] = None, seed: int = 42):
        super().__init__(max_samples=max_samples, seed=seed)

    def _load_hf_dataset(self):
        """Try loading the dataset from HuggingFace with fallback IDs."""
        last_exc = None
        for hf_id in HF_IDS:
            try:
                logger.info(f"[docbank] Trying HuggingFace ID: {hf_id}")
                ds = load_dataset(hf_id, split="test", )
                logger.info(f"[docbank] Loaded {len(ds)} samples from {hf_id}")
                return ds
            except Exception as e:
                logger.warning(f"[docbank] Failed to load {hf_id}: {e}")
                last_exc = e
        logger.warning(
            f"[docbank] Could not load from HuggingFace. "
            f"DocBank requires manual download (47GB). "
            f"See: https://github.com/doc-analysis/DocBank "
            f"Generating synthetic fallback samples."
        )
        return None

    def _pick_text_excerpt(self, tokens: list[str], labels: list[str], target_label: str) -> str:
        """
        Pick a representative text excerpt for the target label.
        Selects consecutive tokens matching the target label, up to ~8 tokens.
        """
        matching_tokens = []
        for tok, lbl in zip(tokens, labels):
            if lbl == target_label:
                matching_tokens.append(tok.strip())
                if len(matching_tokens) >= 8:
                    break
        excerpt = " ".join(t for t in matching_tokens if t)
        return excerpt[:120]  # cap length for query clarity

    def iter_samples(self) -> Iterator[dict]:
        """
        Yield one sample per DocBank page record.

        DocBank field names may vary by version. We handle the common variants:
          - tokens / token / words / word
          - tags / labels / tag / label
          - image / img / page_image
        """
        ds = self._load_hf_dataset()

        # Synthetic fallback when dataset unavailable
        if ds is None:
            import random
            rng = random.Random(self.seed)
            labels = list(DOCBANK_LABELS)
            for idx in range(self.max_samples or 100):
                label = rng.choice(labels)
                sample_id = f"docbank_synthetic_{idx:06d}"
                from PIL import Image, ImageDraw
                img = Image.new("RGB", (595, 842), "white")
                draw = ImageDraw.Draw(img)
                draw.text((50, 50), f"[{label.upper()}] Sample academic paper text", fill="black")
                draw.text((50, 100), "This is a synthetic DocBank sample.", fill="gray")
                yield {
                    "sample_id": sample_id,
                    "query": f"What structural element type does this text region represent in the academic paper?",
                    "gt_answer": label,
                    "gt_answer_aliases": [label],
                    "image": img,
                    "task_type": "T3",
                    "correctness_metric": "exact_match",
                    "source_split": "synthetic",
                    "doc_type": "academic_paper",
                    "has_table": label == "table",
                    "has_figure": label == "figure",
                    "num_pages": 1,
                }
            return

        count = 0
        for idx, row in enumerate(ds):
            # --- Resolve token field ---
            tokens = None
            for field in ("tokens", "token", "words", "word"):
                if field in row and row[field]:
                    tokens = row[field]
                    break
            if not tokens:
                logger.warning(f"[docbank] Row {idx}: no token field found, skipping")
                continue

            # Normalise to list of strings
            if isinstance(tokens, str):
                tokens = tokens.split()

            # --- Resolve label field ---
            labels_raw = None
            for field in ("tags", "labels", "tag", "label"):
                if field in row and row[field]:
                    labels_raw = row[field]
                    break
            if not labels_raw:
                logger.warning(f"[docbank] Row {idx}: no label field found, skipping")
                continue

            # Normalise labels to strings
            if isinstance(labels_raw[0], int):
                # Integer-encoded: map via dataset features if possible
                try:
                    feature_names = ds.features[
                        next(f for f in ("tags", "labels", "tag", "label") if f in ds.features)
                    ].names
                    labels = [feature_names[l] for l in labels_raw]
                except Exception:
                    # Fall back to string conversion; downstream filtering will drop unknowns
                    labels = [str(l) for l in labels_raw]
            else:
                labels = [str(l) for l in labels_raw]

            # --- Resolve image field ---
            image = None
            for field in ("image", "img", "page_image"):
                if field in row and row[field] is not None:
                    image = row[field]
                    break
            if image is None:
                logger.warning(f"[docbank] Row {idx}: no image field found, skipping")
                continue

            # --- Determine the dominant / most interesting label for the QA pair ---
            label_set = set(labels)
            known_labels = label_set & DOCBANK_LABELS

            # Prefer informative labels over generic "paragraph"
            priority_order = [
                "title", "abstract", "section", "author", "caption",
                "table", "figure", "equation", "list", "reference",
                "date", "footer", "paragraph",
            ]
            target_label = None
            for candidate in priority_order:
                if candidate in known_labels:
                    target_label = candidate
                    break

            if target_label is None:
                # Nothing recognisable — fall back to first label
                target_label = labels[0] if labels else "paragraph"

            excerpt = self._pick_text_excerpt(tokens, labels, target_label)
            if not excerpt:
                # If excerpt is empty, just join first few tokens
                excerpt = " ".join(str(t) for t in tokens[:6])

            sample_id = f"docbank_test_{idx:06d}"

            yield {
                "sample_id": sample_id,
                "query": (
                    f"What type of document structure element does the text "
                    f"'{excerpt}' belong to?"
                ),
                "gt_answer": target_label,
                "gt_answer_aliases": [target_label],
                "image": image,
                "task_type": self.task_type,
                "correctness_metric": self.metric,
                "source_split": "test",
                "doc_type": "academic_paper",
                "has_figure": "figure" in label_set,
                "has_table": "table" in label_set,
                "has_chart": False,
                "has_handwriting": False,
                "num_pages": 1,
            }

            count += 1
            if self.max_samples and count >= self.max_samples:
                break


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)
    print("=== DocBank Adapter Smoke Test (3 samples) ===\n")

    adapter = DocBankAdapter(max_samples=3)
    for i, sample in enumerate(adapter.iter_samples()):
        print(f"--- Sample {i + 1} ---")
        display = {k: v for k, v in sample.items() if k != "image"}
        print(json.dumps(display, indent=2))
        img = sample["image"]
        print(f"  image type : {type(img).__name__}")
        if hasattr(img, "size"):
            print(f"  image size : {img.size}")
        print()
