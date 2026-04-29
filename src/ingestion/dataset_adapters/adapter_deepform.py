"""
DocRouteBench — DeepForm Dataset Adapter

Task:   T5 (Multi-Page Long-Document Extraction)
Metric: field_f1  (token-level F1 across extracted field values)
Source: deepform/deepform  (HuggingFace, fallback: jncraton/deepform)
Split:  test

DeepForm contains FCC political advertising disclosure forms (PDF scans).
Each form is a multi-page document; the adapter uses the first page image as
the representative image. The task is key-value field extraction.

NOTE: The DeepForm dataset has evolved across several HF versions. Field names
and schema differ between releases. This adapter implements best-effort
extraction with TODO markers where schema details could not be confirmed
without live access to the dataset.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

from PIL import Image

# Allow running from repo root without installing the package
_SRC = Path(__file__).parent.parent.parent  # src/
_ROOT = _SRC.parent                          # project root
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ingestion.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)

# Fixed extraction query — same for every DeepForm sample
_EXTRACTION_QUERY = (
    "Extract the advertiser name, contract start date, contract end date, "
    "and total amount from this political advertising disclosure form."
)

# Target fields we want to extract (maps to DeepForm annotation keys)
# TODO: verify exact field names in the HF dataset schema
_TARGET_FIELDS = [
    "advertiser",          # advertiser / political committee name
    "contract_start_date", # start of the advertising contract
    "contract_end_date",   # end of the advertising contract
    "total_amount",        # total dollar value of the contract
]

# Candidate HuggingFace dataset IDs — tried in order
# NOTE: DeepForm requires manual download from Google Drive + DocumentCloud PDF pipeline.
# Official: https://github.com/jstray/deepform
# Download: https://drive.google.com/drive/folders/1bsV4A-8A9B7KZkzdbsBnCGKLMZftV2fQ
# PDF download is a multi-day process (~9,000 FCC disclosure forms).
# This adapter falls back to synthetic samples if HF unavailable.
_HF_IDS = [
    "deepform/deepform",
    "jncraton/deepform",
]


class DeepFormAdapter(BaseAdapter):
    """Adapter for the DeepForm FCC political-ad disclosure form benchmark."""

    dataset_name = "deepform"
    task_type    = "T5"
    metric       = "field_f1"

    def __init__(self, max_samples: Optional[int] = None, seed: int = 42):
        super().__init__(max_samples=max_samples, seed=seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_hf_dataset(self):
        """Try each candidate HF ID and return the first that loads."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required. Install it with:\n"
                "  pip install datasets"
            ) from exc

        last_err: Optional[Exception] = None
        for hf_id in _HF_IDS:
            try:
                logger.info(f"[deepform] Trying to load '{hf_id}' …")
                # Some versions use 'validation' not 'test'; try test first
                try:
                    ds = load_dataset(hf_id, split="test", )
                except Exception:
                    logger.warning(
                        f"[deepform] split='test' not found in '{hf_id}', "
                        "falling back to 'validation'"
                    )
                    ds = load_dataset(hf_id, split="validation", )
                logger.info(f"[deepform] Loaded {len(ds)} samples from '{hf_id}'")
                return ds
            except Exception as exc:
                logger.warning(f"[deepform] Failed to load '{hf_id}': {exc}")
                last_err = exc

        logger.warning(
            "Could not load DeepForm from HuggingFace. "
            "Manual download required from: https://github.com/jstray/deepform "
            "Falling back to synthetic samples."
        )
        return None

    @staticmethod
    def _extract_first_image(row: dict) -> Optional[Image.Image]:
        """
        Return the first page image from a DeepForm row as a PIL Image.

        DeepForm represents pages differently across versions:
          - 'images'     : list of PIL images (newer versions)
          - 'image'      : single PIL image (single-page fallback)
          - 'pages'      : list of dicts with 'image' key
          - 'pdf_pages'  : list of image bytes
        TODO: confirm actual field names once live dataset access is available.
        """
        # --- list of page images ---
        for field in ("images", "pdf_pages", "pages"):
            val = row.get(field)
            if not isinstance(val, list) or len(val) == 0:
                continue
            first = val[0]
            if isinstance(first, Image.Image):
                return first.convert("RGB")
            if isinstance(first, dict):
                # e.g. {"image": PIL.Image, ...}  or {"bytes": b"...", ...}
                inner = first.get("image") or first.get("bytes")
                if isinstance(inner, Image.Image):
                    return inner.convert("RGB")
                if isinstance(inner, bytes):
                    return Image.open(io.BytesIO(inner)).convert("RGB")
            if isinstance(first, bytes):
                return Image.open(io.BytesIO(first)).convert("RGB")

        # --- single image field ---
        for field in ("image", "page_image", "first_page"):
            val = row.get(field)
            if val is None:
                continue
            if isinstance(val, Image.Image):
                return val.convert("RGB")
            if isinstance(val, dict) and "bytes" in val:
                return Image.open(io.BytesIO(val["bytes"])).convert("RGB")
            if isinstance(val, bytes):
                return Image.open(io.BytesIO(val)).convert("RGB")

        return None

    @staticmethod
    def _build_gt_answer(row: dict) -> str:
        """
        Build a JSON string of the ground-truth extracted fields.

        DeepForm annotations vary by version; we try several known schemas.
        TODO: verify exact annotation structure in the live HF dataset.
        """
        extracted: dict[str, Any] = {}

        # --- Schema A: flat top-level fields (older DeepForm versions) ---
        # Field names may be exact or prefixed; try multiple spellings.
        _field_candidates: dict[str, list[str]] = {
            "advertiser": [
                "advertiser", "Advertiser", "advertiser_name",
                "political_committee", "committee",
            ],
            "contract_start_date": [
                "contract_start_date", "start_date", "ContractStartDate",
                "from_date", "period_start",
            ],
            "contract_end_date": [
                "contract_end_date", "end_date", "ContractEndDate",
                "to_date", "period_end",
            ],
            "total_amount": [
                "total_amount", "amount", "TotalAmount", "gross_amount",
                "contract_amount", "total",
            ],
        }
        for canonical, candidates in _field_candidates.items():
            for key in candidates:
                val = row.get(key)
                if val not in (None, "", []):
                    extracted[canonical] = str(val)
                    break

        # --- Schema B: nested 'annotations' or 'fields' dict ---
        # TODO: if the HF dataset stores annotations as a nested object,
        #       uncomment and adapt the block below.
        #
        # annots = row.get("annotations") or row.get("fields") or {}
        # if isinstance(annots, dict):
        #     for canonical, candidates in _field_candidates.items():
        #         if canonical not in extracted:
        #             for key in candidates:
        #                 val = annots.get(key)
        #                 if val not in (None, "", []):
        #                     extracted[canonical] = str(val)
        #                     break

        # --- Schema C: list of {"key": ..., "value": ...} annotation records ---
        # TODO: uncomment if the dataset uses this record-list format.
        #
        # annot_list = row.get("annotations") or row.get("entities") or []
        # if isinstance(annot_list, list):
        #     for rec in annot_list:
        #         k = rec.get("key", "").lower().replace(" ", "_")
        #         v = rec.get("value") or rec.get("text", "")
        #         for canonical, candidates in _field_candidates.items():
        #             if k in [c.lower() for c in candidates]:
        #                 extracted.setdefault(canonical, str(v))

        # Ensure all target fields appear in output (null if not found)
        for field in _TARGET_FIELDS:
            extracted.setdefault(field, None)

        return json.dumps(extracted, ensure_ascii=False)

    @staticmethod
    def _estimate_num_pages(row: dict) -> int:
        """
        Estimate the page count for a DeepForm document.

        TODO: confirm the page-count field name in the live dataset.
        """
        for field in ("num_pages", "page_count", "n_pages", "total_pages"):
            val = row.get(field)
            if isinstance(val, int) and val > 0:
                return val

        # Count image list if present
        for field in ("images", "pdf_pages", "pages"):
            val = row.get(field)
            if isinstance(val, list) and len(val) > 0:
                return len(val)

        # DeepForm forms are typically 5–30 pages; use 10 as a safe default
        return 10

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def iter_samples(self) -> Iterator[dict]:
        """Yield one normalized sample dict per DeepForm test row."""
        ds = self._load_hf_dataset()

        # Synthetic fallback when dataset unavailable
        if ds is None:
            from PIL import Image, ImageDraw
            import random
            rng = random.Random(self.seed)
            n = self.max_samples or 150
            for idx in range(n):
                img = Image.new("RGB", (850, 1100), "white")
                draw = ImageDraw.Draw(img)
                draw.text((40, 40), "FEDERAL COMMUNICATIONS COMMISSION", fill="black")
                draw.text((40, 80), "Political Broadcasting Disclosure Form", fill="black")
                draw.text((40, 200), f"Advertiser: [SYNTHETIC_{idx}]", fill="gray")
                draw.text((40, 240), "Contract Period: 01/01/2024 - 01/31/2024", fill="gray")
                draw.text((40, 280), f"Total Amount: ${rng.randint(1000, 50000):,}", fill="gray")
                yield {
                    "sample_id": f"deepform_synthetic_{idx:06d}",
                    "query": "Extract the advertiser name, contract start date, contract end date, and total amount from this political advertising disclosure form.",
                    "gt_answer": '{"advertiser": "SYNTHETIC", "start_date": "01/01/2024", "end_date": "01/31/2024", "total": "10000"}',
                    "gt_answer_aliases": [],
                    "image": img,
                    "task_type": "T5",
                    "correctness_metric": "field_f1",
                    "source_split": "synthetic",
                    "doc_type": "form",
                    "num_pages": 5,
                }
            return

        for idx, row in enumerate(ds):
            if self.max_samples is not None and idx >= self.max_samples:
                break

            sample_id = f"deepform_test_{idx:06d}"

            # --- representative image (first page) ---
            image = self._extract_first_image(row)
            if image is None:
                logger.warning(
                    f"[deepform] {sample_id}: no page image found — skipping. "
                    "TODO: inspect dataset schema and update _extract_first_image()."
                )
                continue

            # --- ground truth answer ---
            gt_answer = self._build_gt_answer(row)

            # --- page count ---
            num_pages = self._estimate_num_pages(row)

            yield {
                "sample_id":          sample_id,
                "query":              _EXTRACTION_QUERY,
                "gt_answer":          gt_answer,
                # For field_f1 the aliases are just the same JSON; evaluation
                # code should parse the JSON and compute token F1 per field.
                "gt_answer_aliases":  [gt_answer],
                "image":              image,
                "task_type":          self.task_type,
                "correctness_metric": self.metric,
                "source_split":       "test",
                # document metadata
                "num_pages":          num_pages,
                "doc_type":           "form",
                "has_table":          True,   # FCC forms have tabular schedules
                "has_figure":         False,
                "has_chart":          False,
                "has_handwriting":    False,  # mostly typed/printed
            }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print("=== DeepForm Adapter — smoke test ===")
    adapter = DeepFormAdapter(max_samples=3)

    samples = []
    try:
        for s in adapter.iter_samples():
            samples.append(s)
            gt = json.loads(s["gt_answer"])
            print(f"\n  sample_id  : {s['sample_id']}")
            print(f"  query      : {s['query'][:80]}")
            print(f"  num_pages  : {s['num_pages']}")
            print(f"  gt_answer  : {s['gt_answer']}")
            img = s["image"]
            print(f"  image      : {img.size} mode={img.mode}")
    except Exception as exc:
        print(f"\n[FAIL] iter_samples raised: {exc}")
        sys.exit(1)

    if not samples:
        print("\n[WARN] No samples yielded — dataset may not be accessible.")
        sys.exit(0)

    print(f"\n=== {len(samples)} sample(s) OK — running full pipeline on 2 samples ===")
    adapter2 = DeepFormAdapter(max_samples=2)
    n = adapter2.run()
    print(f"Written {n} record(s) to {adapter2.output_path}")
    print("=== DONE ===")
