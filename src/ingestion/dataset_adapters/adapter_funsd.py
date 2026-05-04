"""
DocRouteBench: FUNSD Dataset Adapter

Task: T2 (Structured Extraction)
HuggingFace ID: nielsr/funsd-layoutlmv3
Split: test
Metric: field_f1

FUNSD is an NER / form-understanding dataset. Each example contains a
form image, a flat list of words, and NER tags. The adapter converts it
into extraction-QA pairs:

  NER tag semantics
  -----------------
  0 = other
  1 = header
  2 = question  (a form field label)
  3 = answer    (the value for the preceding question)

Strategy
--------
The HF dataset stores entities in `words` / `ner_tags` at the token level.
We reconstruct entity spans by grouping consecutive tokens that share the
same non-zero tag, then pair each question-entity with the answer-entity
that immediately follows it (or the nearest answer span in document order).

One DocRouteBench sample is emitted per linked (question, answer) pair.
If no linked pairs can be found in a form, a single fallback sample is
emitted asking the model to list all fields.
"""

from __future__ import annotations
import logging
from typing import Iterator, List, Optional, Tuple

from ..base_adapter import BaseAdapter, make_sample_id

logger = logging.getLogger(__name__)

# BIO NER tag indices (nielsr/funsd-layoutlmv3 encoding)
TAG_OTHER      = 0
TAG_B_HEADER   = 1
TAG_I_HEADER   = 2
TAG_B_QUESTION = 3
TAG_I_QUESTION = 4
TAG_B_ANSWER   = 5
TAG_I_ANSWER   = 6

# Semantic group for each BIO tag (used for entity merging and pair linking)
_TAG_GROUP = {
    TAG_OTHER: "other",
    TAG_B_HEADER: "header", TAG_I_HEADER: "header",
    TAG_B_QUESTION: "question", TAG_I_QUESTION: "question",
    TAG_B_ANSWER: "answer", TAG_I_ANSWER: "answer",
}

# Legacy constants kept for _link_pairs compatibility
TAG_QUESTION = "question"
TAG_ANSWER   = "answer"


def _build_entities(words: List[str], ner_tags: List[int]) -> List[dict]:
    """
    Collapse a flat word/tag sequence into a list of entity dicts:
        {"text": str, "tag": str, "start_tok": int, "end_tok": int}

    BIO encoding: B-* starts a new entity, I-* continues the current one.
    Tokens with tag=0 (other) are discarded.

    The dataset field is 'tokens' (not 'words') in nielsr/funsd-layoutlmv3.
    """
    entities: List[dict] = []
    if not words:
        return entities

    current_group: Optional[str] = None
    current_tokens: List[str] = []
    start_tok: int = 0

    def flush(end_tok: int) -> None:
        if current_group is not None and current_group != "other" and current_tokens:
            entities.append({
                "text": " ".join(current_tokens),
                "tag": current_group,
                "start_tok": start_tok,
                "end_tok": end_tok,
            })

    for i, (word, tag) in enumerate(zip(words, ner_tags)):
        group = _TAG_GROUP.get(tag, "other")
        is_begin = tag in (TAG_B_HEADER, TAG_B_QUESTION, TAG_B_ANSWER)

        if group == current_group and not is_begin:
            # Continuation of the same entity
            current_tokens.append(word)
        else:
            # New entity (or tag changed)
            flush(i)
            current_group = group
            current_tokens = [word] if group != "other" else []
            start_tok = i

    flush(len(words))
    return entities


def _link_pairs(entities: List[dict]) -> List[Tuple[str, str]]:
    """
    Pair each question entity with the answer entity that immediately
    follows it in document order. Returns list of (question_text, answer_text).
    """
    pairs: List[Tuple[str, str]] = []
    i = 0
    while i < len(entities):
        ent = entities[i]
        if ent["tag"] == TAG_QUESTION:
            # Look ahead for the next answer entity
            j = i + 1
            while j < len(entities) and entities[j]["tag"] not in (TAG_QUESTION, TAG_ANSWER):
                j += 1
            if j < len(entities) and entities[j]["tag"] == TAG_ANSWER:
                pairs.append((ent["text"], entities[j]["text"]))
                i = j + 1
                continue
        i += 1
    return pairs


class FunsdAdapter(BaseAdapter):
    dataset_name = "funsd"
    task_type = "T2"
    metric = "field_f1"

    def __init__(self, max_samples: Optional[int] = None, seed: int = 42):
        super().__init__(max_samples=max_samples, seed=seed)

    def iter_samples(self) -> Iterator[dict]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required. Install it with: pip install datasets"
            ) from exc

        logger.info("[funsd] Loading dataset nielsr/funsd-layoutlmv3 (split=test) ...")
        try:
            ds = load_dataset(
                "nielsr/funsd-layoutlmv3",
                split="test",
            )
        except Exception as exc:
            logger.error(f"[funsd] Failed to load dataset: {exc}")
            raise

        logger.info(f"[funsd] Dataset loaded: {len(ds)} forms available")

        emitted = 0

        for form_idx, row in enumerate(ds):
            words    = row.get("tokens", row.get("words", []))
            ner_tags = row.get("ner_tags", [])
            image    = row["image"]  # PIL Image

            # Build entities and link question→answer pairs
            entities = _build_entities(words, ner_tags)
            pairs    = _link_pairs(entities)

            if pairs:
                for pair_idx, (q_text, a_text) in enumerate(pairs):
                    sample_id = f"funsd_test_{form_idx:06d}_{pair_idx:04d}"
                    query = f"What is the value of '{q_text}' in this form?"

                    yield {
                        "sample_id": sample_id,
                        "query": query,
                        "gt_answer": a_text,
                        "gt_answer_aliases": [],
                        "image": image,
                        "task_type": self.task_type,
                        "correctness_metric": self.metric,
                        "source_split": "test",
                        "doc_type": "form",
                        "num_pages": 1,
                        "has_table": False,
                        "has_chart": False,
                        "has_figure": False,
                        "has_handwriting": False,
                    }

                    emitted += 1
                    if self.max_samples and emitted >= self.max_samples:
                        return
            else:
                # Fallback: no linked pairs found, ask model to list all fields
                sample_id = f"funsd_test_{form_idx:06d}_0000"
                query = "List all fields and their values present in this form."

                # Collect any answer text as a best-effort gt
                answer_entities = [e["text"] for e in entities if e["tag"] == TAG_ANSWER]
                gt_answer = "; ".join(answer_entities) if answer_entities else ""

                yield {
                    "sample_id": sample_id,
                    "query": query,
                    "gt_answer": gt_answer,
                    "gt_answer_aliases": [],
                    "image": image,
                    "task_type": self.task_type,
                    "correctness_metric": self.metric,
                    "source_split": "test",
                    "doc_type": "form",
                    "num_pages": 1,
                    "has_table": False,
                    "has_chart": False,
                    "has_figure": False,
                    "has_handwriting": False,
                }

                emitted += 1
                if self.max_samples and emitted >= self.max_samples:
                    return


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=== FUNSD Adapter Smoke Test (5 samples) ===")
    adapter = FunsdAdapter(max_samples=5)

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
            f"  [{i}] id={s['sample_id']}\n"
            f"       query={s['query']!r}\n"
            f"       gt_answer={s['gt_answer']!r}\n"
            f"       img_size={img.size}"
        )
        # Verify required keys
        for key in ("sample_id", "query", "gt_answer", "gt_answer_aliases", "image",
                    "task_type", "correctness_metric"):
            assert key in s, f"Missing key: {key}"
        assert s["task_type"] == "T2", f"Wrong task_type: {s['task_type']}"
        assert s["correctness_metric"] == "field_f1", f"Wrong metric: {s['correctness_metric']}"
        assert s["doc_type"] == "form"
        assert s["has_table"] is False
        assert isinstance(s["gt_answer_aliases"], list)

    print(f"\nAll {len(samples)} samples passed verification.")

    # Quick end-to-end run (writes images + JSONL for 5 samples)
    print("\nRunning adapter.run() for 5 samples ...")
    n = adapter.run()
    print(f"adapter.run() wrote {n} samples to {adapter.output_path}")
    print("Smoke test PASSED.")
