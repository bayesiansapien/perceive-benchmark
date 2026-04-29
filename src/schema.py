"""
DocRouteBench — Unified Data Schema

Four normalized tables:
  Sample         → task definition (no results)
  ModelConfig    → static model config (28 configs)
  Result         → per (sample, config) observation
  Annotation     → VDS/RDS/SES complexity labels
  RoutingGT      → derived GT routing labels
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
import json


# ── Budget level constants ────────────────────────────────────────────
BUDGET_LEVELS = {
    "B0": 0,        # instruct / no thinking
    "B1": 1024,     # light thinking
    "B2": 4096,     # moderate thinking
    "B3": 16384,    # deep thinking
}

TASK_TYPES = ["T1", "T2", "T3", "T4", "T5", "T6"]

VALID_METRICS = [
    "anls",
    "relaxed_accuracy",
    "field_f1",
    "exact_match",
    "teds",
    "iou",
    "rouge_cider",
    "denotation",
    "vqa_accuracy",
    "slidevqa_em",
]


# ── Table 1: Sample ───────────────────────────────────────────────────
@dataclass
class Sample:
    """
    One row per benchmark sample. Contains the task definition only.
    No model results — those live in Result table.
    Question text stored once, not repeated per config.
    """
    sample_id: str              # e.g. "docvqa_val_00412"
    source_dataset: str         # e.g. "DocVQA"
    source_split: str           # "validation", "test"
    task_type: str              # "T1" through "T6"
    query: str                  # the question or instruction
    gt_answer: str              # primary ground truth answer
    gt_answer_aliases: List[str] = field(default_factory=list)  # alternative accepted answers
    correctness_metric: str = "anls"
    image_path: str = ""        # relative path from project root
    num_pages: int = 1
    has_table: bool = False
    has_chart: bool = False
    has_figure: bool = False
    has_handwriting: bool = False
    doc_type: str = "document"  # form, chart, receipt, letter, etc.
    image_bytes_size: int = 0   # proxy for image complexity
    in_anchor_set: bool = False
    in_validation_set: bool = False

    def __post_init__(self):
        assert self.task_type in TASK_TYPES, \
            f"task_type must be one of {TASK_TYPES}, got {self.task_type}"
        assert self.correctness_metric in VALID_METRICS, \
            f"correctness_metric must be one of {VALID_METRICS}"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, s: str) -> "Sample":
        return cls.from_dict(json.loads(s))


# ── Table 2: ModelConfig ──────────────────────────────────────────────
@dataclass
class ModelConfig:
    """
    One row per model configuration. Static — defined once, never changes.
    Total: 28 configs from 11 models.
    """
    config_id: str              # e.g. "c2_gpt54_B2"
    model_name: str             # e.g. "gpt-5.4"
    model_id: str               # API model ID or HuggingFace ID
    provider: str               # "openai", "google", "anthropic", "self_hosted"
    tier: str                   # "A", "B", "C"
    budget_level: str           # "B0", "B1", "B2", "B3"
    budget_tokens: int          # 0, 1024, 4096, 16384
    cost_per_1M_input: float    # USD, 0 for self-hosted
    cost_per_1M_output: float
    cost_per_1M_reasoning: float = 0.0
    supports_grounding: bool = False
    supports_multipage: bool = True
    max_context_tokens: int = 128000
    is_calibration_model: bool = False  # for extension framework

    def estimate_cost(self, input_tokens: int, output_tokens: int,
                      reasoning_tokens: int = 0) -> float:
        return (
            input_tokens / 1_000_000 * self.cost_per_1M_input +
            output_tokens / 1_000_000 * self.cost_per_1M_output +
            reasoning_tokens / 1_000_000 * self.cost_per_1M_reasoning
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ── Table 3: Result ───────────────────────────────────────────────────
@dataclass
class Result:
    """
    One row per (sample, config) evaluation.
    MINIMAL — only foreign keys + direct observation data.
    No redundant fields from Sample or ModelConfig.
    """
    sample_id: str              # FK → Sample.sample_id
    config_id: str              # FK → ModelConfig.config_id
    is_correct: bool
    predicted_answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0   # 0 for B0 (instruct mode)
    total_cost_usd: float = 0.0
    latency_ms: int = 0
    result_type: str = "observed"  # "observed" or "inferred"
    inference_confidence: Optional[float] = None  # from IMC, None if observed
    timestamp: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # For inferred results, predicted_answer and timestamp don't exist
        if self.result_type == "inferred":
            d.pop("predicted_answer", None)
            d.pop("timestamp", None)
        return d

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "Result":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Table 4: ComplexityAnnotation ─────────────────────────────────────
@dataclass
class ComplexityAnnotation:
    """
    One row per sample. VDS/RDS/SES from 3-judge LLM panel.
    """
    sample_id: str              # FK → Sample.sample_id
    vds: int                    # Visual Dependency Score (1-4)
    rds: int                    # Reasoning Depth Score (1-4)
    ses: int                    # Spatial Extent Score (1-4)
    composite: float            # weighted: 0.30*VDS + 0.45*RDS + 0.25*SES
    tier: int                   # 1=easy, 2=medium, 3=hard
    n_judges: int = 2           # 2 if J1+J2 agreed, 3 if Opus tiebreak used
    had_tiebreak: bool = False
    judge_agreement: str = "strong"  # "strong" (range<=1) or "weak" (range>1)

    def __post_init__(self):
        for axis, val in [("VDS", self.vds), ("RDS", self.rds), ("SES", self.ses)]:
            assert 1 <= val <= 4, f"{axis}={val} must be in [1,4]"
        assert self.tier in [1, 2, 3], f"tier={self.tier} must be 1, 2, or 3"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Derived: RoutingGT ────────────────────────────────────────────────
@dataclass
class RoutingGT:
    """
    Derived from Results table. Computed after all model evaluations.
    GT routing labels for a sample.
    """
    sample_id: str
    gt_cost_config: Optional[str]       # cheapest correct config_id
    gt_accuracy_config: Optional[str]   # highest-tier correct config_id
    gt_pareto_config: Optional[str]     # Pareto-optimal config_id
    any_config_correct: bool
    correct_configs: List[str] = field(default_factory=list)
    n_correct: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ── I/O utilities ─────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: str, record: dict) -> None:
    """Append a single record to a JSONL file (atomic per line)."""
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def get_completed_keys(results_path: str) -> set:
    """
    Load set of (sample_id, config_id) already completed.
    Used for resume support — skip already-done calls.
    """
    completed = set()
    try:
        for r in load_jsonl(results_path):
            completed.add((r["sample_id"], r["config_id"]))
    except FileNotFoundError:
        pass
    return completed
