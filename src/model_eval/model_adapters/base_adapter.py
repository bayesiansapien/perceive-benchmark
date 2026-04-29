"""Base adapter interface for all model adapters."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

class BaseModelAdapter(ABC):
    def __init__(self, yaml_key: str, model_cfg: dict, budget_level: str):
        self.yaml_key = yaml_key
        self.config = model_cfg
        self.budget_level = budget_level
        self.config_id = f"{yaml_key}_{budget_level}"
        self.budget_tokens = {"B0": 0, "B1": 1024, "B2": 4096, "B3": 16384}[budget_level]
        self.model_name = model_cfg.get("name", "")
        self.provider = model_cfg.get("provider", "")
        self.tier = yaml_key[0].upper()

    @abstractmethod
    def call(self, image_b64: str, query: str) -> dict:
        """
        Run inference on one sample.
        Returns:
            answer: str           — raw model output (before extraction)
            input_tokens: int
            output_tokens: int
            reasoning_tokens: int — 0 for B0
            latency_ms: int
        """

    def compute_cost(self, input_tokens: int, output_tokens: int, reasoning_tokens: int = 0) -> float:
        cfg = self.config
        return round(
            input_tokens  / 1_000_000 * cfg.get("cost_per_1M_input", 0) +
            output_tokens / 1_000_000 * cfg.get("cost_per_1M_output", 0) +
            reasoning_tokens / 1_000_000 * cfg.get("cost_per_1M_reasoning", 0),
            8
        )
