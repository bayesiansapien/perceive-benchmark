"""
PERCEIVE Router — shared configuration.

Defines the 24 valid (model, budget) configurations, cost model,
feature columns, and tier mappings used by all router modules.

Cost model uses actual measured costs from API evaluation runs
(api_results_anchor/validation/remaining.jsonl), not budget-cap estimates.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Budget definitions ────────────────────────────────────────────────────────

BUDGET_TOKENS = {"B0": 0, "B1": 1024, "B2": 4096, "B3": 16384}

# ── Model definitions ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelSpec:
    yaml_key: str
    name: str
    tier: str          # "A", "B", "C"
    tier_idx: int      # 0, 1, 2
    input_rate: float  # $/1M input tokens
    output_rate: float # $/1M output tokens
    reasoning_rate: float  # $/1M reasoning tokens
    valid_budgets: tuple[str, ...]

MODELS = {
    "a2_flashlite":  ModelSpec("a2_flashlite",  "Flash-Lite",   "A", 0,  0.05,  0.20,  0.20, ("B1","B2","B3")),
    "a4_gpt54nano":  ModelSpec("a4_gpt54nano",  "GPT-nano",     "A", 0,  0.04,  0.16,  0.16, ("B0",)),
    "b1_gpt54mini":  ModelSpec("b1_gpt54mini",  "GPT-mini",     "B", 1,  0.20,  0.80,  0.80, ("B0","B1","B2","B3")),
    "b3_sonnet":     ModelSpec("b3_sonnet",     "Sonnet",       "B", 1,  3.00, 15.00,  3.00, ("B0","B1","B2","B3")),
    "c1_gpt54":      ModelSpec("c1_gpt54",      "GPT-5.4",     "C", 2,  3.00, 12.00, 12.00, ("B0","B1","B2","B3")),
    "c2_opus":       ModelSpec("c2_opus",       "Opus",         "C", 2, 15.00, 75.00, 15.00, ("B0","B1","B2","B3")),
    "c3_gemini_pro": ModelSpec("c3_gemini_pro", "Gemini-Pro",   "C", 2,  1.25,  5.00,  3.50, ("B0","B1","B2","B3")),
}

MODEL_LIST = list(MODELS.keys())  # 7 models, stable order
MODEL_IDX = {k: i for i, k in enumerate(MODEL_LIST)}

# ── 24 valid configurations ──────────────────────────────────────────────────

CONFIG_LIST: list[tuple[str, str]] = []
for yk in MODEL_LIST:
    for bl in MODELS[yk].valid_budgets:
        CONFIG_LIST.append((yk, bl))

CONFIG_IDX = {c: i for i, c in enumerate(CONFIG_LIST)}
N_CONFIGS = len(CONFIG_LIST)  # 24
N_MODELS = len(MODEL_LIST)    # 7

# ── Actual cost model ────────────────────────────────────────────────────────
# Measured average cost per API call from evaluation runs (71,202 total calls).
# Includes actual input tokenization (provider-specific image encoding),
# actual output tokens, and actual reasoning token usage.

ACTUAL_COSTS: dict[tuple[str, str], float] = {
    ("a2_flashlite", "B1"): 0.00014246,
    ("a2_flashlite", "B2"): 0.00009332,
    ("a2_flashlite", "B3"): 0.00043329,
    ("a4_gpt54nano", "B0"): 0.00004894,
    ("b1_gpt54mini", "B0"): 0.00022614,
    ("b1_gpt54mini", "B1"): 0.00043393,
    ("b1_gpt54mini", "B2"): 0.00103158,
    ("b1_gpt54mini", "B3"): 0.00192311,
    ("b3_sonnet",    "B0"): 0.00328967,
    ("b3_sonnet",    "B1"): 0.00390005,
    ("b3_sonnet",    "B2"): 0.00483883,
    ("b3_sonnet",    "B3"): 0.00526220,
    ("c1_gpt54",     "B0"): 0.00316043,
    ("c1_gpt54",     "B1"): 0.00643326,
    ("c1_gpt54",     "B2"): 0.01486339,
    ("c1_gpt54",     "B3"): 0.03047738,
    ("c2_opus",      "B0"): 0.01557899,
    ("c2_opus",      "B1"): 0.02181283,
    ("c2_opus",      "B2"): 0.02325841,
    ("c2_opus",      "B3"): 0.02409178,
    ("c3_gemini_pro","B0"): 0.00361754,
    ("c3_gemini_pro","B1"): 0.00420533,
    ("c3_gemini_pro","B2"): 0.00464909,
    ("c3_gemini_pro","B3"): 0.00534961,
}

CONFIG_COSTS = [ACTUAL_COSTS[c] for c in CONFIG_LIST]

CHEAPEST_CONFIG_IDX = min(range(N_CONFIGS), key=lambda i: CONFIG_COSTS[i])
CHEAPEST_CONFIG = CONFIG_LIST[CHEAPEST_CONFIG_IDX]

MOST_EXPENSIVE_CONFIG_IDX = max(range(N_CONFIGS), key=lambda i: CONFIG_COSTS[i])
MOST_EXPENSIVE_CONFIG = CONFIG_LIST[MOST_EXPENSIVE_CONFIG_IDX]

# Cheapest config per tier (for non-routable tier-matched routing)
TIER_CHEAPEST_CONFIG_IDX: dict[str, int] = {}
for _tier_label in ["A", "B", "C"]:
    _tier_indices = [i for i in range(N_CONFIGS) if MODELS[CONFIG_LIST[i][0]].tier == _tier_label]
    TIER_CHEAPEST_CONFIG_IDX[_tier_label] = min(_tier_indices, key=lambda i: CONFIG_COSTS[i])

COMPLEXITY_TIER_TO_MODEL_TIER = {1: "A", 2: "B", 3: "C"}

# Per-model average cost (across all budgets for that model)
MODEL_AVG_COSTS = {}
for yk in MODEL_LIST:
    model_configs = [(yk, bl) for bl in MODELS[yk].valid_budgets]
    MODEL_AVG_COSTS[yk] = sum(ACTUAL_COSTS[c] for c in model_configs) / len(model_configs)

# ── Tier mapping ──────────────────────────────────────────────────────────────

def model_tier(yaml_key: str) -> int:
    """Return numeric tier: A=0, B=1, C=2."""
    return MODELS[yaml_key].tier_idx


# ── Feature definitions ──────────────────────────────────────────────────────

TASK_TYPES = ["T1", "T2", "T3", "T4", "T5", "element_localization"]
DOC_TYPES_TOP = [
    "academic_paper", "chart", "document", "infographic", "presentation",
    "receipt", "scene_text", "table", "webpage", "form",
]
KEYWORD_FLAGS = ["compare", "across", "locate", "total", "how many",
                 "all", "table", "chart"]
QUESTION_PREFIXES = ["what", "where", "how", "which", "is"]
