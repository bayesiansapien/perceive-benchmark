"""
PERCEIVE Router: evaluation metrics.

Computes accuracy, cost, oracle efficiency, and breakdowns for routing strategies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from scripts.router.config import (
    COMPLEXITY_TIER_TO_MODEL_TIER,
    CONFIG_COSTS,
    CONFIG_LIST,
    MOST_EXPENSIVE_CONFIG_IDX,
    N_CONFIGS,
    TASK_TYPES,
    MODELS,
    TIER_CHEAPEST_CONFIG_IDX,
)

if TYPE_CHECKING:
    from typing import Any


@dataclass
class RouterMetrics:
    """Evaluation metrics for a routing strategy."""
    name: str
    accuracy: float              # fraction of ALL samples where selected config is correct
    accuracy_routable: float     # fraction of ROUTABLE samples where selected config is correct
    total_cost: float            # sum of cost for all queries
    avg_cost: float              # mean cost per query
    cost_per_correct: float      # total_cost / n_correct (inf if n_correct=0)
    oracle_efficiency: float     # % of oracle cost savings achieved (routable only)
    n_correct: int
    n_total: int
    n_routable: int
    # NEW fields:
    total_cost_routable: float       # cost for routable samples only
    avg_cost_routable: float         # mean cost per routable query
    cascade_total_cost: float | None # cumulative cascade cost (only for cascade routers)


def evaluate_router(
    name: str,
    predictions: np.ndarray,
    dataset: Any,  # RouterDataset
    cascade_costs: np.ndarray | None = None,
) -> RouterMetrics:
    """
    Compute metrics for a routing strategy's predictions.

    Args:
        name: Router name for display
        predictions: (n_samples,) int array of config indices
        dataset: RouterDataset with eval_correct, is_routable, etc.
        cascade_costs: Optional (n_samples,) float array of cumulative cascade costs.
                       If provided, stored as cascade_total_cost in metrics.

    Returns:
        RouterMetrics with all computed metrics
    """
    n_samples = len(predictions)
    assert n_samples == len(dataset.eval_correct), "Predictions length mismatch"

    # Compute correctness
    correct_mask = dataset.eval_correct[np.arange(n_samples), predictions]
    n_correct = correct_mask.sum()

    # Routable subset accuracy
    routable_mask = dataset.is_routable
    n_routable = routable_mask.sum()
    if n_routable > 0:
        n_correct_routable = correct_mask[routable_mask].sum()
        accuracy_routable = n_correct_routable / n_routable
    else:
        accuracy_routable = 0.0

    # Cost computation (always use single-call CONFIG_COSTS for total_cost/avg_cost)
    costs = np.array([CONFIG_COSTS[pred] for pred in predictions])
    total_cost = costs.sum()
    avg_cost = total_cost / n_samples

    # Cascade total cost (optional, separate tracking)
    cascade_total_cost_value = float(cascade_costs.sum()) if cascade_costs is not None else None

    cost_per_correct = total_cost / n_correct if n_correct > 0 else float('inf')

    # Routable-only cost metrics
    routable_indices = np.where(routable_mask)[0]
    if n_routable > 0:
        total_cost_routable = sum(costs[i] for i in routable_indices)
        avg_cost_routable = total_cost_routable / n_routable
    else:
        total_cost_routable = 0.0
        avg_cost_routable = 0.0

    # Oracle efficiency computation (routable samples only)
    complexity_tiers = getattr(dataset, "complexity_tiers", None)
    oracle_predictions = compute_oracle_predictions(dataset, complexity_tiers=complexity_tiers)

    # Oracle cost: cheapest correct config cost for each routable sample
    oracle_cost_routable = sum(CONFIG_COSTS[oracle_predictions[i]] for i in routable_indices)

    # Always-best cost for routable samples only
    always_best_cost_routable = len(routable_indices) * CONFIG_COSTS[MOST_EXPENSIVE_CONFIG_IDX]

    # Router cost for routable samples only
    router_cost_routable = total_cost_routable

    if always_best_cost_routable > oracle_cost_routable:
        oracle_efficiency = (
            (always_best_cost_routable - router_cost_routable)
            / (always_best_cost_routable - oracle_cost_routable)
        ) * 100
        # Clamp to [0, 100]
        oracle_efficiency = max(0.0, min(100.0, oracle_efficiency))
    else:
        # Edge case: oracle cost equals or exceeds always-best (shouldn't happen)
        oracle_efficiency = 0.0

    return RouterMetrics(
        name=name,
        accuracy=n_correct / n_samples,
        accuracy_routable=accuracy_routable,
        total_cost=total_cost,
        avg_cost=avg_cost,
        cost_per_correct=cost_per_correct,
        oracle_efficiency=oracle_efficiency,
        n_correct=int(n_correct),
        n_total=n_samples,
        n_routable=int(n_routable),
        total_cost_routable=total_cost_routable,
        avg_cost_routable=avg_cost_routable,
        cascade_total_cost=cascade_total_cost_value,
    )


def compute_oracle_predictions(
    dataset: Any,
    complexity_tiers: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return oracle predictions: cheapest correct config per sample.

    For routable samples: choose cheapest config where eval_correct is True.
    For non-routable samples: choose cheapest config within the sample's
    complexity tier (tier-matched routing, spend appropriately even when
    no config produces a correct answer).

    Args:
        dataset: RouterDataset with eval_correct, is_routable
        complexity_tiers: (n_samples,) int array of tier_final values (1/2/3).
            If None, falls back to cheapest overall config for non-routable.

    Returns:
        (n_samples,) int array of config indices
    """
    n_samples = len(dataset.eval_correct)
    predictions = np.zeros(n_samples, dtype=np.int32)

    # Sort configs by cost (ascending)
    sorted_config_indices = np.argsort(CONFIG_COSTS)

    for i in range(n_samples):
        if dataset.is_routable[i]:
            for config_idx in sorted_config_indices:
                if dataset.eval_correct[i, config_idx]:
                    predictions[i] = config_idx
                    break
        else:
            if complexity_tiers is not None:
                tier_val = int(complexity_tiers[i])
                model_tier = COMPLEXITY_TIER_TO_MODEL_TIER.get(tier_val, "A")
                predictions[i] = TIER_CHEAPEST_CONFIG_IDX[model_tier]
            else:
                predictions[i] = sorted_config_indices[0]

    return predictions


def compute_pareto_frontier(
    all_metrics: list[RouterMetrics],
) -> list[tuple[str, float, float]]:
    """
    Return Pareto-optimal (name, avg_cost, accuracy) points.

    A point is Pareto-optimal if no other point has both lower cost AND higher accuracy.

    Args:
        all_metrics: List of RouterMetrics to analyze

    Returns:
        List of (name, avg_cost, accuracy) tuples that are Pareto-optimal
    """
    points = [(m.name, m.avg_cost, m.accuracy) for m in all_metrics]
    pareto = []

    for i, (name_i, cost_i, acc_i) in enumerate(points):
        is_dominated = False
        for j, (name_j, cost_j, acc_j) in enumerate(points):
            if i == j:
                continue
            # j dominates i if j has lower cost AND higher accuracy
            if cost_j < cost_i and acc_j > acc_i:
                is_dominated = True
                break
        if not is_dominated:
            pareto.append((name_i, cost_i, acc_i))

    # Sort by cost ascending
    pareto.sort(key=lambda x: x[1])
    return pareto


def per_tier_breakdown(
    name: str,
    predictions: np.ndarray,
    dataset: Any,
    benchmark_data: dict[str, dict],
) -> dict[int, dict]:
    """
    Return per-tier accuracy and cost breakdown.

    Args:
        name: Router name
        predictions: (n_samples,) config indices
        dataset: RouterDataset with sample_ids, eval_correct
        benchmark_data: sample_id → record with tier_final field

    Returns:
        {tier: {accuracy, n_samples, avg_cost}} for tier_final = 1, 2, 3
    """
    tier_stats = {1: [], 2: [], 3: []}
    tier_correct = {1: 0, 2: 0, 3: 0}
    tier_costs = {1: [], 2: [], 3: []}

    for i, sample_id in enumerate(dataset.sample_ids):
        record = benchmark_data.get(sample_id)
        if record is None:
            continue

        tier = record.get("tier_final")
        if tier not in [1, 2, 3]:
            continue

        tier_stats[tier].append(i)

        # Check correctness
        pred_config = predictions[i]
        if dataset.eval_correct[i, pred_config]:
            tier_correct[tier] += 1

        # Accumulate cost
        tier_costs[tier].append(CONFIG_COSTS[pred_config])

    result = {}
    for tier in [1, 2, 3]:
        n = len(tier_stats[tier])
        if n > 0:
            result[tier] = {
                "accuracy": tier_correct[tier] / n,
                "n_samples": n,
                "avg_cost": np.mean(tier_costs[tier]),
            }
        else:
            result[tier] = {
                "accuracy": 0.0,
                "n_samples": 0,
                "avg_cost": 0.0,
            }

    return result


def per_task_breakdown(
    name: str,
    predictions: np.ndarray,
    dataset: Any,
    benchmark_data: dict[str, dict],
) -> dict[str, dict]:
    """
    Return per-task-type accuracy and cost breakdown.

    Args:
        name: Router name
        predictions: (n_samples,) config indices
        dataset: RouterDataset with sample_ids, eval_correct
        benchmark_data: sample_id → record with task_type field

    Returns:
        {task_type: {accuracy, n_samples, avg_cost}} for each task type
    """
    task_stats = {tt: [] for tt in TASK_TYPES}
    task_correct = {tt: 0 for tt in TASK_TYPES}
    task_costs = {tt: [] for tt in TASK_TYPES}

    for i, sample_id in enumerate(dataset.sample_ids):
        record = benchmark_data.get(sample_id)
        if record is None:
            continue

        task_type = record.get("task_type")
        if task_type not in TASK_TYPES:
            continue

        task_stats[task_type].append(i)

        # Check correctness
        pred_config = predictions[i]
        if dataset.eval_correct[i, pred_config]:
            task_correct[task_type] += 1

        # Accumulate cost
        task_costs[task_type].append(CONFIG_COSTS[pred_config])

    result = {}
    for task_type in TASK_TYPES:
        n = len(task_stats[task_type])
        if n > 0:
            result[task_type] = {
                "accuracy": task_correct[task_type] / n,
                "n_samples": n,
                "avg_cost": np.mean(task_costs[task_type]),
            }
        else:
            result[task_type] = {
                "accuracy": 0.0,
                "n_samples": 0,
                "avg_cost": 0.0,
            }

    return result


def format_results_table(all_metrics: list[RouterMetrics]) -> str:
    """
    Format a nice ASCII table of all routers' metrics for printing.

    Args:
        all_metrics: List of RouterMetrics to display

    Returns:
        Formatted ASCII table string
    """
    # Sort by accuracy descending
    sorted_metrics = sorted(all_metrics, key=lambda m: m.accuracy, reverse=True)

    lines = []

    # Header
    header = (
        f"{'Router':<25} | "
        f"{'Accuracy':>8} | "
        f"{'Avg Cost':>10} | "
        f"{'Cost/Corr':>10} | "
        f"{'Oracle Eff':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for m in sorted_metrics:
        cost_per_correct_str = f"${m.cost_per_correct:.6f}" if m.cost_per_correct != float('inf') else "inf"

        row = (
            f"{m.name:<25} | "
            f"{m.accuracy * 100:>7.2f}% | "
            f"${m.avg_cost:>9.6f} | "
            f"{cost_per_correct_str:>10} | "
            f"{m.oracle_efficiency:>9.2f}%"
        )
        lines.append(row)

    return "\n".join(lines)
