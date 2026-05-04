"""DocRouteBench: Cost Calculator for model evaluation results."""


def compute_cost(
    model_cfg: dict,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int = 0,
) -> float:
    """
    Compute cost in USD for one API call.

    Args:
        model_cfg:        Model config dict from model_pool.yaml
        input_tokens:     Prompt token count
        output_tokens:    Completion token count
        reasoning_tokens: Internal reasoning/thinking token count (0 for B0)
    """
    return round(
        input_tokens     / 1_000_000 * model_cfg.get("cost_per_1M_input", 0.0)
        + output_tokens  / 1_000_000 * model_cfg.get("cost_per_1M_output", 0.0)
        + reasoning_tokens / 1_000_000 * model_cfg.get("cost_per_1M_reasoning", 0.0),
        8,
    )
