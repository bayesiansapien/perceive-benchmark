"""
DocRouteBench: Unified Scoring Dispatcher

Single entry point for all correctness checks:
    is_correct(predicted, ground_truth, metric, **kwargs) -> bool

Maps metric name → scorer module. Used everywhere in the benchmark.
"""

from typing import Union, List, Optional

from .anls import is_correct_anls
from .relaxed_accuracy import is_correct_relaxed_accuracy
from .field_f1 import is_correct as is_correct_field_f1
from .exact_match import is_correct as is_correct_exact_match, exact_match_any
from .teds import is_correct as is_correct_teds
from .iou import is_correct as is_correct_iou
from .rouge_cider import is_correct as is_correct_rouge
from .denotation import is_correct as is_correct_denotation
from .vqa_accuracy import is_correct as is_correct_vqa
from .slidevqa_em import is_correct as is_correct_slidevqa


def is_correct(
    predicted: Optional[str],
    ground_truth: Union[str, List[str]],
    metric: str,
    dataset: str = "",
    **kwargs,
) -> bool:
    """
    Unified binary correctness check for any metric.

    Args:
        predicted:     Model's answer (string or None)
        ground_truth:  GT answer or list of acceptable answers
        metric:        One of the VALID_METRICS defined in src/schema.py
        dataset:       Optional dataset name for metric-specific handling
        **kwargs:      Passed through to the underlying scorer

    Returns:
        True if the prediction is considered correct.
    """
    if predicted is None:
        return False

    # Normalize ground_truth to list
    if isinstance(ground_truth, str):
        gt_list = [ground_truth]
    else:
        gt_list = list(ground_truth)

    m = metric.lower().strip()

    if m == "anls":
        return is_correct_anls(predicted, gt_list, **kwargs)

    elif m == "relaxed_accuracy":
        return is_correct_relaxed_accuracy(predicted, gt_list, **kwargs)

    elif m == "field_f1":
        # Take max F1 over all GT aliases
        from .field_f1 import compute_f1
        threshold = kwargs.get("threshold", 0.5)
        return any(compute_f1(predicted, gt) >= threshold for gt in gt_list)

    elif m == "exact_match":
        return exact_match_any(predicted, gt_list, dataset=dataset)

    elif m == "teds":
        # Table structure: single GT expected
        return is_correct_teds(predicted, gt_list[0] if gt_list else "", **kwargs)

    elif m == "iou":
        # Pass all GT boxes (list of bbox strings or single bbox)
        return is_correct_iou(predicted, gt_list if len(gt_list) > 1 else gt_list[0], **kwargs)

    elif m == "rouge_cider":
        return is_correct_rouge(predicted, gt_list, **kwargs)

    elif m == "denotation":
        return is_correct_denotation(predicted, gt_list, **kwargs)

    elif m == "vqa_accuracy":
        # gt_list is the list of 10 annotators' answers
        return is_correct_vqa(predicted, gt_list, **kwargs)

    elif m == "slidevqa_em":
        question = kwargs.get("question", "")
        return is_correct_slidevqa(predicted, gt_list[0] if gt_list else "", question)

    else:
        raise ValueError(
            f"Unknown metric: '{metric}'. "
            f"Valid metrics: anls, relaxed_accuracy, field_f1, exact_match, "
            f"teds, iou, rouge_cider, denotation, vqa_accuracy, slidevqa_em"
        )


if __name__ == "__main__":
    # Smoke test: one case per metric
    tests = [
        # (predicted, gt, metric, dataset, expected)
        ("January 15", "January 15, 1994", "anls", "", True),
        ("42.1", "42", "relaxed_accuracy", "", True),
        ("Company ACME Date 01/01", "Company ACME", "field_f1", "", True),
        ("invoice", "11", "exact_match", "rvl-cdip", True),
        ("<table><tr><td>A</td></tr></table>", "<table><tr><td>A</td></tr></table>", "teds", "", True),
        ("[0.1,0.1,0.5,0.5]", "[0.1,0.1,0.5,0.5]", "iou", "", True),
        ("The revenue increased", "Revenue increased significantly", "rouge_cider", "", True),
        ("1234", "1,234", "denotation", "", True),
        ("cat", ["cat", "dog", "bird", "cat", "cat", "mouse", "cat", "cat", "cat", "cat"], "vqa_accuracy", "", True),
        ("42", "42", "slidevqa_em", "", True),
    ]

    passed = 0
    for pred, gt, metric, ds, expected in tests:
        result = is_correct(pred, gt, metric, dataset=ds)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] metric={metric:20s} pred={str(pred)[:20]:<20} expected={expected} got={result}")
        if result == expected:
            passed += 1

    print(f"\n{passed}/{len(tests)} tests passed")
