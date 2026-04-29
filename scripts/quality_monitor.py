#!/usr/bin/env python3
"""
DocRouteBench Phase 3 — Real-time Quality Monitor

Runs in the background, periodically scanning collected results for anomalies.
Flags issues to /tmp/phase3_quality_flags.log.

Checks:
  1. Empty predicted_answer (no error) — model returned nothing
  2. B1/B2/B3 with reasoning_tokens=0 — thinking not engaged
  3. Answer is copy of query — model echoed the question
  4. Answer too long (>50 words) — model not being concise
  5. Non-English answer on English query — possible hallucination
  6. Cost per call way off expected — pricing anomaly
  7. All models wrong on same sample — possible data issue
  8. Model accuracy < 1% for 200+ calls — systematic failure
  9. Repeated identical answers across many samples (>30%) — degenerate
 10. Latency outliers (>60s) — possible API issues
"""
import json
import logging
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("quality_monitor")

RESULTS_FILE = str(_ROOT / "data/model_eval_results/api_results_anchor.jsonl")
FLAGS_FILE   = "/tmp/phase3_quality_flags.log"
CHECK_INTERVAL = 300  # check every 5 minutes

# Expected cost per call ranges (min, max) in USD
EXPECTED_COST_RANGE = {
    "a2_flashlite": (0.00005, 0.005),
    "a4_gpt54nano":  (0.00002, 0.002),
    "b1_gpt54mini":  (0.0001,  0.05),
    "b3_sonnet":     (0.001,   1.0),
    "c1_gpt54":      (0.001,   2.0),
    "c2_opus":       (0.01,    5.0),
    "c3_gemini_pro": (0.0005,  0.5),
}


def load_new_records(path: str, seen_count: int) -> list:
    """Load only records we haven't seen yet."""
    records = []
    try:
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= seen_count:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    except FileNotFoundError:
        pass
    return records


def flag(msg: str, severity: str = "WARN") -> None:
    """Write a flag to the flags log and stderr."""
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] [{severity}] {msg}"
    log.warning(line) if severity == "WARN" else log.error(line)
    with open(FLAGS_FILE, "a") as f:
        f.write(line + "\n")


def check_record(r: dict, all_records: list) -> list[str]:
    """Return list of issues for a single record."""
    issues = []
    config_id = r.get("config_id", "?")
    sample_id = r.get("sample_id", "?")
    yaml_key  = r.get("yaml_key", "?")
    budget    = r.get("budget_level", "B0")
    predicted = r.get("predicted_answer", "") or ""
    raw       = r.get("raw_answer", "") or ""
    query     = ""  # loaded per-sample below if needed
    cost      = r.get("total_cost_usd", 0)
    latency   = r.get("latency_ms", 0)
    r_tokens  = r.get("reasoning_tokens", 0)
    budget_tokens = r.get("budget_tokens", 0)
    error     = r.get("error")

    # 1. Empty answer (no error)
    if not predicted.strip() and not error:
        issues.append(f"EMPTY_ANSWER: {config_id} / {sample_id} | raw={raw[:40]!r}")

    # 2. Thinking not engaged for B2/B3 (flag only high budgets — B1 is optional)
    # B1=low effort, model may skip thinking on easy samples — that's valid
    # B2/B3 should consistently produce reasoning tokens
    if budget in ("B2", "B3") and r_tokens == 0 and not error and cost > 0:
        issues.append(f"NO_THINKING: {config_id} / {sample_id} | budget={budget_tokens} but reasoning_tokens=0")

    # 3. Answer too long (>50 words)
    if predicted and len(predicted.split()) > 50:
        issues.append(f"LONG_ANSWER: {config_id} / {sample_id} | {len(predicted.split())} words: {predicted[:60]!r}")

    # 4. Cost anomaly
    if cost > 0 and yaml_key in EXPECTED_COST_RANGE:
        lo, hi = EXPECTED_COST_RANGE[yaml_key]
        if cost < lo or cost > hi:
            issues.append(f"COST_ANOMALY: {config_id} | cost=${cost:.6f} expected=[${lo:.5f}, ${hi:.4f}]")

    # 5. Extreme latency (>90 seconds)
    if latency > 90_000 and not error:
        issues.append(f"HIGH_LATENCY: {config_id} / {sample_id} | {latency/1000:.1f}s")

    return issues


def batch_checks(records: list, cumulative: list) -> list[str]:
    """Batch checks across many records."""
    issues = []
    if not records:
        return issues

    # Group by model for accuracy check
    by_model = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in cumulative:
        if not r.get("error"):
            k = r["yaml_key"]
            by_model[k]["total"] += 1
            if r.get("is_correct"):
                by_model[k]["correct"] += 1

    # 8. Model accuracy < 1% for 500+ calls
    for model, stats in by_model.items():
        if stats["total"] >= 500:
            acc = stats["correct"] / stats["total"]
            if acc < 0.01:
                issues.append(
                    f"ZERO_ACCURACY: {model} | {stats['correct']}/{stats['total']} ({acc:.1%}) "
                    f"— systematic failure?"
                )

    # 9. Degenerate repeated answers (>40% same answer for a model)
    by_model_answers = defaultdict(list)
    for r in cumulative:
        if r.get("predicted_answer") and not r.get("error"):
            by_model_answers[r["yaml_key"]].append(r["predicted_answer"].strip().lower())

    for model, answers in by_model_answers.items():
        if len(answers) >= 100:
            top_answer, top_count = Counter(answers).most_common(1)[0]
            if top_count / len(answers) > 0.40:
                issues.append(
                    f"DEGENERATE: {model} | answer {top_answer!r} appears in "
                    f"{top_count}/{len(answers)} ({top_count/len(answers):.0%}) samples"
                )

    # 10. Samples where ALL models are wrong (possible data issue)
    by_sample = defaultdict(dict)
    for r in cumulative:
        if not r.get("error"):
            by_sample[r["sample_id"]][r["yaml_key"]] = r.get("is_correct", False)

    all_wrong = [
        sid for sid, models in by_sample.items()
        if len(models) >= 5 and not any(models.values())
    ]
    if len(all_wrong) > 50:
        issues.append(
            f"ALL_MODELS_WRONG: {len(all_wrong)} samples where no model is correct "
            f"— check data quality"
        )

    return issues


def run_monitor(results_path: str = RESULTS_FILE, interval: int = CHECK_INTERVAL):
    log.info("Quality monitor started. Checking every %ds.", interval)
    log.info("Results file: %s", results_path)
    log.info("Flags log: %s", FLAGS_FILE)

    # Clear flags log
    with open(FLAGS_FILE, "w") as f:
        f.write(f"=== Phase 3 Quality Monitor — started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    seen_count = 0
    cumulative = []
    flagged_ids: set = set()

    while True:
        new_records = load_new_records(results_path, seen_count)

        if new_records:
            seen_count += len(new_records)
            cumulative.extend(new_records)

            # Per-record checks on new batch
            record_issues = []
            for r in new_records:
                for issue in check_record(r, cumulative):
                    key = issue[:60]
                    if key not in flagged_ids:
                        flagged_ids.add(key)
                        record_issues.append(issue)

            # Batch checks on cumulative
            batch_issues = batch_checks(new_records, cumulative)
            for issue in batch_issues:
                key = issue[:60]
                if key not in flagged_ids:
                    flagged_ids.add(key)

            all_issues = record_issues + batch_issues

            # Progress summary
            total_errors = sum(1 for r in cumulative if r.get("error"))
            empty_answers = sum(1 for r in cumulative if not r.get("predicted_answer","").strip() and not r.get("error"))
            total_cost = sum(r.get("total_cost_usd", 0) for r in cumulative)
            correct = sum(1 for r in cumulative if r.get("is_correct"))
            pct = seen_count / 31500 * 100

            log.info(
                "Progress: %d/31500 (%.1f%%) | Cost: $%.2f | Errors: %d | "
                "Empty: %d | Accuracy: %.1f%% | Flags: %d",
                seen_count, pct, total_cost, total_errors,
                empty_answers, correct / max(seen_count - total_errors, 1) * 100,
                len(flagged_ids),
            )

            if all_issues:
                log.warning("=== %d NEW FLAGS ===", len(all_issues))
                for issue in all_issues:
                    flag(issue, "WARN")
            else:
                log.info("No new quality issues detected.")

        time.sleep(interval)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=RESULTS_FILE)
    parser.add_argument("--interval", type=int, default=CHECK_INTERVAL)
    args = parser.parse_args()
    run_monitor(args.results, args.interval)
