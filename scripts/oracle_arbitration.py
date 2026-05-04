#!/usr/bin/env python3
"""
Oracle arbitration: run Mistral Large 2512 on all disagreement records
from the neuro-symbolic evaluation (rule_correct != neural_correct).

Input:  data/model_eval_results/all_models_judgments_v2.jsonl  (needs_oracle=True records)
        OR data/model_eval_results/all_models_judgments.jsonl  (fallback, computes disagreements)
        data/benchmark/benchmark_5000.jsonl                    (for query context)
Output: data/model_eval_results/oracle_verdicts.jsonl
"""
import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Oracle prompt (exact, do not modify)
# ---------------------------------------------------------------------------

ORACLE_SYSTEM = """You are an authoritative evaluator resolving a scoring disagreement for a document understanding benchmark.

A rule-based scorer and a semantic judge gave conflicting verdicts on whether a model response is correct.
You must make the definitive final call.

Context you will receive:
- The question asked
- The ground truth answer
- The model's raw response
- Rule scorer verdict (YES/NO)
- Semantic judge verdict (YES/NO) and its reason

Evaluation criteria:
- If the model response contains or clearly implies the correct answer: CORRECT
- For classification: exact category match required (no partial credit)
- For numeric: within 5% rounding acceptable
- For bounding boxes: IoU >= 0.25 with ground truth
- For text extraction: core value present, extra context acceptable
- Prioritize semantic correctness over formatting

Reply with EXACTLY three lines:
Line 1: correct or wrong
Line 2: confidence: high or medium or low
Line 3: one sentence explaining why you ruled against the minority verdict"""


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_oracle(user_msg: str, api_key: str) -> tuple:
    """Returns (oracle_correct: bool, confidence: str, reason: str)"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mistral-large-2512",
        "messages": [
            {"role": "system", "content": ORACLE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 120,
        "temperature": 0.0,
    }
    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=45,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip().lower()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    is_correct = lines[0].startswith("correct") if lines else False
    confidence = "high"
    if len(lines) > 1:
        conf_line = lines[1]
        if "medium" in conf_line:
            confidence = "medium"
        elif "low" in conf_line:
            confidence = "low"
    reason = lines[2] if len(lines) > 2 else ""
    return is_correct, confidence, reason


def call_oracle_with_retry(user_msg: str, api_key: str) -> tuple:
    """call_oracle with 3 retries and exponential backoff (2s, 4s, 8s)."""
    delays = [2, 4, 8]
    last_exc = None
    for attempt, delay in enumerate([0] + delays):
        if delay:
            time.sleep(delay)
        try:
            return call_oracle(user_msg, api_key)
        except Exception as exc:
            last_exc = exc
            if attempt < len(delays):
                continue
    raise RuntimeError(f"Oracle API failed after retries: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_disagreements(v2_path: Path, v1_path: Path) -> list:
    """
    Load disagreement records.

    Prefers v2 file (needs_oracle=True, neural_correct field).
    Falls back to v1 file: computes disagreements as rule_correct != nano_correct,
    and maps nano_correct -> neural_correct, reason -> neural_reason.
    """
    if v2_path.exists():
        print(f"Reading disagreements from {v2_path}")
        records = []
        with open(v2_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("needs_oracle"):
                    records.append(r)
        print(f"  {len(records)} records with needs_oracle=True")
        return records

    # Fallback to v1: nano_correct is the neural judge
    print(f"v2 file not found, falling back to {v1_path}")
    if not v1_path.exists():
        sys.exit(f"ERROR: neither {v2_path} nor {v1_path} found.")

    records = []
    with open(v1_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rule_correct = r.get("rule_correct", False)
            nano_correct = r.get("nano_correct")
            if nano_correct is None:
                continue  # judgment failed, skip
            if rule_correct != nano_correct:
                # Normalise field names to what the rest of the script expects
                r["neural_correct"] = nano_correct
                r["neural_reason"] = r.get("reason", "")
                r["needs_oracle"] = True
                records.append(r)

    print(f"  {len(records)} disagreement records (rule_correct != nano_correct)")
    return records


def load_benchmark(bench_path: Path) -> dict:
    bench = {}
    with open(bench_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            bench[s["sample_id"]] = s
    return bench


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def build_user_msg(record: dict, bench_sample: dict) -> str:
    query = bench_sample.get("query", record.get("query", "")) or ""
    gt = bench_sample.get("gt_answer", record.get("gt_answer", "")) or ""
    raw = record.get("raw_answer", record.get("predicted_answer", "")) or ""
    rule_correct = record.get("rule_correct", False)
    neural_correct = record.get("neural_correct", record.get("nano_correct", False))
    neural_reason = record.get("neural_reason", record.get("reason", "")) or ""

    return (
        f"Question: {query[:200]}\n"
        f"Ground truth: {gt}\n"
        f"Model response: {raw[:400]}\n\n"
        f"Rule scorer: {'CORRECT' if rule_correct else 'WRONG'}\n"
        f"Semantic judge: {'CORRECT' if neural_correct else 'WRONG'}, {neural_reason}\n\n"
        f"Make the definitive call:"
    )


def make_verdict_record(record: dict, oracle_correct: bool,
                        oracle_confidence: str, oracle_reason: str) -> dict:
    return {
        "sample_id": record["sample_id"],
        "yaml_key": record["yaml_key"],
        "budget_level": record.get("budget_level", "?"),
        "rule_correct": record.get("rule_correct", False),
        "neural_correct": record.get("neural_correct", record.get("nano_correct", False)),
        "oracle_correct": oracle_correct,
        "oracle_confidence": oracle_confidence,
        "oracle_reason": oracle_reason,
        "eval_correct": oracle_correct,  # oracle is final authority on disagreements
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Oracle arbitration via Mistral Large 2512")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process 5 records, print output, no writes")
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        sys.exit("ERROR: MISTRAL_API_KEY environment variable not set.")

    # Paths
    v2_path   = _ROOT / "data/model_eval_results/all_models_judgments_v2.jsonl"
    v1_path   = _ROOT / "data/model_eval_results/all_models_judgments.jsonl"
    bench_path = _ROOT / "data/benchmark/benchmark_5000.jsonl"
    out_path  = _ROOT / "data/model_eval_results/oracle_verdicts.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load inputs
    all_disagreements = load_disagreements(v2_path, v1_path)
    bench = load_benchmark(bench_path)
    print(f"Benchmark loaded: {len(bench)} samples")

    # Resume: load already-done keys
    done_keys = set()
    if not args.dry_run and out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                done_keys.add((j["sample_id"], j["yaml_key"], j["budget_level"]))
        print(f"Resuming: {len(done_keys)} already done")

    pending = [
        r for r in all_disagreements
        if (r["sample_id"], r["yaml_key"], r.get("budget_level", "?")) not in done_keys
    ]

    if args.dry_run:
        pending = pending[:5]
        print(f"\n[DRY RUN] Processing {len(pending)} records (no writes)\n")
    else:
        print(f"Total disagreements: {len(all_disagreements)} | Pending: {len(pending)}")

    if not pending:
        print("Nothing to process: all disagreements already arbitrated.")
        _print_summary(out_path)
        return

    # Shared state
    write_lock = threading.Lock()
    counter = [0]
    dry_run_results = []

    def process_one(record):
        sample = bench.get(record["sample_id"], {})
        user_msg = build_user_msg(record, sample)

        oracle_correct, confidence, reason = call_oracle_with_retry(user_msg, api_key)
        verdict = make_verdict_record(record, oracle_correct, confidence, reason)

        with write_lock:
            if args.dry_run:
                dry_run_results.append(verdict)
            else:
                with open(out_path, "a") as f:
                    f.write(json.dumps(verdict) + "\n")
            counter[0] += 1
            n = counter[0]
            if n % 200 == 0:
                pct = n / len(pending) * 100
                print(f"  Progress: {n}/{len(pending)} ({pct:.1f}%)")

        return verdict

    print(f"\nLaunching 8 parallel workers...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process_one, r): r for r in pending}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                r = futures[fut]
                print(f"  ERROR [{r['sample_id']} / {r['yaml_key']}]: {exc}")

    # Dry-run: print results only
    if args.dry_run:
        print("\n[DRY RUN] Results (not written):")
        for v in dry_run_results:
            print(json.dumps(v, indent=2))
        return

    print(f"\nDone. {counter[0]} verdicts written to {out_path}")
    _print_summary(out_path)


def _print_summary(out_path: Path):
    if not out_path.exists():
        print("(no verdicts file to summarise)")
        return

    verdicts = []
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            verdicts.append(json.loads(line))

    if not verdicts:
        print("(verdicts file is empty)")
        return

    total = len(verdicts)
    agreed_rule = sum(
        1 for v in verdicts
        if v["oracle_correct"] == v["rule_correct"]
    )
    agreed_neural = sum(
        1 for v in verdicts
        if v["oracle_correct"] == v["neural_correct"]
    )
    conf_counts = {"high": 0, "medium": 0, "low": 0}
    for v in verdicts:
        c = v.get("oracle_confidence", "high")
        conf_counts[c] = conf_counts.get(c, 0) + 1

    print("\n" + "=" * 60)
    print("ORACLE ARBITRATION SUMMARY")
    print("=" * 60)
    print(f"Total disagreements processed : {total}")
    print(f"Oracle agreed with rule       : {agreed_rule:>6} ({agreed_rule/total*100:5.1f}%)")
    print(f"Oracle agreed with neural     : {agreed_neural:>6} ({agreed_neural/total*100:5.1f}%)")
    print(f"Confidence (high)            : {conf_counts['high']:>6} ({conf_counts['high']/total*100:5.1f}%)")
    print(f"Confidence (medium)          : {conf_counts['medium']:>6} ({conf_counts['medium']/total*100:5.1f}%)")
    print(f"Confidence (low)             : {conf_counts['low']:>6} ({conf_counts['low']/total*100:5.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
