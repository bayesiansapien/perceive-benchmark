#!/usr/bin/env python3
"""
Judge sensitivity experiment (task 1.3).

Re-judges the ~7,974 oracle-arbitrated records using an alternate frontier arbiter.
Original arbiter: Mistral Large 2512.
Alternate arbiters: GPT-5.4 (OpenAI) or Claude Sonnet 4.6 (Anthropic via Vertex AI).

Usage:
    python scripts/judge_sensitivity.py --model gpt54
    python scripts/judge_sensitivity.py --model sonnet

Output per model:
    data/judge_sensitivity_{model}.jsonl
    data/judge_sensitivity_{model}_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ORACLE_FILE    = ROOT / "data/model_eval_results/oracle_verdicts.jsonl"
BENCHMARK_FILE = ROOT / "data/benchmark/benchmark_5000.jsonl"
API_FILES      = [
    ROOT / "data/model_eval_results/api_results_anchor.jsonl",
    ROOT / "data/model_eval_results/api_results_validation.jsonl",
    ROOT / "data/model_eval_results/api_results_remaining.jsonl",
]

MAX_WORKERS = 12
MAX_RETRIES = 3

JUDGE_SYSTEM = """You are a precise answer evaluator for a document understanding benchmark.
Compare a model's response to the ground truth answer and determine correctness.

Rules:
- Focus on semantic correctness, not formatting or markdown
- Ignore capitalization differences and minor punctuation
- For classification tasks (document type, layout element): exact category match required
- For extraction/OCR: the core value must match; extra context is acceptable
- For yes/no questions: "Yes, because..." counts as "yes"
- For numeric values: allow minor rounding (within 5%)
- For bounding boxes: coordinates must be approximately correct (within 0.05 per coord)
- For multi-word answers: all key words must be present
- If the response contains the correct answer within it, count as correct

Reply with EXACTLY: "correct" or "wrong" on the first line, then one sentence reason."""

MODEL_CONFIGS = {
    "gpt54": {
        "label":    "GPT-5.4",
        "model_id": "gpt-5.4",
        "provider": "openai",
    },
    "sonnet": {
        "label":    "Claude Sonnet 4.6",
        "model_id": "claude-sonnet-4-6",
        "provider": "anthropic_vertex",
    },
}


# ── clients ───────────────────────────────────────────────────────────────────

def make_openai_client():
    import openai
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return openai.OpenAI(api_key=api_key)


def make_anthropic_client():
    from anthropic import AnthropicVertex
    project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    region  = os.environ.get("CLOUD_ML_REGION", "us-east5")
    if not project:
        raise RuntimeError("ANTHROPIC_VERTEX_PROJECT_ID not set")
    return AnthropicVertex(project_id=project, region=region)


# ── judge calls ───────────────────────────────────────────────────────────────

def _user_msg(raw: str, gt: str, query: str, task: str, metric: str, dataset: str) -> str:
    return (
        f"Dataset: {dataset} | Task: {task} | Metric: {metric}\n"
        f"Question: {query[:200]}\n"
        f"Ground truth: {gt}\n"
        f"Model response: {raw[:500]}\n\n"
        f"Is the model's response correct?"
    )


def call_openai(client, model_id: str, raw: str, gt: str,
                query: str, task: str, metric: str, dataset: str) -> tuple[bool | None, str]:
    msg = _user_msg(raw, gt, query, task, metric, dataset)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": msg},
                ],
                max_completion_tokens=80,
                temperature=0.0,
            )
            text = (resp.choices[0].message.content or "").strip().lower()
            is_correct = text.startswith("correct")
            reason = text.split("\n", 1)[1].strip() if "\n" in text else ""
            return is_correct, reason
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return None, str(exc)[:120]


def call_anthropic(client, model_id: str, raw: str, gt: str,
                   query: str, task: str, metric: str, dataset: str) -> tuple[bool | None, str]:
    msg = _user_msg(raw, gt, query, task, metric, dataset)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=model_id,
                max_tokens=80,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": msg}],
            )
            text = (resp.content[0].text or "").strip().lower()
            is_correct = text.startswith("correct")
            reason = text.split("\n", 1)[1].strip() if "\n" in text else ""
            return is_correct, reason
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return None, str(exc)[:120]


# ── data helpers ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_raw_index(api_files: list[Path]) -> dict[tuple, str]:
    idx: dict[tuple, str] = {}
    for f in api_files:
        for r in load_jsonl(f):
            key = (r["sample_id"], r.get("yaml_key", ""), r.get("budget_level", ""))
            if key not in idx:
                idx[key] = r.get("raw_answer", "") or ""
    return idx


def load_done_keys(out_path: Path) -> set[tuple]:
    done: set[tuple] = set()
    for r in load_jsonl(out_path):
        done.add((r["sample_id"], r["yaml_key"], r["budget_level"]))
    return done


# ── report ────────────────────────────────────────────────────────────────────

def write_report(results: list[dict], model_label: str,
                 bench_idx: dict, out_json: Path) -> None:
    judged = [r for r in results if r.get("new_correct") is not None]
    if not judged:
        print("No judged records: cannot write report.")
        return

    n_total   = len(judged)
    n_flipped = sum(r["flipped"] for r in judged)
    flip_rate = n_flipped / n_total

    by_conf = defaultdict(lambda: {"n": 0, "flipped": 0})
    by_task = defaultdict(lambda: {"n": 0, "flipped": 0})
    for r in judged:
        c = r.get("oracle_confidence", "unknown") or "unknown"
        by_conf[c]["n"] += 1
        by_conf[c]["flipped"] += int(r["flipped"])
        tt = bench_idx.get(r["sample_id"], {}).get("task_type", "unknown")
        by_task[tt]["n"] += 1
        by_task[tt]["flipped"] += int(r["flipped"])

    print(f"\n{'═'*62}")
    print(f"Judge sensitivity: {model_label} vs Mistral Large 2512")
    print(f"{'═'*62}")
    print(f"  Records re-judged:  {n_total:,}")
    print(f"  Flipped:            {n_flipped:,}")
    print(f"  Flip rate:          {flip_rate*100:.2f}%")
    stability = ("CONFIRMED (<5%)" if flip_rate < 0.05 else
                 "BORDERLINE (5-10%)" if flip_rate < 0.10 else "HIGH (>10%)")
    print(f"  Label stability:    {stability}")
    print("\nBy oracle confidence:")
    for conf in sorted(by_conf):
        s = by_conf[conf]
        print(f"  {conf:<10}: n={s['n']:>5}, flip={s['flipped']/s['n']*100:.1f}%")
    print("\nBy task type:")
    for tt in sorted(by_task):
        s = by_task[tt]
        print(f"  {tt:<32}: n={s['n']:>5}, flip={s['flipped']/s['n']*100:.1f}%")

    report = {
        "alternate_model":  model_label,
        "original_model":   "mistral-large-2512",
        "n_judged":         n_total,
        "n_flipped":        n_flipped,
        "flip_rate_pct":    round(flip_rate * 100, 3),
        "stability":        stability,
        "by_confidence":    {k: {"n": v["n"],
                                 "flip_pct": round(v["flipped"]/v["n"]*100, 2)}
                             for k, v in by_conf.items()},
        "by_task_type":     {k: {"n": v["n"],
                                 "flip_pct": round(v["flipped"]/v["n"]*100, 2)}
                             for k, v in by_task.items()},
    }
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out_json}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIGS), required=True,
                        help="Alternate arbiter to use")
    args = parser.parse_args()

    cfg        = MODEL_CONFIGS[args.model]
    model_id   = cfg["model_id"]
    model_label = cfg["label"]
    provider   = cfg["provider"]

    out_jsonl = ROOT / f"data/judge_sensitivity_{args.model}.jsonl"
    out_json  = ROOT / f"data/judge_sensitivity_{args.model}_report.json"

    print(f"Judge sensitivity: {model_label} vs Mistral Large 2512")
    print(f"Output: {out_jsonl}")

    # Build clients and judge function
    if provider == "openai":
        client     = make_openai_client()
        judge_fn   = lambda raw, gt, q, t, m, d: call_openai(
            client, model_id, raw, gt, q, t, m, d)
    else:
        client     = make_anthropic_client()
        judge_fn   = lambda raw, gt, q, t, m, d: call_anthropic(
            client, model_id, raw, gt, q, t, m, d)

    # Load data
    print("Loading data ...")
    oracle_rows = load_jsonl(ORACLE_FILE)
    bench_idx   = {r["sample_id"]: r for r in load_jsonl(BENCHMARK_FILE)}
    raw_idx     = build_raw_index(API_FILES)
    done_keys   = load_done_keys(out_jsonl)

    pending = [r for r in oracle_rows
               if (r["sample_id"], r["yaml_key"], r["budget_level"]) not in done_keys]

    print(f"  Oracle records: {len(oracle_rows):,} | Already done: {len(done_keys):,} | Pending: {len(pending):,}")

    if pending:
        write_lock  = threading.Lock()
        done_count  = [0]

        def process(r):
            key   = (r["sample_id"], r["yaml_key"], r["budget_level"])
            bench = bench_idx.get(r["sample_id"], {})
            raw   = raw_idx.get(key, "")
            gt    = bench.get("gt_answer", "") or ""
            if not gt or not raw:
                return

            new_correct, new_reason = judge_fn(
                raw, gt,
                bench.get("query", "") or "",
                bench.get("task_type", "") or "",
                bench.get("correctness_metric", "") or "",
                bench.get("source_dataset", "") or "",
            )

            orig_correct = r.get("oracle_correct")
            flipped = (new_correct is not None and orig_correct is not None
                       and bool(new_correct) != bool(orig_correct))

            row = {
                "sample_id":         r["sample_id"],
                "yaml_key":          r["yaml_key"],
                "budget_level":      r["budget_level"],
                "orig_correct":      orig_correct,
                "new_correct":       new_correct,
                "flipped":           flipped,
                "oracle_confidence": r.get("oracle_confidence", ""),
                "orig_reason":       r.get("oracle_reason", ""),
                "new_reason":        new_reason,
            }
            with write_lock:
                with open(out_jsonl, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                done_count[0] += 1
                if done_count[0] % 500 == 0:
                    print(f"  {done_count[0]:,}/{len(pending):,} "
                          f"({done_count[0]/len(pending)*100:.1f}%) ...")

        print(f"Re-judging with {MAX_WORKERS} workers ...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(process, r) for r in pending]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    print(f"Worker error: {exc}")
        print("Done.\n")

    results = load_jsonl(out_jsonl)
    write_report(results, model_label, bench_idx, out_json)


if __name__ == "__main__":
    main()
