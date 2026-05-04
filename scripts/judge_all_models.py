#!/usr/bin/env python3
"""
Judge all model responses (except Opus, already done) using GPT-5.4-mini
as a semantic correctness evaluator.

Outputs: data/model_eval_results/all_models_judgments.jsonl
"""
import json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import openai

client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

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


def judge_one(record, bench_sample):
    gt = bench_sample.get('gt_answer', '') or ''
    raw = record.get('raw_answer', '') or ''
    query = bench_sample.get('query', '') or ''
    task = bench_sample.get('task_type', '')
    metric = bench_sample.get('correctness_metric', '')
    dataset = bench_sample.get('source_dataset', '')

    if not gt or not raw:
        return None, "missing gt or raw"

    user_msg = (
        f"Dataset: {dataset} | Task: {task} | Metric: {metric}\n"
        f"Question: {query[:200]}\n"
        f"Ground truth: {gt}\n"
        f"Model response: {raw[:500]}\n\n"
        f"Is the model's response correct?"
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_completion_tokens=60,
                temperature=0.0,
            )
            text = (resp.choices[0].message.content or '').strip().lower()
            is_correct = text.startswith('correct')
            reason = text.split('\n', 1)[1].strip() if '\n' in text else ''
            return is_correct, reason
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return None, str(e)[:100]


def main():
    # Load data: anchor + validation
    results = []
    for results_file in [
        _ROOT / 'data/model_eval_results/api_results_anchor.jsonl',
        _ROOT / 'data/model_eval_results/api_results_validation.jsonl',
    ]:
        if results_file.exists():
            with open(results_file) as f:
                for line in f:
                    results.append(json.loads(line.strip()))

    bench = {}
    with open(_ROOT / 'data/benchmark/benchmark_5000.jsonl') as f:
        for line in f:
            s = json.loads(line.strip())
            bench[s['sample_id']] = s

    # Filter: exclude errors, require raw_answer (Opus now included: B2 is new)
    to_judge = [
        r for r in results
        if not r.get('error')
        and r.get('raw_answer', '').strip()
    ]

    out_path = _ROOT / 'data/model_eval_results/all_models_judgments.jsonl'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already done
    done_keys = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                j = json.loads(line.strip())
                done_keys.add((j['sample_id'], j['yaml_key'], j['budget_level']))

    pending = [r for r in to_judge
               if (r['sample_id'], r['yaml_key'], r.get('budget_level')) not in done_keys]

    print(f"Total to judge: {len(to_judge)} | Already done: {len(done_keys)} | Pending: {len(pending)}")

    if not pending:
        print("All done!")
        return

    lock = threading.Lock()
    done_count = [0]
    write_lock = threading.Lock()

    def process(r):
        s = bench.get(r['sample_id'], {})
        is_correct, reason = judge_one(r, s)
        result = {
            'sample_id': r['sample_id'],
            'yaml_key': r['yaml_key'],
            'budget_level': r.get('budget_level', '?'),
            'rule_correct': r.get('is_correct', False),
            'nano_correct': is_correct,
            'reason': reason,
            'source_dataset': s.get('source_dataset', ''),
            'task_type': s.get('task_type', ''),
            'tier_final': s.get('tier_final', 0),
            'correctness_metric': s.get('correctness_metric', ''),
            'gt_answer': s.get('gt_answer', ''),
            'raw_answer': r.get('raw_answer', '')[:300],
            'predicted_answer': r.get('predicted_answer', ''),
        }
        with write_lock:
            with open(out_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
        with lock:
            done_count[0] += 1
            if done_count[0] % 500 == 0:
                pct = done_count[0] / len(pending) * 100
                print(f"  {done_count[0]}/{len(pending)} ({pct:.1f}%) judged...")

    print("Starting parallel judging with 12 workers...")
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(process, r) for r in pending]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"Worker error: {e}")

    print(f"\nDone. Judgments saved to {out_path}")

    # Quick summary
    judgments = []
    with open(out_path) as f:
        for line in f:
            judgments.append(json.loads(line.strip()))

    by_model = defaultdict(lambda: {'rule_c': 0, 'nano_c': 0, 'n': 0})
    for j in judgments:
        if j.get('nano_correct') is None:
            continue
        k = j['yaml_key']
        by_model[k]['n'] += 1
        if j.get('rule_correct'): by_model[k]['rule_c'] += 1
        if j.get('nano_correct'): by_model[k]['nano_c'] += 1

    print(f"\n{'Model':<25} {'N':>6}  {'Rule Acc':>9}  {'Nano Acc':>9}  {'Delta':>8}")
    print('─' * 65)
    for k, v in sorted(by_model.items(), key=lambda x: -x[1]['nano_c']/max(x[1]['n'],1)):
        rule_a = v['rule_c']/v['n']*100
        nano_a = v['nano_c']/v['n']*100
        print(f"  {k:<25} {v['n']:>6}  {rule_a:>8.1f}%  {nano_a:>8.1f}%  {nano_a-rule_a:>+7.1f}pp")


if __name__ == '__main__':
    main()
