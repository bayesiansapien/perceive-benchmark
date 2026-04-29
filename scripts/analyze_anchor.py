#!/usr/bin/env python3
"""
DocRouteBench — Anchor Set Analysis
Generates a multi-dimensional report on the 1,500-sample anchor dataset.

Usage:
    python scripts/analyze_anchor.py
Output saved to data/analysis/anchor_analysis.txt
"""

import json
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_PATH = os.path.join(REPO_ROOT, "data", "benchmark", "benchmark_5000.jsonl")
API_RESULTS_PATH = os.path.join(
    REPO_ROOT, "data", "model_eval_results", "api_results_anchor.jsonl"
)
JUDGMENTS_PATH = os.path.join(
    REPO_ROOT, "data", "model_eval_results", "all_models_judgments.jsonl"
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "analysis")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "anchor_analysis.txt")

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------
MODEL_META = {
    "a2_flashlite":  {"display": "Flash-Lite",  "tier": "A", "cost_per_m": 0.05},
    "a4_gpt54nano":  {"display": "GPT-nano",     "tier": "A", "cost_per_m": 0.04},
    "b1_gpt54mini":  {"display": "GPT-mini",     "tier": "B", "cost_per_m": 0.20},
    "b3_sonnet":     {"display": "Sonnet",        "tier": "B", "cost_per_m": 3.00},
    "c1_gpt54":      {"display": "GPT-5.4",       "tier": "C", "cost_per_m": 3.00},
    "c2_opus":       {"display": "Opus",           "tier": "C", "cost_per_m": 15.00},
    "c3_gemini_pro": {"display": "Gemini Pro",    "tier": "C", "cost_per_m": 1.25},
}
MODEL_ORDER = [
    "a2_flashlite", "a4_gpt54nano",
    "b1_gpt54mini", "b3_sonnet",
    "c1_gpt54", "c2_opus", "c3_gemini_pro",
]
BUDGET_ORDER = ["B0", "B1", "B2", "B3"]
BUDGET_LABELS = {"B0": "B0(0)", "B1": "B1(1k)", "B2": "B2(4k)", "B3": "B3(16k)"}

TASK_ORDER = ["T1", "T2", "T3", "T4", "T5", "element_localization"]
TASK_LABELS = {
    "T1": "T1-DocClass",
    "T2": "T2-TextSpot",
    "T3": "T3-TableParse",
    "T4": "T4-DocVQA",
    "T5": "T5-LayoutUnd",
    "element_localization": "T6-ElemLocal",
}
TIER_ORDER = [1, 2, 3]
TIER_LABELS = {1: "T1(easy)", 2: "T2(med)", 3: "T3(hard)"}

# ---------------------------------------------------------------------------
# Output collector (tee to stdout + file)
# ---------------------------------------------------------------------------
output_lines = []


def emit(line=""):
    print(line)
    output_lines.append(line)


def div(char="=", width=90):
    emit(char * width)


def header(title):
    div()
    emit(f"  {title}")
    div()


def subheader(title):
    emit("")
    emit(f"--- {title} ---")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_benchmark_anchor():
    """Return dict sample_id -> {task_type, tier_final, source_dataset}."""
    meta = {}
    with open(BENCHMARK_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get("in_anchor_set"):
                meta[r["sample_id"]] = {
                    "task_type": r.get("task_type"),
                    "tier_final": r.get("tier_final"),
                    "source_dataset": r.get("source_dataset"),
                }
    return meta


def load_judgments():
    """Return lookup (sample_id, yaml_key, budget_level) -> nano_correct (bool|None)
    and a full record list for aggregation.

    Also returns per-key metadata dict: (sample_id,yaml_key,budget_level) -> record fields.
    """
    lookup = {}          # (sid, yk, bl) -> nano_correct
    records = []         # all judgment records
    with open(JUDGMENTS_PATH) as f:
        for line in f:
            r = json.loads(line)
            key = (r["sample_id"], r["yaml_key"], r["budget_level"])
            nc = r.get("nano_correct")
            lookup[key] = nc
            records.append(r)
    return lookup, records


def load_api_results():
    """Return list of api result records and cost lookup."""
    records = []
    cost_lookup = {}  # (sample_id, yaml_key, budget_level) -> total_cost_usd
    with open(API_RESULTS_PATH) as f:
        for line in f:
            r = json.loads(line)
            key = (r["sample_id"], r["yaml_key"], r["budget_level"])
            cost_lookup[key] = r.get("total_cost_usd", 0) or 0
            records.append(r)
    return records, cost_lookup


def build_eval_records(bench_meta, judgment_lookup, api_records):
    """
    Merge api_records with judgment_lookup and bench_meta.
    Returns a list of dicts, each representing one (sample, model, budget) evaluation:
      - sample_id, yaml_key, budget_level
      - task_type, tier_final, source_dataset  (from bench_meta)
      - nano_correct  (from judgments if available, else is_correct from api)
      - rule_correct  (is_correct from api)
      - total_cost_usd
    """
    merged = []
    for r in api_records:
        sid = r["sample_id"]
        yk = r["yaml_key"]
        bl = r["budget_level"]
        key = (sid, yk, bl)

        bm = bench_meta.get(sid, {})
        task_type = bm.get("task_type")
        tier_final = bm.get("tier_final")
        source_dataset = bm.get("source_dataset")

        rule_correct = r.get("is_correct", False)

        # Primary metric: nano_correct from judgments; fallback to rule_correct
        if key in judgment_lookup:
            nc = judgment_lookup[key]
            nano_correct = nc if nc is not None else rule_correct
        else:
            nano_correct = rule_correct

        merged.append({
            "sample_id": sid,
            "yaml_key": yk,
            "budget_level": bl,
            "task_type": task_type,
            "tier_final": tier_final,
            "source_dataset": source_dataset,
            "nano_correct": nano_correct,
            "rule_correct": rule_correct,
            "total_cost_usd": r.get("total_cost_usd", 0) or 0,
        })
    return merged


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def safe_pct(num, den):
    if den == 0:
        return None
    return num / den


def fmt_pct(val, width=7):
    if val is None:
        return "—".rjust(width)
    return f"{val*100:.1f}%".rjust(width)


def fmt_n(val, width=6):
    if val is None:
        return "—".rjust(width)
    return str(val).rjust(width)


def agg_acc(records, key_fn=None):
    """
    Aggregate accuracy from a list of records.
    key_fn: optional grouping function record -> group_key.
    Returns dict group_key -> (n_total, n_nano_correct, n_rule_correct) if key_fn,
    else tuple (n_total, n_nano_correct, n_rule_correct).
    """
    if key_fn is None:
        total = 0
        nano_sum = 0
        rule_sum = 0
        for r in records:
            total += 1
            nano_sum += int(bool(r["nano_correct"]))
            rule_sum += int(bool(r["rule_correct"]))
        return (total, nano_sum, rule_sum)
    else:
        groups = defaultdict(lambda: [0, 0, 0])
        for r in records:
            k = key_fn(r)
            groups[k][0] += 1
            groups[k][1] += int(bool(r["nano_correct"]))
            groups[k][2] += int(bool(r["rule_correct"]))
        return dict(groups)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def section1_dataset_overview(bench_meta):
    header("SECTION 1 — Dataset Overview")
    n = len(bench_meta)
    emit(f"  Total anchor samples: {n}")

    subheader("By Task Type")
    task_counts = defaultdict(int)
    for v in bench_meta.values():
        task_counts[v["task_type"]] += 1
    emit(f"  {'Task Type':<22}  {'Count':>6}  {'%':>7}")
    emit(f"  {'-'*22}  {'-'*6}  {'-'*7}")
    for t in TASK_ORDER:
        c = task_counts.get(t, 0)
        emit(f"  {TASK_LABELS.get(t, t):<22}  {c:>6}  {c/n*100:>6.1f}%")

    subheader("By Complexity Tier")
    tier_counts = defaultdict(int)
    for v in bench_meta.values():
        tier_counts[v["tier_final"]] += 1
    emit(f"  {'Tier':<14}  {'Count':>6}  {'%':>7}")
    emit(f"  {'-'*14}  {'-'*6}  {'-'*7}")
    for t in TIER_ORDER:
        c = tier_counts.get(t, 0)
        label = {1: "Tier 1 (easy)", 2: "Tier 2 (med)", 3: "Tier 3 (hard)"}.get(t, str(t))
        emit(f"  {label:<14}  {c:>6}  {c/n*100:>6.1f}%")

    subheader("By Source Dataset")
    src_counts = defaultdict(int)
    for v in bench_meta.values():
        src_counts[v["source_dataset"]] += 1
    srcs = sorted(src_counts.items(), key=lambda x: -x[1])
    emit(f"  {'Source Dataset':<22}  {'Count':>6}  {'%':>7}")
    emit(f"  {'-'*22}  {'-'*6}  {'-'*7}")
    for src, c in srcs:
        emit(f"  {(src or 'Unknown'):<22}  {c:>6}  {c/n*100:>6.1f}%")

    subheader("Cross-tab: Task Type × Complexity Tier (count)")
    tiers_in_data = [t for t in TIER_ORDER if tier_counts.get(t, 0) > 0]
    task_tier = defaultdict(lambda: defaultdict(int))
    for v in bench_meta.values():
        task_tier[v["task_type"]][v["tier_final"]] += 1

    tier_col_w = 9
    hdr_cols = "".join(f"  {TIER_LABELS.get(t, str(t)):>{tier_col_w}}" for t in tiers_in_data)
    emit(f"  {'Task Type':<22}{hdr_cols}  {'Total':>7}")
    emit(f"  {'-'*22}" + "".join([f"  {'-'*tier_col_w}"] * len(tiers_in_data)) + f"  {'-'*7}")
    for tt in TASK_ORDER:
        row_total = sum(task_tier[tt].values())
        cols = "".join(f"  {task_tier[tt].get(t, 0):>{tier_col_w}}" for t in tiers_in_data)
        emit(f"  {TASK_LABELS.get(tt, tt):<22}{cols}  {row_total:>7}")
    # totals row
    grand_cols = "".join(f"  {tier_counts.get(t,0):>{tier_col_w}}" for t in tiers_in_data)
    emit(f"  {'TOTAL':<22}{grand_cols}  {n:>7}")


def section2_model_overview(eval_records):
    header("SECTION 2 — Model Overview (Overall Accuracy)")

    # Aggregate per model (all budgets combined)
    per_model = defaultdict(lambda: {"nano": [], "rule": []})
    for r in eval_records:
        yk = r["yaml_key"]
        per_model[yk]["nano"].append(int(bool(r["nano_correct"])))
        per_model[yk]["rule"].append(int(bool(r["rule_correct"])))

    col_w = [15, 6, 8, 8, 9, 9, 8]
    hdr = (f"  {'Model':<{col_w[0]}} {'Tier':>{col_w[1]}} {'Cost/M':>{col_w[2]}}"
           f" {'N_evals':>{col_w[3]}} {'Rule_Acc':>{col_w[4]}} {'Nano_Acc':>{col_w[5]}} {'Delta':>{col_w[6]}}")
    emit(hdr)
    emit("  " + "-" * (sum(col_w) + len(col_w)))

    rows = []
    for yk in MODEL_ORDER:
        if yk not in per_model:
            continue
        meta = MODEL_META.get(yk, {})
        nano_vals = per_model[yk]["nano"]
        rule_vals = per_model[yk]["rule"]
        n = len(nano_vals)
        nano_acc = sum(nano_vals) / n if n else None
        rule_acc = sum(rule_vals) / n if n else None
        delta = (nano_acc - rule_acc) if (nano_acc is not None and rule_acc is not None) else None
        rows.append((yk, meta, n, rule_acc, nano_acc, delta))

    rows_sorted = sorted(rows, key=lambda x: (x[4] or 0), reverse=True)
    for yk, meta, n, rule_acc, nano_acc, delta in rows_sorted:
        disp = meta.get("display", yk)
        tier = meta.get("tier", "?")
        cost = meta.get("cost_per_m", 0)
        delta_str = (f"{delta*100:+.1f}pp" if delta is not None else "—").rjust(col_w[6])
        emit(f"  {disp:<{col_w[0]}} {tier:>{col_w[1]}} ${cost:>{col_w[2]-1}.2f}"
             f" {n:>{col_w[3]}} {fmt_pct(rule_acc, col_w[4])} {fmt_pct(nano_acc, col_w[5])} {delta_str}")


def section3_accuracy_by_task(eval_records):
    header("SECTION 3 — Accuracy by Task Type (nano_correct)")

    # (yaml_key, task_type) -> [nano_correct ints]
    acc = defaultdict(list)
    task_n = defaultdict(int)  # task_type -> total evals
    for r in eval_records:
        tt = r["task_type"] or "unknown"
        yk = r["yaml_key"]
        acc[(yk, tt)].append(int(bool(r["nano_correct"])))
        task_n[tt] += 1

    models_present = [yk for yk in MODEL_ORDER if any((yk, tt) in acc for tt in TASK_ORDER)]
    short = [MODEL_META[yk]["display"][:9] for yk in models_present]

    col_w = 10
    hdr = f"  {'Task Type':<22}  {'N_evals':>7}"
    for s in short:
        hdr += f"  {s:>{col_w}}"
    emit(hdr)
    emit("  " + "-" * (22 + 9 + len(short) * (col_w + 2)))

    task_best = {}   # task_type -> (best_model_display, best_acc)
    task_worst = {}  # task_type -> (worst_model_display, worst_acc)

    for tt in TASK_ORDER:
        n_evals = task_n.get(tt, 0)
        row = f"  {TASK_LABELS.get(tt, tt):<22}  {n_evals:>7}"
        model_accs = []
        for yk in models_present:
            vals = acc.get((yk, tt), [])
            a = (sum(vals) / len(vals)) if vals else None
            model_accs.append((yk, a))
            row += fmt_pct(a, col_w + 2)
        emit(row)

        valid = [(yk, a) for yk, a in model_accs if a is not None]
        if valid:
            best_yk, best_a = max(valid, key=lambda x: x[1])
            worst_yk, worst_a = min(valid, key=lambda x: x[1])
            task_best[tt] = (MODEL_META[best_yk]["display"], best_a)
            task_worst[tt] = (MODEL_META[worst_yk]["display"], worst_a)

    emit("")
    emit("  Best / Worst model per task type:")
    emit(f"  {'Task Type':<22}  {'Best Model':<14}  {'Best%':>7}  {'Worst Model':<14}  {'Worst%':>7}  {'Spread':>8}")
    emit("  " + "-" * 80)
    for tt in TASK_ORDER:
        b_name, b_acc = task_best.get(tt, ("—", None))
        w_name, w_acc = task_worst.get(tt, ("—", None))
        spread = ((b_acc - w_acc) if (b_acc is not None and w_acc is not None) else None)
        emit(f"  {TASK_LABELS.get(tt,tt):<22}  {b_name:<14}  {fmt_pct(b_acc,7)}  {w_name:<14}  {fmt_pct(w_acc,7)}  {fmt_pct(spread,8)}")


def section4_accuracy_by_tier(eval_records):
    header("SECTION 4 — Accuracy by Complexity Tier (nano_correct)")

    acc = defaultdict(list)   # (yaml_key, tier) -> [nano_correct ints]
    for r in eval_records:
        tier = r["tier_final"]
        yk = r["yaml_key"]
        if tier is not None:
            acc[(yk, tier)].append(int(bool(r["nano_correct"])))

    models_present = [yk for yk in MODEL_ORDER if any((yk, t) in acc for t in TIER_ORDER)]
    short = [MODEL_META[yk]["display"][:11] for yk in models_present]

    col_w = 10
    hdr = f"  {'Tier':<15}"
    for s in short:
        hdr += f"  {s:>{col_w}}"
    emit(hdr)
    emit("  " + "-" * (15 + len(short) * (col_w + 2)))

    tier_model_acc = {}  # (tier, yk) -> acc
    for tier in TIER_ORDER:
        label = {1: "T1 (easy)", 2: "T2 (medium)", 3: "T3 (hard)"}.get(tier, str(tier))
        row = f"  {label:<15}"
        for yk in models_present:
            vals = acc.get((yk, tier), [])
            a = (sum(vals) / len(vals)) if vals else None
            tier_model_acc[(tier, yk)] = a
            row += fmt_pct(a, col_w + 2)
        emit(row)

    emit("")
    emit("  Accuracy drop T1→T3 per model (routing signal strength):")
    emit(f"  {'Model':<15}  {'T1(easy)':>9}  {'T3(hard)':>9}  {'Drop':>9}  {'Signal':>10}")
    emit("  " + "-" * 60)
    for yk in models_present:
        disp = MODEL_META[yk]["display"]
        a1 = tier_model_acc.get((1, yk))
        a3 = tier_model_acc.get((3, yk))
        drop = ((a1 - a3) if (a1 is not None and a3 is not None) else None)
        signal = "Strong" if (drop is not None and drop > 0.15) else \
                 "Moderate" if (drop is not None and drop > 0.07) else \
                 "Weak" if drop is not None else "—"
        emit(f"  {disp:<15}  {fmt_pct(a1,9)}  {fmt_pct(a3,9)}  {fmt_pct(drop,9)}  {signal:>10}")


def section5_accuracy_by_budget(eval_records):
    header("SECTION 5 — Accuracy by Reasoning Budget (nano_correct)")

    acc = defaultdict(list)  # (yaml_key, budget_level) -> [nano_correct ints]
    for r in eval_records:
        bl = r["budget_level"]
        yk = r["yaml_key"]
        acc[(yk, bl)].append(int(bool(r["nano_correct"])))

    models_present = [yk for yk in MODEL_ORDER if any((yk, bl) in acc for bl in BUDGET_ORDER)]
    short = [MODEL_META[yk]["display"][:11] for yk in models_present]

    col_w = 10
    hdr = f"  {'Budget':<12}"
    for s in short:
        hdr += f"  {s:>{col_w}}"
    emit(hdr)
    emit("  " + "-" * (12 + len(short) * (col_w + 2)))

    budget_model_acc = {}
    for bl in BUDGET_ORDER:
        label = BUDGET_LABELS.get(bl, bl)
        row = f"  {label:<12}"
        for yk in models_present:
            vals = acc.get((yk, bl), [])
            a = (sum(vals) / len(vals)) if vals else None
            budget_model_acc[(bl, yk)] = a
            row += fmt_pct(a, col_w + 2)
        emit(row)

    emit("")
    emit("  Marginal gain per budget step (B0→B1, B1→B2, B2→B3):")
    emit(f"  {'Model':<15}  {'B0→B1':>8}  {'B1→B2':>8}  {'B2→B3':>8}")
    emit("  " + "-" * 46)
    pairs = [("B0", "B1"), ("B1", "B2"), ("B2", "B3")]
    for yk in models_present:
        disp = MODEL_META[yk]["display"]
        gains = []
        for bl_from, bl_to in pairs:
            a_from = budget_model_acc.get((bl_from, yk))
            a_to = budget_model_acc.get((bl_to, yk))
            if a_from is not None and a_to is not None:
                gain = a_to - a_from
                gains.append(f"{gain*100:+.1f}pp".rjust(8))
            else:
                gains.append("—".rjust(8))
        emit(f"  {disp:<15}  {'  '.join(gains)}")


def section6_3d_task_tier(eval_records):
    header("SECTION 6 — 3D Analysis: Task × Tier per Model (nano_correct)")

    # (yaml_key, task_type, tier) -> [nano_correct ints]
    acc = defaultdict(list)
    for r in eval_records:
        tt = r["task_type"]
        tier = r["tier_final"]
        yk = r["yaml_key"]
        if tt and tier:
            acc[(yk, tt, tier)].append(int(bool(r["nano_correct"])))

    models_present = [yk for yk in MODEL_ORDER if any(yk == k[0] for k in acc)]
    tiers_used = [t for t in TIER_ORDER]

    tier_col_w = 9
    for yk in models_present:
        disp = MODEL_META[yk]["display"]
        meta = MODEL_META[yk]
        emit(f"\n  Model: {disp} (Tier {meta['tier']}, ${meta['cost_per_m']}/M tokens)")
        emit(f"  {'Task Type':<22}" + "".join(f"  {TIER_LABELS.get(t,'T'+str(t)):>{tier_col_w}}" for t in tiers_used) + f"  {'Mean':>{tier_col_w}}")
        emit(f"  {'-'*22}" + f"  {'-'*tier_col_w}" * (len(tiers_used) + 1))
        for tt in TASK_ORDER:
            row = f"  {TASK_LABELS.get(tt, tt):<22}"
            all_vals = []
            for tier in tiers_used:
                vals = acc.get((yk, tt, tier), [])
                a = (sum(vals) / len(vals)) if vals else None
                if a is not None:
                    all_vals.append(a)
                row += fmt_pct(a, tier_col_w + 2)
            mean_a = (sum(all_vals) / len(all_vals)) if all_vals else None
            row += fmt_pct(mean_a, tier_col_w + 2)
            emit(row)


def section7_cost_efficiency(eval_records):
    header("SECTION 7 — Cost Efficiency")

    # (yaml_key, budget_level) -> (n, nano_correct_sum, total_cost)
    agg = defaultdict(lambda: {"n": 0, "nano": 0, "cost": 0.0})
    for r in eval_records:
        k = (r["yaml_key"], r["budget_level"])
        agg[k]["n"] += 1
        agg[k]["nano"] += int(bool(r["nano_correct"]))
        agg[k]["cost"] += r.get("total_cost_usd", 0) or 0

    # Build table rows
    rows = []
    for yk in MODEL_ORDER:
        meta = MODEL_META.get(yk, {})
        disp = meta.get("display", yk)
        tier = meta.get("tier", "?")
        cost_per_m = meta.get("cost_per_m", 0)

        # B0 accuracy for marginal gain column
        b0_key = (yk, "B0")
        b0_data = agg.get(b0_key)
        b0_acc = (b0_data["nano"] / b0_data["n"]) if (b0_data and b0_data["n"] > 0) else None

        for bl in BUDGET_ORDER:
            k = (yk, bl)
            if k not in agg:
                continue
            d = agg[k]
            if d["n"] == 0:
                continue
            nano_acc = d["nano"] / d["n"]
            total_cost = d["cost"]
            # Acc per $M: accuracy percentage points per $1M spent
            acc_per_m = (nano_acc / total_cost * 1e6) if total_cost > 0 else None
            marginal = ((nano_acc - b0_acc) if (b0_acc is not None and bl != "B0") else None)
            rows.append({
                "yk": yk, "bl": bl, "disp": disp, "tier": tier, "cost_per_m": cost_per_m,
                "nano_acc": nano_acc, "total_cost": total_cost, "acc_per_m": acc_per_m,
                "marginal": marginal,
            })

    # Pareto-optimal: for each accuracy level, cheapest config
    # Sort by acc desc, cost asc; mark Pareto front
    rows_sorted_by_acc = sorted(rows, key=lambda x: (-x["nano_acc"], x["total_cost"]))
    pareto = set()
    min_cost_seen = float("inf")
    for i, row in enumerate(rows_sorted_by_acc):
        if row["total_cost"] < min_cost_seen:
            min_cost_seen = row["total_cost"]
            pareto.add((row["yk"], row["bl"]))

    emit(f"  {'Model':<14}  {'Tier':>4}  {'$/M':>6}  {'Budget':>8}  {'Nano_Acc':>9}  {'Total$':>8}  {'Acc/$M':>9}  {'vs B0':>8}  {'Pareto':>6}")
    emit("  " + "-" * 85)
    for row in rows_sorted_by_acc:
        is_pareto = "*" if (row["yk"], row["bl"]) in pareto else " "
        marginal_str = (f"{row['marginal']*100:+.1f}pp" if row["marginal"] is not None else "—").rjust(8)
        acc_per_m_str = (f"{row['acc_per_m']:.0f}" if row["acc_per_m"] else "—").rjust(9)
        emit(f"  {row['disp']:<14}  {row['tier']:>4}  ${row['cost_per_m']:>5.2f}"
             f"  {BUDGET_LABELS.get(row['bl'], row['bl']):>8}"
             f"  {fmt_pct(row['nano_acc'], 9)}"
             f"  ${row['total_cost']:>7.2f}"
             f"  {acc_per_m_str}"
             f"  {marginal_str}"
             f"  {is_pareto:>6}")
    emit("")
    emit("  * = Pareto-optimal (cheapest config at that accuracy level)")


def section8_routing_signal(eval_records):
    header("SECTION 8 — Routing Signal Strength by Task Type")

    # (yaml_key, task_type) -> [nano_correct]
    acc = defaultdict(list)
    for r in eval_records:
        tt = r["task_type"]
        yk = r["yaml_key"]
        if tt:
            acc[(yk, tt)].append(int(bool(r["nano_correct"])))

    # Tier-A models
    tier_a = [yk for yk in MODEL_ORDER if MODEL_META.get(yk, {}).get("tier") == "A"]
    tier_c = [yk for yk in MODEL_ORDER if MODEL_META.get(yk, {}).get("tier") == "C"]

    emit(f"  {'Task Type':<22}  {'Best Model':<14}  {'Best%':>7}  {'Worst Model':<14}  {'Worst%':>7}  {'Spread':>8}  {'Recommendation':<20}")
    emit("  " + "-" * 100)

    for tt in TASK_ORDER:
        model_accs = {}
        for yk in MODEL_ORDER:
            vals = acc.get((yk, tt), [])
            if vals:
                model_accs[yk] = sum(vals) / len(vals)
        if not model_accs:
            continue

        best_yk = max(model_accs, key=lambda k: model_accs[k])
        worst_yk = min(model_accs, key=lambda k: model_accs[k])
        best_acc = model_accs[best_yk]
        worst_acc = model_accs[worst_yk]
        spread = best_acc - worst_acc

        # Recommendation logic
        tier_a_mean = (sum(model_accs[yk] for yk in tier_a if yk in model_accs) /
                       max(1, sum(1 for yk in tier_a if yk in model_accs)))
        tier_c_mean = (sum(model_accs[yk] for yk in tier_c if yk in model_accs) /
                       max(1, sum(1 for yk in tier_c if yk in model_accs)))
        gap_c_vs_a = tier_c_mean - tier_a_mean

        if spread < 0.05:
            rec = "Tier A sufficient"
        elif gap_c_vs_a > 0.10:
            rec = "Route to Tier C"
        elif gap_c_vs_a > 0.05:
            rec = "Route to Tier B/C"
        else:
            rec = "Tier B sufficient"

        emit(f"  {TASK_LABELS.get(tt, tt):<22}  {MODEL_META[best_yk]['display']:<14}  {fmt_pct(best_acc,7)}"
             f"  {MODEL_META[worst_yk]['display']:<14}  {fmt_pct(worst_acc,7)}  {fmt_pct(spread,8)}  {rec:<20}")


def section9_summary(eval_records, bench_meta):
    header("SECTION 9 — Summary Statistics")

    total_calls = len(eval_records)
    total_cost = sum(r.get("total_cost_usd", 0) or 0 for r in eval_records)
    emit(f"  Total API evaluation calls in anchor: {total_calls:,}")
    emit(f"  Total estimated cost:                 ${total_cost:.2f}")

    # Models where reasoning helps (B0 -> B3 gain > 3pp)
    subheader("Reasoning benefit (B0 → B3 gain > 3pp)")
    acc_by_key = defaultdict(list)
    for r in eval_records:
        acc_by_key[(r["yaml_key"], r["budget_level"])].append(int(bool(r["nano_correct"])))

    reasoning_helps = []
    for yk in MODEL_ORDER:
        b0_vals = acc_by_key.get((yk, "B0"), [])
        b3_vals = acc_by_key.get((yk, "B3"), [])
        if b0_vals and b3_vals:
            b0_acc = sum(b0_vals) / len(b0_vals)
            b3_acc = sum(b3_vals) / len(b3_vals)
            gain = (b3_acc - b0_acc) * 100
            disp = MODEL_META[yk]["display"]
            if gain > 3.0:
                reasoning_helps.append((disp, gain))
                emit(f"  + {disp:<14}: {gain:+.1f}pp (B0={b0_acc*100:.1f}% -> B3={b3_acc*100:.1f}%)")
    if not reasoning_helps:
        emit("  None — reasoning budget provides < 3pp lift across all models")

    # Task types where frontier models justify cost (gap > 10pp vs Tier A)
    subheader("Frontier model value (gap > 10pp vs Tier-A average)")
    tier_a_keys = [yk for yk in MODEL_ORDER if MODEL_META[yk]["tier"] == "A"]
    tier_c_keys = [yk for yk in MODEL_ORDER if MODEL_META[yk]["tier"] == "C"]

    acc_by_task_model = defaultdict(list)
    for r in eval_records:
        tt = r["task_type"]
        yk = r["yaml_key"]
        if tt:
            acc_by_task_model[(tt, yk)].append(int(bool(r["nano_correct"])))

    frontier_value_tasks = []
    for tt in TASK_ORDER:
        a_vals = [sum(acc_by_task_model[(tt, yk)]) / len(acc_by_task_model[(tt, yk)])
                  for yk in tier_a_keys if acc_by_task_model[(tt, yk)]]
        c_vals = [sum(acc_by_task_model[(tt, yk)]) / len(acc_by_task_model[(tt, yk)])
                  for yk in tier_c_keys if acc_by_task_model[(tt, yk)]]
        if not a_vals or not c_vals:
            continue
        gap = (sum(c_vals) / len(c_vals)) - (sum(a_vals) / len(a_vals))
        if gap > 0.10:
            frontier_value_tasks.append((TASK_LABELS.get(tt, tt), gap))
            emit(f"  + {TASK_LABELS.get(tt,tt):<22}: Tier-C vs Tier-A gap = {gap*100:.1f}pp")
    if not frontier_value_tasks:
        emit("  None — Tier-C gap over Tier-A is < 10pp on all task types")

    # Hardest / easiest task type (by mean nano_correct across all models)
    subheader("Task difficulty ranking (mean nano_acc across all models)")
    task_mean = {}
    for tt in TASK_ORDER:
        vals = []
        for yk in MODEL_ORDER:
            vals.extend(acc_by_task_model.get((tt, yk), []))
        if vals:
            task_mean[tt] = sum(vals) / len(vals)

    ranked = sorted(task_mean.items(), key=lambda x: x[1])
    for rank, (tt, mean_acc) in enumerate(ranked, 1):
        marker = " <- HARDEST" if rank == 1 else (" <- EASIEST" if rank == len(ranked) else "")
        emit(f"  #{rank} {TASK_LABELS.get(tt,tt):<22}: {mean_acc*100:.1f}%{marker}")

    # Overall stats
    subheader("Overall")
    all_nano = [int(bool(r["nano_correct"])) for r in eval_records]
    all_rule = [int(bool(r["rule_correct"])) for r in eval_records]
    emit(f"  Mean nano_acc (all models/budgets): {sum(all_nano)/len(all_nano)*100:.1f}%")
    emit(f"  Mean rule_acc (all models/budgets): {sum(all_rule)/len(all_rule)*100:.1f}%")
    emit(f"  Nano vs rule delta:                 {(sum(all_nano)-sum(all_rule))/len(all_nano)*100:+.1f}pp")

    unique_samples = len(bench_meta)
    emit(f"  Unique anchor samples:              {unique_samples}")
    emit(f"  Avg evals per sample:               {total_calls/unique_samples:.1f}")
    emit(f"  Avg cost per sample:                ${total_cost/unique_samples:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    banner = "=" * 90
    title = "  DocRouteBench — Anchor Set Analysis (1500 samples)"
    emit(banner)
    emit(title)
    emit(banner)
    emit(f"  Data sources:")
    emit(f"    Benchmark:    {BENCHMARK_PATH}")
    emit(f"    API results:  {API_RESULTS_PATH}")
    emit(f"    Judgments:    {JUDGMENTS_PATH}")
    emit("")

    # Load data
    emit("  Loading data ...")
    bench_meta = load_benchmark_anchor()
    judgment_lookup, judgment_records = load_judgments()
    api_records, cost_lookup = load_api_results()
    emit(f"  Benchmark anchor samples:   {len(bench_meta)}")
    emit(f"  Judgment records:           {len(judgment_records)}")
    emit(f"  API result records:         {len(api_records)}")
    emit("")

    # Build merged eval records
    eval_records = build_eval_records(bench_meta, judgment_lookup, api_records)

    # Run all sections
    section1_dataset_overview(bench_meta)
    section2_model_overview(eval_records)
    section3_accuracy_by_task(eval_records)
    section4_accuracy_by_tier(eval_records)
    section5_accuracy_by_budget(eval_records)
    section6_3d_task_tier(eval_records)
    section7_cost_efficiency(eval_records)
    section8_routing_signal(eval_records)
    section9_summary(eval_records, bench_meta)

    div()
    emit("  END OF REPORT")
    div()

    # Save output
    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(output_lines) + "\n")
    print(f"\n  Report saved to: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
