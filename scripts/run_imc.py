"""
Inductive Matrix Completion (IMC) for PERCEIVE benchmark extension.

Validates that the (sample × config) evaluation matrix has learnable structure
exploitable for efficient benchmark extension — new models or new samples can
be evaluated selectively rather than exhaustively.

Experiments run:
  1. Random 30% entry hold-out        → AUC, accuracy (can IMC fill missing entries?)
  2. Per-model column hold-out (×7)   → per-model AUC (can IMC predict new model?)
  3. Sample row hold-out (300 rows)   → AUC (can IMC predict new samples?)
  4. Confidence threshold sweep       → skip%, skip accuracy (extension cost savings)
  5. Learning curve (anchor size)     → AUC vs training samples
  6. Label churn analysis             → routing label stability under model extension

Usage:
    python scripts/run_imc.py
    python scripts/run_imc.py --out-dir results/imc --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from scripts.router.config import BUDGET_TOKENS, MODEL_LIST, MODELS

BENCHMARK_PATH = _ROOT / "data" / "benchmark" / "benchmark_5000.jsonl"
EVAL_PATH = _ROOT / "data" / "model_eval_results" / "final_eval_correct.jsonl"

TASK_TYPES = ["T1", "T2", "T3", "T4", "T5", "element_localization"]

# Set to True via --no-probe-features flag to ablate VDS/RDS/SES probe scores
_NO_PROBE_FEATURES: bool = False

PROVIDER = {
    "a2_flashlite": "Google",
    "a4_gpt54nano": "OpenAI",
    "b1_gpt54mini": "OpenAI",
    "b3_sonnet":    "Anthropic",
    "c1_gpt54":     "OpenAI",
    "c2_opus":      "Anthropic",
    "c3_gemini_pro":"Google",
}
TIER_INT = {"A": 0, "B": 1, "C": 2}


# ── Feature builders ─────────────────────────────────────────────────────────

def sample_features(s: dict) -> np.ndarray:
    """13-dim sample feature vector (or 10-dim when _NO_PROBE_FEATURES=True).

    Full (13-dim): VDS, RDS, SES, tier, has_table, has_chart, has_figure, task×6
    No-probe (10-dim): tier, has_table, has_chart, has_figure, task×6
      — all derivable from document parsing alone, zero LLM probe calls required.
    """
    if _NO_PROBE_FEATURES:
        feats = [
            float(s["tier_final"]),
            float(s["has_table_detected"]),
            float(s["has_chart_detected"]),
            float(s["has_figure_detected"]),
        ]
    else:
        feats = [
            float(s["vds_probe_avg"]),
            float(s["rds_probe_avg"]),
            float(s["ses_probe_avg"]),
            float(s["tier_final"]),
            float(s["has_table_detected"]),
            float(s["has_chart_detected"]),
            float(s["has_figure_detected"]),
        ]
    task = s["task_type"]
    feats += [1.0 if task == t else 0.0 for t in TASK_TYPES]  # 6 dims
    return np.array(feats, dtype=np.float32)  # 13-dim or 10-dim


def config_features(yaml_key: str, budget_level: str) -> np.ndarray:
    """6-dim config feature vector."""
    m = MODELS[yaml_key]
    tier = TIER_INT[m.tier]
    log_cost = np.log1p(m.reasoning_rate)
    log_budget = np.log1p(BUDGET_TOKENS[budget_level])
    provider = PROVIDER[yaml_key]
    prov_feats = [
        1.0 if provider == "OpenAI" else 0.0,
        1.0 if provider == "Anthropic" else 0.0,
        1.0 if provider == "Google" else 0.0,
    ]
    return np.array([tier, log_cost, log_budget] + prov_feats, dtype=np.float32)


def bilinear_features(sx: np.ndarray, cx: np.ndarray) -> np.ndarray:
    """Outer product interaction: 13 × 6 = 78 dims."""
    return np.outer(sx, cx).ravel()


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    """Load anchor evaluation matrix + feature matrices."""
    # Load benchmark metadata for anchor samples
    samples = {}
    with open(BENCHMARK_PATH) as f:
        for line in f:
            s = json.loads(line)
            if s.get("in_anchor_set"):
                samples[s["sample_id"]] = s

    # Load eval_correct matrix (anchor only)
    eval_matrix: dict[str, dict[tuple, bool]] = {sid: {} for sid in samples}
    with open(EVAL_PATH) as f:
        for line in f:
            r = json.loads(line)
            sid = r["sample_id"]
            if sid not in samples:
                continue
            key = (r["yaml_key"], r["budget_level"])
            eval_matrix[sid][key] = bool(r["eval_correct"])

    # Build ordered lists
    sample_ids = sorted(samples.keys())
    configs = sorted({k for ev in eval_matrix.values() for k in ev.keys()})

    # Build M matrix [N_samples × N_configs]
    N, K = len(sample_ids), len(configs)
    M = np.full((N, K), np.nan)
    for i, sid in enumerate(sample_ids):
        for j, cfg in enumerate(configs):
            if cfg in eval_matrix[sid]:
                M[i, j] = float(eval_matrix[sid][cfg])

    # Build sample feature matrix X_s [N × 13]
    X_s = np.stack([sample_features(samples[sid]) for sid in sample_ids])

    # Build config feature matrix X_c [K × 6]
    X_c = np.stack([config_features(yk, bl) for yk, bl in configs])

    # Build full bilinear feature matrix X [N*K × 78] and labels y [N*K]
    rows, labels = [], []
    entry_idx = []  # (i, j) for each row
    for i in range(N):
        for j in range(K):
            if not np.isnan(M[i, j]):
                rows.append(bilinear_features(X_s[i], X_c[j]))
                labels.append(M[i, j])
                entry_idx.append((i, j))

    X = np.stack(rows)
    y = np.array(labels)

    return {
        "M": M, "X_s": X_s, "X_c": X_c, "X": X, "y": y,
        "sample_ids": sample_ids, "configs": configs,
        "entry_idx": np.array(entry_idx),
        "N": N, "K": K,
    }


# ── IMC fitter ───────────────────────────────────────────────────────────────

def fit_imc(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=500, C=1.0, solver="liblinear")
    clf.fit(X_train, y_train)
    return clf


def eval_imc(clf, X_test, y_test):
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(float)
    auc = roc_auc_score(y_test, probs)
    acc = float((preds == y_test).mean())
    return {"auc": round(auc, 4), "accuracy": round(acc, 4), "n": len(y_test)}


# ── Experiments ──────────────────────────────────────────────────────────────

def exp_random_holdout(data: dict, seed: int) -> dict:
    """Exp 1: Random 30% entry hold-out."""
    rng = np.random.default_rng(seed)
    n = len(data["y"])
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(0.3 * n)
    test_idx, train_idx = idx[:split], idx[split:]

    clf = fit_imc(data["X"][train_idx], data["y"][train_idx])
    result = eval_imc(clf, data["X"][test_idx], data["y"][test_idx])
    print(f"  [Exp 1] Random 30% hold-out: AUC={result['auc']:.4f}  Acc={result['accuracy']:.4f}  n={result['n']}")
    return result


def exp_model_holdout(data: dict) -> dict:
    """Exp 2: Per-model column hold-out (all 7 models)."""
    results = {}
    configs = data["configs"]
    entry_idx = data["entry_idx"]

    for mk in MODEL_LIST:
        # find config indices for this model (all budget levels)
        model_cfg_idx = [j for j, (yk, _) in enumerate(configs) if yk == mk]

        test_mask = np.isin(entry_idx[:, 1], model_cfg_idx)
        train_mask = ~test_mask

        X_train, y_train = data["X"][train_mask], data["y"][train_mask]
        X_test, y_test = data["X"][test_mask], data["y"][test_mask]

        if len(np.unique(y_test)) < 2:
            continue

        clf = fit_imc(X_train, y_train)
        res = eval_imc(clf, X_test, y_test)
        results[mk] = res
        print(f"  [Exp 2] Model hold-out {mk:15s}: AUC={res['auc']:.4f}  Acc={res['accuracy']:.4f}")

    aucs = [v["auc"] for v in results.values()]
    accs = [v["accuracy"] for v in results.values()]
    results["mean"] = {"auc": round(np.mean(aucs), 4), "accuracy": round(np.mean(accs), 4)}
    results["min_auc"] = round(min(aucs), 4)
    results["max_auc"] = round(max(aucs), 4)
    print(f"  [Exp 2] Mean AUC={results['mean']['auc']:.4f}  Min={results['min_auc']:.4f}  Max={results['max_auc']:.4f}")
    return results


def exp_sample_holdout(data: dict, n_holdout: int = 300, seed: int = 42) -> dict:
    """Exp 3: Sample row hold-out."""
    rng = np.random.default_rng(seed)
    sample_ids = np.arange(data["N"])
    rng.shuffle(sample_ids)
    holdout_samples = set(sample_ids[:n_holdout])

    entry_idx = data["entry_idx"]
    test_mask = np.array([i in holdout_samples for i, _ in entry_idx])
    train_mask = ~test_mask

    clf = fit_imc(data["X"][train_mask], data["y"][train_mask])
    result = eval_imc(clf, data["X"][test_mask], data["y"][test_mask])
    result["n_holdout_samples"] = n_holdout
    print(f"  [Exp 3] Sample row hold-out ({n_holdout} samples): AUC={result['auc']:.4f}  Acc={result['accuracy']:.4f}")
    return result


def exp_confidence_threshold(data: dict, seed: int) -> dict:
    """Exp 4: Confidence threshold sweep — how much can we skip?"""
    rng = np.random.default_rng(seed)
    n = len(data["y"])
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(0.3 * n)
    test_idx, train_idx = idx[:split], idx[split:]

    clf = fit_imc(data["X"][train_idx], data["y"][train_idx])
    probs = clf.predict_proba(data["X"][test_idx])[:, 1]
    y_test = data["y"][test_idx]

    results = {}
    for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        high_conf = (probs > (1 - t)) | (probs < t)
        skip_pct = float(high_conf.mean() * 100)
        if high_conf.sum() > 0:
            skip_acc = float((((probs[high_conf] >= 0.5).astype(float)) == y_test[high_conf]).mean())
        else:
            skip_acc = 0.0
        verify_pct = 100.0 - skip_pct
        results[str(t)] = {
            "threshold": t,
            "skip_pct": round(skip_pct, 1),
            "skip_accuracy": round(skip_acc, 4),
            "verify_pct": round(verify_pct, 1),
        }
        print(f"  [Exp 4] Threshold ±{t}: skip={skip_pct:.1f}%  skip_acc={skip_acc:.4f}  verify={verify_pct:.1f}%")
    return results


def exp_learning_curve(data: dict, seed: int) -> dict:
    """Exp 5: Learning curve — AUC vs number of anchor samples."""
    sizes = [150, 300, 450, 750, 1050, 1500]
    rng = np.random.default_rng(seed)
    entry_idx = data["entry_idx"]

    # Use a fixed 30% test set
    all_samples = np.arange(data["N"])
    rng.shuffle(all_samples)
    test_samples = set(all_samples[:300])
    test_mask = np.array([i in test_samples for i, _ in entry_idx])
    X_test, y_test = data["X"][test_mask], data["y"][test_mask]

    train_samples_pool = [s for s in all_samples if s not in test_samples]

    results = {}
    for size in sizes:
        train_samples = set(train_samples_pool[:size])
        train_mask = np.array([i in train_samples for i, _ in entry_idx])
        clf = fit_imc(data["X"][train_mask], data["y"][train_mask])
        res = eval_imc(clf, X_test, y_test)
        results[str(size)] = {"n_samples": size, "auc": res["auc"], "pct_anchor": round(size / 1500 * 100)}
        print(f"  [Exp 5] Learning curve n={size:4d} ({size/1500*100:.0f}%): AUC={res['auc']:.4f}")
    return results


def exp_label_churn(data: dict) -> dict:
    """Exp 6: Routing label churn — how many labels change when each model is removed?"""
    M = data["M"]
    sample_ids = data["sample_ids"]
    configs = data["configs"]
    N = data["N"]

    # Compute full routing labels (cheapest correct config per sample)
    def routing_labels(M_sub, configs_sub):
        labels = {}
        for i, sid in enumerate(sample_ids):
            correct = [(j, configs_sub[j]) for j in range(len(configs_sub))
                       if not np.isnan(M_sub[i, j]) and M_sub[i, j] == 1.0]
            if not correct:
                labels[sid] = None
                continue
            # cheapest = lowest reasoning_rate × budget_tokens as proxy
            def cost(cfg):
                yk, bl = cfg
                return MODELS[yk].reasoning_rate * (BUDGET_TOKENS[bl] + 1)
            labels[sid] = min(correct, key=lambda x: cost(x[1]))[1]
        return labels

    full_labels = routing_labels(M, configs)
    routable = {sid for sid, lbl in full_labels.items() if lbl is not None}

    results = {}
    for mk in MODEL_LIST:
        # Remove this model's columns
        keep_cfg_idx = [j for j, (yk, _) in enumerate(configs) if yk != mk]
        M_sub = M[:, keep_cfg_idx]
        configs_sub = [configs[j] for j in keep_cfg_idx]

        reduced_labels = routing_labels(M_sub, configs_sub)
        routable_both = {sid for sid in routable if reduced_labels.get(sid) is not None}

        changed = sum(
            1 for sid in routable_both
            if full_labels[sid] != reduced_labels[sid]
        )
        churn_pct = round(changed / len(routable_both) * 100, 1) if routable_both else 0.0
        results[mk] = {
            "n_routable": len(routable_both),
            "n_changed": changed,
            "churn_pct": churn_pct,
            "model_cost": MODELS[mk].reasoning_rate,
        }
        print(f"  [Exp 6] Label churn without {mk:15s}: {changed}/{len(routable_both)} = {churn_pct:.1f}%")
    return results


def exp_kshot_model_extension(data: dict, seed: int) -> dict:
    """Exp 7: K-shot model extension — partial observability sweep.

    Simulates adding a new model to the benchmark when only K anchor-set
    evaluations are available (not the full 1,500). For each of the 7 models,
    we hold out its entire column, reveal K of its anchor observations, train
    IMC on the remaining 6 models + those K revealed entries, then predict the
    held-out (N - K) entries. Sweep K ∈ {25, 50, 100, 150, 200, 300, 500, 750,
    1000, 1500}. K=1500 should reproduce Exp 2 (full column hold-out).
    """
    configs = data["configs"]
    entry_idx = data["entry_idx"]
    K_values = [25, 50, 100, 150, 200, 300, 500, 750, 1000, 1500]
    results: dict = {}

    for mk in MODEL_LIST:
        model_cfg_set = frozenset(j for j, (yk, _) in enumerate(configs) if yk == mk)
        # all sample indices that have at least one entry for this model
        model_sample_ids = sorted({int(i) for i, j in entry_idx if j in model_cfg_set})
        n_avail = len(model_sample_ids)

        results[mk] = {}
        for K in K_values:
            if K > n_avail:
                continue
            rng = np.random.default_rng(seed * 1000 + K)
            revealed = set(rng.choice(model_sample_ids, size=K, replace=False).tolist())

            train_mask = np.array([
                (j not in model_cfg_set) or (int(i) in revealed)
                for i, j in entry_idx
            ], dtype=bool)
            test_mask = np.array([
                (j in model_cfg_set) and (int(i) not in revealed)
                for i, j in entry_idx
            ], dtype=bool)

            if test_mask.sum() == 0 or len(np.unique(data["y"][test_mask])) < 2:
                continue

            clf = fit_imc(data["X"][train_mask], data["y"][train_mask])
            res = eval_imc(clf, data["X"][test_mask], data["y"][test_mask])
            results[mk][str(K)] = {
                "K": K, "auc": res["auc"], "accuracy": res["accuracy"], "n_test": res["n"],
            }
            print(f"  [Exp 7] {mk:15s}  K={K:4d}: AUC={res['auc']:.4f}  Acc={res['accuracy']:.4f}  n_test={res['n']}")

    # Mean across models at each K
    mean_by_K: dict = {}
    for K in K_values:
        ks = str(K)
        aucs = [results[mk][ks]["auc"] for mk in MODEL_LIST if ks in results.get(mk, {})]
        if aucs:
            mean_by_K[ks] = {"K": K, "mean_auc": round(float(np.mean(aucs)), 4), "n_models": len(aucs)}
            print(f"  [Exp 7] MEAN K={K:4d}: AUC={mean_by_K[ks]['mean_auc']:.4f}  ({len(aucs)} models)")
    results["mean_by_K"] = mean_by_K
    return results


def exp_trivial_baselines(data: dict, seed: int) -> dict:
    """Exp 8: Trivial baselines vs IMC on random 30% hold-out.

    Baselines:
      B1 — Global mean: predict overall training correctness rate for every entry.
      B2 — Config mean: predict per-config (column) mean correctness.
      B3 — Sample mean: predict per-sample (row) mean correctness.
      B4 — Tier × tier lookup: predict mean correctness for (doc_tier, model_tier) cell.
    All baselines use the same 70/30 split as IMC for direct comparison.
    """
    rng = np.random.default_rng(seed)
    n = len(data["y"])
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(0.3 * n)
    test_idx, train_idx = idx[:split], idx[split:]

    y_train = data["y"][train_idx]
    y_test  = data["y"][test_idx]
    ei = data["entry_idx"]          # shape [n_entries, 2]
    train_ei = ei[train_idx]        # (i, j) for training entries
    test_ei  = ei[test_idx]         # (i, j) for test entries

    results: dict = {}

    # B1: global mean
    gm = float(y_train.mean())
    preds = np.full(len(y_test), gm)
    auc = roc_auc_score(y_test, preds)
    results["global_mean"] = {"auc": round(auc, 4), "label": "Global mean"}
    print(f"  [Exp 8] B1 Global mean:        AUC={auc:.4f}")

    # B2: config mean (per-column)
    cfg_sums: dict[int, list] = {}
    for k in range(len(train_idx)):
        cfg_sums.setdefault(int(train_ei[k, 1]), []).append(y_train[k])
    cfg_means = {j: float(np.mean(v)) for j, v in cfg_sums.items()}
    preds = np.array([cfg_means.get(int(test_ei[k, 1]), gm) for k in range(len(test_idx))])
    auc = roc_auc_score(y_test, preds)
    results["config_mean"] = {"auc": round(auc, 4), "label": "Config mean (per-column)"}
    print(f"  [Exp 8] B2 Config mean:        AUC={auc:.4f}")

    # B3: sample mean (per-row)
    smp_sums: dict[int, list] = {}
    for k in range(len(train_idx)):
        smp_sums.setdefault(int(train_ei[k, 0]), []).append(y_train[k])
    smp_means = {i: float(np.mean(v)) for i, v in smp_sums.items()}
    preds = np.array([smp_means.get(int(test_ei[k, 0]), gm) for k in range(len(test_idx))])
    auc = roc_auc_score(y_test, preds)
    results["sample_mean"] = {"auc": round(auc, 4), "label": "Sample mean (per-row)"}
    print(f"  [Exp 8] B3 Sample mean:        AUC={auc:.4f}")

    # B4: doc_tier × model_tier lookup
    # X_s[:, 3] = tier_final (0=A,1=B,2=C,3=D); X_c[:, 0] = model tier (0=A,1=B,2=C)
    tier_sums: dict[tuple, list] = {}
    for k in range(len(train_idx)):
        i, j = int(train_ei[k, 0]), int(train_ei[k, 1])
        key = (int(round(float(data["X_s"][i, 3]))), int(round(float(data["X_c"][j, 0]))))
        tier_sums.setdefault(key, []).append(y_train[k])
    tier_means = {k: float(np.mean(v)) for k, v in tier_sums.items()}
    preds = np.array([
        tier_means.get(
            (int(round(float(data["X_s"][int(test_ei[k, 0]), 3]))),
             int(round(float(data["X_c"][int(test_ei[k, 1]), 0])))),
            gm,
        )
        for k in range(len(test_idx))
    ])
    auc = roc_auc_score(y_test, preds)
    results["tier_lookup"] = {"auc": round(auc, 4), "label": "Doc-tier × model-tier lookup"}
    print(f"  [Exp 8] B4 Tier×tier lookup:   AUC={auc:.4f}")

    # IMC on same split
    clf = fit_imc(data["X"][train_idx], data["y"][train_idx])
    imc_res = eval_imc(clf, data["X"][test_idx], data["y"][test_idx])
    results["imc"] = {"auc": imc_res["auc"], "label": "Bilinear IMC (ours)"}
    print(f"  [Exp 8] IMC (ours):            AUC={imc_res['auc']:.4f}")

    # Gaps
    for key in ["global_mean", "config_mean", "sample_mean", "tier_lookup"]:
        results[key]["delta_vs_imc"] = round(imc_res["auc"] - results[key]["auc"], 4)
    return results


def exp_routing_regret(data: dict, seed: int) -> dict:
    """Exp 9: Routing regret — does IMC produce good routing decisions?

    Holds out 300 samples entirely (same protocol as Exp 3). For each held-out
    routable sample IMC predicts correctness probabilities across all 24 configs.
    IMC-routing picks the cheapest config with predicted prob ≥ 0.5; we then
    check whether that config actually solved the query.

    Also reports a 'fallback' variant: if IMC predicts no config as correct,
    fall back to the cheapest model (AlwaysCheapest on that sample) — measuring
    practical routing quality with a safety net.

    Per-model breakdown: for each model, what fraction of entries routed TO that
    model by IMC are actually correct (precision) and what fraction of entries
    that model actually solves does IMC route to it (recall).
    """
    rng = np.random.default_rng(seed)
    configs = data["configs"]
    M = data["M"]
    entry_idx = data["entry_idx"]

    sample_ids = np.arange(data["N"])
    rng.shuffle(sample_ids)
    holdout_samples = set(sample_ids[:300].tolist())

    test_mask  = np.array([int(i) in holdout_samples for i, _ in entry_idx], dtype=bool)
    train_mask = ~test_mask

    clf = fit_imc(data["X"][train_mask], data["y"][train_mask])
    probs_all = clf.predict_proba(data["X"][test_mask])[:, 1]

    # Group test entries by sample
    by_sample: dict[int, list] = {}
    for k, (i, j) in enumerate(entry_idx[test_mask]):
        by_sample.setdefault(int(i), []).append((int(j), k))

    def config_cost(j: int) -> float:
        yk, bl = configs[j]
        return MODELS[yk].reasoning_rate * (BUDGET_TOKENS[bl] + 1)

    cheapest_j = min(range(data["K"]), key=config_cost)

    n_routable = 0
    imc_correct = 0
    imc_no_route = 0
    fallback_correct = 0

    # Per-model routing precision / recall
    model_routed_to:   dict[str, int] = {mk: 0 for mk in MODEL_LIST}
    model_routed_correct: dict[str, int] = {mk: 0 for mk in MODEL_LIST}
    model_oracle_count: dict[str, int]  = {mk: 0 for mk in MODEL_LIST}

    for i, jk_list in by_sample.items():
        correct_js = [j for j, _ in jk_list if not np.isnan(M[i, j]) and M[i, j] == 1.0]
        if not correct_js:
            continue  # non-routable
        n_routable += 1

        # oracle cheapest
        oracle_j = min(correct_js, key=config_cost)
        oracle_mk = configs[oracle_j][0]
        model_oracle_count[oracle_mk] += 1

        # IMC routing: cheapest predicted-correct
        predicted_ok = [(j, probs_all[k]) for j, k in jk_list if probs_all[k] >= 0.5]
        if not predicted_ok:
            imc_no_route += 1
            # fallback: cheapest available config for this sample
            avail_js = [j for j, _ in jk_list]
            fb_j = min(avail_js, key=config_cost)
            if not np.isnan(M[i, fb_j]) and M[i, fb_j] == 1.0:
                fallback_correct += 1
            continue

        imc_j = min(predicted_ok, key=lambda x: config_cost(x[0]))[0]
        imc_mk = configs[imc_j][0]
        model_routed_to[imc_mk] += 1
        is_correct = not np.isnan(M[i, imc_j]) and M[i, imc_j] == 1.0
        if is_correct:
            imc_correct += 1
            model_routed_correct[imc_mk] += 1
        else:
            # fallback counted as imc already chose, no additional fallback
            pass

    routing_acc  = imc_correct / n_routable if n_routable else 0.0
    fallback_acc = (imc_correct + fallback_correct) / n_routable if n_routable else 0.0
    no_route_pct = imc_no_route / n_routable * 100 if n_routable else 0.0

    print(f"  [Exp 9] n_routable={n_routable}  IMC routing acc={routing_acc*100:.1f}%  "
          f"regret={((1-routing_acc)*100):.1f}%  no-route={no_route_pct:.1f}%")
    print(f"  [Exp 9] With cheapest-fallback:  acc={fallback_acc*100:.1f}%  "
          f"regret={((1-fallback_acc)*100):.1f}%")

    per_model: dict = {}
    for mk in MODEL_LIST:
        rt = model_routed_to[mk]
        rc = model_routed_correct[mk]
        oc = model_oracle_count[mk]
        precision = rc / rt if rt > 0 else None
        recall    = rc / oc if oc > 0 else None
        per_model[mk] = {
            "n_routed_to": rt,
            "n_correct": rc,
            "n_oracle": oc,
            "precision": round(precision, 4) if precision is not None else None,
            "recall":    round(recall,    4) if recall    is not None else None,
        }
        print(f"  [Exp 9] {mk:15s}: routed={rt:3d}  correct={rc:3d}  "
              f"prec={precision*100:.1f}%  oracle={oc:3d}  recall={recall*100:.1f}%"
              if precision is not None and recall is not None
              else f"  [Exp 9] {mk:15s}: routed={rt}  oracle={oc}  insufficient data")

    return {
        "imc_routing_acc":  round(routing_acc,  4),
        "routing_regret":   round(1 - routing_acc, 4),
        "fallback_acc":     round(fallback_acc, 4),
        "fallback_regret":  round(1 - fallback_acc, 4),
        "no_route_pct":     round(no_route_pct, 1),
        "n_routable":       n_routable,
        "per_model":        per_model,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IMC validation experiments for PERCEIVE")
    parser.add_argument("--out-dir", default="results/imc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-probe-features", action="store_true",
        help="Ablation: drop VDS/RDS/SES probe scores, use only document-parsing features "
             "(tier, has_table, has_chart, has_figure, task_type). Runs Exp 3 only.",
    )
    args = parser.parse_args()

    global _NO_PROBE_FEATURES
    _NO_PROBE_FEATURES = args.no_probe_features

    out_dir = _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data()
    feat_dim = data["X_s"].shape[1]
    print(f"Feature mode: {'no-probe (10-dim)' if _NO_PROBE_FEATURES else 'full (13-dim)'} | X_s shape: {data['X_s'].shape}")

    # Ablation mode: run only Exp 3 (sample row hold-out) and write to separate file
    if _NO_PROBE_FEATURES:
        print("\n=== Ablation: Exp 3 (sample row hold-out, no probe features) ===")
        result = exp_sample_holdout(data, n_holdout=300, seed=args.seed)
        out_path = out_dir / "imc_noprobe_ablation.json"
        with open(out_path, "w") as f:
            json.dump({"sample_holdout_noprobe": result, "feat_dim": feat_dim}, f, indent=2)
        print(f"\nAblation result saved to {out_path}")
        print(f"No-probe AUC={result['auc']:.4f}  Acc={result['accuracy']:.4f}  n={result['n']}")
        print("Compare to full-feature Exp 3: AUC=0.8878  Acc=0.8400")
        return
    print(f"Anchor matrix: {data['N']} samples × {data['K']} configs = {len(data['y'])} observed entries")
    print(f"Matrix density: {len(data['y']) / (data['N'] * data['K']):.1%}")
    print()

    results = {}

    print("=== Exp 1: Random 30% hold-out ===")
    results["random_holdout"] = exp_random_holdout(data, args.seed)
    print()

    print("=== Exp 2: Per-model column hold-out ===")
    results["model_holdout"] = exp_model_holdout(data)
    print()

    print("=== Exp 3: Sample row hold-out (300 samples) ===")
    results["sample_holdout"] = exp_sample_holdout(data, n_holdout=300, seed=args.seed)
    print()

    print("=== Exp 4: Confidence threshold sweep ===")
    results["confidence_threshold"] = exp_confidence_threshold(data, args.seed)
    print()

    print("=== Exp 5: Learning curve ===")
    results["learning_curve"] = exp_learning_curve(data, args.seed)
    print()

    print("=== Exp 6: Label churn analysis ===")
    results["label_churn"] = exp_label_churn(data)
    print()

    print("=== Exp 7: K-shot model extension (partial observability) ===")
    results["kshot_model_extension"] = exp_kshot_model_extension(data, args.seed)
    print()

    print("=== Exp 8: Trivial baselines comparison ===")
    results["trivial_baselines"] = exp_trivial_baselines(data, args.seed)
    print()

    print("=== Exp 9: Routing regret ===")
    results["routing_regret"] = exp_routing_regret(data, args.seed)
    print()

    out_path = out_dir / "imc_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Exp 1 (random hold-out):      AUC={results['random_holdout']['auc']:.4f}")
    print(f"Exp 2 (model hold-out):       AUC mean={results['model_holdout']['mean']['auc']:.4f}  "
          f"min={results['model_holdout']['min_auc']:.4f}  max={results['model_holdout']['max_auc']:.4f}")
    print(f"Exp 3 (sample hold-out):      AUC={results['sample_holdout']['auc']:.4f}")
    t10 = results["confidence_threshold"]["0.1"]
    print(f"Exp 4 (threshold ±0.10):      skip={t10['skip_pct']}%  skip_acc={t10['skip_accuracy']:.4f}")
    lc = results["learning_curve"]
    print(f"Exp 5 (learning curve):       AUC@150={lc['150']['auc']:.4f}  AUC@300={lc['300']['auc']:.4f}  AUC@1500={lc['1500']['auc']:.4f}")
    churn = results["label_churn"]
    print(f"Exp 6 (label churn):          GPT-nano={churn['a4_gpt54nano']['churn_pct']}%  Opus={churn['c2_opus']['churn_pct']}%")
    ks = results["kshot_model_extension"]["mean_by_K"]
    k25  = ks.get("25",  {}).get("mean_auc", "n/a")
    k100 = ks.get("100", {}).get("mean_auc", "n/a")
    k300 = ks.get("300", {}).get("mean_auc", "n/a")
    print(f"Exp 7 (K-shot mean AUC):      K=25→{k25}  K=100→{k100}  K=300→{k300}")
    tb = results["trivial_baselines"]
    print(f"Exp 8 (trivial baselines):    GlobalMean={tb['global_mean']['auc']}  "
          f"TierLookup={tb['tier_lookup']['auc']}  IMC={tb['imc']['auc']}")
    rr = results["routing_regret"]
    print(f"Exp 9 (routing regret):       IMC routing acc={rr['imc_routing_acc']*100:.1f}%  "
          f"regret={rr['routing_regret']*100:.1f}%  "
          f"fallback acc={rr['fallback_acc']*100:.1f}%  no-route={rr['no_route_pct']}%")


if __name__ == "__main__":
    main()
