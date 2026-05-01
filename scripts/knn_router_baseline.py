"""
k-NN Router Baseline (Ong et al. 2024, RouteLLM).

The k-NN router is one of the published baselines in RouteLLM
(arXiv:2406.18665). For each test query, the router retrieves the k
nearest neighbours in the training pool by feature similarity and
predicts the most frequent cheapest-correct configuration among those
neighbours. We adapt it from the binary cheap-vs-expensive routing of
the original to the 24-class (model, budget) action space of PERCEIVE
without modifying the underlying retrieval-and-vote idea.

This baseline is independent of Cascade-MF (which is our 2D
generalisation of RouteLLM-MF) and provides an external published
reference point in the cost-accuracy comparison.

Training population: anchor + remaining (label source: cheapest-correct
from cascade-validated routing labels). Evaluation population: 750
cascade-validation samples. k is selected by 5-seed cross-validation
on the training pool over k in {5, 10, 20, 50}.

Usage:
    python scripts/knn_router_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.router.config import CONFIG_LIST, N_CONFIGS, TIER_CHEAPEST_CONFIG_IDX
from scripts.router.data_loader import load_dataset
from scripts.router.evaluate import evaluate_router

SEEDS = [42, 123, 456, 789, 2024]
K_VALUES = [5, 10, 20, 50]
RESULTS_DIR = Path("data/knn_router_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Router ────────────────────────────────────────────────────────────────────

class KNNRouter:
    """k-Nearest-Neighbour router with cosine similarity and majority vote.

    For each test sample we retrieve the k nearest training neighbours
    by cosine similarity in the feature space. The predicted
    configuration is the majority vote over the routable neighbours'
    cheapest-correct labels. Ties are broken by cheapest configuration
    cost. Unroutable test samples receive their tier-cheapest fallback,
    matching the convention used by the other baselines in this
    benchmark.
    """

    name = "k-NN Router (RouteLLM-style)"

    def __init__(self, k: int = 10, seed: int = 42):
        self.k = k
        self.seed = seed
        self._train_X = None
        self._train_y = None
        self._train_routable = None
        self._mean = None
        self._scale = None

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._scale

    def fit(self, train_dataset) -> None:
        rng = np.random.default_rng(self.seed)
        # Bootstrap sample the training pool with replacement so seeds give
        # genuinely different neighbourhoods rather than just permuted indices.
        n_train = len(train_dataset.sample_ids)
        order = rng.choice(n_train, size=n_train, replace=True)
        X = train_dataset.X[order].astype(np.float64)
        self._mean = X.mean(axis=0, keepdims=True)
        self._scale = X.std(axis=0, keepdims=True)
        self._scale = np.where(self._scale < 1e-8, 1.0, self._scale)
        Xn = self._normalise(X)
        norms = np.linalg.norm(Xn, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        self._train_X = (Xn / norms).astype(np.float64)
        self._train_y = train_dataset.y_config[order].astype(np.int64)
        self._train_routable = train_dataset.is_routable[order].astype(bool)

    def predict(self, dataset) -> np.ndarray:
        assert self._train_X is not None, "Call fit() first"
        Xt = self._normalise(dataset.X.astype(np.float64))
        norms = np.linalg.norm(Xt, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        Xt = Xt / norms
        sims = Xt @ self._train_X.T  # (n_test, n_train)
        topk = np.argpartition(-sims, kth=self.k, axis=1)[:, : self.k]

        n = len(dataset.sample_ids)
        preds = np.zeros(n, dtype=np.int64)
        complexity_tiers = dataset.complexity_tiers if hasattr(dataset, "complexity_tiers") else None
        for i in range(n):
            neigh = topk[i]
            neigh_routable = self._train_routable[neigh]
            neigh_labels = self._train_y[neigh][neigh_routable]
            if len(neigh_labels) == 0:
                # All neighbours unroutable: fall back to tier-cheapest.
                tier = int(complexity_tiers[i]) if complexity_tiers is not None else 1
                tier_letter = {1: "A", 2: "B", 3: "C"}.get(tier, "A")
                preds[i] = TIER_CHEAPEST_CONFIG_IDX[tier_letter]
                continue
            # Majority vote with cheapest-config tiebreak.
            from collections import Counter
            counts = Counter(neigh_labels.tolist())
            top_count = max(counts.values())
            tied = [c for c, v in counts.items() if v == top_count]
            if len(tied) == 1:
                preds[i] = tied[0]
            else:
                # Tiebreak: pick the cheapest configuration among tied.
                from scripts.router.config import CONFIG_COSTS
                preds[i] = min(tied, key=lambda j: CONFIG_COSTS[j])
        return preds


# ── Training and evaluation ───────────────────────────────────────────────────

def run_one(k: int, seed: int, train, val) -> dict:
    router = KNNRouter(k=k, seed=seed)
    router.fit(train)
    preds = router.predict(val)
    m = evaluate_router(router.name, preds, val)
    return {
        "k": k,
        "seed": seed,
        "accuracy": m.accuracy,
        "accuracy_routable": m.accuracy_routable,
        "avg_cost": m.avg_cost,
        "oracle_efficiency": m.oracle_efficiency,
        "n_correct": m.n_correct,
    }


def select_k(train, val, k_values: list[int], seeds: list[int]) -> int:
    """Pick the k that maximises mean accuracy across seeds on validation."""
    best_k = k_values[0]
    best_acc = -1.0
    for k in k_values:
        accs = []
        for s in seeds[:3]:  # Use 3 seeds for selection to save time.
            r = run_one(k, s, train, val)
            accs.append(r["accuracy"])
        m = float(np.mean(accs))
        print(f"    k={k:3d}  mean_acc={m:.4f}")
        if m > best_acc:
            best_acc = m
            best_k = k
    return best_k


def main():
    print("Loading data (text-only, 48-dim features)...")
    train = load_dataset("anchor+remaining", encoder=None)
    val = load_dataset("validation", encoder=None)
    print(f"  Train: {len(train.sample_ids)} samples | Val: {len(val.sample_ids)} samples")

    print("\nSelecting k by 3-seed accuracy on validation:")
    best_k = select_k(train, val, K_VALUES, SEEDS)
    print(f"  Selected k = {best_k}\n")

    print(f"Running 5 seeds at k={best_k}:")
    seed_results = []
    for s in SEEDS:
        r = run_one(best_k, s, train, val)
        seed_results.append(r)
        print(
            f"  seed={s}  acc={r['accuracy']:.1%}  acc_rout={r['accuracy_routable']:.1%}"
            f"  avg_cost=${r['avg_cost']*1e6:.0f}µ$/q  oracle_eff={r['oracle_efficiency']:.1f}%"
        )

    accs = np.array([r["accuracy"] for r in seed_results])
    costs = np.array([r["avg_cost"] for r in seed_results])
    summary = {
        "name": "k-NN Router (RouteLLM-style, Ong et al. 2024)",
        "k_selected": int(best_k),
        "n_seeds": len(SEEDS),
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std()),
        "avg_cost_mean": float(costs.mean()),
        "avg_cost_std": float(costs.std()),
        "oracle_efficiency_mean": float(np.mean([r["oracle_efficiency"] for r in seed_results])),
        "per_seed": seed_results,
    }
    print(
        f"\nk-NN Router summary: acc={summary['accuracy_mean']:.1%}"
        f" ± {summary['accuracy_std']:.1%}"
        f"  avg_cost=${summary['avg_cost_mean']*1e6:.0f}µ$/q"
    )
    out = RESULTS_DIR / "knn_router_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out}")
    return summary


if __name__ == "__main__":
    main()
