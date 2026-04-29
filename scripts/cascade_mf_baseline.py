"""
Cascade-MF Baseline for PERCEIVE Router.

Matrix Factorization adapted to the 2D routing setting:
  - RouteLLM-MF does binary routing (cheap vs. expensive) on text queries.
  - Cascade-MF generalises MF to a 24-class routing problem over 7 models x 4 budgets,
    using the PERCEIVE preference signal: for each query, the cheapest-correct config.

The model learns a low-rank factorisation of the (n_samples x 24) correctness matrix:
    P_hat[i, j] = sigmoid(u_i @ v_j + b_j)

where u_i is inferred from sample features X_i via a linear encoder (NOT a lookup —
this enables prediction on unseen queries at test time), and v_j is a learned config
embedding. Routing decision: pick cheapest j where P_hat[i,j] >= threshold.

Framed as: "natural generalisation of RouteLLM-MF to the 2D (model x budget) routing
setting enabled by PERCEIVE's joint routing signal."

Usage:
    python scripts/cascade_mf_baseline.py

Results saved to:
    data/cascade_mf_results/cascade_mf_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.router.config import CONFIG_COSTS, CONFIG_LIST, N_CONFIGS
from scripts.router.data_loader import load_dataset
from scripts.router.evaluate import evaluate_router

SEEDS = [42, 123, 456, 789, 2024]
RESULTS_DIR = Path("data/cascade_mf_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Sorted config costs for cheapest-above-threshold routing
_CONFIG_COST_ORDER = sorted(range(N_CONFIGS), key=lambda j: CONFIG_COSTS[j])


# ── Model ─────────────────────────────────────────────────────────────────────

class CascadeMFModel(nn.Module):
    """
    Sample features -> low-rank user vector -> dot with config embeddings.
    Inductive: works on unseen queries at test time.
    """

    def __init__(self, n_features: int, rank: int = 32, n_configs: int = N_CONFIGS):
        super().__init__()
        self.encoder = nn.Linear(n_features, rank)
        self.config_emb = nn.Embedding(n_configs, rank)
        self.config_bias = nn.Embedding(n_configs, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (batch, n_features)
        Returns:
            logits: (batch, n_configs) — raw scores before sigmoid
        """
        u = self.encoder(X)                            # (batch, rank)
        V = self.config_emb.weight                     # (n_configs, rank)
        b = self.config_bias.weight.squeeze(1)         # (n_configs,)
        return u @ V.T + b                             # (batch, n_configs)


# ── Router wrapper ────────────────────────────────────────────────────────────

class CascadeMFRouter:

    def __init__(self, n_features: int, rank: int = 32, seed: int = 42,
                 threshold: float = 0.5):
        self.rank = rank
        self.seed = seed
        self.threshold = threshold
        self.n_features = n_features
        self._model: CascadeMFModel | None = None
        self._X_mean: np.ndarray | None = None
        self._X_std: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "Cascade-MF"

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._X_mean) / (self._X_std + 1e-8)

    def fit(self, train_dataset, n_epochs: int = 60, lr: float = 3e-3,
            batch_size: int = 256, weight_decay: float = 1e-4) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = train_dataset.X.astype(np.float32)
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0)
        X = self._normalise(X)

        # Build binary correctness matrix (n_samples, 24)
        Y = train_dataset.eval_correct.astype(np.float32)

        X_t = torch.tensor(X)
        Y_t = torch.tensor(Y)

        model = CascadeMFModel(n_features=self.n_features, rank=self.rank)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loader = DataLoader(TensorDataset(X_t, Y_t), batch_size=batch_size, shuffle=True)

        model.train()
        for _ in range(n_epochs):
            for xb, yb in loader:
                logits = model(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        self._model = model

    def predict(self, dataset) -> np.ndarray:
        assert self._model is not None, "Call fit() first"
        X = self._normalise(dataset.X.astype(np.float32))
        X_t = torch.tensor(X)

        self._model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self._model(X_t)).numpy()  # (n, 24)

        # Route to cheapest config where predicted P(correct) >= threshold
        n = len(dataset.sample_ids)
        preds = np.full(n, _CONFIG_COST_ORDER[-1], dtype=np.int32)  # fallback: most expensive
        for i in range(n):
            for j in _CONFIG_COST_ORDER:
                if probs[i, j] >= self.threshold:
                    preds[i] = j
                    break
        return preds


# ── Training and evaluation ───────────────────────────────────────────────────

def run(rank: int = 32, threshold: float = 0.5, seeds: list[int] = SEEDS) -> dict:
    print("Loading data (text-only, 48-dim features)...")
    train = load_dataset("anchor", encoder=None)
    val = load_dataset("validation", encoder=None)
    print(f"  Train: {len(train.sample_ids)} samples | Val: {len(val.sample_ids)} samples")

    n_features = train.X.shape[1]
    print(f"  Features: {n_features} | Rank: {rank} | Threshold: {threshold}")

    metrics_list = []
    for seed in seeds:
        print(f"\n  Seed {seed}...")
        router = CascadeMFRouter(n_features=n_features, rank=rank,
                                 seed=seed, threshold=threshold)
        router.fit(train)
        preds = router.predict(val)
        m = evaluate_router("Cascade-MF", preds, val)
        metrics_list.append({
            "seed": seed,
            "accuracy": m.accuracy,
            "accuracy_routable": m.accuracy_routable,
            "avg_cost": m.avg_cost,
            "oracle_efficiency": m.oracle_efficiency,
            "n_correct": m.n_correct,
        })
        print(f"    acc={m.accuracy:.1%}  acc_rout={m.accuracy_routable:.1%}"
              f"  avg_cost=${m.avg_cost*1e6:.0f}µ$/q  oracle_eff={m.oracle_efficiency:.1f}%")

    accs = [r["accuracy"] for r in metrics_list]
    costs = [r["avg_cost"] for r in metrics_list]
    summary = {
        "name": "Cascade-MF",
        "rank": rank,
        "threshold": threshold,
        "n_seeds": len(seeds),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "avg_cost_mean": float(np.mean(costs)),
        "avg_cost_std": float(np.std(costs)),
        "oracle_efficiency_mean": float(np.mean([r["oracle_efficiency"] for r in metrics_list])),
        "per_seed": metrics_list,
    }
    print(f"\nCascade-MF summary: acc={summary['accuracy_mean']:.1%}"
          f" ± {summary['accuracy_std']:.1%}"
          f"  avg_cost=${summary['avg_cost_mean']*1e6:.0f}µ$/q")
    return summary


def threshold_sweep(ranks: list[int] = [16, 32, 64],
                    thresholds: list[float] = [0.3, 0.4, 0.5, 0.6]) -> list[dict]:
    results = []
    for rank in ranks:
        for thresh in thresholds:
            print(f"\n── rank={rank}  threshold={thresh} ──")
            r = run(rank=rank, threshold=thresh, seeds=[42, 123, 456])
            results.append(r)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep", action="store_true",
                        help="Run rank x threshold sweep to find best hyperparams")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    if args.sweep:
        print("Running rank x threshold sweep (3 seeds each)...")
        sweep_results = threshold_sweep()
        best = max(sweep_results, key=lambda r: r["accuracy_mean"])
        print(f"\nBest config: rank={best['rank']}  threshold={best['threshold']}"
              f"  acc={best['accuracy_mean']:.1%}")
        out = {"sweep": sweep_results, "best": best}
        (RESULTS_DIR / "sweep_results.json").write_text(json.dumps(out, indent=2))
        print(f"Sweep saved to {RESULTS_DIR}/sweep_results.json")

        # Re-run best config with all 5 seeds
        print(f"\nRe-running best config with {len(SEEDS)} seeds...")
        final = run(rank=best["rank"], threshold=best["threshold"], seeds=SEEDS)
        (RESULTS_DIR / "cascade_mf_results.json").write_text(json.dumps(final, indent=2))
        print(f"Final results saved to {RESULTS_DIR}/cascade_mf_results.json")
    else:
        result = run(rank=args.rank, threshold=args.threshold, seeds=args.seeds)
        (RESULTS_DIR / "cascade_mf_results.json").write_text(json.dumps(result, indent=2))
        print(f"\nResults saved to {RESULTS_DIR}/cascade_mf_results.json")
