#!/usr/bin/env python3
"""
Train PERCEIVE-IPS (TwoPhaseRouter), text-only, two-phase cascade-aligned router.

Paper configuration (reproduces 61.6%):
    encoder = None   (text-only, 48 features)
    cost_strength = 0.0  (accuracy-focused)
    epochs_p1 = 200, epochs_p2 = 200
    lr_p1 = 1e-3,    lr_p2 = 2e-3
    seeds  = 5  (report mean ± std)

Source for paper number: encoder ablation study, scenario "full+none"
(scripts/experiment_encoder_ablation.py on the router-modelling branch).
CLIP/MobileNet embeddings are available in data/embeddings/ for research
but reduce accuracy by ~2pp at the 1,500-sample anchor scale.

Usage:
    python scripts/train_perceive_router.py             # 1 seed (~30 s)
    python scripts/train_perceive_router.py --seeds 5   # 5-seed mean ± std (~2 min)
    python scripts/train_perceive_router.py --encoder clip   # CLIP ablation (~2 min)

Results saved to:
    results/router/perceive_ips_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.router.data_loader import load_dataset
from scripts.router.perceive_twophase import TwoPhaseRouter

RESULTS_DIR = ROOT / "results" / "router"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Ablation hyperparameters that reproduce the paper's 61.6%
_EPOCHS_P1 = 200
_EPOCHS_P2 = 200
_LR_P1     = 1e-3
_LR_P2     = 2e-3


def train_once(encoder: str | None, seed: int, verbose: bool) -> float:
    """Train one TwoPhaseRouter and return validation accuracy."""
    print(f"\n--- seed={seed}, encoder={encoder} ---")

    anchor = load_dataset("anchor",           encoder=encoder)
    ar     = load_dataset("anchor+remaining", encoder=encoder)
    val    = load_dataset("validation",       encoder=encoder)

    n_features = anchor.X.shape[1]
    print(f"  Features: {n_features}  (anchor={len(anchor.sample_ids)}, "
          f"ar={len(ar.sample_ids)}, val={len(val.sample_ids)})")

    router = TwoPhaseRouter(n_features=n_features, seed=seed)
    router.fit(
        anchor, ar, val,
        epochs_p1=_EPOCHS_P1, epochs_p2=_EPOCHS_P2,
        lr_p1=_LR_P1, lr_p2=_LR_P2,
        verbose=verbose,
    )

    acc = router._eval_accuracy(val)
    print(f"  Final val accuracy: {acc:.4f} ({acc*100:.2f}%)")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Train PERCEIVE-IPS router")
    parser.add_argument(
        "--encoder", default="none",
        help="Image encoder: clip, mobilenet, or none (default: none = text-only, matches paper)",
    )
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="Number of random seeds to average over (5 for paper figure)",
    )
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--no-verbose", dest="verbose", action="store_false")
    args = parser.parse_args()

    encoder = None if args.encoder.lower() == "none" else args.encoder

    accs = []
    for i in range(args.seeds):
        seed = [42, 123, 456, 789, 2024][i % 5]
        acc = train_once(encoder, seed, args.verbose)
        accs.append(acc)

    import numpy as np
    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs)) if len(accs) > 1 else 0.0

    print(f"\n{'='*60}")
    print(f"PERCEIVE-IPS  encoder={encoder}  seeds={args.seeds}")
    print(f"  mean accuracy: {mean_acc*100:.2f}%  std: {std_acc*100:.2f}%")
    print(f"{'='*60}")

    result = {
        "encoder": str(encoder),
        "seeds": args.seeds,
        "seed_list": [42, 123, 456, 789, 2024][:args.seeds],
        "accuracies": [round(a, 6) for a in accs],
        "mean_accuracy": round(mean_acc, 6),
        "std_accuracy": round(std_acc, 6),
        "mean_accuracy_pct": round(mean_acc * 100, 2),
        "hyperparameters": {
            "epochs_p1": _EPOCHS_P1, "epochs_p2": _EPOCHS_P2,
            "lr_p1": _LR_P1,         "lr_p2": _LR_P2,
            "cost_strength": 0.0,    "n_features": 48,
        },
    }

    out_path = RESULTS_DIR / "perceive_ips_results.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Results written to {out_path}")
    return result


if __name__ == "__main__":
    main()
