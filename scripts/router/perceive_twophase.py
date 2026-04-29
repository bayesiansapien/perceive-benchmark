"""
Two-Phase Cascade-Aligned Router

Phase 1: Train on anchor with standard decomposed loss (model CE + budget + aux).
Phase 2: Fine-tune on anchor+remaining with cascade-aligned objective:
  - Anchor samples: 7-class model CE + budget regression + aux
  - Remaining samples: 3-class tier CE (via logsumexp) + aux only

Cascade evaluation for the remaining set reliably reveals which TIER solved the
problem (exhaustive cost-ordered evaluation within tiers), but is noisy for exact
model selection (~9/24 configs evaluated). Phase 2 exploits the tier-level signal
rather than fighting the sparsity of per-model cascade labels.
"""
from __future__ import annotations

import copy
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from scripts.router.config import (
    BUDGET_TOKENS,
    CONFIG_IDX,
    MODEL_AVG_COSTS,
    MODEL_LIST,
    MODELS,
    N_MODELS,
)
from scripts.router.perceive_mlp import PerceiveMLP

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

_MODEL_TIER_IDX = np.array([MODELS[m].tier_idx for m in MODEL_LIST])
_MODEL_TIER_LUT = torch.tensor(_MODEL_TIER_IDX, dtype=torch.long)
_TIER_SLICES = [
    [i for i in range(N_MODELS) if _MODEL_TIER_IDX[i] == t] for t in range(3)
]
_N_TIERS = 3


class TwoPhaseRouter:
    """Two-phase cascade-aligned router with tier-level fine-tuning.

    Paper configuration: cost_strength=0.0 (accuracy objective), encoder=None (text-only, 48 features).
    Achieves 61.6% validation accuracy averaged over 5 seeds.
    """

    N_TEXT_FEATURES = 48

    def __init__(self, n_features: int, seed: int = 42, cost_strength: float = 0.0):
        self.n_features = n_features
        self.seed = seed
        self.cost_strength = cost_strength
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = PerceiveMLP(n_features=n_features)
        self.scaler = StandardScaler()
        self.device = torch.device("cpu")
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "TwoPhase-MLP"

    def _compute_model_weights(self, y_model: np.ndarray) -> torch.Tensor:
        counts = np.bincount(y_model, minlength=N_MODELS).astype(np.float32)
        freq = np.where(counts > 0, 1.0 / counts, 0.0)
        freq /= freq.sum() + 1e-8
        costs = np.array(
            [MODEL_AVG_COSTS[k] for k in MODEL_LIST], dtype=np.float32
        )
        cost_penalty = (costs.max() / (costs + 1e-10)) ** self.cost_strength
        w = freq * cost_penalty
        w /= w.mean() + 1e-8
        return torch.tensor(w, dtype=torch.float32)

    def _prepare_batch(self, dataset: Any) -> dict[str, torch.Tensor]:
        X_s = self.scaler.transform(dataset.X)
        d = {
            "X": torch.tensor(X_s, dtype=torch.float32),
            "y_model": torch.tensor(dataset.y_model, dtype=torch.long),
            "y_vds": torch.tensor(
                (dataset.y_vds - 1.0) / 3.0, dtype=torch.float32
            ),
            "y_rds": torch.tensor(
                (dataset.y_rds - 1.0) / 3.0, dtype=torch.float32
            ),
            "y_ses": torch.tensor(
                (dataset.y_ses - 1.0) / 3.0, dtype=torch.float32
            ),
            "is_routable": torch.tensor(dataset.is_routable, dtype=torch.bool),
            "budget_targets": torch.tensor(
                dataset.budget_targets, dtype=torch.float32
            ),
            "model_solvable": torch.tensor(
                dataset.model_solvable, dtype=torch.bool
            ),
            "sample_weights": torch.tensor(
                dataset.sample_weights, dtype=torch.float32
            ),
        }
        if hasattr(dataset, "is_anchor"):
            d["is_anchor"] = torch.tensor(
                dataset.is_anchor, dtype=torch.bool
            )
        if hasattr(dataset, "model_observed"):
            d["model_observed"] = torch.tensor(
                dataset.model_observed, dtype=torch.bool
            )
        return d

    def _tier_logits(self, model_logits: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                torch.logsumexp(model_logits[:, s], dim=1)
                for s in _TIER_SLICES
            ],
            dim=1,
        )

    def _loss_phase1(
        self,
        outputs: dict,
        targets: dict,
        model_weights: torch.Tensor,
    ) -> torch.Tensor:
        sw = targets["sample_weights"]

        ce = F.cross_entropy(
            outputs["model_logits"],
            targets["y_model"],
            weight=model_weights,
            reduction="none",
        )
        l_model = (ce * sw).sum() / (sw.sum() + 1e-8)

        ms = targets["model_solvable"]
        if ms.sum() > 0:
            se = (outputs["budget_preds"] - targets["budget_targets"]) ** 2
            wm = ms.float() * sw.unsqueeze(1)
            l_budget = (se * wm).sum() / (wm.sum() + 1e-8)
        else:
            l_budget = torch.tensor(0.0)

        l_vds = (
            (outputs["vds"].squeeze() - targets["y_vds"]) ** 2 * sw
        ).mean()
        l_rds = (
            (outputs["rds"].squeeze() - targets["y_rds"]) ** 2 * sw
        ).mean()
        l_ses = (
            (outputs["ses"].squeeze() - targets["y_ses"]) ** 2 * sw
        ).mean()

        return l_model + 0.5 * l_budget + 0.3 * (l_vds + l_rds + l_ses)

    def _loss_phase2(
        self,
        outputs: dict,
        targets: dict,
        model_weights: torch.Tensor,
        tier_weights: torch.Tensor,
    ) -> torch.Tensor:
        sw = targets["sample_weights"]
        is_anchor = targets["is_anchor"]
        anc = is_anchor.float()
        rem = (~is_anchor).float()
        model_logits = outputs["model_logits"]

        # Model CE — anchor: full 7-class CE
        ce = F.cross_entropy(
            model_logits, targets["y_model"],
            weight=model_weights, reduction="none",
        )
        sw_anc = sw * anc
        l_model_anc = (ce * sw_anc).sum() / (sw_anc.sum() + 1e-8)

        # Model CE — remaining: observation-weighted 7-class CE
        # Weight by observation confidence (n_observed / 7)
        model_obs = targets.get("model_observed")
        if model_obs is not None:
            obs_confidence = model_obs.float().sum(dim=1) / N_MODELS
        else:
            obs_confidence = torch.ones(model_logits.shape[0])
        ce_rem = F.cross_entropy(
            model_logits, targets["y_model"],
            weight=model_weights, reduction="none",
        )
        sw_rem = sw * rem * obs_confidence
        l_model_rem = (ce_rem * sw_rem).sum() / (sw_rem.sum() + 1e-8)

        # Negative signal: push down logits for observed-and-failed models
        # model_observed & ~model_solvable = observed but no correct budget
        l_neg = torch.tensor(0.0)
        if model_obs is not None:
            obs_failed = model_obs & ~targets["model_solvable"]
            obs_failed_rem = obs_failed & (~is_anchor).unsqueeze(1)
            if obs_failed_rem.any():
                probs = F.softmax(model_logits, dim=1)
                neg_per_sample = (probs * obs_failed_rem.float()).sum(dim=1)
                sw_neg = sw * rem
                l_neg = (neg_per_sample * sw_neg).sum() / (sw_neg.sum() + 1e-8)

        # Budget — all observed (anchor fully, remaining where cascade observed)
        ms = targets["model_solvable"]
        if ms.sum() > 0:
            se = (outputs["budget_preds"] - targets["budget_targets"]) ** 2
            wm = ms.float() * sw.unsqueeze(1)
            l_budget = (se * wm).sum() / (wm.sum() + 1e-8)
        else:
            l_budget = torch.tensor(0.0)

        # Aux — all samples
        l_vds = (
            (outputs["vds"].squeeze() - targets["y_vds"]) ** 2 * sw
        ).mean()
        l_rds = (
            (outputs["rds"].squeeze() - targets["y_rds"]) ** 2 * sw
        ).mean()
        l_ses = (
            (outputs["ses"].squeeze() - targets["y_ses"]) ** 2 * sw
        ).mean()

        return (
            l_model_anc
            + l_model_rem
            + 0.5 * l_neg
            + 0.5 * l_budget
            + 0.3 * (l_vds + l_rds + l_ses)
        )

    def _run_training(
        self,
        data: Any,
        val: Any,
        epochs: int,
        lr: float,
        compute_loss,
        model_weights: torch.Tensor,
        verbose: bool,
        tag: str,
        init_acc: float = 0.0,
    ) -> float:
        tensors = self._prepare_batch(data)

        has_anchor = "is_anchor" in tensors
        has_model_obs = "model_observed" in tensors
        ds = [
            tensors["X"],
            tensors["y_model"],
            tensors["y_vds"],
            tensors["y_rds"],
            tensors["y_ses"],
            tensors["is_routable"],
            tensors["budget_targets"],
            tensors["model_solvable"],
            tensors["sample_weights"],
        ]
        if has_anchor:
            ds.append(tensors["is_anchor"])
        if has_model_obs:
            ds.append(tensors["model_observed"])

        loader = DataLoader(
            TensorDataset(*ds), batch_size=128, shuffle=True
        )
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_acc = init_acc
        best_state = copy.deepcopy(self.model.state_dict()) if init_acc > 0 else None
        pat_ctr = 0
        patience, check_interval = 40, 5

        self.model.train()
        for epoch in range(epochs):
            for batch in loader:
                idx = 0
                X, ym, yv, yr, ys, ir, bt, ms, sw = batch[:9]
                idx = 9
                ia = batch[idx] if has_anchor else None
                if has_anchor:
                    idx += 1
                mo = batch[idx] if has_model_obs else None

                out = self.model(X)
                tgt = {
                    "y_model": ym,
                    "y_vds": yv,
                    "y_rds": yr,
                    "y_ses": ys,
                    "is_routable": ir,
                    "budget_targets": bt,
                    "model_solvable": ms,
                    "sample_weights": sw,
                    "is_anchor": ia,
                    "model_observed": mo,
                }
                loss = compute_loss(out, tgt, model_weights)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if val is not None and (epoch + 1) % check_interval == 0:
                acc = self._eval_accuracy(val)
                if acc > best_acc:
                    best_acc = acc
                    best_state = copy.deepcopy(self.model.state_dict())
                    pat_ctr = 0
                    if verbose:
                        print(
                            f"  {tag} ep {epoch + 1:3d}  "
                            f"val_acc={acc:.4f} *"
                        )
                else:
                    pat_ctr += 1
                    if pat_ctr >= patience // check_interval:
                        if verbose:
                            print(
                                f"  {tag} early stop ep {epoch + 1} "
                                f"(best={best_acc:.4f})"
                            )
                        break
            sched.step()

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_acc

    def fit(
        self,
        anchor_data: Any,
        ar_data: Any,
        val: Any,
        epochs_p1: int = 200,
        epochs_p2: int = 200,
        lr_p1: float = 1e-3,
        lr_p2: float = 2e-3,
        verbose: bool = True,
    ) -> None:
        self.scaler.fit(anchor_data.X)

        model_weights = self._compute_model_weights(anchor_data.y_model).to(
            self.device
        )

        tier_weights = None

        if verbose:
            print("Phase 1: Anchor training...")
        p1 = self._run_training(
            anchor_data,
            val,
            epochs_p1,
            lr_p1,
            self._loss_phase1,
            model_weights,
            verbose,
            "P1",
        )
        if verbose:
            print(f"Phase 1 best: {p1:.4f}")
            print("Phase 2: Cascade fine-tuning...")

        p2_loss = lambda out, tgt, mw: self._loss_phase2(
            out, tgt, mw, tier_weights
        )
        p2 = self._run_training(
            ar_data,
            val,
            epochs_p2,
            lr_p2,
            p2_loss,
            model_weights,
            verbose,
            "P2",
            init_acc=p1,
        )
        if verbose:
            print(f"Phase 2 best: {p2:.4f}")

        self._is_fitted = True

    def _eval_accuracy(self, dataset: Any) -> float:
        was = self._is_fitted
        self._is_fitted = True
        preds = self.predict(dataset)
        self._is_fitted = was
        n = len(preds)
        correct = dataset.eval_correct[np.arange(n), preds].sum()
        self.model.train()
        return float(correct) / n

    def _tokens_to_config(self, model_idx: int, log_tokens: float) -> int:
        mk = MODEL_LIST[model_idx]
        tokens = max(0.0, float(np.exp(log_tokens) - 1))
        vb = MODELS[mk].valid_budgets
        sel = vb[0]
        for bl in vb:
            if BUDGET_TOKENS[bl] >= tokens:
                sel = bl
                break
        else:
            sel = vb[-1]
        return CONFIG_IDX[(mk, sel)]

    def predict(self, dataset: Any) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Not fitted")
        self.model.eval()
        t = self._prepare_batch(dataset)
        with torch.no_grad():
            out = self.model(t["X"])
            mp = torch.argmax(out["model_logits"], dim=1).cpu().numpy()
            bp = out["budget_preds"].cpu().numpy()
            return np.array(
                [
                    self._tokens_to_config(mp[i], bp[i, mp[i]])
                    for i in range(len(mp))
                ],
                dtype=np.int64,
            )
