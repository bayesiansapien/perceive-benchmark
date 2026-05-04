"""
PERCEIVE Decomposed MLP Router

Cost-aware neural router with decomposed model+budget architecture.

Architecture:
  - Shared trunk: feature extraction with dropout regularization
  - Stage 1: Three regression heads for VDS/RDS/SES (auxiliary supervision)
  - Stage 2: Shared projection from trunk+complexity, then:
    - Model head (7-class classifier)
    - Budget regressors (7 independent regressors, one per model)

Key innovation: Instead of predicting 24 configs directly, we decompose into:
  1. Which model to use (7-class classification with cost weighting)
  2. What budget to use for that model (per-model regression on reasoning tokens)

Training loss:
  L = lambda_model * L_model + lambda_budget * L_budget + lambda_aux * (L_vds + L_rds + L_ses)

where:
  - L_model uses cost-weighted cross-entropy to prefer cheaper models
  - L_budget is masked MSE on log(reasoning_tokens+1), trained only on solvable models
  - L_aux provides auxiliary supervision from complexity metrics
"""
from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
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

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


class PerceiveMLP(nn.Module):
    """
    PERCEIVE decomposed architecture (~270K params with CLIP).

    3-layer shared trunk with BatchNorm. Decomposed into:
      Stage 1: trunk -> complexity heads (VDS/RDS/SES auxiliary supervision)
      Stage 2: projection(trunk + complexity) -> model head + budget regressors
    """

    def __init__(self, n_features: int, n_models: int = N_MODELS):
        super().__init__()

        # Shared trunk: 3-layer MLP with BatchNorm
        self.trunk = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Stage 1: Complexity heads (auxiliary supervision)
        self.vds_head = nn.Linear(128, 1)
        self.rds_head = nn.Linear(128, 1)
        self.ses_head = nn.Linear(128, 1)

        # Stage 2: Projection from trunk(128) + complexity(3) = 131 -> 128
        self.projection = nn.Sequential(
            nn.Linear(131, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Stage 2a: Model classifier
        self.model_head = nn.Linear(64, n_models)

        # Stage 2b: Budget regressors (7 independent heads)
        self.budget_regressors = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(n_models)
        ])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        trunk_features = self.trunk(x)

        # Stage 1: Predict complexity metrics
        vds_pred = torch.sigmoid(self.vds_head(trunk_features))
        rds_pred = torch.sigmoid(self.rds_head(trunk_features))
        ses_pred = torch.sigmoid(self.ses_head(trunk_features))

        # Concatenate trunk + complexity for stage 2
        combined = torch.cat([trunk_features, vds_pred, rds_pred, ses_pred], dim=1)
        stage2_features = self.projection(combined)

        # Stage 2a: Model classification
        model_logits = self.model_head(stage2_features)

        # Stage 2b: Budget regression (7 independent predictions)
        budget_preds = torch.stack([
            regressor(stage2_features).squeeze(-1)
            for regressor in self.budget_regressors
        ], dim=1)

        return {
            "vds": vds_pred,
            "rds": rds_pred,
            "ses": ses_pred,
            "model_logits": model_logits,
            "budget_preds": budget_preds,
            "stage2_features": stage2_features,
        }


class PerceiveRouter:
    """
    PERCEIVE Router: decomposed model+budget architecture with cost-aware training.

    Predicts:
      1. Which model to use (7-class classification)
      2. What budget to use for that model (per-model regression)

    Then maps (model, budget) to one of 24 valid configs.
    """

    # Number of text-only features (CLIP features start after this index)
    N_TEXT_FEATURES = 48

    def __init__(
        self,
        n_features: int,
        seed: int = 42,
        clip_mask_prob: float = 0.0,
        remaining_loss_mode: str = "all",
        cost_strength: float = 1.0,
    ):
        """
        Initialize router.

        Args:
            n_features: Number of input features (48 text-only, 560 with CLIP)
            seed: Random seed for reproducibility
            clip_mask_prob: Probability of zeroing out CLIP features per sample during
                training (modality dropout). No-op when n_features <= N_TEXT_FEATURES.
            remaining_loss_mode: What losses non-anchor samples contribute to.
                "all", model CE on anchor, budget+aux on all (default)
                "aux_only", model+budget on anchor, aux on all
                "none", all losses on anchor only (remaining for BN enrichment)
            cost_strength: Exponent on cost penalty in model class weights.
                0.0 = pure inverse-frequency (no cost awareness)
                1.0 = full cost penalty (default, original behavior)
        """
        self.n_features = n_features
        self.seed = seed
        self.clip_mask_prob = clip_mask_prob
        self.remaining_loss_mode = remaining_loss_mode
        self.cost_strength = cost_strength

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Model and preprocessing
        self.model = PerceiveMLP(n_features=n_features, n_models=N_MODELS)
        self.scaler = StandardScaler()
        self.device = torch.device("cpu")  # CPU-only, small model

        # Training state
        self._is_fitted = False
        self._best_val_loss = float("inf")

    @property
    def name(self) -> str:
        return "PERCEIVE-MLP"

    def _compute_model_weights(self, y_model: np.ndarray) -> torch.Tensor:
        """
        Compute cost-aware class weights for model prediction.

        weight[i] = (1 / freq[i]) * (max_cost / model_cost[i])

        This upweights cheap, under-represented models.

        Args:
            y_model: (n_samples,) target model indices (-1 for non-routable)

        Returns:
            (n_models,) tensor of class weights
        """
        # Inverse frequency weights (all samples have valid targets now)
        counts = np.bincount(y_model, minlength=N_MODELS)
        freq_weights = np.zeros(N_MODELS, dtype=np.float32)
        for i in range(N_MODELS):
            if counts[i] > 0:
                freq_weights[i] = 1.0 / counts[i]

        # Normalize frequency weights
        freq_weights = freq_weights / (freq_weights.sum() + 1e-8)

        # Cost penalty: upweight cheaper models (controlled by cost_strength)
        model_costs = np.array(
            [MODEL_AVG_COSTS[yk] for yk in MODEL_LIST], dtype=np.float32
        )
        max_cost = model_costs.max()
        cost_penalty = max_cost / (model_costs + 1e-10)
        cost_penalty = cost_penalty ** self.cost_strength

        # Combined weight
        class_weights = freq_weights * cost_penalty

        # Normalize to mean=1.0 for stability
        class_weights = class_weights / (class_weights.mean() + 1e-8)

        return torch.tensor(class_weights, dtype=torch.float32)

    def _scale_complexity(self, y: np.ndarray) -> np.ndarray:
        """Scale complexity metric from [1, 4] to [0, 1]."""
        return (y - 1.0) / 3.0

    def _prepare_batch(self, dataset: Any) -> dict[str, torch.Tensor]:
        """Convert RouterDataset to torch tensors."""
        X_scaled = self.scaler.transform(dataset.X)

        batch = {
            "X": torch.tensor(X_scaled, dtype=torch.float32),
            "y_model": torch.tensor(dataset.y_model, dtype=torch.long),
            "y_vds": torch.tensor(
                self._scale_complexity(dataset.y_vds), dtype=torch.float32
            ),
            "y_rds": torch.tensor(
                self._scale_complexity(dataset.y_rds), dtype=torch.float32
            ),
            "y_ses": torch.tensor(
                self._scale_complexity(dataset.y_ses), dtype=torch.float32
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
            batch["is_anchor"] = torch.tensor(dataset.is_anchor, dtype=torch.bool)
        return batch

    def _compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        model_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute decomposed loss.

        L = lambda_model * L_model + lambda_budget * L_budget
            + lambda_aux * (L_vds + L_rds + L_ses)

        Args:
            outputs: Model outputs from forward pass
            targets: Ground truth targets
            model_weights: (n_models,) class weights for model loss

        Returns:
            (total_loss, loss_dict) where loss_dict contains individual components
        """
        lambda_model = 1.0
        lambda_budget = 0.5
        lambda_aux = 0.3

        sw = targets["sample_weights"]  # (batch,)
        is_anchor = targets.get("is_anchor")  # (batch,) bool or None

        # Per-component sample weights based on remaining_loss_mode
        if is_anchor is not None:
            anchor_float = is_anchor.float()
            sw_model = sw * anchor_float  # model CE always anchor-only
            if self.remaining_loss_mode == "none":
                sw_budget = sw * anchor_float
                sw_aux = sw * anchor_float
            elif self.remaining_loss_mode == "aux_only":
                sw_budget = sw * anchor_float
                sw_aux = sw
            else:  # "all"
                sw_budget = sw
                sw_aux = sw
        else:
            sw_model = sw
            sw_budget = sw
            sw_aux = sw

        # ---- Model classification loss (anchor-only when is_anchor provided) ----
        per_sample_ce = F.cross_entropy(
            outputs["model_logits"],
            targets["y_model"],
            weight=model_weights,
            reduction="none",
        )
        loss_model = (per_sample_ce * sw_model).sum() / (sw_model.sum() + 1e-8)

        # ---- Budget regression loss (masked per-sample, per-model) ----
        budget_preds = outputs["budget_preds"]      # (batch, 7)
        budget_targets = targets["budget_targets"]   # (batch, 7)
        model_solvable = targets["model_solvable"]   # (batch, 7)

        if model_solvable.sum() > 0:
            squared_errors = (budget_preds - budget_targets) ** 2
            weighted_mask = model_solvable.float() * sw_budget.unsqueeze(1)
            loss_budget = (squared_errors * weighted_mask).sum() / (weighted_mask.sum() + 1e-8)
        else:
            loss_budget = torch.tensor(0.0)

        # ---- Auxiliary losses ----
        loss_vds = ((outputs["vds"].squeeze() - targets["y_vds"]) ** 2 * sw_aux).mean()
        loss_rds = ((outputs["rds"].squeeze() - targets["y_rds"]) ** 2 * sw_aux).mean()
        loss_ses = ((outputs["ses"].squeeze() - targets["y_ses"]) ** 2 * sw_aux).mean()

        # ---- Total loss ----
        total_loss = (
            lambda_model * loss_model
            + lambda_budget * loss_budget
            + lambda_aux * (loss_vds + loss_rds + loss_ses)
        )

        loss_dict = {
            "total": total_loss.item(),
            "model": loss_model.item(),
            "budget": loss_budget.item() if model_solvable.sum() > 0 else 0.0,
            "vds": loss_vds.item(),
            "rds": loss_rds.item(),
            "ses": loss_ses.item(),
        }

        return total_loss, loss_dict

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        checkpoint_path: str,
        model_weights: torch.Tensor,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "scaler_state": {
                "mean_": self.scaler.mean_,
                "scale_": self.scaler.scale_,
                "var_": self.scaler.var_,
                "n_features_in_": self.scaler.n_features_in_,
                "n_samples_seen_": self.scaler.n_samples_seen_,
            },
            "epoch": epoch,
            "best_val_loss": val_loss,
            "config_weights": model_weights.cpu().numpy(),
            "seed": self.seed,
            "n_features": self.n_features,
        }
        torch.save(checkpoint, checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Restore model
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Restore scaler
        scaler_state = checkpoint["scaler_state"]
        self.scaler.mean_ = scaler_state["mean_"]
        self.scaler.scale_ = scaler_state["scale_"]
        self.scaler.var_ = scaler_state["var_"]
        self.scaler.n_features_in_ = scaler_state["n_features_in_"]
        self.scaler.n_samples_seen_ = scaler_state["n_samples_seen_"]

        self._is_fitted = True
        self._best_val_loss = checkpoint["best_val_loss"]

        return checkpoint

    def _log_epoch(
        self,
        log_path: Path,
        epoch: int,
        train_loss: float,
        val_loss: float | None,
        loss_dict: dict[str, float],
        lr: float,
        is_best: bool,
    ) -> None:
        """Append one JSON line per epoch to the training log."""
        record = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
            "loss_model": round(loss_dict.get("model", 0.0), 6),
            "loss_budget": round(loss_dict.get("budget", 0.0), 6),
            "loss_vds": round(loss_dict.get("vds", 0.0), 6),
            "loss_rds": round(loss_dict.get("rds", 0.0), 6),
            "loss_ses": round(loss_dict.get("ses", 0.0), 6),
            "lr": round(lr, 8),
            "is_best": is_best,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def fit(
        self,
        train: Any,
        val: Any | None = None,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 128,
        verbose: bool = True,
        checkpoint_dir: str | None = None,
        resume_from: str | None = None,
        scaler_data: Any | None = None,
    ) -> None:
        """
        Train the router.

        Args:
            train: Training dataset (RouterDataset or duck-typed equivalent)
            val: Optional validation dataset for early stopping
            epochs: Maximum number of training epochs
            lr: Learning rate
            batch_size: Batch size
            verbose: Print training progress
            checkpoint_dir: If provided, save checkpoints to this directory
            resume_from: If provided, path to checkpoint to resume from
            scaler_data: If provided, fit scaler on this dataset instead of train.
                         Useful for fitting on anchor distribution when training on A+R.
        """
        start_epoch = 0

        # Resolve checkpoint paths
        _repo_root = Path(__file__).resolve().parents[2]
        if checkpoint_dir is not None:
            ckpt_dir = Path(checkpoint_dir)
        else:
            ckpt_dir = _repo_root / "data" / "phase4_results" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(ckpt_dir / "perceive_best.pt")

        # Training log
        log_dir = _repo_root / "data" / "phase4_results"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "training_log.jsonl"

        # Resume from checkpoint if specified
        if resume_from is not None and Path(resume_from).exists():
            if verbose:
                print(f"Resuming from checkpoint: {resume_from}")
            checkpoint = self._load_checkpoint(resume_from)
            start_epoch = checkpoint["epoch"] + 1
            if verbose:
                print(
                    f"  Resumed at epoch {start_epoch}, "
                    f"best_val_loss={self._best_val_loss:.4f}"
                )
        else:
            # Fit scaler on specified data (or training data by default)
            scaler_source = scaler_data if scaler_data is not None else train
            self.scaler.fit(scaler_source.X)

        # Prepare data
        train_tensors = self._prepare_batch(train)
        model_weights = self._compute_model_weights(train.y_model).to(
            self.device
        )

        # Create DataLoader
        has_anchor_mask = "is_anchor" in train_tensors
        dataset_tensors = [
            train_tensors["X"],
            train_tensors["y_model"],
            train_tensors["y_vds"],
            train_tensors["y_rds"],
            train_tensors["y_ses"],
            train_tensors["is_routable"],
            train_tensors["budget_targets"],
            train_tensors["model_solvable"],
            train_tensors["sample_weights"],
        ]
        if has_anchor_mask:
            dataset_tensors.append(train_tensors["is_anchor"])
        train_dataset = TensorDataset(*dataset_tensors)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        # Skip to the right scheduler state if resuming
        if resume_from is not None and start_epoch > 0:
            for _ in range(start_epoch):
                scheduler.step()

        # Early stopping based on validation accuracy
        best_val_acc = 0.0
        best_model_state: dict | None = None
        patience_counter = 0
        patience = 40
        acc_check_interval = 5

        # Training loop
        self.model.train()
        for epoch in range(start_epoch, epochs):
            epoch_losses = []
            last_loss_dict: dict[str, float] = {}

            for batch in train_loader:
                if has_anchor_mask:
                    (X_batch, y_model_batch, y_vds_batch, y_rds_batch,
                     y_ses_batch, is_routable_batch, budget_targets_batch,
                     model_solvable_batch, sample_weights_batch,
                     is_anchor_batch) = batch
                else:
                    (X_batch, y_model_batch, y_vds_batch, y_rds_batch,
                     y_ses_batch, is_routable_batch, budget_targets_batch,
                     model_solvable_batch, sample_weights_batch) = batch
                    is_anchor_batch = None

                # Modality dropout: zero out CLIP features with probability clip_mask_prob
                if self.clip_mask_prob > 0 and X_batch.shape[1] > self.N_TEXT_FEATURES:
                    mask = (torch.rand(X_batch.shape[0], 1) > self.clip_mask_prob).float()
                    X_batch = X_batch.clone()
                    X_batch[:, self.N_TEXT_FEATURES:] *= mask

                # Forward pass
                outputs = self.model(X_batch)

                # Compute loss
                targets = {
                    "y_model": y_model_batch,
                    "y_vds": y_vds_batch,
                    "y_rds": y_rds_batch,
                    "y_ses": y_ses_batch,
                    "is_routable": is_routable_batch,
                    "budget_targets": budget_targets_batch,
                    "model_solvable": model_solvable_batch,
                    "sample_weights": sample_weights_batch,
                    "is_anchor": is_anchor_batch,
                }
                loss, last_loss_dict = self._compute_loss(
                    outputs, targets, model_weights
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(last_loss_dict["total"])

            # Epoch statistics
            mean_train_loss = float(np.mean(epoch_losses))
            current_lr = scheduler.get_last_lr()[0]

            # Validation: accuracy-based model selection
            val_loss: float | None = None
            is_best = False

            if val is not None:
                val_loss = self._evaluate(val, model_weights)

                if (epoch + 1) % acc_check_interval == 0:
                    val_acc = self._eval_accuracy(val)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model_state = copy.deepcopy(self.model.state_dict())
                        patience_counter = 0
                        is_best = True
                        if verbose:
                            print(
                                f"Epoch {epoch + 1:3d}/{epochs}  "
                                f"train={mean_train_loss:.4f}  "
                                f"val_acc={val_acc:.4f}  * best"
                            )
                    else:
                        patience_counter += 1
                        if patience_counter >= patience // acc_check_interval:
                            if verbose:
                                print(f"Early stopping at epoch {epoch + 1} "
                                      f"(best val_acc={best_val_acc:.4f})")
                            self._log_epoch(
                                log_path, epoch, mean_train_loss, val_loss,
                                last_loss_dict, current_lr, is_best,
                            )
                            break

            # Log epoch to JSONL
            self._log_epoch(
                log_path, epoch, mean_train_loss, val_loss,
                last_loss_dict, current_lr, is_best,
            )

            # Learning rate step
            scheduler.step()

            # Print progress every 20 epochs
            if verbose and (epoch + 1) % 20 == 0 and not is_best:
                if val is not None:
                    print(
                        f"Epoch {epoch + 1:3d}/{epochs}  "
                        f"train={mean_train_loss:.4f}  "
                        f"val={val_loss:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1:3d}/{epochs}  "
                        f"train={mean_train_loss:.4f}"
                    )

        # Restore best-accuracy model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        self._is_fitted = True

        if verbose:
            if val is not None:
                print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
            else:
                print(
                    f"Training complete. Final loss: {mean_train_loss:.4f}"
                )

    def _evaluate(
        self, dataset: Any, model_weights: torch.Tensor
    ) -> float:
        """
        Evaluate total loss on a dataset.

        Args:
            dataset: Dataset to evaluate
            model_weights: Model class weights

        Returns:
            Total validation loss (float)
        """
        self.model.eval()

        tensors = self._prepare_batch(dataset)

        with torch.no_grad():
            outputs = self.model(tensors["X"])

            targets = {
                "y_model": tensors["y_model"],
                "y_vds": tensors["y_vds"],
                "y_rds": tensors["y_rds"],
                "y_ses": tensors["y_ses"],
                "is_routable": tensors["is_routable"],
                "budget_targets": tensors["budget_targets"],
                "model_solvable": tensors["model_solvable"],
                "sample_weights": tensors["sample_weights"],
            }

            _, loss_dict = self._compute_loss(outputs, targets, model_weights)

        self.model.train()
        return loss_dict["total"]

    def _eval_accuracy(self, dataset: Any) -> float:
        """Compute routing accuracy on a dataset (fraction of correct predictions)."""
        was_fitted = self._is_fitted
        self._is_fitted = True
        preds = self.predict(dataset)
        self._is_fitted = was_fitted
        n = len(preds)
        correct = dataset.eval_correct[np.arange(n), preds].sum()
        self.model.train()
        return float(correct) / n

    def _tokens_to_config(
        self, model_idx: int, predicted_log_tokens: float
    ) -> int:
        """
        Map (model, predicted_log_tokens) to config index.

        Strategy: Find the cheapest valid budget that accommodates the
        predicted tokens. If predicted tokens exceed all caps, use the
        highest available budget.

        Args:
            model_idx: Index into MODEL_LIST
            predicted_log_tokens: log(reasoning_tokens + 1)

        Returns:
            Config index (0-23)
        """
        model_key = MODEL_LIST[model_idx]
        tokens = max(0.0, float(np.exp(predicted_log_tokens) - 1))
        valid_budgets = MODELS[model_key].valid_budgets

        # Find cheapest budget that accommodates the predicted tokens
        selected_budget = valid_budgets[0]  # default: cheapest
        for bl in valid_budgets:
            if BUDGET_TOKENS[bl] >= tokens:
                selected_budget = bl
                break
        else:
            # All budgets < tokens: use max budget
            selected_budget = valid_budgets[-1]

        return CONFIG_IDX[(model_key, selected_budget)]

    def predict(self, dataset: Any) -> np.ndarray:
        """
        Predict optimal config for each sample.

        Args:
            dataset: Dataset to predict on

        Returns:
            (n_samples,) int array of config indices into CONFIG_LIST
        """
        if not self._is_fitted:
            raise RuntimeError("Router must be fitted before prediction")

        self.model.eval()

        tensors = self._prepare_batch(dataset)

        with torch.no_grad():
            outputs = self.model(tensors["X"])
            model_logits = outputs["model_logits"]
            budget_preds = outputs["budget_preds"]

            # Predict model (argmax of 7-class logits)
            model_predictions = (
                torch.argmax(model_logits, dim=1).cpu().numpy()
            )

            # Get budget predictions
            budget_log_tokens = budget_preds.cpu().numpy()

            # Map to config indices
            config_predictions = np.array(
                [
                    self._tokens_to_config(
                        model_idx, budget_log_tokens[i, model_idx]
                    )
                    for i, model_idx in enumerate(model_predictions)
                ],
                dtype=np.int64,
            )

        return config_predictions

    def predict_model_budget(
        self, dataset: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict model and reasoning tokens for detailed analysis.

        Args:
            dataset: Dataset to predict on

        Returns:
            (model_indices, predicted_reasoning_tokens)
              - model_indices: (n_samples,) int array of model indices (0-6)
              - predicted_reasoning_tokens: (n_samples,) float array
        """
        if not self._is_fitted:
            raise RuntimeError("Router must be fitted before prediction")

        self.model.eval()

        tensors = self._prepare_batch(dataset)

        with torch.no_grad():
            outputs = self.model(tensors["X"])
            model_logits = outputs["model_logits"]
            budget_preds = outputs["budget_preds"]

            # Predict model (argmax of 7-class logits)
            model_predictions = (
                torch.argmax(model_logits, dim=1).cpu().numpy()
            )

            # Get budget prediction for the predicted model
            budget_log_tokens = budget_preds.cpu().numpy()
            predicted_tokens = np.array(
                [
                    max(0.0, float(np.exp(budget_log_tokens[i, model_idx]) - 1))
                    for i, model_idx in enumerate(model_predictions)
                ],
                dtype=np.float32,
            )

        return model_predictions, predicted_tokens
