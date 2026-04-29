"""
Baseline routing strategies for PERCEIVE router benchmarking (NeurIPS 2026).

Implements 8 baselines:
1. AlwaysCheapest - always route to cheapest config
2. AlwaysBest - always route to most expensive config
3. RandomRouter - uniform random selection
4. ComplexityThresholdRouter - threshold cascade based on complexity
5. LogisticRouter - multinomial logistic regression
6. OracleCascadeRouter - oracle FrugalGPT-style cascade (uses eval_correct)
7. LearnedCascadeRouter - learned tier escalation (FrugalGPT-style)
8. CostSensitiveMLP - flat MLP with cost-weighted cross-entropy
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Optional

from scripts.router.config import (
    COMPLEXITY_TIER_TO_MODEL_TIER,
    CONFIG_COSTS, CONFIG_IDX, CONFIG_LIST, N_CONFIGS,
    CHEAPEST_CONFIG_IDX, MOST_EXPENSIVE_CONFIG_IDX, MODELS, MODEL_LIST,
    TIER_CHEAPEST_CONFIG_IDX,
)


class AlwaysCheapest:
    """Always routes to the cheapest config (a4_gpt54nano B0)."""

    @property
    def name(self) -> str:
        return "AlwaysCheapest"

    def fit(self, train) -> None:
        """No training needed for static baseline."""
        pass

    def predict(self, dataset) -> np.ndarray:
        """Return cheapest config for all samples."""
        n_samples = len(dataset.sample_ids)
        return np.full(n_samples, CHEAPEST_CONFIG_IDX, dtype=np.int32)


class AlwaysBest:
    """Always routes to the most expensive config (c2_opus B3)."""

    @property
    def name(self) -> str:
        return "AlwaysBest"

    def fit(self, train) -> None:
        """No training needed for static baseline."""
        pass

    def predict(self, dataset) -> np.ndarray:
        """Return most expensive config for all samples."""
        n_samples = len(dataset.sample_ids)
        return np.full(n_samples, MOST_EXPENSIVE_CONFIG_IDX, dtype=np.int32)


class RandomRouter:
    """Uniform random selection from all 24 configs."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @property
    def name(self) -> str:
        return f"RandomRouter(seed={self.seed})"

    def fit(self, train) -> None:
        """No training needed for random baseline."""
        pass

    def predict(self, dataset) -> np.ndarray:
        """Return uniform random config for all samples."""
        n_samples = len(dataset.sample_ids)
        return self.rng.randint(0, N_CONFIGS, size=n_samples, dtype=np.int32)


class ComplexityThresholdRouter:
    """
    FrugalGPT-style threshold cascade based on composite complexity score.

    Uses two thresholds (t1, t2) to route to cheapest config in each tier:
    - composite <= t1 → cheapest Tier A config
    - t1 < composite <= t2 → cheapest Tier B config
    - composite > t2 → cheapest Tier C config

    Thresholds are tuned on training data to maximize accuracy.
    """

    def __init__(self):
        self.t1: Optional[float] = None
        self.t2: Optional[float] = None
        self.tier_configs: Optional[dict] = None

    @property
    def name(self) -> str:
        return "ComplexityThreshold"

    def fit(self, train) -> None:
        """
        Grid-search thresholds t1, t2 to maximize accuracy on routable samples.
        """
        # Find cheapest config in each tier
        tier_a_configs = [(i, CONFIG_COSTS[i]) for i, (model, budget) in enumerate(CONFIG_LIST)
                          if MODELS[model].tier == 'A']
        tier_b_configs = [(i, CONFIG_COSTS[i]) for i, (model, budget) in enumerate(CONFIG_LIST)
                          if MODELS[model].tier == 'B']
        tier_c_configs = [(i, CONFIG_COSTS[i]) for i, (model, budget) in enumerate(CONFIG_LIST)
                          if MODELS[model].tier == 'C']

        cheapest_a = min(tier_a_configs, key=lambda x: x[1])[0]
        cheapest_b = min(tier_b_configs, key=lambda x: x[1])[0]
        cheapest_c = min(tier_c_configs, key=lambda x: x[1])[0]

        self.tier_configs = {
            'A': cheapest_a,
            'B': cheapest_b,
            'C': cheapest_c
        }

        # Extract composite scores from routable training samples
        routable_mask = train.is_routable
        if not np.any(routable_mask):
            # No routable samples - use default thresholds
            self.t1 = 1.5
            self.t2 = 2.5
            return

        composite_scores = train.X[routable_mask, 3]  # composite_est is at index 3
        eval_correct = train.eval_correct[routable_mask]

        # Get unique composite values to try as thresholds
        unique_scores = np.unique(composite_scores)

        # Grid search over all combinations
        best_acc = -1
        best_t1, best_t2 = 1.5, 2.5  # defaults

        for t1_val in unique_scores:
            for t2_val in unique_scores:
                if t2_val <= t1_val:
                    continue

                # Predict configs using these thresholds
                predictions = np.where(
                    composite_scores <= t1_val,
                    cheapest_a,
                    np.where(
                        composite_scores <= t2_val,
                        cheapest_b,
                        cheapest_c
                    )
                )

                # Calculate accuracy: fraction where predicted config is correct
                correct = eval_correct[np.arange(len(predictions)), predictions]
                acc = np.mean(correct)

                if acc > best_acc:
                    best_acc = acc
                    best_t1 = t1_val
                    best_t2 = t2_val

        self.t1 = best_t1
        self.t2 = best_t2

    def predict(self, dataset) -> np.ndarray:
        """Route based on composite complexity score and learned thresholds."""
        if self.t1 is None or self.t2 is None or self.tier_configs is None:
            raise RuntimeError("ComplexityThresholdRouter must be fitted before prediction")

        composite_scores = dataset.X[:, 3]  # composite_est at index 3

        predictions = np.where(
            composite_scores <= self.t1,
            self.tier_configs['A'],
            np.where(
                composite_scores <= self.t2,
                self.tier_configs['B'],
                self.tier_configs['C']
            )
        )

        return predictions.astype(np.int32)


class OracleCascadeRouter:
    """
    Oracle cascade: try configs in cost order, stop at first correct.

    This is the BEST POSSIBLE cascade — it has a perfect confidence checker.
    The cost is cumulative (every failed attempt adds cost).
    """

    def __init__(self):
        # Pre-sort config indices by cost
        self._sorted_configs = sorted(range(N_CONFIGS), key=lambda i: CONFIG_COSTS[i])

    @property
    def name(self) -> str:
        return "OracleCascade"

    def fit(self, train) -> None:
        pass  # no training needed

    def predict(self, dataset) -> np.ndarray:
        """
        Return the first correct config (in cost order) for each sample.
        For non-routable: tier-matched cheapest config.
        """
        n = len(dataset.sample_ids)
        complexity_tiers = getattr(dataset, "complexity_tiers", None)
        predictions = np.zeros(n, dtype=np.int32)
        for i in range(n):
            for cfg_idx in self._sorted_configs:
                if dataset.eval_correct[i, cfg_idx]:
                    predictions[i] = cfg_idx
                    break
            else:
                if complexity_tiers is not None:
                    tier_val = int(complexity_tiers[i])
                    model_tier = COMPLEXITY_TIER_TO_MODEL_TIER.get(tier_val, "A")
                    predictions[i] = TIER_CHEAPEST_CONFIG_IDX[model_tier]
                else:
                    predictions[i] = self._sorted_configs[0]
        return predictions

    def cascade_costs(self, dataset) -> np.ndarray:
        """
        Return the CUMULATIVE cascade cost per sample.
        This is different from the standard cost (which is just the selected config's cost).
        Cascade cost = sum of all configs tried up to and including first correct.
        """
        n = len(dataset.sample_ids)
        costs = np.zeros(n, dtype=np.float64)
        for i in range(n):
            cumulative = 0.0
            for cfg_idx in self._sorted_configs:
                cumulative += CONFIG_COSTS[cfg_idx]
                if dataset.eval_correct[i, cfg_idx]:
                    break
            costs[i] = cumulative
        return costs


class LogisticRouter:
    """
    Multinomial logistic regression predicting cheapest-correct config.

    Trains on routable samples only, using StandardScaler for feature normalization.
    """

    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self.classifier: Optional[LogisticRegression] = None

    @property
    def name(self) -> str:
        return "LogisticRouter"

    def fit(self, train) -> None:
        """
        Train logistic regression on routable samples.
        """
        # Filter to routable samples only
        routable_mask = train.is_routable
        if not np.any(routable_mask):
            raise ValueError("No routable samples in training data")

        X_train = train.X[routable_mask]
        y_train = train.y_config[routable_mask]

        # Fit scaler and transform features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # Train multinomial logistic regression
        # Note: sklearn 1.8.0 handles multinomial automatically with lbfgs solver
        self.classifier = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        )
        self.classifier.fit(X_scaled, y_train)

    def predict(self, dataset) -> np.ndarray:
        """Predict most likely config for all samples."""
        if self.scaler is None or self.classifier is None:
            raise RuntimeError("LogisticRouter must be fitted before prediction")

        X_scaled = self.scaler.transform(dataset.X)
        predictions = self.classifier.predict(X_scaled)

        return predictions.astype(np.int32)


class LearnedCascadeRouter:
    """
    FrugalGPT-style learned cascade with tier escalation.

    Two logistic regression gates predict whether Tier A or Tier B configs
    can solve a query. At inference: try Tier A first, escalate to B if not
    confident, fall through to C. Gate thresholds tuned on training data.

    Routes to cheapest config within the selected tier — tests whether
    sequential tier escalation outperforms flat classification.
    """

    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self.gate_a: Optional[LogisticRegression] = None
        self.gate_b: Optional[LogisticRegression] = None
        self.threshold_a: float = 0.5
        self.threshold_b: float = 0.5
        self.cheapest_a: int = 0
        self.cheapest_b: int = 0
        self.cheapest_c: int = 0

    @property
    def name(self) -> str:
        return "LearnedCascade"

    def _find_cheapest_tier_configs(self):
        for tier_label, attr in [('A', 'cheapest_a'), ('B', 'cheapest_b'), ('C', 'cheapest_c')]:
            tier_configs = [
                (i, CONFIG_COSTS[i]) for i, (model, _) in enumerate(CONFIG_LIST)
                if MODELS[model].tier == tier_label
            ]
            setattr(self, attr, min(tier_configs, key=lambda x: x[1])[0])

    def _compute_tier_labels(self, dataset) -> tuple[np.ndarray, np.ndarray]:
        n = len(dataset.sample_ids)
        tier_a_correct = np.zeros(n, dtype=bool)
        tier_b_correct = np.zeros(n, dtype=bool)

        for i in range(n):
            for cfg_idx in range(N_CONFIGS):
                if dataset.eval_correct[i, cfg_idx]:
                    tier = MODELS[CONFIG_LIST[cfg_idx][0]].tier
                    if tier == 'A':
                        tier_a_correct[i] = True
                    elif tier == 'B':
                        tier_b_correct[i] = True

        return tier_a_correct, tier_b_correct

    def fit(self, train) -> None:
        self._find_cheapest_tier_configs()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(train.X)

        tier_a_correct, tier_b_correct = self._compute_tier_labels(train)

        self.gate_a = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
        self.gate_a.fit(X_scaled, tier_a_correct.astype(int))

        self.gate_b = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
        self.gate_b.fit(X_scaled, tier_b_correct.astype(int))

        self._tune_thresholds(X_scaled, train)

    def _tune_thresholds(self, X_scaled: np.ndarray, dataset) -> None:
        prob_a = self.gate_a.predict_proba(X_scaled)[:, 1]
        prob_b = self.gate_b.predict_proba(X_scaled)[:, 1]

        best_acc = -1.0
        best_ta, best_tb = 0.5, 0.5

        for ta in np.arange(0.1, 0.9, 0.05):
            for tb in np.arange(0.1, 0.9, 0.05):
                preds = np.where(
                    prob_a >= ta, self.cheapest_a,
                    np.where(prob_b >= tb, self.cheapest_b, self.cheapest_c)
                )
                correct = dataset.eval_correct[np.arange(len(preds)), preds]
                acc = correct.mean()
                if acc > best_acc:
                    best_acc = acc
                    best_ta, best_tb = ta, tb

        self.threshold_a = best_ta
        self.threshold_b = best_tb

    def predict(self, dataset) -> np.ndarray:
        if self.scaler is None or self.gate_a is None:
            raise RuntimeError("LearnedCascadeRouter must be fitted before prediction")

        X_scaled = self.scaler.transform(dataset.X)
        prob_a = self.gate_a.predict_proba(X_scaled)[:, 1]
        prob_b = self.gate_b.predict_proba(X_scaled)[:, 1]

        preds = np.where(
            prob_a >= self.threshold_a, self.cheapest_a,
            np.where(prob_b >= self.threshold_b, self.cheapest_b, self.cheapest_c)
        )
        return preds.astype(np.int32)


class CostSensitiveMLP:
    """
    Flat 24-class MLP with cost-weighted cross-entropy loss.

    Unlike PERCEIVE's decomposed model+budget architecture, this predicts
    config indices directly. The loss upweights cheap configs so the router
    prefers cost-efficient choices even when expensive configs are also correct.
    """

    N_TEXT_FEATURES = 48

    def __init__(self, n_features: int, seed: int = 42, clip_mask_prob: float = 0.0):
        self.n_features = n_features
        self.seed = seed
        self.clip_mask_prob = clip_mask_prob
        self.scaler: Optional[StandardScaler] = None
        self._model = None
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "CostSensitiveMLP"

    def fit(self, train) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(train.X)

        self._model = nn.Sequential(
            nn.Linear(self.n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, N_CONFIGS),
        )

        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(train.y_config, dtype=torch.long)
        sw_t = torch.tensor(train.sample_weights, dtype=torch.float32)

        max_cost = max(CONFIG_COSTS)
        class_weights = torch.tensor(
            [max_cost / (CONFIG_COSTS[i] + 1e-10) for i in range(N_CONFIGS)],
            dtype=torch.float32,
        )
        class_weights = class_weights / class_weights.mean()

        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        loader = DataLoader(TensorDataset(X_t, y_t, sw_t), batch_size=128, shuffle=True)

        self._model.train()
        for _epoch in range(100):
            for X_batch, y_batch, sw_batch in loader:
                if self.clip_mask_prob > 0 and X_batch.shape[1] > self.N_TEXT_FEATURES:
                    mask = (torch.rand(X_batch.shape[0], 1) > self.clip_mask_prob).float()
                    X_batch = X_batch.clone()
                    X_batch[:, self.N_TEXT_FEATURES:] *= mask

                logits = self._model(X_batch)
                loss = (criterion(logits, y_batch) * sw_batch).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        self._is_fitted = True

    def predict(self, dataset) -> np.ndarray:
        import torch

        if not self._is_fitted:
            raise RuntimeError("CostSensitiveMLP must be fitted before prediction")

        self._model.eval()
        X_scaled = self.scaler.transform(dataset.X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = self._model(X_t)
            predictions = torch.argmax(logits, dim=1).numpy()
        return predictions.astype(np.int32)
