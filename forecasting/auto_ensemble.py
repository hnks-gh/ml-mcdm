# -*- coding: utf-8 -*-
"""
Bayesian Optimization-based Automatic Ensemble Selection
=========================================================

Automatically searches for the optimal subset of base models and
their combination weights using Bayesian Optimization. Instead of
using all models (which can introduce redundancy and overfitting),
this module intelligently selects the best 2-5 models and their
optimal weights.

Algorithm:
    1. Define search space: binary model selectors + continuous weights
    2. Objective: maximize time-series CV R² score
    3. Use Tree-structured Parzen Estimator (TPE) for efficient search
    4. Constraint: select 2-6 models to prevent overfitting
    5. Run 50-100 optimization trials

Key Features:
    - Sparse ensembles: Often finds 2-3 models are sufficient
    - Prevents redundancy: Removes correlated models
    - Better generalization: Fewer models reduce overfitting
    - Principled: Bayesian optimization is sample-efficient

Implementation uses scipy.optimize when Optuna is not available,
providing a gradient-free optimization fallback.

References:
    - Caruana et al. (2004). "Ensemble Selection from Libraries of
      Models" ICML
    - Feurer et al. (2015). "Efficient and Robust Automated Machine
      Learning" NeurIPS (Auto-sklearn)
    - Akiba et al. (2019). "Optuna: A Next-generation Hyperparameter
      Optimization Framework" KDD
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import copy
import warnings

from .base import BaseForecaster

warnings.filterwarnings("ignore")


class AutoEnsembleSelector:
    """
    Automatic ensemble model selection via Bayesian Optimization.

    Searches over:
    1. Which models to include (binary selection)
    2. What weights to assign (continuous optimization)
    3. What regularization to use (meta-learner hyperparameters)

    Uses either Optuna (if available) or scipy differential evolution
    for the optimization backend.

    Parameters:
        base_models: Dictionary of {name: BaseForecaster} instances
        n_trials: Number of optimization trials (default 50)
        cv_folds: Number of temporal CV folds
        min_models: Minimum models in ensemble (default 2)
        max_models: Maximum models in ensemble (default 6)
        metric: Evaluation metric ('r2', 'neg_mse', 'neg_mae')
        n_random_starts: Number of random initial trials
        random_state: Random seed
        verbose: Print progress

    Example:
        >>> selector = AutoEnsembleSelector(base_models=models, n_trials=50)
        >>> selector.fit(X_train, y_train)
        >>> best_pred = selector.predict(X_test)
    """

    def __init__(
        self,
        base_models: Dict[str, BaseForecaster],
        n_trials: int = 50,
        cv_folds: int = 5,
        min_models: int = 2,
        max_models: int = 6,
        metric: str = "r2",
        n_random_starts: int = 10,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.base_models = base_models
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.min_models = min_models
        self.max_models = max_models
        self.metric = metric
        self.n_random_starts = n_random_starts
        self.random_state = random_state
        self.verbose = verbose

        # Results
        self._best_model_selection: Dict[str, bool] = {}
        self._best_weights: Dict[str, float] = {}
        self._best_score: float = -np.inf
        self._fitted_models: Dict[str, BaseForecaster] = {}
        self._trial_history: List[Dict[str, Any]] = []
        self._fitted: bool = False
        self._n_outputs: int = 1

        # Precomputed CV predictions for efficiency
        self._cv_predictions: Optional[Dict[str, np.ndarray]] = None

    def _precompute_cv_predictions(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Pre-compute cross-validated predictions for all base models.

        This avoids re-training models during each optimization trial,
        reducing computational cost from O(n_trials × n_models × n_folds)
        to O(n_models × n_folds + n_trials).

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Dictionary mapping model names to OOF prediction arrays
        """
        n_samples = X.shape[0]
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        cv_preds = {}

        for name, model in self.base_models.items():
            oof = np.full((n_samples, self._n_outputs), np.nan)

            for train_idx, val_idx in tscv.split(X):
                try:
                    model_copy = copy.deepcopy(model)
                    model_copy.fit(X[train_idx], y[train_idx])
                    pred = model_copy.predict(X[val_idx])
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, 1)
                    for c in range(min(pred.shape[1], self._n_outputs)):
                        oof[val_idx, c] = pred[:, c]
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: {name} CV failed: {e}")

            cv_preds[name] = oof

        return cv_preds

    def _evaluate_ensemble(
        self,
        selection: Dict[str, bool],
        weights: Dict[str, float],
        cv_predictions: Dict[str, np.ndarray],
        y: np.ndarray,
    ) -> float:
        """
        Evaluate an ensemble configuration using precomputed CV predictions.

        Args:
            selection: Which models to include
            weights: Model weights
            cv_predictions: Precomputed OOF predictions
            y: True target values

        Returns:
            Evaluation metric score
        """
        selected = [name for name, include in selection.items() if include]
        if len(selected) == 0:
            return -np.inf

        # Normalize weights for selected models
        selected_weights = {name: weights.get(name, 1.0) for name in selected}
        weight_sum = sum(selected_weights.values())
        if weight_sum <= 0:
            return -np.inf
        selected_weights = {n: w / weight_sum for n, w in selected_weights.items()}

        n_samples = y.shape[0]
        ensemble_pred = np.zeros((n_samples, self._n_outputs))

        for name in selected:
            pred = cv_predictions[name]
            w = selected_weights[name]
            valid = ~np.isnan(pred).any(axis=1)
            ensemble_pred[valid] += w * pred[valid]

        # Only evaluate on rows where all selected models have predictions
        valid_mask = np.ones(n_samples, dtype=bool)
        for name in selected:
            valid_mask &= ~np.isnan(cv_predictions[name]).any(axis=1)

        if valid_mask.sum() < 5:
            return -np.inf

        # Compute metric
        scores = []
        for c in range(self._n_outputs):
            if self.metric == "r2":
                score = r2_score(y[valid_mask, c], ensemble_pred[valid_mask, c])
            elif self.metric == "neg_mse":
                score = -np.mean((y[valid_mask, c] - ensemble_pred[valid_mask, c]) ** 2)
            elif self.metric == "neg_mae":
                score = -np.mean(np.abs(y[valid_mask, c] - ensemble_pred[valid_mask, c]))
            else:
                score = r2_score(y[valid_mask, c], ensemble_pred[valid_mask, c])
            scores.append(score)

        return np.mean(scores)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AutoEnsembleSelector":
        """
        Run Bayesian Optimization to find the best ensemble.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Self for method chaining
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._n_outputs = y.shape[1]

        if self.verbose:
            print(f"  AutoEnsemble: Precomputing CV predictions for {len(self.base_models)} models...")

        # Pre-compute CV predictions
        self._cv_predictions = self._precompute_cv_predictions(X, y)

        if self.verbose:
            print(f"  AutoEnsemble: Running {self.n_trials} optimization trials...")

        # Try Optuna first, fall back to random search + refinement
        try:
            self._optimize_with_optuna(y)
        except ImportError:
            self._optimize_random_search(y)

        if self.verbose:
            print(f"  AutoEnsemble: Best score = {self._best_score:.4f}")
            print(f"  Selected models:")
            for name, w in sorted(
                self._best_weights.items(), key=lambda x: x[1], reverse=True
            ):
                if self._best_model_selection.get(name, False):
                    bar = "█" * int(w * 30)
                    print(f"    {name:25s}: {w:.4f} {bar}")

        # Fit selected models on full training data
        for name, include in self._best_model_selection.items():
            if include:
                try:
                    model = copy.deepcopy(self.base_models[name])
                    model.fit(X, y)
                    self._fitted_models[name] = model
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: {name} final fit failed: {e}")

        self._fitted = True
        return self

    def _optimize_with_optuna(self, y: np.ndarray):
        """Run optimization using Optuna's TPE sampler."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        model_names = list(self.base_models.keys())

        def objective(trial):
            # Binary model selection
            selection = {}
            for name in model_names:
                selection[name] = trial.suggest_int(f"use_{name}", 0, 1) == 1

            # Check model count constraint
            n_selected = sum(selection.values())
            if n_selected < self.min_models or n_selected > self.max_models:
                return -np.inf

            # Continuous weights for selected models
            weights = {}
            for name in model_names:
                if selection[name]:
                    weights[name] = trial.suggest_float(f"w_{name}", 0.01, 1.0)
                else:
                    weights[name] = 0.0

            score = self._evaluate_ensemble(
                selection, weights, self._cv_predictions, y
            )
            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=self.random_state,
                n_startup_trials=self.n_random_starts,
            ),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        # Extract best configuration
        best_trial = study.best_trial
        self._best_score = best_trial.value

        model_names = list(self.base_models.keys())
        self._best_model_selection = {
            name: best_trial.params.get(f"use_{name}", 0) == 1
            for name in model_names
        }

        raw_weights = {
            name: best_trial.params.get(f"w_{name}", 0.0)
            for name in model_names
            if self._best_model_selection.get(name, False)
        }
        w_sum = sum(raw_weights.values())
        self._best_weights = (
            {n: w / w_sum for n, w in raw_weights.items()}
            if w_sum > 0
            else {n: 1.0 / len(raw_weights) for n in raw_weights}
        )

        # Store trial history
        for trial in study.trials:
            self._trial_history.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
            })

    def _optimize_random_search(self, y: np.ndarray):
        """Fallback: random search + greedy refinement."""
        rng = np.random.RandomState(self.random_state)
        model_names = list(self.base_models.keys())
        n_models = len(model_names)

        best_score = -np.inf
        best_selection = {}
        best_weights = {}

        for trial in range(self.n_trials):
            # Random model selection
            n_select = rng.randint(self.min_models, min(self.max_models + 1, n_models + 1))
            selected_idx = rng.choice(n_models, size=n_select, replace=False)

            selection = {name: False for name in model_names}
            for idx in selected_idx:
                selection[model_names[idx]] = True

            # Random weights
            raw_weights = rng.dirichlet(np.ones(n_select))
            weights = {name: 0.0 for name in model_names}
            for i, idx in enumerate(selected_idx):
                weights[model_names[idx]] = raw_weights[i]

            score = self._evaluate_ensemble(
                selection, weights, self._cv_predictions, y
            )

            self._trial_history.append({
                "number": trial,
                "value": score,
                "selection": {k: v for k, v in selection.items() if v},
            })

            if score > best_score:
                best_score = score
                best_selection = selection.copy()
                best_weights = weights.copy()

        # Greedy weight refinement on best selection
        selected_names = [n for n, v in best_selection.items() if v]
        if len(selected_names) >= 2:
            # Grid search over weight combinations
            n_sel = len(selected_names)
            best_refined_score = best_score
            best_refined_weights = best_weights.copy()

            for _ in range(100):
                raw_w = rng.dirichlet(np.ones(n_sel) * 2.0)
                test_weights = {name: 0.0 for name in model_names}
                for i, name in enumerate(selected_names):
                    test_weights[name] = raw_w[i]

                s = self._evaluate_ensemble(
                    best_selection, test_weights, self._cv_predictions, y
                )
                if s > best_refined_score:
                    best_refined_score = s
                    best_refined_weights = test_weights.copy()

            best_score = best_refined_score
            best_weights = best_refined_weights

        self._best_score = best_score
        self._best_model_selection = best_selection

        # Normalize final weights
        selected_weights = {
            n: w for n, w in best_weights.items()
            if best_selection.get(n, False) and w > 0
        }
        w_sum = sum(selected_weights.values())
        self._best_weights = (
            {n: w / w_sum for n, w in selected_weights.items()}
            if w_sum > 0
            else {n: 1.0 / max(1, len(selected_weights)) for n in selected_weights}
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the optimized ensemble.

        Args:
            X: Feature matrix

        Returns:
            Weighted ensemble predictions
        """
        if not self._fitted:
            raise ValueError("Not fitted. Call fit() first.")

        n_samples = X.shape[0]
        result = np.zeros((n_samples, self._n_outputs))

        for name, model in self._fitted_models.items():
            weight = self._best_weights.get(name, 0.0)
            if weight <= 0:
                continue
            try:
                pred = model.predict(X)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                for c in range(min(pred.shape[1], self._n_outputs)):
                    result[:, c] += weight * pred[:, c]
            except Exception:
                continue

        if self._n_outputs == 1:
            return result.ravel()
        return result

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty from model disagreement.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (mean prediction, standard deviation)
        """
        all_preds = []
        all_weights = []

        for name, model in self._fitted_models.items():
            weight = self._best_weights.get(name, 0.0)
            if weight <= 0:
                continue
            try:
                pred = model.predict(X)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                all_preds.append(pred[:, :self._n_outputs])
                all_weights.append(weight)
            except Exception:
                continue

        if not all_preds:
            raise ValueError("No models produced predictions")

        weights = np.array(all_weights)
        weights /= weights.sum()

        pred_stack = np.stack(all_preds, axis=0)
        mean_pred = np.average(pred_stack, weights=weights, axis=0)
        diff = pred_stack - mean_pred[np.newaxis, :, :]
        weighted_var = np.average(diff ** 2, weights=weights, axis=0)

        if self._n_outputs == 1:
            return mean_pred.ravel(), np.sqrt(weighted_var).ravel()
        return mean_pred, np.sqrt(weighted_var)

    def get_selected_models(self) -> List[str]:
        """Get names of selected models."""
        return [n for n, v in self._best_model_selection.items() if v]

    def get_best_weights(self) -> Dict[str, float]:
        """Get optimized weights for selected models."""
        return self._best_weights.copy()

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get full trial history."""
        return self._trial_history

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get optimization diagnostics."""
        scores = [t["value"] for t in self._trial_history if t["value"] != -np.inf]
        return {
            "best_score": self._best_score,
            "n_trials": len(self._trial_history),
            "n_valid_trials": len(scores),
            "selected_models": self.get_selected_models(),
            "weights": self._best_weights,
            "score_progression": {
                "first_10_mean": np.mean(scores[:10]) if len(scores) >= 10 else np.nan,
                "last_10_mean": np.mean(scores[-10:]) if len(scores) >= 10 else np.nan,
                "improvement": (
                    (np.mean(scores[-10:]) - np.mean(scores[:10])) / abs(np.mean(scores[:10]) + 1e-10)
                    if len(scores) >= 20
                    else np.nan
                ),
            },
            "min_models": self.min_models,
            "max_models": self.max_models,
            "metric": self.metric,
        }
