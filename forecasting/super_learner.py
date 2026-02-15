# -*- coding: utf-8 -*-
"""
Super Learner (Stacked Generalization) Meta-Ensemble
====================================================

Implements the Super Learner algorithm (van der Laan et al., 2007),
a principled approach to ensemble learning that uses a meta-learner
trained on out-of-fold predictions to optimally combine base models.

Architecture:
    Base Models → Out-of-Fold Predictions → Meta-Learner → Final Prediction
         ├─ Gradient Boosting  (ŷ₁)           ↓
         ├─ Random Forest      (ŷ₂)      ElasticNet / Ridge
         ├─ Bayesian Ridge     (ŷ₃)      (learns α₁...αₙ)
         ├─ Panel VAR          (ŷ₄)           ↓
         ├─ Hier. Bayes        (ŷ₅)      ŷ_final = Σ αᵢŷᵢ
         └─ NAM                (ŷ₆)

Key Properties:
    1. Oracle inequality: Super Learner performs asymptotically
       as well as the best weighted combination of base models
    2. Cross-validated meta-features prevent information leakage
    3. Non-negative weights ensure interpretability
    4. Temporal ordering preserved via TimeSeriesSplit

Variants:
    - Standard: ElasticNet meta-learner with positive weights
    - Bayesian Stacking: Dirichlet-weighted meta-learner for uncertainty
    - Dynamic: Time-varying weights via exponential weighting

References:
    - van der Laan, Polley & Hubbard (2007). "Super Learner"
      Statistical Applications in Genetics and Molecular Biology
    - Naimi & Balzer (2018). "Stacked Generalization: An Introduction
      to Super Learning" European Journal of Epidemiology
    - Yao et al. (2018). "Using Stacking to Average Bayesian Predictive
      Distributions" Bayesian Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNetCV, RidgeCV, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import copy
import warnings

from .base import BaseForecaster

warnings.filterwarnings("ignore")


class SuperLearner:
    """
    Super Learner meta-ensemble combining multiple base forecasters.

    The algorithm:
    1. Generate out-of-fold (OOF) predictions using temporal CV
    2. Train a meta-learner on OOF predictions as features
    3. Re-train all base models on full training data
    4. Combine base model predictions using meta-learner weights

    Parameters:
        base_models: Dictionary of {name: BaseForecaster} instances
        meta_learner_type: Type of meta-learner ('elasticnet', 'ridge',
                          'bayesian_stacking')
        n_cv_folds: Number of CV folds for OOF predictions
        positive_weights: If True, constrain meta-weights to be non-negative
        normalize_weights: If True, meta-weights sum to 1
        meta_alpha_range: Range of regularization values for meta-learner CV
        temperature: Temperature for Bayesian stacking softmax
        random_state: Random seed
        verbose: Print progress messages

    Example:
        >>> from forecasting.tree_ensemble import GradientBoostingForecaster
        >>> from forecasting.linear import BayesianForecaster
        >>>
        >>> base = {
        ...     'gb': GradientBoostingForecaster(),
        ...     'bayesian': BayesianForecaster(),
        ... }
        >>> sl = SuperLearner(base_models=base)
        >>> sl.fit(X_train, y_train)
        >>> predictions = sl.predict(X_test)
    """

    def __init__(
        self,
        base_models: Dict[str, BaseForecaster],
        meta_learner_type: str = "ridge",
        n_cv_folds: int = 5,
        positive_weights: bool = True,
        normalize_weights: bool = True,
        meta_alpha_range: Optional[List[float]] = None,
        temperature: float = 5.0,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.base_models = base_models
        self.meta_learner_type = meta_learner_type
        self.n_cv_folds = n_cv_folds
        self.positive_weights = positive_weights
        self.normalize_weights = normalize_weights
        self.meta_alpha_range = meta_alpha_range or [
            0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0
        ]
        self.temperature = temperature
        self.random_state = random_state
        self.verbose = verbose

        # Fitted components
        self._fitted_base_models: Dict[str, BaseForecaster] = {}
        self._meta_learner = None
        self._meta_weights: Dict[str, float] = {}
        self._cv_scores: Dict[str, List[float]] = {}
        self._fitted: bool = False
        self._n_outputs: int = 1
        self._oof_r2: Dict[str, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SuperLearner":
        """
        Fit the Super Learner ensemble.

        Stage 1: Generate out-of-fold predictions via temporal CV
        Stage 2: Train meta-learner on OOF predictions
        Stage 3: Re-train base models on full training data

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) or (n_samples, n_outputs)

        Returns:
            Self for method chaining
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._n_outputs = y.shape[1]
        n_samples = X.shape[0]
        n_models = len(self.base_models)

        if self.verbose:
            print(f"  Super Learner: {n_models} base models, {self.n_cv_folds} CV folds")

        # ============================================================
        # Stage 1: Generate out-of-fold predictions
        # ============================================================
        tscv = TimeSeriesSplit(n_splits=self.n_cv_folds)

        # OOF prediction storage: (n_samples, n_models * n_outputs)
        oof_predictions = np.full(
            (n_samples, n_models * self._n_outputs), np.nan
        )
        self._cv_scores = {name: [] for name in self.base_models}

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

            for m_idx, (name, model) in enumerate(self.base_models.items()):
                try:
                    model_copy = copy.deepcopy(model)
                    model_copy.fit(X_train_cv, y_train_cv)

                    pred = model_copy.predict(X_val_cv)
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, 1)

                    # Store OOF predictions
                    for out_col in range(self._n_outputs):
                        col_idx = m_idx * self._n_outputs + out_col
                        pred_col = min(out_col, pred.shape[1] - 1)
                        oof_predictions[val_idx, col_idx] = pred[:, pred_col]

                    # Compute CV score
                    for out_col in range(y_val_cv.shape[1]):
                        pred_col = min(out_col, pred.shape[1] - 1)
                        r2 = r2_score(y_val_cv[:, out_col], pred[:, pred_col])
                        self._cv_scores[name].append(r2)

                except Exception as e:
                    if self.verbose:
                        print(f"    Warning: {name} failed on fold {fold_idx}: {e}")
                    self._cv_scores[name].append(-1.0)

        # Compute OOF R² for each model
        valid_mask = ~np.isnan(oof_predictions).any(axis=1)
        for m_idx, name in enumerate(self.base_models):
            model_cols = slice(
                m_idx * self._n_outputs, (m_idx + 1) * self._n_outputs
            )
            valid = valid_mask & ~np.isnan(oof_predictions[:, model_cols]).any(axis=1)
            if valid.sum() > 0:
                r2_vals = []
                for out_col in range(self._n_outputs):
                    col_idx = m_idx * self._n_outputs + out_col
                    r2 = r2_score(y[valid, out_col], oof_predictions[valid, col_idx])
                    r2_vals.append(r2)
                self._oof_r2[name] = np.mean(r2_vals)
            else:
                self._oof_r2[name] = -1.0

        # ============================================================
        # Stage 2: Train meta-learner on OOF predictions
        # ============================================================
        # Use only rows with valid OOF predictions
        valid_rows = valid_mask
        if valid_rows.sum() < 5:
            # Fallback: use simple averaging if not enough OOF data
            if self.verbose:
                print("  Warning: Not enough OOF data, falling back to weighted avg")
            self._meta_weights = self._fallback_weights()
        else:
            oof_X = oof_predictions[valid_rows]
            oof_y = y[valid_rows]

            self._fit_meta_learner(oof_X, oof_y)

        # ============================================================
        # Stage 3: Re-train all base models on full data
        # ============================================================
        for name, model in self.base_models.items():
            try:
                fitted_model = copy.deepcopy(model)
                fitted_model.fit(X, y)
                self._fitted_base_models[name] = fitted_model
            except Exception as e:
                if self.verbose:
                    print(f"    Warning: {name} failed on full data: {e}")

        self._fitted = True

        if self.verbose:
            print("  Super Learner meta-weights:")
            for name, w in sorted(
                self._meta_weights.items(), key=lambda x: x[1], reverse=True
            ):
                bar = "█" * int(w * 30)
                print(f"    {name:25s}: {w:.4f} {bar}")

        return self

    def _fit_meta_learner(self, oof_X: np.ndarray, oof_y: np.ndarray):
        """Fit the second-level meta-learner on OOF predictions."""
        # Fit per output column, average weights
        all_coefs = []

        for out_col in range(self._n_outputs):
            # Extract model predictions for this output
            model_preds = np.column_stack([
                oof_X[:, m_idx * self._n_outputs + out_col]
                for m_idx in range(len(self.base_models))
            ])

            y_col = oof_y[:, out_col]

            # Handle NaN values
            valid = ~np.isnan(model_preds).any(axis=1) & ~np.isnan(y_col)
            if valid.sum() < 3:
                all_coefs.append(np.ones(len(self.base_models)) / len(self.base_models))
                continue

            model_preds_valid = model_preds[valid]
            y_valid = y_col[valid]

            if self.meta_learner_type == "elasticnet":
                meta = ElasticNetCV(
                    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                    alphas=self.meta_alpha_range,
                    positive=self.positive_weights,
                    cv=min(3, valid.sum() // 2),
                    max_iter=5000,
                    random_state=self.random_state,
                )
            elif self.meta_learner_type == "bayesian_stacking":
                # Bayesian stacking: softmax-weighted by OOF R²
                scores = np.array([
                    np.mean(self._cv_scores[name])
                    for name in self.base_models
                ])
                scores = np.clip(scores, 0, None)
                exp_scores = np.exp(self.temperature * scores)
                weights = exp_scores / exp_scores.sum()
                all_coefs.append(weights)
                continue
            else:  # ridge
                meta = RidgeCV(
                    alphas=self.meta_alpha_range,
                    cv=min(3, valid.sum() // 2),
                )

            try:
                meta.fit(model_preds_valid, y_valid)
                coefs = meta.coef_.copy()

                # Enforce non-negative weights if required
                if self.positive_weights:
                    coefs = np.maximum(coefs, 0)

                all_coefs.append(coefs)

                if out_col == 0:
                    self._meta_learner = meta
            except Exception:
                all_coefs.append(
                    np.ones(len(self.base_models)) / len(self.base_models)
                )

        # Average coefficients across outputs
        avg_coefs = np.mean(all_coefs, axis=0)

        # Normalize to sum to 1
        if self.normalize_weights:
            coef_sum = avg_coefs.sum()
            if coef_sum > 0:
                avg_coefs /= coef_sum
            else:
                avg_coefs = np.ones(len(self.base_models)) / len(self.base_models)

        self._meta_weights = dict(zip(self.base_models.keys(), avg_coefs))

    def _fallback_weights(self) -> Dict[str, float]:
        """Compute fallback weights from CV scores when meta-learner fails."""
        scores = {
            name: max(0, np.mean(s)) for name, s in self._cv_scores.items()
        }
        total = sum(scores.values())
        if total > 0:
            return {name: s / total for name, s in scores.items()}
        return {name: 1.0 / len(scores) for name in scores}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the Super Learner ensemble.

        Combines base model predictions using learned meta-weights.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_outputs)
        """
        if not self._fitted:
            raise ValueError("Super Learner not fitted. Call fit() first.")

        all_predictions = []
        model_names = []

        for name, model in self._fitted_base_models.items():
            try:
                pred = model.predict(X)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                all_predictions.append(pred)
                model_names.append(name)
            except Exception:
                continue

        if not all_predictions:
            raise ValueError("No base models produced predictions")

        # Weighted combination
        n_samples = X.shape[0]
        result = np.zeros((n_samples, self._n_outputs))

        for pred, name in zip(all_predictions, model_names):
            weight = self._meta_weights.get(name, 0.0)
            for out_col in range(self._n_outputs):
                pred_col = min(out_col, pred.shape[1] - 1)
                result[:, out_col] += weight * pred[:, pred_col]

        if self._n_outputs == 1:
            return result.ravel()
        return result

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty from model disagreement.

        Uncertainty = weighted standard deviation of base model predictions.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (mean prediction, standard deviation)
        """
        all_predictions = []
        all_weights = []

        for name, model in self._fitted_base_models.items():
            try:
                pred = model.predict(X)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                all_predictions.append(pred)
                all_weights.append(self._meta_weights.get(name, 0.0))
            except Exception:
                continue

        if not all_predictions:
            raise ValueError("No base models produced predictions")

        # Weighted mean
        weights = np.array(all_weights)
        weights /= weights.sum() + 1e-10

        pred_stack = np.stack(
            [p[:, :self._n_outputs] if p.shape[1] >= self._n_outputs
             else np.column_stack([p] * self._n_outputs)
             for p in all_predictions],
            axis=0,
        )

        mean_pred = np.average(pred_stack, weights=weights, axis=0)

        # Weighted standard deviation (model disagreement)
        diff = pred_stack - mean_pred[np.newaxis, :, :]
        weighted_var = np.average(diff ** 2, weights=weights, axis=0)
        std_pred = np.sqrt(weighted_var)

        if self._n_outputs == 1:
            return mean_pred.ravel(), std_pred.ravel()
        return mean_pred, std_pred

    def get_meta_weights(self) -> Dict[str, float]:
        """Get the learned meta-learner weights."""
        return self._meta_weights.copy()

    def get_cv_scores(self) -> Dict[str, List[float]]:
        """Get cross-validation R² scores for each base model."""
        return self._cv_scores.copy()

    def get_oof_performance(self) -> Dict[str, float]:
        """Get out-of-fold R² for each base model."""
        return self._oof_r2.copy()

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics."""
        return {
            "meta_learner_type": self.meta_learner_type,
            "n_cv_folds": self.n_cv_folds,
            "n_base_models": len(self.base_models),
            "meta_weights": self._meta_weights,
            "oof_r2": self._oof_r2,
            "mean_cv_scores": {
                name: np.mean(scores)
                for name, scores in self._cv_scores.items()
            },
            "positive_weights": self.positive_weights,
        }
