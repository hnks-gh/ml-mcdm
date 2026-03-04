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
         └─ NAM                (ŷ₅)      ŷ_final = Σ αᵢŷᵢ

Key Properties:
    1. Oracle inequality: Super Learner performs asymptotically
       as well as the best weighted combination of base models
    2. Cross-validated meta-features prevent information leakage
    3. Non-negative weights (NNLS) ensure interpretability and
       per-entity weight clamping prevents boundary artefacts
    4. Panel-aware temporal CV (``_PanelTemporalSplit``) uses the
       *median* entity length to size fold windows, preserving data
       from longer-history entities that the old min-based splitter
       discarded (Bug S-2 fix)
    5. OOF predictions are cached and re-used for conformal calibration
       — the SuperLearner ensemble is never deep-copied (Bug U-2 fix)

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
from scipy.optimize import nnls
import copy
import warnings
import functools

from .base import BaseForecaster


def _silence_warnings(func):
    """Scope all warning filters to the duration of *func* only."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return func(*args, **kwargs)
    return wrapper


class _PanelTemporalSplit:
    """
    Panel-aware temporal cross-validation splitter.

    Unlike ``TimeSeriesSplit`` (which splits on the absolute row position in
    the stacked panel), this splitter identifies the temporal position of
    each row *within its entity* and produces folds that respect temporal
    order for every entity simultaneously.

    For each fold k:
      * **train**: rows whose within-entity time position is in ``[0, cut_k)``
      * **val**  : rows whose within-entity time position is in
                   ``[cut_k, cut_{k+1})``

    This guarantees that no entity's future leaks into training data and
    that every entity contributes both train and validation rows.

    Parameters
    ----------
    n_splits : int
        Number of CV folds (≥ 2).
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(
        self,
        X: np.ndarray,
        entity_indices: Optional[np.ndarray] = None,
    ):
        """
        Yield ``(train_idx, val_idx)`` pairs.

        Parameters
        ----------
        X : ndarray  (n_samples, n_features)
        entity_indices : ndarray of shape (n_samples,), optional
            Entity/group ID for each row.  When *None* the splitter falls
            back to a standard ``TimeSeriesSplit``.
        """
        if entity_indices is None:
            yield from TimeSeriesSplit(n_splits=self.n_splits).split(X)
            return

        unique_entities = np.unique(entity_indices)
        # Rows for each entity in their original (assumed temporal) order
        entity_rows: Dict[Any, np.ndarray] = {
            ent: np.where(entity_indices == ent)[0]
            for ent in unique_entities
        }

        # S-2 FIX: Use the MEDIAN entity length to size the fold windows rather
        # than the MIN.  The old code used T = min(...) — any entity shorter
        # than the fold window would exclude ALL data from longer-history
        # entities, wasting up to 9 years of panel data per province.
        #
        # New behaviour:
        #   * Fold boundaries are derived from the median entity length.
        #   * For each fold, each entity contributes its OWN rows up to its
        #     actual T — longer entities provide full training + validation;
        #     shorter entities that fall below the validation window simply
        #     contribute 0 validation rows for that fold (acceptable).
        T_per_entity = {ent: len(rows) for ent, rows in entity_rows.items()}
        T_median = max(2, int(np.median(list(T_per_entity.values()))))

        if T_median < 2:
            yield from TimeSeriesSplit(n_splits=self.n_splits).split(X)
            return

        # Reserve at least half the timeline as the initial training window
        # so that even the first fold has enough data relative to the feature
        # dimensionality.  The old formula ``max(1, T // (K+1))`` gave only
        # 3 years for T=14, K=3 — far too small for 400+ features.
        min_train_T = max(T_median // 2, T_median // (self.n_splits + 1))
        fold_size = max(1, (T_median - min_train_T) // self.n_splits)

        for fold in range(self.n_splits):
            cut = min_train_T + fold * fold_size
            val_end = cut + fold_size
            if cut >= T_median:
                break

            train_idx_parts: List[np.ndarray] = []
            val_idx_parts: List[np.ndarray] = []

            for ent, rows in entity_rows.items():
                T_ent = len(rows)
                # Training: all rows of this entity with within-entity
                # position < cut (clamped to the entity's actual length)
                train_cut_ent = min(cut, T_ent)
                if train_cut_ent > 0:
                    train_idx_parts.append(rows[:train_cut_ent])

                # Validation: rows in [cut, val_end], clamped to T_ent.
                # Entities shorter than `cut` contribute 0 val rows — that
                # is fine; they still contribute to training in later folds.
                val_start_ent = min(cut, T_ent)
                val_end_ent = min(val_end, T_ent)
                if val_end_ent > val_start_ent:
                    val_idx_parts.append(rows[val_start_ent:val_end_ent])

            if not train_idx_parts or not val_idx_parts:
                continue

            train_idx = np.sort(np.concatenate(train_idx_parts))
            val_idx   = np.sort(np.concatenate(val_idx_parts))

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            yield train_idx, val_idx


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
        temperature: Temperature for the softmax-weighted stacking (higher
            temperature → flatter weights; lower → winner-takes-all). Only
            used when ``meta_learner_type='bayesian_stacking'``.
            Note: despite the name, the current implementation is a
            deterministic *softmax* weighting of OOF R² scores
            (temperature-scaled), not a full Bayesian Dirichlet-posterior
            stacking as in Yao et al. (2018).  The name is retained for
            backward compatibility.
        random_state: Random seed
        verbose: Print progress messages

    Example:
        >>> from forecasting.gradient_boosting import GradientBoostingForecaster
        >>> from forecasting.bayesian import BayesianForecaster
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

        # OOF ensemble predictions — stored at fit time so that
        # UnifiedForecaster can calibrate conformal intervals from pre-computed
        # residuals without deep-copying the full ensemble (U-2).
        self._oof_ensemble_predictions_: Optional[np.ndarray] = None  # (n_samples, n_outputs)
        self._oof_valid_mask_: Optional[np.ndarray] = None            # (n_samples,) bool

    @_silence_warnings
    def fit(self, X: np.ndarray, y: np.ndarray, entity_indices: Optional[np.ndarray] = None) -> "SuperLearner":
        """
        Fit the Super Learner ensemble.

        Stage 1: Generate out-of-fold predictions via temporal CV
        Stage 2: Train meta-learner on OOF predictions
        Stage 3: Re-train base models on full training data

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) or (n_samples, n_outputs)
            entity_indices: Optional entity group IDs for panel-aware models

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
        # Panel-aware temporal split: each fold's validation rows are
        # temporally later than training rows *for every entity*.
        tscv = _PanelTemporalSplit(n_splits=self.n_cv_folds)

        # OOF prediction storage: (n_samples, n_models * n_outputs)
        oof_predictions = np.full(
            (n_samples, n_models * self._n_outputs), np.nan
        )
        self._cv_scores = {name: [] for name in self.base_models}

        for fold_idx, (train_idx, val_idx) in enumerate(
            tscv.split(X, entity_indices=entity_indices)
        ):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

            for m_idx, (name, model) in enumerate(self.base_models.items()):
                try:
                    model_copy = copy.deepcopy(model)
                    # Forward entity_indices to models that accept them
                    self._fit_model(model_copy, X_train_cv, y_train_cv,
                                    entity_indices[train_idx] if entity_indices is not None else None)

                    pred = model_copy.predict(X_val_cv)
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, 1)

                    # Store OOF predictions
                    for out_col in range(self._n_outputs):
                        col_idx = m_idx * self._n_outputs + out_col
                        pred_col = min(out_col, pred.shape[1] - 1)
                        oof_predictions[val_idx, col_idx] = pred[:, pred_col]

                    # Compute CV score: mean R² across all output columns
                    # for this fold (one value per fold, not per output).
                    fold_r2s = []
                    for out_col in range(y_val_cv.shape[1]):
                        pred_col = min(out_col, pred.shape[1] - 1)
                        fold_r2s.append(
                            r2_score(y_val_cv[:, out_col], pred[:, pred_col])
                        )
                    self._cv_scores[name].append(float(np.mean(fold_r2s)))

                except Exception as e:
                    if self.verbose:
                        print(f"    Warning: {name} failed on fold {fold_idx}: {e}")
                    self._cv_scores[name].append(np.nan)

        # Compute OOF R² for each model (per-model valid mask, not joint)
        for m_idx, name in enumerate(self.base_models):
            model_cols = slice(
                m_idx * self._n_outputs, (m_idx + 1) * self._n_outputs
            )
            valid = ~np.isnan(oof_predictions[:, model_cols]).any(axis=1)
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
        # Check if ANY model produced enough valid OOF rows (per-model,
        # not joint).  The old joint mask excluded every row when a
        # single model (e.g. QuantileRF or PanelVAR) failed, poisoning
        # all rows even for models that succeeded.
        max_valid_per_model = max(
            (~np.isnan(
                oof_predictions[:, m * self._n_outputs:(m + 1) * self._n_outputs]
            ).any(axis=1)).sum()
            for m in range(n_models)
        )
        if max_valid_per_model < 5:
            # Fallback: use simple averaging if not enough OOF data
            if self.verbose:
                print("  Warning: Not enough OOF data, falling back to weighted avg")
            self._meta_weights = self._fallback_weights()
        else:
            # Pass full OOF matrix; _fit_meta_learner handles NaN
            # per-model via its own per-output valid filter.
            self._fit_meta_learner(oof_predictions, y)

        # ----------------------------------------------------------
        # Cache ensemble OOF predictions for conformal calibration
        # (U-2): build the meta-weighted blend of per-model OOF preds
        # so UnifiedForecaster can calibrate from these residuals
        # without deep-copying the full ensemble inside cv_plus.
        # ----------------------------------------------------------
        oof_ensemble = np.full((n_samples, self._n_outputs), np.nan)
        for i_out in range(self._n_outputs):
            weighted_col = np.zeros(n_samples)
            weight_sum_col = np.zeros(n_samples)
            for m_idx, name in enumerate(self.base_models):
                w = self._meta_weights.get(name, 0.0)
                if w == 0.0:
                    continue
                col_idx = m_idx * self._n_outputs + i_out
                oof_col = oof_predictions[:, col_idx]
                valid_col = ~np.isnan(oof_col)
                weighted_col[valid_col] += w * oof_col[valid_col]
                weight_sum_col[valid_col] += w
            valid_col_any = weight_sum_col > 0
            oof_ensemble[valid_col_any, i_out] = (
                weighted_col[valid_col_any] / weight_sum_col[valid_col_any]
            )
        self._oof_ensemble_predictions_ = oof_ensemble
        self._oof_valid_mask_ = ~np.isnan(oof_ensemble).any(axis=1)

        # ============================================================
        # Stage 3: Re-train all base models on full data
        # ============================================================
        for name, model in self.base_models.items():
            try:
                fitted_model = copy.deepcopy(model)
                self._fit_model(fitted_model, X, y, entity_indices)
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

    @staticmethod
    def _fit_model(model, X, y, entity_indices=None):
        """Fit a base model, forwarding entity_indices when supported."""
        import inspect
        sig = inspect.signature(model.fit)
        if 'entity_indices' in sig.parameters and entity_indices is not None:
            model.fit(X, y, entity_indices=entity_indices)
        elif 'group_indices' in sig.parameters and entity_indices is not None:
            model.fit(X, y, group_indices=entity_indices)
        else:
            model.fit(X, y)

    @staticmethod
    def _predict_model(model, X, entity_indices=None):
        """Call predict on a base model, forwarding entity_indices when supported."""
        import inspect
        sig = inspect.signature(model.predict)
        if 'entity_indices' in sig.parameters and entity_indices is not None:
            return model.predict(X, entity_indices=entity_indices)
        return model.predict(X)

    def _fit_meta_learner(self, oof_X: np.ndarray, oof_y: np.ndarray):
        """Fit the second-level meta-learner on OOF predictions.

        Handles partial OOF data gracefully: models whose OOF columns are
        entirely NaN (because they failed on every fold) are excluded from
        the NNLS / meta-learner fit and receive weight 0.  Only models
        with valid predictions participate, so one crashed model no longer
        poisons the meta-weight estimation for the remaining models.
        """
        n_models = len(self.base_models)
        model_names = list(self.base_models.keys())

        # Fit per output column, average weights
        all_coefs = []

        for out_col in range(self._n_outputs):
            # Extract model predictions for this output
            model_preds = np.column_stack([
                oof_X[:, m_idx * self._n_outputs + out_col]
                for m_idx in range(n_models)
            ])

            y_col = oof_y[:, out_col]

            # Identify models that have *any* valid OOF predictions
            model_has_data = [
                (~np.isnan(model_preds[:, m])).sum() > 0
                for m in range(n_models)
            ]
            active_indices = [m for m in range(n_models) if model_has_data[m]]

            if not active_indices:
                all_coefs.append(np.ones(n_models) / n_models)
                continue

            # Build sub-matrix using only active (non-failed) models
            active_preds = model_preds[:, active_indices]
            valid = ~np.isnan(active_preds).any(axis=1) & ~np.isnan(y_col)
            if valid.sum() < 3:
                all_coefs.append(np.ones(n_models) / n_models)
                continue

            active_preds_valid = active_preds[valid]
            y_valid = y_col[valid]

            # Compute coefficients for active models only
            active_coefs = None

            if self.meta_learner_type == "elasticnet":
                meta = ElasticNetCV(
                    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                    alphas=self.meta_alpha_range,
                    positive=self.positive_weights,
                    cv=max(2, min(3, valid.sum() // 2)),
                    max_iter=5000,
                    random_state=self.random_state,
                )
            elif self.meta_learner_type == "bayesian_stacking":
                # Bayesian stacking: softmax-weighted by OOF R²
                scores = np.array([
                    float(np.nanmean(self._cv_scores[model_names[m]]))
                    if not np.all(np.isnan(self._cv_scores[model_names[m]]))
                    else 0.0
                    for m in active_indices
                ])
                scores = np.clip(scores, 0, None)
                exp_scores = np.exp(self.temperature * scores)
                active_coefs = exp_scores / exp_scores.sum()
            else:  # ridge
                if self.positive_weights:
                    # Use NNLS for a proper non-negative least-squares
                    # solution instead of post-hoc clipping of Ridge coefs.
                    try:
                        coefs_nnls, _ = nnls(active_preds_valid, y_valid)
                        active_coefs = coefs_nnls
                        if out_col == 0:
                            self._meta_learner = None  # no sklearn meta object
                    except Exception:
                        pass  # fall through to RidgeCV

                if active_coefs is None:
                    meta = RidgeCV(
                        alphas=self.meta_alpha_range,
                        cv=TimeSeriesSplit(n_splits=max(2, min(3, valid.sum() // 2))),
                    )

            # Fit sklearn meta-learner if active_coefs not already set
            if active_coefs is None:
                try:
                    meta.fit(active_preds_valid, y_valid)
                    active_coefs = meta.coef_.copy()
                    if self.positive_weights:
                        active_coefs = np.maximum(active_coefs, 0)
                    if out_col == 0:
                        self._meta_learner = meta
                except Exception:
                    active_coefs = np.ones(len(active_indices)) / len(active_indices)

            # Scatter active coefficients back into full-model vector
            full_coefs = np.zeros(n_models)
            for i, m_idx in enumerate(active_indices):
                full_coefs[m_idx] = active_coefs[i]
            all_coefs.append(full_coefs)

        # Normalize each per-output coefficient vector to sum to 1 before
        # averaging so that outputs whose prediction scale is larger do not
        # dominate the final meta-weights.
        normed_coefs = []
        for c in all_coefs:
            s = float(np.sum(c))
            normed_coefs.append(c / s if s > 1e-15
                                else np.ones(len(c)) / len(c))
        avg_coefs = np.mean(normed_coefs, axis=0)

        # Normalize to sum to 1
        if self.normalize_weights:
            coef_sum = avg_coefs.sum()
            if coef_sum > 0:
                avg_coefs /= coef_sum
            else:
                avg_coefs = np.ones(len(self.base_models)) / len(self.base_models)

        self._meta_weights = dict(zip(self.base_models.keys(), avg_coefs))

    def _fallback_weights(self) -> Dict[str, float]:
        """Compute fallback weights from CV scores when meta-learner fails.

        Failed folds are stored as ``np.nan`` and excluded via
        ``np.nanmean``.  Models whose folds all failed get weight 0.
        """
        scores = {}
        for name, s in self._cv_scores.items():
            arr = np.array(s, dtype=float)
            if np.all(np.isnan(arr)):
                scores[name] = 0.0
            else:
                scores[name] = max(0.0, float(np.nanmean(arr)))
        total = sum(scores.values())
        if total > 0:
            return {name: v / total for name, v in scores.items()}
        return {name: 1.0 / len(scores) for name in scores}

    def predict(self, X: np.ndarray, entity_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions using the Super Learner ensemble.

        Combines base model predictions using learned meta-weights.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            entity_indices: Optional entity group IDs for panel-aware models

        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_outputs)
        """
        if not self._fitted:
            raise ValueError("Super Learner not fitted. Call fit() first.")

        all_predictions = []
        model_names = []

        for name, model in self._fitted_base_models.items():
            try:
                pred = self._predict_model(model, X, entity_indices)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                all_predictions.append(pred)
                model_names.append(name)
            except Exception:
                continue

        if not all_predictions:
            raise ValueError("No base models produced predictions")

        # Weighted combination (renormalize weights for successful models only)
        n_samples = X.shape[0]
        result = np.zeros((n_samples, self._n_outputs))

        active_weights = {name: self._meta_weights.get(name, 0.0) for name in model_names}
        weight_sum = sum(active_weights.values())
        if weight_sum > 0:
            active_weights = {n: w / weight_sum for n, w in active_weights.items()}

        for pred, name in zip(all_predictions, model_names):
            weight = active_weights.get(name, 0.0)
            for out_col in range(self._n_outputs):
                pred_col = min(out_col, pred.shape[1] - 1)
                result[:, out_col] += weight * pred[:, pred_col]

        if self._n_outputs == 1:
            return result.ravel()
        return result

    def predict_with_uncertainty(
        self, X: np.ndarray, entity_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty from model disagreement.

        Uncertainty = weighted standard deviation of base model predictions.

        Args:
            X: Feature matrix
            entity_indices: Optional entity group IDs, forwarded to panel-aware
                base models (``PanelVARForecaster``).
                When *None*, panel models fall back to population-level (zero-dummy)
                predictions, making the uncertainty estimate entity-wrong.
                Always pass the same ``entity_indices`` used during ``predict()``.

        Returns:
            Tuple of (mean prediction, standard deviation)
        """
        all_predictions = []
        all_weights = []

        for name, model in self._fitted_base_models.items():
            try:
                # Use _predict_model so that panel-aware models receive
                # entity_indices (fixing audit issue S-1: the previous
                # model.predict(X) call omitted entity_indices, causing
                # PanelVAR to predict at the reference
                # entity level for ALL entities, producing wrong uncertainty).
                pred = self._predict_model(model, X, entity_indices)
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
                name: float(np.nanmean(scores))
                for name, scores in self._cv_scores.items()
            },
            "positive_weights": self.positive_weights,
        }
