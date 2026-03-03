# -*- coding: utf-8 -*-
"""
Panel Vector Autoregression (Panel VAR) Forecaster
===================================================

Exploits the panel structure of the data by modeling each province
with fixed effects (province-specific intercepts) and temporal
autoregressive dynamics.

Model:
    y_it = α_i + Σ_k Γ_k * y_{i,t-k} + β * X_it + ε_it

Where:
    - α_i = province fixed effect (time-invariant)
    - Γ_k = autoregressive coefficient matrices at lag k
    - X_it = exogenous features (momentum, trend, etc.)
    - ε_it = idiosyncratic error

This approach:
    - Captures province-level heterogeneity via fixed effects
    - Models temporal dependencies via autoregressive lags
    - Works well with small T (14 years) and moderate N (63 provinces)
    - Provides interpretable coefficients

References:
    - Holtz-Eakin, Newey & Rosen (1988). "Estimating VARs with Panel Data"
    - Abrigo & Love (2016). "Estimation of Panel VAR in Stata"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

from .base import BaseForecaster


class PanelVARForecaster(BaseForecaster):
    """
    Panel VAR forecaster with fixed effects and ML-augmented features.

    Combines classical panel econometric structure (fixed effects + AR lags)
    with machine learning regularization (Ridge/ElasticNet) to handle
    the curse of dimensionality in high-dimensional panel VAR.

    Parameters:
        n_lags: Number of autoregressive lags (1-3)
        alpha: Regularization strength for Ridge/ElasticNet
        use_fixed_effects: Whether to include entity fixed effects
        lag_selection: Method for choosing optimal lag ('aic', 'bic', or 'fixed')
        max_lags: Maximum lags to consider during selection
        regularizer: Type of regularizer ('ridge' or 'elasticnet')
        l1_ratio: L1/L2 mix for ElasticNet (0=Ridge, 1=Lasso)
        random_state: Random seed

    Example:
        >>> forecaster = PanelVARForecaster(n_lags=2, use_fixed_effects=True)
        >>> forecaster.fit(X_train, y_train, entity_ids=entity_ids)
        >>> predictions = forecaster.predict(X_test)
    """

    def __init__(
        self,
        n_lags: int = 2,
        alpha: float = 1.0,
        use_fixed_effects: bool = True,
        lag_selection: str = "bic",
        max_lags: int = 3,
        regularizer: str = "ridge",
        l1_ratio: float = 0.5,
        random_state: int = 42,
    ):
        self.n_lags = n_lags
        self.alpha = alpha
        self.use_fixed_effects = use_fixed_effects
        self.lag_selection = lag_selection
        self.max_lags = max_lags
        self.regularizer = regularizer
        self.l1_ratio = l1_ratio
        self.random_state = random_state

        self.models_: Dict[int, object] = {}
        self.scalers_: Dict[int, StandardScaler] = {}
        self.fixed_effects_: Optional[Dict[str, np.ndarray]] = None
        self.feature_importance_: Optional[np.ndarray] = None
        self.selected_lags_: int = n_lags
        self._n_outputs: int = 0
        self._entity_encoder: Dict = {}   # maps entity_id → column index in dummies
        self._unique_entities: Optional[np.ndarray] = None
        self._n_entities: int = 0
        self._n_base_features: int = 0     # original X.shape[1] to trim feature_importance_
        self._fitted: bool = False

    def _build_panel_features(
        self,
        X: np.ndarray,
        y_history: Optional[np.ndarray] = None,
        entity_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Augment feature matrix with panel-specific components.

        Adds entity fixed effect dummies (within-transformation) and
        ensures autoregressive lag features are properly structured.
        Uses the entity mapping stored during fit() so that predict()
        produces the same number of columns.

        Args:
            X: Base feature matrix (n_samples, n_features)
            y_history: Historical target values for AR lags (optional)
            entity_indices: Entity index for each sample (optional)

        Returns:
            Augmented feature matrix
        """
        features_list = [X]

        # Add fixed effects as entity dummies (LSDV approach)
        if self.use_fixed_effects and self._n_entities > 1:
            n_dummies = self._n_entities - 1  # drop first for identification
            dummies = np.zeros((X.shape[0], n_dummies))

            if entity_indices is not None:
                for ent_id, col_idx in self._entity_encoder.items():
                    if col_idx is not None:  # col_idx is None for the reference entity
                        mask = entity_indices == ent_id
                        dummies[mask, col_idx] = 1.0

            features_list.append(dummies)

        return np.hstack(features_list)

    @staticmethod
    def _build_lag_matrix(X: np.ndarray, n_lags: int) -> np.ndarray:
        """
        Construct a lagged feature matrix from *X*.

        For lag order *p*, row *t* of the result is::

            [X[t], X[t-1], X[t-2], ..., X[t-p]]

        The first *p* rows are dropped because they lack sufficient
        history.

        Args:
            X: Feature matrix of shape (T, d).
            n_lags: Number of lags *p* to append.

        Returns:
            Lagged matrix of shape (T - p, d * (1 + p)).
        """
        n_rows, n_cols = X.shape
        parts = [X[n_lags:]]
        for lag in range(1, n_lags + 1):
            parts.append(X[n_lags - lag : n_rows - lag])
        return np.hstack(parts)

    def _select_lag_order(
        self, X: np.ndarray, y: np.ndarray,
        entity_indices: Optional[np.ndarray] = None,
    ) -> int:
        """
        Select optimal lag order using information criteria.

        For each candidate lag order *p* in ``1 .. max_lags``:
          1. Build a proper lag matrix per entity (no cross-entity leakage).
          2. Fit a Ridge model and compute residual sum of squares.
          3. Evaluate AIC or BIC.

        Args:
            X: Feature matrix (may include fixed-effect dummies).
            y: Target values.
            entity_indices: Optional entity IDs (enables per-entity lag build).

        Returns:
            Optimal number of lags.
        """
        if self.lag_selection == "fixed":
            return self.n_lags

        best_ic = np.inf
        best_lag = 1

        y_target = y if y.ndim == 1 else y[:, 0]

        for lag in range(1, min(self.max_lags + 1, 4)):
            try:
                if entity_indices is not None:
                    # Build pep-entity lag matrices to avoid cross-entity
                    # boundary contamination during IC evaluation.
                    X_parts, y_parts = [], []
                    for ent in np.unique(entity_indices):
                        mask = entity_indices == ent
                        X_ent = X[mask]
                        y_ent = y_target[mask]
                        if X_ent.shape[0] <= lag:
                            continue
                        X_parts.append(self._build_lag_matrix(X_ent, lag))
                        y_parts.append(y_ent[lag:])
                    if not X_parts:
                        continue
                    X_lag = np.vstack(X_parts)
                    y_lag = np.concatenate(y_parts)
                else:
                    X_lag = self._build_lag_matrix(X, lag)
                    y_lag = y_target[lag:]

                n_samples = X_lag.shape[0]
                if n_samples < lag + 2:
                    continue

                # Hold-out CV: train on first ~80 %, evaluate on last ~20 %.
                # This replaces the former in-sample Ridge AIC/BIC for two
                # reasons:
                #   (a) Ridge in-sample RSS is not monotone in lag order
                #       (shrinkage inflates in-sample RSS vs OLS).
                #   (b) Standard AIC/BIC penalties assume OLS and count raw
                #       parameters; under Ridge the effective df is
                #       tr(X(X'X+λI)⁻¹X') ≪ k, so the penalty severely
                #       over-penalises complexity and biases toward lag=1.
                h = max(1, n_samples // 5)
                if n_samples - h < lag + 2:
                    continue
                X_tr, X_val = X_lag[:-h], X_lag[-h:]
                y_tr, y_val = y_lag[:-h], y_lag[-h:]

                model = Ridge(alpha=self.alpha, random_state=self.random_state)
                model.fit(X_tr, y_tr)
                y_val_pred = model.predict(X_val)
                hold_out_mse = float(np.mean((y_val - y_val_pred) ** 2))

                if hold_out_mse < best_ic:
                    best_ic = hold_out_mse
                    best_lag = lag
            except Exception:
                continue

        return best_lag

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        entity_indices: Optional[np.ndarray] = None,
    ) -> "PanelVARForecaster":
        """
        Fit the Panel VAR model.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) or (n_samples, n_outputs)
            entity_indices: Optional entity IDs for fixed effects

        Returns:
            Self for method chaining
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._n_outputs = y.shape[1]
        self._n_base_features = X.shape[1]

        # Track entity encoding for fixed effects
        if entity_indices is not None:
            unique = np.unique(entity_indices)
            self._unique_entities = unique
            self._n_entities = len(unique)
            # Build encoder: reference entity (first) gets None, rest get column indices
            self._entity_encoder = {unique[0]: None}
            for col_idx, ent in enumerate(unique[1:]):
                self._entity_encoder[ent] = col_idx
        else:
            self._n_entities = 0
            self._unique_entities = None
            self._entity_encoder = {}

        # Build panel features with fixed effects
        X_panel = self._build_panel_features(X, entity_indices=entity_indices)

        # Select optimal lag order (entity-aware to avoid boundary contamination)
        self.selected_lags_ = self._select_lag_order(
            X_panel, y, entity_indices=entity_indices
        )

        # Build lag-augmented feature matrix per entity to avoid cross-entity
        # boundary contamination (lag of entity[i+1] row 0 must not include
        # entity[i] last rows).
        if self.selected_lags_ > 0:
            if entity_indices is not None:
                X_parts, y_parts = [], []
                for ent in np.unique(entity_indices):
                    mask = entity_indices == ent
                    X_ent = X_panel[mask]
                    y_ent = y[mask]
                    if X_ent.shape[0] <= self.selected_lags_:
                        continue   # entity too short to produce any lag rows
                    X_parts.append(self._build_lag_matrix(X_ent, self.selected_lags_))
                    y_parts.append(y_ent[self.selected_lags_:])
                if X_parts:
                    X_fit = np.vstack(X_parts)
                    y_fit = np.vstack(y_parts)
                else:            # fallback: no entity had enough rows
                    X_fit = self._build_lag_matrix(X_panel, self.selected_lags_)
                    y_fit = y[self.selected_lags_:]
            else:
                X_fit = self._build_lag_matrix(X_panel, self.selected_lags_)
                y_fit = y[self.selected_lags_:]
        else:
            X_fit = X_panel
            y_fit = y

        # Store per-entity training tails so that predict() can prepend the
        # correct lag history for each entity without cross-entity leakage.
        if entity_indices is not None:
            self._X_panel_tail_ = {
                ent: X_panel[entity_indices == ent][-max(self.selected_lags_, 1):]
                for ent in np.unique(entity_indices)
            }
        else:
            self._X_panel_tail_ = X_panel[-max(self.selected_lags_, 1):]

        # Fit one model per output dimension
        all_importances = []
        for col in range(self._n_outputs):
            y_col = y_fit[:, col]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_fit)
            self.scalers_[col] = scaler

            if self.regularizer == "elasticnet":
                model = ElasticNet(
                    alpha=self.alpha,
                    l1_ratio=self.l1_ratio,
                    max_iter=2000,
                    random_state=self.random_state,
                )
            else:
                model = Ridge(alpha=self.alpha, random_state=self.random_state)

            model.fit(X_scaled, y_col)
            self.models_[col] = model
            all_importances.append(np.abs(model.coef_))

        # Average feature importance across outputs (trim to original X size)
        avg_imp = np.mean(all_importances, axis=0)
        self.feature_importance_ = avg_imp[: self._n_base_features]
        self._fitted = True
        return self

    def predict(
        self, X: np.ndarray, entity_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Make predictions using the fitted Panel VAR model.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            entity_indices: Optional entity IDs. If provided, entity-specific
                           fixed effects are applied. If None, predictions
                           are at the population level (reference entity).

        Returns:
            Predictions of shape (n_samples, n_outputs)
        """
        if not self._fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        X_panel = self._build_panel_features(X, entity_indices=entity_indices)

        # Build lag-augmented features for prediction.
        # Use per-entity tails to avoid cross-entity boundary contamination.
        if self.selected_lags_ > 0:
            if entity_indices is not None and isinstance(self._X_panel_tail_, dict):
                X_parts: List[np.ndarray] = []
                row_positions: List[np.ndarray] = []
                for ent in np.unique(entity_indices):
                    ent_mask = np.where(entity_indices == ent)[0]
                    X_ent = X_panel[ent_mask]
                    tail = self._X_panel_tail_.get(ent)
                    if tail is not None and len(tail) >= self.selected_lags_:
                        tail_rows = tail[-self.selected_lags_:]
                        X_extended = np.vstack([tail_rows, X_ent])
                        X_parts.append(
                            self._build_lag_matrix(X_extended, self.selected_lags_)
                        )
                    else:
                        # Insufficient history: zero-pad lag columns
                        n_lag_feats = X_panel.shape[1] * (1 + self.selected_lags_)
                        X_parts.append(np.zeros((len(X_ent), n_lag_feats)))
                    row_positions.append(ent_mask)
                # Reassemble rows in their original order
                stacked = np.vstack(X_parts)
                original_positions = np.concatenate(row_positions)
                inv_order = np.argsort(original_positions)
                X_lagged = stacked[inv_order]
            else:
                if isinstance(self._X_panel_tail_, dict):
                    # Fitted with entity_indices but predict() called without:
                    # zero-pad lag columns to preserve sample count.
                    # (population-level / reference-entity predictions)
                    n_lag_feats = X_panel.shape[1] * (1 + self.selected_lags_)
                    X_lagged = np.zeros((X_panel.shape[0], n_lag_feats))
                    X_lagged[:, : X_panel.shape[1]] = X_panel
                else:
                    tail_arr = (
                        self._X_panel_tail_
                        if isinstance(self._X_panel_tail_, np.ndarray)
                        else np.empty((0, X_panel.shape[1]))
                    )
                    tail_rows = tail_arr[-self.selected_lags_:]
                    X_extended = np.vstack([tail_rows, X_panel])
                    X_lagged = self._build_lag_matrix(X_extended, self.selected_lags_)
        else:
            X_lagged = X_panel

        predictions = np.zeros((X.shape[0], self._n_outputs))

        for col in range(self._n_outputs):
            scaler = self.scalers_[col]
            model = self.models_[col]
            X_scaled = scaler.transform(X_lagged)
            predictions[:, col] = model.predict(X_scaled)

        if self._n_outputs == 1:
            return predictions.ravel()
        return predictions

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from absolute coefficient magnitudes."""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet")
        return self.feature_importance_

    def get_fixed_effects(self) -> Optional[Dict[str, float]]:
        """
        Extract estimated fixed effects for each entity.

        Returns:
            Dictionary mapping entity identifiers to their fixed effect values,
            or None if fixed effects are not used.
        """
        if not self.use_fixed_effects or not self._fitted:
            return None

        # After lag expansion, the Ridge coefficient vector has the structure:
        #   Block 0 (current time t) : [base_feats(0..n_base-1), dummies(n_base..n_base+n_dum-1)]
        #   Block 1 (lag 1, t-1)     : [base_feats, dummies]
        #   ...
        # The LSDV fixed effects are the coefficients on the entity dummies in the
        # CURRENT-TIME block (block 0), i.e. positions n_base_features through
        # n_base_features + n_dummies - 1.  Using coefs[-(n_entities-1):] was
        # incorrect because those are the entity-dummy coefficients from the
        # *last* lag block, not the fixed effects.
        n_dummies = self._n_entities - 1  # reference entity is absorbed into intercept
        effects = {}
        for col in range(self._n_outputs):
            model = self.models_[col]
            coefs = model.coef_
            if self._n_entities > 1:
                fe_slice = coefs[
                    self._n_base_features : self._n_base_features + n_dummies
                ]
                effects[col] = fe_slice
            else:
                effects[col] = np.array([0.0])
        return effects

    def get_diagnostics(self) -> Dict[str, any]:
        """
        Return model diagnostics.

        Returns:
            Dictionary with selected lag order and model info.
        """
        return {
            "selected_lags": self.selected_lags_,
            "n_outputs": self._n_outputs,
            "regularizer": self.regularizer,
            "alpha": self.alpha,
            "use_fixed_effects": self.use_fixed_effects,
        }
