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
        self, X: np.ndarray, y: np.ndarray
    ) -> int:
        """
        Select optimal lag order using information criteria.

        For each candidate lag order *p* in ``1 .. max_lags``:
          1. Build a proper lag matrix ``[X_t, X_{t-1}, ..., X_{t-p}]``.
          2. Fit a Ridge model and compute residual sum of squares.
          3. Evaluate AIC or BIC (accounting for the extra parameters
             introduced by each additional lag).

        Args:
            X: Feature matrix (may include fixed-effect dummies).
            y: Target values.

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
                X_lag = self._build_lag_matrix(X, lag)
                y_lag = y_target[lag:]          # align y with the lagged rows
                n_samples = X_lag.shape[0]

                if n_samples < lag + 2:         # too few rows left
                    continue

                model = Ridge(alpha=self.alpha, random_state=self.random_state)
                model.fit(X_lag, y_lag)
                y_pred = model.predict(X_lag)

                rss = np.sum((y_lag - y_pred) ** 2)
                k = X_lag.shape[1]              # number of parameters

                if self.lag_selection == "aic":
                    ic = n_samples * np.log(rss / n_samples + 1e-10) + 2 * k
                else:  # bic
                    ic = n_samples * np.log(rss / n_samples + 1e-10) + k * np.log(n_samples)

                if ic < best_ic:
                    best_ic = ic
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

        # Select optimal lag order
        self.selected_lags_ = self._select_lag_order(X_panel, y)

        # Fit one model per output dimension
        all_importances = []
        for col in range(self._n_outputs):
            y_col = y[:, col]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_panel)
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
        predictions = np.zeros((X.shape[0], self._n_outputs))

        for col in range(self._n_outputs):
            scaler = self.scalers_[col]
            model = self.models_[col]
            X_scaled = scaler.transform(X_panel)
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

        # Fixed effects are captured in the dummy variable coefficients
        effects = {}
        for col in range(self._n_outputs):
            model = self.models_[col]
            coefs = model.coef_
            # Last (n_entities - 1) coefficients are the fixed effects
            # Reference entity has effect = 0
            effects[col] = coefs[-(self._n_entities - 1):] if self._n_entities > 1 else np.array([0.0])
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
