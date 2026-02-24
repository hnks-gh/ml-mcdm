# -*- coding: utf-8 -*-
"""
Quantile Regression Forests Forecaster
=======================================

Extends Random Forest to predict the full conditional distribution
rather than just the conditional mean. Provides distributional forecasts
with asymmetric prediction intervals and heteroscedastic uncertainty.

Instead of averaging tree predictions (point estimate), QRF retains
the full set of training observations that fall in each leaf, allowing
estimation of arbitrary quantiles of the conditional distribution.

Algorithm:
    1. Train Random Forest as usual
    2. For each test point x:
       - Identify all training samples y_i in the same leaf as x
       - Estimate quantiles from the empirical distribution of {y_i}
    3. Output: Q(τ|x) for τ ∈ {0.05, 0.10, ..., 0.95}

Key Advantages:
    - Non-parametric: No distributional assumptions
    - Heteroscedastic: Uncertainty varies with input
    - Asymmetric intervals: Captures skewness in predictions
    - Naturally calibrated: Well-calibrated uncertainty
    - No additional training cost beyond standard RF

References:
    - Meinshausen (2006). "Quantile Regression Forests" JMLR
    - Athey, Tibshirani & Wager (2019). "Generalized Random Forests"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler

from .base import BaseForecaster


class QuantileRandomForestForecaster(BaseForecaster):
    """
    Quantile Random Forest for distributional forecasting.

    Provides full predictive distributions via quantile estimation from
    the empirical distribution of training samples within tree leaves.

    Parameters:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees (None = unlimited)
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples in a leaf node
        quantiles: Quantile levels to estimate
        random_state: Random seed
        n_jobs: Number of parallel jobs (-1 = all cores)

    Example:
        >>> qrf = QuantileRandomForestForecaster(n_estimators=200)
        >>> qrf.fit(X_train, y_train)
        >>> predictions = qrf.predict(X_test)
        >>> quantile_preds = qrf.predict_quantiles(X_test, quantiles=[0.05, 0.5, 0.95])
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 3,
        quantiles: List[float] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.quantiles = quantiles or [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model_ = None
        self.scaler_ = RobustScaler()
        self.feature_importance_: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._X_train_scaled: Optional[np.ndarray] = None
        self._leaf_indices_train: Optional[np.ndarray] = None
        self._is_multi_output: bool = False
        self._n_outputs: int = 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileRandomForestForecaster":
        """
        Fit the Quantile Random Forest.

        Trains a standard Random Forest and stores training data leaf
        assignments for subsequent quantile estimation.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) or (n_samples, n_outputs)

        Returns:
            Self for method chaining
        """
        X_scaled = self.scaler_.fit_transform(X)
        self._X_train_scaled = X_scaled

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._y_train = y
        self._n_outputs = y.shape[1]
        self._is_multi_output = y.shape[1] > 1

        # Fit standard Random Forest (handles multi-output natively)
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            bootstrap=True,
            oob_score=True,
        )
        self.model_.fit(X_scaled, y.ravel() if y.shape[1] == 1 else y)
        self.feature_importance_ = self.model_.feature_importances_

        # Store leaf indices for training data (for quantile estimation)
        self._leaf_indices_train = self.model_.apply(X_scaled)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make point predictions (conditional median).

        Returns the 50th-percentile (median) of the conditional
        distribution, which is the theoretically appropriate point
        estimator for distributional forecasting (minimises MAE).
        Falls back to the RF conditional mean when quantile
        estimation fails.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_outputs)
        """
        try:
            q_preds = self.predict_quantiles(X, quantiles=[0.50])
            return q_preds[0.50]
        except Exception:
            # Fallback to RF mean if quantile estimation fails
            X_scaled = self.scaler_.transform(X)
            return self.model_.predict(X_scaled)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """
        Return the conditional mean (standard RF prediction).

        Useful when the mean is preferred over the median, e.g. for
        squared-error scoring or comparison with other models.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_outputs)
        """
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: Optional[List[float]] = None,
    ) -> Dict[float, np.ndarray]:
        """
        Predict specific quantiles of the conditional distribution.

        For each test point, identifies which training observations share
        the same leaf in each tree, then computes weighted quantiles
        from those observations.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            quantiles: List of quantile levels (default: self.quantiles)

        Returns:
            Dictionary mapping quantile level to prediction array.
            Each array has shape (n_samples,) for single output or
            (n_samples, n_outputs) for multi-output.
        """
        if quantiles is None:
            quantiles = self.quantiles

        X_scaled = self.scaler_.transform(X)
        leaf_indices_test = self.model_.apply(X_scaled)

        n_test = X_scaled.shape[0]
        n_trees = self.n_estimators

        results = {q: np.zeros((n_test, self._n_outputs)) for q in quantiles}

        for i in range(n_test):
            for out_col in range(self._n_outputs):
                # Compute sample weights: fraction of trees where each
                # training sample shares a leaf with test sample i
                weights = np.zeros(len(self._y_train))

                for t in range(n_trees):
                    test_leaf = leaf_indices_test[i, t]
                    train_leaves = self._leaf_indices_train[:, t]
                    same_leaf = train_leaves == test_leaf
                    n_in_leaf = same_leaf.sum()
                    if n_in_leaf > 0:
                        weights[same_leaf] += 1.0 / n_in_leaf

                # Normalize weights
                weight_sum = weights.sum()
                if weight_sum > 0:
                    weights /= weight_sum

                # Compute weighted quantiles
                y_col = self._y_train[:, out_col]
                sorted_indices = np.argsort(y_col)
                sorted_y = y_col[sorted_indices]
                sorted_weights = weights[sorted_indices]
                cumulative_weights = np.cumsum(sorted_weights)

                for q in quantiles:
                    # Find the value where cumulative weight >= q
                    idx = np.searchsorted(cumulative_weights, q)
                    idx = min(idx, len(sorted_y) - 1)
                    results[q][i, out_col] = sorted_y[idx]

        # Squeeze single-output results
        if self._n_outputs == 1:
            results = {q: v.ravel() for q, v in results.items()}

        return results

    def predict_intervals(
        self,
        X: np.ndarray,
        coverage: float = 0.90,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute prediction intervals at specified coverage level.

        Args:
            X: Feature matrix
            coverage: Desired coverage probability (default 0.90)

        Returns:
            Tuple of (lower_bound, median, upper_bound) arrays
        """
        alpha = (1.0 - coverage) / 2.0
        quantiles_needed = [alpha, 0.50, 1.0 - alpha]
        qpreds = self.predict_quantiles(X, quantiles=quantiles_needed)
        return qpreds[alpha], qpreds[0.50], qpreds[1.0 - alpha]

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate prediction uncertainty as IQR-based standard deviation proxy.

        Uses the interquartile range (Q75 - Q25) scaled to approximate
        standard deviation under normality: σ ≈ IQR / 1.349.

        Args:
            X: Feature matrix

        Returns:
            Uncertainty estimates (pseudo standard deviation)
        """
        qpreds = self.predict_quantiles(X, quantiles=[0.25, 0.75])
        iqr = qpreds[0.75] - qpreds[0.25]
        return iqr / 1.349  # IQR to σ conversion under normality

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet")
        return self.feature_importance_

    @property
    def oob_score(self) -> float:
        """Get out-of-bag R² score."""
        if self.model_ is None:
            raise ValueError("Model not fitted yet")
        return self.model_.oob_score_

    def get_prediction_distribution(
        self, X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get comprehensive distributional summary for predictions.

        Returns:
            Dictionary with keys: 'mean' (RF conditional mean), 'median'
            (QRF conditional median), 'std', 'q05', 'q10', 'q25', 'q50',
            'q75', 'q90', 'q95'
        """
        median_pred   = self.predict(X)        # conditional median
        mean_pred     = self.predict_mean(X)   # conditional mean (RF average)
        quantile_preds = self.predict_quantiles(X)
        uncertainty   = self.predict_uncertainty(X)

        return {
            "mean":   mean_pred,
            "median": median_pred,
            "std":    uncertainty,
            **{f"q{int(q * 100):02d}": v for q, v in quantile_preds.items()},
        }
