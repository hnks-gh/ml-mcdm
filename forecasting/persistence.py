# -*- coding: utf-8 -*-
"""
Persistence (Carry-Forward) Baseline Forecaster
==============================================

Implements a naive "last-value-carried-forward" baseline for benchmarking
ensemble performance. The persistence model predicts the last non-NaN value
from the training set for all prediction rows.

This is the standard industry baseline for time-series governance data:
it captures the unconditional temporal mean and is theoretically the
strongest non-learning baseline. A skill score > 0 indicates the ensemble
learns beyond simple temporal inertia.

Mathematical Formulation
------------------------
For each output criterion c and validation row v:

    ŷ_v,c = y_train[last_non_nan_idx, c]

where last_non_nan_idx is the index of the last training row with a
non-NaN value for criterion c. All validation rows receive the same
carry-forward prediction (constant per criterion).

References
----------
- Hyndman & Athanasopoulos (2021). Forecasting: Principles and Practice,
  Chapter 3: Benchmarks and Forecast Accuracy Measures
"""

import numpy as np
import logging
from typing import Optional
from .base import BaseForecaster

logger = logging.getLogger('ml_mcdm')


class PersistenceForecaster(BaseForecaster):
    """
    Naive persistence (last-value-carried-forward) baseline forecaster.

    Stores the last non-NaN value from the training set for each output
    criterion, then predicts this constant value for all samples in the
    prediction set.

    This baseline is appropriate for:
    - Panel data with temporal structure (time-series governance)
    - Benchmarking ensemble performance via skill scores
    - Diagnostic evaluation (if ensemble << persistence, target is mostly noise)

    Parameters
    ----------
    verbose : bool, default=True
        Print progress messages.

    Attributes
    ----------
    last_values_ : ndarray of shape (n_outputs,)
        Last non-NaN value for each output criterion.
    n_outputs_ : int
        Number of output criteria.
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    >>> from forecasting.persistence import PersistenceForecaster
    >>> from sklearn.metrics import r2_score
    >>>
    >>> # Create synthetic data: 100 samples, 5 criteria
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randn(100, 5)
    >>>
    >>> # Fit and predict
    >>> model = PersistenceForecaster()
    >>> model.fit(X[:80], y[:80])
    >>> y_pred = model.predict(X[80:])
    >>>
    >>> # All 20 prediction rows should have identical predictions
    >>> assert y_pred.shape == (20, 5)
    >>> assert np.allclose(y_pred, model.last_values_, equal_nan=True)
    """

    def __init__(self, verbose: bool = True):
        """Initialize the persistence forecaster."""
        self.verbose = verbose
        self.last_values_: Optional[np.ndarray] = None
        self.n_outputs_: int = 1
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceForecaster":
        """
        Fit the persistence model by storing the last non-NaN value per output.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (unused; included for API compatibility).
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Target values. May contain NaN entries; the last non-NaN
            value per criterion is stored.

        Returns
        -------
        self : PersistenceForecaster
            Fitted model.
        """
        # Normalize y to 2-D
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_outputs_ = y.shape[1]
        self.last_values_ = np.full(self.n_outputs_, np.nan)

        # For each output criterion, find the last non-NaN value
        for j in range(self.n_outputs_):
            valid_indices = np.where(~np.isnan(y[:, j]))[0]
            if len(valid_indices) > 0:
                last_idx = valid_indices[-1]  # index of last non-NaN
                self.last_values_[j] = y[last_idx, j]

        self.is_fitted_ = True

        if self.verbose:
            n_nan = np.isnan(self.last_values_).sum()
            logger.debug(
                f"PersistenceForecaster fitted: {self.n_outputs_} outputs, "
                f"{n_nan} with NaN last-values"
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate persistence predictions (carry-forward last training value).

        For all prediction rows, returns the last non-NaN value from training
        for each criterion. This is a constant prediction matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples_pred, n_features)
            Prediction feature matrix. Shape[0] determines the number of
            output rows; features are unused.

        Returns
        -------
        y_pred : ndarray of shape (n_samples_pred, n_outputs)
            Persistence predictions: each row is identical to last_values_.
        """
        if not self.is_fitted_:
            raise ValueError(
                "Model must be fitted before calling predict(). "
                "Call fit() first."
            )

        n_samples = X.shape[0]

        # Tile the last values across all prediction rows
        y_pred = np.tile(self.last_values_, (n_samples, 1))

        return y_pred

    def predict_quantiles(
        self, X: np.ndarray, quantiles: Optional[list] = None
    ) -> dict:
        """
        Generate persistence quantile predictions (point predictions only).

        The persistence model is deterministic, so all quantiles return the
        same carry-forward prediction. This method is provided for API
        compatibility with models that support probabilistic predictions
        (e.g., QuantileRF).

        Parameters
        ----------
        X : ndarray of shape (n_samples_pred, n_features)
            Prediction feature matrix.
        quantiles : list of float, optional
            Quantile levels (0.025, 0.5, 0.975, etc.). Default: [0.025, 0.5, 0.975].

        Returns
        -------
        dict
            Mapping quantile → prediction array. All quantiles return the
            same carry-forward value.
        """
        if quantiles is None:
            quantiles = [0.025, 0.5, 0.975]

        y_pred = self.predict(X)

        return {q: y_pred for q in quantiles}

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator (scikit-learn API compatibility)."""
        return {'verbose': self.verbose}

    def set_params(self, **params) -> "PersistenceForecaster":
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_feature_importance(self) -> np.ndarray:
        """
        Feature importance for persistence forecaster.
        
        Returns uniform (zero) importance since persistence does not depend
        on any features — it only uses the last observed target value.
        
        A deterministic model has no feature-based learned importance.
        This method is provided for API consistency with BaseForecaster.
        
        Returns
        -------
        ndarray, shape (n_features,) or (n_features, n_outputs)
            Zero importance for all features, since persistence ignores
            features entirely. Returns an empty array to signal 
            "non-feature-based" model.
        """
        # Return empty array to indicate no feature importance
        # (persistence doesn't use features)
        return np.array([])

    def __repr__(self) -> str:
        """String representation."""
        state = "fitted" if self.is_fitted_ else "unfitted"
        return (
            f"PersistenceForecaster(n_outputs={self.n_outputs_}, "
            f"state={state})"
        )
