"""
Persistence (Carry-Forward) Baseline Forecaster.

This module implements a naive "last-value-carried-forward" baseline for 
benchmarking ensemble performance. The persistence model predicts the 
last non-NaN value from the training set for all prediction rows.

This is the standard industry baseline for time-series data: it captures 
the unconditional temporal mean and is theoretically a strong 
non-learning baseline.

Mathematical Formulation
------------------------
For each output criterion c and validation row v:
    ŷ_v,c = y_train[last_non_nan_idx, c]

where last_non_nan_idx is the index of the last training row with a
non-NaN value for criterion c. All validation rows receive the same
carry-forward prediction (constant per criterion).

References
----------
- Hyndman & Athanasopoulos (2021). "Forecasting: Principles and Practice", 
  Chapter 3: Benchmarks and Forecast Accuracy Measures.
"""

import numpy as np
import logging
from typing import Optional, Dict, List, Any
from .base import BaseForecaster

logger = logging.getLogger('ml_mcdm')


class PersistenceForecaster(BaseForecaster):
    """
    Naive persistence (last-value-carried-forward) baseline forecaster.

    Stores the last non-NaN value from the training set for each output
    criterion, then predicts this constant value for all samples in the
    prediction set.

    This baseline is appropriate for:
    - Panel data with temporal structure (time-series governance).
    - Benchmarking ensemble performance via skill scores.
    - Diagnostic evaluation (if ensemble << persistence, target is mostly noise).

    Parameters
    ----------
    verbose : bool, default=True
        Whether to print progress messages.

    Attributes
    ----------
    last_values_ : np.ndarray, shape (n_outputs,)
        Last non-NaN value for each output criterion.
    n_outputs_ : int
        Number of output criteria.
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    >>> from forecasting.persistence import PersistenceForecaster
    >>> model = PersistenceForecaster()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """

    def __init__(self, verbose: bool = True) -> None:
        """
        Initialize the persistence forecaster.

        Parameters
        ----------
        verbose : bool, default=True
            Whether to print progress messages.
        """
        self.verbose = verbose
        self.last_values_: Optional[np.ndarray] = None
        self.n_outputs_: int = 1
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceForecaster":
        """
        Fit by storing the last non-NaN value per output.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix (unused; included for API compatibility).
        y : np.ndarray, shape (n_samples, n_outputs)
            Target values.

        Returns
        -------
        PersistenceForecaster
            Fitted estimator.
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
            n_nan = np.sum(np.isnan(self.last_values_))
            logger.debug(
                f"PersistenceForecaster fitted: {self.n_outputs_} outputs, "
                f"{n_nan} with NaN last-values"
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate persistence predictions.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples_pred, n_features)
            Prediction feature matrix.

        Returns
        -------
        np.ndarray, shape (n_samples_pred, n_outputs)
            Persistence predictions (constant per criterion).
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling predict.")

        n_samples = X.shape[0]

        # Tile the last values across all prediction rows
        y_pred = np.tile(self.last_values_, (n_samples, 1))

        return y_pred

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: Optional[List[float]] = None,
    ) -> Dict[float, np.ndarray]:
        """
        Generate persistence quantile predictions.

        The persistence model is deterministic, so all quantiles return 
        the same carry-forward prediction.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples_pred, n_features)
            Prediction feature matrix.
        quantiles : List[float], optional
            Quantile levels (e.g., [0.025, 0.5, 0.975]).

        Returns
        -------
        dict
            Mapping quantile → prediction array.
        """
        if quantiles is None:
            quantiles = [0.025, 0.5, 0.975]

        y_pred = self.predict(X)

        return {q: y_pred for q in quantiles}

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            Whether to return parameters for sub-estimators.

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        return {'verbose': self.verbose}

    def set_params(self, **params) -> "PersistenceForecaster":
        """
        Set parameters for this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        PersistenceForecaster
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_feature_importance(self) -> np.ndarray:
        """
        Return feature importance (none for persistence).

        Returns
        -------
        np.ndarray
            Empty array (persistence depends only on previous target values).
        """
        return np.array([])

    def __repr__(self) -> str:
        """String representation."""
        state = "fitted" if self.is_fitted_ else "unfitted"
        return (
            f"PersistenceForecaster(n_outputs={self.n_outputs_}, "
            f"state={state})"
        )
