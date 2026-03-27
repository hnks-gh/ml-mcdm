# -*- coding: utf-8 -*-
"""
ElasticNet Forecaster with Automatic Regularization Tuning
===========================================================

L1+L2 penalized linear regression with explicit feature selection.
Complementary to BayesianRidge despite both being linear:

* **Explicit feature selection** — L1 penalty directly zeroes weak features
  (LASSO effect); typically retains 40–60 % of features on sparse panel data.
  BayesianRidge uses automatic relevance determination (Bayesian shrinkage),
  which is softer but less interpretable.

* **L1+L2 regularization path** — ElasticNet's hybrid penalty differs
  fundamentally from Ridge's pure L2 path, increasing ensemble diversity.

* **Cross-validated alpha/l1_ratio** — ElasticNetCV automatically tunes the
  regularization strength and L1/L2 balance per-output column via internal CV,
  adapting to each criterion's signal-to-noise ratio.

Multi-output regression via ``MultiOutputRegressor`` for C01–C08 targets.
Feature track: **PLS-reduced (PCA track)** alongside BayesianForecaster and
SVRForecaster.  Dimensionality reduction improves sparsity and interpretability.
"""

import numpy as np
import warnings
from typing import Optional, List
from sklearn.linear_model import ElasticNetCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from .base import BaseForecaster


class ElasticNetForecaster(BaseForecaster):
    """
    Multi-output ElasticNet forecaster with automatic regularization tuning.

    Wraps ``sklearn.linear_model.ElasticNetCV`` inside a
    ``MultiOutputRegressor`` to handle multi-output targets (C01–C08).
    An internal ``StandardScaler`` normalises features before fitting.

    Parameters
    ----------
    l1_ratios : List[float], optional
        L1/L2 weight ratios to try during cross-validation.
        ``l1_ratio=0`` → pure Ridge (L2).
        ``l1_ratio=1`` → pure LASSO (L1).
        ``l1_ratio=0.5`` → balanced ElasticNet.
        Default: [0.2, 0.5, 0.8] (favor L1 for feature selection).
    alphas : np.ndarray or List[float], optional
        Regularization strengths to try.  Smaller α → weaker regularization.
        Default: 20 logarithmically-spaced values in [1e-4, 10].
    cv : int
        Number of folds for internal cross-validation.  Default: 5.
    random_state : int, optional
        Random seed for CV splits.  ElasticNetCV itself is deterministic.
    max_iter : int
        Maximum solver iterations per fit.  Default: 5000 (sufficient for
        most panel datasets).
    tol : float
        Convergence tolerance.  Default: 1e-3.

    Notes
    -----
    Sets ``positive=False`` to allow both positive and negative coefficients,
    required for general regression.  If predictions must be non-negative,
    apply post-prediction clipping.

    Attributes
    ----------
    feature_importance_ : np.ndarray
        Mean absolute coefficient magnitudes across outputs.  High values
        indicate features that strongly influence predictions.
    """

    def __init__(
        self,
        l1_ratios: Optional[List[float]] = None,
        alphas: Optional[np.ndarray] = None,
        cv: int = 5,
        random_state: Optional[int] = None,
        max_iter: int = 5000,
        tol: float = 1e-3,
    ):
        if l1_ratios is None:
            l1_ratios = [0.2, 0.5, 0.8]
        if alphas is None:
            alphas = np.logspace(-4, 1, 20)

        self.l1_ratios = l1_ratios
        self.alphas = alphas
        self.cv = cv
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

        self._scaler = StandardScaler()
        self._model: Optional[MultiOutputRegressor] = None
        self.feature_importance_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.n_outputs_: Optional[int] = None

    # ------------------------------------------------------------------
    # BaseForecaster interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ElasticNetForecaster':
        """Fit ElasticNetCV model after standard-normalising X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        self

        Notes
        -----
        ``sample_weight`` is intentionally not in the signature —
        ElasticNetCV does not directly support per-sample weights.
        To use sample weights, pre-multiply X and y by sqrt(weights) before calling.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_features_ = X.shape[1]
        self.n_outputs_  = y.shape[1]

        X_scaled = self._scaler.fit_transform(X)

        # Suppress convergence warnings for short CV runs
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)

            base = ElasticNetCV(
                l1_ratio=self.l1_ratios,
                alphas=self.alphas,
                cv=self.cv,
                random_state=self.random_state,
                max_iter=self.max_iter,
                tol=self.tol,
                fit_intercept=True,
                positive=False,  # Allow negative coefficients for flexibility
                verbose=0,
            )
            self._model = MultiOutputRegressor(base, n_jobs=1)
            self._model.fit(X_scaled, y)

        # Feature importance: mean absolute coefficient magnitude per feature
        # across all outputs.  Coefficients are often sparse (many near zero);
        # absolute magnitude indicates "strength of influence".
        try:
            coef_list = [est.coef_ for est in self._model.estimators_]
            coef_array = np.array(coef_list)  # (n_outputs, n_features)
            self.feature_importance_ = np.mean(np.abs(coef_array), axis=0)
        except Exception:
            self.feature_importance_ = np.ones(self.n_features_)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_outputs)
        """
        if self._model is None:
            raise RuntimeError(
                "ElasticNetForecaster must be fitted before predict()"
            )
        X = np.asarray(X, dtype=float)
        X_scaled = self._scaler.transform(X)
        pred = self._model.predict(X_scaled)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred

    def get_feature_importance(self) -> np.ndarray:
        """Return mean-absolute-coefficient feature importance proxy.

        Returns
        -------
        np.ndarray, shape (n_features,)

        Raises
        ------
        RuntimeError
            If model is not fitted.
        """
        if self.feature_importance_ is None:
            raise RuntimeError("ElasticNetForecaster is not fitted.")
        return self.feature_importance_
