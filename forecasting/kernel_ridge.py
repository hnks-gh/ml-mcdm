# -*- coding: utf-8 -*-
"""
Kernel Ridge Regression Forecaster
====================================

Closed-form L2-regularised regression in a Reproducing Kernel Hilbert
Space via the RBF (Gaussian) kernel.  Complementary to GBM-based models
because:

* Non-parametric — can capture any smooth relationship without specifying
  a functional form.
* Closed-form dual solution — no iterative optimisation; exact fit.
* RBF kernel = ensemble of Gaussian bumps → universal approximator.
* Internal ``StandardScaler`` prevents scale sensitivity.

Feature track assignment (unified.py)
--------------------------------------
``KernelRidgeForecaster`` is routed to the **PLS-reduced (PCA) track**,
sharing the compressed linear subspace with ``BayesianForecaster``.
This exploits the fact that smooth kernel methods benefit from
dimensionality reduction (fewer eigen-directions to interpolate across).

References
----------
Saunders et al. (1998).  "Ridge regression learning algorithm in dual
variables."  In *Proceedings of the 15th ICML*, pp. 515–521.
"""

import numpy as np
from typing import Optional
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from .base import BaseForecaster


class KernelRidgeForecaster(BaseForecaster):
    """
    Multi-output Kernel Ridge Regression forecaster.

    Wraps ``sklearn.kernel_ridge.KernelRidge`` with RBF kernel inside a
    ``MultiOutputRegressor`` to handle multi-output targets (C01–C08).
    An internal ``StandardScaler`` normalises features before kernel
    evaluation (kernel methods are scale-sensitive).

    Parameters
    ----------
    alpha : float
        Tikhonov regularisation strength.  Larger → more regularised
        (higher bias, lower variance).  Default 1.0.
    gamma : str or float
        RBF bandwidth:
        ``'scale'`` → γ = 1 / (n_features × Var[X_scaled]), adapts to
        the empirical feature variance after scaling.
        ``'auto'``  → γ = 1 / n_features.
        Float → fixed bandwidth.
    random_state : int, optional
        Unused (KernelRidge is deterministic); stored for API consistency.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: object = "scale",
        kernel: str = "rbf",
        random_state: Optional[int] = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.kernel = kernel
        self.random_state = random_state

        self._scaler = StandardScaler()
        self._model: Optional[MultiOutputRegressor] = None
        self.feature_importance_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.n_outputs_: Optional[int] = None

    # ------------------------------------------------------------------
    # BaseForecaster interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KernelRidgeForecaster':
        """Fit the KernelRidge model after standard-normalising X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_features_ = X.shape[1]
        self.n_outputs_  = y.shape[1]

        X_scaled = self._scaler.fit_transform(X)

        # Resolve gamma: 'scale' → 1 / (n_features × Var[X_scaled])
        if self.gamma == "scale":
            _var = float(np.var(X_scaled))
            _var = _var if _var > 0 else 1.0
            _gamma = 1.0 / (X_scaled.shape[1] * _var)
        elif self.gamma == "auto":
            _gamma = 1.0 / X_scaled.shape[1]
        else:
            _gamma = float(self.gamma)

        base = KernelRidge(kernel=self.kernel, alpha=self.alpha, gamma=_gamma)
        self._model = MultiOutputRegressor(base, n_jobs=1)
        self._model.fit(X_scaled, y)

        # Feature importance proxy: uniform (dual coefficients live in
        # sample space, not feature space for RBF kernel; no direct
        # feature-space attribution is possible without approximation).
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
                "KernelRidgeForecaster must be fitted before predict()"
            )
        X = np.asarray(X, dtype=float)
        X_scaled = self._scaler.transform(X)
        pred = self._model.predict(X_scaled)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance array (uniform proxy for RBF kernel).

        Returns
        -------
        np.ndarray, shape (n_features,)
        """
        if self.feature_importance_ is None:
            raise RuntimeError("KernelRidgeForecaster is not fitted.")
        return self.feature_importance_
