# -*- coding: utf-8 -*-
"""
Support Vector Regression Forecaster
=====================================

ε-insensitive RBF kernel SVR via ``sklearn.svm.SVR`` wrapped in
``MultiOutputRegressor``.  Complementary to ``KernelRidgeForecaster``
despite sharing the same RBF kernel:

* **Sparsity** — only support vectors (typically 30–60 % of training
  samples) define the prediction function; the solution is sparse in
  the dual space.  Kernel Ridge uses *all* training points.
* **ε-insensitive loss** — residuals within ε contribute zero gradient,
  biasing toward smooth fits and improving robustness to label noise.
* **Different inductive bias** — SVR's Hinge-derived regularisation
  path differs from KRR's L2 path, increasing ensemble diversity.

No ``sample_weight`` support
-----------------------------
``sklearn.svm.SVR`` does not accept sample weights in ``fit()``.
``SuperLearner._fit_model()`` dispatches via ``inspect.signature``,
so the absence of a ``sample_weight`` parameter is handled automatically.

Feature track assignment (unified.py)
--------------------------------------
``SVRForecaster`` is routed to the **PLS-reduced (PCA) track**,
sharing the compressed linear subspace with ``BayesianForecaster`` and
``KernelRidgeForecaster``.  Dimensionality reduction suppresses
feature noise that would otherwise inflate the number of support vectors.
"""

import numpy as np
from typing import Optional
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from .base import BaseForecaster


class SVRForecaster(BaseForecaster):
    """
    Multi-output Support Vector Regression forecaster.

    Wraps ``sklearn.svm.SVR`` with RBF kernel inside a
    ``MultiOutputRegressor`` to handle multi-output targets (C01–C08).
    An internal ``StandardScaler`` normalises features before kernel
    evaluation.

    Parameters
    ----------
    C : float
        Regularisation parameter.  Larger C → tighter fit, more support
        vectors; smaller C → smoother fit, fewer support vectors.
        Default 1.0.
    epsilon : float
        ε-tube half-width.  Residuals |yᵢ − ŷᵢ| ≤ ε contribute zero
        loss.  Default 0.1.
    gamma : str or float
        RBF bandwidth:
        ``'scale'`` → γ = 1 / (n_features × Var[X_scaled]).
        ``'auto'``  → γ = 1 / n_features.
        Float → fixed bandwidth.
    random_state : int, optional
        Unused (SVR is deterministic); stored for API consistency.

    Notes
    -----
    ``SVR.fit()`` does **not** accept ``sample_weight`` — this is by
    sklearn design.  ``SuperLearner`` handles this transparently via
    ``inspect.signature`` dispatch and will call ``fit(X, y)`` without
    the weight argument for this model.
    """

    def __init__(
        self,
        C: float = 1.0,
        epsilon: float = 0.1,
        gamma: object = "scale",
        random_state: Optional[int] = None,
    ):
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.random_state = random_state

        self._scaler = StandardScaler()
        self._model: Optional[MultiOutputRegressor] = None
        self.feature_importance_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.n_outputs_: Optional[int] = None

    # ------------------------------------------------------------------
    # BaseForecaster interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVRForecaster':
        """Fit SVR model after standard-normalising X.

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
        sklearn SVR does not support it.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_features_ = X.shape[1]
        self.n_outputs_  = y.shape[1]

        X_scaled = self._scaler.fit_transform(X)

        base = SVR(
            kernel='rbf',
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma,
        )
        self._model = MultiOutputRegressor(base, n_jobs=1)
        self._model.fit(X_scaled, y)

        # Feature importance proxy: mean std of support vectors per feature
        # across outputs — wider SV spread on a feature implies it is more
        # discriminative.
        try:
            sv_stds = [
                np.std(est.support_vectors_, axis=0)
                for est in self._model.estimators_
            ]
            self.feature_importance_ = np.mean(sv_stds, axis=0)
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
                "SVRForecaster must be fitted before predict()"
            )
        X = np.asarray(X, dtype=float)
        X_scaled = self._scaler.transform(X)
        pred = self._model.predict(X_scaled)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred

    def get_feature_importance(self) -> np.ndarray:
        """Return SV-spread feature importance proxy.

        Returns
        -------
        np.ndarray, shape (n_features,)
        """
        if self.feature_importance_ is None:
            raise RuntimeError("SVRForecaster is not fitted.")
        return self.feature_importance_
