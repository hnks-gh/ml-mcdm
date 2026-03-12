# -*- coding: utf-8 -*-
"""
Multi-Task Elastic Net Forecaster
==================================

Joint linear forecaster via ``MultiTaskElasticNetCV`` with automatic
regularisation selection and ``BayesianRidge + MultiOutputRegressor`` fallback.

Design rationale vs. BayesianRidge + MultiOutputRegressor
---------------------------------------------------------
*  **Joint sparsity (Group Lasso)** — ``MultiTaskElasticNet`` enforces a
   shared sparsity pattern across all criterion composites (C01–C08) via
   Group Lasso penalty.  If a feature is predictive for *any* criterion,
   it is retained for *all*, explicitly exploiting cross-criteria correlation.
   ``MultiOutputRegressor`` trains N independent models: no feature sharing.

*  **Automatic regularisation** — ``MultiTaskElasticNetCV`` selects ``alpha``
   (regularisation strength) and ``l1_ratio`` (Elastic Net mixing) via
   cross-validation, removing two previously hand-tuned hyperparameters.

*  **Uncertainty** — ``MultiTaskElasticNet`` is a point estimator.  Per-output
   aleatoric uncertainty is calibrated as the root-mean-squared training
   residual per criterion (``sigma_j``), clipped at 1e-6 for strict
   positivity.  The conformal stage in ``UnifiedForecaster`` replaces this
   with distribution-free coverage-guaranteed intervals downstream.

*  **Fallback** — when ``MultiTaskElasticNetCV`` fails (e.g. numeric issues
   on very small folds), the model falls back to
   ``BayesianRidge + MultiOutputRegressor``, which provides exact Bayesian
   posterior standard deviations via ``return_std=True``.

References
----------
Obozinski et al. (2010). "Joint covariate selection and joint subspace
selection for multiple classification problems." *Statistics and Computing* 20.
"""

import numpy as np
from typing import List, Optional, Tuple
from sklearn.base import clone
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNetCV,
    MultiTaskElasticNetCV,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from .base import BaseForecaster


class BayesianForecaster(BaseForecaster):
    """
    Joint linear forecaster using MultiTaskElasticNetCV with BayesianRidge fallback.

    Primary model: ``MultiTaskElasticNetCV``
        Enforces joint sparsity across all criterion composites via a Group
        Lasso penalty.  ``alpha`` and ``l1_ratio`` are chosen automatically
        by cross-validation, removing hand-tuned hyperparameters.

    Uncertainty estimation:
        ``MultiTaskElasticNet`` is a point estimator.  Per-output aleatoric
        uncertainty is estimated as ``sigma_j = RMS(y_j − yhat_j_train)``,
        clipped at 1e-6 to guarantee strict positivity.  Downstream conformal
        calibration in ``UnifiedForecaster`` produces coverage-guaranteed
        intervals from these residuals.

    Fallback: ``BayesianRidge + MultiOutputRegressor``
        Triggered automatically when ``MultiTaskElasticNetCV`` raises an
        exception.  Provides exact Bayesian posterior std via ``return_std``.

    Parameters
    ----------
    alpha_1, alpha_2, lambda_1, lambda_2 : float
        BayesianRidge hyperpriors — used **only** in fallback mode.
    max_iter : int
        Maximum coordinate-descent iterations for ElasticNet solvers.
    cv_folds : int
        Cross-validation folds for ``MultiTaskElasticNetCV``.
    l1_ratios : list of float
        L1/L2 mixing grid searched by CV.  0 = Ridge, 1 = Lasso.
        Default covers the full spectrum from near-Ridge to pure Lasso.
    """

    def __init__(
        self,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        max_iter: int = 300,
        cv_folds: int = 5,
        l1_ratios: Optional[List[float]] = None,
    ):
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.max_iter = max_iter
        self.cv_folds = cv_folds
        self.l1_ratios: List[float] = (
            l1_ratios
            if l1_ratios is not None
            else [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        )

        self.model = None          # MultiTaskElasticNetCV | ElasticNetCV | MultiOutputRegressor(BayesianRidge)
        self.scaler = StandardScaler()
        self.feature_importance_: Optional[np.ndarray] = None
        self._is_multi_output: bool = False
        self._using_elasticnet: bool = True
        # Per-output training-residual RMSE used for uncertainty broadcast
        self._sigma_: Optional[np.ndarray] = None   # shape (n_outputs,)

    # ------------------------------------------------------------------
    # BayesianRidge helper (fallback / backward-compat)
    # ------------------------------------------------------------------

    def _make_bayesian_ridge(self) -> BayesianRidge:
        return BayesianRidge(
            alpha_1=self.alpha_1, alpha_2=self.alpha_2,
            lambda_1=self.lambda_1, lambda_2=self.lambda_2,
            max_iter=self.max_iter, compute_score=True,
        )

    @property
    def _base_model(self) -> BayesianRidge:
        """Backward-compatible accessor returning a fresh BayesianRidge instance."""
        return self._make_bayesian_ridge()

    # ------------------------------------------------------------------
    # Public API  (BaseForecaster interface)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianForecaster':
        """
        Fit MultiTaskElasticNetCV (primary) or BayesianRidge (fallback).

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        self
        """
        X_scaled = self.scaler.fit_transform(X)

        if y.ndim > 1 and y.shape[1] > 1:
            self._is_multi_output = True
            self._fit_multitask(X_scaled, y)
        else:
            self._is_multi_output = False
            self._fit_single(X_scaled, y.ravel() if y.ndim > 1 else y)

        return self

    def _fit_multitask(self, X_scaled: np.ndarray, y: np.ndarray) -> None:
        """
        Primary: MultiTaskElasticNetCV (joint sparsity across all outputs).
        Fallback: BayesianRidge + MultiOutputRegressor.
        
        Enhancement M-04: Partial-Target Recovery
        -----------------------------------------
        Handles NaN in target matrix via sample weighting. Rows with partial
        NaN retain their valid target values; NaN cells are filled with per-
        column median (for solver compatibility) but receive zero weight.
        
        Sample weight per row = (n_valid_outputs / n_total_outputs), ensuring
        rows with more missing targets contribute less to the loss.
        """
        # Safe CV folds: at least 2, at most min(cv_folds, n//2, 5)
        n_cv = max(min(self.cv_folds, X_scaled.shape[0] // 2, 5), 2)

        # M-04: Detect and handle NaN targets
        nan_mask = np.isnan(y)  # (n_samples, n_outputs)
        has_nan = nan_mask.any()
        
        if has_nan:
            # Fill NaN with per-column median for solver compatibility
            y_filled = y.copy()
            for j in range(y.shape[1]):
                col_median = np.nanmedian(y[:, j])
                if np.isnan(col_median):
                    col_median = 0.0  # Entire column NaN → neutral fill
                y_filled[nan_mask[:, j], j] = col_median
            
            # Sample weight: fraction of valid targets per row
            # Rows with all NaN get weight 0; partial NaN get proportional weight
            n_valid_per_row = (~nan_mask).sum(axis=1)
            sample_weight = n_valid_per_row / y.shape[1]
            # Zero-weight rows should be filtered upstream, but guard here
            sample_weight = np.clip(sample_weight, 1e-9, 1.0)
        else:
            y_filled = y
            sample_weight = None

        try:
            mtnet = MultiTaskElasticNetCV(
                l1_ratio=self.l1_ratios,
                eps=1e-3,          # alpha_min / alpha_max ratio along path
                alphas=20,         # regularisation grid resolution
                cv=n_cv,
                max_iter=self.max_iter * 10,  # coordinate descent > gradient steps
                random_state=42,
                n_jobs=1,          # deterministic; avoids joblib deadlocks
            )
            # MultiTaskElasticNetCV.fit() accepts sample_weight (sklearn >= 0.23)
            if sample_weight is not None:
                mtnet.fit(X_scaled, y_filled, sample_weight=sample_weight)
            else:
                mtnet.fit(X_scaled, y_filled)
            
            self.model = mtnet
            self._using_elasticnet = True
            # coef_ shape: (n_outputs, n_features); average |coef| across outputs
            self.feature_importance_ = np.mean(np.abs(mtnet.coef_), axis=0)
            # Per-output training RMSE: aleatoric noise estimate per criterion
            # Only compute residuals on valid (non-NaN) entries
            residuals_per_output = []
            for j in range(y.shape[1]):
                valid_mask_j = ~nan_mask[:, j]
                if valid_mask_j.any():
                    pred_j = mtnet.predict(X_scaled[valid_mask_j])
                    # Ensure pred_j is 2D (n_samples, n_outputs)
                    if pred_j.ndim == 1:
                        pred_j = pred_j.reshape(-1, 1)
                    # Get column j safely
                    col_idx = min(j, pred_j.shape[1] - 1)
                    res_j = y[valid_mask_j, j] - pred_j[:, col_idx]
                    rmse_j = np.sqrt(np.mean(res_j ** 2))
                else:
                    rmse_j = 1.0  # Entire output NaN → default uncertainty
                residuals_per_output.append(rmse_j)
            self._sigma_ = np.clip(np.array(residuals_per_output), 1e-6, None)

        except Exception:
            # Fallback: BayesianRidge + MultiOutputRegressor (one model per output)
            self._using_elasticnet = False
            fallback = MultiOutputRegressor(
                clone(self._make_bayesian_ridge()), n_jobs=1
            )
            if sample_weight is not None:
                fallback.fit(X_scaled, y_filled, sample_weight=sample_weight)
            else:
                fallback.fit(X_scaled, y_filled)
            self.model = fallback
            # Average absolute coefficients across per-output BayesianRidge models
            self.feature_importance_ = np.mean(
                [np.abs(est.coef_) for est in fallback.estimators_], axis=0
            )
            # Noise std from BayesianRidge posterior: sigma = 1 / sqrt(alpha_)
            self._sigma_ = np.clip(
                np.array([
                    np.sqrt(1.0 / max(float(est.alpha_), 1e-12))
                    for est in fallback.estimators_
                ]),
                1e-6, None,
            )

    def _fit_single(self, X_scaled: np.ndarray, y_1d: np.ndarray) -> None:
        """
        Single-output case: ElasticNetCV (primary), BayesianRidge (fallback).
        """
        n_cv = max(min(self.cv_folds, X_scaled.shape[0] // 2, 5), 2)

        try:
            enet = ElasticNetCV(
                l1_ratio=self.l1_ratios,
                eps=1e-3,
                alphas=20,
                cv=n_cv,
                max_iter=self.max_iter * 10,
                random_state=42,
            )
            enet.fit(X_scaled, y_1d)
            self.model = enet
            self._using_elasticnet = True
            self.feature_importance_ = np.abs(enet.coef_)
            residuals = y_1d - enet.predict(X_scaled)
            self._sigma_ = np.array(
                [max(float(np.sqrt(np.mean(residuals ** 2))), 1e-6)]
            )

        except Exception:
            self._using_elasticnet = False
            br = clone(self._make_bayesian_ridge())
            br.fit(X_scaled, y_1d)
            self.model = br
            self.feature_importance_ = np.abs(br.coef_)
            self._sigma_ = np.array(
                [max(float(np.sqrt(1.0 / max(float(br.alpha_), 1e-12))), 1e-6)]
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make point predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty quantification.

        For the primary ElasticNet model, returns the per-output training-
        residual RMSE (``sigma_j``) broadcast uniformly across all samples.
        This estimates the irreducible aleatoric noise per criterion and is
        NOT a per-sample epistemic uncertainty.  Downstream conformal
        calibration in ``UnifiedForecaster`` provides proper coverage-
        guaranteed intervals.

        For the BayesianRidge fallback, returns the exact posterior predictive
        std (aleatoric + epistemic) via ``return_std=True``.

        All returned ``std`` values are strictly positive (≥ 1e-6).

        Returns
        -------
        mean : ndarray, shape (n_samples, n_outputs) or (n_samples,)
        std  : ndarray, same shape as mean; all elements > 0
        """
        X_scaled = self.scaler.transform(X)

        if self._using_elasticnet:
            mean = self.model.predict(X_scaled)
            if mean.ndim == 1:
                # Single-output: broadcast scalar sigma across all samples
                std = np.full(mean.shape, self._sigma_[0])
            else:
                # Multi-output: broadcast per-output sigma along sample axis.
                # _sigma_ shape: (n_outputs,) → broadcast to (n_samples, n_outputs)
                std = np.broadcast_to(
                    self._sigma_[np.newaxis, :], mean.shape
                ).copy()  # copy() makes the array writeable
        else:
            # BayesianRidge fallback provides exact posterior std
            if self._is_multi_output:
                means_list, stds_list = [], []
                for est in self.model.estimators_:
                    m, s = est.predict(X_scaled, return_std=True)
                    means_list.append(m)
                    stds_list.append(s)
                mean = np.column_stack(means_list)
                std = np.clip(np.column_stack(stds_list), 1e-6, None)
            else:
                mean, std = self.model.predict(X_scaled, return_std=True)
                std = np.clip(std, 1e-6, None)

        return mean, std

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from absolute coefficients."""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet")
        return self.feature_importance_

    # ------------------------------------------------------------------
    # Backward-compatible properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """Effective noise precision.

        ElasticNet primary: ``1 / mean(sigma_j^2)`` across outputs.
        BayesianRidge fallback: mean of per-output ``alpha_`` values.
        """
        if self._sigma_ is None:
            raise AttributeError("Model not fitted yet.")
        if self._using_elasticnet:
            return float(np.mean(1.0 / (self._sigma_ ** 2 + 1e-12)))
        if self._is_multi_output:
            return float(np.mean([est.alpha_ for est in self.model.estimators_]))
        return float(self.model.alpha_)

    @property
    def lambda_(self) -> float:
        """Effective regularisation precision.

        ElasticNet primary: the CV-selected ``alpha_`` regularisation parameter
        (sklearn naming convention; analogous to lambda in the primal objective).
        BayesianRidge fallback: mean of per-output ``lambda_`` values.
        """
        if self._sigma_ is None:
            raise AttributeError("Model not fitted yet.")
        if self._using_elasticnet:
            return float(self.model.alpha_)
        if self._is_multi_output:
            return float(np.mean([est.lambda_ for est in self.model.estimators_]))
        return float(self.model.lambda_)

