# -*- coding: utf-8 -*-
"""
CatBoost Forecaster
===================

Joint multi-output gradient boosting via CatBoost MultiRMSE.

Phase 2 enhancement: chronological early stopping is applied by reserving
the most recent validation_fraction of each training fold as a temporal
holdout monitor set.
"""

import warnings
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Backend availability check
_CATBOOST_AVAILABLE: bool = False

try:
    from catboost import CatBoostRegressor as _CatBoostRegressor  # type: ignore[import]
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CatBoostRegressor = None  # type: ignore[assignment,misc]
    warnings.warn(
        "CatBoost not found. CatBoostForecaster will not be usable.\n"
        "Install CatBoost for joint multi-output (MultiRMSE) support:\n"
        "    pip install catboost>=1.2.0",
        ImportWarning,
        stacklevel=2,
    )

from .base import BaseForecaster


def _chronological_es_split(
    X,
    y,
    validation_fraction: float,
    sample_weight,
    min_n_train: int = 30,
    min_n_val: int = 10,
) -> tuple:
    """Split an (X, y) training block into train and early-stopping val sets."""
    n = X.shape[0]
    n_val = max(min_n_val, int(round(n * validation_fraction)))
    n_tr = n - n_val

    if n_tr < min_n_train or n_val < min_n_val:
        return None

    def _slice_rows(arr, start: int, end: Optional[int] = None):
        if hasattr(arr, "iloc"):
            return arr.iloc[start:end]
        return arr[start:end]

    X_tr, X_va = _slice_rows(X, 0, n_tr), _slice_rows(X, n_tr, None)
    y_tr, y_va = _slice_rows(y, 0, n_tr), _slice_rows(y, n_tr, None)
    sw_tr = _slice_rows(sample_weight, 0, n_tr) if sample_weight is not None else None
    return X_tr, y_tr, sw_tr, X_va, y_va


class CatBoostForecaster(BaseForecaster):
    """CatBoost gradient boosting forecaster with joint multi-output regression."""

    def __init__(
        self,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        subsample: float = 0.8,
        early_stopping_rounds: int = 20,
        validation_fraction: float = 0.20,
        random_state: int = 42,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_state = random_state

        self.model = None
        self.feature_importance_: Optional[np.ndarray] = None
        self._n_outputs: int = 1
        self._fitted: bool = False
        self._const_mask_: Optional[np.ndarray] = None
        self._const_vals_: Optional[np.ndarray] = None
        self._best_iteration_: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> "CatBoostForecaster":
        """Fit the CatBoost model."""
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._n_outputs = y.shape[1]

        self._const_vals_ = y[0].copy()
        self._const_mask_ = np.ptp(y, axis=0) < 1e-12
        if self._const_mask_.all():
            self.feature_importance_ = np.zeros(X.shape[1])
            self._backend_ = 'none'
            self._fitted = True
            return self

        y_fit = y[:, ~self._const_mask_] if self._const_mask_.any() else y

        n_samples = X.shape[0]
        if n_samples < 200:
            eff_depth, eff_iter, eff_min_leaf = 3, 100, max(8, n_samples // 16)
        elif n_samples < 400:
            eff_depth, eff_iter, eff_min_leaf = 4, 150, max(5, n_samples // 32)
        elif n_samples < 600:
            eff_depth, eff_iter, eff_min_leaf = 5, 200, max(4, n_samples // 64)
        else:
            eff_depth, eff_iter, eff_min_leaf = self.depth, self.iterations, max(4, n_samples // 64)

        if not _CATBOOST_AVAILABLE:
            raise RuntimeError(
                "CatBoostForecaster requires CatBoost to be installed.\n"
                "Install it with: pip install catboost>=1.2.0"
            )

        self._n_outputs = y_fit.shape[1]
        self._fit_catboost(X, y_fit, eff_depth, eff_iter, eff_min_leaf,
                           sample_weight=sample_weight)
        self._fitted = True
        return self

    def _fit_catboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eff_depth: int,
        eff_iter: int,
        eff_min_leaf: int,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Fit CatBoostRegressor with joint MultiRMSE (or RMSE)."""
        loss_function = "MultiRMSE" if self._n_outputs > 1 else "RMSE"

        use_es = self.early_stopping_rounds > 0
        es_split = None
        if use_es:
            es_split = _chronological_es_split(
                X, y,
                validation_fraction=self.validation_fraction,
                sample_weight=sample_weight,
                min_n_train=30,
                min_n_val=10,
            )
            if es_split is None:
                use_es = False

        self.model = _CatBoostRegressor(
            iterations=eff_iter,
            depth=eff_depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            min_data_in_leaf=eff_min_leaf,
            loss_function=loss_function,
            bootstrap_type="Bernoulli",
            subsample=self.subsample,
            boosting_type="Plain",
            random_seed=self.random_state,
            verbose=0,
            allow_writing_files=False,
            use_best_model=use_es,
            early_stopping_rounds=(
                self.early_stopping_rounds if use_es else None
            ),
        )

        if use_es:
            X_tr, y_tr, sw_tr, X_va, y_va = es_split  # type: ignore[misc]
            try:
                self.model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    sample_weight=sw_tr,
                )
            except TypeError:
                self.model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
            try:
                self._best_iteration_ = int(self.model.get_best_iteration())
                logger.debug(
                    "CatBoost ES: best_iteration=%d / max=%d",
                    self._best_iteration_, eff_iter,
                )
            except Exception:
                pass
        else:
            try:
                self.model.fit(X, y, sample_weight=sample_weight)
            except TypeError:
                self.model.fit(X, y)

        self.feature_importance_ = self.model.get_feature_importance()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make point predictions and always return a 2-D array."""
        if not self._fitted:
            raise RuntimeError(
                "CatBoostForecaster.predict() called before fit(). "
                "Call fit(X, y) first."
            )
        n_test = X.shape[0]
        if self.model is None:
            return np.tile(self._const_vals_, (n_test, 1))

        pred = self.model.predict(X)
        if np.ndim(pred) == 0:
            pred = np.asarray([[pred[()]]] * n_test)
        elif pred.ndim == 1:
            pred = pred.reshape(-1, 1)

        if self._const_mask_ is not None and self._const_mask_.any():
            n_orig = len(self._const_vals_)
            full = np.empty((n_test, n_orig), dtype=pred.dtype)
            full[:, self._const_mask_] = self._const_vals_[self._const_mask_]
            full[:, ~self._const_mask_] = pred
            return full
        return pred

    def get_feature_importance(self) -> np.ndarray:
        """Return pooled feature importance."""
        if self.feature_importance_ is None:
            raise RuntimeError(
                "CatBoostForecaster.get_feature_importance() called before "
                "fit(). Call fit(X, y) first."
            )
        return self.feature_importance_
