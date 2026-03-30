"""
CatBoost Forecaster for Joint Multi-Output Targets.

This module provides a `CatBoostForecaster` that implements joint 
multi-output gradient boosting via the CatBoost 'MultiRMSE' loss. 

Key Features
------------
- **Joint Multi-Output**: Optimizes a shared ensemble for multiple criterion 
  simultaneously, preserving inter-target correlations.
- **Temporal Early Stopping**: Reserves the most recent fraction of training 
  data as a monitor set to prevent overfitting.
- **Adaptive Hyperparameters**: Automatically scales tree depth, iterations, 
  and leaf constraints based on the size of the individual cross-validation 
  folds.

References
----------
- Prokhorenkova et al. (2018). "CatBoost: unbiased boosting with 
  categorical features." NeurIPS.
- CatBoost Documentation: https://catboost.ai/en/docs/concepts/loss-functions-regression#multirmse
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
        """
        Initialize the CatBoost forecaster.

        Parameters
        ----------
        iterations : int, default=300
            Maximum number of boosting iterations.
        depth : int, default=6
            Depth of the trees.
        learning_rate : float, default=0.05
            Boosting learning rate.
        l2_leaf_reg : float, default=3.0
            L2 regularization coefficient for the leaves.
        subsample : float, default=0.8
            Subsampling rate for Bernoulli bootstrap.
        early_stopping_rounds : int, default=20
            Stop training if validation loss does not improve for this many 
            rounds.
        validation_fraction : float, default=0.20
            Fraction of recent training data to reserve for early stopping.
        random_state : int, default=42
            Random seed for reproducibility.
        """
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

        # FIX #7 (TIER 3): Adaptive hyperparameter scaling based on fold size
        # Prevents overfitting on small folds and underfitting on large folds
        # by adjusting tree depth, number of iterations, and minimum leaf size.
        n_samples = X.shape[0]

        # Logarithmic depth scaling formula (from ACTION_PLAN TIER 3 FIX #7):
        # depth = max(3, min(7, int(np.log2(max(10, n_train / 5)))))
        # This scales depth from 3 (small folds) to 7 (large folds) based on
        # the effective training set size per tree (n/5 handles fold structure).
        adaptive_depth = max(3, min(7, int(np.log2(max(10, n_samples / 5)))))

        # Early stopping rounds scale with sqrt(n): fewer rounds for tiny folds,
        # more for large folds. Prevents premature stopping on n<100 and wasted
        # rounds on n>1000.
        adaptive_es_rounds = max(10, int(np.sqrt(n_samples / 25)))

        # Effective hyperparameters for this fold
        if n_samples < 200:
            # Very small fold: conservative model to prevent memorization
            eff_depth = min(3, adaptive_depth)
            eff_iter = 100
            eff_min_leaf = max(8, n_samples // 16)
        elif n_samples < 400:
            # Small fold: moderate complexity
            eff_depth = min(4, adaptive_depth)
            eff_iter = 150
            eff_min_leaf = max(5, n_samples // 32)
        elif n_samples < 600:
            # Medium fold: allow deeper trees
            eff_depth = min(5, adaptive_depth)
            eff_iter = 200
            eff_min_leaf = max(4, n_samples // 64)
        else:
            # Large fold: use adaptive depth + configured iterations
            eff_depth = adaptive_depth
            eff_iter = self.iterations
            eff_min_leaf = max(4, n_samples // 64)

        logger.debug(
            f"CatBoost adaptive scaling (n={n_samples}): "
            f"depth={eff_depth}, iter={eff_iter}, min_leaf={eff_min_leaf}"
        )

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
