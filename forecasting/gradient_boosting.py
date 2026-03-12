# -*- coding: utf-8 -*-
"""
CatBoost Forecaster
===================

Joint multi-output gradient boosting via CatBoost's ``MultiRMSE`` loss.

Design rationale vs. sklearn GradientBoostingRegressor + MultiOutputRegressor
-----------------------------------------------------------------------------
*  **Joint multi-output (MultiRMSE)** — a single ``CatBoostRegressor`` fits
   all 8 criterion composites simultaneously through one shared oblivious-tree
   structure.  Splits are chosen to minimise total RMSE across *all* outputs,
   so cross-criterion correlations (provinces that rank high on infrastructure
   tend to rank high on economic criteria) are exploited automatically.
   ``sklearn.multioutput.MultiOutputRegressor`` trains N independent trees
   and cannot exploit these correlations.

*  **No feature scaling required** — CatBoost oblivious decision trees are
   invariant to monotone feature transformations; ``RobustScaler`` /
   ``StandardScaler`` pre-processing is unnecessary and has been removed.

*  **Plain boosting + Bernoulli bootstrap** — standard gradient boosting
   identical in structure to XGBoost / LightGBM.  CatBoost *Ordered* boosting
   was considered but is not used here because temporal ordering is already
   enforced by ``_PanelTemporalSplit`` and ``_WalkForwardYearlySplit``.

*  **allow_writing_files=False** — prevents CatBoost from creating
   ``catboost_info/`` directories and ``catboost_training.log`` files in
   the working directory during production runs.

References
----------
Prokhorenkova et al. (2018). "CatBoost: unbiased boosting with categorical
features." *Advances in Neural Information Processing Systems* 31.
"""

import warnings
import numpy as np
from typing import Optional

# ── Backend availability checks ──────────────────────────────────────────────
# CatBoostForecaster and LightGBMForecaster are independent ensemble members.
# Each requires its respective library to be installed; there is no fallback
# chain between them.
_CATBOOST_AVAILABLE: bool = False
_LIGHTGBM_AVAILABLE: bool = False

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

try:
    from lightgbm import LGBMRegressor as _LGBMRegressor  # type: ignore[import]
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LGBMRegressor = None  # type: ignore[assignment,misc]
    warnings.warn(
        "LightGBM not found. LightGBMForecaster will not be usable.\n"
        "Install LightGBM for leaf-wise multi-output support:\n"
        "    pip install lightgbm>=4.0.0",
        ImportWarning,
        stacklevel=2,
    )

from sklearn.multioutput import MultiOutputRegressor as _MultiOutputRegressor

from .base import BaseForecaster


class CatBoostForecaster(BaseForecaster):
    """
    CatBoost gradient boosting forecaster with joint multi-output regression.

    Trains a single ``CatBoostRegressor`` that predicts all criterion
    composites simultaneously via the **MultiRMSE** loss function.  The joint
    formulation allows tree splits to leverage cross-criterion correlation —
    a province that improves on C01 tends to improve on C04 — which
    per-output independent models cannot exploit.

    CatBoost requires no feature pre-scaling; its oblivious decision trees
    are invariant to monotone transformations of the input features.

    Sample-adaptive hyperparameter scaling
    ----------------------------------------
    Training-subset sizes vary across CV folds.  ``fit()`` automatically
    adjusts ``depth`` and ``iterations`` to prevent leaf underpopulation:

    +----------------+-------+------------+---------------------+
    | Training n     | depth | iterations | Min samples / leaf  |
    +================+=======+============+=====================+
    | < 200          |   4   |    100     |   ≥ 12 (n/16)       |
    | 200 – 499      |   5   |    200     |   ≥ 6  (n/32)       |
    | ≥ 500          |   6   |    300     |   ≥ 8  (n/64)       |
    +----------------+-------+------------+---------------------+

    Parameters
    ----------
    iterations : int
        Maximum boosting rounds.  Default 300 (targets n ≥ 500).
    depth : int
        Oblivious tree depth.  Default 6 (CatBoost standard default).
        Each level doubles the number of leaf clusters (2^depth).
    learning_rate : float
        Shrinkage factor.  Default 0.05 — lower than sklearn's 0.1
        default because ``l2_leaf_reg`` provides direct leaf regularisation.
    l2_leaf_reg : float
        L2 regularisation coefficient on leaf weight values (analogous
        to ``min_child_weight`` in XGBoost).  Default 3.0.
    subsample : float
        Fraction of training samples drawn without replacement per tree
        (Bernoulli bootstrap).  Default 0.8.
    random_state : int
        Random seed forwarded to CatBoost as ``random_seed``.

    Example
    -------
    >>> cb = CatBoostForecaster(iterations=300, depth=6)
    >>> cb.fit(X_train, y_train)           # y_train shape (n, 8)
    >>> preds = cb.predict(X_test)         # shape (n_test, 8)
    """

    def __init__(
        self,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        subsample: float = 0.8,
        random_state: int = 42,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.random_state = random_state

        # CatBoostRegressor | None (before fit)
        self.model = None
        self.feature_importance_: Optional[np.ndarray] = None
        self._n_outputs: int = 1
        self._fitted: bool = False
        # Constant-output bookkeeping (set during fit)
        self._const_mask_: Optional[np.ndarray] = None   # True = constant col
        self._const_vals_: Optional[np.ndarray] = None   # per-col fallback val

    # ------------------------------------------------------------------
    # Public API  (BaseForecaster interface)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostForecaster":
        """
        Fit the CatBoost model.

        Multi-dimensional ``y`` triggers ``MultiRMSE`` loss (joint
        multi-output); 1-D ``y`` triggers standard ``RMSE``.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.  No pre-scaling required.
        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        Returns
        -------
        self
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._n_outputs = y.shape[1]

        # Detect constant output columns (peak-to-peak ≈ 0) before dispatching
        # to the backend.  CatBoost raises "All train targets are equal" when
        # any output column has zero variance; other backends silently produce
        # a single-value leaf anyway, so we short-circuit all of them.
        self._const_vals_ = y[0].copy()
        self._const_mask_ = np.ptp(y, axis=0) < 1e-12
        if self._const_mask_.all():
            # Every output is constant — constant predictor, no model needed.
            self.feature_importance_ = np.zeros(X.shape[1])
            self._backend_ = 'none'
            self._fitted = True
            return self

        # Restrict training targets to non-constant columns only.
        y_fit = y[:, ~self._const_mask_] if self._const_mask_.any() else y

        # Sample-adaptive hyperparameter scaling (see class docstring table)
        n_samples = X.shape[0]
        if n_samples < 200:
            eff_depth, eff_iter = 4, 100
        elif n_samples < 500:
            eff_depth, eff_iter = 5, 200
        else:
            eff_depth, eff_iter = self.depth, self.iterations

        if not _CATBOOST_AVAILABLE:
            raise RuntimeError(
                "CatBoostForecaster requires CatBoost to be installed.\n"
                "Install it with: pip install catboost>=1.2.0"
            )
        # _n_outputs must reflect the model's actual output count (may be
        # smaller than the original if some columns are constant).
        self._n_outputs = y_fit.shape[1]

        self._fit_catboost(X, y_fit, eff_depth, eff_iter)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Backend-specific fit helper
    # ------------------------------------------------------------------

    def _fit_catboost(
        self, X: np.ndarray, y: np.ndarray, eff_depth: int, eff_iter: int
    ) -> None:
        """
        Fit ``CatBoostRegressor`` with joint MultiRMSE (multi-output) or RMSE.

        MultiRMSE fits all outputs simultaneously through a single shared
        oblivious-tree structure, exploiting cross-criterion correlation.
        RMSE is used for single-output targets.
        """
        loss_function = "MultiRMSE" if self._n_outputs > 1 else "RMSE"
        self.model = _CatBoostRegressor(
            iterations=eff_iter,
            depth=eff_depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function=loss_function,
            bootstrap_type="Bernoulli",    # required when subsample < 1.0
            subsample=self.subsample,
            boosting_type="Plain",         # standard gradient boosting;
                                           # temporal ordering enforced
                                           # externally by _PanelTemporalSplit
            random_seed=self.random_state,
            verbose=0,                     # suppress per-iteration logs
            allow_writing_files=False,     # no catboost_info/ directory
        )
        self.model.fit(X, y)
        # PredictionValuesChange pooled across all outputs for MultiRMSE
        # → shape (n_features,) as required by inverse_importance().
        self.feature_importance_ = self.model.get_feature_importance()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make point predictions.

        Always returns a 2-D array so ``SuperLearner`` can index
        predictions uniformly across all base models.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        ndarray, shape (n_samples, n_outputs)
            CatBoost returns ``(n_samples, n_outputs)`` for MultiRMSE
            and ``(n_samples,)`` for RMSE; both are normalised to 2-D.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self._fitted:
            raise RuntimeError(
                "CatBoostForecaster.predict() called before fit(). "
                "Call fit(X, y) first."
            )
        n_test = X.shape[0]
        # All outputs were constant during training — return stored constants.
        if self.model is None:
            return np.tile(self._const_vals_, (n_test, 1))
        pred = self.model.predict(X)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        # Reconstruct full output array when some columns were held constant.
        if self._const_mask_ is not None and self._const_mask_.any():
            n_orig = len(self._const_vals_)
            full = np.empty((n_test, n_orig), dtype=pred.dtype)
            full[:, self._const_mask_] = self._const_vals_[self._const_mask_]
            full[:, ~self._const_mask_] = pred
            return full
        return pred

    def get_feature_importance(self) -> np.ndarray:
        """
        Return the pooled feature importance vector.

        For ``MultiRMSE``, CatBoost aggregates ``PredictionValuesChange``
        across all output dimensions, yielding a single 1-D vector of
        shape ``(n_features,)``.  This is broadcast across all output
        columns in ``_get_per_output_importance`` (unified.py).

        Returns
        -------
        ndarray, shape (n_features,)

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self.feature_importance_ is None:
            raise RuntimeError(
                "CatBoostForecaster.get_feature_importance() called before "
                "fit().  Call fit(X, y) first."
            )
        return self.feature_importance_


class LightGBMForecaster(BaseForecaster):
    """
    LightGBM gradient boosting forecaster with leaf-wise multi-output regression.

    Trains independent per-output ``LGBMRegressor`` models wrapped by
    ``MultiOutputRegressor``, using LightGBM's leaf-wise tree growth strategy.
    As an independent ensemble member alongside ``CatBoostForecaster``, it
    provides a complementary inductive bias:

    *  **Leaf-wise growth** — LightGBM grows the leaf with the maximum loss
       reduction at each step, producing asymmetric trees that fit complex
       interactions more efficiently at the same leaf count compared to
       CatBoost's symmetric oblivious trees.
    *  **Per-output independence** — fits 8 separate ``LGBMRegressor``
       instances; each criterion is optimised independently, complementing
       CatBoost's joint ``MultiRMSE`` formulation that exploits
       cross-criterion correlation.

    Together the two gradient-boosting members give the Super Learner
    maximum ensemble diversity from the tree-based model family.

    Sample-adaptive hyperparameter scaling
    ----------------------------------------
    +----------------+-------+------------+---------------------+
    | Training n     | depth | iterations | Min samples / leaf  |
    +================+=======+============+=====================+
    | < 200          |   4   |    100     |   ≥ 12 (n/16)       |
    | 200 – 499      |   5   |    200     |   ≥ 6  (n/32)       |
    | ≥ 500          |   6   |    300     |   ≥ 8  (n/64)       |
    +----------------+-------+------------+---------------------+

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.  Default 300.
    max_depth : int
        Maximum tree depth.  Default 6.  Controls ``num_leaves`` via
        ``min(2^max_depth, 64)``.
    learning_rate : float
        Shrinkage factor.  Default 0.05.
    l2_reg : float
        L2 regularisation coefficient (``reg_lambda``).  Default 3.0.
    subsample : float
        Fraction of training rows sampled per tree (Bernoulli).  Default 0.8.
    random_state : int
        Random seed.

    Example
    -------
    >>> lgbm = LightGBMForecaster(n_estimators=300, max_depth=6)
    >>> lgbm.fit(X_train, y_train)           # y_train shape (n, 8)
    >>> preds = lgbm.predict(X_test)         # shape (n_test, 8)
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        l2_reg: float = 3.0,
        subsample: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.subsample = subsample
        self.random_state = random_state

        # MultiOutputRegressor(LGBMRegressor) | None (before fit)
        self.model = None
        self.feature_importance_: Optional[np.ndarray] = None
        self._n_outputs: int = 1
        self._fitted: bool = False
        # Constant-output bookkeeping (set during fit)
        self._const_mask_: Optional[np.ndarray] = None   # True = constant col
        self._const_vals_: Optional[np.ndarray] = None   # per-col fallback val

    # ------------------------------------------------------------------
    # Public API  (BaseForecaster interface)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMForecaster":
        """
        Fit independent per-output LightGBM models.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.  No pre-scaling required.
        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        Returns
        -------
        self
        """
        if not _LIGHTGBM_AVAILABLE:
            raise RuntimeError(
                "LightGBMForecaster requires LightGBM to be installed.\n"
                "Install it with: pip install lightgbm>=4.0.0"
            )
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._n_outputs = y.shape[1]

        # Detect constant output columns before fitting (avoids degenerate trees).
        self._const_vals_ = y[0].copy()
        self._const_mask_ = np.ptp(y, axis=0) < 1e-12
        if self._const_mask_.all():
            self.feature_importance_ = np.zeros(X.shape[1])
            self._fitted = True
            return self

        y_fit = y[:, ~self._const_mask_] if self._const_mask_.any() else y

        # Sample-adaptive hyperparameter scaling (see class docstring table)
        n_samples = X.shape[0]
        if n_samples < 200:
            eff_depth, eff_iter = 4, 100
        elif n_samples < 500:
            eff_depth, eff_iter = 5, 200
        else:
            eff_depth, eff_iter = self.max_depth, self.n_estimators

        # _n_outputs must reflect the model's actual output count (may be
        # smaller than the original if some columns are constant).
        self._n_outputs = y_fit.shape[1]

        self._fit_lightgbm(X, y_fit, eff_depth, eff_iter)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Backend-specific fit helper
    # ------------------------------------------------------------------

    def _fit_lightgbm(
        self, X: np.ndarray, y: np.ndarray, eff_depth: int, eff_iter: int
    ) -> None:
        """Fit ``MultiOutputRegressor(LGBMRegressor)`` — independent per output."""
        lgbm = _LGBMRegressor(
            n_estimators=eff_iter,
            max_depth=eff_depth,
            num_leaves=min(2 ** eff_depth, 64),  # leaf-wise complexity cap
            learning_rate=self.learning_rate,
            reg_lambda=self.l2_reg,
            subsample=self.subsample,
            subsample_freq=1,           # required to activate row subsampling
            random_state=self.random_state,
            verbose=-1,
            n_jobs=1,                   # deterministic; avoids joblib deadlocks
        )
        self.model = _MultiOutputRegressor(lgbm, n_jobs=1)
        self.model.fit(X, y)
        # Average gain-based importance across per-output LightGBM models
        self.feature_importance_ = np.mean(
            [est.feature_importances_ for est in self.model.estimators_], axis=0
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make point predictions.

        Always returns a 2-D array so ``SuperLearner`` can index
        predictions uniformly across all base models.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        ndarray, shape (n_samples, n_outputs)

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self._fitted:
            raise RuntimeError(
                "LightGBMForecaster.predict() called before fit(). "
                "Call fit(X, y) first."
            )
        n_test = X.shape[0]
        # All outputs were constant during training — return stored constants.
        if self.model is None:
            return np.tile(self._const_vals_, (n_test, 1))
        pred = self.model.predict(X)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        # Reconstruct full output array when some columns were held constant.
        if self._const_mask_ is not None and self._const_mask_.any():
            n_orig = len(self._const_vals_)
            full = np.empty((n_test, n_orig), dtype=pred.dtype)
            full[:, self._const_mask_] = self._const_vals_[self._const_mask_]
            full[:, ~self._const_mask_] = pred
            return full
        return pred

    def get_feature_importance(self) -> np.ndarray:
        """
        Return averaged feature importance across per-output LightGBM models.

        Importance values are gain-based (split gain summed over trees),
        averaged across the 8 per-criterion ``LGBMRegressor`` estimators.

        Returns
        -------
        ndarray, shape (n_features,)

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self.feature_importance_ is None:
            raise RuntimeError(
                "LightGBMForecaster.get_feature_importance() called before "
                "fit().  Call fit(X, y) first."
            )
        return self.feature_importance_
