# -*- coding: utf-8 -*-
"""
CatBoost / LightGBM Forecasters
================================

Joint multi-output gradient boosting via CatBoost's ``MultiRMSE`` loss, and
independent per-output gradient boosting via LightGBM's leaf-wise trees.

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

Phase 2 Enhancement: Chronological Early Stopping
--------------------------------------------------
Both CatBoostForecaster and LightGBMForecaster now support **chronological
early stopping** to prevent overfitting on small cross-validation folds.

Problem: with n_train ≈ 150–500, training 200–300 boosting rounds causes
the model to memorise fold-specific noise, producing negative CV R²
(LightGBM: −0.07, QRF: −0.088 per the audit).

Solution: hold out the last ``validation_fraction`` of each fold's training
rows — which are chronologically the most recent year-cohorts, because rows
are sorted ``(year, entity)`` by the feature engineer — as an internal
validation set.  Early stopping halts when validation loss fails to improve
for ``early_stopping_rounds`` consecutive rounds.

**Temporal integrity**: because the monitor set is drawn from the *tail* of
the training window rather than randomly, it always represents years *later*
than the training region, consistently with the walk-forward CV philosophy.

Sample-adaptive hyperparameter scaling
----------------------------------------
Training-subset sizes vary across CV folds.  ``fit()`` automatically
adjusts ``depth``, ``iterations``, and ``min_data_in_leaf`` to prevent
leaf underpopulation and excessive model capacity:

+----------------+-------+------------+--------------------+
| Training n     | depth | iterations | min_data_in_leaf   |
+================+=======+============+====================+
| < 200          |   3   |    100     | max(8,  n // 16)   |
| 200 – 399      |   4   |    150     | max(5,  n // 32)   |
| 400 – 599      |   5   |    200     | max(4,  n // 64)   |
| ≥ 600          |   6   |    300     | max(4,  n // 64)   |
+----------------+-------+------------+--------------------+

References
----------
Prokhorenkova et al. (2018). "CatBoost: unbiased boosting with categorical
features." *Advances in Neural Information Processing Systems* 31.

Ke et al. (2017). "LightGBM: A highly efficient gradient boosting decision
tree." *Advances in Neural Information Processing Systems* 30.
"""

import warnings
import logging
import numpy as np
from typing import List, Optional

try:
    import pandas as pd  # type: ignore[import]
except ImportError:
    pd = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

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
    import lightgbm as lgb  # type: ignore[import]
    from lightgbm import LGBMRegressor as _LGBMRegressor  # type: ignore[import]
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None  # type: ignore[assignment]
    _LGBMRegressor = None  # type: ignore[assignment,misc]
    warnings.warn(
        "LightGBM not found. LightGBMForecaster will not be usable.\n"
        "Install LightGBM for leaf-wise multi-output support:\n"
        "    pip install lightgbm>=4.0.0",
        ImportWarning,
        stacklevel=2,
    )

from .base import BaseForecaster


# ---------------------------------------------------------------------------
# Internal helper: chronological early-stopping split
# ---------------------------------------------------------------------------

def _chronological_es_split(
    X,
    y,
    validation_fraction: float,
    sample_weight,
    min_n_train: int = 30,
    min_n_val: int = 10,
) -> tuple:
    """Split an (X, y) training block into train and early-stopping val sets.

    Rows are assumed to be ordered chronologically ``(year ASC, entity ASC)``
    by the feature engineer, so slicing the *last* n_val rows always produces
    the most-recent year-cohorts.  This preserves temporal integrity: the
    validation set lies strictly *after* the training window, consistent with
    walk-forward cross-validation.

    Parameters
    ----------
    X : ndarray, shape (n, p)
    y : ndarray, shape (n,) or (n, k)
    validation_fraction : float
        Fraction of rows to hold out as validation.
    sample_weight : ndarray or None
    min_n_train : int
        Minimum rows that must remain in the training split.
        If splitting would leave fewer rows, returns None (skip early stop).
    min_n_val : int
        Minimum rows required in the validation split.
        If the computed n_val < min_n_val, returns None.

    Returns
    -------
    tuple of (X_tr, y_tr, sw_tr, X_va, y_va) or None
        None signals that the dataset is too small for early stopping.
    """
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


# ---------------------------------------------------------------------------
# CatBoostForecaster
# ---------------------------------------------------------------------------

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

    Phase 2 Early Stopping
    -----------------------
    When ``early_stopping_rounds > 0`` and ``n_train ≥ 40``, fits on the
    chronological training split (first 80%) and monitors validation loss on
    the chronological holdout (last 20%).  CatBoost reverts weights to the
    best-iteration checkpoint automatically (``use_best_model=True``).
    Falls back to full-iteration training for tiny folds (n_train < 40).

    Sample-adaptive hyperparameter scaling
    ----------------------------------------
    +----------------+-------+------------+--------------------+
    | Training n     | depth | iterations | min_data_in_leaf   |
    +================+=======+============+====================+
    | < 200          |   3   |    100     | max(8,  n // 16)   |
    | 200 – 399      |   4   |    150     | max(5,  n // 32)   |
    | 400 – 599      |   5   |    200     | max(4,  n // 64)   |
    | ≥ 600          |   6   |    300     | max(4,  n // 64)   |
    +----------------+-------+------------+--------------------+

    Parameters
    ----------
    iterations : int
        Maximum boosting rounds (upper bound with early stopping enabled).
    depth : int
        Oblivious tree depth.  Default 6 (used only when n ≥ 600).
    learning_rate : float
        Shrinkage factor.  Default 0.05.
    l2_leaf_reg : float
        L2 regularisation on leaf weights.  Default 3.0.
    subsample : float
        Bernoulli row-subsampling fraction.  Default 0.8.
    early_stopping_rounds : int
        Consecutive rounds with no improvement before halting.
        Set to 0 to disable.  Default 20.
    validation_fraction : float
        Fraction of training rows used as chronological ES validation.
        Default 0.20.
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

        # CatBoostRegressor | None (before fit)
        self.model = None
        self.feature_importance_: Optional[np.ndarray] = None
        self._n_outputs: int = 1
        self._fitted: bool = False
        # Constant-output bookkeeping (set during fit)
        self._const_mask_: Optional[np.ndarray] = None   # True = constant col
        self._const_vals_: Optional[np.ndarray] = None   # per-col fallback val
        # Early stopping diagnostics (set during fit when ES fires)
        self._best_iteration_: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API  (BaseForecaster interface)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> "CatBoostForecaster":
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
        sample_weight : ndarray, shape (n_samples,), optional
            Per-sample importance weights (e.g., from
            ``PanelCovariateShiftDetector``).  ``None`` = uniform weights.

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
        # _n_outputs must reflect the model's actual output count (may be
        # smaller than the original if some columns are constant).
        self._n_outputs = y_fit.shape[1]

        self._fit_catboost(X, y_fit, eff_depth, eff_iter, eff_min_leaf,
                           sample_weight=sample_weight)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Backend-specific fit helper
    # ------------------------------------------------------------------

    def _fit_catboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eff_depth: int,
        eff_iter: int,
        eff_min_leaf: int,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit ``CatBoostRegressor`` with joint MultiRMSE (multi-output) or RMSE.

        Implements chronological early stopping (Phase 2.1): when enabled and
        n_train ≥ 40, holds out the last ``validation_fraction`` of training
        rows as a temporal evaluation set.  Smaller datasets fall back to
        full-iteration training without early stopping.

        Parameters
        ----------
        X : ndarray
        y : ndarray (non-constant columns only)
        eff_depth : int
            Sample-adaptive tree depth.
        eff_iter : int
            Maximum boosting rounds (early stopping may halt before this).
        eff_min_leaf : int
            Minimum training samples per leaf (sample-adaptive).
        sample_weight : ndarray or None
        """
        loss_function = "MultiRMSE" if self._n_outputs > 1 else "RMSE"

        # ── Phase 2.1: Chronological early-stopping setup ─────────────────
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
                # Dataset too small — disable ES for this fold
                use_es = False

        self.model = _CatBoostRegressor(
            iterations=eff_iter,
            depth=eff_depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            min_data_in_leaf=eff_min_leaf,
            loss_function=loss_function,
            bootstrap_type="Bernoulli",    # required when subsample < 1.0
            subsample=self.subsample,
            boosting_type="Plain",         # standard gradient boosting;
                                           # temporal ordering enforced
                                           # externally by _PanelTemporalSplit
            random_seed=self.random_state,
            verbose=0,                     # suppress per-iteration logs
            allow_writing_files=False,     # no catboost_info/ directory
            use_best_model=use_es,         # revert to best checkpoint when ES
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
                # Some CatBoost versions / loss functions don't accept
                # sample_weight; fall back to unweighted fit.
                self.model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
            # Record best iteration for diagnostics
            try:
                self._best_iteration_ = int(self.model.get_best_iteration())
                logger.debug(
                    "CatBoost ES: best_iteration=%d / max=%d",
                    self._best_iteration_, eff_iter,
                )
            except Exception:
                pass
        else:
            # Full-iteration training (no ES or dataset too small)
            try:
                self.model.fit(X, y, sample_weight=sample_weight)
            except TypeError:
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
        # Handle 0-D scalars and ensure 2-D shape
        if np.ndim(pred) == 0:
            pred = np.asarray([[pred[()]]] * n_test)
        elif pred.ndim == 1:
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


# ---------------------------------------------------------------------------
# LightGBMForecaster
# ---------------------------------------------------------------------------

class LightGBMForecaster(BaseForecaster):
    """
    LightGBM gradient boosting forecaster with leaf-wise multi-output regression.

    Trains independent per-output ``LGBMRegressor`` models, using LightGBM's
    leaf-wise tree growth strategy.  As an independent ensemble member
    alongside ``CatBoostForecaster``, it provides a complementary inductive
    bias:

    *  **Leaf-wise growth** — LightGBM grows the leaf with the maximum loss
       reduction at each step, producing asymmetric trees that fit complex
       interactions more efficiently at the same leaf count compared to
       CatBoost's symmetric oblivious trees.
    *  **Per-output independence** — fits separate ``LGBMRegressor``
       instances; each criterion is optimised independently, complementing
       CatBoost's joint ``MultiRMSE`` formulation that exploits
       cross-criterion correlation.

    Together the two gradient-boosting members give the Super Learner
    maximum ensemble diversity from the tree-based model family.

    Phase 2 Early Stopping
    -----------------------
    When ``early_stopping_rounds > 0`` and ``n_train ≥ 40``, each per-output
    ``LGBMRegressor`` uses the chronological holdout (last 20% of rows) as
    its ``eval_set``.  LightGBM's callback mechanism halts training at the
    round with minimum validation loss and restores the best model weights.
    Falls back to full-iteration training for tiny folds (n_train < 40).

    Why per-estimator instead of MultiOutputRegressor?
    ``sklearn.multioutput.MultiOutputRegressor`` does not reliably propagate
    ``callbacks`` and ``eval_set`` to nested estimators across sklearn
    versions.  We fit each output directly and store estimators in
    ``self._estimators_`` (also exposed via ``self.model.estimators_`` for
    API compatibility with existing tests and ``_get_per_output_importance``).

    Sample-adaptive hyperparameter scaling
    ----------------------------------------
    +----------------+-------+------------+--------------------+
    | Training n     | depth | iterations | min_child_samples  |
    +================+=======+============+====================+
    | < 200          |   3   |    100     | max(8,  n // 16)   |
    | 200 – 399      |   4   |    150     | max(5,  n // 32)   |
    | 400 – 599      |   5   |    200     | max(4,  n // 64)   |
    | ≥ 600          |   6   |    300     | max(4,  n // 64)   |
    +----------------+-------+------------+--------------------+

    Parameters
    ----------
    n_estimators : int
        Maximum boosting rounds (upper bound with early stopping).
    max_depth : int
        Maximum tree depth.  Controls ``num_leaves`` via
        ``min(2^max_depth, 64)``.  Default 6 (used only when n ≥ 600).
    learning_rate : float
        Shrinkage factor.  Default 0.05.
    l2_reg : float
        L2 regularisation coefficient (``reg_lambda``).  Default 3.0.
    subsample : float
        Bernoulli row-subsampling fraction.  Default 0.8.
    early_stopping_rounds : int
        Consecutive rounds with no improvement before halting.
        Set to 0 to disable.  Default 20.
    validation_fraction : float
        Fraction of training rows used as chronological ES validation.
        Default 0.20.
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
        early_stopping_rounds: int = 20,
        validation_fraction: float = 0.20,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.subsample = subsample
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_state = random_state

        # Per-output estimators + compatibility wrapper | None (before fit)
        self._estimators_: List[Optional[object]] = []
        self.model = None   # _LGBMCompatWrapper | None
        self.feature_importance_: Optional[np.ndarray] = None
        self._n_outputs: int = 1
        self._fitted: bool = False
        # Constant-output bookkeeping (set during fit)
        self._const_mask_: Optional[np.ndarray] = None   # True = constant col
        self._const_vals_: Optional[np.ndarray] = None   # per-col fallback val
        # Early stopping diagnostics
        self._best_iterations_: List[int] = []
        # If fit with a DataFrame, preserve feature names for ndarray inference.
        self._feature_names_: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Public API  (BaseForecaster interface)
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "LightGBMForecaster":
        """
        Fit independent per-output LightGBM models with chronological ES.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.  No pre-scaling required.
        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            Target values.
        sample_weight : ndarray, shape (n_samples,), optional
            Per-sample importance weights.  ``None`` = uniform weights.

        Returns
        -------
        self
        """
        if not _LIGHTGBM_AVAILABLE:
            raise RuntimeError(
                "LightGBMForecaster requires LightGBM to be installed.\n"
                "Install it with: pip install lightgbm>=4.0.0"
            )

        columns = getattr(X, 'columns', None)
        self._feature_names_ = [str(c) for c in columns] if columns is not None else None

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._n_outputs = y.shape[1]

        # Detect constant output columns before fitting (avoids degenerate trees).
        self._const_vals_ = y[0].copy()
        self._const_mask_ = np.ptp(y, axis=0) < 1e-12
        if self._const_mask_.all():
            self.feature_importance_ = np.zeros(X.shape[1])
            self._estimators_ = []
            self.model = _LGBMCompatWrapper([])
            self._fitted = True
            return self

        y_fit = y[:, ~self._const_mask_] if self._const_mask_.any() else y

        # Sample-adaptive hyperparameter scaling
        n_samples = X.shape[0]
        if n_samples < 200:
            eff_depth, eff_iter, eff_min_child = 3, 100, max(8, n_samples // 16)
        elif n_samples < 400:
            eff_depth, eff_iter, eff_min_child = 4, 150, max(5, n_samples // 32)
        elif n_samples < 600:
            eff_depth, eff_iter, eff_min_child = 5, 200, max(4, n_samples // 64)
        else:
            eff_depth, eff_iter, eff_min_child = self.max_depth, self.n_estimators, max(4, n_samples // 64)

        # _n_outputs must reflect the model's actual output count (may be
        # smaller than the original if some columns are constant).
        self._n_outputs = y_fit.shape[1]

        self._fit_lightgbm(X, y_fit, eff_depth, eff_iter, eff_min_child,
                           sample_weight=sample_weight)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Backend-specific fit helper
    # ------------------------------------------------------------------

    def _fit_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eff_depth: int,
        eff_iter: int,
        eff_min_child: int,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Fit per-output LGBMRegressors with optional chronological ES.

        Fits one ``LGBMRegressor`` per output column directly, stored in
        ``self._estimators_``.  A lightweight compatibility wrapper
        ``_LGBMCompatWrapper`` exposes ``estimators_`` for code paths in
        unified.py that iterate over model components for feature importance.

        Parameters
        ----------
        sample_weight : ndarray, shape (n_samples,), optional
            Forwarded to each LGBMRegressor.fit().
        """
        n_outputs = y.shape[1]
        n_samples = X.shape[0]

        # ── Phase 2.1: Chronological early-stopping setup ─────────────────
        use_es = self.early_stopping_rounds > 0 and _LIGHTGBM_AVAILABLE
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

        if use_es:
            X_tr, y_tr, sw_tr, X_va, y_va = es_split  # type: ignore[misc]
        else:
            X_tr, y_tr, sw_tr = X, y, sample_weight
            X_va, y_va = None, None

        self._estimators_ = []
        self._best_iterations_ = []

        for col in range(n_outputs):
            lgbm = _LGBMRegressor(
                n_estimators=eff_iter,
                max_depth=eff_depth,
                num_leaves=min(2 ** eff_depth, 64),  # leaf-wise complexity cap
                learning_rate=self.learning_rate,
                reg_lambda=self.l2_reg,
                min_child_samples=eff_min_child,
                subsample=self.subsample,
                subsample_freq=1,           # required to activate row subsampling
                random_state=self.random_state,
                verbose=-1,
                n_jobs=1,                   # deterministic; avoids joblib deadlocks
            )
            if use_es:
                callbacks = [
                    lgb.early_stopping(
                        stopping_rounds=self.early_stopping_rounds,
                        verbose=False,
                    ),
                    lgb.log_evaluation(period=-1),
                ]
                lgbm.fit(
                    X_tr, y_tr[:, col],
                    eval_set=[(X_va, y_va[:, col])],
                    callbacks=callbacks,
                    sample_weight=sw_tr,
                )
                best_iter = getattr(lgbm, 'best_iteration_', eff_iter)
                self._best_iterations_.append(best_iter)
                logger.debug(
                    "LightGBM output=%d ES: best_iter=%d / max=%d",
                    col, best_iter, eff_iter,
                )
            else:
                lgbm.fit(X_tr, y_tr[:, col], sample_weight=sw_tr)
                self._best_iterations_.append(eff_iter)
            self._estimators_.append(lgbm)

        # Compatibility wrapper: exposes .estimators_ for _get_per_output_importance
        self.model = _LGBMCompatWrapper(self._estimators_)

        # Average gain-based importance across per-output LightGBM models
        importances = []
        for est in self._estimators_:
            fi = getattr(est, 'feature_importances_', None)
            if fi is not None:
                importances.append(fi)
        if importances:
            self.feature_importance_ = np.mean(importances, axis=0)
        else:
            self.feature_importance_ = np.zeros(X.shape[1])

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
        if not self._estimators_:
            return np.tile(self._const_vals_, (n_test, 1))

        X_model = X
        if not hasattr(X, 'columns') and pd is not None:
            feature_names = self._feature_names_
            if feature_names is None and self._estimators_:
                first_est = self._estimators_[0]
                est_names = getattr(first_est, 'feature_name_', None)
                if est_names is None:
                    booster = getattr(first_est, 'booster_', None)
                    if booster is not None and hasattr(booster, 'feature_name'):
                        est_names = booster.feature_name()
                if est_names is not None:
                    feature_names = [str(c) for c in est_names]

            if feature_names is not None:
                if X.shape[1] != len(feature_names):
                    raise ValueError(
                        "LightGBMForecaster.predict() received input with "
                        f"{X.shape[1]} features, expected {len(feature_names)}"
                    )
                X_model = pd.DataFrame(X, columns=feature_names)

        cols = [est.predict(X_model) for est in self._estimators_]
        pred = np.column_stack(cols)

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
        averaged across the per-criterion ``LGBMRegressor`` estimators.

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


# ---------------------------------------------------------------------------
# _LGBMCompatWrapper
# ---------------------------------------------------------------------------

class _LGBMCompatWrapper:
    """Thin wrapper to expose per-output LGBMRegressor estimators_list.

    ``_get_per_output_importance`` in unified.py checks for
    ``model.model.estimators_`` to extract per-output feature importances.
    Previously this was a ``MultiOutputRegressor``; now that we fit
    LGBMRegressors directly, this wrapper replicates the ``.estimators_``
    interface without the sklearn overhead.

    Also exposes ``.predict()`` routing to individual estimators for any
    code path that calls ``model.predict()`` directly on the wrapper.
    """

    def __init__(self, estimators: list):
        self.estimators_ = estimators

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("_LGBMCompatWrapper has no estimators.")
        cols = [est.predict(X) for est in self.estimators_]
        return np.column_stack(cols)
