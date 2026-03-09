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

import numpy as np
from typing import Optional
from catboost import CatBoostRegressor

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

        self.model: Optional[CatBoostRegressor] = None
        self.feature_importance_: Optional[np.ndarray] = None
        self._n_outputs: int = 1
        self._fitted: bool = False

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

        # MultiRMSE for joint multi-output; RMSE for single-output
        loss_function = "MultiRMSE" if self._n_outputs > 1 else "RMSE"

        # Sample-adaptive hyperparameter scaling (see class docstring)
        n_samples = X.shape[0]
        if n_samples < 200:
            eff_depth, eff_iter = 4, 100
        elif n_samples < 500:
            eff_depth, eff_iter = 5, 200
        else:
            eff_depth, eff_iter = self.depth, self.iterations

        self.model = CatBoostRegressor(
            iterations=eff_iter,
            depth=eff_depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function=loss_function,
            bootstrap_type="Bernoulli",    # required when subsample < 1.0
            subsample=self.subsample,
            boosting_type="Plain",         # standard gradient boosting;
                                           # temporal order enforced externally
                                           # by _PanelTemporalSplit
            random_seed=self.random_state,
            verbose=0,                     # suppress per-iteration console logs
            allow_writing_files=False,     # no catboost_info/ directory
        )
        self.model.fit(X, y)

        # PredictionValuesChange (CatBoost default): pooled across all
        # outputs for MultiRMSE → shape (n_features,).
        self.feature_importance_ = self.model.get_feature_importance()
        self._fitted = True
        return self

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
        if not self._fitted or self.model is None:
            raise RuntimeError(
                "CatBoostForecaster.predict() called before fit(). "
                "Call fit(X, y) first."
            )
        pred = self.model.predict(X)
        if pred.ndim == 1:
            return pred.reshape(-1, 1)
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
