# -*- coding: utf-8 -*-
"""
Quantile Regression Forests Forecaster
=======================================

Extends Random Forest to predict the full conditional distribution
rather than just the conditional mean. Provides distributional forecasts
with asymmetric prediction intervals and heteroscedastic uncertainty.

Built on ``sklearn_quantile.RandomForestQuantileRegressor``, which
implements the Meinshausen (2006) QRF algorithm at the C level —
10–100× faster than a pure-Python leaf-weight implementation and more
memory-efficient (no dense (n_test × n_train) weight matrix).

Instead of averaging tree predictions (point estimate), QRF retains
the full set of training observations that fall in each leaf, allowing
estimation of arbitrary quantiles of the conditional distribution.

Algorithm:
    1. Train Random Forest as usual
    2. For each test point x:
       - Identify all training samples y_i in the same leaf as x
       - Estimate quantiles from the empirical distribution of {y_i}
    3. Output: Q(τ|x) for τ ∈ {0.05, 0.10, ..., 0.95}

Point-prediction semantics:
    ``predict(X)``        → conditional **mean** (standard RF average)
                            Used by Super Learner meta-learner (MSE criterion)
    ``predict_median(X)`` → conditional **median** at q=0.5
    ``predict_mean(X)``   → identical to ``predict()``

Phase 2 Stabilization (Phase 2.4)
----------------------------------
Two changes eliminate the negative CV R² (−0.088 observed in the audit):

1. **RobustScaler removed** — ``RandomForestQuantileRegressor`` is a tree
   model and is invariant to monotone feature transformations.  The scaler
   added unnecessary computation, made feature importances less
   interpretable, and could introduce train-test distribution shift when
   the prediction set is out-of-distribution.  ``X`` is now passed directly.

2. **Adaptive min_samples_leaf** — with n_train ≈ 150–500, the default
   ``min_samples_leaf=3`` allows highly specific leaf nodes that memorise
   fold-specific noise.  ``fit()`` now auto-scales the leaf threshold:

   +-------------+---------------------+
   | n_train     | effective_min_leaf  |
   +=============+=====================+
   | < 200       | max(5, n // 20)     |
   | 200 – 399   | max(3, n // 30)     |
   | ≥ 400       | min_samples_leaf    |
   +-------------+---------------------+

   Auto-scaling only fires when ``min_samples_leaf`` retains the default
   value (3); explicit overrides are always honoured.

Key Advantages:
    - Non-parametric: No distributional assumptions
    - Heteroscedastic: Uncertainty varies with input
    - Asymmetric intervals: Captures skewness in predictions
    - Naturally calibrated: Well-calibrated uncertainty
    - No additional training cost beyond standard RF

References:
    - Meinshausen (2006). "Quantile Regression Forests" JMLR
    - Athey, Tibshirani & Wager (2019). "Generalized Random Forests"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# sklearn_quantile is listed as a core dependency in pyproject.toml
# (sklearn-quantile >= 0.0.22).  The try/except converts a cryptic
# ModuleNotFoundError at import time into a clear, actionable message.
try:
    from sklearn_quantile import RandomForestQuantileRegressor
    _SKLEARN_QUANTILE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_QUANTILE_AVAILABLE = False
    RandomForestQuantileRegressor = None  # type: ignore[assignment,misc]

from .base import BaseForecaster


class QuantileRandomForestForecaster(BaseForecaster):
    """
    Quantile Random Forest for distributional forecasting.

    Provides full predictive distributions via quantile estimation from
    the empirical distribution of training samples within tree leaves.

    Phase 2 changes:
    - **RobustScaler removed**: tree models are scale-invariant.
    - **Adaptive min_samples_leaf**: auto-scaled with n_train to prevent
      leaf memorisation on small CV folds.
    - ``n_estimators`` default raised to 200 (class) / 300 (when read from
      ``ForecastConfig.qrf_n_estimators``).

    Parameters:
        n_estimators: Number of trees in the forest (default 200 here;
            ``_create_models()`` in unified.py passes 300 from config).
        max_depth: Maximum depth of trees (None = unlimited).
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples in a leaf node (default 3).
            Auto-scaled with training set size when using default value.
        quantiles: Quantile levels to estimate.
        random_state: Random seed.
        n_jobs: Number of parallel jobs (-1 = all cores).

    Example:
        >>> qrf = QuantileRandomForestForecaster(n_estimators=300)
        >>> qrf.fit(X_train, y_train)
        >>> predictions = qrf.predict(X_test)
        >>> quantile_preds = qrf.predict_quantiles(X_test, quantiles=[0.05, 0.5, 0.95])
    """

    # Default min_samples_leaf sentinel — when the user passes exactly this
    # value, auto-scaling is enabled; any other value is treated as explicit
    # and respected verbatim.
    _DEFAULT_MIN_LEAF: int = 3

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 3,
        quantiles: List[float] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        if not _SKLEARN_QUANTILE_AVAILABLE:
            raise ImportError(
                "QuantileRandomForestForecaster requires the 'sklearn-quantile' package.\n"
                "Install it with:  pip install sklearn-quantile\n"
                "It is listed as a core dependency in pyproject.toml; if you are "
                "working in a custom environment ensure it is installed there."
            )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.quantiles = quantiles or [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model_: Optional[RandomForestQuantileRegressor] = None
        self._models_per_output_: List[RandomForestQuantileRegressor] = []
        # Phase 2.4: scaler_ retained as None for backward compatibility with
        # any pickled legacy models that inspected this attribute.
        self.scaler_ = None
        self.feature_importance_: Optional[np.ndarray] = None
        self._is_multi_output: bool = False
        self._n_outputs: int = 1

    def _compute_effective_min_leaf(self, n_samples: int) -> int:
        """Compute adaptive min_samples_leaf based on training set size.

        Auto-scaling is enabled only when ``min_samples_leaf`` retains its
        default sentinel value (3).  Explicit user-provided values bypass
        this logic and are always used as-is.

        +-------------+---------------------+
        | n_train     | effective_min_leaf  |
        +=============+=====================+
        | < 200       | max(5, n // 20)     |
        | 200 – 399   | max(3, n // 30)     |
        | ≥ 400       | min_samples_leaf    |
        +-------------+---------------------+
        """
        if self.min_samples_leaf != self._DEFAULT_MIN_LEAF:
            # Explicit override — honour without modification.
            return self.min_samples_leaf
        if n_samples < 200:
            return max(5, n_samples // 20)
        elif n_samples < 400:
            return max(3, n_samples // 30)
        else:
            return self.min_samples_leaf  # default is fine for larger datasets

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileRandomForestForecaster":
        """
        Fit the Quantile Random Forest.

        Trains a standard Random Forest and stores training data leaf
        assignments for subsequent quantile estimation.

        Phase 2.4: ``X`` is passed directly without scaling (tree models are
        scale-invariant).  ``min_samples_leaf`` is auto-scaled with n_train.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) or (n_samples, n_outputs)

        Returns:
            Self for method chaining
        """
        n_samples = X.shape[0]
        eff_min_leaf = self._compute_effective_min_leaf(n_samples)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._n_outputs = y.shape[1]
        self._is_multi_output = y.shape[1] > 1

        # RandomForestQuantileRegressor only supports single-output targets.
        # For multi-output (n_outputs > 1), fit one QRF per output column
        # independently.  We avoid MultiOutputRegressor because RFQR's
        # overridden predict() requires quantiles and is incompatible with
        # sklearn's generic MultiOutputRegressor.predict() delegation.
        self._models_per_output_ = []
        for col in range(self._n_outputs):
            qrf = RandomForestQuantileRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=eff_min_leaf,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                bootstrap=True,
                oob_score=True,
            )
            # Phase 2.4: pass X directly — no scaling for tree models.
            qrf.fit(X, y[:, col])
            self._models_per_output_.append(qrf)

        # For API compat, keep model_ pointing to the first estimator
        self.model_ = self._models_per_output_[0]

        # Average feature importance across per-output estimators
        self.feature_importance_ = np.mean(
            [est.feature_importances_ for est in self._models_per_output_],
            axis=0,
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make point predictions (conditional mean).

        Returns the RF conditional mean so that ``predict()`` is
        consistent with the MSE-based Super Learner meta-learner.
        Use :meth:`predict_quantiles` for distributional forecasting
        or :meth:`predict_median` for the MAE-optimal point estimate.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_outputs)
        """
        return self.predict_mean(X)

    def predict_median(self, X: np.ndarray) -> np.ndarray:
        """
        Return the conditional median (50th percentile).

        The median minimises MAE and is the natural point summary
        for distributional / quantile regression models.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_outputs)
        """
        try:
            q_preds = self.predict_quantiles(X, quantiles=[0.50])
            return q_preds[0.50]
        except Exception:
            return self.predict_mean(X)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """
        Return the conditional mean (standard RF prediction).

        Useful when the mean is preferred over the median, e.g. for
        squared-error scoring or comparison with other models.

        Phase 2.4: X is passed directly without scaling.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_outputs)
        """
        # Use the underlying sklearn RandomForestRegressor.predict (via super)
        # which returns the tree-averaged conditional mean.
        from sklearn.ensemble import RandomForestRegressor
        if self._is_multi_output:
            cols = [
                RandomForestRegressor.predict(est, X)
                for est in self._models_per_output_
            ]
            return np.column_stack(cols)
        return RandomForestRegressor.predict(self.model_, X)

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: Optional[List[float]] = None,
    ) -> Dict[float, np.ndarray]:
        """
        Predict specific quantiles of the conditional distribution.

        Delegates to ``RandomForestQuantileRegressor.predict()`` with a
        ``quantiles`` argument — all leaf-weight bookkeeping is handled
        internally by the C-level implementation, which is 10–100× faster
        than the previous manual NumPy loop.

        Phase 2.4: X is passed directly without scaling.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            quantiles: List of quantile levels (default: self.quantiles)

        Returns:
            Dictionary mapping quantile level to prediction array.
            Each array has shape (n_samples,) for single output or
            (n_samples, n_outputs) for multi-output.
        """
        if quantiles is None:
            quantiles = self.quantiles

        q_array = np.asarray(quantiles, dtype=np.float64)

        if self._is_multi_output:
            # One QRF per output column — set q then call predict on each.
            per_output = []
            for est in self._models_per_output_:
                est.q = q_array
                raw_col = np.asarray(est.predict(X))
                # When len(q)==1, sklearn_quantile returns (n_samples,)
                # instead of (1, n_samples).  Normalise to 2-D.
                if raw_col.ndim == 1:
                    raw_col = raw_col.reshape(1, -1)  # (1, n_samples)
                per_output.append(raw_col)

            results: Dict[float, np.ndarray] = {}
            for i, q in enumerate(quantiles):
                # Stack the i-th quantile across outputs → (n_samples, n_outputs)
                col = np.column_stack([po[i] for po in per_output])
                results[float(q)] = col
            return results

        # Single-output path
        self.model_.q = q_array
        raw = np.asarray(self.model_.predict(X))
        # Normalise to 2-D when a single quantile was requested
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        results: Dict[float, np.ndarray] = {}
        for i, q in enumerate(quantiles):
            results[float(q)] = raw[i]  # (n_samples,)

        return results

    def predict_intervals(
        self,
        X: np.ndarray,
        lower_q: float = 0.025,
        upper_q: float = 0.975,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute asymmetric prediction intervals directly from QRF quantiles.

        Produces **heteroscedastic** intervals: entities (provinces) whose
        feature vector lands in high-variance leaves receive wider bands, while
        those in stable, low-variance leaves receive narrower ones.  This
        mirrors the true conditional distribution of the target, unlike
        homoscedastic methods that add a constant half-width to every entity.

        For Bonferroni-corrected joint coverage across D criteria at overall
        level 1 − α, pass::

            lower_q = α / (2 * D)
            upper_q = 1 − α / (2 * D)

        which guarantees joint coverage ≥ 1 − α by the union bound.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            lower_q: Lower tail quantile level.  Default 0.025 gives a 95%
                symmetric single-component interval.  For Bonferroni-adjusted
                joint coverage use ``conformal_alpha / (2 * n_components)``.
            upper_q: Upper tail quantile level.  Must satisfy upper_q > lower_q.

        Returns:
            Tuple ``(lower_arr, upper_arr)`` each of shape
            ``(n_samples, n_outputs)``.  Guaranteed ``lower_arr ≤ upper_arr``
            element-wise (monotonicity is enforced defensively; it may be
            violated only when the leaf has fewer distinct observed values than
            needed for reliable extreme-quantile estimation).

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self._models_per_output_:
            raise ValueError(
                "QuantileRandomForestForecaster: call fit() before predict_intervals()."
            )

        qpreds = self.predict_quantiles(X, quantiles=[lower_q, upper_q])
        lower = qpreds[lower_q]
        upper = qpreds[upper_q]

        # Normalise to 2-D (n_samples, n_outputs) for consistent multi-output handling.
        # predict_quantiles returns (n_samples,) for single-output models.
        if lower.ndim == 1:
            lower = lower.reshape(-1, 1)
        if upper.ndim == 1:
            upper = upper.reshape(-1, 1)

        # Enforce monotonicity: lower ≤ upper.
        # With very extreme quantiles and small leaf sets the RFQR estimator may
        # return identical values or (rarely) swap them at minimal leaf sizes.
        lower_out = np.minimum(lower, upper)
        upper_out = np.maximum(lower, upper)
        return lower_out, upper_out

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate prediction uncertainty as IQR-based standard deviation proxy.

        Uses the interquartile range (Q75 - Q25) scaled to approximate
        standard deviation under normality: σ ≈ IQR / 1.349.

        Args:
            X: Feature matrix

        Returns:
            Uncertainty estimates (pseudo standard deviation)
        """
        qpreds = self.predict_quantiles(X, quantiles=[0.25, 0.75])
        iqr = qpreds[0.75] - qpreds[0.25]
        return iqr / 1.349  # IQR to σ conversion under normality

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet")
        return self.feature_importance_

    @property
    def oob_score(self) -> float:
        """Get out-of-bag R² score."""
        if self.model_ is None:
            raise ValueError("Model not fitted yet")
        if self._is_multi_output:
            # Average OOB score across per-output estimators
            return float(np.mean(
                [est.oob_score_ for est in self._models_per_output_]
            ))
        return self.model_.oob_score_

    def get_prediction_distribution(
        self, X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get comprehensive distributional summary for predictions.

        Returns:
            Dictionary with keys: 'mean' (RF conditional mean), 'median'
            (QRF conditional median), 'std', 'q05', 'q10', 'q25', 'q50',
            'q75', 'q90', 'q95'
        """
        # predict_mean() → RF conditional mean (average of tree outputs)
        # predict_median() → QRF conditional median (leaf-weight quantile at τ=0.5)
        # These are distinct statistics; using predict() here was a bug because
        # predict() returns the mean, not the median.
        mean_pred      = self.predict_mean(X)
        median_pred    = self.predict_median(X)   # true QRF conditional median
        quantile_preds = self.predict_quantiles(X)
        uncertainty    = self.predict_uncertainty(X)

        return {
            "mean":   mean_pred,
            "median": median_pred,
            "std":    uncertainty,
            **{f"q{int(q * 100):02d}": v for q, v in quantile_preds.items()},
        }
