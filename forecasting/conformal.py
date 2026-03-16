# -*- coding: utf-8 -*-
"""
Conformal Prediction for Uncertainty Calibration
=================================================

Provides distribution-free, finite-sample valid prediction intervals
that guarantee coverage at any desired level (e.g., 95%), regardless
of the underlying model or data distribution.

Unlike Bayesian credible intervals or bootstrap confidence intervals,
conformal prediction intervals have a theoretical coverage guarantee:

    P(Y_new ∈ C(X_new)) ≥ 1 - α

This guarantee holds for any model, any distribution, and any
finite sample size, under the (weak) assumption of exchangeability.

Algorithm (Split Conformal):
    1. Split data into proper training and calibration sets
    2. Train base model on proper training set
    3. Compute conformity scores on calibration set:
       s_i = |y_i - ŷ_i| (absolute residual)
    4. Compute q = (1-α)(1 + 1/n_cal)-quantile of {s_i}
    5. Prediction interval: [ŷ_new - q, ŷ_new + q]

Adaptive Conformal Inference (ACI) for Time Series:
    Implements the *additive* gradient step from Gibbs & Candès (2021, Eq. 3):

        α_{t+1} = α_t + γ(α − err_t)

    where ``err_t ∈ {0, 1}`` is the miscoverage indicator.  The default
    γ = 0.02 is appropriate for moderate distribution shifts (literature
    recommends γ ∈ [0.005, 0.05]).  The old default of 0.95 was incorrectly
    borrowed from exponential-smoothing forgetting factors (multiplicative
    formula) and caused wild oscillation with this additive update rule.

Additional calibration path:
    ``calibrate_residuals(residuals)`` accepts pre-computed OOF residuals
    so the full model never needs to be re-fitted or deep-copied during
    conformal calibration (used by ``UnifiedForecaster``).

References:
    - Vovk, Gammerman & Shafer (2005). "Algorithmic Learning in a
      Random World" Springer
    - Barber et al. (2023). "Conformal Prediction Beyond
      Exchangeability" Annals of Statistics
    - Xu & Xie (2021). "Conformal Prediction Interval for Dynamic
      Time-Series" ICML
    - Romano, Patterson & Candès (2019). "Conformalized Quantile
      Regression" NeurIPS
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

logger = logging.getLogger('ml_mcdm')


# ---------------------------------------------------------------------------
# E-06: Student-t predictive quantile for small-n conformal calibration.
# ---------------------------------------------------------------------------

def _calibrate_with_studentt(
    residuals: np.ndarray,
    alpha: float,
) -> float:
    """Student-t predictive quantile for small-n conformal calibration.

    For signed OOF residuals, converts to absolute values internally and fits
    a Student-t distribution via MLE with location forced to 0.  Returns the
    (1 - alpha) predictive quantile using degrees-of-freedom df_pred = df_fit - 1
    to penalise for scale estimation uncertainty (Meeker & Escobar, 1998).

    Falls back to the Papadopoulos empirical quantile when:
    - ``len(residuals) > 100`` (large sample — empirical is reliable)
    - ``scipy.stats`` is unavailable
    - MLE optimisation fails or returns non-finite result

    Parameters
    ----------
    residuals : ndarray, shape (n,)
        Signed ``y_true − ŷ`` OOF residuals.  NaNs must be removed by caller.
    alpha : float
        Miscoverage level.  Returns the ``1 - alpha`` predictive quantile of
        the fitted Student-t applied to |residuals|.

    Returns
    -------
    q_hat : float
        Conformal half-width threshold (≥ 0).

    References
    ----------
    Meeker & Escobar (1998). "Statistical Intervals: A Guide for
    Practitioners" — Section 4.3: Predictive interval for a future observation.
    """
    abs_res = np.abs(residuals)
    n = len(abs_res)

    # Large-sample path: empirical Papadopoulos quantile is unbiased and
    # consistent; the Student-t approximation provides no additional benefit.
    if n > 100:
        q_level = min(np.ceil((1.0 - alpha) * (n + 1)) / n, 1.0)
        return float(np.quantile(abs_res, q_level))

    try:
        from scipy.stats import t as _t_dist  # type: ignore[import]

        # MLE with loc=0 forced: scale captures median absolute deviation.
        # Note: scipy t.fit minimises the negative log-likelihood; floc=0
        # constraints the location to zero (mean-zero absolute residuals).
        df_fit, _loc, scale_fit = _t_dist.fit(abs_res, floc=0)

        # Predictive df: subtract 1 to account for having estimated scale
        # (analogous to using s instead of σ in a normal-theory prediction
        # interval, which inflates the effective df by 1).
        df_pred = max(df_fit - 1.0, 1.0)

        q_hat = float(_t_dist.ppf(1.0 - alpha, df=df_pred, loc=0.0, scale=scale_fit))

        # Guard: non-positive or infinite quantile signals degenerate fit.
        if not np.isfinite(q_hat) or q_hat < 0.0:
            raise ValueError(f"Degenerate Student-t quantile: {q_hat}")

    except Exception:
        # Fallback: Papadopoulos empirical quantile (always valid).
        q_level = min(np.ceil((1.0 - alpha) * (n + 1)) / n, 1.0)
        q_hat = float(np.quantile(abs_res, q_level))

    return q_hat


# ---------------------------------------------------------------------------
# Panel-aware walk-forward splitter (local copy, avoids circular import from
# super_learner.py).  Identical semantics to _WalkForwardYearlySplit.
# ---------------------------------------------------------------------------
class _ConformalWalkForwardSplit:
    """
    Walk-forward yearly splitter for conformal calibration.

    Produces expanding-window folds where training contains all rows with
    year_label < val_year and validation contains rows where
    year_label == val_year.

    Unlike ``TimeSeriesSplit``, fold boundaries are calendar-year-aligned,
    so no cross-entity temporal leakage can occur in the stacked panel.

    Parameters
    ----------
    min_train_years : int
        Minimum unique year-label cohorts in the *first* training window.
        Default 2 starts early to maximise the conformal calibration set
        (residuals from all training years are needed for coverage).
    max_folds : int
        Hard cap on the number of folds yielded.  Set large (e.g. 999) to
        exhaust all calendar years after ``min_train_years``.
    """

    def __init__(self, min_train_years: int = 2, max_folds: int = 999):
        self.min_train_years = min_train_years
        self.max_folds = max_folds

    def split(self, X: np.ndarray, year_labels: np.ndarray):
        """Yield (train_idx, val_idx) in calendar-year order."""
        unique_years = np.sort(np.unique(year_labels))
        n_years = len(unique_years)

        if n_years < 2:
            mid = len(X) // 2
            if mid > 0:
                yield np.arange(mid), np.arange(mid, len(X))
            return

        first_val_pos = min(self.min_train_years, n_years - 1)
        n_yielded = 0
        for k in range(self.max_folds):
            val_pos = first_val_pos + k
            if val_pos >= n_years:
                break
            val_year = int(unique_years[val_pos])
            train_idx = np.where(year_labels < val_year)[0]
            val_idx   = np.where(year_labels == val_year)[0]
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            yield np.sort(train_idx), np.sort(val_idx)
            n_yielded += 1

        if n_yielded == 0 and n_years >= 2:
            val_year = int(unique_years[-1])
            train_idx = np.where(year_labels < val_year)[0]
            val_idx   = np.where(year_labels == val_year)[0]
            if len(train_idx) > 0 and len(val_idx) > 0:
                yield np.sort(train_idx), np.sort(val_idx)


class ConformalPredictor:
    """
    Conformal prediction wrapper for any point forecaster.

    Wraps any base model/ensemble to produce prediction intervals
    with guaranteed finite-sample coverage.

    Supports three methods:
    1. Split Conformal: Simple split into train/calibration
    2. CV+ (Cross-Validation+): Uses CV residuals for tighter intervals
    3. Adaptive Conformal Inference (ACI): For time series with drift

    Parameters:
        method: Conformal method ('split', 'cv_plus', 'adaptive')
        alpha: Miscoverage rate (default 0.05 for 95% intervals)
        calibration_fraction: Fraction of data for calibration (split method)
        gamma: Adaptation rate for ACI.  The ACI update rule is the additive
            gradient step from Gibbs & Candès (2021, Eq. 3):

                α_{t+1} = α_t + γ(α − err_t)

            where ``err_t ∈ {0, 1}`` is the miscoverage indicator.  With
            this *additive* formulation a single miss shifts α by ``γ``
            (not ``γ × α``), so the step size must be small.  Typical
            values from the literature are γ ∈ [0.005, 0.05].  The
            default 0.02 tracks moderate distribution shifts while
            remaining stable.  (The old default 0.95 was borrowed from
            exponential-smoothing forgetting factors where the formula is
            *multiplicative*; it caused wild oscillation here.)
        symmetric: If True, symmetric intervals; if False, asymmetric
        random_state: Random seed for reproducibility

    Example:
        >>> from forecasting.super_learner import SuperLearner
        >>> cp = ConformalPredictor(method='cv_plus', alpha=0.05)
        >>> cp.calibrate(model, X_train, y_train)
        >>> lower, upper = cp.predict_intervals(X_test)
    """

    def __init__(
        self,
        method: str = "cv_plus",
        alpha: float = 0.05,
        calibration_fraction: float = 0.25,
        gamma: float = 0.02,
        symmetric: bool = True,
        random_state: int = 42,
        stratify_by_missingness: bool = False,
        n_strata: int = 3,
        use_studentt_small_n: bool = False,
        studentt_threshold: int = 50,
    ):
        if alpha <= 0 or alpha >= 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.method = method
        self.alpha = alpha
        self.calibration_fraction = calibration_fraction
        self.gamma = gamma
        self.symmetric = symmetric
        self.random_state = random_state
        self.stratify_by_missingness = stratify_by_missingness
        self.n_strata = n_strata
        # E-06: Student-t predictive distribution for small calibration sets.
        # When enabled and n_cal < studentt_threshold, _calibrate_with_studentt()
        # replaces the empirical Papadopoulos quantile in calibrate_residuals().
        self.use_studentt_small_n = use_studentt_small_n
        self.studentt_threshold = studentt_threshold

        # Calibration state
        self._conformity_scores: Optional[np.ndarray] = None
        self._q_hat: Optional[float] = None
        self._q_lower: Optional[float] = None
        self._q_upper: Optional[float] = None
        self._calibrated: bool = False
        self._base_model = None
        self._n_cal: int = 0

        # ACI tracking state
        self._aci_alpha_t: float = alpha
        self._aci_history: List[float] = []
        
        # M-13: Stratified conformal state
        self._stratified: bool = False
        self._stratum_quantiles: Optional[Dict[int, Tuple[float, float]]] = None
        self._stratum_boundaries: Optional[np.ndarray] = None
        self._missingness_rates_cal: Optional[np.ndarray] = None

    def calibrate(
        self,
        base_model,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        year_labels: Optional[np.ndarray] = None,
        entity_indices: Optional[np.ndarray] = None,
    ) -> "ConformalPredictor":
        """
        Calibrate the conformal predictor using training data.

        The target ``y`` must be 1-D (single output component).  For
        multi-output problems, create one ``ConformalPredictor`` per
        component and calibrate each on a single column of ``y``.

        Args:
            base_model: Fitted model with .predict() method.
                        For multi-output models, wrap the model so that
                        ``.predict()`` returns only the relevant column.
            X: Feature matrix
            y: True target values — must be 1-D or shape (n, 1)
            cv_folds: Number of CV folds for cv_plus method (used only
                when ``year_labels`` is None).
            year_labels: Optional integer calendar-year label for each row
                (the *target* year).  When provided, ``_calibrate_cv_plus``
                uses :class:`_ConformalWalkForwardSplit` (panel-aware,
                calendar-year-aligned) instead of ``TimeSeriesSplit``
                (row-position-based), eliminating cross-entity temporal
                leakage in the stacked panel.
            entity_indices: Optional integer entity indices.  Currently
                stored for forward-compatibility; forwarded to models that
                accept an ``entity_indices`` keyword in ``.fit()`` /
                ``.predict()``.

        Returns:
            Self for method chaining

        Raises:
            ValueError:
                If ``y`` has more than one column (multi-output).
        """
        self._base_model = base_model

        if y.ndim > 1 and y.shape[1] > 1:
            raise ValueError(
                f"ConformalPredictor.calibrate() requires single-output y, "
                f"got y.shape={y.shape}.  Calibrate one predictor per "
                f"output component (with Bonferroni-corrected alpha)."
            )
        if y.ndim > 1:
            y = y.ravel()

        if self.method == "split":
            self._calibrate_split(base_model, X, y)
        elif self.method == "cv_plus":
            self._calibrate_cv_plus(
                base_model, X, y, cv_folds,
                year_labels=year_labels,
                entity_indices=entity_indices,
            )
        elif self.method == "adaptive":
            self._calibrate_adaptive(base_model, X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._calibrated = True
        return self

    def calibrate_residuals(
        self,
        residuals: np.ndarray,
        base_model=None,
    ) -> "ConformalPredictor":
        """
        Calibrate from pre-computed out-of-fold residuals.

        This is a lightweight alternative to :meth:`calibrate` that
        avoids re-fitting the model entirely.  It is used by
        ``UnifiedForecaster`` to calibrate conformal intervals from the
        SuperLearner's cached OOF residuals so that the entire
        SuperLearner ensemble is **never** deep-copied during conformal
        calibration (U-2 performance fix).

        The same finite-sample ``(n+1)/n`` Papadopoulos quantile
        correction is applied as in :meth:`_calibrate_cv_plus`.

        Args:
            residuals: 1-D array of signed ``y_true - y_pred`` residuals
                from held-out (OOF) data.  NaNs are automatically removed.
            base_model: Optional fitted model to store as ``_base_model``.
                If ``predict_intervals`` is called without
                ``point_predictions``, this model is used.  When the
                caller always passes pre-computed point predictions (as
                ``UnifiedForecaster`` does), this can be ``None``.

        Returns:
            Self for method chaining.
        """
        residuals = np.asarray(residuals, dtype=np.float64)
        residuals = residuals[~np.isnan(residuals)]

        # F-07: Hard minimum (3) kept for correctness; practical minimum (30)
        # triggers a warning so the caller knows interval reliability is low.
        # With n_cal < 10 the Papadopoulos quantile is clamped to the maximum
        # residual, producing infinitely conservative intervals.
        _HARD_MIN_CAL = 3
        _SOFT_MIN_CAL = 30
        if len(residuals) < _HARD_MIN_CAL:
            raise ValueError(
                "calibrate_residuals() needs at least 3 valid residuals; "
                f"got {len(residuals)}."
            )
        if len(residuals) < _SOFT_MIN_CAL:
            logger.warning(
                "ConformalPredictor.calibrate_residuals(): only %d calibration "
                "residuals (<%d recommended). The Papadopoulos quantile may be "
                "clamped to the observed maximum, producing overly wide "
                "prediction intervals. Consider using the E-01 extended "
                "walk-forward sweep or Mondrian stratification to increase n_cal.",
                len(residuals), _SOFT_MIN_CAL,
            )

        if base_model is not None:
            self._base_model = base_model

        n_cal = len(residuals)

        if self.symmetric:
            abs_residuals = np.abs(residuals)
            self._conformity_scores = abs_residuals
            # E-06: Student-t predictive distribution for small calibration sets.
            # When n_cal < studentt_threshold and use_studentt_small_n is True,
            # fit a Student-t MLE and use the predictive (1-alpha) quantile
            # with df_pred = df_fit - 1 (penalise for scale estimation).
            # This provides heavier-tail coverage guarantees for strata with
            # n_cal < 50, where the empirical Papadopoulos quantile has high
            # variance and may undershoot the required coverage level.
            if self.use_studentt_small_n and n_cal < self.studentt_threshold:
                self._q_hat = _calibrate_with_studentt(residuals, self.alpha)
            else:
                q_level = np.ceil((1 - self.alpha) * (n_cal + 1)) / n_cal
                q_level = min(q_level, 1.0)
                self._q_hat = np.quantile(abs_residuals, q_level)
        else:
            # Asymmetric: same (n+1)/n Papadopoulos correction as
            # _calibrate_split / _calibrate_cv_plus.
            self._conformity_scores = residuals
            alpha_half = self.alpha / 2
            q_upper_level = min(
                np.ceil((1.0 - alpha_half) * (n_cal + 1)) / n_cal, 1.0
            )
            q_lower_level = max(
                np.floor(alpha_half * (n_cal + 1)) / n_cal, 0.0
            )
            self._q_lower = np.quantile(residuals, q_lower_level)
            self._q_upper = np.quantile(residuals, q_upper_level)

        self._n_cal = n_cal
        self._calibrated = True
        return self

    def _calibrate_split(self, model, X: np.ndarray, y: np.ndarray):
        """Calibrate using split conformal method.
        
        Re-fits the model on the proper training portion so that
        calibration residuals are computed on truly held-out data.
        """
        import copy

        n = len(y)
        n_cal = max(5, int(n * self.calibration_fraction))

        # Split: train on first (n - n_cal), calibrate on last n_cal
        X_train = X[:-n_cal]
        y_train = y[:-n_cal]
        X_cal = X[-n_cal:]
        y_cal = y[-n_cal:]

        # Re-fit model on proper training set only
        model_proper = copy.deepcopy(model)
        if hasattr(model_proper, 'fit'):
            model_proper.fit(X_train, y_train)

        # Replace base model so predict_intervals uses the correctly
        # re-fitted model whose residuals we are calibrating against.
        self._base_model = model_proper

        # Get predictions on held-out calibration set
        y_pred = model_proper.predict(X_cal)
        if y_pred.ndim > 1:
            y_pred = y_pred.mean(axis=1) if y_pred.shape[1] > 1 else y_pred.ravel()

        # Compute conformity scores
        residuals = y_cal - y_pred

        if self.symmetric:
            self._conformity_scores = np.abs(residuals)
            # Compute conformal quantile
            n_cal = len(self._conformity_scores)
            q_level = np.ceil((1 - self.alpha) * (n_cal + 1)) / n_cal
            q_level = min(q_level, 1.0)
            self._q_hat = np.quantile(self._conformity_scores, q_level)
        else:
            # Asymmetric: separate upper and lower quantiles.
            # Apply the finite-sample (n+1)/n Papadopoulos correction to
            # both quantile levels.  Without this correction the marginal
            # coverage guarantee P(Y ∈ C(X)) ≥ 1 − α does NOT hold;
            # intervals are systematically too narrow.
            self._conformity_scores = residuals
            alpha_half = self.alpha / 2
            # Upper tail: we want the α/2-upper quantile of the calibration
            # signed residuals, i.e. the high-end threshold.
            q_upper_level = min(
                np.ceil((1.0 - alpha_half) * (n_cal + 1)) / n_cal, 1.0
            )
            # Lower tail: floor ensures we do not clip the lower tail too
            # aggressively (conservative on both sides).
            q_lower_level = max(
                np.floor(alpha_half * (n_cal + 1)) / n_cal, 0.0
            )
            self._q_lower = np.quantile(residuals, q_lower_level)
            self._q_upper = np.quantile(residuals, q_upper_level)

        self._n_cal = n_cal

    def _calibrate_cv_plus(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int,
        year_labels: Optional[np.ndarray] = None,
        entity_indices: Optional[np.ndarray] = None,
    ):
        """
        Calibrate using a panel-aware CV+ adaptation for time-series data.

        **Panel-aware splitting (E-04 fix)**

        When ``year_labels`` is provided, fold boundaries follow calendar
        years via :class:`_ConformalWalkForwardSplit` — identical to the
        walk-forward scheme used by the SuperLearner ensemble.  This
        guarantees:

        * Fold train/val splits are aligned on the same calendar-year
          boundaries as the ensemble's OOF predictions, preserving
          exchangeability between calibration residuals and test residuals.
        * No cross-entity temporal leakage: entities are not split mid-way
          through their time series by row-position arithmetic.
        * Extended calibration set: starting from ``min_train_years=2``
          instead of the ensemble's ``min_train_years=8`` provides
          residuals from ALL training years (e.g. 2013–2024 for a
          2012–2024 panel), not just 2020–2024 (E-02 coverage extension).

        When ``year_labels`` is None, falls back to ``TimeSeriesSplit``
        on the row position — valid for non-panel usage.

        **Coverage guarantee**

        The finite-sample ``(n+1)/n`` Papadopoulos correction is applied
        to the pooled OOF residuals, matching the correction in
        :meth:`calibrate_residuals`.  Marginal coverage is guaranteed
        under temporal exchangeability within each fold.

        References
        ----------
        * Barber et al. (2023), "Conformal Prediction Beyond Exchangeability"
        * Gibbon et al. (2023), "Online Conformal Prediction for PAL"
        """
        import copy

        n = len(y)

        # ------------------------------------------------------------------
        # Choose splitter: panel-aware when year_labels provided (E-04 fix)
        # ------------------------------------------------------------------
        if year_labels is not None:
            # Panel-aware walk-forward; start from min_train_years=2 so all
            # training years contribute calibration residuals (E-02 coverage
            # extension).  max_folds=999 exhaust all remaining years.
            splitter = _ConformalWalkForwardSplit(min_train_years=2, max_folds=999)
            splits = list(splitter.split(X, year_labels))
        else:
            # Row-position fallback — used for non-panel data / unit tests
            from sklearn.model_selection import TimeSeriesSplit as _TSS
            n_splits_safe = min(cv_folds, max(2, n // 5))
            splits = list(_TSS(n_splits=n_splits_safe).split(X))

        if not splits:
            self._calibrate_split(model, X, y)
            return

        # ------------------------------------------------------------------
        # Collect out-of-fold residuals
        # ------------------------------------------------------------------
        oof_residuals = np.full(n, np.nan)

        for train_idx, val_idx in splits:
            try:
                model_copy = copy.deepcopy(model)

                # Forward entity_indices to panel-aware models when supported
                if hasattr(model_copy, "fit"):
                    train_ent = (
                        entity_indices[train_idx]
                        if entity_indices is not None else None
                    )
                    if train_ent is not None:
                        import inspect
                        sig = inspect.signature(model_copy.fit)
                        if 'entity_indices' in sig.parameters:
                            model_copy.fit(
                                X[train_idx], y[train_idx],
                                entity_indices=train_ent,
                            )
                        else:
                            model_copy.fit(X[train_idx], y[train_idx])
                    else:
                        model_copy.fit(X[train_idx], y[train_idx])

                # Forward entity_indices to predict when supported
                val_ent = (
                    entity_indices[val_idx]
                    if entity_indices is not None else None
                )
                if val_ent is not None:
                    import inspect
                    sig_p = inspect.signature(model_copy.predict)
                    if 'entity_indices' in sig_p.parameters:
                        pred = model_copy.predict(
                            X[val_idx], entity_indices=val_ent
                        )
                    else:
                        pred = model_copy.predict(X[val_idx])
                else:
                    pred = model_copy.predict(X[val_idx])

                if pred.ndim > 1:
                    pred = pred.mean(axis=1) if pred.shape[1] > 1 else pred.ravel()

                oof_residuals[val_idx] = y[val_idx] - pred

            except Exception:
                continue

        # Use only valid (non-NaN) residuals
        valid = ~np.isnan(oof_residuals)
        residuals = oof_residuals[valid]

        if len(residuals) < 3:
            # Fallback to split method
            self._calibrate_split(model, X, y)
            return

        # ------------------------------------------------------------------
        # Compute Papadopoulos-corrected conformal quantile from pooled OOF
        # residuals.  The finite-sample correction ⌈(1-α)(n+1)⌉/n guarantees
        # marginal coverage P(Y_new ∈ C(X_new)) ≥ 1-α for any finite n.
        # ------------------------------------------------------------------
        n_cal = len(residuals)

        if self.symmetric:
            abs_residuals = np.abs(residuals)
            self._conformity_scores = abs_residuals
            q_level = min(
                np.ceil((1 - self.alpha) * (n_cal + 1)) / n_cal, 1.0
            )
            self._q_hat = np.quantile(abs_residuals, q_level)
        else:
            # Asymmetric: apply (n+1)/n correction to both tails.
            self._conformity_scores = residuals
            alpha_half = self.alpha / 2
            q_upper_level = min(
                np.ceil((1.0 - alpha_half) * (n_cal + 1)) / n_cal, 1.0
            )
            q_lower_level = max(
                np.floor(alpha_half * (n_cal + 1)) / n_cal, 0.0
            )
            self._q_lower = np.quantile(residuals, q_lower_level)
            self._q_upper = np.quantile(residuals, q_upper_level)

        self._n_cal = n_cal

    def _calibrate_adaptive(self, model, X: np.ndarray, y: np.ndarray):
        """
        Calibrate using Adaptive Conformal Inference (ACI).

        For time series data where the distribution may drift over time.
        Tracks effective coverage and adapts the quantile threshold.

        Splits the data into a proper-training portion and a calibration
        portion.  The model is re-fitted on the training portion so that
        conformity scores are computed on genuinely held-out data (audit
        fix H6: the previous version used in-sample residuals, which
        produced anti-conservative / too-tight intervals).
        """
        import copy

        n = len(y)
        # Reserve the last `n_cal` points for calibration (temporal split).
        n_cal = max(5, int(n * self.calibration_fraction))

        X_train = X[:-n_cal]
        y_train = y[:-n_cal]
        X_cal = X[-n_cal:]
        y_cal = y[-n_cal:]

        # Re-fit on proper training set so residuals are out-of-sample
        model_proper = copy.deepcopy(model)
        if hasattr(model_proper, 'fit'):
            model_proper.fit(X_train, y_train)

        # Use re-fitted model for future predict_intervals calls
        self._base_model = model_proper

        # Compute out-of-sample residuals on calibration set
        y_pred_cal = model_proper.predict(X_cal)
        if y_pred_cal.ndim > 1:
            y_pred_cal = (
                y_pred_cal.mean(axis=1) if y_pred_cal.shape[1] > 1
                else y_pred_cal.ravel()
            )

        residuals = np.abs(y_cal - y_pred_cal)
        self._conformity_scores = residuals

        # Use first half of calibration residuals for initial quantile,
        # then run the ACI online update on the second half.
        n_init = max(3, n_cal // 2)
        init_residuals = residuals[:n_init]
        q_level_init = min(
            np.ceil((1 - self.alpha) * (n_init + 1)) / n_init, 1.0
        )
        self._q_hat = np.quantile(init_residuals, q_level_init)

        # ACI: track adaptive alpha
        self._aci_alpha_t = self.alpha
        self._aci_history = []

        # Online update on second half of calibration residuals
        for t in range(n_init, n_cal):
            # Check if observation was covered
            covered = residuals[t] <= self._q_hat
            error_indicator = 0.0 if covered else 1.0

            # Update adaptive miscoverage rate (Gibbs & Candès, 2021, Eq. 3).
            # Gradient step: α_{t+1} = α_t + γ(α − error_t)
            self._aci_alpha_t = (
                self._aci_alpha_t + self.gamma * (self.alpha - error_indicator)
            )
            self._aci_alpha_t = np.clip(self._aci_alpha_t, 0.001, 0.999)

            # Update quantile threshold using calibration residuals seen so far
            seen = residuals[:t + 1]
            q_level = min(
                np.ceil((1 - self._aci_alpha_t) * (len(seen) + 1)) / len(seen),
                1.0,
            )
            self._q_hat = np.quantile(seen, q_level)

            self._aci_history.append({
                "alpha_t": self._aci_alpha_t,
                "q_hat": self._q_hat,
                "covered": covered,
            })

        self._n_cal = n_cal

    def calibrate_stratified(
        self,
        residuals: np.ndarray,
        missingness_rates: np.ndarray,
    ) -> "ConformalPredictor":
        """
        Calibrate conformal intervals with stratification by missingness rate.

        Enhancement M-13: Missingness-Stratified Conformal Intervals
        -------------------------------------------------------------
        Computes per-stratum quantiles to provide honest uncertainty
        quantification that reflects feature missingness. Samples with
        higher missingness receive wider intervals.

        Theoretical Foundation:
            - Barber et al. (2023): Conformal prediction with conditional
              coverage guarantees
            - Stratification ensures valid coverage within each missingness
              regime (low/medium/high)
            - Uses adaptive quantiles: q_s = ⌈(1-α)(n_s+1)⌉/n_s per stratum

        Args:
            residuals: Calibration absolute residuals or conformity scores
            missingness_rates: Per-sample missingness rates (fraction of NaN
                             features) for calibration set

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If not initialized with stratify_by_missingness=True
        """
        if not self._stratified:
            raise ValueError(
                "calibrate_stratified() requires stratify_by_missingness=True"
            )

        if len(residuals) != len(missingness_rates):
            raise ValueError(
                f"Residuals ({len(residuals)}) and missingness_rates "
                f"({len(missingness_rates)}) must have same length"
            )

        # Store calibration missingness rates for boundary computation
        self._missingness_rates_cal = missingness_rates

        # Assign calibration samples to strata based on missingness quantiles
        strata = self._assign_to_strata(missingness_rates)

        # Compute per-stratum quantiles with finite-sample correction
        self._stratum_quantiles = {}
        for s in range(self.n_strata):
            stratum_mask = strata == s
            stratum_residuals = residuals[stratum_mask]

            if len(stratum_residuals) < 3:
                # Too few samples in stratum — use global quantile
                if self.symmetric:
                    q_level = min(
                        np.ceil((1 - self.alpha) * (len(residuals) + 1))
                        / len(residuals),
                        1.0,
                    )
                    q_s = np.quantile(np.abs(residuals), q_level)
                    self._stratum_quantiles[s] = (q_s, q_s)
                else:
                    alpha_half = self.alpha / 2
                    q_upper_level = min(
                        np.ceil((1.0 - alpha_half) * (len(residuals) + 1))
                        / len(residuals),
                        1.0,
                    )
                    q_lower_level = max(
                        np.floor(alpha_half * (len(residuals) + 1))
                        / len(residuals),
                        0.0,
                    )
                    q_lower_s = np.quantile(residuals, q_lower_level)
                    q_upper_s = np.quantile(residuals, q_upper_level)
                    self._stratum_quantiles[s] = (q_lower_s, q_upper_s)
                continue

            # Compute stratum-specific quantiles with (n+1)/n correction
            n_s = len(stratum_residuals)
            if self.symmetric:
                q_level = min(
                    np.ceil((1 - self.alpha) * (n_s + 1)) / n_s, 1.0
                )
                q_s = np.quantile(np.abs(stratum_residuals), q_level)
                self._stratum_quantiles[s] = (q_s, q_s)
            else:
                alpha_half = self.alpha / 2
                q_upper_level = min(
                    np.ceil((1.0 - alpha_half) * (n_s + 1)) / n_s, 1.0
                )
                q_lower_level = max(
                    np.floor(alpha_half * (n_s + 1)) / n_s, 0.0
                )
                q_lower_s = np.quantile(stratum_residuals, q_lower_level)
                q_upper_s = np.quantile(stratum_residuals, q_upper_level)
                self._stratum_quantiles[s] = (q_lower_s, q_upper_s)

        # Store conformity scores
        self._conformity_scores = residuals
        self._n_cal = len(residuals)

        return self

    def _assign_to_strata(self, missingness_rates: np.ndarray) -> np.ndarray:
        """
        Assign samples to missingness strata.

        Uses quantile-based binning to create balanced strata. If stratum
        boundaries are already computed (from calibration), uses those.
        Otherwise, computes new boundaries from input data.

        Args:
            missingness_rates: Per-sample missingness rates (fraction of NaN)

        Returns:
            Stratum assignments (0 to n_strata-1) for each sample
        """
        # Compute stratum boundaries if not already set
        if self._stratum_boundaries is None:
            # Use quantile-based boundaries
            quantiles = np.linspace(0, 1, self.n_strata + 1)
            self._stratum_boundaries = np.quantile(missingness_rates, quantiles)
            # Ensure boundaries are unique (if data has limited unique values)
            self._stratum_boundaries = np.unique(self._stratum_boundaries)

        # Assign samples to strata using digitize (bins are right-inclusive)
        # digitize returns 0 for values < boundaries[0], so subtract 1
        strata = np.digitize(missingness_rates, self._stratum_boundaries) - 1

        # Clip to valid range [0, n_strata-1]
        strata = np.clip(strata, 0, self.n_strata - 1)

        return strata

    def predict_intervals(
        self,
        X: np.ndarray,
        point_predictions: Optional[np.ndarray] = None,
        missingness_rates: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute prediction intervals with guaranteed coverage.
        
        Enhancement M-13: Missingness-Stratified Conformal Intervals
        -------------------------------------------------------------
        When stratify_by_missingness=True, prediction intervals are
        calibrated per missingness stratum. Samples with higher feature
        missingness receive wider intervals, reflecting higher uncertainty.

        Args:
            X: Feature matrix for new predictions
            point_predictions: Optional pre-computed point predictions.
                             If None, uses base_model.predict(X).
            missingness_rates: Optional per-sample missingness rates (fraction
                             of NaN features). Required if stratify_by_missingness=True.

        Returns:
            Tuple of (lower_bound, upper_bound) arrays with
            guaranteed P(Y ∈ [lower, upper]) ≥ 1 - α
        """
        if not self._calibrated:
            raise ValueError("Not calibrated. Call calibrate() first.")

        if point_predictions is None:
            point_predictions = self._base_model.predict(X)

        if point_predictions.ndim > 1:
            point_predictions = (
                point_predictions.mean(axis=1)
                if point_predictions.shape[1] > 1
                else point_predictions.ravel()
            )
        
        n_test = len(point_predictions)
        lower = np.zeros(n_test)
        upper = np.zeros(n_test)
        
        # M-13: Stratified conformal intervals
        if self._stratified and missingness_rates is not None:
            # Assign each test sample to a stratum
            test_strata = self._assign_to_strata(missingness_rates)
            
            for stratum_id in range(self.n_strata):
                stratum_mask = test_strata == stratum_id
                if not stratum_mask.any():
                    continue
                
                # Get stratum-specific quantiles
                if stratum_id in self._stratum_quantiles:
                    q_lower_s, q_upper_s = self._stratum_quantiles[stratum_id]
                else:
                    # Fallback to global quantiles if stratum not calibrated
                    q_lower_s, q_upper_s = self._q_lower, self._q_upper
                
                if self.symmetric:
                    q_s = (q_upper_s - q_lower_s) / 2.0 if q_lower_s is not None else q_upper_s
                    lower[stratum_mask] = point_predictions[stratum_mask] - q_s
                    upper[stratum_mask] = point_predictions[stratum_mask] + q_s
                else:
                    lower[stratum_mask] = point_predictions[stratum_mask] + q_lower_s
                    upper[stratum_mask] = point_predictions[stratum_mask] + q_upper_s
        
        else:
            # Standard (unstratified) intervals
            if self.symmetric:
                lower = point_predictions - self._q_hat
                upper = point_predictions + self._q_hat
            else:
                # Asymmetric intervals
                lower = point_predictions + self._q_lower  # q_lower is negative
                upper = point_predictions + self._q_upper

        return lower, upper

    def predict_with_intervals(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return point predictions and prediction intervals together.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        predictions = self._base_model.predict(X)
        if predictions.ndim > 1:
            predictions = (
                predictions.mean(axis=1)
                if predictions.shape[1] > 1
                else predictions.ravel()
            )

        lower, upper = self.predict_intervals(X, point_predictions=predictions)
        return predictions, lower, upper

    def get_interval_width(self) -> float:
        """Get the width of the prediction interval."""
        if self.symmetric:
            return 2.0 * self._q_hat if self._q_hat is not None else np.nan
        else:
            return (
                self._q_upper - self._q_lower
                if self._q_upper is not None
                else np.nan
            )

    def evaluate_coverage(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate the empirical coverage and sharpness on test data.

        Args:
            X_test: Test features
            y_test: True test values

        Returns:
            Dictionary with coverage metrics:
                - empirical_coverage: Fraction of test points covered
                - target_coverage: 1 - alpha
                - mean_interval_width: Average interval width
                - median_interval_width: Median interval width
                - coverage_gap: empirical - target
        """
        if y_test.ndim > 1:
            y_test = y_test.ravel() if y_test.shape[1] == 1 else y_test.mean(axis=1)

        predictions, lower, upper = self.predict_with_intervals(X_test)

        covered = (y_test >= lower) & (y_test <= upper)
        empirical_coverage = covered.mean()

        widths = upper - lower
        target_coverage = 1.0 - self.alpha

        return {
            "empirical_coverage": empirical_coverage,
            "target_coverage": target_coverage,
            "coverage_gap": empirical_coverage - target_coverage,
            "mean_interval_width": np.mean(widths),
            "median_interval_width": np.median(widths),
            "std_interval_width": np.std(widths),
            "n_test": len(y_test),
            "n_calibration": self._n_cal,
            "method": self.method,
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get conformal prediction diagnostics."""
        diag = {
            "method": self.method,
            "alpha": self.alpha,
            "calibrated": self._calibrated,
            "n_calibration_samples": self._n_cal,
            "interval_width": self.get_interval_width(),
        }

        if self.symmetric:
            diag["q_hat"] = self._q_hat
        else:
            diag["q_lower"] = self._q_lower
            diag["q_upper"] = self._q_upper

        if self.method == "adaptive" and self._aci_history:
            diag["final_aci_alpha"] = self._aci_alpha_t
            diag["n_aci_adjustments"] = len(self._aci_history)
            coverages = [h["covered"] for h in self._aci_history]
            diag["aci_rolling_coverage"] = np.mean(coverages[-20:])

        return diag


# ---------------------------------------------------------------------------
# E-03: Mondrian Conformal Predictor stratified by feature missingness rate.
# ---------------------------------------------------------------------------

class MissingnessStratifiedConformal:
    """Mondrian Conformal Predictor stratified by feature missingness rate.

    Provides stratum-conditional coverage guarantees separately for provinces
    with different levels of missing input features::

        P(Y ∈ C(X) | stratum(X) = s) ≥ 1 - α  for each stratum s.

    Strata are defined by quantile boundaries of the calibration missingness
    distribution (equal-frequency binning), so each stratum receives
    approximately ``n_cal / n_strata`` calibration residuals.

    When a stratum has fewer than ``_MIN_STRATUM_CAL`` calibration samples,
    the global Papadopoulos quantile is used as a conservative fallback —
    this ensures coverage is never violated at the expense of interval width.

    Parameters
    ----------
    alpha : float
        Marginal miscoverage level.  Each stratum achieves ≥ 1 - alpha
        conditional coverage.
    n_strata : int ≥ 2
        Number of missingness strata.  Default 3 (low / medium / high).

    References
    ----------
    Vovk et al. (2005). "Algorithmic Learning in a Random World" — Chapter 3
        (Mondrian CP, stratum-conditional coverage).
    Barber et al. (2023). "Conformal Prediction Beyond Exchangeability" —
        Section 4 (validity under covariate stratification).
    """

    _MIN_STRATUM_CAL: int = 5

    def __init__(self, alpha: float = 0.05, n_strata: int = 3) -> None:
        if alpha <= 0 or alpha >= 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if n_strata < 2:
            raise ValueError(f"n_strata must be ≥ 2, got {n_strata}")
        self.alpha = alpha
        self.n_strata = n_strata

        self._stratum_boundaries: Optional[np.ndarray] = None
        self._stratum_quantiles: Dict[int, float] = {}
        self._global_quantile: float = np.inf
        self._calibrated: bool = False
        self._n_cal: int = 0
        self._n_cal_per_stratum: Dict[int, int] = {}

    def calibrate(
        self,
        residuals: np.ndarray,
        missingness_rates: np.ndarray,
    ) -> "MissingnessStratifiedConformal":
        """Calibrate from OOF residuals and per-sample missingness fractions.

        Parameters
        ----------
        residuals : ndarray, shape (n_cal,)
            Signed ``y_true − ŷ`` OOF residuals.  Converted to abs internally.
        missingness_rates : ndarray, shape (n_cal,)
            Per-sample fraction of NaN feature columns in [0, 1].

        Returns
        -------
        self
        """
        residuals = np.asarray(residuals, dtype=np.float64).ravel()
        missingness_rates = np.asarray(missingness_rates, dtype=np.float64).ravel()

        if len(residuals) != len(missingness_rates):
            raise ValueError(
                f"residuals ({len(residuals)}) and missingness_rates "
                f"({len(missingness_rates)}) must have the same length."
            )

        # Remove NaN rows (should be pre-filtered but guard defensively)
        valid = ~(np.isnan(residuals) | np.isnan(missingness_rates))
        residuals = residuals[valid]
        missingness_rates = missingness_rates[valid]

        n_cal = len(residuals)
        if n_cal < self._MIN_STRATUM_CAL:
            raise ValueError(
                f"MissingnessStratifiedConformal.calibrate() needs ≥ "
                f"{self._MIN_STRATUM_CAL} valid samples, got {n_cal}."
            )

        abs_res = np.abs(residuals)

        # Global quantile: Papadopoulos (n+1)/n correction.
        # Used as fallback for strata with too few samples.
        q_glob = np.ceil((1.0 - self.alpha) * (n_cal + 1)) / n_cal
        q_glob = min(q_glob, 1.0)
        self._global_quantile = float(np.quantile(abs_res, q_glob))

        # Equal-frequency stratum boundaries from the calibration distribution.
        quantile_cuts = np.linspace(0.0, 1.0, self.n_strata + 1)
        raw_boundaries = np.quantile(missingness_rates, quantile_cuts)

        # Deduplicate: constant missingness collapses to a single stratum.
        unique_boundaries = np.unique(raw_boundaries)
        if len(unique_boundaries) < 2:
            # No variation — one global stratum, global quantile for all.
            self._stratum_boundaries = np.array([
                float(missingness_rates.min()), float(missingness_rates.max()) + 1e-9
            ])
            self._stratum_quantiles = {0: self._global_quantile}
            self._n_cal_per_stratum = {0: n_cal}
        else:
            self._stratum_boundaries = unique_boundaries
            n_actual = len(unique_boundaries) - 1
            for s in range(n_actual):
                lo = unique_boundaries[s]
                hi = unique_boundaries[s + 1]
                # Last stratum: right-inclusive to capture the maximum value.
                if s == n_actual - 1:
                    mask = (missingness_rates >= lo) & (missingness_rates <= hi)
                else:
                    mask = (missingness_rates >= lo) & (missingness_rates < hi)

                res_s = abs_res[mask]
                n_s = int(mask.sum())
                self._n_cal_per_stratum[s] = n_s

                if n_s < self._MIN_STRATUM_CAL:
                    # Too sparse — use global quantile as conservative fallback.
                    self._stratum_quantiles[s] = self._global_quantile
                else:
                    q_s = np.ceil((1.0 - self.alpha) * (n_s + 1)) / n_s
                    q_s = min(q_s, 1.0)
                    self._stratum_quantiles[s] = float(np.quantile(res_s, q_s))

        self._n_cal = n_cal
        self._calibrated = True
        return self

    def _assign_strata(self, rates: np.ndarray) -> np.ndarray:
        """Map each missingness rate to its stratum index (0-based).

        Uses ``np.searchsorted`` on the upper stratum boundaries so that
        rates exactly at a boundary are assigned to the higher stratum.
        The result is clipped to [0, n_strata - 1].
        """
        if self._stratum_boundaries is None:
            return np.zeros(len(rates), dtype=int)
        # Upper boundaries (excluding the leftmost): searchsorted finds the
        # first upper bound strictly greater than the rate.
        upper_bounds = self._stratum_boundaries[1:]
        strata = np.searchsorted(upper_bounds, rates, side='left')
        max_s = len(self._stratum_boundaries) - 2  # = n_actual - 1
        return np.clip(strata, 0, max(max_s, 0)).astype(int)

    def predict_intervals(
        self,
        point_predictions: np.ndarray,
        missingness_rates: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stratum-conditional symmetric prediction intervals.

        Parameters
        ----------
        point_predictions : ndarray, shape (n_test,)
            Ensemble point forecast for each test province.
        missingness_rates : ndarray, shape (n_test,)
            Per-province fraction of NaN input features in [0, 1].

        Returns
        -------
        lower, upper : ndarray, each shape (n_test,)
        """
        if not self._calibrated:
            raise RuntimeError(
                "MissingnessStratifiedConformal: call calibrate() first."
            )
        point_predictions = np.asarray(point_predictions, dtype=np.float64).ravel()
        missingness_rates = np.asarray(missingness_rates, dtype=np.float64).ravel()

        strata = self._assign_strata(missingness_rates)
        q_arr = np.array([
            self._stratum_quantiles.get(int(s), self._global_quantile)
            for s in strata
        ])

        return point_predictions - q_arr, point_predictions + q_arr

    def get_diagnostics(self) -> Dict[str, Any]:
        """Calibration statistics per stratum."""
        return {
            "alpha": self.alpha,
            "n_strata": self.n_strata,
            "n_cal_total": self._n_cal,
            "global_quantile": self._global_quantile,
            "stratum_quantiles": dict(self._stratum_quantiles),
            "n_cal_per_stratum": dict(self._n_cal_per_stratum),
            "stratum_boundaries": (
                self._stratum_boundaries.tolist()
                if self._stratum_boundaries is not None else []
            ),
        }


# ---------------------------------------------------------------------------
# E-02: Conformalized Quantile Regression (CQR).
# ---------------------------------------------------------------------------

class CQRConformalPredictor:
    """Conformalized Quantile Regression (Romano, Patterson & Candès, NeurIPS 2019).

    Provides adaptive-width prediction intervals using a Quantile Random Forest
    (QRF) as the base regression quantile estimator.  A single conformity
    adjustment ``q̂_CQR`` is calibrated so that the shifted QRF intervals::

        [Q̂_{α/2}(x) − q̂_CQR,  Q̂_{1−α/2}(x) + q̂_CQR]

    achieve marginal coverage ≥ 1 − α over exchangeable test–calibration pairs.

    CQR conformity score (Eq. 2 of Romano et al. 2019)::

        s_i = max(Q̂_{α/2}(x_i) − y_i,   y_i − Q̂_{1−α/2}(x_i))

    * ``s_i < 0``: y_i inside the QRF interval (comfortable coverage).
    * ``s_i ≥ 0``: y_i outside the interval; value is the overshoot distance.

    ``q̂_CQR`` is the Papadopoulos ``(1-α)(1+1/n)``-quantile of ``{s_i}``.

    Key advantage over split conformal: the interval widths are *adaptive* —
    entities whose feature vectors land in high-variance QRF leaves receive
    automatically wider intervals, those in low-variance leaves receive
    narrower ones.  This is the heteroscedastic analogue of split conformal,
    maintaining the same marginal coverage guarantee.

    Note on calibration data
    ------------------------
    For theoretically valid (exchangeable) calibration the QRF quantile
    predictions on the calibration set should be OOF (out-of-bag or out-of-
    fold).  When only in-sample QRF predictions are available, the calibration
    is approximate: in-sample QRF intervals are slightly wider (the training
    point contributes to its own leaf distribution), causing ``q̂_CQR`` to be
    slightly smaller than the honest estimate.  The Papadopoulos correction
    adds conservatism that compensates in practice.

    Parameters
    ----------
    alpha : float
        Miscoverage level (same ``alpha`` as used for the QRF quantile levels).

    References
    ----------
    Romano, Patterson & Candès (2019). "Conformalized Quantile Regression."
        NeurIPS 2019, Advances in Neural Information Processing Systems.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if alpha <= 0 or alpha >= 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self._q_cqr: Optional[float] = None
        self._calibrated: bool = False
        self._n_cal: int = 0
        self._cal_miscoverage_rate: float = float("nan")  # diagnostic

    def calibrate(
        self,
        qrf_lower_cal: np.ndarray,
        qrf_upper_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "CQRConformalPredictor":
        """Calibrate from QRF quantile predictions and true targets.

        Parameters
        ----------
        qrf_lower_cal : ndarray, shape (n_cal,)
            QRF lower-quantile predictions (at level α/2 or similar) on the
            calibration set.
        qrf_upper_cal : ndarray, shape (n_cal,)
            QRF upper-quantile predictions (at level 1 − α/2 or similar).
        y_cal : ndarray, shape (n_cal,)
            True target values on the calibration set.

        Returns
        -------
        self
        """
        qrf_lower_cal = np.asarray(qrf_lower_cal, dtype=np.float64).ravel()
        qrf_upper_cal = np.asarray(qrf_upper_cal, dtype=np.float64).ravel()
        y_cal = np.asarray(y_cal, dtype=np.float64).ravel()

        # Joint NaN removal (any NaN in the triplet invalidates the row).
        valid = ~(
            np.isnan(qrf_lower_cal)
            | np.isnan(qrf_upper_cal)
            | np.isnan(y_cal)
        )
        qrf_lower_cal = qrf_lower_cal[valid]
        qrf_upper_cal = qrf_upper_cal[valid]
        y_cal = y_cal[valid]

        n_cal = len(y_cal)
        if n_cal < 3:
            raise ValueError(
                f"CQRConformalPredictor.calibrate() requires ≥ 3 valid samples, "
                f"got {n_cal}."
            )

        # CQR conformity score (Romano et al. 2019, Eq. 2):
        # positive when y is outside [lower, upper]; negative when inside.
        scores = np.maximum(qrf_lower_cal - y_cal, y_cal - qrf_upper_cal)

        # Diagnostic: fraction of cal points outside QRF interval.
        self._cal_miscoverage_rate = float((scores > 0).mean())

        # Papadopoulos (n+1)/n finite-sample correction.
        q_level = np.ceil((1.0 - self.alpha) * (n_cal + 1)) / n_cal
        q_level = min(q_level, 1.0)
        self._q_cqr = float(np.quantile(scores, q_level))

        self._n_cal = n_cal
        self._calibrated = True
        return self

    def predict_intervals(
        self,
        qrf_lower_test: np.ndarray,
        qrf_upper_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CQR prediction intervals for the test set.

        Interval: ``[Q̂_{α/2}(x) − q̂_CQR,  Q̂_{1−α/2}(x) + q̂_CQR]``.

        When ``q̂_CQR < 0`` (QRF intervals over-cover on average), the CQR
        interval is strictly *inside* the QRF interval but still ≥ 1 − α coverage.

        Parameters
        ----------
        qrf_lower_test : ndarray, shape (n_test,)
        qrf_upper_test : ndarray, shape (n_test,)

        Returns
        -------
        lower, upper : ndarray, each shape (n_test,)
        """
        if not self._calibrated:
            raise RuntimeError(
                "CQRConformalPredictor: call calibrate() before predict_intervals()."
            )
        qrf_lower_test = np.asarray(qrf_lower_test, dtype=np.float64).ravel()
        qrf_upper_test = np.asarray(qrf_upper_test, dtype=np.float64).ravel()
        return qrf_lower_test - self._q_cqr, qrf_upper_test + self._q_cqr

    @property
    def q_cqr(self) -> Optional[float]:
        """Calibrated CQR adjustment (shifted to both interval endpoints)."""
        return self._q_cqr

    def get_diagnostics(self) -> Dict[str, Any]:
        """CQR diagnostics."""
        return {
            "alpha": self.alpha,
            "q_cqr": self._q_cqr,
            "n_cal": self._n_cal,
            "cal_miscoverage_rate": self._cal_miscoverage_rate,
        }


# ---------------------------------------------------------------------------
# E-05: Locally Weighted Conformal Prediction (Tibshirani et al. 2019).
# ---------------------------------------------------------------------------

class LocallyWeightedConformalPredictor:
    """Weighted Split Conformal Predictor (Tibshirani, Barber, Candès & Ramdas, 2019).

    Assigns per-calibration-point importance weights based on RBF feature-space
    similarity to the test point.  Same-entity calibration points receive a
    multiplicative boost (``entity_weight`` ≥ 1), reflecting the higher
    exchangeability between a province's own historical observations and its
    future prediction.

    Weighted quantile (Tibshirani et al. 2019, Eq. 3):
        - Calibration weights:  w_i ∝ K_h(||x_i - x_test||) × entity_boost_i
        - Normalised to probability simplex (Σ w_i = 1).
        - An extra ``+∞`` residual with weight ``1/(n+1)`` serves as the
          Papadopoulos guard, ensuring marginal coverage ≥ 1 − α.

    Coverage guarantee (Theorem 1, Tibshirani et al. 2019):
        Under covariate shift with p_test/p_cal density ratio bounded by the
        kernel weights, weighted conformal achieves ≥ 1 − α marginal coverage.

    For panel data (province × year):
        - ``x_test`` = T+1 feature vector for province p.
        - ``x_cal_i`` = historical features for province q at year t.
        - Same-entity weighting (p = q) identifies the province's own
          calibration residuals as most representative.

    Computational complexity: O(n_test × n_cal × p) per call.  With
    n_test ≤ 63 provinces and n_cal ≤ 750, this is ≤ 0.1 s.

    Parameters
    ----------
    alpha : float
        Miscoverage level.
    entity_weight : float ≥ 1.0
        Multiplicative up-weight for same-entity calibration points.
        Default 2.0 doubles the probability mass assigned to the test
        province's own historical residuals.
    min_bandwidth : float
        Lower bound on the RBF bandwidth to prevent degenerate weights.

    References
    ----------
    Tibshirani, Barber, Candès & Ramdas (2019). "Conformal Prediction
        Under Covariate Shift." NeurIPS 2019.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        entity_weight: float = 2.0,
        min_bandwidth: float = 1e-6,
    ) -> None:
        if alpha <= 0 or alpha >= 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if entity_weight < 1.0:
            raise ValueError(
                f"entity_weight must be ≥ 1.0, got {entity_weight}"
            )
        self.alpha = alpha
        self.entity_weight = entity_weight
        self.min_bandwidth = min_bandwidth

        self._X_cal: Optional[np.ndarray] = None
        self._abs_residuals_cal: Optional[np.ndarray] = None
        self._entity_cal: Optional[np.ndarray] = None
        self._bandwidth: float = 1.0
        self._calibrated: bool = False
        self._n_cal: int = 0

    def calibrate(
        self,
        X_cal: np.ndarray,
        residuals: np.ndarray,
        entity_indices: Optional[np.ndarray] = None,
    ) -> "LocallyWeightedConformalPredictor":
        """Calibrate from OOF features, residuals, and optional entity IDs.

        The RBF bandwidth is estimated via the median heuristic computed on
        a random subsample of ``min(n_cal, 500)`` rows.

        Parameters
        ----------
        X_cal : ndarray, shape (n_cal, n_features)
            OOF feature matrix (tree-track; aligned with residuals).
        residuals : ndarray, shape (n_cal,)
            Signed OOF residuals ``y − ŷ``.  Converted to abs internally.
        entity_indices : ndarray, shape (n_cal,), optional
            Integer entity (province) IDs; enables same-entity up-weighting.

        Returns
        -------
        self
        """
        X_cal = np.asarray(X_cal, dtype=np.float64)
        residuals = np.asarray(residuals, dtype=np.float64).ravel()

        if X_cal.ndim != 2:
            raise ValueError(
                f"X_cal must be 2-D, got shape {X_cal.shape}"
            )
        if len(residuals) != X_cal.shape[0]:
            raise ValueError(
                f"X_cal rows ({X_cal.shape[0]}) and residuals ({len(residuals)}) "
                f"must match."
            )

        # Joint NaN removal
        valid = ~np.isnan(residuals)
        if X_cal.ndim == 2:
            valid = valid & ~np.isnan(X_cal).any(axis=1)

        X_cal = X_cal[valid]
        residuals = residuals[valid]
        entity_cal = (
            np.asarray(entity_indices, dtype=object).ravel()[valid]
            if entity_indices is not None else None
        )

        n_cal = len(residuals)
        if n_cal < 3:
            raise ValueError(
                f"LocallyWeightedConformalPredictor.calibrate() requires "
                f"≥ 3 valid samples, got {n_cal}."
            )

        self._X_cal = X_cal
        self._abs_residuals_cal = np.abs(residuals)
        self._entity_cal = entity_cal
        self._n_cal = n_cal

        # Bandwidth via median pairwise distance (median heuristic).
        # Subsampled to ≤ 500 rows for O(n²) cost control.
        if n_cal <= 500:
            X_bw = X_cal
        else:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_cal, 500, replace=False)
            X_bw = X_cal[idx]

        # Squared pairwise distances via ||a-b||² = ||a||² + ||b||² - 2a·b
        sq_norms = np.sum(X_bw ** 2, axis=1)
        dists_sq = (
            sq_norms[:, np.newaxis]
            + sq_norms[np.newaxis, :]
            - 2.0 * (X_bw @ X_bw.T)
        )
        np.clip(dists_sq, 0.0, None, out=dists_sq)
        dists = np.sqrt(dists_sq)
        upper_tri = dists[np.triu_indices_from(dists, k=1)]
        nonzero = upper_tri[upper_tri > 0.0]
        self._bandwidth = max(
            float(np.median(nonzero)) if len(nonzero) > 0 else 1.0,
            self.min_bandwidth,
        )

        self._calibrated = True
        return self

    def _compute_weights(
        self,
        x_q: np.ndarray,
        entity_q: Any = None,
    ) -> np.ndarray:
        """Per-calibration-point importance weights for test point x_q.

        w_i ∝ exp(−||x_q − x_cal_i||² / (2h²)) × entity_boost_i

        Parameters
        ----------
        x_q : ndarray, shape (n_features,)
            A single test feature vector.
        entity_q : optional
            Entity ID of the test point; triggers same-entity up-weighting.

        Returns
        -------
        w : ndarray, shape (n_cal,)  — probability simplex weights (Σw=1).
        """
        gamma = 1.0 / (2.0 * self._bandwidth ** 2)
        diff = self._X_cal - x_q[np.newaxis, :]   # (n_cal, p)
        d_sq = np.sum(diff ** 2, axis=1)           # (n_cal,)
        w = np.exp(-gamma * d_sq)                  # (n_cal,)

        # Entity-aware boost: same-entity calibration points are up-weighted.
        if entity_q is not None and self._entity_cal is not None:
            same = self._entity_cal == entity_q
            w[same] *= self.entity_weight

        w_sum = w.sum()
        if w_sum < 1e-30:
            # All weights near-zero: test point is far from all cal points →
            # fall back to equal weights (standard split conformal).
            return np.ones(self._n_cal) / self._n_cal
        return w / w_sum

    def _weighted_quantile(self, w: np.ndarray) -> float:
        """Weighted (1 − α) quantile of abs_residuals_cal (Tibshirani et al. 2019).

        Follows Eq. 3: q̂ = inf{q : Σ_{i: r_i ≤ q} w_i ≥ 1 − α}.
        An extra +∞ residual with weight 1/(n+1) ensures ≥ 1 − α coverage.

        Parameters
        ----------
        w : ndarray, shape (n_cal,)  — probability simplex weights (already normalised).

        Returns
        -------
        q_hat : float  — half-width of the symmetric interval.
        """
        n_cal = self._n_cal
        # Rescale existing weights to accommodate the extra +∞ point.
        # Total mass = 1; extra point gets mass 1/(n+1); remaining rescaled.
        w_inf = 1.0 / (n_cal + 1)
        w_scaled = w * (n_cal / (n_cal + 1))  # renormalise to 1 - w_inf

        sorted_idx = np.argsort(self._abs_residuals_cal)
        sorted_r = self._abs_residuals_cal[sorted_idx]
        sorted_w = w_scaled[sorted_idx]

        aug_r = np.concatenate([sorted_r, [np.inf]])
        aug_w = np.concatenate([sorted_w, [w_inf]])

        cum_w = np.cumsum(aug_w)
        target = 1.0 - self.alpha
        idx_q = np.searchsorted(cum_w, target, side='left')
        idx_q = min(idx_q, len(aug_r) - 1)
        return float(aug_r[idx_q])

    def predict_intervals(
        self,
        X_test: np.ndarray,
        point_predictions: np.ndarray,
        entity_indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Locally-weighted symmetric prediction intervals.

        O(n_test × n_cal × p).  Acceptable for n_test ≤ 63, n_cal ≤ 750,
        p ≤ 150 (≈ 0.05 s).

        Parameters
        ----------
        X_test : ndarray, shape (n_test, n_features)
        point_predictions : ndarray, shape (n_test,)
            Ensemble point forecasts (centre of each interval).
        entity_indices : ndarray, shape (n_test,), optional
            Entity IDs of the prediction-year provinces.

        Returns
        -------
        lower, upper : ndarray, each shape (n_test,)
        """
        if not self._calibrated:
            raise RuntimeError(
                "LocallyWeightedConformalPredictor: call calibrate() first."
            )
        X_test = np.asarray(X_test, dtype=np.float64)
        point_predictions = np.asarray(point_predictions, dtype=np.float64).ravel()
        n_test = len(point_predictions)

        lower = np.empty(n_test)
        upper = np.empty(n_test)

        for i in range(n_test):
            ent_q = entity_indices[i] if entity_indices is not None else None
            w = self._compute_weights(X_test[i], ent_q)
            q_hat = self._weighted_quantile(w)
            lower[i] = point_predictions[i] - q_hat
            upper[i] = point_predictions[i] + q_hat

        return lower, upper

    def get_diagnostics(self) -> Dict[str, Any]:
        """LWCP calibration diagnostics."""
        return {
            "alpha": self.alpha,
            "entity_weight": self.entity_weight,
            "bandwidth": self._bandwidth,
            "n_cal": self._n_cal,
        }
