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

        if len(residuals) < 3:
            raise ValueError(
                "calibrate_residuals() needs at least 3 valid residuals; "
                f"got {len(residuals)}."
            )

        if base_model is not None:
            self._base_model = base_model

        n_cal = len(residuals)

        if self.symmetric:
            abs_residuals = np.abs(residuals)
            self._conformity_scores = abs_residuals
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
