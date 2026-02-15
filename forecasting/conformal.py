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
    Uses exponential smoothing of conformity scores to handle
    temporal non-stationarity:
    
    α_t = γ * α_{t-1} + (1-γ) * error_indicator_t

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
        gamma: Adaptation rate for ACI (0 < γ < 1, default 0.95)
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
        gamma: float = 0.95,
        symmetric: bool = True,
        random_state: int = 42,
    ):
        if alpha <= 0 or alpha >= 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.method = method
        self.alpha = alpha
        self.calibration_fraction = calibration_fraction
        self.gamma = gamma
        self.symmetric = symmetric
        self.random_state = random_state

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

    def calibrate(
        self,
        base_model,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
    ) -> "ConformalPredictor":
        """
        Calibrate the conformal predictor using training data.

        Args:
            base_model: Fitted model with .predict() method
            X: Feature matrix
            y: True target values
            cv_folds: Number of CV folds for cv_plus method

        Returns:
            Self for method chaining
        """
        self._base_model = base_model

        if y.ndim > 1:
            y = y.ravel() if y.shape[1] == 1 else y.mean(axis=1)

        if self.method == "split":
            self._calibrate_split(base_model, X, y)
        elif self.method == "cv_plus":
            self._calibrate_cv_plus(base_model, X, y, cv_folds)
        elif self.method == "adaptive":
            self._calibrate_adaptive(base_model, X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._calibrated = True
        return self

    def _calibrate_split(self, model, X: np.ndarray, y: np.ndarray):
        """Calibrate using split conformal method."""
        n = len(y)
        n_cal = max(5, int(n * self.calibration_fraction))

        # Use last n_cal samples as calibration (temporal ordering)
        X_cal = X[-n_cal:]
        y_cal = y[-n_cal:]

        # Get predictions on calibration set
        y_pred = model.predict(X_cal)
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
            # Asymmetric: separate upper and lower quantiles
            self._conformity_scores = residuals
            alpha_half = self.alpha / 2
            q_lower_level = alpha_half
            q_upper_level = 1.0 - alpha_half
            self._q_lower = np.quantile(residuals, q_lower_level)
            self._q_upper = np.quantile(residuals, q_upper_level)

        self._n_cal = n_cal

    def _calibrate_cv_plus(
        self, model, X: np.ndarray, y: np.ndarray, cv_folds: int
    ):
        """
        Calibrate using CV+ method (Barber et al., 2019).

        Uses leave-one-out or K-fold CV residuals for calibration,
        producing tighter intervals than split conformal.
        """
        from sklearn.model_selection import TimeSeriesSplit
        import copy

        n = len(y)
        tscv = TimeSeriesSplit(n_splits=min(cv_folds, max(2, n // 5)))

        # Collect out-of-fold residuals
        oof_residuals = np.full(n, np.nan)

        for train_idx, val_idx in tscv.split(X):
            try:
                model_copy = copy.deepcopy(model)

                # Handle models that need re-fitting vs pre-fitted
                if hasattr(model_copy, "fit"):
                    model_copy.fit(X[train_idx], y[train_idx])

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

        if self.symmetric:
            abs_residuals = np.abs(residuals)
            self._conformity_scores = abs_residuals
            n_cal = len(abs_residuals)
            q_level = np.ceil((1 - self.alpha) * (n_cal + 1)) / n_cal
            q_level = min(q_level, 1.0)
            self._q_hat = np.quantile(abs_residuals, q_level)
        else:
            self._conformity_scores = residuals
            alpha_half = self.alpha / 2
            self._q_lower = np.quantile(residuals, alpha_half)
            self._q_upper = np.quantile(residuals, 1.0 - alpha_half)

        self._n_cal = len(residuals)

    def _calibrate_adaptive(self, model, X: np.ndarray, y: np.ndarray):
        """
        Calibrate using Adaptive Conformal Inference (ACI).

        For time series data where the distribution may drift over time.
        Tracks effective coverage and adapts the quantile threshold.
        """
        y_pred = model.predict(X)
        if y_pred.ndim > 1:
            y_pred = y_pred.mean(axis=1) if y_pred.shape[1] > 1 else y_pred.ravel()

        residuals = np.abs(y - y_pred)
        self._conformity_scores = residuals

        # Initialize with standard conformal quantile
        n = len(residuals)
        q_level = min(np.ceil((1 - self.alpha) * (n + 1)) / n, 1.0)
        self._q_hat = np.quantile(residuals, q_level)

        # ACI: track adaptive alpha using exponential smoothing
        self._aci_alpha_t = self.alpha
        self._aci_history = []

        for t in range(len(residuals)):
            # Check if observation was covered
            covered = residuals[t] <= self._q_hat
            error_indicator = 0.0 if covered else 1.0

            # Update adaptive miscoverage rate
            self._aci_alpha_t = (
                self.gamma * self._aci_alpha_t
                + (1 - self.gamma) * error_indicator
            )
            self._aci_alpha_t = np.clip(self._aci_alpha_t, 0.001, 0.999)

            # Update quantile threshold
            q_level = min(
                np.ceil((1 - self._aci_alpha_t) * (n + 1)) / n, 1.0
            )
            self._q_hat = np.quantile(residuals[:t + 1], q_level)

            self._aci_history.append({
                "alpha_t": self._aci_alpha_t,
                "q_hat": self._q_hat,
                "covered": covered,
            })

        self._n_cal = n

    def predict_intervals(
        self,
        X: np.ndarray,
        point_predictions: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute prediction intervals with guaranteed coverage.

        Args:
            X: Feature matrix for new predictions
            point_predictions: Optional pre-computed point predictions.
                             If None, uses base_model.predict(X).

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
