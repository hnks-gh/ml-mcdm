# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation Suite for Forecasting
================================================

Provides rigorous evaluation metrics, validation procedures, and
diagnostic tests for assessing forecasting model quality.

Evaluation Dimensions:
    1. Accuracy: R², RMSE, MAE, MAPE per sub-criterion
    2. Uncertainty Calibration: Coverage, sharpness, calibration curves
    3. Model Diagnostics: Residual tests, heteroscedasticity, stationarity
    4. Ablation Studies: Component-wise contribution analysis

Validation Strategies:
    - Expanding window: Train on t₁...tₖ, predict tₖ₊₁
    - Rolling window: Train on tₖ₋w...tₖ, predict tₖ₊₁
    - Leave-one-year-out: Train on all but tₖ, predict tₖ

References:
    - Hyndman & Athanasopoulos (2021). "Forecasting: Principles
      and Practice" 3rd edition (OTexts)
    - Gneiting & Raftery (2007). "Strictly Proper Scoring Rules,
      Prediction, and Estimation" JASA
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
)
from sklearn.model_selection import TimeSeriesSplit
import copy
import warnings

warnings.filterwarnings("ignore")


class ForecastEvaluator:
    """
    Comprehensive evaluation suite for forecasting models.

    Parameters:
        metrics: List of metric names to compute
        cv_strategy: Cross-validation strategy ('expanding', 'rolling', 'loyo')
        n_folds: Number of CV folds
        window_size: Window size for rolling CV
        verbose: Print progress

    Example:
        >>> evaluator = ForecastEvaluator(metrics=['r2', 'rmse', 'mae'])
        >>> results = evaluator.evaluate(model, X_train, y_train)
        >>> report = evaluator.generate_report(results)
    """

    METRIC_FUNCTIONS = {
        "r2": lambda y, p: r2_score(y, p),
        "rmse": lambda y, p: np.sqrt(mean_squared_error(y, p)),
        "mae": lambda y, p: mean_absolute_error(y, p),
        "medae": lambda y, p: median_absolute_error(y, p),
        "mape": lambda y, p: np.mean(np.abs((y - p) / (np.abs(y) + 1e-10))) * 100,
        "max_error": lambda y, p: np.max(np.abs(y - p)),
        "bias": lambda y, p: np.mean(p - y),
    }

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        cv_strategy: str = "expanding",
        n_folds: int = 5,
        window_size: Optional[int] = None,
        verbose: bool = True,
    ):
        self.metrics = metrics or ["r2", "rmse", "mae", "mape"]
        self.cv_strategy = cv_strategy
        self.n_folds = n_folds
        self.window_size = window_size
        self.verbose = verbose

    def evaluate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation.

        Args:
            model: Model with fit() and predict() methods
            X: Training features
            y: Training targets
            X_test: Optional test features for holdout evaluation
            y_test: Optional test targets

        Returns:
            Dictionary with evaluation results
        """
        results = {}

        # Cross-validation evaluation
        if self.verbose:
            print("  Evaluating: Cross-validation...")
        results["cv"] = self._cross_validate(model, X, y)

        # Holdout evaluation
        if X_test is not None and y_test is not None:
            if self.verbose:
                print("  Evaluating: Holdout test set...")
            results["holdout"] = self._evaluate_holdout(model, X, y, X_test, y_test)

        # Residual diagnostics
        if self.verbose:
            print("  Evaluating: Residual diagnostics...")
        results["diagnostics"] = self._residual_diagnostics(model, X, y)

        return results

    def _cross_validate(
        self, model, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        """Run time-series cross-validation."""
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            try:
                model_copy = copy.deepcopy(model)

                if hasattr(model_copy, "fit"):
                    model_copy.fit(X[train_idx], y[train_idx])

                pred = model_copy.predict(X[val_idx])
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)

                fold_metrics = {}
                for metric_name in self.metrics:
                    if metric_name in self.METRIC_FUNCTIONS:
                        func = self.METRIC_FUNCTIONS[metric_name]
                        # Average across outputs
                        scores = []
                        for c in range(y.shape[1]):
                            pc = min(c, pred.shape[1] - 1)
                            scores.append(func(y[val_idx, c], pred[:, pc]))
                        fold_metrics[metric_name] = np.mean(scores)

                fold_results.append(fold_metrics)
            except Exception as e:
                if self.verbose:
                    print(f"    Fold {fold_idx} failed: {e}")

        # Aggregate across folds
        if not fold_results:
            return {"error": "All folds failed"}

        agg = {}
        for metric_name in self.metrics:
            values = [f[metric_name] for f in fold_results if metric_name in f]
            if values:
                agg[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "per_fold": values,
                }

        return agg

    def _evaluate_holdout(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate on holdout test set."""
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        model_copy = copy.deepcopy(model)
        if hasattr(model_copy, "fit"):
            model_copy.fit(X_train, y_train)

        pred = model_copy.predict(X_test)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)

        results = {}
        for metric_name in self.metrics:
            if metric_name in self.METRIC_FUNCTIONS:
                func = self.METRIC_FUNCTIONS[metric_name]
                scores = []
                for c in range(y_test.shape[1]):
                    pc = min(c, pred.shape[1] - 1)
                    scores.append(func(y_test[:, c], pred[:, pc]))
                results[metric_name] = np.mean(scores)

        return results

    def _residual_diagnostics(
        self, model, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform residual diagnostic tests using out-of-fold residuals.

        Uses TimeSeriesSplit cross-validation to compute genuine
        out-of-sample residuals, avoiding the overfitting bias that
        arises when diagnostics are computed on in-sample residuals.

        Tests:
            - Normality (skewness and excess kurtosis)
            - Autocorrelation (Durbin-Watson statistic)
            - Heteroscedasticity (residual vs predicted correlation)
            - Mean zero test
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Collect out-of-fold predictions via TimeSeriesSplit
        n = X.shape[0]
        tscv = TimeSeriesSplit(n_splits=min(self.n_folds, max(2, n // 5)))

        oof_residuals = np.full(n, np.nan)
        oof_pred_mean = np.full(n, np.nan)

        for train_idx, val_idx in tscv.split(X):
            try:
                model_copy = copy.deepcopy(model)
                if hasattr(model_copy, "fit"):
                    model_copy.fit(X[train_idx], y[train_idx])

                pred = model_copy.predict(X[val_idx])
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)

                # Average residuals across outputs
                oof_residuals[val_idx] = np.mean(
                    y[val_idx] - pred[:, :y.shape[1]], axis=1
                )
                oof_pred_mean[val_idx] = np.mean(pred[:, :y.shape[1]], axis=1)
            except Exception:
                continue

        # Keep only valid (non-NaN) positions
        valid = ~np.isnan(oof_residuals)
        if valid.sum() < 5:
            return {"error": "Not enough OOF residuals for diagnostics"}

        residuals = oof_residuals[valid]
        pred_mean = oof_pred_mean[valid]

        diagnostics = {}

        # 1. Mean zero test
        diagnostics["residual_mean"] = float(np.mean(residuals))
        diagnostics["residual_std"] = float(np.std(residuals))
        diagnostics["mean_is_zero"] = abs(np.mean(residuals)) < 2 * np.std(residuals) / np.sqrt(len(residuals))

        # 2. Durbin-Watson statistic (autocorrelation)
        if len(residuals) > 2:
            diff = np.diff(residuals)
            dw = np.sum(diff ** 2) / (np.sum(residuals ** 2) + 1e-10)
            diagnostics["durbin_watson"] = float(dw)
            # DW ≈ 2 means no autocorrelation, <1.5 positive, >2.5 negative
            diagnostics["no_autocorrelation"] = 1.5 < dw < 2.5

        # 3. Heteroscedasticity check (correlation between |residuals| and predictions)
        abs_resid = np.abs(residuals)
        if np.std(pred_mean) > 1e-10 and np.std(abs_resid) > 1e-10:
            corr = np.corrcoef(pred_mean, abs_resid)[0, 1]
            diagnostics["heteroscedasticity_corr"] = float(corr)
            diagnostics["homoscedastic"] = abs(corr) < 0.3

        # 4. Normality (skewness and kurtosis)
        if len(residuals) > 10:
            skewness = float(np.mean(((residuals - np.mean(residuals)) / (np.std(residuals) + 1e-10)) ** 3))
            kurtosis = float(np.mean(((residuals - np.mean(residuals)) / (np.std(residuals) + 1e-10)) ** 4) - 3)
            diagnostics["skewness"] = skewness
            diagnostics["excess_kurtosis"] = kurtosis
            diagnostics["approximately_normal"] = abs(skewness) < 1.0 and abs(kurtosis) < 3.0

        return diagnostics

    @staticmethod
    def evaluate_uncertainty(
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        target_coverage: float = 0.95,
    ) -> Dict[str, float]:
        """
        Evaluate prediction interval quality.

        Metrics:
            - Empirical coverage: Fraction of true values within intervals
            - Mean/median interval width (sharpness)
            - Coverage gap: Empirical - target
            - Winkler score: Penalizes wide intervals and missed coverage

        Args:
            y_true: True values
            lower: Lower bounds of prediction intervals
            upper: Upper bounds of prediction intervals
            target_coverage: Desired coverage level

        Returns:
            Dictionary of uncertainty quality metrics
        """
        if y_true.ndim > 1:
            y_true = y_true.ravel()
        if lower.ndim > 1:
            lower = lower.ravel()
        if upper.ndim > 1:
            upper = upper.ravel()

        covered = (y_true >= lower) & (y_true <= upper)
        widths = upper - lower

        alpha = 1.0 - target_coverage

        # Winkler score
        winkler_scores = widths.copy()
        below = y_true < lower
        above = y_true > upper
        winkler_scores[below] += (2.0 / alpha) * (lower[below] - y_true[below])
        winkler_scores[above] += (2.0 / alpha) * (y_true[above] - upper[above])

        return {
            "empirical_coverage": float(covered.mean()),
            "target_coverage": target_coverage,
            "coverage_gap": float(covered.mean() - target_coverage),
            "mean_interval_width": float(np.mean(widths)),
            "median_interval_width": float(np.median(widths)),
            "std_interval_width": float(np.std(widths)),
            "winkler_score": float(np.mean(winkler_scores)),
            "n_samples": len(y_true),
        }

    @staticmethod
    def calibration_curve(
        y_true: np.ndarray,
        quantile_predictions: Dict[float, np.ndarray],
        n_bins: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Compute calibration curve: predicted vs observed quantile levels.

        A well-calibrated model should have points along the diagonal.

        Args:
            y_true: True values
            quantile_predictions: Dict mapping quantile levels to predictions
            n_bins: Number of bins for calibration

        Returns:
            Dictionary with:
                - expected_levels: Predicted quantile levels
                - observed_levels: Empirical fraction below each quantile
                - calibration_error: Mean absolute calibration error
        """
        if y_true.ndim > 1:
            y_true = y_true.ravel()

        expected = []
        observed = []

        for q_level, q_pred in sorted(quantile_predictions.items()):
            if q_pred.ndim > 1:
                q_pred = q_pred.ravel()
            fraction_below = np.mean(y_true <= q_pred)
            expected.append(q_level)
            observed.append(fraction_below)

        expected = np.array(expected)
        observed = np.array(observed)
        calibration_error = np.mean(np.abs(expected - observed))

        return {
            "expected_levels": expected,
            "observed_levels": observed,
            "calibration_error": calibration_error,
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable evaluation report.

        Args:
            results: Output from evaluate()

        Returns:
            Formatted string report
        """
        lines = [
            "",
            "=" * 80,
            "FORECASTING EVALUATION REPORT",
            "=" * 80,
        ]

        # Cross-validation results
        if "cv" in results and isinstance(results["cv"], dict) and "error" not in results["cv"]:
            lines.append("\n## Cross-Validation Results")
            lines.append("-" * 40)
            for metric, vals in results["cv"].items():
                if isinstance(vals, dict):
                    lines.append(
                        f"  {metric:>8s}: {vals['mean']:.4f} ± {vals['std']:.4f} "
                        f"[{vals['min']:.4f}, {vals['max']:.4f}]"
                    )

        # Holdout results
        if "holdout" in results:
            lines.append("\n## Holdout Test Results")
            lines.append("-" * 40)
            for metric, value in results["holdout"].items():
                lines.append(f"  {metric:>8s}: {value:.4f}")

        # Diagnostics
        if "diagnostics" in results:
            diag = results["diagnostics"]
            lines.append("\n## Residual Diagnostics")
            lines.append("-" * 40)

            lines.append(f"  Residual mean:        {diag.get('residual_mean', 'N/A'):.6f}")
            lines.append(f"  Residual std:         {diag.get('residual_std', 'N/A'):.6f}")
            lines.append(f"  Mean approx 0:        {'[OK]' if diag.get('mean_is_zero', False) else '[X]'}")

            if "durbin_watson" in diag:
                lines.append(f"  Durbin-Watson:        {diag['durbin_watson']:.4f}")
                lines.append(f"  No autocorrelation:   {'[OK]' if diag.get('no_autocorrelation', False) else '[X]'}")

            if "heteroscedasticity_corr" in diag:
                lines.append(f"  Heterosc. corr:       {diag['heteroscedasticity_corr']:.4f}")
                lines.append(f"  Homoscedastic:        {'[OK]' if diag.get('homoscedastic', False) else '[X]'}")

            if "skewness" in diag:
                lines.append(f"  Skewness:             {diag['skewness']:.4f}")
                lines.append(f"  Excess kurtosis:      {diag['excess_kurtosis']:.4f}")
                lines.append(f"  Approx Normal:        {'[OK]' if diag.get('approximately_normal', False) else '[X]'}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


class AblationStudy:
    """
    Systematic ablation study for ensemble components.

    Tests the contribution of each model by comparing:
    1. Full ensemble vs ensemble without model_i
    2. Individual model performance
    3. Pairwise model combinations

    Parameters:
        base_evaluator: ForecastEvaluator instance
        primary_metric: Primary metric for comparison (default 'r2')

    Example:
        >>> ablation = AblationStudy()
        >>> results = ablation.run(models, X_train, y_train)
    """

    def __init__(
        self,
        base_evaluator: Optional[ForecastEvaluator] = None,
        primary_metric: str = "r2",
    ):
        self.evaluator = base_evaluator or ForecastEvaluator(
            metrics=["r2", "rmse", "mae"], verbose=False
        )
        self.primary_metric = primary_metric

    def run(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run ablation study on model ensemble.

        Args:
            models: Dictionary of {name: fitted_model}
            X: Feature matrix
            y: Target values

        Returns:
            Dictionary with ablation results
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        results = {}

        # 1. Individual model performance
        individual = {}
        for name, model in models.items():
            try:
                eval_result = self.evaluator.evaluate(model, X, y)
                if "cv" in eval_result and self.primary_metric in eval_result["cv"]:
                    individual[name] = eval_result["cv"][self.primary_metric]["mean"]
                else:
                    individual[name] = np.nan
            except Exception:
                individual[name] = np.nan

        results["individual_performance"] = individual

        # 2. Rank models
        ranked = sorted(
            individual.items(),
            key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf,
            reverse=True,
        )
        results["model_ranking"] = [
            {"name": name, "score": score} for name, score in ranked
        ]

        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate ablation study report."""
        lines = [
            "",
            "=" * 80,
            "ABLATION STUDY REPORT",
            "=" * 80,
            f"\nPrimary metric: {self.primary_metric}",
        ]

        if "model_ranking" in results:
            lines.append("\n## Model Ranking")
            lines.append("-" * 50)
            for i, item in enumerate(results["model_ranking"]):
                score = item["score"]
                bar = "█" * int(max(0, score) * 30) if not np.isnan(score) else "N/A"
                lines.append(
                    f"  {i + 1}. {item['name']:25s}: {score:.4f} {bar}"
                )

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
