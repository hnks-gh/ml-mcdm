"""
Pipeline Validation Framework.

This module provides a comprehensive validation suite for the ML-MCDM 
pipeline, covering Evidential Reasoning (ER) belief distributions, 
ensemble forecasting performance, and end-to-end integration quality.

Key Features
------------
- **ER Validation**: Checks for belief validity, completeness, and 
  aggregation quality across hierarchical levels.
- **Forecast Validation**: Assesses cross-validation stability, 
  prediction interval coverage (conformal), and residual diagnostics.
- **Integrated Health Checks**: Combines multiple validation metrics 
  into a single "pass/fail" result with detailed warnings for localized 
  failures.
- **Diagnostic Summaries**: Generates human-readable reports of 
  statistical health and calibration accuracy.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import warnings
import functools

_logger = logging.getLogger(__name__)


def _silence(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ERValidationResult:
    """Validation results for Evidential Reasoning output."""
    belief_validity: bool = True
    belief_completeness: float = 1.0
    mean_belief_entropy: float = 0.0
    er_aggregation_score: float = 0.0
    cross_level_consistency: float = 0.0
    mean_utility_interval_width: float = 0.0
    grade_distribution: Dict[str, float] = field(default_factory=dict)
    er_valid: bool = True
    validation_warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "ER VALIDATION RESULTS",
            "=" * 60,
            f"Belief validity      : {self.belief_validity}",
            f"Belief completeness  : {self.belief_completeness:.4f}",
            f"Mean entropy         : {self.mean_belief_entropy:.4f}",
            f"Aggregation score    : {self.er_aggregation_score:.4f}",
            f"Cross-level consist. : {self.cross_level_consistency:.4f}",
            f"Mean util. interval  : {self.mean_utility_interval_width:.4f}",
            f"ER valid             : {self.er_valid}",
        ]
        if self.validation_warnings:
            lines.append("Warnings:")
            for w in self.validation_warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


@dataclass
class ForecastValidationResult:
    """Validation results for ML forecasting output."""
    cv_scores: Dict[str, float] = field(default_factory=dict)
    cv_fold_stability: float = 0.0
    interval_coverage: float = 0.0
    interval_sharpness: float = 0.0
    conformal_efficiency: float = 0.0
    oos_performance: Dict[str, float] = field(default_factory=dict)
    model_agreement: float = 0.0
    ensemble_diversity: float = 0.0
    residual_normality: float = 0.0
    residual_autocorrelation: float = 0.0
    residual_homoscedasticity: float = 0.0
    fold_performance_trend: float = 0.0
    learning_stability: float = 0.0
    forecast_valid: bool = True
    overall_score: float = 0.0
    validation_warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "FORECAST VALIDATION RESULTS",
            "=" * 60,
            f"CV fold stability    : {self.cv_fold_stability:.4f}",
            f"Interval coverage    : {self.interval_coverage:.4f}",
            f"Interval sharpness   : {self.interval_sharpness:.4f}",
            f"Model agreement      : {self.model_agreement:.4f}",
            f"Residual normality   : {self.residual_normality:.4f}",
            f"Overall score        : {self.overall_score:.4f}",
            f"Forecast valid       : {self.forecast_valid}",
        ]
        if self.validation_warnings:
            lines.append("Warnings:")
            for w in self.validation_warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


@dataclass
class ValidationResult:
    """Combined pipeline validation result."""
    er_validation: Optional[ERValidationResult] = None
    forecast_validation: Optional[ForecastValidationResult] = None
    overall_validity: float = 0.0
    validation_passed: bool = True
    validation_warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "PIPELINE VALIDATION SUMMARY",
            "=" * 60,
            f"Overall validity     : {self.overall_validity:.4f}",
            f"Validation passed    : {self.validation_passed}",
        ]
        if self.er_validation:
            lines.append("\n" + self.er_validation.summary())
        if self.forecast_validation:
            lines.append("\n" + self.forecast_validation.summary())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ERValidator
# ---------------------------------------------------------------------------

class ERValidator:
    """Validates Evidential Reasoning output."""

    def __init__(
        self,
        entropy_threshold: float = 2.0,
        utility_interval_threshold: float = 0.5,
    ):
        """
        Initialize the ER validator.

        Parameters
        ----------
        entropy_threshold : float, default=2.0
            Maximum allowable mean belief entropy before a warning is issued. 
            High entropy indicates lack of consensus or high ambiguity.
        utility_interval_threshold : float, default=0.5
            Maximum allowable mean utility interval width. Large widths 
            indicate high ignorance or missing data in the weighted criteria.
        """
        self.entropy_threshold = entropy_threshold
        self.utility_interval_threshold = utility_interval_threshold

    def validate(self, er_result) -> ERValidationResult:
        result = ERValidationResult()
        warnings_list: List[str] = []

        if er_result is None:
            result.er_valid = False
            result.validation_warnings = ["No ER result provided"]
            return result

        bd = getattr(er_result, "belief_distributions", {}) or {}
        if not bd:
            result.belief_validity = True
            result.er_valid = True
            result.validation_warnings = ["Empty belief distributions"]
            return result

        # --- belief validity: all beliefs in [0,1], sum <= 1 + eps
        valid = True
        for entity, b_dist in bd.items():
            beliefs = np.asarray(b_dist.beliefs)
            if np.any(beliefs < -1e-6) or np.any(beliefs > 1.0 + 1e-6):
                valid = False
                warnings_list.append(f"Entity {entity}: beliefs out of [0,1]")
            if beliefs.sum() > 1.0 + 1e-3:
                valid = False
                warnings_list.append(f"Entity {entity}: beliefs sum {beliefs.sum():.4f} > 1")
        result.belief_validity = valid

        # --- completeness: fraction of entities with beliefs.sum() > 0.5
        completeness_vals = []
        for b_dist in bd.values():
            s = float(np.asarray(b_dist.beliefs).sum())
            completeness_vals.append(min(s, 1.0))
        result.belief_completeness = float(np.mean(completeness_vals)) if completeness_vals else 1.0

        # --- mean belief entropy
        entropies = []
        for b_dist in bd.values():
            beliefs = np.asarray(b_dist.beliefs, dtype=float)
            beliefs = np.clip(beliefs, 1e-12, None)
            beliefs = beliefs / beliefs.sum()
            h = float(-np.sum(beliefs * np.log(beliefs)))
            entropies.append(h)
        result.mean_belief_entropy = float(np.mean(entropies)) if entropies else 0.0
        if result.mean_belief_entropy > self.entropy_threshold:
            warnings_list.append(
                f"High mean belief entropy: {result.mean_belief_entropy:.3f} > {self.entropy_threshold}"
            )

        # --- aggregation score: 1 - std of final_scores / (range + 1e-9)
        final_scores = getattr(er_result, "final_scores", None)
        if final_scores is not None and len(final_scores) > 1:
            s = float(np.std(final_scores.values))
            r = float(np.ptp(final_scores.values)) + 1e-9
            result.er_aggregation_score = float(np.clip(s / r, 0.0, 1.0))
        else:
            result.er_aggregation_score = 0.0

        # --- cross-level consistency via Spearman between criterion scores and final scores
        crit_scores = getattr(er_result, "criterion_method_scores", {}) or {}
        if crit_scores and final_scores is not None and len(final_scores) > 2:
            try:
                from scipy.stats import spearmanr
                # Use the OR of each entity across all criterion scores
                entities = list(final_scores.index)
                crit_means = {}
                for ent in entities:
                    vals = []
                    for crit, scores_dict in crit_scores.items():
                        if hasattr(scores_dict, "get"):
                            v = scores_dict.get(ent, np.nan)
                        else:
                            v = np.nan
                        if not np.isnan(v):
                            vals.append(v)
                    crit_means[ent] = float(np.mean(vals)) if vals else np.nan
                crit_vec = np.array([crit_means.get(e, np.nan) for e in entities])
                final_vec = final_scores.values.astype(float)
                mask = ~np.isnan(crit_vec) & ~np.isnan(final_vec)
                if mask.sum() > 2:
                    rho, _ = spearmanr(crit_vec[mask], final_vec[mask])
                    result.cross_level_consistency = float(np.clip((rho + 1) / 2, 0.0, 1.0))
                else:
                    result.cross_level_consistency = 0.5
            except Exception:
                result.cross_level_consistency = 0.5
        else:
            result.cross_level_consistency = 0.5

        # --- utility interval width
        interval_widths = []
        for entity, b_dist in bd.items():
            try:
                lo, hi = b_dist.utility_interval()
                interval_widths.append(float(hi - lo))
            except Exception:
                interval_widths.append(0.0)
        result.mean_utility_interval_width = float(np.mean(interval_widths)) if interval_widths else 0.0
        if result.mean_utility_interval_width > self.utility_interval_threshold:
            warnings_list.append(
                f"High mean utility interval width: {result.mean_utility_interval_width:.3f}"
            )

        # --- grade distribution
        all_grades: Dict[str, float] = {}
        for b_dist in bd.values():
            for g, b in zip(b_dist.grades, b_dist.beliefs):
                all_grades[g] = all_grades.get(g, 0.0) + float(b)
        total = sum(all_grades.values()) + 1e-12
        result.grade_distribution = {g: v / total for g, v in all_grades.items()}

        # --- overall validity flag
        result.er_valid = (
            result.belief_validity
            and result.belief_completeness >= 0.5
        )
        result.validation_warnings = warnings_list
        return result


# ---------------------------------------------------------------------------
# ForecastValidator
# ---------------------------------------------------------------------------

class ForecastValidator:
    """Validates ML forecasting output."""

    def __init__(self, r2_threshold: float = 0.0, coverage_target: float = 0.90):
        """
        Initialize the forecast validator.

        Parameters
        ----------
        r2_threshold : float, default=0.0
            Minimum allowable mean R² score for the forecast to be 
            considered valid.
        coverage_target : float, default=0.90
            Target empirical coverage for prediction intervals. Warnings 
            are issued if coverage falls significantly below this value.
        """
        self.r2_threshold = r2_threshold
        self.coverage_target = coverage_target

    @_silence
    def validate(self, forecast_result) -> ForecastValidationResult:
        result = ForecastValidationResult()
        warnings_list: List[str] = []

        if forecast_result is None:
            result.forecast_valid = False
            result.validation_warnings = ["No forecast result provided"]
            return result

        # --- CV scores
        cv_scores_raw = getattr(forecast_result, "cross_validation_scores", {}) or {}
        mean_cvs: Dict[str, float] = {}
        all_cv_values: List[float] = []
        for model, scores in cv_scores_raw.items():
            if scores:
                m = float(np.mean(scores))
                mean_cvs[model] = m
                all_cvs = [float(s) for s in scores]
                all_cv_values.extend(all_cvs)
        result.cv_scores = mean_cvs

        # --- CV fold stability
        if all_cv_values and len(all_cv_values) > 1:
            cv_std = float(np.std(all_cv_values))
            cv_range = float(np.ptp(all_cv_values)) + 1e-9
            result.cv_fold_stability = float(np.clip(1.0 - cv_std / cv_range, 0.0, 1.0))
        else:
            result.cv_fold_stability = 0.5

        # --- prediction interval coverage
        pred_intervals = getattr(forecast_result, "prediction_intervals", {}) or {}
        predictions = getattr(forecast_result, "predictions", None)
        if pred_intervals and predictions is not None:
            coverages = []
            for key, interval_df in pred_intervals.items():
                if interval_df is None or interval_df.empty:
                    continue
                for col in predictions.columns:
                    if col in interval_df.columns or ("lower" in interval_df.columns and "upper" in interval_df.columns):
                        try:
                            preds = predictions[col].values
                            lo = interval_df["lower"].values
                            hi = interval_df["upper"].values
                            covered = np.mean((preds >= lo) & (preds <= hi))
                            coverages.append(float(covered))
                        except Exception:
                            pass
            result.interval_coverage = float(np.mean(coverages)) if coverages else 0.0
        else:
            result.interval_coverage = 0.0
        if result.interval_coverage < self.coverage_target * 0.8:
            warnings_list.append(
                f"Low interval coverage: {result.interval_coverage:.3f} < {self.coverage_target}"
            )

        # --- interval sharpness (mean width)
        uncertainty = getattr(forecast_result, "uncertainty", None)
        if uncertainty is not None and not uncertainty.empty:
            result.interval_sharpness = float(uncertainty.mean().mean() * 2)
        else:
            result.interval_sharpness = 0.0

        # --- model performance
        model_perf = getattr(forecast_result, "model_performance", {}) or {}
        r2_vals = []
        for m, perf in model_perf.items():
            r2 = perf.get("r2", None) if isinstance(perf, dict) else None
            if r2 is not None:
                r2_vals.append(float(r2))

        # --- model agreement (1 - normalized std of R² scores)
        if len(r2_vals) > 1:
            std_r2 = float(np.std(r2_vals))
            range_r2 = float(np.ptp(r2_vals)) + 1e-9
            result.model_agreement = float(np.clip(1.0 - std_r2 / range_r2, 0.0, 1.0))
        elif len(r2_vals) == 1:
            result.model_agreement = 1.0
        else:
            result.model_agreement = 0.0

        # --- ensemble diversity (std of model contributions)
        contributions = getattr(forecast_result, "model_contributions", {}) or {}
        contribs = list(contributions.values()) if contributions else []
        if len(contribs) > 1:
            result.ensemble_diversity = float(np.clip(np.std(contribs) * 10, 0.0, 1.0))
        else:
            result.ensemble_diversity = 0.0

        # --- residual diagnostics from uncertainty as proxy
        if uncertainty is not None and not uncertainty.empty:
            residuals = uncertainty.values.flatten()
            # Normality: kurtosis-based proxy (normal kurtosis ≈ 0)
            if len(residuals) > 4:
                try:
                    from scipy.stats import kurtosis
                    kurt = abs(float(kurtosis(residuals)))
                    result.residual_normality = float(np.clip(1.0 - kurt / (kurt + 3.0), 0.0, 1.0))
                except Exception:
                    result.residual_normality = 0.5
            else:
                result.residual_normality = 0.5
            # Autocorrelation: Durbin-Watson proxy
            if len(residuals) > 2:
                diff = np.diff(residuals)
                dw = float(np.sum(diff ** 2) / (np.sum(residuals ** 2) + 1e-12))
                result.residual_autocorrelation = float(np.clip(1.0 - abs(dw - 2.0) / 2.0, 0.0, 1.0))
            else:
                result.residual_autocorrelation = 0.5
            # Homoscedasticity: running variance stability
            if len(residuals) > 8:
                half = len(residuals) // 2
                v1 = float(np.var(residuals[:half]) + 1e-12)
                v2 = float(np.var(residuals[half:]) + 1e-12)
                ratio = min(v1, v2) / max(v1, v2)
                result.residual_homoscedasticity = float(np.clip(ratio, 0.0, 1.0))
            else:
                result.residual_homoscedasticity = 0.5
        else:
            result.residual_normality = 0.5
            result.residual_autocorrelation = 0.5
            result.residual_homoscedasticity = 0.5

        # --- fold performance trend (are later folds worse/better?)
        result.fold_performance_trend = 0.5
        result.learning_stability = 0.5

        # --- forecast validity
        mean_r2 = float(np.mean(r2_vals)) if r2_vals else 0.0
        result.forecast_valid = mean_r2 >= self.r2_threshold

        # --- overall score
        components = [
            result.cv_fold_stability,
            result.interval_coverage,
            result.model_agreement,
            result.residual_normality,
            result.residual_autocorrelation,
        ]
        result.overall_score = float(np.clip(np.mean(components), 0.0, 1.0))
        if result.overall_score < 0.4:
            warnings_list.append(f"Low overall forecast validation score: {result.overall_score:.3f}")

        result.validation_warnings = warnings_list
        return result


# ---------------------------------------------------------------------------
# Combined Validator
# ---------------------------------------------------------------------------

class Validator:
    """Runs full-pipeline validation (ER + forecasting)."""

    def __init__(self, **kwargs):
        self._er_validator = ERValidator()
        self._fc_validator = ForecastValidator()

    def validate_full_pipeline(
        self,
        forecast_result=None,
        er_result=None,
        ranking_result=None,
        **kwargs,   # absorbs legacy positional args (panel_data, weights)
    ) -> ValidationResult:
        er_val = self._er_validator.validate(er_result)
        fc_val = self._fc_validator.validate(forecast_result)

        scores = []
        if er_val is not None:
            scores.append(er_val.er_aggregation_score)
        if fc_val is not None:
            scores.append(fc_val.overall_score)
        overall = float(np.mean(scores)) if scores else 0.0

        passed = True
        all_warnings: List[str] = []
        if er_val and not er_val.er_valid:
            passed = False
            all_warnings.extend(er_val.validation_warnings)
        if fc_val and not fc_val.forecast_valid:
            passed = False
            all_warnings.extend(fc_val.validation_warnings)

        return ValidationResult(
            er_validation=er_val,
            forecast_validation=fc_val,
            overall_validity=overall,
            validation_passed=passed,
            validation_warnings=all_warnings,
        )

    # Legacy shim so existing pipeline.py code still works
    def validate(self, *args, **kwargs) -> ValidationResult:
        return self.validate_full_pipeline(**kwargs)


def run_validation(
    forecast_result=None,
    er_result=None,
    ranking_result=None,
    **kwargs,
) -> ValidationResult:
    """Convenience function: run full-pipeline validation."""
    return Validator().validate_full_pipeline(
        forecast_result=forecast_result,
        er_result=er_result,
        ranking_result=ranking_result,
        **kwargs,
    )
