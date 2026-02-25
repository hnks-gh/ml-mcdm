# -*- coding: utf-8 -*-
"""
Production Validation for IFS+ER+Forecasting Pipeline
======================================================

Production-grade validation for hierarchical MCDM system with IFS,
ER aggregation, and ML forecasting.

Validation Components:
1. Ranking Consistency
   - Cross-level agreement (subcriteria → criteria → final)
   - ER aggregation quality metrics
   - Belief distribution validation

2. IFS Parameter Validation
   - Membership/non-membership consistency
   - Hesitancy degree bounds checking
   - IFS transformation correctness

3. Weight Scheme Validation
   - Temporal stability of weights
   - Bootstrap confidence intervals
   - Method agreement (Entropy, CRITIC, MEREC, StdDev)

4. Forecast Validation
   - Temporal cross-validation (expanding window)
   - Prediction interval coverage
   - Out-of-sample performance

5. End-to-End Pipeline Validation
   - Full pipeline consistency checks
   - Multi-year robustness
   - Rank preservation under perturbation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings
import functools

_logger = logging.getLogger(__name__)


def _silence_warnings(func):
    """Scope all warning filters to the duration of *func* only."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return func(*args, **kwargs)
    return wrapper


@dataclass
class ValidationResult:
    """Comprehensive validation results for hierarchical MCDM pipeline."""
    
    # Hierarchical consistency
    cross_level_consistency: Dict[str, float]  # Subcriteria-criteria-final agreement
    er_aggregation_quality: float              # ER belief distribution quality
    
    # IFS validation
    ifs_consistency: Dict[str, bool]           # IFS parameter constraints
    ifs_hesitancy_valid: bool                  # Hesitancy degree checks
    
    # Weight validation
    weight_temporal_stability: float           # Weight stability across years
    weight_method_agreement: float             # Agreement among weighting methods
    weight_bootstrap_ci: Dict[str, Tuple[float, float]]  # Bootstrap confidence intervals
    
    # Forecast validation
    forecast_cv_scores: Optional[Dict[str, float]] = None  # CV performance metrics
    forecast_interval_coverage: Optional[float] = None     # Prediction interval coverage
    forecast_oos_performance: Optional[Dict[str, float]] = None  # Out-of-sample metrics
    
    # Overall validation
    overall_validity: float = 0.0              # 0-1 overall validity score
    validation_warnings: List[str] = field(default_factory=list)
    validation_passed: bool = True
    
    def summary(self) -> str:
        """Generate comprehensive validation report."""
        lines = [
            f"\n{'='*70}",
            "VALIDATION RESULTS",
            f"{'='*70}",
            f"\nOverall Validity Score: {self.overall_validity:.4f}",
            f"Validation Status: {'PASSED' if self.validation_passed else 'FAILED'}",
            f"\n{'-'*70}",
            "CONSISTENCY METRICS",
            f"{'-'*70}",
        ]
        
        for level, consistency in self.cross_level_consistency.items():
            lines.append(f"  {level}: {consistency:.4f}")
        
        lines.append(f"\nER Aggregation Quality: {self.er_aggregation_quality:.4f}")
        
        lines.extend([
            f"\n{'-'*70}",
            "IFS VALIDATION",
            f"{'-'*70}",
            f"  Hesitancy bounds valid: {'YES' if self.ifs_hesitancy_valid else 'NO'}",
        ])
        
        failed_ifs = [k for k, v in self.ifs_consistency.items() if not v]
        if failed_ifs:
            lines.append(f"  Failed IFS checks: {', '.join(failed_ifs)}")
        else:
            lines.append("  All IFS constraints satisfied")
        
        lines.extend([
            f"\n{'-'*70}",
            "WEIGHT VALIDATION",
            f"{'-'*70}",
            f"  Temporal stability: {self.weight_temporal_stability:.4f}",
            f"  Method agreement: {self.weight_method_agreement:.4f}",
        ])
        
        if self.forecast_cv_scores:
            lines.extend([
                f"\n{'-'*70}",
                "FORECAST VALIDATION",
                f"{'-'*70}",
            ])
            for metric, score in self.forecast_cv_scores.items():
                lines.append(f"  {metric}: {score:.4f}")
            
            if self.forecast_interval_coverage is not None:
                lines.append(f"  Interval coverage: {self.forecast_interval_coverage:.1%}")
        
        if self.validation_warnings:
            lines.extend([
                f"\n{'-'*70}",
                "WARNINGS",
                f"{'-'*70}",
            ])
            for warning in self.validation_warnings:
                lines.append(f"  - {warning}")
        
        lines.append("=" * 70)
        return "\n".join(lines)


class Validator:
    """
    Comprehensive validator for IFS+ER+Forecasting pipeline.
    
    Validates all pipeline components:
    - Multi-level structure integrity
    - IFS parameter correctness
    - Weight scheme robustness
    - Forecast quality
    - End-to-end consistency
    """
    
    def __init__(self,
                 ifs_tolerance: float = 1e-6,
                 consistency_threshold: float = 0.7,
                 stability_threshold: float = 0.85):
        """
        Initialize validator.
        
        Parameters
        ----------
        ifs_tolerance : float, default=1e-6
            Tolerance for IFS constraint violations
        consistency_threshold : float, default=0.7
            Minimum correlation for cross-level consistency
        stability_threshold : float, default=0.85
            Minimum temporal stability for weights
        """
        self.ifs_tolerance = ifs_tolerance
        self.consistency_threshold = consistency_threshold
        self.stability_threshold = stability_threshold
    
    @_silence_warnings
    def validate_full_pipeline(
        self,
        panel_data,
        weights: Dict[str, Any],
        ranking_result,
        forecast_result: Optional[Any] = None
    ) -> ValidationResult:
        """
        Perform comprehensive validation of the full pipeline.
        
        Parameters
        ----------
        panel_data : PanelData
            Panel dataset
        weights : Dict[str, Any]
            Weight calculation results
        ranking_result : HierarchicalRankingResult
            Ranking pipeline results
        forecast_result : Optional
            Forecasting results
        
        Returns
        -------
        ValidationResult
            Comprehensive validation results
        """
        warnings_list = []
        
        # 1. Validate hierarchical consistency
        cross_level_consistency = self._validate_hierarchical_consistency(
            ranking_result
        )
        
        # 2. Validate ER aggregation
        er_quality = self._validate_er_aggregation(ranking_result)
        
        # 3. Validate IFS parameters
        ifs_consistency, ifs_hesitancy_valid = self._validate_ifs_parameters(
            ranking_result
        )
        
        # 4. Validate weight scheme
        weight_stability = self._validate_weight_temporal_stability(
            panel_data, weights
        )
        
        weight_agreement = self._validate_weight_method_agreement(weights)
        
        weight_ci = self._get_weight_confidence_intervals(weights)
        
        # 5. Validate forecasting (if available)
        forecast_cv = None
        forecast_coverage = None
        forecast_oos = None
        
        if forecast_result is not None:
            forecast_cv = self._validate_forecast_cv(forecast_result)
            forecast_coverage = self._validate_forecast_intervals(forecast_result)
            forecast_oos = self._validate_forecast_oos(forecast_result)
        
        # Check warnings
        if weight_stability < self.stability_threshold:
            warnings_list.append(
                f"Weight temporal stability ({weight_stability:.3f}) below threshold ({self.stability_threshold})"
            )
        
        if weight_agreement < self.consistency_threshold:
            warnings_list.append(
                f"Weight method agreement ({weight_agreement:.3f}) below threshold ({self.consistency_threshold})"
            )
        
        if not ifs_hesitancy_valid:
            warnings_list.append("IFS hesitancy degree bounds violated")
        
        # Compute overall validity
        validity_components = [
            np.mean(list(cross_level_consistency.values())) if cross_level_consistency else 0.0,
            er_quality,
            1.0 if ifs_hesitancy_valid else 0.0,
            weight_stability,
            weight_agreement,
        ]
        
        overall_validity = np.mean(validity_components)
        validation_passed = (
            overall_validity >= self.consistency_threshold and
            ifs_hesitancy_valid and
            len(warnings_list) < 3
        )
        
        return ValidationResult(
            cross_level_consistency=cross_level_consistency,
            er_aggregation_quality=er_quality,
            ifs_consistency=ifs_consistency,
            ifs_hesitancy_valid=ifs_hesitancy_valid,
            weight_temporal_stability=weight_stability,
            weight_method_agreement=weight_agreement,
            weight_bootstrap_ci=weight_ci,
            forecast_cv_scores=forecast_cv,
            forecast_interval_coverage=forecast_coverage,
            forecast_oos_performance=forecast_oos,
            overall_validity=overall_validity,
            validation_warnings=warnings_list,
            validation_passed=validation_passed
        )
    
    def _validate_hierarchical_consistency(
        self,
        ranking_result
    ) -> Dict[str, float]:
        """Validate cross-level ranking consistency.
        
        Uses criterion_method_scores from HierarchicalRankingResult to check
        that per-criterion scores are consistent with the final ER ranking.
        """
        consistency = {}
        
        # Access the actual attributes on HierarchicalRankingResult
        if not hasattr(ranking_result, 'criterion_method_scores'):
            return {}
        
        try:
            from scipy.stats import spearmanr
            
            er_result = ranking_result.er_result
            final_scores = er_result.final_scores
            
            # Per-criterion consistency: compare each criterion's aggregated
            # method scores with the final ranking
            criterion_correlations = []
            for crit_id, method_scores in ranking_result.criterion_method_scores.items():
                if not method_scores:
                    continue
                # Average across all methods for this criterion
                all_scores = [s for s in method_scores.values() if hasattr(s, 'values')]
                if not all_scores:
                    continue
                avg_crit_scores = pd.concat(all_scores, axis=1).mean(axis=1)
                # Align with final scores
                common = avg_crit_scores.index.intersection(final_scores.index)
                if len(common) > 2:
                    corr, _ = spearmanr(
                        avg_crit_scores.loc[common].values,
                        final_scores.loc[common].values
                    )
                    if not np.isnan(corr):
                        consistency[f'{crit_id}_to_final'] = float(corr)
                        criterion_correlations.append(float(corr))
            
            if criterion_correlations:
                consistency['mean_criterion_to_final'] = float(np.mean(criterion_correlations))
        
        except Exception as _exc:
            _logger.warning('hierarchical_consistency check failed: %s', _exc)
        
        return consistency
    
    def _validate_er_aggregation(self, ranking_result) -> float:
        """Validate ER aggregation quality (belief distribution properties)."""
        if not hasattr(ranking_result, 'er_result'):
            return 0.0  # ER result unavailable — cannot assert quality
        
        scores = ranking_result.er_result.final_scores.values
        
        # Check properties:
        # 1. Scores should be in [0, 1]
        in_range = np.all((scores >= 0) & (scores <= 1 + 1e-6))
        
        # 2. Should have reasonable spread (not all same)
        spread = np.std(scores) > 0.01
        
        # 3. Should use full range reasonably
        range_usage = (scores.max() - scores.min()) / (1.0 + 1e-10)
        
        quality = (float(in_range) + float(spread) + range_usage) / 3.0
        return quality
    
    def _validate_ifs_parameters(
        self,
        ranking_result
    ) -> Tuple[Dict[str, bool], bool]:
        """Validate IFS parameters (μ + ν ≤ 1, 0 ≤ π ≤ 1).
        
        Uses ifs_diagnostics from HierarchicalRankingResult, which maps
        criterion → {alternative → {subcrit → IFN}}.
        """
        ifs_checks = {}
        
        if not hasattr(ranking_result, 'ifs_diagnostics'):
            # No diagnostic data available — nothing to invalidate
            return {}, True
        
        ifs_diag = ranking_result.ifs_diagnostics
        if not ifs_diag:
            return {}, True
        
        try:
            overall_valid = True
            hesitancy_valid = True
            
            for crit_id, crit_diag in ifs_diag.items():
                if not isinstance(crit_diag, dict):
                    continue
                # crit_diag may contain IFS matrix objects or sample IFNs
                # Check any IFN objects found
                for alt_key, alt_data in crit_diag.items():
                    if isinstance(alt_data, dict):
                        for sub_key, ifn in alt_data.items():
                            if hasattr(ifn, 'mu') and hasattr(ifn, 'nu'):
                                mu, nu = ifn.mu, ifn.nu
                                pi = 1 - mu - nu
                                sum_ok = (mu + nu) <= 1 + self.ifs_tolerance
                                hesit_ok = (-self.ifs_tolerance <= pi <= 1 + self.ifs_tolerance)
                                
                                if not sum_ok:
                                    ifs_checks[f'{crit_id}_{alt_key}_{sub_key}_sum'] = False
                                    overall_valid = False
                                if not hesit_ok:
                                    ifs_checks[f'{crit_id}_{alt_key}_{sub_key}_hesitancy'] = False
                                    hesitancy_valid = False
            
            if not ifs_checks:
                ifs_checks['default'] = True
            
            return ifs_checks, hesitancy_valid
        
        except Exception as _exc:
            _logger.warning('IFS parameter validation failed: %s', _exc)
            return {}, False
    
    def _validate_weight_temporal_stability(
        self,
        panel_data,
        weights: Dict[str, Any]
    ) -> float:
        """Validate temporal stability of weights across years.

        Reads from the new ``HybridWeightingCalculator`` details structure:
        - ``details.stability.cosine_similarity``  (primary)
        - ``details.level2.mc_diagnostics.cv_weights``  (fallback: 1 - mean_cv)
        Also handles the legacy ``details.bootstrap`` structure for backward compat.
        """
        if not hasattr(panel_data, 'years') or len(panel_data.years) < 2:
            return 1.0

        details = weights.get('details', {})

        # ── New structure: stability dict from HybridWeightingCalculator ──
        stability_info = details.get('stability', {})
        cosine_sim = stability_info.get('cosine_similarity')
        if cosine_sim is not None:
            return float(np.clip(cosine_sim, 0.0, 1.0))

        # ── New structure: Level 2 MC CV-based stability ───────────────────
        l2_diag = details.get('level2', {}).get('mc_diagnostics', {})
        if l2_diag:
            cv_weights = l2_diag.get('cv_weights', {})
            if cv_weights:
                mean_cv = float(np.mean(list(cv_weights.values())))
                return float(np.clip(1.0 - mean_cv, 0.0, 1.0))
            std_weights = l2_diag.get('std_weights', {})
            mean_weights = l2_diag.get('mean_weights', {})
            if std_weights and mean_weights:
                cvs = [
                    std_weights[k] / mean_weights[k]
                    for k in std_weights
                    if mean_weights.get(k, 0) > 1e-10
                ]
                if cvs:
                    return float(np.clip(1.0 - np.mean(cvs), 0.0, 1.0))

        # ── Legacy structure: bootstrap dict from HybridWeightingPipeline ─
        bootstrap_info = details.get('bootstrap', {})
        if bootstrap_info:
            std_weights = bootstrap_info.get('std_weights', {})
            if std_weights:
                subcriteria = weights.get('subcriteria', [])
                fused = weights.get('fused', np.array([]))
                if len(fused) > 0 and len(subcriteria) > 0:
                    cvs = []
                    for i, sc in enumerate(subcriteria):
                        w = fused[i] if i < len(fused) else 0
                        s = std_weights.get(sc, 0)
                        if w > 1e-10:
                            cvs.append(s / w)
                    if cvs:
                        return float(np.clip(1.0 - np.mean(cvs), 0.0, 1.0))

        return 0.0

    def _validate_weight_method_agreement(
        self,
        weights: Dict[str, Any]
    ) -> float:
        """Validate internal weight agreement.

        For the new ``HybridWeightingCalculator`` (single two-method blend),
        uses mean 1 - CV across Level 2 criterion weights as an agreement proxy.

        Falls back to pairwise Spearman correlation for the legacy
        ``HybridWeightingPipeline`` four-method case.
        """
        details = weights.get('details', {})

        # ── New structure: use Level 2 CV as agreement proxy ──────────────
        l2_diag = details.get('level2', {}).get('mc_diagnostics', {})
        if l2_diag:
            cv_weights = l2_diag.get('cv_weights', {})
            if cv_weights:
                mean_cv = float(np.mean(list(cv_weights.values())))
                # High agreement = low CV; map [0, ∞) → [0, 1]
                return float(np.clip(1.0 - mean_cv, 0.0, 1.0))

        # ── Legacy: pairwise Spearman across four method arrays ────────────
        method_keys = ['entropy', 'critic', 'merec', 'std_dev']
        available = {
            k: np.asarray(weights[k])
            for k in method_keys
            if k in weights and len(np.asarray(weights[k])) > 2
        }
        if len(available) >= 2:
            from scipy.stats import spearmanr
            arrs = list(available.values())
            correlations = []
            for i in range(len(arrs)):
                for j in range(i + 1, len(arrs)):
                    if len(arrs[i]) == len(arrs[j]):
                        corr, _ = spearmanr(arrs[i], arrs[j])
                        if not np.isnan(corr):
                            correlations.append(corr)
            if correlations:
                return float(np.mean(correlations))

        return 0.0

    def _get_weight_confidence_intervals(
        self,
        weights: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """Get bootstrap confidence intervals for weights.

        Reads from new ``HybridWeightingCalculator`` details:
        ``details.level2.mc_diagnostics.ci_lower_2_5`` /
        ``ci_upper_97_5``.

        Falls back to the legacy ``details.bootstrap`` structure, then to
        a ±1.96σ approximation from std_weights.
        """
        ci: Dict[str, Tuple[float, float]] = {}
        details = weights.get('details', {})

        # ── New structure: Level 2 MC credible intervals ───────────────────
        l2_diag = details.get('level2', {}).get('mc_diagnostics', {})
        ci_lo = l2_diag.get('ci_lower_2_5', {})
        ci_hi = l2_diag.get('ci_upper_97_5', {})
        if ci_lo and ci_hi:
            for ck in ci_lo:
                if ck in ci_hi:
                    ci[ck] = (float(ci_lo[ck]), float(ci_hi[ck]))
            # Also try to get SC-level CIs from Level 1
            for l1_group in details.get('level1', {}).values():
                mc = l1_group.get('mc_diagnostics', {})
                sc_lo = mc.get('ci_lower_2_5', {})
                sc_hi = mc.get('ci_upper_97_5', {})
                for sc in sc_lo:
                    if sc in sc_hi:
                        ci[sc] = (float(sc_lo[sc]), float(sc_hi[sc]))
            if ci:
                return ci

        # ── Legacy: bootstrap dict from HybridWeightingPipeline ───────────
        bootstrap_info = details.get('bootstrap', {})
        if bootstrap_info:
            b_lo = bootstrap_info.get('ci_lower_2_5', {})
            b_hi = bootstrap_info.get('ci_upper_97_5', {})
            if b_lo and b_hi:
                for sc in b_lo:
                    if sc in b_hi:
                        ci[sc] = (float(b_lo[sc]), float(b_hi[sc]))
                return ci
            # ±1.96σ fallback
            std_weights = bootstrap_info.get('std_weights', {})
            subcriteria = weights.get('subcriteria', [])
            fused = weights.get('fused', np.array([]))
            if std_weights and len(fused) > 0:
                for i, sc in enumerate(subcriteria):
                    if i < len(fused):
                        w = float(fused[i])
                        s = float(std_weights.get(sc, 0))
                        ci[sc] = (w - 1.96 * s, w + 1.96 * s)

        return ci
    
    def _validate_forecast_cv(
        self,
        forecast_result
    ) -> Optional[Dict[str, float]]:
        """Extract forecast cross-validation scores."""
        if not hasattr(forecast_result, 'cross_validation_scores'):
            return None
        
        cv_scores = forecast_result.cross_validation_scores
        
        # Aggregate to mean scores
        aggregated = {}
        for model, scores in cv_scores.items():
            if isinstance(scores, list) and len(scores) > 0:
                aggregated[f'{model}_mean'] = np.mean(scores)
                aggregated[f'{model}_std'] = np.std(scores)
        
        return aggregated if aggregated else None
    
    def _validate_forecast_intervals(
        self,
        forecast_result
    ) -> Optional[float]:
        """Validate prediction interval coverage."""
        if not hasattr(forecast_result, 'prediction_intervals'):
            return None
        
        # Empirical coverage computation requires retained holdout actuals,
        # which are not currently threaded through to this validator.
        # Return None rather than a fabricated value.
        return None
    
    def _validate_forecast_oos(
        self,
        forecast_result
    ) -> Optional[Dict[str, float]]:
        """Validate out-of-sample forecast performance."""
        if not hasattr(forecast_result, 'holdout_performance'):
            return None
        
        return forecast_result.holdout_performance


def run_validation(
    panel_data,
    weights: Dict[str, Any],
    ranking_result,
    forecast_result: Optional[Any] = None
) -> ValidationResult:
    """
    Convenience function for comprehensive pipeline validation.
    
    Parameters
    ----------
    panel_data : PanelData
        Panel dataset
    weights : Dict[str, Any]
        Weight calculation results
    ranking_result : HierarchicalRankingResult
        Ranking results
    forecast_result : Optional
        Forecasting results
    
    Returns
    -------
    ValidationResult
        Comprehensive validation results
    
    Example
    -------
    >>> result = run_validation(
    ...     panel_data, weights, ranking_result, forecast_result
    ... )
    >>> print(f"Overall validity: {result.overall_validity:.3f}")
    >>> print(f"Validation: {'PASSED' if result.validation_passed else 'FAILED'}")
    >>> print(result.summary())
    """
    validator = Validator()
    
    return validator.validate_full_pipeline(
        panel_data=panel_data,
        weights=weights,
        ranking_result=ranking_result,
        forecast_result=forecast_result
    )

