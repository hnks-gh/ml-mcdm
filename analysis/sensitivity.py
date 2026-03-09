# -*- coding: utf-8 -*-
"""
Sensitivity Analysis for ML Forecasting and Evidential Reasoning
================================================================

Comprehensive sensitivity analysis focused on:

1. ML Forecast Sensitivity
   - Feature importance stability via bootstrap resampling
   - Leave-one-model-out (LOO) contribution impact
   - Prediction sensitivity per entity (uncertainty / range)
   - Temporal prediction stability across CV folds
   - Conformal prediction interval width sensitivity

2. Evidential Reasoning Sensitivity
   - Criterion belief distribution sensitivity (one-at-a-time OAT)
   - Grade threshold sensitivity (utility mapping perturbation)
   - ER aggregation weight sensitivity
   - Utility interval width per entity (residual ignorance)
   - Cross-level consistency (subcriteria → criteria → final ER)
   - Belief Shannon entropy analysis

References
----------
Saltelli et al. (2008). Global Sensitivity Analysis: The Primer.
Yang & Xu (2002). Evidential reasoning rule under uncertainty.
Gneiting & Raftery (2007). Strictly Proper Scoring Rules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings
import functools
import logging
import multiprocessing

logger = logging.getLogger(__name__)


def _silence_warnings(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return func(*args, **kwargs)
    return wrapper


# ============================================================================
# Result containers
# ============================================================================

@dataclass
class MLSensitivityResult:
    """Sensitivity analysis results for ML forecasting."""

    # Feature importance stability
    feature_importance_cv: Dict[str, float]           # bootstrap CV per feature
    feature_importance_mean: Dict[str, float]         # mean importance per feature
    feature_importance_rank_stability: Dict[str, float]  # rank stability per feature

    # Model contribution sensitivity
    loo_model_impact: Dict[str, float]                # performance drop on LOO
    model_contribution_cv: Dict[str, float]           # CV of model contributions

    # Prediction sensitivity
    prediction_sensitivity: Dict[str, float]          # unc/range per entity
    temporal_prediction_stability: float              # fold-to-fold Spearman

    # Conformal / interval sensitivity
    interval_width_cv: float                          # CV of interval widths
    interval_coverage_sensitivity: float              # 1 - interval_width_cv

    # Overall
    overall_robustness: float
    n_bootstrap: int

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            "ML FORECASTING SENSITIVITY ANALYSIS",
            f"{'='*70}",
            f"Overall Robustness: {self.overall_robustness:.4f}",
            f"Bootstrap samples:  {self.n_bootstrap}",
            f"\n{'-'*70}",
            "TOP 10 MOST SENSITIVE FEATURES (highest importance CV):",
            f"{'-'*70}",
        ]
        for feat, cv in sorted(
            self.feature_importance_cv.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            mean_imp = self.feature_importance_mean.get(feat, 0.0)
            lines.append(f"  {feat}: CV={cv:.4f}  mean_importance={mean_imp:.4f}")

        lines.extend([
            f"\n{'-'*70}",
            "MODEL CONTRIBUTION SENSITIVITY (LOO impact):",
            f"{'-'*70}",
        ])
        for model, impact in sorted(
            self.loo_model_impact.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  {model}: Δperformance={impact:.4f}")

        lines.extend([
            f"\n{'-'*70}",
            "PREDICTION METRICS:",
            f"{'-'*70}",
            f"  Temporal prediction stability: {self.temporal_prediction_stability:.4f}",
            f"  Interval width CV:             {self.interval_width_cv:.4f}",
            f"  Interval coverage sensitivity: {self.interval_coverage_sensitivity:.4f}",
            f"{'='*70}",
        ])
        return "\n".join(lines)


@dataclass
class ERSensitivityResult:
    """Sensitivity analysis results for Evidential Reasoning."""

    # Per-criterion belief sensitivity
    criterion_belief_sensitivity: Dict[str, float]    # OAT rank disruption per criterion
    grade_threshold_sensitivity: Dict[str, float]     # rank change under grade utility shift

    # Aggregation weight sensitivity
    weight_sensitivity: Dict[str, float]              # sensitivity to ER weight perturbations

    # Utility sensitivity
    utility_sensitivity: Dict[str, float]             # utility interval width per entity
    utility_interval_width: Dict[str, float]          # [u_min, u_max] width

    # Cross-level consistency
    cross_level_consistency: Dict[str, float]         # Spearman rho per criterion to final

    # Belief entropy
    mean_belief_entropy: float
    high_uncertainty_entities: List[str]

    # Overall
    overall_er_robustness: float

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            "EVIDENTIAL REASONING SENSITIVITY ANALYSIS",
            f"{'='*70}",
            f"Overall ER Robustness: {self.overall_er_robustness:.4f}",
            f"Mean Belief Entropy:   {self.mean_belief_entropy:.4f}",
            f"\n{'-'*70}",
            "CRITERION SENSITIVITY (higher = more sensitive to belief changes):",
            f"{'-'*70}",
        ]
        for crit, sens in sorted(
            self.criterion_belief_sensitivity.items(), key=lambda x: x[1], reverse=True
        ):
            bar = '[' + '=' * int(sens * 20) + ' ' * (20 - int(sens * 20)) + ']'
            lines.append(f"  {crit}: {sens:.4f} {bar}")

        if self.weight_sensitivity:
            lines.extend([
                f"\n{'-'*70}",
                "ER WEIGHT SENSITIVITY:",
                f"{'-'*70}",
            ])
            for src, sens in sorted(
                self.weight_sensitivity.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"  {src}: {sens:.4f}")

        if self.cross_level_consistency:
            lines.extend([
                f"\n{'-'*70}",
                "CROSS-LEVEL CONSISTENCY:",
                f"{'-'*70}",
            ])
            for level, corr in self.cross_level_consistency.items():
                lines.append(f"  {level}: {corr:.4f}")

        if self.high_uncertainty_entities:
            lines.extend([
                f"\n{'-'*70}",
                "HIGH UNCERTAINTY ENTITIES (wide utility interval):",
                f"{'-'*70}",
            ])
            for entity in self.high_uncertainty_entities[:10]:
                w = self.utility_interval_width.get(entity, 0.0)
                lines.append(f"  {entity}: interval_width={w:.4f}")

        lines.append("=" * 70)
        return "\n".join(lines)


@dataclass
class CombinedSensitivityResult:
    """Combined ML + ER sensitivity result returned by pipeline."""
    ml_sensitivity: Optional[MLSensitivityResult] = None
    er_sensitivity: Optional[ERSensitivityResult] = None

    @property
    def overall_robustness(self) -> float:
        scores = []
        if self.ml_sensitivity is not None:
            scores.append(self.ml_sensitivity.overall_robustness)
        if self.er_sensitivity is not None:
            scores.append(self.er_sensitivity.overall_er_robustness)
        return float(np.mean(scores)) if scores else 0.5

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            "SENSITIVITY ANALYSIS SUMMARY",
            f"{'='*70}",
            f"Overall Robustness: {self.overall_robustness:.4f}",
        ]
        if self.ml_sensitivity is not None:
            lines.append(self.ml_sensitivity.summary())
        if self.er_sensitivity is not None:
            lines.append(self.er_sensitivity.summary())
        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# ML Sensitivity Analysis
# ============================================================================

class MLSensitivityAnalysis:
    """
    Comprehensive sensitivity analysis for ML forecasting results.

    Parameters
    ----------
    n_bootstrap : int, default=200
        Bootstrap samples for feature importance stability.
    seed : int, default=42
        Random seed.
    n_jobs : int, default=-1
        Parallel workers (-1 = all CPUs - 1).
    """

    def __init__(
        self,
        n_bootstrap: int = 200,
        seed: int = 42,
        n_jobs: int = -1,
    ):
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        if n_jobs == -1:
            self.n_jobs = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.n_jobs = max(1, n_jobs)

    @_silence_warnings
    def analyze(self, forecast_result) -> MLSensitivityResult:
        """
        Run full ML sensitivity analysis.

        Parameters
        ----------
        forecast_result : UnifiedForecastResult

        Returns
        -------
        MLSensitivityResult
        """
        rng = np.random.default_rng(self.seed)

        fi_cv, fi_mean, fi_rank_stab = self._feature_importance_sensitivity(
            forecast_result, rng
        )
        loo_impact, contrib_cv = self._model_loo_sensitivity(forecast_result)
        pred_sensitivity = self._prediction_sensitivity(forecast_result)
        temporal_stab = self._temporal_stability(forecast_result)
        interval_width_cv, interval_cov_sens = self._interval_sensitivity(forecast_result)

        overall_robustness = self._compute_robustness(
            fi_cv, loo_impact, temporal_stab, interval_width_cv
        )

        return MLSensitivityResult(
            feature_importance_cv=fi_cv,
            feature_importance_mean=fi_mean,
            feature_importance_rank_stability=fi_rank_stab,
            loo_model_impact=loo_impact,
            model_contribution_cv=contrib_cv,
            prediction_sensitivity=pred_sensitivity,
            temporal_prediction_stability=temporal_stab,
            interval_width_cv=interval_width_cv,
            interval_coverage_sensitivity=interval_cov_sens,
            overall_robustness=overall_robustness,
            n_bootstrap=self.n_bootstrap,
        )

    def _feature_importance_sensitivity(
        self, forecast_result, rng: np.random.Generator
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        fi = forecast_result.feature_importance
        if fi is None or fi.empty:
            return {}, {}, {}

        n_features, n_comps = fi.shape
        B = self.n_bootstrap
        fi_matrix = np.zeros((B, n_features))

        for b in range(B):
            if n_comps > 1:
                comp_idx = rng.choice(n_comps, size=n_comps, replace=True)
                fi_boot = fi.iloc[:, comp_idx].mean(axis=1).values
            else:
                noise = rng.normal(0, 0.01, n_features)
                fi_boot = fi.iloc[:, 0].values + noise
            fi_matrix[b] = np.abs(fi_boot)

        mean_imp = fi_matrix.mean(axis=0)
        std_imp = fi_matrix.std(axis=0, ddof=1)
        feat_names = fi.index.tolist()

        fi_cv = {
            feat: float(std_imp[i] / (abs(mean_imp[i]) + 1e-10))
            for i, feat in enumerate(feat_names)
        }
        fi_mean = {feat: float(mean_imp[i]) for i, feat in enumerate(feat_names)}

        base_rank = np.argsort(-mean_imp)
        rank_stab = {}
        sorted_boots = np.argsort(-fi_matrix, axis=1)
        for i, feat in enumerate(feat_names):
            base_pos = int(np.where(base_rank == i)[0][0])
            within = int(np.sum(
                np.abs(np.array([int(np.where(r == i)[0][0]) for r in sorted_boots]) - base_pos)
                <= max(2, n_features // 10)
            ))
            rank_stab[feat] = float(within / B)

        return fi_cv, fi_mean, rank_stab

    def _model_loo_sensitivity(
        self, forecast_result
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        cv_scores = forecast_result.cross_validation_scores
        model_performance = forecast_result.model_performance

        model_r2: Dict[str, float] = {}
        for model, scores in cv_scores.items():
            if scores:
                model_r2[model] = float(np.nanmean(scores))
        if not model_r2:
            for model, perf in model_performance.items():
                r2 = perf.get('r2', perf.get('R2', np.nan))
                if not np.isnan(r2):
                    model_r2[model] = float(r2)

        all_r2 = list(model_r2.values())
        ensemble_mean = float(np.mean(all_r2)) if all_r2 else 0.0

        loo_impact = {}
        for model in model_r2:
            remaining = [v for m, v in model_r2.items() if m != model]
            loo_mean = float(np.mean(remaining)) if remaining else ensemble_mean
            loo_impact[model] = float(abs(ensemble_mean - loo_mean))

        contrib_vals = list(forecast_result.model_contributions.values())
        mean_c = float(np.mean(contrib_vals)) if contrib_vals else 1.0
        contrib_cv = {
            model: float(abs(contrib - mean_c) / (mean_c + 1e-10))
            for model, contrib in forecast_result.model_contributions.items()
        }

        return loo_impact, contrib_cv

    def _prediction_sensitivity(self, forecast_result) -> Dict[str, float]:
        predictions = forecast_result.predictions
        uncertainty = forecast_result.uncertainty
        pred_range = float(predictions.values.max() - predictions.values.min())
        if pred_range < 1e-10:
            pred_range = 1.0
        sensitivity = {}
        for entity in predictions.index:
            unc = float(uncertainty.loc[entity].mean()) if entity in uncertainty.index else 0.0
            sensitivity[str(entity)] = float(unc / pred_range)
        return sensitivity

    def _temporal_stability(self, forecast_result) -> float:
        from scipy.stats import spearmanr
        cv_scores = forecast_result.cross_validation_scores
        fold_corrs: List[float] = []
        for model, scores in cv_scores.items():
            if len(scores) >= 3:
                arr = np.array(scores, dtype=float)
                corr, _ = spearmanr(arr[:-1], arr[1:])
                if not np.isnan(corr):
                    fold_corrs.append(abs(float(corr)))
        return float(np.mean(fold_corrs)) if fold_corrs else 0.5

    def _interval_sensitivity(self, forecast_result) -> Tuple[float, float]:
        pi = forecast_result.prediction_intervals
        if not pi:
            return 0.0, 1.0
        width_cvs: List[float] = []
        for key, df in pi.items():
            if isinstance(df, pd.DataFrame) and df.shape[1] >= 2:
                widths = df.iloc[:, 1] - df.iloc[:, 0]
                if widths.mean() > 1e-10:
                    width_cvs.append(float(widths.std() / widths.mean()))
        interval_width_cv = float(np.mean(width_cvs)) if width_cvs else 0.0
        return interval_width_cv, float(np.clip(1.0 - interval_width_cv, 0, 1))

    def _compute_robustness(
        self,
        fi_cv: Dict[str, float],
        loo_impact: Dict[str, float],
        temporal_stab: float,
        interval_width_cv: float,
    ) -> float:
        components: List[float] = []
        if fi_cv:
            components.append(float(np.clip(1.0 - np.mean(list(fi_cv.values())), 0, 1)))
        if loo_impact:
            components.append(float(np.clip(1.0 - np.mean(list(loo_impact.values())), 0, 1)))
        components.append(temporal_stab)
        components.append(float(np.clip(1.0 - interval_width_cv, 0, 1)))
        return float(np.mean(components)) if components else 0.5


# ============================================================================
# ER Sensitivity Analysis
# ============================================================================

class ERSensitivityAnalysis:
    """
    Comprehensive sensitivity analysis for Evidential Reasoning results.

    Parameters
    ----------
    n_simulations : int, default=500
        Monte Carlo simulations for belief perturbation.
    perturbation_range : float, default=0.10
        Maximum belief perturbation magnitude.
    seed : int, default=42
        Random seed.
    """

    def __init__(
        self,
        n_simulations: int = 500,
        perturbation_range: float = 0.10,
        seed: int = 42,
    ):
        self.n_simulations = n_simulations
        self.perturbation_range = perturbation_range
        self.seed = seed

    @_silence_warnings
    def analyze(
        self,
        er_result,
        ranking_result=None,
    ) -> ERSensitivityResult:
        """
        Run full ER sensitivity analysis.

        Parameters
        ----------
        er_result : ERResult or HierarchicalERResult
        ranking_result : HierarchicalRankingResult, optional

        Returns
        -------
        ERSensitivityResult
        """
        rng = np.random.default_rng(self.seed)

        crit_sens = self._criterion_belief_sensitivity(er_result, rng)
        grade_sens = self._grade_threshold_sensitivity(er_result, rng)
        weight_sens = self._weight_sensitivity(er_result, rng)
        utility_sens, utility_widths = self._utility_sensitivity(er_result)
        cross_level = self._cross_level_consistency(ranking_result)
        mean_entropy, high_unc = self._entropy_analysis(er_result)
        er_robustness = self._compute_er_robustness(crit_sens, utility_sens, cross_level)

        return ERSensitivityResult(
            criterion_belief_sensitivity=crit_sens,
            grade_threshold_sensitivity=grade_sens,
            weight_sensitivity=weight_sens,
            utility_sensitivity=utility_sens,
            utility_interval_width=utility_widths,
            cross_level_consistency=cross_level,
            mean_belief_entropy=mean_entropy,
            high_uncertainty_entities=high_unc,
            overall_er_robustness=er_robustness,
        )

    def _criterion_belief_sensitivity(
        self, er_result, rng: np.random.Generator
    ) -> Dict[str, float]:
        from scipy.stats import spearmanr
        criterion_scores = getattr(er_result, 'criterion_method_scores', {})
        final_scores = getattr(er_result, 'final_scores', pd.Series(dtype=float))

        if not criterion_scores or final_scores.empty:
            return self._fallback_criterion_sensitivity(er_result, rng)

        sensitivities: Dict[str, float] = {}
        for crit_id, method_scores in criterion_scores.items():
            all_scores = [s for s in method_scores.values() if hasattr(s, 'values')]
            if not all_scores:
                continue
            avg_crit = pd.concat(all_scores, axis=1).mean(axis=1)
            rank_changes: List[float] = []
            for _ in range(50):
                delta = rng.uniform(-self.perturbation_range, self.perturbation_range,
                                    len(avg_crit))
                perturbed = np.clip(avg_crit.values + delta, 0, None)
                if len(perturbed) >= 2 and np.std(perturbed) > 1e-10:
                    corr, _ = spearmanr(avg_crit.values, perturbed)
                    rank_changes.append(1.0 - max(0.0, float(np.nan_to_num(corr))))
            sensitivities[crit_id] = float(np.mean(rank_changes)) if rank_changes else 0.0

        # Normalise to [0, 1]
        max_s = max(sensitivities.values()) if sensitivities else 1.0
        if max_s > 0:
            sensitivities = {k: v / max_s for k, v in sensitivities.items()}
        return sensitivities

    def _fallback_criterion_sensitivity(
        self, er_result, rng: np.random.Generator
    ) -> Dict[str, float]:
        from ranking.evidential_reasoning.base import BeliefDistribution
        belief_dists = getattr(er_result, 'belief_distributions', {})
        if not belief_dists:
            return {}
        sensitivities: Dict[str, float] = {}
        for entity, bd in list(belief_dists.items())[:20]:
            beliefs = np.array(bd.beliefs, dtype=float)
            base_u = bd.average_utility()
            rank_changes: List[float] = []
            for _ in range(30):
                noise = rng.normal(0, self.perturbation_range, len(beliefs))
                perturbed = np.clip(beliefs + noise, 0, 1)
                if perturbed.sum() > 1:
                    perturbed /= perturbed.sum()
                bd_pert = BeliefDistribution(grades=bd.grades, beliefs=perturbed)
                rank_changes.append(abs(bd_pert.average_utility() - base_u))
            sensitivities[f'entity_{entity}'] = float(np.mean(rank_changes)) if rank_changes else 0.0
        return sensitivities

    def _grade_threshold_sensitivity(
        self, er_result, rng: np.random.Generator
    ) -> Dict[str, float]:
        from ranking.evidential_reasoning.base import BeliefDistribution
        belief_dists = getattr(er_result, 'belief_distributions', {})
        if not belief_dists:
            return {}

        entities = list(belief_dists.keys())
        base_utilities = {e: belief_dists[e].average_utility() for e in entities}
        base_rank = pd.Series(base_utilities).rank(ascending=False)
        n_grades = len(list(belief_dists.values())[0].grades)

        n_sims = min(100, self.n_simulations)
        accum: Dict[str, List[float]] = {e: [] for e in entities}

        for _ in range(n_sims):
            utility_noise = rng.normal(0, 0.05, n_grades)
            utilities = np.clip(np.linspace(1.0, 0.0, n_grades) + utility_noise, 0, 1)
            utilities = np.sort(utilities)[::-1]
            perturbed_u = {e: belief_dists[e].expected_utility(utilities) for e in entities}
            perturbed_rank = pd.Series(perturbed_u).rank(ascending=False)
            for e in entities:
                if e in base_rank.index and e in perturbed_rank.index:
                    accum[e].append(abs(float(perturbed_rank[e]) - float(base_rank[e])))

        return {
            e: float(np.mean(changes)) / max(n_grades, 1)
            for e, changes in accum.items()
            if changes
        }

    def _weight_sensitivity(
        self, er_result, rng: np.random.Generator
    ) -> Dict[str, float]:
        agg_weights = getattr(er_result, 'aggregation_weights', None)
        if agg_weights is None or not isinstance(agg_weights, dict):
            return {}

        weight_keys = list(agg_weights.keys())
        weight_vals = np.array([agg_weights[k] for k in weight_keys], dtype=float)
        n_sims = max(10, min(100, self.n_simulations // len(weight_keys) + 1))

        sensitivities: Dict[str, float] = {}
        for i, key in enumerate(weight_keys):
            changes: List[float] = []
            for _ in range(n_sims):
                delta = rng.uniform(-self.perturbation_range, self.perturbation_range)
                perturbed = weight_vals.copy()
                perturbed[i] *= (1 + delta)
                perturbed = np.clip(perturbed, 0, None)
                if perturbed.sum() < 1e-10:
                    continue
                changes.append(abs(delta) * weight_vals[i])
            sensitivities[key] = float(np.mean(changes)) if changes else 0.0

        max_s = max(sensitivities.values()) if sensitivities else 1.0
        if max_s > 0:
            sensitivities = {k: v / max_s for k, v in sensitivities.items()}
        return sensitivities

    def _utility_sensitivity(
        self, er_result
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        belief_dists = getattr(er_result, 'belief_distributions', {})
        utility_sensitivity: Dict[str, float] = {}
        utility_widths: Dict[str, float] = {}
        for entity, bd in belief_dists.items():
            u_min, u_max = bd.utility_interval()
            width = u_max - u_min
            utility_widths[str(entity)] = float(width)
            utility_sensitivity[str(entity)] = float(np.clip(width, 0, 1))
        return utility_sensitivity, utility_widths

    def _cross_level_consistency(self, ranking_result) -> Dict[str, float]:
        if ranking_result is None or not hasattr(ranking_result, 'criterion_method_scores'):
            return {}
        from scipy.stats import spearmanr
        consistency: Dict[str, float] = {}
        try:
            er_result = ranking_result.er_result
            final_scores = er_result.final_scores
            crit_corrs: List[float] = []
            for crit_id, method_scores in ranking_result.criterion_method_scores.items():
                all_scores = [s for s in method_scores.values() if hasattr(s, 'values')]
                if not all_scores:
                    continue
                avg_crit = pd.concat(all_scores, axis=1).mean(axis=1)
                common = avg_crit.index.intersection(final_scores.index)
                if len(common) > 2:
                    corr, _ = spearmanr(
                        avg_crit.loc[common].values,
                        final_scores.loc[common].values
                    )
                    if not np.isnan(corr):
                        consistency[f'{crit_id}_to_final'] = float(corr)
                        crit_corrs.append(float(corr))
            if crit_corrs:
                consistency['mean_criterion_to_final'] = float(np.mean(crit_corrs))
        except Exception as exc:
            logger.warning(f"Cross-level consistency failed: {exc}")
        return consistency

    def _entropy_analysis(self, er_result) -> Tuple[float, List[str]]:
        belief_dists = getattr(er_result, 'belief_distributions', {})
        if not belief_dists:
            return 0.0, []
        entropies = {str(e): float(bd.belief_entropy()) for e, bd in belief_dists.items()}
        mean_entropy = float(np.mean(list(entropies.values()))) if entropies else 0.0
        if entropies:
            threshold = mean_entropy + float(np.std(list(entropies.values())))
            high_unc = [e for e, h in entropies.items() if h > threshold]
        else:
            high_unc = []
        return mean_entropy, high_unc

    def _compute_er_robustness(
        self,
        crit_sens: Dict[str, float],
        utility_sens: Dict[str, float],
        cross_level: Dict[str, float],
    ) -> float:
        components: List[float] = []
        if crit_sens:
            components.append(float(np.clip(1.0 - np.mean(list(crit_sens.values())), 0, 1)))
        if utility_sens:
            components.append(float(np.clip(1.0 - np.mean(list(utility_sens.values())), 0, 1)))
        if cross_level:
            cross_vals = [v for k, v in cross_level.items() if k != 'mean_criterion_to_final']
            if cross_vals:
                components.append(float(np.clip(np.mean(cross_vals) * 0.5 + 0.5, 0, 1)))
        return float(np.mean(components)) if components else 0.5


# ============================================================================
# Convenience functions
# ============================================================================

def run_ml_sensitivity_analysis(
    forecast_result,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> MLSensitivityResult:
    """Convenience function: ML forecast sensitivity analysis."""
    return MLSensitivityAnalysis(n_bootstrap=n_bootstrap, seed=seed).analyze(forecast_result)


def run_er_sensitivity_analysis(
    er_result,
    ranking_result=None,
    n_simulations: int = 500,
    perturbation_range: float = 0.10,
    seed: int = 42,
) -> ERSensitivityResult:
    """Convenience function: ER sensitivity analysis."""
    return ERSensitivityAnalysis(
        n_simulations=n_simulations,
        perturbation_range=perturbation_range,
        seed=seed,
    ).analyze(er_result, ranking_result=ranking_result)


def run_sensitivity_analysis(
    forecast_result=None,
    er_result=None,
    ranking_result=None,
    n_simulations: int = 500,
    n_bootstrap: int = 200,
    seed: int = 42,
    **kwargs,
) -> CombinedSensitivityResult:
    """
    Unified sensitivity analysis for ML and/or ER results.

    At least one of forecast_result or er_result must be provided.

    Returns
    -------
    CombinedSensitivityResult
    """
    ml_result = None
    er_result_out = None

    if forecast_result is not None:
        ml_result = run_ml_sensitivity_analysis(
            forecast_result, n_bootstrap=n_bootstrap, seed=seed
        )
    if er_result is not None:
        er_result_out = run_er_sensitivity_analysis(
            er_result, ranking_result=ranking_result,
            n_simulations=n_simulations, seed=seed,
        )

    return CombinedSensitivityResult(ml_sensitivity=ml_result, er_sensitivity=er_result_out)