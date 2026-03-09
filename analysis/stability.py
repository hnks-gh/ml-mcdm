# -*- coding: utf-8 -*-
"""
Stability Analysis for ML Forecasting and Evidential Reasoning
==============================================================

Quantifies temporal and cross-sectional stability for:

1. Forecast Stability
   - Temporal fold-to-fold consistency of ML predictions
   - Ensemble model agreement across members
   - Prediction interval width stability

2. Evidential Reasoning Stability
   - Belief distribution cosine similarity between two results
   - Utility ranking rank correlation
   - Grade assignment consistency and entropy stability

References
----------
Hyndman & Athanasopoulos (2021). Forecasting: Principles and Practice.
Yang & Xu (2002). Evidential reasoning under uncertainty.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Result containers
# ============================================================================

@dataclass
class ForecastStabilityResult:
    """Result container for ML forecast temporal stability."""
    is_stable: bool
    temporal_consistency: float            # Mean Spearman rho across folds
    entity_rank_volatility: Dict[str, float]  # Normalised uncertainty per entity
    model_agreement: float                 # 1 - CV across model R2 scores
    interval_width_stability: float        # 1 - CV of prediction interval widths
    component_stability: Dict[str, float]  # Per-model fold-wise stability
    split_point: int
    threshold: float

    @property
    def summary(self) -> str:
        status = "STABLE" if self.is_stable else "UNSTABLE"
        return (
            f"Forecast Stability: {status}\n"
            f"  Temporal Consistency: {self.temporal_consistency:.4f}\n"
            f"  Model Agreement:      {self.model_agreement:.4f}\n"
            f"  Interval Width CV:    {1.0 - self.interval_width_stability:.4f}\n"
            f"  Split Point:          {self.split_point}"
        )


@dataclass
class ERStabilityResult:
    """Result container for ER aggregation stability."""
    is_stable: bool
    belief_cosine_similarity: float        # Cosine similarity between two ER results
    utility_rank_correlation: float        # Spearman rho of utility rankings
    entropy_stability: float               # Stability of belief-entropy values
    grade_consistency: float               # Fraction with same dominant grade
    entity_volatility: Dict[str, float]    # Per-entity Shannon entropy
    threshold: float

    @property
    def summary(self) -> str:
        status = "STABLE" if self.is_stable else "UNSTABLE"
        return (
            f"ER Stability: {status}\n"
            f"  Belief Cosine Similarity: {self.belief_cosine_similarity:.4f}\n"
            f"  Utility Rank Correlation: {self.utility_rank_correlation:.4f}\n"
            f"  Entropy Stability:        {self.entropy_stability:.4f}\n"
            f"  Grade Consistency:        {self.grade_consistency:.4f}"
        )


# ============================================================================
# ML Forecast Stability
# ============================================================================

class ForecastStabilityAnalyzer:
    """
    Analyzes temporal and cross-model stability of ML forecasts.

    Parameters
    ----------
    threshold : float, default=0.8
        Minimum temporal consistency score to declare stability.
    """

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def analyze(
        self,
        forecast_result,
        panel_data=None,
    ) -> ForecastStabilityResult:
        """
        Analyze stability of a UnifiedForecastResult.

        Parameters
        ----------
        forecast_result : UnifiedForecastResult
        panel_data : PanelData, optional

        Returns
        -------
        ForecastStabilityResult
        """
        predictions = forecast_result.predictions
        uncertainty = forecast_result.uncertainty
        cv_scores = forecast_result.cross_validation_scores
        model_performance = forecast_result.model_performance

        temporal_consistency, split_point = self._temporal_consistency(
            cv_scores, panel_data
        )
        entity_rank_volatility = self._entity_rank_volatility(predictions, uncertainty)
        model_agreement = self._model_agreement(model_performance)
        interval_width_stability = self._interval_stability(forecast_result)
        component_stability = self._component_stability(cv_scores)

        is_stable = temporal_consistency >= self.threshold

        return ForecastStabilityResult(
            is_stable=is_stable,
            temporal_consistency=temporal_consistency,
            entity_rank_volatility=entity_rank_volatility,
            model_agreement=model_agreement,
            interval_width_stability=interval_width_stability,
            component_stability=component_stability,
            split_point=split_point,
            threshold=self.threshold,
        )

    def _temporal_consistency(
        self, cv_scores: Dict[str, List[float]], panel_data=None
    ) -> Tuple[float, int]:
        from scipy.stats import spearmanr
        split_point = 0
        if panel_data is not None and hasattr(panel_data, 'years'):
            years = sorted(panel_data.years)
            split_point = years[len(years) // 2] if years else 0

        fold_corrs: List[float] = []
        for model, scores in cv_scores.items():
            if len(scores) < 2:
                continue
            arr = np.array(scores, dtype=float)
            if len(arr) > 2:
                corr, _ = spearmanr(arr[:-1], arr[1:])
                if not np.isnan(corr):
                    fold_corrs.append(abs(float(corr)))
            mean_s = float(np.nanmean(arr))
            std_s = float(np.nanstd(arr))
            if abs(mean_s) > 1e-10:
                cv_metric = 1.0 - min(std_s / abs(mean_s), 1.0)
                fold_corrs.append(cv_metric)

        return (float(np.mean(fold_corrs)) if fold_corrs else 0.5), split_point

    def _entity_rank_volatility(
        self, predictions: pd.DataFrame, uncertainty: pd.DataFrame
    ) -> Dict[str, float]:
        n_entities = len(predictions)
        unc_mean = uncertainty.mean(axis=1)
        volatility = {}
        for entity in predictions.index:
            normalised = float(unc_mean.get(entity, 0)) / max(n_entities / 2, 1)
            volatility[str(entity)] = float(np.clip(normalised, 0, 1))
        return volatility

    def _model_agreement(self, model_performance: Dict[str, Dict[str, float]]) -> float:
        if not model_performance:
            return 1.0
        r2_vals = [
            p.get('r2', p.get('R2', np.nan))
            for p in model_performance.values()
        ]
        valid = [v for v in r2_vals if not np.isnan(v)]
        if len(valid) < 2:
            return 1.0
        mean_v = float(np.mean(valid))
        return float(np.clip(1.0 - np.std(valid) / (abs(mean_v) + 1e-10), 0, 1))

    def _interval_stability(self, forecast_result) -> float:
        pi = forecast_result.prediction_intervals
        if not pi:
            return 1.0
        for key, df in pi.items():
            if isinstance(df, pd.DataFrame) and df.shape[1] >= 2:
                widths = df.iloc[:, 1] - df.iloc[:, 0]
                if widths.mean() < 1e-10:
                    return 1.0
                cv = float(widths.std() / (widths.mean() + 1e-10))
                return float(np.clip(1.0 - cv, 0, 1))
        return 1.0

    def _component_stability(self, cv_scores: Dict[str, List[float]]) -> Dict[str, float]:
        stability = {}
        for model, scores in cv_scores.items():
            if not scores:
                stability[model] = 0.0
                continue
            arr = np.array(scores, dtype=float)
            mean_s = float(np.nanmean(arr))
            std_s = float(np.nanstd(arr))
            if abs(mean_s) > 1e-10:
                stability[model] = float(np.clip(1.0 - std_s / abs(mean_s), 0, 1))
            else:
                stability[model] = 0.0
        return stability


# ============================================================================
# ER Stability
# ============================================================================

class ERStabilityAnalyzer:
    """
    Analyzes stability of Evidential Reasoning aggregations by comparing
    two ER results (e.g., from split time periods or alternative parameterisations).

    Parameters
    ----------
    threshold : float, default=0.85
        Minimum cosine similarity to declare ER results stable.
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def analyze(self, er_result_1, er_result_2) -> ERStabilityResult:
        """
        Compare two ER results to assess stability.

        Parameters
        ----------
        er_result_1 : ERResult or HierarchicalERResult
        er_result_2 : ERResult or HierarchicalERResult

        Returns
        -------
        ERStabilityResult
        """
        from scipy.stats import spearmanr

        beliefs_1 = getattr(er_result_1, 'belief_distributions', {})
        beliefs_2 = getattr(er_result_2, 'belief_distributions', {})
        scores_1 = getattr(er_result_1, 'final_scores', pd.Series(dtype=float))
        scores_2 = getattr(er_result_2, 'final_scores', pd.Series(dtype=float))

        common = sorted(set(scores_1.index) & set(scores_2.index))
        if len(common) < 2:
            return ERStabilityResult(
                is_stable=True, belief_cosine_similarity=1.0,
                utility_rank_correlation=1.0, entropy_stability=1.0,
                grade_consistency=1.0, entity_volatility={}, threshold=self.threshold,
            )

        u1 = scores_1.loc[common].values
        u2 = scores_2.loc[common].values

        # Utility rank correlation
        corr, _ = spearmanr(u1, u2)
        utility_rank_corr = float(np.nan_to_num(corr))

        # Belief cosine similarity and grade consistency
        cos_sims: List[float] = []
        grade_matches: List[bool] = []
        for entity in common:
            if entity in beliefs_1 and entity in beliefs_2:
                b1 = np.array(beliefs_1[entity].beliefs, dtype=float)
                b2 = np.array(beliefs_2[entity].beliefs, dtype=float)
                n = min(len(b1), len(b2))
                b1, b2 = b1[:n], b2[:n]
                n1, n2 = np.linalg.norm(b1), np.linalg.norm(b2)
                if n1 > 0 and n2 > 0:
                    cos_sims.append(float(np.dot(b1, b2) / (n1 * n2)))
                grade_matches.append(int(b1.argmax()) == int(b2.argmax()))

        belief_cosine = float(np.mean(cos_sims)) if cos_sims else 1.0
        grade_consistency = float(np.mean(grade_matches)) if grade_matches else 1.0

        # Entropy stability
        entropy_1 = self._compute_entropy(beliefs_1, common)
        entropy_2 = self._compute_entropy(beliefs_2, common)
        if entropy_1.size > 0 and entropy_2.size > 0 and entropy_1.std() > 1e-10:
            corr_e, _ = spearmanr(entropy_1, entropy_2)
            entropy_stability = float(np.clip(np.nan_to_num(corr_e) * 0.5 + 0.5, 0, 1))
        else:
            entropy_stability = 1.0

        # Per-entity volatility (belief entropy of first result)
        entity_volatility = {
            entity: float(beliefs_1[entity].belief_entropy())
            for entity in common
            if entity in beliefs_1
        }

        is_stable = belief_cosine >= self.threshold

        return ERStabilityResult(
            is_stable=is_stable,
            belief_cosine_similarity=belief_cosine,
            utility_rank_correlation=utility_rank_corr,
            entropy_stability=entropy_stability,
            grade_consistency=grade_consistency,
            entity_volatility=entity_volatility,
            threshold=self.threshold,
        )

    def _compute_entropy(self, belief_distributions: dict, entities: list) -> np.ndarray:
        return np.array([
            belief_distributions[e].belief_entropy()
            for e in entities
            if e in belief_distributions
        ])


# ============================================================================
# Convenience functions
# ============================================================================

def analyze_forecast_stability(
    forecast_result,
    panel_data=None,
    threshold: float = 0.8,
) -> ForecastStabilityResult:
    """Convenience function: analyze ML forecast stability."""
    return ForecastStabilityAnalyzer(threshold=threshold).analyze(
        forecast_result, panel_data=panel_data
    )


def analyze_er_stability(
    er_result_1,
    er_result_2,
    threshold: float = 0.85,
) -> ERStabilityResult:
    """Convenience function: compare two ER results for stability."""
    return ERStabilityAnalyzer(threshold=threshold).analyze(er_result_1, er_result_2)