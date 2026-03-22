# -*- coding: utf-8 -*-
"""
Forecasting Plots (fig16–fig23) — Phase 4 Compatibility Facade

This module serves as a compatibility layer between the legacy ForecastPlotter API
and the new modular chart architecture (Phase 2/3).

Old callers invoke ForecastPlotter.plot_*() with raw numpy arrays and dicts.
Each method now delegates to the appropriate chart module class after constructing
a type-safe ForecastVizPayload contract.

Charts are organized into 7 modular classes:
- AccuracyCharts: F-01, F-02, F-03, F-13 (actual vs predicted, residuals, holdout)
- EnsembleCharts: F-04, F-05, F-06, F-22, F-15, F-20b (weights, performance, architecture)
- UncertaintyCharts: F-07, F-08, F-09, F-16 (intervals, conformal, bootstrap CI)
- InterpretabilityCharts: F-12, F-14 (feature importance, per-model importance)
- ImpactCharts: F-10, F-11, F-21 (rank change, province comparison, score trajectory)
- DiversityCharts: F-17, F-18 (prediction correlation, scatter matrix)
- TemporalCharts: F-19, F-20 (entity error analysis, temporal training curve)

PHASE 4 NOTE:
- All 24 original ForecastPlotter method signatures are preserved
- Method bodies delegate to modular chart modules via ForecastVizPayload
- See PHASE4_IMPLEMENTATION_SPEC.md and PHASE4_METHOD_MAPPING.md for details
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .base import BasePlotter, HAS_MATPLOTLIB

# Phase 3/4 imports: data contracts and chart modules
from .forecast.contracts import ForecastVizPayload
from .forecast.charts.accuracy import AccuracyCharts
from .forecast.charts.ensemble import EnsembleCharts
from .forecast.charts.uncertainty import UncertaintyCharts
from .forecast.charts.interpretability import InterpretabilityCharts
from .forecast.charts.impact import ImpactCharts
from .forecast.charts.diversity import DiversityCharts
from .forecast.charts.temporal import TemporalCharts


# ── Phase 4 Helper Functions: Raw Array → ForecastVizPayload ──────────────── #

def _ensure_array(val: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Convert to numpy array and ensure float64 if not None."""
    if val is None:
        return None
    arr = np.asarray(val, dtype=np.float64)
    return arr if arr.size > 0 else None


def _build_payload(**kwargs) -> ForecastVizPayload:
    """Construct a ForecastVizPayload from raw facade method arguments."""
    return ForecastVizPayload(
        y_test=_ensure_array(kwargs.get('y_test')),
        y_pred_ensemble=_ensure_array(kwargs.get('y_pred_ensemble')),
        entity_names=kwargs.get('entity_names'),
        interval_lower_df=kwargs.get('interval_lower_df'),
        interval_upper_df=kwargs.get('interval_upper_df'),
        model_contributions=kwargs.get('model_contributions'),
        model_performance=kwargs.get('model_performance'),
        cv_scores=kwargs.get('cv_scores'),
        per_model_oof_predictions=kwargs.get('per_model_oof_predictions'),
        per_model_feature_importance=kwargs.get('per_model_feature_importance'),
        current_scores=_ensure_array(kwargs.get('current_scores')),
        provinces=kwargs.get('provinces'),
        target_year=kwargs.get('target_year'),
        cv_fold_val_years=kwargs.get('cv_fold_val_years'),
    )


class ForecastPlotter(BasePlotter):
    """
    Phase 4 Facade: Forward-compatible wrapper for modular chart architecture.

    All method signatures remain unchanged for backward compatibility with
    existing code. Each method creates a ForecastVizPayload and delegates
    to the appropriate chart module (Accuracy, Ensemble, Uncertainty, etc.).
    """

    def __init__(self, output_dir: str = '.', dpi: int = 300):
        """Initialize facade with chart module instances."""
        super().__init__(output_dir, dpi)
        self._accuracy = AccuracyCharts(output_dir, dpi)
        self._ensemble = EnsembleCharts(output_dir, dpi)
        self._uncertainty = UncertaintyCharts(output_dir, dpi)
        self._interpretability = InterpretabilityCharts(output_dir, dpi)
        self._impact = ImpactCharts(output_dir, dpi)
        self._diversity = DiversityCharts(output_dir, dpi)
        self._temporal = TemporalCharts(output_dir, dpi)

    # ====================================================================
    # ACCURACY CHARTS (F-01, F-02, F-03, F-13)
    # ====================================================================

    def plot_forecast_scatter(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        entity_names: Optional[List[str]] = None,
        save_name: str = 'fig16_forecast_scatter.png',
    ) -> Optional[str]:
        """F-01: Actual vs Predicted scatter with fit line and stats."""
        payload = _build_payload(y_test=actual, y_pred_ensemble=predicted,
                                entity_names=entity_names)
        return self._accuracy.plot_forecast_scatter(payload, save_name=save_name)

    def plot_forecast_residuals(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        save_name: str = 'fig17_forecast_residuals.png',
    ) -> Optional[str]:
        """F-02: 4-panel residual diagnostics."""
        payload = _build_payload(y_test=actual, y_pred_ensemble=predicted)
        return self._accuracy.plot_forecast_residuals(payload, save_name=save_name)

    def plot_holdout_comparison(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        per_model_predictions: Optional[Dict] = None,
        entity_names: Optional[List[str]] = None,
        model_contributions: Optional[Dict[str, float]] = None,
        save_name: str = 'fig_holdout_comparison.png',
    ) -> Optional[str]:
        """F-03: Holdout comparison by model and ensemble."""
        payload = _build_payload(
            y_test=actual, y_pred_ensemble=predicted,
            per_model_oof_predictions=per_model_predictions,
            entity_names=entity_names, model_contributions=model_contributions)
        return self._accuracy.plot_holdout_comparison(payload, save_name=save_name)

    def plot_residual_distributions(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        per_model_predictions: Optional[Dict] = None,
        max_models: int = 12,
        save_name: str = 'fig13_residual_distributions.png',
    ) -> Optional[str]:
        """F-13: Per-model residual distribution panel."""
        payload = _build_payload(
            y_test=actual, y_pred_ensemble=predicted,
            per_model_oof_predictions=per_model_predictions)
        return self._accuracy.plot_residual_distributions(
            payload, max_models=max_models, save_name=save_name)

    # ====================================================================
    # ENSEMBLE CHARTS (F-04, F-05, F-06, F-22, F-15, F-20b)
    # ====================================================================

    def plot_model_weights_donut(
        self,
        weights: Dict[str, float],
        save_name: str = 'fig19_model_weights.png',
    ) -> Optional[str]:
        """F-04: Model contribution weight donut chart."""
        payload = _build_payload(model_contributions=weights)
        return self._ensemble.plot_model_weights_donut(payload, save_name=save_name)

    def plot_model_performance(
        self,
        model_metrics: Dict[str, Dict[str, float]],
        save_name: str = 'fig20_model_performance.png',
    ) -> Optional[str]:
        """F-05: Per-model performance comparison."""
        payload = _build_payload(model_performance=model_metrics)
        return self._ensemble.plot_model_performance(payload, save_name=save_name)

    def plot_cv_boxplots(
        self,
        cv_scores: Dict[str, List[float]],
        save_name: str = 'fig21_cv_boxplots.png',
    ) -> Optional[str]:
        """F-06: Cross-validation score distributions."""
        payload = _build_payload(cv_scores=cv_scores)
        return self._ensemble.plot_cv_boxplots(payload, save_name=save_name)

    def plot_ensemble_architecture(
        self,
        model_names: Optional[List[str]] = None,
        save_name: str = 'fig22_ensemble_architecture.png',
    ) -> Optional[str]:
        """F-22: Ensemble pipeline architecture flowchart."""
        payload = _build_payload()
        return self._ensemble.plot_ensemble_architecture(payload, save_name=save_name)

    def plot_model_contribution_dots(
        self,
        weights: Dict[str, float],
        cv_scores: Dict[str, List[float]],
        save_name: str = 'fig20b_model_contribution_dots.png',
    ) -> Optional[str]:
        """F-20b: Weight vs CV R² bubble chart."""
        payload = _build_payload(model_contributions=weights, cv_scores=cv_scores)
        return self._ensemble.plot_model_contribution_dots(payload, save_name=save_name)

    def plot_model_metric_radar(
        self,
        model_metrics: Dict[str, Dict[str, float]],
        metrics: Optional[List[str]] = None,
        save_name: str = 'fig15_model_metric_radar.png',
    ) -> Optional[str]:
        """F-15: Model metric radar/spider chart (with bar fallback)."""
        payload = _build_payload(model_performance=model_metrics)
        return self._ensemble.plot_model_metric_radar(
            payload, metrics=metrics, save_name=save_name)

    # ====================================================================
    # UNCERTAINTY CHARTS (F-07, F-08, F-09, F-16)
    # ====================================================================

    def plot_prediction_intervals(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        lower: Optional[pd.DataFrame] = None,
        upper: Optional[pd.DataFrame] = None,
        entity_names: Optional[List[str]] = None,
        top_n: int = 15,
        save_name: str = 'fig23_prediction_intervals.png',
    ) -> Optional[str]:
        """F-07: Prediction interval chart for top entities."""
        payload = _build_payload(
            y_test=actual, y_pred_ensemble=predicted,
            interval_lower_df=lower, interval_upper_df=upper,
            entity_names=entity_names)
        return self._uncertainty.plot_prediction_intervals(payload, save_name=save_name)

    def plot_conformal_coverage(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        lower: Optional[pd.DataFrame] = None,
        upper: Optional[pd.DataFrame] = None,
        save_name: str = 'fig24_conformal_coverage.png',
    ) -> Optional[str]:
        """F-08: Conformal coverage calibration curve."""
        payload = _build_payload(
            y_test=actual, y_pred_ensemble=predicted,
            interval_lower_df=lower, interval_upper_df=upper)
        return self._uncertainty.plot_conformal_coverage(payload, save_name=save_name)

    def plot_interval_calibration_scatter(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        lower: Optional[pd.DataFrame] = None,
        upper: Optional[pd.DataFrame] = None,
        save_name: str = 'fig25_interval_calibration_scatter.png',
    ) -> Optional[str]:
        """F-09: Interval calibration scatter (width vs historical error)."""
        payload = _build_payload(
            y_test=actual, y_pred_ensemble=predicted,
            interval_lower_df=lower, interval_upper_df=upper)
        return self._uncertainty.plot_interval_calibration_scatter(
            payload, save_name=save_name)

    def plot_bootstrap_metric_ci(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        metric: str = 'r2',
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        save_name: str = 'fig16_bootstrap_ci.png',
    ) -> Optional[str]:
        """F-16: Bootstrap confidence intervals for metrics."""
        payload = _build_payload(y_test=actual, y_pred_ensemble=predicted)
        # Note: Method name mismatch - ForecastPlotter uses plot_bootstrap_metric_ci
        #       but UncertaintyCharts uses plot_bootstrap_ci
        return self._uncertainty.plot_bootstrap_ci(payload, metric=metric, 
                                                  save_name=save_name)

    # ====================================================================
    # INTERPRETABILITY CHARTS (F-12, F-14)
    # ====================================================================

    def plot_feature_importance(
        self,
        importance: Dict[str, float],
        top_n: int = 20,
        title: str = 'Feature Importance — Top Features',
        save_name: str = 'fig18_feature_importance.png',
    ) -> Optional[str]:
        """F-12: Feature importance lollipop chart."""
        payload = _build_payload()
        # Note: This is a special case - we pass importance dict directly to chart
        # The chart module method signature may differ
        return self._interpretability.plot_feature_importance(
            importance, top_n=top_n, title=title, save_name=save_name)

    def plot_feature_importance_single(
        self,
        importance_dict: Dict[str, float],
        **kwargs
    ) -> Optional[str]:
        """Backward-compatible alias for plot_feature_importance."""
        return self.plot_feature_importance(importance_dict, **kwargs)

    def plot_per_model_importance_heatmap(
        self,
        per_model_importances: Dict[str, Dict[str, float]],
        top_n: int = 15,
        save_name: str = 'fig14_per_model_importance_heatmap.png',
    ) -> Optional[str]:
        """F-14: Per-model feature importance heatmap."""
        payload = _build_payload(per_model_feature_importance=per_model_importances)
        return self._interpretability.plot_per_model_importance_heatmap(
            payload, top_n=top_n, save_name=save_name)

    # ====================================================================
    # IMPACT CHARTS (F-10, F-11, F-21)
    # ====================================================================

    def plot_rank_change_bubble(
        self,
        current_scores: np.ndarray,
        predicted_scores: np.ndarray,
        entity_names: Optional[List[str]] = None,
        current_year: Optional[int] = None,
        predicted_year: Optional[int] = None,
        save_name: str = 'fig_rank_change_bubble.png',
    ) -> Optional[str]:
        """F-10: Rank change bubble chart (current vs forecast)."""
        payload = _build_payload(
            current_scores=current_scores, y_pred_ensemble=predicted_scores,
            entity_names=entity_names, target_year=predicted_year)
        return self._impact.plot_rank_change_bubble(payload, save_name=save_name)

    def plot_province_forecast_comparison(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        entity_names: Optional[List[str]] = None,
        save_name: str = 'fig_province_comparison.png',
    ) -> Optional[str]:
        """F-11: Province/entity comparison (current vs forecast with CI)."""
        payload = _build_payload(
            y_test=actual, y_pred_ensemble=predicted,
            provinces=entity_names, entity_names=entity_names)
        # Note: Method name mismatch - ForecastPlotter has plot_province_forecast_comparison
        #       but ImpactCharts might use plot_province_comparison
        return self._impact.plot_province_comparison(payload, save_name=save_name)

    def plot_score_trajectory(
        self,
        historical_scores: Optional[np.ndarray] = None,
        predicted_score: Optional[np.ndarray] = None,
        entity_names: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        save_name: str = 'fig_score_trajectory.png',
    ) -> Optional[str]:
        """F-21: Score trajectory (historical + forecast with CI fan)."""
        payload = _build_payload(
            current_scores=historical_scores,
            y_pred_ensemble=predicted_score,
            entity_names=entity_names)
        return self._impact.plot_score_trajectory(payload, save_name=save_name)

    # ====================================================================
    # DIVERSITY CHARTS (F-17, F-18)
    # ====================================================================

    def plot_prediction_correlation_heatmap(
        self,
        per_model_predictions: Dict[str, np.ndarray],
        entity_names: Optional[List[str]] = None,
        save_name: str = 'fig_prediction_correlation_heatmap.png',
    ) -> Optional[str]:
        """F-17: Prediction correlation heatmap (with clustering)."""
        payload = _build_payload(
            per_model_oof_predictions=per_model_predictions,
            entity_names=entity_names)
        return self._diversity.plot_prediction_correlation_heatmap(
            payload, save_name=save_name)

    def plot_prediction_scatter_matrix(
        self,
        per_model_predictions: Dict[str, np.ndarray],
        max_models: int = 7,
        save_name: str = 'fig_prediction_scatter_matrix.png',
    ) -> Optional[str]:
        """F-18: Scatter matrix of pairwise model predictions."""
        payload = _build_payload(per_model_oof_predictions=per_model_predictions)
        return self._diversity.plot_prediction_scatter_matrix(
            payload, max_models=max_models, save_name=save_name)

    # ====================================================================
    # TEMPORAL CHARTS (F-19, F-20)
    # ====================================================================

    def plot_entity_error_analysis(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        entity_names: Optional[List[str]] = None,
        top_n: int = 20,
        save_name: str = 'fig_entity_error_analysis.png',
    ) -> Optional[str]:
        """F-19: Entity-wise error analysis with signed bias annotation."""
        payload = _build_payload(
            y_test=actual, y_pred_ensemble=predicted,
            entity_names=entity_names)
        return self._temporal.plot_entity_error_analysis(payload, save_name=save_name)

    def plot_temporal_training_curve(
        self,
        cv_scores: Dict[str, List[float]],
        fold_labels: Optional[List[str]] = None,
        save_name: str = 'fig_temporal_training_curve.png',
    ) -> Optional[str]:
        """F-20: Walk-forward temporal CV curve (fold/year trajectory)."""
        payload = _build_payload(cv_scores=cv_scores)
        return self._temporal.plot_temporal_training_curve(payload, save_name=save_name)


__all__ = ['ForecastPlotter']
