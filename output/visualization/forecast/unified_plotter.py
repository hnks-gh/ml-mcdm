# -*- coding: utf-8 -*-
"""Unified Forecast Visualization Orchestrator

Composes all chart modules into a single coordinated visualization pipeline
following the Composite Pattern. Provides high-level API for generating
all forecast analytics figures.

This module bridges the ForecastVizPayload data contract with individual
chart implementations and handles:
- Delegation to specialized chart modules
- Output coordination and naming
- Error handling and graceful degradation
"""

from typing import Dict, Optional, List, Tuple
from pathlib import Path
import logging

from output.visualization.forecast.contracts import ForecastVizPayload
from output.visualization.forecast.charts.accuracy import AccuracyCharts
from output.visualization.forecast.charts.ensemble import EnsembleCharts
from output.visualization.forecast.charts.uncertainty import UncertaintyCharts
from output.visualization.forecast.charts.interpretability import InterpretabilityCharts
from output.visualization.forecast.charts.impact import ImpactCharts
from output.visualization.forecast.charts.diversity import DiversityCharts
from output.visualization.forecast.charts.temporal import TemporalCharts


logger = logging.getLogger(__name__)


class UnifiedForecastPlotter:
    """Orchestrates all forecast visualization modules.
    
    Composite pattern: delegates to specialized chart modules per domain.
    Provides unified API: single call to generate all figures.
    
    Attributes:
        output_dir: Output directory for all generated figures
        accuracy_charts: Accuracy and error diagnostics module
        ensemble_charts: Ensemble composition and model comparison module
        uncertainty_charts: Calibration and uncertainty quantification module
        interpretability_charts: Feature importance and attribution module
        impact_charts: Forecast impact and business metrics module
        diversity_charts: Model diversity and prediction analysis module
        temporal_charts: Temporal reliability and entity-level analysis module
    """
    
    def __init__(self, output_dir: str = 'output/visualization/forecast/result'):
        """Initialize unified plotter with output directory.
        
        Args:
            output_dir: Directory to save all generated figures.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize chart modules
        self.accuracy_charts = AccuracyCharts(output_dir=str(self.output_dir))
        self.ensemble_charts = EnsembleCharts(output_dir=str(self.output_dir))
        self.uncertainty_charts = UncertaintyCharts(output_dir=str(self.output_dir))
        self.interpretability_charts = InterpretabilityCharts(output_dir=str(self.output_dir))
        self.impact_charts = ImpactCharts(output_dir=str(self.output_dir))
        self.diversity_charts = DiversityCharts(output_dir=str(self.output_dir))
        self.temporal_charts = TemporalCharts(output_dir=str(self.output_dir))
    
    def plot_all(
        self,
        payload: ForecastVizPayload,
    ) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Generate all forecast visualization figures.
        
        Calls all specialized chart modules to produce a comprehensive
        set of diagnostics figures (F-01 through F-22).
        
        Args:
            payload: Unified data contract for all visualization inputs.
            
        Returns:
            Dictionary mapping figure descriptions to (success, output_path) tuples.
            Success is True if figure was generated, False if skipped or errored.
        """
        results = {}
        
        # Accuracy diagnostics (F-01, F-02, F-03)
        logger.info('Generating accuracy diagnostics figures...')
        try:
            out_path = self.accuracy_charts.plot_forecast_scatter(payload)
            results['F-01: Actual vs Predicted'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-01: {e}', exc_info=True)
            results['F-01: Actual vs Predicted'] = (False, None)
        
        try:
            out_path = self.accuracy_charts.plot_forecast_residuals(payload)
            results['F-02: Residual Diagnostics'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-02: {e}', exc_info=True)
            results['F-02: Residual Diagnostics'] = (False, None)
        
        try:
            out_path = self.accuracy_charts.plot_holdout_comparison(payload)
            results['F-03: Holdout Comparison'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-03: {e}', exc_info=True)
            results['F-03: Holdout Comparison'] = (False, None)
        
        # Ensemble diagnostics (F-04, F-05, F-06, F-22)
        logger.info('Generating ensemble diagnostics figures...')
        try:
            out_path = self.ensemble_charts.plot_model_weights_donut(payload)
            results['F-04: Model Weights'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-04: {e}', exc_info=True)
            results['F-04: Model Weights'] = (False, None)
        
        try:
            out_path = self.ensemble_charts.plot_model_performance(payload)
            results['F-05: Model Performance'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-05: {e}', exc_info=True)
            results['F-05: Model Performance'] = (False, None)
        
        try:
            out_path = self.ensemble_charts.plot_cv_boxplots(payload)
            results['F-06: CV Score Distributions'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-06: {e}', exc_info=True)
            results['F-06: CV Score Distributions'] = (False, None)
        
        # Uncertainty and calibration diagnostics (F-07, F-08, F-09, F-16)
        logger.info('Generating uncertainty and calibration figures...')
        try:
            out_path = self.uncertainty_charts.plot_prediction_intervals(payload)
            results['F-07: Prediction Intervals'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-07: {e}', exc_info=True)
            results['F-07: Prediction Intervals'] = (False, None)
        
        try:
            out_path = self.uncertainty_charts.plot_conformal_coverage(payload)
            results['F-08: Conformal Coverage'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-08: {e}', exc_info=True)
            results['F-08: Conformal Coverage'] = (False, None)
        
        try:
            out_path = self.uncertainty_charts.plot_interval_calibration_scatter(payload)
            results['F-09: Interval Calibration'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-09: {e}', exc_info=True)
            results['F-09: Interval Calibration'] = (False, None)
        
        try:
            out_path = self.uncertainty_charts.plot_bootstrap_ci(payload)
            results['F-16: Bootstrap CI'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-16: {e}', exc_info=True)
            results['F-16: Bootstrap CI'] = (False, None)
        
        # Impact diagnostics (F-10, F-11, F-21)
        logger.info('Generating impact diagnostics figures...')
        try:
            out_path = self.impact_charts.plot_rank_change_bubble(payload)
            results['F-10: Rank Change Bubble'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-10: {e}', exc_info=True)
            results['F-10: Rank Change Bubble'] = (False, None)
        
        try:
            out_path = self.impact_charts.plot_province_comparison(payload)
            results['F-11: Province Comparison'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-11: {e}', exc_info=True)
            results['F-11: Province Comparison'] = (False, None)
        
        try:
            out_path = self.impact_charts.plot_score_trajectory(payload)
            results['F-21: Score Trajectory'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-21: {e}', exc_info=True)
            results['F-21: Score Trajectory'] = (False, None)
        
        # Interpretability diagnostics (F-12, F-14)
        logger.info('Generating interpretability diagnostics figures...')
        try:
            out_path = self.interpretability_charts.plot_feature_importance(payload)
            results['F-12: Feature Importance'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-12: {e}', exc_info=True)
            results['F-12: Feature Importance'] = (False, None)
        
        try:
            out_path = self.interpretability_charts.plot_per_model_importance_heatmap(payload)
            results['F-14: Per-Model Importance'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-14: {e}', exc_info=True)
            results['F-14: Per-Model Importance'] = (False, None)
        
        # Diversity diagnostics (F-17, F-18)
        logger.info('Generating diversity diagnostics figures...')
        try:
            out_path = self.diversity_charts.plot_prediction_correlation_heatmap(payload)
            results['F-17: Prediction Correlation'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-17: {e}', exc_info=True)
            results['F-17: Prediction Correlation'] = (False, None)
        
        try:
            out_path = self.diversity_charts.plot_prediction_scatter_matrix(payload)
            results['F-18: Prediction Scatter Matrix'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-18: {e}', exc_info=True)
            results['F-18: Prediction Scatter Matrix'] = (False, None)
        
        # Temporal diagnostics (F-19, F-20)
        logger.info('Generating temporal diagnostics figures...')
        try:
            out_path = self.temporal_charts.plot_entity_error_analysis(payload)
            results['F-19: Entity Error Analysis'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-19: {e}', exc_info=True)
            results['F-19: Entity Error Analysis'] = (False, None)
        
        try:
            out_path = self.temporal_charts.plot_temporal_training_curve(payload)
            results['F-20: Temporal Training Curve'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-20: {e}', exc_info=True)
            results['F-20: Temporal Training Curve'] = (False, None)
        
        # Ensemble architecture (F-22)
        try:
            out_path = self.ensemble_charts.plot_ensemble_architecture(payload)
            results['F-22: Ensemble Architecture'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-22: {e}', exc_info=True)
            results['F-22: Ensemble Architecture'] = (False, None)
        
        logger.info(f'Visualization generation complete. {sum(r[0] for r in results.values())}/{len(results)} figures generated.')
        return results
    
    def plot_quick_diagnostics(
        self,
        payload: ForecastVizPayload,
    ) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Generate essential diagnostic figures (quick subset).
        
        Generates a minimal set of critical figures:
        - F-01: Actual vs Predicted
        - F-02: Residual Diagnostics
        - F-07: Prediction Intervals
        - F-17: Prediction Correlation
        
        Args:
            payload: Unified data contract for visualization inputs.
            
        Returns:
            Dictionary mapping figure descriptions to (success, output_path) tuples.
        """
        results = {}
        
        logger.info('Generating quick diagnostic figures...')
        
        try:
            out_path = self.accuracy_charts.plot_forecast_scatter(payload)
            results['F-01: Actual vs Predicted'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-01: {e}', exc_info=True)
            results['F-01: Actual vs Predicted'] = (False, None)
        
        try:
            out_path = self.accuracy_charts.plot_forecast_residuals(payload)
            results['F-02: Residual Diagnostics'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-02: {e}', exc_info=True)
            results['F-02: Residual Diagnostics'] = (False, None)
        
        try:
            out_path = self.uncertainty_charts.plot_prediction_intervals(payload)
            results['F-07: Prediction Intervals'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-07: {e}', exc_info=True)
            results['F-07: Prediction Intervals'] = (False, None)
        
        try:
            out_path = self.diversity_charts.plot_prediction_correlation_heatmap(payload)
            results['F-17: Prediction Correlation'] = (out_path is not None, out_path)
        except Exception as e:
            logger.warning(f'Failed to generate F-17: {e}', exc_info=True)
            results['F-17: Prediction Correlation'] = (False, None)
        
        logger.info(f'Quick diagnostics complete. {sum(r[0] for r in results.values())}/{len(results)} figures generated.')
        return results


__all__ = ['UnifiedForecastPlotter']
