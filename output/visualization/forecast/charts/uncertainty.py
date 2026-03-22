# -*- coding: utf-8 -*-
"""
Forecast Visualization: Uncertainty Quantification and Calibration

Figures F-07, F-08, F-09, F-16 covering prediction intervals,
conformal coverage calibration, and uncertainty analysis.

Phase 2 scaffold — to be filled in subsequent iterations.
"""

from typing import Optional
from output.visualization.base import BasePlotter, HAS_MATPLOTLIB
from output.visualization.forecast.contracts import ForecastVizPayload


class UncertaintyCharts(BasePlotter):
    """
    Uncertainty quantification and calibration diagnostics.
    
    Figures:
    - F-07: Prediction intervals for top entities
    - F-08: Conformal coverage calibration curve
    - F-09: Interval calibration scatter plot
    - F-16: Bootstrap confidence intervals for metrics
    """
    
    def plot_prediction_intervals(
        self,
        payload: ForecastVizPayload,
        save_name: str = 'fig07_prediction_intervals.png',
    ) -> Optional[str]:
        """Prediction interval chart for top entities."""
        if not HAS_MATPLOTLIB:
            return None
        # TODO: Phase 2 implementation
        return None
    
    def plot_conformal_coverage(
        self,
        payload: ForecastVizPayload,
        save_name: str = 'fig08_conformal_coverage.png',
    ) -> Optional[str]:
        """Conformal coverage calibration curve."""
        if not HAS_MATPLOTLIB:
            return None
        # TODO: Phase 2 implementation
        return None
    
    def plot_interval_calibration_scatter(
        self,
        payload: ForecastVizPayload,
        save_name: str = 'fig09_interval_calibration_scatter.png',
    ) -> Optional[str]:
        """Interval calibration scatter plot."""
        if not HAS_MATPLOTLIB:
            return None
        # TODO: Phase 2 implementation
        return None
    
    def plot_bootstrap_ci(
        self,
        payload: ForecastVizPayload,
        metric: str = 'r2',
        save_name: str = 'fig16_bootstrap_ci.png',
    ) -> Optional[str]:
        """Bootstrap confidence intervals for model metrics."""
        if not HAS_MATPLOTLIB:
            return None
        # TODO: Phase 2 implementation
        return None


__all__ = ['UncertaintyCharts']
