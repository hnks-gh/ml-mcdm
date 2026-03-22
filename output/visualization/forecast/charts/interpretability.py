# -*- coding: utf-8 -*-
"""Forecast Visualization: Feature Importance and Model Interpretability

Figures F-12, F-14 covering feature attribution and per-model importance analysis.
Phase 2 scaffold — to be filled in subsequent iterations.
"""

from typing import Optional
from output.visualization.base import BasePlotter, HAS_MATPLOTLIB
from output.visualization.forecast.contracts import ForecastVizPayload


class InterpretabilityCharts(BasePlotter):
    """Feature importance and model interpretability diagnostics.
    
    Figures:
    - F-12: Global feature importance (top-N)
    - F-14: Per-model feature importance heatmap
    """
    
    def plot_feature_importance(
        self,
        payload: ForecastVizPayload,
        save_name: str = 'fig12_feature_importance.png',
    ) -> Optional[str]:
        """Global feature importance lollipop chart."""
        if not HAS_MATPLOTLIB:
            return None
        # TODO: Phase 2 implementation
        return None
    
    def plot_per_model_importance_heatmap(
        self,
        payload: ForecastVizPayload,
        save_name: str = 'fig14_per_model_importance_heatmap.png',
    ) -> Optional[str]:
        """Per-model feature importance heatmap."""
        if not HAS_MATPLOTLIB:
            return None
        # TODO: Phase 2 implementation
        return None


__all__ = ['InterpretabilityCharts']
