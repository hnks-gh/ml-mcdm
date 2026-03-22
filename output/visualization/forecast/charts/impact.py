# -*- coding: utf-8 -*-
"""Forecast Visualization: Forecast Impact and Business Metrics

Figures F-10, F-11, F-21 covering rank changes, province comparisons, and trajectories.
Phase 2 scaffold — to be filled in subsequent iterations.
"""

from typing import Optional
from output.visualization.base import BasePlotter, HAS_MATPLOTLIB
from output.visualization.forecast.contracts import ForecastVizPayload


class ImpactCharts(BasePlotter):
    """Forecast impact and business metric diagnostics.
    
    Figures:
    - F-10: Rank change bubble chart
    - F-11: Province/entity comparison (current vs forecast)
    - F-21: Score trajectory with confidence intervals
    """
    
    def plot_rank_change_bubble(
        self,
        payload: ForecastVizPayload,
        save_name: str = 'fig10_rank_change_bubble.png',
    ) -> Optional[str]:
        """Rank change bubble chart."""
        if not HAS_MATPLOTLIB:
            return None
        # TODO: Phase 2 implementation
        return None
    
    def plot_province_comparison(
        self,
        payload: ForecastVizPayload,
        save_name: str = 'fig11_province_comparison.png',
    ) -> Optional[str]:
        """Province/entity comparison with CI bars."""
        if not HAS_MATPLOTLIB:
            return None
        # TODO: Phase 2 implementation
        return None
    
    def plot_score_trajectory(
        self,
        payload: ForecastVizPayload,
        save_name: str = 'fig21_score_trajectory.png',
    ) -> Optional[str]:
        """Score trajectory with confidence interval fan."""
        if not HAS_MATPLOTLIB:
            return None
        # TODO: Phase 2 implementation
        return None


__all__ = ['ImpactCharts']
