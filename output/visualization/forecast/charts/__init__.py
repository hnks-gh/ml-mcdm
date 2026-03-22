# -*- coding: utf-8 -*-
"""
Forecast Visualization Chart Modules

Modular chart generation components extracted from the monolithic
forecast_plots.py. Each module focuses on a specific concern:

- accuracy: Predictive accuracy and error structure (F-01, F-02, F-03)
- ensemble: Ensemble composition and model comparison (F-04, F-05, F-06, F-22)
- uncertainty: Uncertainty quantification and calibration (F-07, F-08, F-09, F-16)
- interpretability: Feature attribution and model internals (F-12, F-14)
- impact: Forecast impact and business metrics (F-10, F-11, F-21)
- diversity: Model diversity and pairwise analysis (F-17, F-18)
- temporal: Temporal reliability and entity analysis (F-19, F-20)
"""

from output.visualization.forecast.charts.accuracy import AccuracyCharts
from output.visualization.forecast.charts.ensemble import EnsembleCharts
from output.visualization.forecast.charts.uncertainty import UncertaintyCharts
from output.visualization.forecast.charts.interpretability import InterpretabilityCharts
from output.visualization.forecast.charts.impact import ImpactCharts
from output.visualization.forecast.charts.diversity import DiversityCharts
from output.visualization.forecast.charts.temporal import TemporalCharts

__version__ = "1.0.0"

__all__ = [
    'AccuracyCharts',
    'EnsembleCharts',
    'UncertaintyCharts',
    'InterpretabilityCharts',
    'ImpactCharts',
    'DiversityCharts',
    'TemporalCharts',
]
