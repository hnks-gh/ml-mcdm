# -*- coding: utf-8 -*-
"""
Forecast Visualization Package

Modular, production-grade visualization system for ensemble forecasting
diagnostic and communication figures.

Architecture:
- contracts.py: Typed data payload structures
- validators.py: Input validation and invariant checks
- metrics.py: Centralized metric implementations
- orchestrator.py: Figure generation coordinator (Phase 3)
- charts/: Modular chart modules (Phase 2)
- adapters/: Data transformation utilities (Phase 3)

All modules follow single-responsibility and explicit contract patterns.
"""

from output.visualization.forecast.contracts import ForecastVizPayload
from output.visualization.forecast.validators import (
    PayloadValidationError,
    ValidatorWarning,
    validate_payload_essential,
    validate_payload_for_figure,
)
from output.visualization.forecast.metrics import (
    r2_score,
    rmse_score,
    mae_score,
    mape_score,
    bias_score,
    correlation_score,
    conformal_quantile,
    conformal_coverage,
    conformal_interval_size,
    bootstrap_ci,
    compute_metric_summary,
)
from output.visualization.forecast.unified_plotter import UnifiedForecastPlotter
from output.visualization.forecast.orchestrator import (
    ForecastVisualizationOrchestrator,
    ExecutionReport,
)
from output.visualization.forecast.adapters import UnifiedResultAdapter
from output.visualization.forecast.io_manifest import (
    get_manifest,
    ForecastFigureManifest,
    FigureCategory,
    FigureSpec,
)
from output.visualization.forecast.charts import (
    AccuracyCharts,
    EnsembleCharts,
    UncertaintyCharts,
    InterpretabilityCharts,
    ImpactCharts,
    DiversityCharts,
    TemporalCharts,
)

__version__ = "1.0.0"

__all__ = [
    # Contracts
    'ForecastVizPayload',
    # Validation
    'PayloadValidationError',
    'ValidatorWarning',
    'validate_payload_essential',
    'validate_payload_for_figure',
    # Metrics
    'r2_score',
    'rmse_score',
    'mae_score',
    'mape_score',
    'bias_score',
    'correlation_score',
    'conformal_quantile',
    'conformal_coverage',
    'conformal_interval_size',
    'bootstrap_ci',
    'compute_metric_summary',
    # Orchestrator (Phase 3)
    'ForecastVisualizationOrchestrator',
    'ExecutionReport',
    # Adapter (Phase 3)
    'UnifiedResultAdapter',
    # Manifest (Phase 3)
    'get_manifest',
    'ForecastFigureManifest',
    'FigureCategory',
    'FigureSpec',
    # Unified plotter (Phase 2)
    'UnifiedForecastPlotter',
    # Chart modules (Phase 2)
    'AccuracyCharts',
    'EnsembleCharts',
    'UncertaintyCharts',
    'InterpretabilityCharts',
    'ImpactCharts',
    'DiversityCharts',
    'TemporalCharts',
]
