# -*- coding: utf-8 -*-
"""
ML-MCDM: Machine Learning Enhanced Multi-Criteria Decision Making
=================================================================

A comprehensive framework for MCDM analysis with ML-powered forecasting.

Architecture
------------
Traditional MCDM + Evidential Reasoning (Yang & Xu, 2002) two-stage hierarchy:
  Stage 1: Within each of 8 criteria, combine 6 traditional method scores via ER
  Stage 2: Combine 8 criterion beliefs via ER with criterion weights

Package Structure
-----------------
ml-mcdm/
├── weighting/          # Criterion weighting methods
│   ├── critic.py       # CRITIC weight calculation (single-level)
│   ├── critic_weighting.py  # Two-level deterministic CRITIC pipeline
│   ├── adaptive.py     # NaN-aware adaptive weight utility
│   ├── normalization.py
│   ├── bootstrap.py
│   └── validation.py
│
├── ranking/
│   ├── hierarchical_pipeline.py  # Unified ranking orchestrator
│   ├── topsis.py       # TOPSIS
│   ├── vikor.py        # VIKOR
│   ├── promethee.py    # PROMETHEE II
│   ├── copras.py       # COPRAS
│   ├── edas.py         # EDAS
│   ├── saw.py          # Simple Additive Weighting (ensemble surrogate)
│   └── evidential_reasoning/  # ER aggregation
│       ├── base.py            # BeliefDistribution, ER engine
│       └── hierarchical_er.py # Two-stage hierarchical ER
│
├── forecasting/        # ML forecasting methods
│   ├── base.py
│   ├── features.py        # Feature engineering
│   ├── unified.py         # Super Learner ensemble
│   ├── conformal.py       # Conformal prediction intervals
│   ├── catboost_forecaster.py
│   ├── bayesian.py
│   └── quantile_forest.py
│
├── analysis/           # Validation & sensitivity
│   ├── sensitivity.py  # Monte Carlo sensitivity analysis
│   └── validation.py   # Cross-validation, bootstrap
│
└── output/             # Result export + visualization
    ├── csv_writer.py
    ├── report_writer.py
    ├── orchestrator.py
    └── visualization/  # Publication-quality figures

Quick Start
-----------
>>> from ml_mcdm.config import Config
>>> from ml_mcdm.pipeline import run_pipeline
>>> result = run_pipeline('data/data.csv', Config())
>>> print(result.summary())

For detailed usage, see individual module documentation.
"""

try:
    from .config import Config, get_default_config, get_config, set_config, reset_config
    from .loggers import (
        setup_logging,
        ConsoleLogger,
        DebugLogger,
        # Backward-compatible shims
        setup_logger,
        get_logger,
        get_module_logger,
        ProgressLogger,
        log_execution,
        log_exceptions,
        log_context,
        timed_operation,
    )
    from .data import DataLoader, PanelData, HierarchyMapping, load_data
    from .pipeline import MLMCDMPipeline, run_pipeline, PipelineResult
    from .output import OutputOrchestrator, CsvWriter, ReportWriter
    from .output.visualization import VisualizationOrchestrator, create_visualizer
except ImportError as _import_err:
    import warnings as _warnings
    _warnings.warn(
        f"ml-mcdm: partial import — some symbols are unavailable: {_import_err}",
        ImportWarning,
        stacklevel=2,
    )

try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNotFoundError
    __version__: str = _pkg_version('ml-mcdm')
except _PkgNotFoundError:
    __version__ = '0.0.0+unknown'  # package not installed (running from source)

__all__ = [
    # Configuration
    'Config',
    'get_default_config',
    'get_config',
    'set_config',
    'reset_config',

    # Logging
    'setup_logging',
    'ConsoleLogger',
    'DebugLogger',
    'setup_logger',
    'get_logger',
    'get_module_logger',
    'ProgressLogger',
    'log_execution',
    'log_exceptions',
    'log_context',
    'timed_operation',

    # Data Loading
    'DataLoader',
    'PanelData',
    'HierarchyMapping',
    'load_data',

    # Pipeline
    'MLMCDMPipeline',
    'run_pipeline',
    'PipelineResult',

    # Output Management
    'OutputOrchestrator',
    'CsvWriter',
    'ReportWriter',

    # Visualization
    'VisualizationOrchestrator',
    'create_visualizer',
]
