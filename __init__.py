"""
ML-MCDM: Machine Learning Enhanced Multi-Criteria Decision Making.

This hierarchical framework integrates traditional MCDM methods with 
advanced Machine Learning forecasting. It provides a multi-stage 
pipeline for assessing and predicting regional performance metrics 
(e.g., provincial rankings).

Architecture
------------
The framework implements a two-stage hierarchical ranking system:
1. **Stage 1**: At the subcriteria level, 6 traditional MCDM methods 
   (TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, SAW) are applied to 
   generate per-criterion scores.
2. **Stage 2**: At the final level, 8 criteria are aggregated using 
   specific criterion weights to produce the overall ranking.

Packages
--------
- **`weighting`**: Statistical and deterministic weight calculation (CRITIC).
- **`ranking`**: Implementation of MCDM methods and hierarchical ranking pipeline.
- **`forecasting`**: ML ensemble (Super Learner) for predictive analytics.
- **`analysis`**: Validation, bootstrap, and sensitivity diagnostics.
- **`output`**: Result orchestration, reporting, and publication-quality 
  visualizations.

Usage
-----
>>> from ml_mcdm import Config, run_pipeline
>>> result = run_pipeline('data/provinces.csv', Config())
>>> result.summary()
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
