# -*- coding: utf-8 -*-
"""
ML-MCDM: Machine Learning Enhanced Multi-Criteria Decision Making
=================================================================

A comprehensive framework for MCDM analysis with ML-powered forecasting.

Architecture
------------
IFS-MCDM + Evidential Reasoning (Yang & Xu, 2002) two-stage hierarchy:
  Stage 1: Within each of 8 criteria, combine 12 method scores via ER
  Stage 2: Combine 8 criterion beliefs via ER with criterion weights

Package Structure
-----------------
ml-mcdm/
├── weighting/          # Criterion weighting methods
│   ├── entropy.py      # Entropy weight calculation
│   ├── critic.py       # CRITIC weight calculation
│   ├── merec.py        # MEREC weight calculation
│   ├── standard_deviation.py
│   └── fusion.py       # Reliability-weighted fusion
│
├── mcdm/
│   ├── traditional/    # Traditional MCDM methods
│   │   ├── topsis.py   # TOPSIS
│   │   ├── vikor.py    # VIKOR
│   │   ├── promethee.py
│   │   ├── copras.py
│   │   ├── edas.py
│   │   └── saw.py      # Simple Additive Weighting
│   │
│   └── ifs/            # IFS MCDM methods (Atanassov, 1986)
│       ├── base.py     # IFN, IFSDecisionMatrix
│       ├── topsis.py   # IFS-TOPSIS
│       ├── vikor.py    # IFS-VIKOR
│       └── ...
│
├── forecasting/        # ML forecasting methods (experimental)
│   ├── tree_ensemble.py   # GB, RF, ExtraTrees
│   ├── linear.py          # Bayesian, Huber, Ridge
│   ├── neural.py          # MLP, Attention
│   └── unified.py         # Unified forecaster
│
├── evidential_reasoning/  # ER aggregation
│   ├── base.py            # BeliefDistribution, ER engine
│   └── hierarchical_er.py # Two-stage hierarchical ER
│
├── ranking/
│   └── pipeline.py     # Unified ranking orchestrator
│
└── analysis/           # Validation & sensitivity
    ├── sensitivity.py  # Monte Carlo sensitivity
    └── validation.py   # Cross-validation, bootstrap

Quick Start
-----------
>>> from config import Config
>>> from pipeline import run_pipeline
>>> result = run_pipeline('data/data.csv', Config())
>>> print(result.summary())

For detailed usage, see individual module documentation.
"""

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
from .data_loader import DataLoader, PanelData, HierarchyMapping, load_data
from .pipeline import MLMCDMPipeline, run_pipeline, PipelineResult
from .output import OutputOrchestrator, CsvWriter, ReportWriter
from .visualization import VisualizationOrchestrator, create_visualizer

__version__ = '5.0.0'

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
