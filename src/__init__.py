# -*- coding: utf-8 -*-
"""
ML-MCDM: Machine Learning Enhanced Multi-Criteria Decision Making
=================================================================

A comprehensive framework for MCDM analysis with ML-powered forecasting.

Package Structure
-----------------
src/
├── weighting/          # Criterion weighting methods
│   ├── entropy.py      # Entropy weight calculation
│   ├── critic.py       # CRITIC weight calculation
│   └── ensemble.py     # Ensemble weight combination
│
├── mcdm/
│   ├── traditional/    # Traditional MCDM methods
│   │   ├── topsis.py   # TOPSIS with multi-period variants
│   │   ├── vikor.py    # VIKOR with multi-period variants
│   │   ├── promethee.py
│   │   ├── copras.py
│   │   └── edas.py
│   │
│   └── fuzzy/          # Fuzzy MCDM methods
│       ├── base.py     # Triangular Fuzzy Numbers
│       ├── topsis.py   # Fuzzy TOPSIS
│       ├── vikor.py    # Fuzzy VIKOR
│       └── ...
│
├── ml/
│   └── forecasting/    # ML forecasting methods
│       ├── tree_ensemble.py   # GB, RF, ExtraTrees
│       ├── linear.py          # Bayesian, Huber, Ridge
│       ├── neural.py          # MLP, Attention
│       └── unified.py         # Unified forecaster
│
├── ensemble/
│   └── aggregation/    # Rank aggregation & stacking
│       ├── borda.py    # Borda Count
│       ├── copeland.py # Copeland method
│       ├── kemeny.py   # Kemeny-Young
│       └── stacking.py # Stacking ensemble
│
└── analysis/           # Validation & sensitivity
    ├── sensitivity.py  # Monte Carlo sensitivity
    └── validation.py   # Cross-validation, bootstrap

Quick Start
-----------
>>> from src import Config, run_pipeline
>>> result = run_pipeline('data/data.csv', Config())
>>> print(result.summary())

For detailed usage, see individual module documentation.
"""

from .config import Config, get_default_config
from .logger import (
    setup_logger, 
    get_logger, 
    get_module_logger,
    ProgressLogger, 
    PipelineLogger,
    LoggerFactory,
    log_execution,
    log_exceptions,
    log_context,
    timed_operation,
)
from .data_loader import PanelDataLoader, PanelData
from .pipeline import MLTOPSISPipeline, run_pipeline, PipelineResult
from .output_manager import OutputManager, create_output_manager
from .visualization import PanelVisualizer, create_visualizer

__version__ = '2.1.0'

__all__ = [
    # Configuration
    'Config', 
    'get_default_config',
    
    # Logging
    'setup_logger', 
    'get_logger', 
    'get_module_logger',
    'ProgressLogger', 
    'PipelineLogger', 
    'LoggerFactory',
    'log_execution', 
    'log_exceptions', 
    'log_context', 
    'timed_operation',
    
    # Data Loading
    'PanelDataLoader', 
    'PanelData',
    
    # Pipeline
    'MLTOPSISPipeline', 
    'run_pipeline', 
    'PipelineResult',
    
    # Output Management
    'OutputManager', 
    'create_output_manager',
    
    # Visualization
    'PanelVisualizer', 
    'create_visualizer',
]
