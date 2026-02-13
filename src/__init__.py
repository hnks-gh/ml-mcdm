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
src/
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
├── ml/
│   └── forecasting/    # ML forecasting methods
│       ├── tree_ensemble.py   # GB, RF, ExtraTrees
│       ├── linear.py          # Bayesian, Huber, Ridge
│       ├── neural.py          # MLP, Attention
│       └── unified.py         # Unified forecaster
│
├── ensemble/
│   └── evidential_reasoning/  # ER aggregation
│       ├── base.py            # BeliefDistribution, ER engine
│       └── hierarchical_er.py # Two-stage hierarchical ER
│
├── ranking/
│   └── pipeline.py     # Unified ranking orchestrator
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
from .data_loader import DataLoader, PanelData, HierarchyMapping, load_data
from .pipeline import MLMCDMPipeline, run_pipeline, PipelineResult
from .output_manager import OutputManager, create_output_manager
from .visualization import PanelVisualizer, create_visualizer

__version__ = '3.0.0'

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
    'DataLoader', 
    'PanelData',
    'HierarchyMapping',
    'load_data',
    
    # Pipeline
    'MLMCDMPipeline',
    'run_pipeline', 
    'PipelineResult',
    
    # Output Management
    'OutputManager', 
    'create_output_manager',
    
    # Visualization
    'PanelVisualizer', 
    'create_visualizer',
]
