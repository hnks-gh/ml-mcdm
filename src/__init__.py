# -*- coding: utf-8 -*-
"""
ML-MCDM: Panel Data Multi-Criteria Decision Analysis Framework
=================================================================

A production-ready framework combining Machine Learning and TOPSIS methods
for comprehensive panel data analysis and sustainability assessment.

Main Components:
- MCDM Methods: TOPSIS, Dynamic TOPSIS, VIKOR, Fuzzy TOPSIS
- ML Methods: Panel Regression, Random Forest, LSTM, Rough Sets
- Ensemble: Stacking, Rank Aggregation (Borda, Copeland)
- Analysis: Convergence, Sensitivity, Validation

Output Structure:
- outputs/figures/ : High-resolution individual charts (300 DPI)
- outputs/results/ : Complete numerical results (CSV)
- outputs/reports/ : Comprehensive analysis reports
"""

from .config import Config, get_default_config
from .logger import setup_logger
from .data_loader import PanelDataLoader, PanelData
from .main import MLTOPSISPipeline, run_pipeline, PipelineResult
from .output_manager import OutputManager, create_output_manager
from .visualization import PanelVisualizer, create_visualizer

__version__ = '2.0.0'
__author__ = 'ML-MCDM Team'

__all__ = [
    # Configuration
    'Config',
    'get_default_config',
    # Logging
    'setup_logger',
    # Data
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
