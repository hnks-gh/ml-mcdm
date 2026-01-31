# -*- coding: utf-8 -*-
"""ML-MCDM: Machine Learning enhanced Multi-Criteria Decision Making framework."""

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

__version__ = '2.0.0'

__all__ = [
    'Config', 'get_default_config',
    'setup_logger', 'get_logger', 'get_module_logger',
    'ProgressLogger', 'PipelineLogger', 'LoggerFactory',
    'log_execution', 'log_exceptions', 'log_context', 'timed_operation',
    'PanelDataLoader', 'PanelData',
    'MLTOPSISPipeline', 'run_pipeline', 'PipelineResult',
    'OutputManager', 'create_output_manager',
    'PanelVisualizer', 'create_visualizer',
]
