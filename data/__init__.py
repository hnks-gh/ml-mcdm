# -*- coding: utf-8 -*-
"""
Data Management and Ingestion Package
=====================================

This package provides tools for loading hierarchical CSV datasets and 
handling missing data (NaN) across the ML-MCDM pipeline.

Key Modules:
------------
- :mod:`data_loader`: High-level orchestrator for dataset assembly.
- :mod:`missing_data`: Centralized utilities for NaN filtering and imputation.
"""

from .missing_data import (
    MatrixFilterReport,
    filter_all_nan_rows,
    filter_all_nan_columns,
    prepare_decision_matrix,
    fill_missing_features,
    has_complete_target,
    build_ml_panel_data,
)
from .data_loader import (
    DataLoader,
    PanelData,
    HierarchyMapping,
    YearContext,
    load_data,
)

__all__ = [
    # Missing-data utilities
    "MatrixFilterReport",
    "filter_all_nan_rows",
    "filter_all_nan_columns",
    "prepare_decision_matrix",
    "fill_missing_features",
    "has_complete_target",
    "build_ml_panel_data",
    # Data loading
    "DataLoader",
    "PanelData",
    "HierarchyMapping",
    "YearContext",
    "load_data",
]
