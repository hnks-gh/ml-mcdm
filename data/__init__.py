# -*- coding: utf-8 -*-
"""Data package — CSV dataset loading and shared missing-data utilities.

Directory layout::

    data/
    ├── csv/            ← input panel data (2011.csv … 2024.csv)
    ├── codebook/       ← codebook_criteria.csv, codebook_subcriteria.csv
    ├── data_loader.py  ← :class:`DataLoader`, :class:`PanelData`, …
    └── missing_data.py ← centralised NaN-handling primitives

The :mod:`data.missing_data` module provides NaN-filtering utilities shared
across the weighting, ranking, and forecasting phases.
The :mod:`data.data_loader` module loads the full hierarchical panel dataset.
"""

from .missing_data import (
    MatrixFilterReport,
    filter_all_nan_rows,
    filter_all_nan_columns,
    impute_column_mean,
    prepare_decision_matrix,
    impute_neutral_score,
    fill_missing_features,
    has_complete_target,
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
    "impute_column_mean",
    "prepare_decision_matrix",
    "impute_neutral_score",
    "fill_missing_features",
    "has_complete_target",
    # Data loading
    "DataLoader",
    "PanelData",
    "HierarchyMapping",
    "YearContext",
    "load_data",
]
