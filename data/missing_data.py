# -*- coding: utf-8 -*-
"""Centralized missing-data (NaN) handling utilities.

All three phases of the pipeline need NaN-aware data preparation.  Rather
than scattering ad-hoc ``fillna`` / ``notna`` calls throughout the codebase,
this module provides a single, tested set of primitives that each phase calls:

**Weighting phase** (``weighting/adaptive.py``)
    :func:`prepare_decision_matrix` — filter all-NaN rows/columns then impute
    remaining partial cells with the per-column mean.

**Ranking phase** (``ranking/pipeline.py``)
    :func:`impute_neutral_score` — fill NaN in a normalized decision matrix
    with 0.5, resulting in a neutral mid-point score that does not bias ranking.

**Forecasting / ML phase** (``forecasting/features.py``)
    :func:`fill_missing_features` — replace NaN feature values with 0.0,
    encoding "no prior information" for the model without fabricating data.
    :func:`has_complete_target`   — validate that a target vector is NaN-free
    before using it as a training label.

Notes
-----
The dataset uses NaN (not zero) to represent missing observations.  A value
of exactly 0.0 is a legitimate governance score and is NEVER treated as
missing.  All checks therefore use ``pd.notna`` / ``np.isnan`` rather than
comparisons against zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class MatrixFilterReport:
    """Records what was kept and excluded during decision-matrix NaN cleanup.

    Attributes
    ----------
    included_rows : list of str
        Row labels (e.g. province names) that survived filtering.
    excluded_rows : list of str
        Row labels dropped because every column value was NaN.
    included_columns : list of str
        Column names (criteria / subcriteria) that survived filtering.
    excluded_columns : list of str
        Column names dropped because every row value was NaN.
    """

    included_rows: List[str] = field(default_factory=list)
    excluded_rows: List[str] = field(default_factory=list)
    included_columns: List[str] = field(default_factory=list)
    excluded_columns: List[str] = field(default_factory=list)

    @property
    def n_included_rows(self) -> int:
        return len(self.included_rows)

    @property
    def n_excluded_rows(self) -> int:
        return len(self.excluded_rows)

    @property
    def n_included_columns(self) -> int:
        return len(self.included_columns)

    @property
    def n_excluded_columns(self) -> int:
        return len(self.excluded_columns)

    def to_dict(self) -> dict:
        """Serialisable summary for logging / output."""
        return {
            "included_rows":    self.n_included_rows,
            "excluded_rows":    self.n_excluded_rows,
            "included_columns": self.n_included_columns,
            "excluded_columns": self.n_excluded_columns,
            "note": "excluded = all-NaN; partial NaN imputed with column mean",
        }


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------

def filter_all_nan_rows(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List, List]:
    """Drop rows where every cell is NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input matrix.  Must *not* contain a non-numeric entity column.

    Returns
    -------
    filtered : pd.DataFrame
        Copy of *df* with all-NaN rows removed.
    included : list
        Index labels of retained rows.
    excluded : list
        Index labels of dropped rows.
    """
    valid = df.notna().any(axis=1)
    return df[valid].copy(), df.index[valid].tolist(), df.index[~valid].tolist()


def filter_all_nan_columns(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Drop columns where every cell is NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input matrix.

    Returns
    -------
    filtered : pd.DataFrame
        Copy of *df* with all-NaN columns removed.
    included : list of str
        Column names of retained columns.
    excluded : list of str
        Column names of dropped columns.
    """
    valid = df.notna().any(axis=0)
    return df.loc[:, valid].copy(), df.columns[valid].tolist(), df.columns[~valid].tolist()


def impute_column_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Fill partial NaN cells with the per-column mean.

    Preserves each column's central tendency without artificially reducing its
    variance.  Rows or columns that are *entirely* NaN should be removed with
    :func:`filter_all_nan_rows` / :func:`filter_all_nan_columns` first.

    Parameters
    ----------
    df : pd.DataFrame
        Matrix that may contain partial NaN cells.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with remaining NaN cells replaced by column means.
    """
    col_means = df.mean(skipna=True)
    return df.fillna(col_means)


# ---------------------------------------------------------------------------
# High-level: weighting / ranking decision-matrix preparation
# ---------------------------------------------------------------------------

def prepare_decision_matrix(
    df: pd.DataFrame,
    entity_col: Optional[str] = None,
    min_rows: int = 2,
    min_cols: int = 2,
) -> Tuple[pd.DataFrame, MatrixFilterReport]:
    """Full missing-data preparation pipeline for a numeric decision matrix.

    Applies three sequential steps:

    1. **Strip entity column** — remove the province/entity identifier column
       (if *entity_col* is given and present) so only numeric criterion values
       remain.
    2. **Filter all-NaN rows** — provinces with no valid data for any criterion
       are excluded entirely.
    3. **Filter all-NaN columns** — criteria where every province is NaN are
       excluded entirely.
    4. **Impute partial NaN** — remaining isolated NaN cells are filled with the
       column mean so the downstream calculator receives a complete numeric
       matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Decision matrix.  May include an entity identifier column.
    entity_col : str, optional
        Name of the entity identifier column (e.g. ``'Province'``).  Stripped
        before filtering; *not* present in the returned DataFrame.
    min_rows : int
        Minimum number of rows required after filtering.
    min_cols : int
        Minimum number of columns required after filtering.

    Returns
    -------
    data : pd.DataFrame
        Cleaned, NaN-free numeric decision matrix.
    report : MatrixFilterReport
        Details of what was included and excluded.

    Raises
    ------
    ValueError
        If fewer than *min_rows* or *min_cols* remain after NaN filtering.
    """
    # Separate entity label column
    if entity_col is not None and entity_col in df.columns:
        row_labels: list = df[entity_col].tolist()
        data = df.drop(columns=[entity_col]).copy()
    elif df.index.name == entity_col and entity_col is not None:
        row_labels = df.index.tolist()
        data = df.copy()
    else:
        row_labels = list(df.index)
        data = df.copy()

    original_columns = data.columns.tolist()

    # Step 1: Filter all-NaN rows
    row_mask = data.notna().any(axis=1)
    included_rows = [row_labels[i] for i, v in enumerate(row_mask) if v]
    excluded_rows = [row_labels[i] for i, v in enumerate(row_mask) if not v]

    n_valid_rows = int(row_mask.sum())
    if n_valid_rows < min_rows:
        raise ValueError(
            f"Insufficient rows after NaN filtering: {n_valid_rows} < {min_rows}"
        )
    data = data[row_mask].copy()

    # Step 2: Filter all-NaN columns
    col_mask = data.notna().any(axis=0)
    included_cols = [c for c, v in zip(original_columns, col_mask) if v]
    excluded_cols = [c for c, v in zip(original_columns, col_mask) if not v]

    n_valid_cols = int(col_mask.sum())
    if n_valid_cols < min_cols:
        raise ValueError(
            f"Insufficient columns after NaN filtering: {n_valid_cols} < {min_cols}"
        )
    data = data.loc[:, col_mask].copy()

    # Step 3: Impute remaining partial NaN cells with column mean
    data = impute_column_mean(data)

    report = MatrixFilterReport(
        included_rows=included_rows,
        excluded_rows=excluded_rows,
        included_columns=included_cols,
        excluded_columns=excluded_cols,
    )
    return data, report


# ---------------------------------------------------------------------------
# Ranking phase utilities
# ---------------------------------------------------------------------------

def impute_neutral_score(
    df: pd.DataFrame,
    neutral: float = 0.5,
) -> pd.DataFrame:
    """Fill NaN cells with *neutral* (default ``0.5``).

    Used in the ranking phase after min-max normalisation: provinces with
    partially-missing sub-criterion data receive the neutral mid-point score
    so they are not artificially promoted or demoted relative to provinces
    that have complete data.

    Parameters
    ----------
    df : pd.DataFrame
        Normalised decision matrix (values in ``[0, 1]``) that may contain NaN
        cells for provinces lacking a particular sub-criterion score.
    neutral : float
        Fill value.  Defaults to ``0.5`` (mid-point of the ``[0, 1]`` scale).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with NaN replaced by *neutral*.
    """
    return df.fillna(neutral)


# ---------------------------------------------------------------------------
# Forecasting / ML phase utilities
# ---------------------------------------------------------------------------

def fill_missing_features(X: "np.ndarray | pd.DataFrame") -> np.ndarray:
    """Replace NaN feature values with ``0.0`` ("no prior information").

    Used in the forecasting/ML phase when a lag value or cross-entity statistic
    cannot be computed (e.g. the first observed year for a province).  Setting
    the feature to zero tells the model the information is unavailable without
    fabricating a plausible value — this is distinct from statistical imputation.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix that may contain NaN for missing lag / rolling values.

    Returns
    -------
    np.ndarray
        Array with dtype ``float64``; all NaN replaced by ``0.0``.
    """
    arr = np.asarray(X, dtype=float)
    return np.where(np.isnan(arr), 0.0, arr)


def has_complete_target(target: "np.ndarray | list") -> bool:
    """Return ``True`` if *target* contains no NaN values.

    Used in the forecasting/ML phase to decide whether a training sample
    can be used.  No imputation is performed on target (label) values —
    incomplete targets are excluded from training entirely.

    Parameters
    ----------
    target : array-like
        Target vector (sub-criterion scores for a single province-year).

    Returns
    -------
    bool
        ``True`` if every element is a finite real number; ``False`` otherwise.
    """
    arr = np.asarray(target, dtype=float)
    return not bool(np.any(np.isnan(arr)))
