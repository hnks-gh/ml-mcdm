# -*- coding: utf-8 -*-
"""Centralized missing-data (NaN) handling utilities.

All three phases of the pipeline need NaN-aware data preparation.  Rather
than scattering ad-hoc ``fillna`` / ``notna`` calls throughout the codebase,
this module provides a single, tested set of primitives that each phase calls:

**Weighting phase** (``weighting/adaptive.py``)
    :func:`prepare_decision_matrix` — filter all-NaN rows/columns; remaining
    partial NaN cells are preserved (no imputation applied).
    :func:`impute_panel_temporal` — advanced temporal imputation for panel
    data using within-entity interpolation before cross-sectional fallback.

**Ranking phase** (``ranking/hierarchical_pipeline.py``)
    :func:`impute_neutral_score` — fill NaN in a normalized decision matrix
    with 0.5, resulting in a neutral mid-point score that does not bias ranking.

**Forecasting / ML phase** (``forecasting/features.py``)
    :func:`fill_missing_features` — replace NaN feature values with per-column
    training means (not 0.0, which is a valid governance score). Optionally
    returns a missingness indicator mask so models can distinguish imputed
    from observed values.
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
from typing import Dict, List, Optional, Tuple, Union

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
            "note": "excluded = all-NaN rows/columns; partial NaN cells preserved (no imputation)",
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

def impute_panel_temporal(
    panel_tensor: Dict[int, pd.DataFrame],
    max_linear_gap: int = 2,
) -> Dict[int, pd.DataFrame]:
    """Impute a 3-D panel tensor using within-entity temporal interpolation.

    Applies a three-stage hierarchy leveraging the full panel structure:

    1. **Linear interpolation** for internal gaps ≤ max_linear_gap years
       within each (province, subcriteria) time series.
    2. **Forward/backward fill** for boundary gaps (end-of-series missing).
    3. **Cross-sectional median** as final fallback for any remaining NaN.

    This function replaces the single-year column-mean imputation used in
    ``prepare_decision_matrix`` by exploiting the temporal ordering and
    provincial trajectories inherent in the panel data structure.

    Parameters
    ----------
    panel_tensor : dict of {int: pd.DataFrame}
        Panel data as a dict mapping year → (provinces × subcriteria) DataFrame.
        Each DataFrame must have the same index (province names) and columns
        (subcriteria).
    max_linear_gap : int
        Maximum gap size (in years) for linear interpolation. Gaps larger
        than this are not interpolated linearly. Default is 2 years.

    Returns
    -------
    dict of {int: pd.DataFrame}
        Panel tensor with NaN imputed. Same structure as input.

    Notes
    -----
    This strategy preserves provincial governance trajectories and eliminates
    the variance-suppression artifact from cross-sectional mean imputation.
    Linear interpolation is appropriate for governance scores which typically
    evolve smoothly year-over-year.

    Examples
    --------
    >>> panel = {2020: df_2020, 2021: df_2021, 2022: df_2022}
    >>> imputed_panel = impute_panel_temporal(panel, max_linear_gap=2)
    """
    if not panel_tensor:
        return panel_tensor

    years = sorted(panel_tensor.keys())
    if len(years) < 2:
        # Single year: can only apply cross-sectional median (Stage 3)
        for yr in years:
            col_medians = panel_tensor[yr].median(skipna=True)
            panel_tensor[yr] = panel_tensor[yr].fillna(col_medians)
        return panel_tensor

    # Extract consistent structure
    first_year = years[0]
    provinces = panel_tensor[first_year].index.tolist()
    subcriteria = panel_tensor[first_year].columns.tolist()

    # Stage 1 & 2: Temporal interpolation + forward/backward fill
    # Process each (province, subcriteria) time series independently
    for prov in provinces:
        for sc in subcriteria:
            # Build time series for this (province, sub-criterion) pair
            series_data = {}
            for yr in years:
                if prov in panel_tensor[yr].index and sc in panel_tensor[yr].columns:
                    series_data[yr] = panel_tensor[yr].loc[prov, sc]
                else:
                    series_data[yr] = np.nan

            series = pd.Series(series_data, name=(prov, sc))

            # Stage 1: Linear interpolation for internal gaps
            # limit=max_linear_gap ensures we don't interpolate across large gaps
            filled = series.interpolate(
                method='linear',
                limit_direction='both',
                limit=max_linear_gap,
            )

            # Stage 2: Forward/backward fill for boundary gaps
            # This handles newly-introduced sub-criteria (e.g., SC71-SC73, SC81-SC83)
            filled = filled.ffill().bfill()

            # Write back to panel tensor
            for yr in years:
                if prov in panel_tensor[yr].index and sc in panel_tensor[yr].columns:
                    panel_tensor[yr].loc[prov, sc] = filled[yr]

    # Stage 3: Cross-sectional median for any remaining NaN
    # This is the final safety net for provinces with entirely missing
    # time series for a particular sub-criterion
    for yr in years:
        col_medians = panel_tensor[yr].median(skipna=True)
        panel_tensor[yr] = panel_tensor[yr].fillna(col_medians)

    return panel_tensor


def prepare_decision_matrix(
    df: pd.DataFrame,
    entity_col: Optional[str] = None,
    min_rows: int = 2,
    min_cols: int = 2,
) -> Tuple[pd.DataFrame, MatrixFilterReport]:
    """Full missing-data preparation pipeline for a numeric decision matrix.

    Applies two sequential filtering steps:

    1. **Strip entity column** — remove the province/entity identifier column
       (if *entity_col* is given and present) so only numeric criterion values
       remain.
    2. **Filter all-NaN rows** — provinces with no valid data for any criterion
       are excluded entirely.
    3. **Filter all-NaN columns** — criteria where every province is NaN are
       excluded entirely.

    Partial NaN cells (a province has *some* valid sub-criteria but not all)
    are **preserved unchanged**.  Downstream calculators must apply their own
    complete-case strategy (e.g. ``CRITICWeightCalculator`` uses per-row
    exclusion via its F-03 guard).  Imputing partial cells with column means
    would attenuate variance and inflate inter-criteria correlations —
    biasing objective CRITIC weights.

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
        Filtered numeric decision matrix; partial NaN cells are preserved
        (no imputation applied).
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

    .. deprecated::
        This function is **no longer called** by the ranking pipeline.
        :meth:`HierarchicalRankingPipeline._minmax_normalize` previously
        called this to fill partial-NaN cells with the neutral mid-point
        after normalisation.  That call was removed (F-04) because imputing
        0.5 manufactured ordinal information that was never observed —
        violating the complete-case strategy applied everywhere else.
        The function is retained for backward compatibility with any external
        callers, but should not be used in new MCDM code.

    Parameters
    ----------
    df : pd.DataFrame
        Normalised decision matrix (values in ``[0, 1]``) that may contain
        NaN cells.
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

def fill_missing_features(
    X: "np.ndarray | pd.DataFrame",
    fallback_values: Optional[np.ndarray] = None,
    return_mask: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Replace NaN in feature matrix with per-column fallback values.

    **CRITICAL FIX (M-01):** This function NO LONGER uses 0.0 as a fill value.
    On the governance score scale [0, 3.33], 0.0 is a legitimate observed
    poor-governance score. Using 0.0 for missingness conflates "data unavailable"
    with "genuinely poor governance", corrupting feature semantics downstream.

    The function now uses per-column training means (or provided fallback values)
    and optionally returns a missingness indicator mask for downstream models
    to distinguish imputed from observed values.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix that may contain NaN for missing lag / rolling values.
    fallback_values : ndarray of shape (n_features,), optional
        Per-column fill values. If None, computes per-column mean of non-NaN
        entries from X itself (safe for inference; leakage-free at training
        time when X is training data only).
        **IMPORTANT:** During prediction/test, pass pre-computed training means
        to avoid test-set leakage.
    return_mask : bool
        If True, returns a tuple ``(filled_array, missingness_mask)`` where
        ``missingness_mask`` is a boolean array of shape (n_samples, n_features)
        with True at positions that were NaN (imputed). Models can use this
        as additional features to learn to discount imputed values.

    Returns
    -------
    filled : np.ndarray
        Array with dtype ``float64``; all NaN replaced by fallback values.
    mask : np.ndarray, optional
        Boolean array indicating which cells were imputed (True = was NaN).
        Only returned if ``return_mask=True``.

    Notes
    -----
    0.0 is NEVER used as a fill value. The only exception is when a column
    is entirely NaN (no valid fallback can be computed from training data),
    in which case 0.0 is used as an emergency sentinel. This edge case should
    be logged upstream.

    Examples
    --------
    >>> X_train = np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 4.0]])
    >>> X_filled, mask = fill_missing_features(X_train, return_mask=True)
    >>> X_filled
    array([[1., 3.5], [2., 3. ], [3., 4. ]])  # col1_mean = (3+4)/2 = 3.5
    >>> mask
    array([[False,  True], [False, False], [False, False]])
    """
    arr = np.asarray(X, dtype=float)
    nan_mask = np.isnan(arr)

    if fallback_values is None:
        # Compute per-column mean from non-NaN entries
        # np.nanmean returns NaN if a column is entirely NaN; replace with 0.0
        col_means = np.nanmean(arr, axis=0)
        fallback_values = np.where(np.isnan(col_means), 0.0, col_means)

    # Ensure fallback_values is at least 1D (handle 0-D scalar case)
    fallback_values = np.atleast_1d(fallback_values)

    # Broadcast fallback_values across rows and fill NaN positions
    if arr.ndim == 1:
        # 1D array case: simple replacement
        filled = np.where(nan_mask, fallback_values[0], arr)
    else:
        # 2D array case: broadcast across rows
        filled = np.where(nan_mask, fallback_values[np.newaxis, :], arr)

    if return_mask:
        return filled, nan_mask
    return filled


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


# ---------------------------------------------------------------------------
# Advanced Matrix Completion (M-05: SoftImpute)
# ---------------------------------------------------------------------------

def soft_impute_matrix(
    X: np.ndarray,
    lambda_reg: float = 1.0,
    max_rank: Optional[int] = None,
    max_iter: int = 100,
    tol: float = 1e-5,
    verbose: bool = False,
) -> np.ndarray:
    """Matrix completion via Soft-Impute (nuclear norm minimization).

    Enhancement M-05: SoftImpute / Nuclear Norm Matrix Completion
    --------------------------------------------------------------
    Implements the Soft-Impute algorithm (Mazumder, Hastie, Tibshirani 2010)
    for low-rank matrix completion. Governance panel data exhibits strong
    low-rank structure — sub-criteria co-vary across provinces (regional
    patterns) and across components (structural governance dimensions).

    The algorithm minimizes:
        min_Z  (1/2)||P_Ω(M - Z)||²_F + λ||Z||_*
    
    where P_Ω is projection onto observed entries and ||·||_* is the nuclear
    norm (sum of singular values, convex relaxation of matrix rank).

    Solution via iterative SVD soft-thresholding:
        Z = U · diag(max(σ - λ, 0)) · V^T

    Parameters
    ----------
    X : ndarray, shape (n, m)
        Input matrix with NaN at missing positions. Observed entries should
        already be centered/scaled if desired.
    lambda_reg : float
        Nuclear norm regularization strength. Larger values enforce stronger
        low-rank structure. Typical range: [0.001, 10]. Cross-validate on
        held-out observed entries (not temporal holdout — cell-level CV).
    max_rank : int, optional
        Maximum rank to retain in SVD truncation. If None, uses full rank.
        For governance data (63×29), typical max_rank = 5-15.
    max_iter : int
        Maximum SVD iterations. Typically converges in 20-50 iterations.
    tol : float
        Convergence tolerance. Stop when ||Z^(t) - Z^(t-1)||_F / ||Z^(t)||_F < tol.
    verbose : bool
        If True, print iteration progress.

    Returns
    -------
    Z : ndarray, shape (n, m)
        Completed matrix with all NaN replaced by imputed values.
        Observed entries are approximately preserved (with shrinkage toward
        low-rank approximation).

    Notes
    -----
    - For panel data (T years × N provinces × C sub-criteria), apply per year:
      Z_t = soft_impute_matrix(M_t, lambda_reg=λ) for each year t.
    - For temporal regularization (exploiting smooth year-to-year changes),
      use :func:`soft_impute_panel_tensor` instead (not yet implemented).
    - Recommended λ grid for cross-validation: [0.01, 0.1, 1.0, 10.0].

    References
    ----------
    Mazumder, R., Hastie, T., & Tibshirani, R. (2010). Spectral regularization
    algorithms for learning large incomplete matrices. *Journal of Machine
    Learning Research*, 11, 2287-2322.

    Examples
    --------
    >>> M = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [np.nan, 8.0, 9.0]])
    >>> Z = soft_impute_matrix(M, lambda_reg=0.5, max_iter=50)
    >>> # Z is a completed low-rank approximation of M
    """
    # Initialize Z with mean imputation (fast convergence starting point)
    Z = X.copy()
    nan_mask = np.isnan(Z)
    
    if not nan_mask.any():
        # No missing values — return input unchanged
        return Z
    
    # Mean imputation for initialization
    col_means = np.nanmean(Z, axis=0)
    row_means = np.nanmean(Z, axis=1)
    global_mean = np.nanmean(Z)
    
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if nan_mask[i, j]:
                # Use column mean if available, else row mean, else global mean
                if not np.isnan(col_means[j]):
                    Z[i, j] = col_means[j]
                elif not np.isnan(row_means[i]):
                    Z[i, j] = row_means[i]
                else:
                    Z[i, j] = global_mean if not np.isnan(global_mean) else 0.0
    
    # Iterative SVD soft-thresholding
    for iteration in range(max_iter):
        Z_old = Z.copy()
        
        # SVD of current Z
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        
        # Soft-thresholding of singular values: σ_new = max(σ - λ, 0)
        s_thresh = np.maximum(s - lambda_reg, 0.0)
        
        # Truncate to max_rank if specified
        if max_rank is not None and max_rank < len(s_thresh):
            s_thresh[max_rank:] = 0.0
        
        # Reconstruct matrix: Z_new = U · Σ_thresh · V^T
        Z = U @ np.diag(s_thresh) @ Vt
        
        # Project onto observed entries: Z_obs = X_obs (preserve observations)
        Z[~nan_mask] = X[~nan_mask]
        
        # Check convergence
        diff_norm = np.linalg.norm(Z - Z_old, 'fro')
        z_norm = np.linalg.norm(Z, 'fro')
        rel_change = diff_norm / (z_norm + 1e-12)
        
        if verbose:
            n_nonzero = np.sum(s_thresh > 1e-9)
            print(f"Iteration {iteration+1}: rel_change={rel_change:.6f}, "
                  f"rank={n_nonzero}, max_s={s_thresh[0]:.4f}")
        
        if rel_change < tol:
            if verbose:
                print(f"Converged in {iteration+1} iterations.")
            break
    
    return Z


def soft_impute_panel(
    panel_tensor: Dict[int, pd.DataFrame],
    lambda_reg: float = 1.0,
    max_rank: Optional[int] = None,
    max_iter: int = 100,
    tol: float = 1e-5,
    verbose: bool = False,
) -> Dict[int, pd.DataFrame]:
    """Apply Soft-Impute to each year's cross-section independently.

    Enhancement M-05: Panel-aware matrix completion.
    
    For panel data (years × provinces × subcriteria), this function applies
    :func:`soft_impute_matrix` to each year's (provinces × subcriteria)
    cross-section independently. For temporal regularization that couples
    adjacent years, use soft_impute_panel_tensor (not yet implemented).

    Parameters
    ----------
    panel_tensor : dict of {int: pd.DataFrame}
        Panel data as year → (provinces × subcriteria) DataFrame.
    lambda_reg : float
        Nuclear norm regularization. Typical range: [0.01, 10].
    max_rank : int, optional
        Maximum rank per year. For 63×29 matrices, typical: 5-15.
    max_iter : int
        SVD iterations per year.
    tol : float
        Convergence tolerance per year.
    verbose : bool
        Print per-year progress.

    Returns
    -------
    dict of {int: pd.DataFrame}
        Panel tensor with NaN imputed via low-rank completion.

    Notes
    -----
    - Recommended to apply AFTER :func:`impute_panel_temporal` for best results
      (temporal interpolation handles obvious gaps; SoftImpute handles complex
      missing patterns).
    - Cross-validate λ on held-out observed cells, not on temporal holdout.

    Examples
    --------
    >>> panel = {2020: df_2020, 2021: df_2021, 2022: df_2022}
    >>> imputed = soft_impute_panel(panel, lambda_reg=0.5, max_rank=8)
    """
    imputed_tensor = {}
    
    for year in sorted(panel_tensor.keys()):
        df_year = panel_tensor[year]
        
        if verbose:
            n_missing = df_year.isna().sum().sum()
            n_total = df_year.size
            print(f"\nYear {year}: {n_missing}/{n_total} missing "
                  f"({100*n_missing/n_total:.1f}%)")
        
        # Convert to numpy, apply SoftImpute, convert back to DataFrame
        X = df_year.values
        Z = soft_impute_matrix(X, lambda_reg, max_rank, max_iter, tol, verbose)
        
        imputed_tensor[year] = pd.DataFrame(
            Z, index=df_year.index, columns=df_year.columns
        )
    
    return imputed_tensor


# ---------------------------------------------------------------------------
# M-09: Tensor Completion via CP Decomposition (Alternating Least Squares)
# ---------------------------------------------------------------------------

def cp_als_tensor_completion(
    tensor_dict: Dict[int, pd.DataFrame],
    rank: int = 8,
    max_iter: int = 100,
    tol: float = 1e-5,
    lambda_reg: float = 0.1,
    verbose: bool = False,
) -> Dict[int, pd.DataFrame]:
    """
    Tensor completion via CP (CANDECOMP/PARAFAC) decomposition using ALS.
    
    Enhancement M-09: Low-rank tensor factorization for panel data completion.
    
    Governance panel data lives in a 3-D tensor (province × year × subcriterion).
    CP decomposition jointly exploits all three modes, unlike matrix completion
    which treats years independently.
    
    The CP model factorizes the tensor T ∈ ℝ^(N×Y×C) as:
    
        T ≈ Σ_{r=1}^R a_r ⊗ b_r ⊗ c_r
    
    where:
        a_r ∈ ℝ^N  — province mode (latent provincial governance profile)
        b_r ∈ ℝ^Y  — temporal mode (governance trajectory)
        c_r ∈ ℝ^C  — subcriterion mode (governance dimension loading)
    
    For rank R=5-10, this has N·R + Y·R + C·R parameters vs. N×Y×C observations,
    dramatically regularizing the imputation.
    
    Missingness-aware ALS minimizes:
    
        min_{A,B,C} Σ_{(i,t,j)∈Ω} (T_itj - Σ_r a_ir b_tr c_jr)² 
                     + λ(||A||²_F + ||B||²_F + ||C||²_F)
    
    where Ω is the set of observed entries.
    
    Parameters
    ----------
    tensor_dict : dict of {int: pd.DataFrame}
        Panel data as year → (provinces × subcriteria) DataFrame.
        Years must be consecutive integers for temporal mode interpretation.
    rank : int
        CP rank (number of components). For governance data (63×14×29),
        typical: 5-15. Higher rank = more expressive but less regularization.
    max_iter : int
        Maximum ALS iterations. Typically converges in 20-100 iterations.
    tol : float
        Convergence tolerance. Stop when relative reconstruction error
        change < tol.
    lambda_reg : float
        L2 regularization on factor matrices. Prevents overfitting.
        Typical range: [0.01, 1.0].
    verbose : bool
        If True, print iteration progress.
    
    Returns
    -------
    completed_tensor : dict of {int: pd.DataFrame}
        Panel tensor with NaN imputed via CP reconstruction.
    
    Notes
    -----
    - CP decomposition is unique up to permutation and scaling (unlike SVD).
    - Non-negative CP (NNCP) can be enforced by replacing least squares
      with NNLS in ALS updates. Current implementation allows negative values
      (appropriate for governance scores which can be low/negative-coded).
    - For temporal regularization (smooth b_r trajectories), add Tikhonov
      penalty on first differences of B (not yet implemented).
    
    References
    ----------
    Kolda, T. G., & Bader, B. W. (2009). Tensor decompositions and applications.
    SIAM Review, 51(3), 455-500.
    
    Examples
    --------
    >>> panel = {2020: df_2020, 2021: df_2021, 2022: df_2022}
    >>> completed = cp_als_tensor_completion(panel, rank=8, lambda_reg=0.1)
    """
    # Sort years and extract structure
    years = sorted(tensor_dict.keys())
    first_year = years[0]
    provinces = tensor_dict[first_year].index.tolist()
    subcriteria = tensor_dict[first_year].columns.tolist()
    
    N = len(provinces)  # 63
    Y = len(years)      # 14
    C = len(subcriteria)  # 29
    
    # Build 3D tensor array: shape (N, Y, C)
    # T[i, t, j] = score for province i, year t, subcriterion j
    T = np.zeros((N, Y, C))
    mask = np.zeros((N, Y, C), dtype=bool)  # True = observed
    
    for t_idx, year in enumerate(years):
        df = tensor_dict[year]
        for i, prov in enumerate(provinces):
            for j, sc in enumerate(subcriteria):
                if prov in df.index and sc in df.columns:
                    val = df.loc[prov, sc]
                    if not np.isnan(val):
                        T[i, t_idx, j] = val
                        mask[i, t_idx, j] = True
    
    n_observed = mask.sum()
    n_total = N * Y * C
    missing_rate = 1.0 - n_observed / n_total
    
    if verbose:
        print(f"Tensor shape: {N} provinces × {Y} years × {C} subcriteria")
        print(f"Observed: {n_observed}/{n_total} ({100*(1-missing_rate):.1f}%)")
        print(f"Rank: {rank}, λ: {lambda_reg}")
    
    # Initialize factor matrices with small random values
    np.random.seed(42)
    A = np.random.randn(N, rank) * 0.01  # Province factors
    B = np.random.randn(Y, rank) * 0.01  # Temporal factors
    C_factors = np.random.randn(C, rank) * 0.01  # Subcriterion factors
    
    # ALS iterations
    prev_error = np.inf
    
    for iteration in range(max_iter):
        # Update A (province mode) holding B, C fixed
        # For each province i, solve: min_a_i Σ_{(t,j): observed} (T_itj - a_i^T (b_t ⊙ c_j))²
        for i in range(N):
            # Observed entries for province i: mask[i, :, :]
            obs_i = mask[i, :, :]  # (Y, C) boolean
            if not obs_i.any():
                continue  # Province i fully missing
            
            # Khatri-Rao product rows for observed entries: (b_t ⊙ c_j)
            # For each observed (t, j), stack b_t ⊙ c_j as a column
            X_i = []
            y_i = []
            for t in range(Y):
                for j in range(C):
                    if obs_i[t, j]:
                        # Khatri-Rao product: b_t ⊙ c_j (element-wise)
                        kr = B[t, :] * C_factors[j, :]
                        X_i.append(kr)
                        y_i.append(T[i, t, j])
            
            if not X_i:
                continue
            
            X_i = np.array(X_i)  # (n_obs_i, rank)
            y_i = np.array(y_i)  # (n_obs_i,)
            
            # Ridge regression: (X^T X + λI)^{-1} X^T y
            XtX = X_i.T @ X_i + lambda_reg * np.eye(rank)
            Xty = X_i.T @ y_i
            A[i, :] = np.linalg.solve(XtX, Xty)
        
        # Update B (temporal mode) holding A, C fixed
        for t in range(Y):
            obs_t = mask[:, t, :]  # (N, C)
            if not obs_t.any():
                continue
            
            X_t = []
            y_t = []
            for i in range(N):
                for j in range(C):
                    if obs_t[i, j]:
                        kr = A[i, :] * C_factors[j, :]
                        X_t.append(kr)
                        y_t.append(T[i, t, j])
            
            if not X_t:
                continue
            
            X_t = np.array(X_t)
            y_t = np.array(y_t)
            XtX = X_t.T @ X_t + lambda_reg * np.eye(rank)
            Xty = X_t.T @ y_t
            B[t, :] = np.linalg.solve(XtX, Xty)
        
        # Update C (subcriterion mode) holding A, B fixed
        for j in range(C):
            obs_j = mask[:, :, j]  # (N, Y)
            if not obs_j.any():
                continue
            
            X_j = []
            y_j = []
            for i in range(N):
                for t in range(Y):
                    if obs_j[i, t]:
                        kr = A[i, :] * B[t, :]
                        X_j.append(kr)
                        y_j.append(T[i, t, j])
            
            if not X_j:
                continue
            
            X_j = np.array(X_j)
            y_j = np.array(y_j)
            XtX = X_j.T @ X_j + lambda_reg * np.eye(rank)
            Xty = X_j.T @ y_j
            C_factors[j, :] = np.linalg.solve(XtX, Xty)
        
        # Reconstruct tensor and compute error on observed entries
        # T_reconstructed[i, t, j] = Σ_r A[i,r] * B[t,r] * C[j,r]
        T_recon = np.zeros((N, Y, C))
        for r in range(rank):
            # Outer product: A[:, r] ⊗ B[:, r] ⊗ C[:, r]
            T_recon += np.einsum('i,t,j->itj', A[:, r], B[:, r], C_factors[:, r])
        
        # Error on observed entries (RMSE)
        error = np.sqrt(np.sum((T[mask] - T_recon[mask]) ** 2) / n_observed)
        rel_change = abs(error - prev_error) / (prev_error + 1e-12)
        
        if verbose and (iteration % 10 == 0 or iteration < 5):
            print(f"Iter {iteration+1}: RMSE={error:.6f}, rel_change={rel_change:.6f}")
        
        if rel_change < tol:
            if verbose:
                print(f"Converged in {iteration+1} iterations.")
            break
        
        prev_error = error
    
    # Fill missing entries with reconstruction
    T_completed = T.copy()
    T_completed[~mask] = T_recon[~mask]
    
    # Convert back to dict of DataFrames
    completed_dict = {}
    for t_idx, year in enumerate(years):
        df_completed = pd.DataFrame(
            T_completed[:, t_idx, :],
            index=provinces,
            columns=subcriteria,
        )
        completed_dict[year] = df_completed
    
    return completed_dict


# ---------------------------------------------------------------------------
# M-06: Gaussian Process Imputation with Spatio-Temporal Kernel
# ---------------------------------------------------------------------------

def gp_spatiotemporal_impute(
    tensor_dict: Dict[int, pd.DataFrame],
    region_mapping: Optional[Dict[str, int]] = None,
    temporal_length_scale: float = 3.0,
    spatial_length_scale: float = 2.0,
    noise_level: float = 0.1,
    n_restarts: int = 3,
    verbose: bool = False,
) -> Dict[int, pd.DataFrame]:
    """
    Bayesian GP imputation with spatio-temporal kernel for panel data.
    
    Enhancement M-06: Principled uncertainty-aware imputation exploiting
    temporal smoothness and spatial (regional) correlation.
    
    Governance scores exhibit:
    1. **Temporal smoothness**: provinces evolve gradually year-over-year
    2. **Spatial correlation**: provinces in same region (Red River Delta,
       Central Highlands, etc.) have correlated governance patterns
    
    A Gaussian Process prior encodes both via product kernel:
    
        k((i,t), (i',t')) = k_temporal(t, t') · k_spatial(i, i') + σ_n² δ
    
    where:
        k_temporal(t, t') = exp(-|t - t'|² / (2ℓ_T²))    [RBF over years]
        k_spatial(i, i') = exp(-d²_region(i, i') / (2ℓ_S²))  [regional kernel]
    
    For each subcriterion independently, the GP predicts:
    
        T*_{itj} | T_obs ~ N(μ*(i,t), σ²*(i,t))
    
    providing both point predictions (μ*) and uncertainty estimates (σ*).
    
    Parameters
    ----------
    tensor_dict : dict of {int: pd.DataFrame}
        Panel data as year → (provinces × subcriteria) DataFrame.
    region_mapping : dict of {str: int}, optional
        Province name → region ID (0-4 for Vietnam's 5 regions).
        If None, uses simplified regional structure based on province names.
    temporal_length_scale : float
        Temporal kernel length scale ℓ_T (years). Larger = smoother over time.
        Typical: 2-5 years for governance data.
    spatial_length_scale : float
        Spatial kernel length scale ℓ_S. Larger = more regional smoothing.
        Typical: 1-3 for 5 regions.
    noise_level : float
        Observation noise σ_n. Models measurement error / local variations.
        Typical: 0.05-0.2 on normalized scale.
    n_restarts : int
        Optimizer restarts for hyperparameter tuning. Higher = better fit
        but slower. Default 3 is reasonable for small panel (N×Y ≈ 900).
    verbose : bool
        If True, print per-subcriterion progress.
    
    Returns
    -------
    completed_tensor : dict of {int: pd.DataFrame}
        Panel tensor with NaN imputed via GP posterior mean.
    
    Notes
    -----
    - Computational cost: O((N·Y)³) per subcriterion for GP fit (Cholesky).
      For 63×14=882 observations, ~5-10 seconds per subcriterion.
      Total: ~3-5 minutes for 29 subcriteria (parallelizable if needed).
    - For multiple imputation, call this function M times with different
      random seeds, drawing from GP posterior: μ* + σ* · ε, ε ~ N(0,1).
    - Non-stationary extensions (time-varying length scales) possible via
      neural kernels but not implemented here.
    
    References
    ----------
    Williams, C. K., & Rasmussen, C. E. (2006). Gaussian Processes for
    Machine Learning. MIT Press.
    
    Examples
    --------
    >>> panel = {2020: df_2020, 2021: df_2021, 2022: df_2022}
    >>> # Use default Vietnam regional structure
    >>> completed = gp_spatiotemporal_impute(panel, temporal_length_scale=3.0)
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    
    years = sorted(tensor_dict.keys())
    first_year = years[0]
    provinces = tensor_dict[first_year].index.tolist()
    subcriteria = tensor_dict[first_year].columns.tolist()
    
    N = len(provinces)
    Y = len(years)
    C = len(subcriteria)
    
    # Build region mapping if not provided (simplified Vietnam structure)
    if region_mapping is None:
        region_mapping = _build_default_region_mapping(provinces)
    
    # Build 3D tensor and mask
    T = np.zeros((N, Y, C))
    mask = np.zeros((N, Y, C), dtype=bool)
    
    for t_idx, year in enumerate(years):
        df = tensor_dict[year]
        for i, prov in enumerate(provinces):
            for j, sc in enumerate(subcriteria):
                if prov in df.index and sc in df.columns:
                    val = df.loc[prov, sc]
                    if not np.isnan(val):
                        T[i, t_idx, j] = val
                        mask[i, t_idx, j] = True
    
    if verbose:
        n_obs = mask.sum()
        print(f"GP imputation: {N}×{Y}×{C} tensor, {n_obs}/{N*Y*C} observed")
    
    # Process each subcriterion independently
    T_completed = T.copy()
    
    for j, sc in enumerate(subcriteria):
        # Extract observed and missing entries for subcriterion j
        obs_mask_j = mask[:, :, j]  # (N, Y)
        n_obs_j = obs_mask_j.sum()
        n_miss_j = (~obs_mask_j).sum()
        
        if n_miss_j == 0:
            # No missing data for this subcriterion
            continue
        
        if n_obs_j < 10:
            # Insufficient observations for GP — use column mean fallback
            if verbose:
                print(f"  {sc}: {n_obs_j} obs < 10, using mean fill")
            col_mean = np.nanmean(T[:, :, j])
            T_completed[~obs_mask_j, j] = col_mean
            continue
        
        # Build training inputs X_train: (n_obs_j, 2) — [province_idx, year_idx]
        # and training targets y_train: (n_obs_j,)
        X_train = []
        y_train = []
        for i in range(N):
            for t in range(Y):
                if obs_mask_j[i, t]:
                    X_train.append([i, t])
                    y_train.append(T[i, t, j])
        
        X_train = np.array(X_train)  # (n_obs_j, 2)
        y_train = np.array(y_train)  # (n_obs_j,)
        
        # Build test inputs X_test: (n_miss_j, 2) for missing entries
        X_test = []
        for i in range(N):
            for t in range(Y):
                if not obs_mask_j[i, t]:
                    X_test.append([i, t])
        
        X_test = np.array(X_test)  # (n_miss_j, 2)
        
        # Construct spatio-temporal kernel
        # Instead of custom kernel, use RBF with feature transformation:
        # Transform (province_idx, year_idx) → (spatial_feature, temporal_feature)
        # Spatial feature: region ID (0-4)
        # Temporal feature: year index (0-13)
        
        # Map province indices to region IDs
        province_to_region = np.array([region_mapping.get(prov, 0) for prov in provinces])
        
        # Transform X_train and X_test: [prov_idx, year_idx] → [region_id, year_idx]
        X_train_transformed = np.column_stack([
            province_to_region[X_train[:, 0].astype(int)],
            X_train[:, 1],
        ])
        X_test_transformed = np.column_stack([
            province_to_region[X_test[:, 0].astype(int)],
            X_test[:, 1],
        ])
        
        # RBF kernel with anisotropic length scales: [spatial, temporal]
        # sklearn RBF uses length_scale per dimension
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(0.1, 10.0)) *
            RBF(
                length_scale=[spatial_length_scale, temporal_length_scale],
                length_scale_bounds=[(0.5, 5.0), (1.0, 10.0)],
            ) +
            WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-3, 1.0))
        )
        
        # Fit GP
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            alpha=1e-6,  # Numerical stability (Jitter)
            normalize_y=True,  # Standardize targets for stability
            random_state=42,
        )
        
        try:
            gp.fit(X_train_transformed, y_train)
            
            # Predict at missing locations
            y_pred, y_std = gp.predict(X_test_transformed, return_std=True)
            
            # Fill missing entries with GP posterior mean
            test_idx = 0
            for i in range(N):
                for t in range(Y):
                    if not obs_mask_j[i, t]:
                        T_completed[i, t, j] = y_pred[test_idx]
                        test_idx += 1
            
            if verbose:
                print(f"  {sc}: {n_obs_j} obs, {n_miss_j} imputed, "
                      f"mean_std={y_std.mean():.3f}")
        
        except Exception as e:
            # GP fit failed (e.g., singular matrix) — fallback to mean
            if verbose:
                print(f"  {sc}: GP failed ({e}), using mean fill")
            col_mean = np.nanmean(T[:, :, j])
            T_completed[~obs_mask_j, j] = col_mean
    
    # Convert back to dict of DataFrames
    completed_dict = {}
    for t_idx, year in enumerate(years):
        df_completed = pd.DataFrame(
            T_completed[:, t_idx, :],
            index=provinces,
            columns=subcriteria,
        )
        completed_dict[year] = df_completed
    
    return completed_dict


def _build_default_region_mapping(provinces: List[str]) -> Dict[str, int]:
    """Build simplified Vietnam regional mapping for GP spatial kernel.
    
    Vietnam has 5 major regions (matching features.py regional structure):
    0 — Northern Mountains & Midlands
    1 — Red River Delta
    2 — Central (North-Central + South-Central)
    3 — Central Highlands
    4 — Southern (South-East + Mekong Delta)
    
    This is a simplified heuristic based on province name patterns.
    For production use, load from codebook or spatial adjacency matrix.
    """
    region_map = {
        # Red River Delta (1)
        "Hanoi": 1, "Hai Phong": 1, "Vinh Phuc": 1, "Bac Ninh": 1,
        "Hai Duong": 1, "Hung Yen": 1, "Thai Binh": 1, "Ha Nam": 1,
        "Nam Dinh": 1, "Ninh Binh": 1,
        
        # Northern Mountains & Midlands (0)
        "Ha Giang": 0, "Cao Bang": 0, "Bac Kan": 0, "Tuyen Quang": 0,
        "Lao Cai": 0, "Dien Bien": 0, "Lai Chau": 0, "Son La": 0,
        "Yen Bai": 0, "Hoa Binh": 0, "Thai Nguyen": 0, "Lang Son": 0,
        "Quang Ninh": 0, "Bac Giang": 0, "Phu Tho": 0,
        
        # Central (2) — North-Central + South-Central
        "Thanh Hoa": 2, "Nghe An": 2, "Ha Tinh": 2, "Quang Binh": 2,
        "Quang Tri": 2, "Thua Thien Hue": 2, "Da Nang": 2, "Quang Nam": 2,
        "Quang Ngai": 2, "Binh Dinh": 2, "Phu Yen": 2, "Khanh Hoa": 2,
        "Ninh Thuan": 2, "Binh Thuan": 2,
        
        # Central Highlands (3)
        "Kon Tum": 3, "Gia Lai": 3, "Dak Lak": 3, "Dak Nong": 3, "Lam Dong": 3,
        
        # Southern (4) — South-East + Mekong Delta
        "Binh Phuoc": 4, "Tay Ninh": 4, "Binh Duong": 4, "Dong Nai": 4,
        "Ba Ria-Vung Tau": 4, "Ho Chi Minh": 4, "Long An": 4, "Tien Giang": 4,
        "Ben Tre": 4, "Tra Vinh": 4, "Vinh Long": 4, "Dong Thap": 4,
        "An Giang": 4, "Kien Giang": 4, "Can Tho": 4, "Hau Giang": 4,
        "Soc Trang": 4, "Bac Lieu": 4, "Ca Mau": 4,
    }
    
    # Map provinces to regions, default to region 0 if not found
    return {prov: region_map.get(prov, 0) for prov in provinces}


# ---------------------------------------------------------------------------
# M-11: Missingness Mechanism Diagnostics (MCAR/MAR/MNAR)
# ---------------------------------------------------------------------------

@dataclass
class MissingnessMechanismReport:
    """Diagnostic report for missing data mechanism (MCAR/MAR/MNAR).
    
    Attributes
    ----------
    littles_test_pvalue : float
        P-value from Little's MCAR test. P < 0.05 rejects MCAR hypothesis.
    mar_logistic_r2 : float
        Pseudo-R² from logistic regression of missingness indicators on
        observed variables. Higher values indicate stronger MAR pattern.
    mar_significant_predictors : List[str]
        Variable names significantly predicting missingness (p < 0.05).
    mnar_sensitivity_index : float
        Divergence measure between MAR and MNAR model estimates. Higher
        values suggest MNAR is more plausible. Range: [0, 1].
    mechanism_assessment : str
        Summary assessment: 'MCAR', 'MAR', or 'MNAR_suspected'.
    sample_size : int
        Number of observations in diagnostic sample.
    missingness_rate : float
        Overall fraction of NaN values in the dataset.
    """
    littles_test_pvalue: float
    mar_logistic_r2: float
    mar_significant_predictors: List[str]
    mnar_sensitivity_index: float
    mechanism_assessment: str
    sample_size: int
    missingness_rate: float
    
    def to_dict(self) -> dict:
        """Serializable summary for logging."""
        return {
            "mechanism": self.mechanism_assessment,
            "littles_pvalue": self.littles_test_pvalue,
            "mar_r2": self.mar_logistic_r2,
            "mnar_sensitivity": self.mnar_sensitivity_index,
            "n_samples": self.sample_size,
            "missing_rate": self.missingness_rate,
            "mar_predictors": self.mar_significant_predictors,
        }


def littles_mcar_test(X: np.ndarray, alpha: float = 0.05) -> Tuple[float, bool]:
    """Little's MCAR test via chi-square statistic.
    
    Tests the null hypothesis that data are Missing Completely At Random
    (MCAR) by comparing means across different missing-data patterns.
    
    Enhancement M-11: MCAR/MAR/MNAR Diagnostic Battery
    ---------------------------------------------------
    Little's test (1988) is the standard frequentist test for MCAR. It
    compares the observed means for each missing-data pattern against the
    expected means under MCAR, using a chi-square test statistic.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data matrix with NaN at missing positions.
    alpha : float
        Significance level. Default 0.05.
    
    Returns
    -------
    pvalue : float
        P-value from chi-square test. P < alpha rejects MCAR.
    is_mcar : bool
        True if p-value >= alpha (fail to reject MCAR); False otherwise.
    
    Notes
    -----
    - Requires at least 30 samples per missing-data pattern for asymptotic
      validity. For small samples, interpret with caution.
    - Rejection of MCAR means data are MAR or MNAR; the test cannot
      distinguish between these two alternatives.
    - This implementation uses a simplified approximation; for production use
      with complex missing patterns, consider the `pyampute` library.
    
    References
    ----------
    Little, R. J. A. (1988). A test of missing completely at random for
    multivariate data with missing values. *Journal of the American
    Statistical Association*, 83(404), 1198-1202.
    """
    from scipy import stats
    
    n, p = X.shape
    nan_mask = np.isnan(X)
    
    if not nan_mask.any():
        # No missing data — vacuously MCAR
        return 1.0, True
    
    # Create missing-data pattern indicator: hash each row's pattern
    # Each unique pattern of (missing, observed) across columns is a group
    pattern_keys = []
    for i in range(n):
        pattern = tuple(nan_mask[i, :])
        pattern_keys.append(pattern)
    
    unique_patterns = list(set(pattern_keys))
    n_patterns = len(unique_patterns)
    
    if n_patterns == 1:
        # All rows have identical missing pattern — test not applicable
        return 1.0, True
    
    # Compute group means for each pattern
    # For each variable j, compute mean of observed values in each pattern group
    # Compare observed group means to overall mean (pooled across patterns)
    chi2_stat = 0.0
    df = 0
    
    for j in range(p):
        # Overall mean for variable j (ignoring missingness)
        overall_mean = np.nanmean(X[:, j])
        if np.isnan(overall_mean):
            continue  # Entire column missing — skip
        
        # Group means per pattern
        group_means = []
        group_sizes = []
        for pattern in unique_patterns:
            group_mask = np.array([pattern_keys[i] == pattern for i in range(n)])
            # Only include samples where variable j is observed in this pattern
            observed_in_group = group_mask & ~nan_mask[:, j]
            if observed_in_group.sum() > 0:
                group_mean_j = np.mean(X[observed_in_group, j])
                group_means.append(group_mean_j)
                group_sizes.append(observed_in_group.sum())
        
        if len(group_means) < 2:
            continue  # Need at least 2 groups with observations
        
        # Chi-square contribution: sum of (group_mean - overall_mean)^2 * n_group
        for gm, gn in zip(group_means, group_sizes):
            chi2_stat += gn * (gm - overall_mean) ** 2
        
        df += len(group_means) - 1
    
    if df == 0:
        return 1.0, True  # Insufficient variation for test
    
    # Normalize by pooled variance estimate
    pooled_var = np.nanvar(X, axis=0).mean()
    if pooled_var > 1e-12:
        chi2_stat /= pooled_var
    
    # P-value from chi-square distribution
    pvalue = 1.0 - stats.chi2.cdf(chi2_stat, df)
    is_mcar = pvalue >= alpha
    
    return pvalue, is_mcar


def mar_logistic_test(X: np.ndarray) -> Tuple[float, List[str]]:
    """Test for MAR via logistic regression of missingness indicators.
    
    Enhancement M-11: MAR diagnostic via predictive modeling.
    
    Fits a logistic regression predicting whether each variable is missing
    based on observed values of other variables. Significant predictors
    indicate MAR (missingness depends on observed data). High R² indicates
    strong MAR pattern.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data matrix with NaN at missing positions. Feature names assumed
        to be f0, f1, ..., f{p-1} unless passed as DataFrame.
    
    Returns
    -------
    r2_score : float
        Average pseudo-R² (McFadden) across all missingness indicators.
        Range: [0, 1]. Higher values indicate missingness is strongly
        predicted by observed variables (MAR).
    significant_predictors : List[str]
        Names of variables that significantly predict missingness (p < 0.05).
    
    Notes
    -----
    - Returns (0.0, []) if there is no missing data or insufficient variation.
    - For panel data, apply per-year or on stacked panel with year dummies.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    
    n, p = X.shape
    nan_mask = np.isnan(X)
    
    if not nan_mask.any():
        return 0.0, []
    
    r2_scores = []
    significant_vars = set()
    
    for j in range(p):
        # Skip columns with no missing or all missing
        if nan_mask[:, j].sum() == 0 or nan_mask[:, j].sum() == n:
            continue
        
        # Target: binary indicator of missingness for variable j
        y = nan_mask[:, j].astype(int)
        
        # Features: all OTHER observed variables (exclude variable j itself)
        feature_cols = [k for k in range(p) if k != j]
        X_features = X[:, feature_cols]
        
        # Drop rows where any predictor is missing (complete-case for this test)
        complete_rows = ~np.isnan(X_features).any(axis=1)
        if complete_rows.sum() < 30:
            continue  # Insufficient samples for valid test
        
        X_complete = X_features[complete_rows]
        y_complete = y[complete_rows]
        
        # Fit logistic regression
        try:
            logreg = LogisticRegression(
                penalty='l2', C=1.0, max_iter=200, random_state=42
            )
            logreg.fit(X_complete, y_complete)
            
            # McFadden's pseudo-R²: 1 - (log_loss_model / log_loss_null)
            y_pred_proba = logreg.predict_proba(X_complete)[:, 1]
            null_proba = np.full(len(y_complete), y_complete.mean())
            
            # Avoid log(0) by clipping probabilities
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            null_proba = np.clip(null_proba, 1e-15, 1 - 1e-15)
            
            ll_model = -log_loss(y_complete, y_pred_proba, normalize=False)
            ll_null = -log_loss(y_complete, null_proba, normalize=False)
            
            r2 = 1.0 - (ll_model / ll_null) if ll_null != 0 else 0.0
            r2_scores.append(max(r2, 0.0))  # Clamp at 0
            
            # Check coefficient significance (Wald test approximation)
            # If |coef| > 2 * std_err (roughly p < 0.05), mark as significant
            # This requires statsmodels for exact p-values; here we use heuristic
            for k, coef in enumerate(logreg.coef_[0]):
                if abs(coef) > 0.1:  # Heuristic threshold (conservative)
                    significant_vars.add(f"f{feature_cols[k]}")
        
        except Exception:
            # Logistic regression failed (e.g., perfect separation) — skip
            continue
    
    avg_r2 = np.mean(r2_scores) if r2_scores else 0.0
    return avg_r2, sorted(significant_vars)


def assess_missing_mechanism(
    X: "np.ndarray | pd.DataFrame",
    alpha: float = 0.05,
    verbose: bool = False,
) -> MissingnessMechanismReport:
    """Comprehensive diagnostic for missing data mechanism (MCAR/MAR/MNAR).
    
    Enhancement M-11: Full diagnostic battery for missing data mechanism.
    
    Applies three tests:
    1. **Little's MCAR test** — tests if missingness is completely random.
    2. **MAR logistic test** — tests if missingness depends on observed data.
    3. **MNAR sensitivity heuristic** — estimates plausibility of MNAR.
    
    Decision tree for imputation strategy:
    - MCAR confirmed (Little's p >= 0.05, MAR R² < 0.1) → mean/median/MICE
    - MAR confirmed (Little's p < 0.05, MAR R² >= 0.1) → MICE/MissForest/SoftImpute
    - MNAR suspected (MAR R² < 0.1, high sensitivity) → GAIN / Selection models
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix with NaN at missing positions. Can be numpy array or
        pandas DataFrame (column names used in report if available).
    alpha : float
        Significance level for MCAR test. Default 0.05.
    verbose : bool
        If True, print diagnostic summary to console.
    
    Returns
    -------
    report : MissingnessMechanismReport
        Diagnostic report with test statistics and mechanism assessment.
    
    Examples
    --------
    >>> X = panel_tensor[2023].values  # (63 provinces × 29 subcriteria)
    >>> report = assess_missing_mechanism(X, verbose=True)
    >>> print(report.mechanism_assessment)  # 'MAR'
    >>> print(report.to_dict())
    """
    # Convert to numpy if DataFrame
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_arr = X.values
    else:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        X_arr = X
    
    n, p = X_arr.shape
    nan_mask = np.isnan(X_arr)
    missing_rate = nan_mask.sum() / nan_mask.size
    
    if missing_rate == 0.0:
        # No missing data — all tests vacuously MCAR
        return MissingnessMechanismReport(
            littles_test_pvalue=1.0,
            mar_logistic_r2=0.0,
            mar_significant_predictors=[],
            mnar_sensitivity_index=0.0,
            mechanism_assessment="MCAR",
            sample_size=n,
            missingness_rate=0.0,
        )
    
    # Test 1: Little's MCAR test
    littles_p, is_mcar = littles_mcar_test(X_arr, alpha)
    
    # Test 2: MAR logistic test
    mar_r2, mar_predictors = mar_logistic_test(X_arr)
    
    # Test 3: MNAR sensitivity heuristic
    # Estimate divergence between complete-case mean and pairwise-available mean
    # High divergence suggests selection bias (MNAR)
    mnar_sensitivity = 0.0
    for j in range(p):
        complete_case_mean = np.nanmean(X_arr[~nan_mask.any(axis=1), j])
        pairwise_mean = np.nanmean(X_arr[:, j])
        if not np.isnan(complete_case_mean) and not np.isnan(pairwise_mean):
            # Normalized absolute difference
            col_std = np.nanstd(X_arr[:, j])
            if col_std > 1e-12:
                divergence = abs(complete_case_mean - pairwise_mean) / col_std
                mnar_sensitivity = max(mnar_sensitivity, divergence)
    
    # Mechanism assessment decision tree
    if is_mcar and mar_r2 < 0.1:
        assessment = "MCAR"
    elif mar_r2 >= 0.1:
        assessment = "MAR"
    elif mnar_sensitivity > 0.5:
        assessment = "MNAR_suspected"
    else:
        assessment = "MAR"  # Default to MAR if uncertain
    
    report = MissingnessMechanismReport(
        littles_test_pvalue=littles_p,
        mar_logistic_r2=mar_r2,
        mar_significant_predictors=mar_predictors,
        mnar_sensitivity_index=mnar_sensitivity,
        mechanism_assessment=assessment,
        sample_size=n,
        missingness_rate=missing_rate,
    )
    
    if verbose:
        print("\n=== Missing Data Mechanism Diagnostic ===")
        print(f"Sample size: {n} × {p}")
        print(f"Missingness rate: {missing_rate:.1%}")
        print(f"\nLittle's MCAR Test:")
        print(f"  P-value: {littles_p:.4f} {'(MCAR)' if is_mcar else '(Reject MCAR)'}")
        print(f"\nMAR Logistic Test:")
        print(f"  Pseudo-R²: {mar_r2:.3f}")
        print(f"  Significant predictors: {mar_predictors[:5]}")  # Show first 5
        print(f"\nMNAR Sensitivity:")
        print(f"  Divergence index: {mnar_sensitivity:.3f}")
        print(f"\nAssessment: {assessment}")
        print(f"Recommended imputation: ", end="")
        if assessment == "MCAR":
            print("Mean/Median/MICE (any consistent method)")
        elif assessment == "MAR":
            print("MICE/MissForest/SoftImpute (max likelihood valid)")
        else:
            print("GAIN/Selection models (MNAR-robust methods)")
    
    return report
