# -*- coding: utf-8 -*-
"""
🚫 DEPRECATED: Panel Sequential MICE Imputation (E-05)
======================================================

**Deprecated as of 2026-03-27**

This multi-phase imputation approach (temporal → spatial → global) has been
superseded by the unified MICE imputation strategy using ExtraTreesRegressor.

REASON FOR DEPRECATION:
- Over-engineered: Three phases achieve the same result as single MICE pass
- Complex: Hard to debug and validate
- Unified MICE with n_nearest_features=30 captures both temporal and spatial
  correlations automatically through multivariate learning
- Production simplified: Single configuration, single validation suite

MIGRATION:
Use unified MICE imputation instead:

    from data.imputation import MICEImputer, ImputationConfig

    config = ImputationConfig(
        mice_estimator='extra_trees',
        mice_max_iter=40,
        mice_n_nearest_features=30,  # Captures temporal + spatial
    )
    imputer = MICEImputer(config)
    X_imputed = imputer.fit_transform(X_train)

REMOVAL SCHEDULE:
- Current (active): 2026-03 to 2026-06 (3-month deprecation period)
- Removal: 2026-06-27

See data.imputation.iterative.MICEImputer for details.
"""

import warnings
warnings.warn(
    "forecasting.panel_mice is DEPRECATED as of 2026-03-27. "
    "Use data.imputation.MICEImputer instead. "
    "Will be removed 2026-06-27.",
    DeprecationWarning,
    stacklevel=2,
)

# Original implementation below (for reference, do not use)
# ============================================================================


Phase 1 — Temporal (within-entity)
    For each entity, interpolate NaN values across the time dimension using
    linear interpolation with nearest-neighbour boundary extension.  Captures
    entity-specific temporal autocorrelation (the strongest signal in panels).

Phase 2 — Spatial (cross-sectional, per year)
    For each calendar year, apply K-Nearest-Neighbour imputation across
    entities using the complete features of neighbouring entities at the
    same time point.  Captures cross-entity spatial correlation.

Phase 3 — Global fallback
    Any residual NaN after phases 1–2 is handled by IterativeImputer with
    HistGradientBoostingRegressor (handles NaN natively, sample_posterior=True
    for stochastic draws needed by Rubin's Rules downstream).

Design notes
------------
* Operates on the engineered feature matrix ``(X, entity_indices, year_labels)``
  after ``TemporalFeatureEngineer.fit_transform()`` but BEFORE
  ``PanelFeatureReducer`` so the imputed values flow into dimensionality
  reduction correctly.
* ``transform()`` uses Phase 2 KNN (same-year entities' feature values are
  available at prediction time for historical years) and Phase 3 global
  imputer's ``transform()`` to handle prediction-year NaN.  For truly
  novel prediction years, Phase 1 temporal extrapolation is applied first.

References
----------
van Buuren, S. (2018). Flexible Imputation of Missing Data, Chapter 9.3.
    CRC Press.  https://stefvanbuuren.name/fimd/
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import HistGradientBoostingRegressor


class PanelSequentialMICE:
    """Three-phase panel-structured MICE imputation.

    Parameters
    ----------
    n_temporal_passes : int
        Repeat the temporal interpolation pass ``n_temporal_passes`` times
        (default 1).  More passes help when a feature has NaN in multiple
        consecutive years.
    knn_neighbors : int
        Number of spatial neighbours for Phase 2 KNNImputer (default 5).
    global_max_iter : int
        IterativeImputer max iterations for Phase 3 (default 20).
    sample_posterior : bool
        Stochastic Phase 3 (required for Rubin's rules, default True).
    add_indicator : bool
        Add missingness indicator columns in Phase 3 (default True).
    random_state : int
    verbose : bool

    Attributes
    ----------
    _global_imputer_ : IterativeImputer
        Fitted Phase 3 imputer (persisted for transform()).
    _knn_imputers_ : dict[int, KNNImputer]
        Per-year Phase 2 imputers (keyed by integer year).
    nan_before_ : int
        NaN count in input X at last fit_transform() call.
    nan_after_ : int
        NaN count in output after all three phases.
    """

    def __init__(
        self,
        n_temporal_passes: int = 1,
        knn_neighbors:      int = 5,
        global_max_iter:    int = 20,
        sample_posterior:   bool = True,
        add_indicator:      bool = True,
        random_state:       int = 42,
        verbose:            bool = False,
    ):
        self.n_temporal_passes = n_temporal_passes
        self.knn_neighbors     = knn_neighbors
        self.global_max_iter   = global_max_iter
        self.sample_posterior  = sample_posterior
        self.add_indicator     = add_indicator
        self.random_state      = random_state
        self.verbose           = verbose

        self._global_imputer_: Optional[IterativeImputer] = None
        self._knn_imputers_:   dict = {}
        self._n_features_in_:  Optional[int] = None
        self.nan_before_: int = 0
        self.nan_after_:  int = 0

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X: np.ndarray,
        entity_indices: np.ndarray,
        year_labels:    np.ndarray,
    ) -> np.ndarray:
        """Fit all three imputation phases on training data and return
        imputed X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix.  May contain NaN.
        entity_indices : ndarray, shape (n_samples,)
            Integer entity (province) IDs.
        year_labels : ndarray, shape (n_samples,)
            Integer calendar year per observation.

        Returns
        -------
        X_imputed : ndarray, shape (n_samples, n_features)
            NaN count reduced or eliminated.
        """
        self._n_features_in_ = X.shape[1]
        self.nan_before_ = int(np.isnan(X).sum())

        if self.nan_before_ == 0:
            # Nothing to impute — fit global imputer on complete data
            # (still needed for transform())
            self._global_imputer_ = self._make_global_imputer()
            self._global_imputer_.fit(X)
            self.nan_after_ = 0
            return X.copy()

        X_out = X.copy()

        # Phase 1: temporal interpolation per entity
        X_out = self._temporal_phase(X_out, entity_indices, year_labels)

        # Phase 2: spatial KNN imputation per year
        X_out = self._spatial_phase_fit(X_out, entity_indices, year_labels)

        # Phase 3: global IterativeImputer for residual NaN
        self._global_imputer_ = self._make_global_imputer()
        n_residual = int(np.isnan(X_out).sum())
        if n_residual > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_out = self._global_imputer_.fit_transform(X_out)
        else:
            self._global_imputer_.fit(X_out)

        self.nan_after_ = int(np.isnan(X_out).sum())

        if self.verbose:
            print(f"  PanelMICE: {self.nan_before_} NaN → {self.nan_after_} "
                  f"after 3-phase imputation.")
        return X_out

    def transform(
        self,
        X: np.ndarray,
        entity_indices: np.ndarray,
        year_labels: np.ndarray,
    ) -> np.ndarray:
        """Apply fitted imputation to new data (e.g., holdout or prediction
        features).

        Phase 1 (temporal): uses within-X linear interpolation / extrapolation
        on the provided rows.  For a single future year row, only boundary
        extension (nearest value) applies.

        Phase 2 (spatial): uses per-year KNN imputers fitted during
        ``fit_transform()``.  For unseen years, skips Phase 2.

        Phase 3 (global): applies the fitted ``IterativeImputer.transform()``.

        Parameters
        ----------
        X, entity_indices, year_labels : same conventions as fit_transform()

        Returns
        -------
        X_imputed : ndarray, shape (n_samples, n_features)
        """
        if self._global_imputer_ is None:
            raise RuntimeError("Call fit_transform() before transform().")
        if not np.isnan(X).any():
            return X.copy()

        X_out = X.copy()

        # Phase 1: temporal
        X_out = self._temporal_phase(X_out, entity_indices, year_labels)

        # Phase 2: spatial (only for years seen during fit)
        X_out = self._spatial_phase_transform(X_out, entity_indices, year_labels)

        # Phase 3: global
        n_residual = int(np.isnan(X_out).sum())
        if n_residual > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_out = self._global_imputer_.transform(X_out)

        return X_out

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _temporal_phase(
        self,
        X: np.ndarray,
        entity_indices: np.ndarray,
        year_labels:    np.ndarray,
    ) -> np.ndarray:
        """Phase 1: per-entity linear time interpolation.

        For each entity, each feature column is treated as a time series
        ordered by ``year_labels``.  NaN values are filled via:
        1. pandas``interpolate(method='linear', limit_direction='both')``
           which handles interior gaps and nearest-neighbour boundary extension.

        The operation is performed ``n_temporal_passes`` times so multiple
        consecutive NaN years receive values from both adjacent non-NaN years.
        """
        X_out = X.copy()
        unique_entities = np.unique(entity_indices)

        for _ in range(self.n_temporal_passes):
            for ent in unique_entities:
                mask = entity_indices == ent
                if not mask.any():
                    continue
                row_idx = np.where(mask)[0]
                yrs     = year_labels[row_idx]
                order   = np.argsort(yrs)
                sorted_rows = row_idx[order]

                X_ent = X_out[sorted_rows, :]
                df_ent = pd.DataFrame(X_ent)
                df_ent = df_ent.interpolate(
                    method='linear',
                    limit_direction='both',
                    axis=0,
                )
                # For boundary NaN that interpolation can't fill (all-NaN
                # column for this entity), forward-fill then back-fill.
                df_ent = df_ent.ffill().bfill()
                X_out[sorted_rows, :] = df_ent.values

        return X_out

    def _spatial_phase_fit(
        self,
        X: np.ndarray,
        entity_indices: np.ndarray,
        year_labels:    np.ndarray,
    ) -> np.ndarray:
        """Phase 2 fit: per-year KNNImputer trained on training observations.

        For each unique year, fit a KNNImputer on the cross-section
        (all entities in that year), then transform to fill remaining NaN.
        The fitted imputers are stored in ``_knn_imputers_`` keyed by year.
        """
        X_out = X.copy()
        unique_years = np.unique(year_labels)
        k = min(self.knn_neighbors, max(2, len(np.unique(entity_indices)) - 1))

        for yr in unique_years:
            yr_mask = year_labels == yr
            if not yr_mask.any():
                continue
            yr_rows = np.where(yr_mask)[0]
            X_yr = X_out[yr_rows, :]

            if not np.isnan(X_yr).any():
                # No missing in this year; fit imputer anyway for transform()
                knn = KNNImputer(n_neighbors=k, keep_empty_features=True)
                knn.fit(X_yr)
                self._knn_imputers_[int(yr)] = knn
                continue

            knn = KNNImputer(n_neighbors=k, keep_empty_features=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_yr_imp = knn.fit_transform(X_yr)
            self._knn_imputers_[int(yr)] = knn
            X_out[yr_rows, :] = X_yr_imp

        return X_out

    def _spatial_phase_transform(
        self,
        X: np.ndarray,
        entity_indices: np.ndarray,
        year_labels:    np.ndarray,
    ) -> np.ndarray:
        """Phase 2 transform: apply per-year KNN imputers fitted during fit()."""
        X_out = X.copy()
        for yr in np.unique(year_labels):
            yr_int = int(yr)
            yr_mask = year_labels == yr
            yr_rows = np.where(yr_mask)[0]
            X_yr = X_out[yr_rows, :]

            if not np.isnan(X_yr).any():
                continue

            if yr_int not in self._knn_imputers_:
                # Unseen year: use the nearest fitted year's imputer
                known_yrs = sorted(self._knn_imputers_.keys())
                if not known_yrs:
                    continue
                nearest = min(known_yrs, key=lambda y: abs(y - yr_int))
                knn = self._knn_imputers_[nearest]
            else:
                knn = self._knn_imputers_[yr_int]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_out[yr_rows, :] = knn.transform(X_yr)

        return X_out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_global_imputer(self) -> IterativeImputer:
        """Create Phase 3 IterativeImputer with HistGradientBoosting."""
        return IterativeImputer(
            estimator=HistGradientBoostingRegressor(
                max_iter=150,              # increased from 100 for stability
                max_leaf_nodes=31,
                random_state=self.random_state,
            ),
            max_iter=self.global_max_iter,
            tol=5e-3,                     # relaxed from 1e-3 for achievability
            initial_strategy='median',
            sample_posterior=self.sample_posterior,
            add_indicator=self.add_indicator,
            keep_empty_features=True,
            random_state=self.random_state,
        )

    @property
    def nan_reduction_pct(self) -> Optional[float]:
        """Percentage of NaN eliminated (0–100)."""
        if self.nan_before_ == 0:
            return 100.0
        return 100.0 * (self.nan_before_ - self.nan_after_) / self.nan_before_
