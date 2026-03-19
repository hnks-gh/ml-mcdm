# -*- coding: utf-8 -*-
"""
Temporal Feature Engineering
============================

Advanced feature engineering for time series panel data, creating
rich feature sets for ML forecasting models.

Features include:
- Lag features (historical values at t-1, t-2, t-3)
- Rolling statistics (mean, std, min, max over 2–3 year windows)
- Momentum and acceleration (first/second differences)
- Trend features (linear slope via polyfit)
- Cross-entity features (percentile rank, z-score relative to panel)

Stationarity features (Phase 2)
---------------------------------
Most criteria panel series exhibit unit-root / trending behaviour.
Three blocks of stationary transformations remove this non-stationarity:

* **First-differences** ``_delta1, _delta2`` — current and lagged annual
  changes.  Together they allow ML models to learn AR dynamics in
  difference space (analogous to ARIMA differencing).

* **Entity-demeaned levels** ``_demeaned`` — subtract each entity's
  historical mean (the "within" / fixed-effect transformation), making
  feature values comparable across provinces with different baselines.

* **Entity-demeaned momentum** ``_demeaned_momentum`` — subtract each
  entity's long-run average first-difference, isolating whether a
  province is currently changing faster or slower than its own trend.

Dynamic exclusion via ``year_contexts``
---------------------------------------
When ``panel_data.year_contexts`` is present, ``fit_transform`` removes:

* **Training rows** whose *target* year excludes the entity (province absent
  from ``year_contexts[t+1].active_provinces``) or whose target sub-criterion
  is NaN for that year.  No imputation is performed on target values.

* **Prediction rows** limited to entities in the *forecast year*'s
  ``active_provinces`` set; excluded entities are simply absent from output.

NaN values in *feature* (input) vectors — e.g. missing lag values for
entities with short histories — are filled with the cross-sectional median
for the missing year, with a binary ``_was_missing`` indicator appended so
the model can learn to discount imputed inputs (Fix D-01).  Any remaining
NaN values (non-lag features with insufficient history) are filled with
``0.0`` ("no prior information") by the safety wrapper.

Phase 1 Enhancement Summary
----------------------------
* **D-01 Fix** — Lag NaN → cross-sectional median + ``_was_missing`` flags
* **D-02 Fix** — ``_delta1`` removed (duplicated ``_momentum``); ``_delta2`` retained
* **D-03 Fix** — Cross-entity cross-sections filtered to ``active_provinces``
* **G-01** — EWMA levels (spans 2, 3, 5) as recency-weighted baselines
* **G-02** — Rolling window=5 added to existing windows {2, 3}
* **G-03** — Expanding window mean (unconditional historical baseline)
* **G-04** — Inter-criterion diversity (std and range across components)
* **G-05** — Rank-change: Δpercentile = pct_t − pct_{t−1}
* **G-06** — Regional cluster dummies (5 Vietnam geographic regions)
* **G-07** — Rolling skewness (5-year window, min 3 valid points)
* **G-08** — Polyfit trend requires ≥ 3 valid points (was ≥ 2)
"""

import numpy as np
import pandas as pd
import logging
from scipy import stats as _scipy_stats
from typing import Dict, List, Optional, Tuple

from data.missing_data import fill_missing_features, has_complete_target
from data.imputation import ImputationConfig, ImputationAudit

logger = logging.getLogger('ml_mcdm')

# ---------------------------------------------------------------------------
# Vietnam province → geographic region mapping (5 regions, 0-indexed)
#
# Region IDs:
#   0 — Northern Mountains & Midlands (Ha Giang, Cao Bang, … Phu Tho)
#   1 — Red River Delta (Hanoi, Vinh Phuc, … Ninh Binh)
#   2 — Central (Thanh Hoa … Binh Thuan — North-Central + South-Central)
#   3 — Central Highlands (Kon Tum, Gia Lai, Dak Lak, Dak Nong, Lam Dong)
#   4 — Southern (Binh Phuoc … Ca Mau — South-East + Mekong Delta)
#
# Source: Vietnam administrative geography; matches province ordering in
# ``data/codebook/codebook_provinces.csv`` (P01–P63).
# ---------------------------------------------------------------------------
_VIETNAM_PROVINCE_REGIONS: Dict[str, int] = {
    # Red River Delta (1)
    "Hanoi": 1,
    # Northern Mountains & Midlands (0)
    "Ha Giang": 0, "Cao Bang": 0, "Bac Kan": 0, "Tuyen Quang": 0,
    "Lao Cai": 0, "Dien Bien": 0, "Lai Chau": 0, "Son La": 0,
    "Yen Bai": 0, "Hoa Binh": 0, "Thai Nguyen": 0, "Lang Son": 0,
    "Quang Ninh": 0, "Bac Giang": 0, "Phu Tho": 0,
    # Red River Delta (1)
    "Vinh Phuc": 1, "Bac Ninh": 1, "Hai Duong": 1, "Hai Phong": 1,
    "Hung Yen": 1, "Thai Binh": 1, "Ha Nam": 1, "Nam Dinh": 1, "Ninh Binh": 1,
    # Central (2)
    "Thanh Hoa": 2, "Nghe An": 2, "Ha Tinh": 2, "Quang Binh": 2,
    "Quang Tri": 2, "Thua Thien Hue": 2, "Da Nang": 2, "Quang Nam": 2,
    "Quang Ngai": 2, "Binh Dinh": 2, "Phu Yen": 2, "Khanh Hoa": 2,
    "Ninh Thuan": 2, "Binh Thuan": 2,
    # Central Highlands (3)
    "Kon Tum": 3, "Gia Lai": 3, "Dak Lak": 3, "Dak Nong": 3, "Lam Dong": 3,
    # Southern (4)
    "Binh Phuoc": 4, "Tay Ninh": 4, "Binh Duong": 4, "Dong Nai": 4,
    "Ba Ria - Vung Tau": 4, "Ho Chi Minh City": 4, "Long An": 4,
    "Tien Giang": 4, "Ben Tre": 4, "Tra Vinh": 4, "Vinh Long": 4,
    "Dong Thap": 4, "An Giang": 4, "Kien Giang": 4, "Can Tho": 4,
    "Hau Giang": 4, "Soc Trang": 4, "Bac Lieu": 4, "Ca Mau": 4,
}
_N_REGIONS: int = 5


class SAWNormalizer:
    """
    Per-year, column-wise minmax normalization of criteria cross-sections.

    Applies the SAW method's minmax normalization (benefit criteria) to each
    year's decision matrix (entities × criteria), producing bounded [0, 1]
    values that:

    * Remove cross-year level differences — each year is rescaled
      independently so predictions from different years are on a common scale.
    * Preserve within-year ordinal structure — relative rankings of
      provinces on each criterion are unchanged.
    * Are free of CRITIC weighting bias — raw criterion composites reflect
      the year-specific CRITIC weights used during data preparation; SAW
      normalization removes this confound.

    This replicates ``SAWCalculator._normalize(normalization='minmax')`` for
    *benefit* criteria only (no cost-criterion inversion for C01–C08).

    Notes
    -----
    Degenerate columns (all values identical or all NaN):
        Mapped to 0.0; no exception is raised.

    NaN preservation:
        A province whose raw value is NaN receives NaN in the output.
        Callers decide on exclusion or imputation.  During training, the
        ``TemporalFeatureEngineer`` applies the same exclusion/imputation
        rules used in non-SAW mode.

    Numerical stability:
        ``np.errstate(invalid='ignore', divide='ignore')`` suppresses the
        0/0 warning for degenerate columns; results are then forced to 0.0.
        Outputs are clipped to [0, 1] to guard against float precision edge
        cases (e.g. a value that is marginally outside the observed range due
        to rounding).
    """

    def normalize_cross_section(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize a (entities × criteria) cross-section via column-wise minmax.

        Parameters
        ----------
        df : pd.DataFrame
            Raw criteria composite values, shape ``(n_entities, n_criteria)``.
            Index = province/entity names; columns = criterion identifiers.

        Returns
        -------
        pd.DataFrame
            Minmax-normalized values in [0, 1], same index and columns.
            NaN cells remain NaN.  Degenerate columns (min == max) → 0.0.

        Examples
        --------
        >>> import pandas as pd
        >>> norm = SAWNormalizer()
        >>> df = pd.DataFrame({'C1': [1.0, 2.0, 3.0], 'C2': [5.0, 5.0, 5.0]})
        >>> norm.normalize_cross_section(df)
             C1   C2
        0  0.0  0.0
        1  0.5  0.0
        2  1.0  0.0
        """
        X = df.values.astype(float)
        nan_mask = np.isnan(X)

        min_v = np.nanmin(X, axis=0)   # shape (n_criteria,)
        max_v = np.nanmax(X, axis=0)
        rng = max_v - min_v            # 0 for degenerate / all-NaN columns

        # Vectorised minmax; degenerate columns yield 0/0 → set to 0.0
        with np.errstate(invalid='ignore', divide='ignore'):
            norm = np.where(rng > 0, (X - min_v) / rng, 0.0)

        # Clip to [0, 1] to guard float precision edge cases
        norm = np.clip(norm, 0.0, 1.0)

        # Restore NaN cells that were present in the original data
        norm[nan_mask] = np.nan

        return pd.DataFrame(norm, index=df.index, columns=df.columns)

    def normalize_year(self, panel_data, year: int) -> pd.DataFrame:
        """
        Retrieve and normalize the criteria cross-section for a given year.

        Parameters
        ----------
        panel_data : PanelData
            Panel data object with a ``criteria_cross_section`` attribute
            (dict keyed by year → province-indexed DataFrame with C01..C08).
        year : int
            Calendar year to normalize.

        Returns
        -------
        pd.DataFrame
            Normalized decision matrix (entities × criteria) in [0, 1].
            Index = province names, columns = criteria names (C01..C08).

        Raises
        ------
        KeyError
            If ``year`` is not present in ``panel_data.criteria_cross_section``.
        """
        cs = panel_data.criteria_cross_section.get(year)
        if cs is None:
            available = sorted(panel_data.criteria_cross_section)
            raise KeyError(
                f"SAWNormalizer: no criteria_cross_section for year {year}. "
                f"Available years: {available}"
            )
        # Ensure province names are the index (some callers may leave Province
        # as a column rather than the index).
        if 'Province' in cs.columns:
            cs = cs.set_index('Province')
        return self.normalize_cross_section(cs)


class TemporalFeatureEngineer:
    """
    Advanced feature engineering for time series panel data.

    Creates a rich feature set spanning 12 structural blocks:

    Block 1  — Current values (raw component levels at t)
    Block 2  — Lag features (t-1, t-2, t-3) + binary missingness indicators
    Block 3  — Rolling statistics (mean, std, min, max) for windows {2, 3, 5}
    Block 4  — Momentum and acceleration (first/second-order changes)
    Block 5  — Stationarity: entity-demeaned level, entity-demeaned momentum,
                lagged first-difference delta2 (delta1 removed — Fix D-02)
    Block 6  — Trend (polyfit slope, min 3 valid points — Fix G-08)
    Block 7  — EWMA levels (spans 2, 3, 5) — recency-weighted baselines
    Block 8  — Expanding window mean — unconditional long-run baseline
    Block 9  — Inter-component diversity (std and range across components)
    Block 10 — Rolling skewness (5-year window) — breakout vs. regression
    Block 11 — Panel-relative: percentile, z-score (Fix D-03), rank-change
    Block 12 — Regional cluster dummies (5 geographic regions of Vietnam)

    Parameters
    ----------
    lag_periods : list of int
        Lag periods to include (default: [1, 2, 3]).
    rolling_windows : list of int
        Window sizes for rolling statistics (default: [2, 3, 5]).
    include_momentum : bool
        Include rate-of-change (momentum + acceleration) features.
    include_cross_entity : bool
        Include percentile rank and z-score relative to the panel.
    include_ewma : bool
        Include EWMA level features (spans 2, 3, 5).  Default True.
    include_expanding : bool
        Include expanding-window (full historical) mean.  Default True.
    include_diversity : bool
        Include inter-component diversity features (std, range).  Default True.
    include_rank_change : bool
        Include Δpercentile rank-change feature.  Default True.
    include_region_dummies : bool
        Include 5 one-hot regional dummies for Vietnam's geographic regions.
    include_rolling_skewness : bool
        Include 5-year rolling skewness per component.  Default True.
    target_level : str
        ``'criteria'`` (default) or ``'subcriteria'``.

    Example
    -------
    >>> engineer = TemporalFeatureEngineer(lag_periods=[1, 2, 3])
    >>> X_train, y_train, X_pred, _, _, _ = engineer.fit_transform(panel_data, 2025)
    """

    def __init__(self,
                 lag_periods: List[int] = [1, 2, 3],
                 rolling_windows: List[int] = [3, 5],
                 include_momentum: bool = True,
                 include_cross_entity: bool = True,
                 include_ewma: bool = True,
                 include_expanding: bool = False,
                 include_diversity: bool = True,
                 include_rank_change: bool = True,
                 include_region_dummies: bool = True,
                 include_rolling_skewness: bool = False,
                 target_level: str = "criteria"):
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows
        self.include_momentum = include_momentum
        self.include_cross_entity = include_cross_entity
        self.include_ewma = include_ewma
        self.include_expanding = include_expanding
        self.include_diversity = include_diversity
        self.include_rank_change = include_rank_change
        self.include_region_dummies = include_region_dummies
        self.include_rolling_skewness = include_rolling_skewness
        self.target_level = target_level
        self.feature_names_: List[str] = []

        # Populated by fit_transform() — entity-level historical statistics
        # used to remove entity fixed effects from features (Phase 2).
        #
        # _entity_means_[(entity, component)] = mean of entity's raw values
        #     over all training feature years (Panel "within" mean).
        # _entity_mean_deltas_[(entity, component)] = mean of consecutive
        #     first-differences (Δ) over training feature years.  Represents
        #     the entity's typical annual rate of change; subtracting it from
        #     the current first-difference isolates the entity-specific
        #     velocity deviation from its long-run trend.
        #
        # Both default to 0.0 for unseen entities (safe fallback: demeaning by
        # zero is a no-op, which is preferable to raising KeyError at runtime).
        self._entity_means_: Dict = {}
        self._entity_mean_deltas_: Dict = {}

        # Per-(year, component) cross-sectional median over active provinces.
        # Populated by fit_transform(); used to fill missing lag slots with a
        # meaningful centre-of-distribution value rather than 0.0 (Fix D-01).
        self._component_year_medians_: Dict = {}
        
        # ===== PHASE A Enhancement: Tiered Imputation Caches =====
        # Tier 2: Per-entity-year medians (temporal blocks: 3, 4, 7, 8, 10)
        # Key: (entity, block_id, component) -> median_value
        self._temporal_medians_: Dict[Tuple[str, int, str], float] = {}
        
        # Tier 3: Cross-sectional medians (sectional blocks: 1, 5, 6, 9, 11)
        # Key: (year, block_id, component) -> median_value
        self._crosssectional_medians_: Dict[Tuple[int, int, str], float] = {}
        
        # Tier 4: Training means per block (fallback cache)
        # Key: (block_id, component) -> mean_value
        self._block_training_means_: Dict[Tuple[int, str], float] = {}
        
        # Imputation configuration and audit trail
        self._imputation_config: Optional['ImputationConfig'] = None
        self._imputation_audit: 'ImputationAudit' = None
    
    def fit_transform(self,
                      panel_data,
                      target_year: int,
                      use_saw_normalization: bool = False,
                      holdout_year: Optional[int] = None,
                      imputation_config: Optional[ImputationConfig] = None,
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                 pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create feature matrix for training and prediction with dynamic exclusion.

        **Missing-data rules:**

        * A training sample ``(entity, year_t → year_{t+1})`` is **excluded**
          if the entity is absent from ``year_{t+1}``'s active province set
          (per ``panel_data.year_contexts``) OR if *any* target sub-criterion
          for ``year_{t+1}`` is NaN.  No imputation is performed.

        * For prediction, only entities in the *target year*'s active province
          set are included.  Excluded provinces are simply absent from the
          prediction output.

        * NaN values in *feature* vectors (e.g. missing lag values because a
          prior year had no data) are replaced with ``0.0`` — representing
          "no prior information" — after all other filters are applied.

        The method logs per-sub-criterion effective training year counts so
        callers can see, e.g., "SC53: 13/14 valid target years (1 year missing,
        excluded from training)".

        **SAW normalization (use_saw_normalization=True, criteria mode only):**

        When ``use_saw_normalization=True``:

        * For each training pair ``(year_t, year_{t+1})``, the target vector
          is taken from ``SAWNormalizer.normalize_year(panel_data, year_{t+1})``
          rather than from raw ``entity_data.loc[year_{t+1}]``.  This means the
          model is trained to predict per-year minmax-normalized criterion scores
          bounded in [0, 1] rather than raw composite values.
        * Per-year normalization is pre-computed once over all years for
          efficiency (O(Y) normalization passes, not O(N × Y)).
        * Fallback: if ``year_{t+1}`` has no cross-section entry (e.g. a
          synthetic future year) or the entity has no row in the cross-section,
          the raw value is used instead.
        * NaN handling for criteria mode is unchanged: rows where the entire
          normalized target vector is NaN are excluded; rows with *partial* NaN
          (some criteria missing) have those cells imputed with the per-column
          median of the normalized training targets (still in [0, 1]).

        **Holdout split (holdout_year is not None):**

        Training samples whose *target* year equals ``holdout_year`` are
        separated into ``X_holdout / y_holdout`` instead of
        ``X_train / y_train``.  The holdout set is never used during fitting;
        it is returned to the caller for post-training evaluation.

        When ``holdout_year`` falls within the training window (typically
        ``holdout_year = max(training_years)``), the effective training set
        shrinks by one year-step.  For example, with data 2011–2024 and
        ``target_year = 2025``:

        * ``holdout_year = 2024`` → training covers targets 2012–2023 (features
          2011–2022); holdout = targets 2024 (features 2023); prediction uses
          features 2024 to forecast 2025.

        Args:
            panel_data: Panel data object with provinces, years, subcriteria,
                        and (optionally) ``year_contexts``.
            target_year: Year to predict.
            use_saw_normalization: When True (and ``target_level == 'criteria'``),
                replace raw target values with SAW per-year minmax-normalized
                scores in [0, 1].  Ignored in sub-criteria mode.
            holdout_year: When not None, training samples whose target year
                equals this value are withheld and returned as
                ``(X_holdout, y_holdout)``.  Typically set to
                ``max(training_years)`` so the most recent complete year
                serves as a true out-of-sample evaluation set.

        Returns:
            Tuple of six DataFrames:

            X_train (n_train × n_features)
                Feature matrix for model fitting.  Excludes holdout rows.
            y_train (n_train × n_components)
                Target matrix.  Values are SAW-normalized [0, 1] when
                ``use_saw_normalization=True``; raw composites otherwise.
            X_pred (n_pred × n_features)
                Feature matrix for the forecast target year, indexed by
                province name.
            entity_info (n_train × 1)
                Training entity integer indices (column ``entity_index``).
            X_holdout (n_holdout × n_features)
                Holdout feature matrix.  Empty DataFrame when
                ``holdout_year`` is None or no samples fall in that year.
            y_holdout (n_holdout × n_components)
                Holdout target matrix (same normalization as ``y_train``).
                Empty DataFrame when ``holdout_year`` is None.
        """
        entities   = panel_data.provinces
        years      = sorted(panel_data.years)

        # Select components and data accessor based on target_level
        if self.target_level == "criteria":
            components = panel_data.criteria_names          # ['C01', ..., 'C08']
            def _get_entity_data(entity):
                return panel_data.get_province_criteria(entity)
        else:
            components = panel_data.subcriteria_names       # ['SC11', ..., 'SC83']
            def _get_entity_data(entity):
                return panel_data.get_province(entity)

        if target_year in years:
            last_year = target_year
        else:
            last_year = years[-1]

        train_feature_years = [y for y in years if y < last_year]

        # ------------------------------------------------------------------
        # Pre-compute per-component effective training year counts (logging)
        # ------------------------------------------------------------------
        if self.target_level != "criteria":
            # Sub-criteria mode: use year_contexts.is_valid() per SC
            sc_target_year_valid: Dict[str, int] = {}
            for sc in components:
                valid_count = 0
                for year in train_feature_years:
                    next_yr = years[years.index(year) + 1]
                    ctx_next = getattr(panel_data, 'year_contexts', {}).get(next_yr)
                    if ctx_next is not None:
                        if any(ctx_next.is_valid(ent, sc) for ent in entities):
                            valid_count += 1
                    else:
                        valid_count += 1
                sc_target_year_valid[sc] = valid_count

            for sc, count in sorted(sc_target_year_valid.items()):
                total = len(train_feature_years)
                if count < total:
                    logger.debug(
                        f"{sc}: {count}/{total} valid target years "
                        f"({total - count} missing → excluded from training)"
                    )

        # ------------------------------------------------------------------
        # Pre-compute entity-level means for stationarity demeaning (Phase 2)
        #
        # For each (entity, component) pair, compute:
        #   entity_mean   = mean of entity's raw values over training feature years
        #   entity_mean_delta = mean first-difference of entity over consecutive
        #                       training feature year pairs
        #
        # Computed here (O(E × C × Y)) rather than per-sample inside
        # _create_features() (which would be O(E × C × Y²)), and stored as
        # instance attributes so they are available during prediction.
        #
        # Leakage-free design:
        #   - Means are taken over `train_feature_years` (raw feature data),
        #     NOT over target years.  Target values (y) are never accessed
        #     here.  This is standard panel econometrics ("within estimator").
        #   - Year pairs where at least one endpoint is NaN in the entity's
        #     time series are skipped in the mean-delta computation (NaN-safe).
        #
        # E-01 (fold-aware correction):
        #   Additionally, per-entity per-year raw values are stored in
        #   ``_entity_yearly_values_`` so that ``compute_fold_entity_corrections``
        #   can recompute entity means restricted to any temporal training
        #   window — correcting the look-ahead bias that arises when the CV
        #   loop validates on year k but entity means include years k+1…T.
        # ------------------------------------------------------------------
        self._entity_means_ = {}
        self._entity_mean_deltas_ = {}

        # E-01: per-entity per-year raw values for fold-aware demean correction
        self._entity_yearly_values_: Dict = {}
        self._entities_: List[str] = list(entities)
        self._components_: List[str] = list(components)
        self._all_train_feature_years_: List[int] = list(train_feature_years)

        for _entity in entities:
            _edata = _get_entity_data(_entity).reindex(years)
            for _c in components:
                # ── Entity mean ────────────────────────────────────────────
                _vals = _edata.loc[train_feature_years, _c].values.astype(float)
                _mean = float(np.nanmean(_vals)) if not np.all(np.isnan(_vals)) else 0.0
                self._entity_means_[(_entity, _c)] = _mean

                # ── Entity mean delta ──────────────────────────────────────
                # Δ_t = val[years[i]] - val[years[i-1]] for every consecutive
                # pair within train_feature_years.  NaN on either endpoint
                # skips that pair so one missing year does not void the mean.
                _deltas = []
                for _i in range(1, len(train_feature_years)):
                    _v_curr = float(_edata.loc[train_feature_years[_i], _c])
                    _v_prev = float(_edata.loc[train_feature_years[_i - 1], _c])
                    if not (np.isnan(_v_curr) or np.isnan(_v_prev)):
                        _deltas.append(_v_curr - _v_prev)
                self._entity_mean_deltas_[(_entity, _c)] = (
                    float(np.mean(_deltas)) if _deltas else 0.0
                )

                # ── E-01: store per-year raw value ─────────────────────────
                for _yr in train_feature_years:
                    _raw = (
                        float(_edata.loc[_yr, _c])
                        if _yr in _edata.index else np.nan
                    )
                    self._entity_yearly_values_[(_entity, _c, _yr)] = _raw

        # ------------------------------------------------------------------
        # Pre-compute cross-sectional medians per (year, component) for
        # lag-feature NaN imputation (Fix D-01).
        #
        # For each year and component, the median is taken over active
        # provinces only (from year_contexts).  Lag slots that are NaN are
        # filled with this median in _create_features(), replacing the
        # previous 0.0 sentinel which conflated "missing data" with a
        # legitimate zero governance score.
        #
        # Leakage-safe: medians are derived from observed feature-year data
        # only; target-year values are never accessed here.
        # ------------------------------------------------------------------
        self._component_year_medians_ = {}
        for _yr in years:
            _ctx_yr = getattr(panel_data, 'year_contexts', {}).get(_yr)
            _active_yr = (
                _ctx_yr.active_provinces if _ctx_yr is not None else entities
            )
            for _c in components:
                _mvals: List[float] = []
                for _ent in _active_yr:
                    _edata_m = _get_entity_data(_ent)
                    if _yr in _edata_m.index:
                        _v = float(_edata_m.loc[_yr, _c])
                        if not np.isnan(_v):
                            _mvals.append(_v)
                self._component_year_medians_[(_yr, _c)] = (
                    float(np.median(_mvals)) if _mvals else 0.0
                )

        # ===== PHASE A Enhancement: Compute Imputation Statistics (Tiers 2-4) =====
        self._imputation_config = imputation_config or ImputationConfig()
        if self._imputation_config.use_advanced_feature_imputation:
            self._compute_imputation_statistics(panel_data)

        # ------------------------------------------------------------------
        # Pre-compute SAW-normalized cross-sections (one pass per year, O(Y))
        # Used only in criteria mode when use_saw_normalization=True.
        # ------------------------------------------------------------------
        saw_norms: Dict[int, pd.DataFrame] = {}
        if use_saw_normalization and self.target_level == "criteria":
            _saw_normalizer = SAWNormalizer()
            for yr in years:
                try:
                    saw_norms[yr] = _saw_normalizer.normalize_year(panel_data, yr)
                except (KeyError, Exception):
                    pass  # Graceful skip (year absent from cross-section dict)

        # ------------------------------------------------------------------
        # Build training, holdout, and prediction samples
        # ------------------------------------------------------------------
        X_train_list:    List[np.ndarray] = []
        y_train_list:    List[np.ndarray] = []
        entity_indices:  List[int] = []
        year_labels_train: List[int] = []
        X_holdout_list:  List[np.ndarray] = []
        y_holdout_list:  List[np.ndarray] = []
        pred_feature_list: List[np.ndarray] = []
        pred_entities: List[str] = []

        # Year context for the prediction target year
        ctx_pred_year = getattr(panel_data, 'year_contexts', {}).get(target_year)
        if ctx_pred_year is None and target_year not in years:
            # Future year: use the last available year's context
            ctx_pred_year = getattr(panel_data, 'year_contexts', {}).get(last_year)

        n_skipped_train = 0
        n_skipped_pred  = 0

        for ent_idx, entity in enumerate(entities):
            entity_data = _get_entity_data(entity)

            # ---- Training samples ----
            for year in train_feature_years:
                next_yr = years[years.index(year) + 1]

                # Check entity is active in target year (next_yr)
                ctx_next = getattr(panel_data, 'year_contexts', {}).get(next_yr)
                if ctx_next is not None and entity not in ctx_next.active_provinces:
                    n_skipped_train += 1
                    continue  # Entity absent this year — not a valid target

                # Check we can form the target vector (no NaN → no imputation)
                if next_yr not in entity_data.index:
                    n_skipped_train += 1
                    continue

                target = entity_data.loc[next_yr, components].values.astype(float)

                # Enhancement M-04: Relaxed partial-NaN handling
                # Skip only if ALL targets are NaN (both criteria and sub-criteria modes)
                # Partial NaN targets are preserved; downstream models handle via
                # sample weighting (BayesianForecaster) or component-wise training
                if np.all(np.isnan(target)):
                    n_skipped_train += 1
                    continue

                # Guard: entity must also be present in the feature year.
                if year not in entity_data.index:
                    n_skipped_train += 1
                    continue

                # ---- Resolve target vector --------------------------------
                # When SAW normalization is enabled and a cross-section is
                # available for next_yr, look up the pre-normalized row.
                # Fallback to raw entity_data values if the entity is absent
                # from the cross-section (should be rare given active-province
                # guard above) or if SAW norms are not pre-computed.
                if (
                    use_saw_normalization
                    and self.target_level == "criteria"
                    and next_yr in saw_norms
                ):
                    saw_cs = saw_norms[next_yr]
                    if entity in saw_cs.index:
                        # Align to the component list in case column ordering
                        # differs between cross-section and entity time series
                        target = (
                            saw_cs.loc[entity, [c for c in components
                                                if c in saw_cs.columns]]
                            .reindex(components)
                            .values
                            .astype(float)
                        )
                    else:
                        # Entity absent from cross-section → raw fallback
                        target = (
                            entity_data.loc[next_yr, components]
                            .values.astype(float)
                        )
                else:
                    target = entity_data.loc[next_yr, components].values.astype(float)

                # Build features; NaN feature cells → 0.0 (no prior info)
                features = self._create_features_safe(
                    entity_data, entity, years, year, entities, panel_data
                )

                # ---- Route to training or holdout -------------------------
                if holdout_year is not None and next_yr == holdout_year:
                    X_holdout_list.append(features)
                    y_holdout_list.append(target)
                    # Note: entity_indices is training-only (panel-aware CV)
                else:
                    X_train_list.append(features)
                    y_train_list.append(target)
                    entity_indices.append(ent_idx)
                    year_labels_train.append(next_yr)

            # ---- Prediction sample ----
            # Determine the per-entity feature year for the prediction sample.
            # Normally this equals `last_year` (e.g. 2024 when predicting 2025).
            #
            # Special case — future-year prediction where a province is entirely
            # absent from the most-recent year context (all sub-criteria NaN in
            # 2024, so DataLoader excluded it from `active_provinces[2024]`):
            # instead of excluding the province, fall back to the most-recent year
            # that has at least one valid observation.  This guarantees all
            # historically-active provinces receive a 2025 prediction.
            #
            # For in-sample predictions (target_year already in the panel) the
            # original hard exclusion is preserved — a missing target year means
            # there is no ground truth to train against, so we skip it.
            is_future_prediction = target_year not in years

            if ctx_pred_year is not None and entity not in ctx_pred_year.active_provinces:
                if not is_future_prediction:
                    # In-sample: entity absent from the prediction year → skip.
                    n_skipped_pred += 1
                    continue

                # Future-year: scan backwards for the most-recent year that has
                # at least one non-NaN value for this entity.
                entity_pred_feature_year: Optional[int] = None
                for yr in reversed(years):
                    if yr in entity_data.index:
                        row_vals = entity_data.loc[yr, components].values.astype(float)
                        if not np.all(np.isnan(row_vals)):
                            entity_pred_feature_year = yr
                            break

                if entity_pred_feature_year is None:
                    # Entity has no valid data across the entire historical panel.
                    n_skipped_pred += 1
                    logger.warning(
                        f"Forecasting: {entity} excluded from {target_year} prediction"
                        f" — no valid observations across entire historical panel."
                    )
                    continue

                logger.info(
                    f"Forecasting: {entity} has no {last_year} data; using "
                    f"{entity_pred_feature_year} features for {target_year} "
                    f"prediction (most-recent valid year fallback)."
                )
            else:
                entity_pred_feature_year = last_year

            if entity_pred_feature_year not in entity_data.index:
                n_skipped_pred += 1
                continue

            pred_features = self._create_features_safe(
                entity_data, entity, years, entity_pred_feature_year, entities, panel_data
            )
            pred_feature_list.append(pred_features)
            pred_entities.append(entity)

        if n_skipped_train > 0:
            logger.info(
                f"Forecasting: {n_skipped_train} training observations "
                f"excluded (missing target values or inactive province-year pairs)"
            )
        if n_skipped_pred > 0:
            logger.info(
                f"Forecasting: {n_skipped_pred} province(s) excluded from prediction"
                f" (no valid historical data)"
            )

        if not X_train_list:
            raise ValueError(
                "No valid training samples after dynamic exclusion. "
                "Check that the panel has sufficient complete observations."
            )
        if not pred_feature_list:
            raise ValueError(
                "No active provinces for prediction in the target year. "
                "All provinces may be missing data."
            )

        # ---- Assemble training arrays ----
        X_train = np.vstack(X_train_list)
        y_train = np.vstack(y_train_list)
        X_pred  = np.vstack(pred_feature_list)

        # Enhancement M-04: Partial-NaN target handling (both modes)
        # Leave NaN in y_train — downstream models (BayesianForecaster) use
        # sample weights to mask NaN cells. Models without NaN support (e.g.,
        # CatBoost MultiRMSE) must handle separately in their fit() methods.
        # 
        # Note: This is a breaking change from the Phase 1 behavior where
        # criteria-mode NaN were imputed with median. Now all models must be
        # NaN-aware or apply their own imputation strategy before fitting.
        pass  # Intentionally preserve NaN in y_train

        # ---- Assemble holdout arrays ----
        if X_holdout_list:
            X_holdout = np.vstack(X_holdout_list)
            y_holdout = np.vstack(y_holdout_list)
            # M-04: Preserve NaN in holdout targets (same as training)
            # Evaluation metrics must be NaN-aware (use np.nanmean, etc.)
            X_holdout_df = pd.DataFrame(X_holdout, columns=self.feature_names_)
            y_holdout_df = pd.DataFrame(y_holdout, columns=components)
        else:
            X_holdout_df = pd.DataFrame(columns=self.feature_names_)
            y_holdout_df = pd.DataFrame(columns=components)

        X_train_df      = pd.DataFrame(X_train, columns=self.feature_names_)
        y_train_df      = pd.DataFrame(y_train, columns=components)
        X_pred_df       = pd.DataFrame(
            X_pred, columns=self.feature_names_, index=pred_entities
        )
        entity_index_df = pd.DataFrame(
            {
                'entity_index': np.array(entity_indices, dtype=int),
                'year_label':   np.array(year_labels_train, dtype=int),
            }
        )

        return X_train_df, y_train_df, X_pred_df, entity_index_df, X_holdout_df, y_holdout_df

    # ------------------------------------------------------------------
    # E-01 helper: fold-aware entity demean corrections
    # ------------------------------------------------------------------

    def compute_fold_entity_corrections(
        self,
        max_feature_year: int,
    ) -> Tuple[Dict, Dict]:
        """
        Compute entity-mean and entity-mean-delta restricted to feature
        years ≤ ``max_feature_year``.

        This corrects the look-ahead bias in entity-demeaned features when
        they are used inside a CV fold whose training window ends before
        the full panel.  The correction offset for each row is::

            Δ_mean  = global_entity_mean  − fold_entity_mean
            Δ_delta = global_entity_delta − fold_entity_delta

        Adding these offsets to the pre-computed ``_demeaned`` /
        ``_demeaned_momentum`` columns in ``X_train_tree_`` yields values
        equivalent to having computed entity statistics from the fold's
        training window only, without re-running the full feature pipeline.

        Parameters
        ----------
        max_feature_year : int
            Maximum feature year (= max CV training target year − 1) whose
            data should be included in the fold-restricted entity means.
            Rows from ``train_feature_years`` with year > this value are
            excluded from the fold mean computation.

        Returns
        -------
        fold_means : dict  {(entity_name, component): float}
            Per-(entity, component) mean computed from feature years
            ≤ ``max_feature_year``.
        fold_mean_deltas : dict  {(entity_name, component): float}
            Per-(entity, component) mean first-difference from feature
            years ≤ ``max_feature_year``.

        Notes
        -----
        * Returns global statistics unchanged when ``max_feature_year``
          equals or exceeds the largest training feature year — no
          correction is needed in that case.
        * Falls back to 0.0 for entities/components with no valid values
          in the restricted window (safe: demeaning by 0 is a no-op).
        """
        if not self._all_train_feature_years_:
            return dict(self._entity_means_), dict(self._entity_mean_deltas_)

        global_max_feat_year = max(self._all_train_feature_years_)
        if max_feature_year >= global_max_feat_year:
            # No future leakage in this fold — global stats are correct
            return dict(self._entity_means_), dict(self._entity_mean_deltas_)

        eligible_years = sorted(
            [y for y in self._all_train_feature_years_ if y <= max_feature_year]
        )
        if not eligible_years:
            return dict(self._entity_means_), dict(self._entity_mean_deltas_)

        fold_means: Dict = {}
        fold_mean_deltas: Dict = {}

        for entity in self._entities_:
            for comp in self._components_:
                vals = [
                    self._entity_yearly_values_.get((entity, comp, yr), np.nan)
                    for yr in eligible_years
                ]
                valid_vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
                fold_means[(entity, comp)] = (
                    float(np.mean(valid_vals)) if valid_vals else 0.0
                )

                deltas: List[float] = []
                for i in range(1, len(eligible_years)):
                    v_curr = self._entity_yearly_values_.get(
                        (entity, comp, eligible_years[i]), np.nan
                    )
                    v_prev = self._entity_yearly_values_.get(
                        (entity, comp, eligible_years[i - 1]), np.nan
                    )
                    if not (np.isnan(v_curr) or np.isnan(v_prev)):
                        deltas.append(float(v_curr) - float(v_prev))
                fold_mean_deltas[(entity, comp)] = (
                    float(np.mean(deltas)) if deltas else 0.0
                )

        return fold_means, fold_mean_deltas

    def _compute_imputation_statistics(self, panel_data) -> None:
        """
        Compute imputation caches for Tiers 2-4 (pre-computed during fit).
        
        This ensures:
        ✓ No test-set leakage (uses training data only)
        ✓ Consistent imputation across folds
        ✓ Production stability (statistics cached, not recomputed at inference)
        
        PHASE A Enhancement M-12: Advanced tiered imputation strategy.
        """
        if not self._imputation_config or not self._imputation_config.use_advanced_feature_imputation:
            return
        
        components = list(panel_data.criteria.columns)
        
        # ===== TIER 2: Temporal medians for time-series blocks =====
        # Blocks: 3 (rolling stats), 4 (momentum), 7 (EWMA), 8 (expanding), 10 (skewness)
        temporal_blocks = {3, 4, 7, 8, 10}
        
        for entity in panel_data.subcriteria.index.get_level_values(0).unique() if hasattr(
            panel_data.subcriteria.index, 'get_level_values') else panel_data.subcriteria.index:
            try:
                if isinstance(panel_data.subcriteria.index, pd.MultiIndex):
                    entity_rows = panel_data.subcriteria.xs(entity, level=0)
                else:
                    entity_rows = panel_data.subcriteria.loc[[entity]]
                
                for component in components:
                    if component in entity_rows.columns:
                        entity_ts = entity_rows[component].dropna()
                        if len(entity_ts) >= self._imputation_config.temporal_imputation_min_periods:
                            temporal_median = entity_ts.median()
                            for block_id in temporal_blocks:
                                self._temporal_medians_[(entity, block_id, component)] = temporal_median
            except (KeyError, IndexError):
                continue
        
        # ===== TIER 3: Cross-sectional medians for feature blocks =====
        # Blocks: 1 (current), 5 (entity-demeaned), 6 (trend), 9 (diversity), 11 (percentile)
        sectional_blocks = {1, 5, 6, 9, 11}
        
        try:
            if isinstance(panel_data.final.index, pd.MultiIndex):
                years = panel_data.final.index.get_level_values(1).unique()
            else:
                years = [y for y in panel_data.final.index if isinstance(y, int)]
            
            for year in years:
                try:
                    if isinstance(panel_data.final.index, pd.MultiIndex):
                        year_data = panel_data.final.xs(year, level=1)
                    else:
                        year_rows = []
                        for idx in panel_data.final.index:
                            if isinstance(idx, tuple) and len(idx) > 1 and idx[1] == year:
                                year_rows.append(idx)
                        if year_rows:
                            year_data = panel_data.final.loc[year_rows]
                        else:
                            continue
                    
                    for component in components:
                        if component in year_data.columns:
                            component_vals = year_data[component].dropna()
                            if len(component_vals) > 0:
                                cs_median = component_vals.median()
                                for block_id in sectional_blocks:
                                    key = (year, block_id, component)
                                    self._crosssectional_medians_[key] = cs_median
                except (KeyError, IndexError):
                    continue
        except Exception:
            pass
        
        # ===== TIER 4: Training means per block =====
        try:
            all_vals = panel_data.final.values.flatten()
            all_vals_valid = all_vals[~np.isnan(all_vals)]
            if len(all_vals_valid) > 0:
                global_mean = float(np.mean(all_vals_valid))
                for block_id in range(1, 12):
                    for component in components:
                        self._block_training_means_[(block_id, component)] = global_mean
        except Exception:
            pass
        
        logger.info(
            f"Imputation statistics computed: "
            f"temporal_medians={len(self._temporal_medians_)}, "
            f"crosssectional_medians={len(self._crosssectional_medians_)}, "
            f"training_means={len(self._block_training_means_)}"
        )

    def _get_imputed_value_for_block(
        self,
        block_id: int,
        entity: str,
        component: str,
        current_year: int,
    ) -> float:
        """
        Retrieve imputation value based on block's tier strategy.
        
        Logic (fallback chain):
        1. Tier 2 (Temporal median) if block is temporal
        2. Tier 3 (Cross-sectional median) if block is sectional
        3. Tier 4 (Training mean) as universal fallback
        4. 0.0 as absolute final fallback
        
        Returns
        -------
        float : imputation value (never NaN, guaranteed scalar)
        """
        tier = self._imputation_config.block_imputation_tiers.get(block_id, "training_mean")
        
        if tier == "temporal_median":
            # Tier 2: Per-entity annual median
            temporal_val = self._temporal_medians_.get((entity, block_id, component))
            if temporal_val is not None and not np.isnan(temporal_val):
                return float(temporal_val)
            
            # Fallback to Tier 3
            cs_val = self._crosssectional_medians_.get((current_year, block_id, component))
            if cs_val is not None and not np.isnan(cs_val):
                return float(cs_val)
            
            # Fallback to Tier 4
            tm_val = self._block_training_means_.get((block_id, component), 0.0)
            return float(tm_val) if not np.isnan(tm_val) else 0.0
        
        elif tier == "cross_sectional_median":
            # Tier 3: Cross-sectional median
            cs_val = self._crosssectional_medians_.get((current_year, block_id, component))
            if cs_val is not None and not np.isnan(cs_val):
                return float(cs_val)
            
            # Fallback to Tier 4
            tm_val = self._block_training_means_.get((block_id, component), 0.0)
            return float(tm_val) if not np.isnan(tm_val) else 0.0
        
        else:  # "training_mean" or default
            # Tier 4: Training means
            tm_val = self._block_training_means_.get((block_id, component), 0.0)
            return float(tm_val) if not np.isnan(tm_val) else 0.0

    def _map_feature_index_to_block(self, feature_index: int) -> Tuple[int, str]:
        """
        Inverse map: feature index → (block_id, component_name).
        
        Uses self.feature_names_ (canonical list built during first _create_features call)
        to identify which block and component a feature belongs to.
        
        Returns
        -------
        tuple : (block_id: int, component: str)
        """
        if not hasattr(self, 'feature_names_') or feature_index >= len(self.feature_names_):
            return (1, "unknown")
        
        feature_name = self.feature_names_[feature_index]
        
        # Parse feature name to extract block and component
        if "_current" in feature_name:
            block_id = 1
        elif "_lag" in feature_name and "_was_missing" not in feature_name:
            block_id = 2
        elif "_roll" in feature_name:
            block_id = 3
        elif "momentum" in feature_name or "acceleration" in feature_name:
            block_id = 4
        elif "_demeaned" in feature_name or "_delta2" in feature_name:
            block_id = 5
        elif "_trend" in feature_name:
            block_id = 6
        elif "_ewma" in feature_name:
            block_id = 7
        elif "_expanding" in feature_name:
            block_id = 8
        elif "_diversity" in feature_name or "_range" in feature_name:
            block_id = 9
        elif "_skew" in feature_name:
            block_id = 10
        elif "_percentile" in feature_name or "_zscore" in feature_name or "_rank" in feature_name:
            block_id = 11
        elif "_region" in feature_name:
            block_id = 12
        else:
            block_id = 1
        
        # Extract component key (e.g., "C01" from "C01_current")
        component = feature_name.split("_")[0]
        
        return (block_id, component)

    def _apply_tiered_imputation(
        self,
        features: np.ndarray,
        entity: str,
        current_year: int,
    ) -> np.ndarray:
        """
        Apply tiered imputation to feature array.
        
        Replaces NaN values using tier strategy (Temporal → Sectional → Training Mean → 0.0)
        with fallback chain ensuring no NaN escapes to downstream models.
        
        PHASE A Enhancement M-12.
        """
        if not self._imputation_config or not self._imputation_config.use_advanced_feature_imputation:
            return features
        
        imputed_count = 0
        for i, val in enumerate(features):
            if np.isnan(val):
                # Map feature index to block and component
                block_id, component = self._map_feature_index_to_block(i)
                
                # Apply tiered logic
                imputed_val = self._get_imputed_value_for_block(
                    block_id, entity, component, current_year
                )
                
                features[i] = imputed_val
                imputed_count += 1
        
        return features

    def _create_features_safe(
        self,
        entity_data: 'pd.DataFrame',
        entity: str,
        years: List[int],
        current_year: int,
        all_entities: List[str],
        panel_data,
    ) -> np.ndarray:
        """
        NaN-safe wrapper around :meth:`_create_features` with tiered imputation support.
        
        PHASE A Enhancement M-12: Uses multi-tier imputation strategy:
        - Tier 1 (MICE): Applied in preprocessing.PanelFeatureReducer
        - Tier 2 (Temporal): Per-entity annual medians (applied here)
        - Tier 3 (Sectional): Cross-sectional medians (applied here)
        - Tier 4 (Fallback): Training means from caches (applied here)

        :meth:`_create_features` handles lag-feature NaN internally by filling
        with the cross-sectional median and appending ``_was_missing`` indicator
        flags.  Any residual NaN values in non-lag blocks (e.g. ``_delta2`` or
        ``_demeaned_momentum`` when lag history is absent) are replaced using
        tiered imputation (if enabled) or per-column means (legacy mode).
        """
        features = self._create_features(
            entity_data, entity, years, current_year, all_entities, panel_data
        )
        
        # Apply advanced tiered imputation if enabled
        if self._imputation_config and self._imputation_config.use_advanced_feature_imputation:
            features = self._apply_tiered_imputation(features, entity, current_year)
        
        # Fallback: any remaining NaN → safety wrapper with training means
        return fill_missing_features(features)

    def _create_features(self,
                         entity_data: pd.DataFrame,
                         entity: str,
                         years: List[int],
                         current_year: int,
                         all_entities: List[str],
                         panel_data) -> np.ndarray:
        """Create feature vector for a single entity-year combination.

        Returns a 1-D float array whose length equals ``len(self.feature_names_)``
        after the first call establishes the canonical feature list.  NaN values
        that cannot be resolved inside this method (e.g. ``_delta2`` when lag-2
        is unavailable) are left as NaN and replaced with ``0.0`` by
        :meth:`_create_features_safe`.
        """
        components = list(entity_data.columns)
        # Reindex to the full year sequence so absent years produce NaN rows.
        entity_data = entity_data.reindex(years)
        features = []
        feature_names: List[str] = []

        year_idx = years.index(current_year)
        available_years = [y for y in years if y <= current_year]
        current_values = entity_data.loc[current_year, components].values.astype(float)

        # ==================================================================
        # Block 1: Current values
        # ==================================================================
        features.extend(current_values)
        feature_names.extend([f"{c}_current" for c in components])

        # ==================================================================
        # Block 2: Lag features with missingness indicators (Fix D-01)
        #
        # NaN lag slots are filled with the training cross-sectional median
        # (not 0.0) to avoid conflating "no history available" with a
        # legitimate zero governance score.  A binary ``_was_missing`` flag
        # (1.0 = imputed, 0.0 = observed) is appended for every lag×component
        # pair so models can learn to discount or adjust for imputed inputs.
        # ==================================================================
        for lag in self.lag_periods:
            if year_idx - lag >= 0:
                lag_year = years[year_idx - lag]
                lag_values = entity_data.loc[lag_year, components].values.astype(float)
            else:
                lag_year = None
                lag_values = np.full(len(components), np.nan)

            was_missing = np.isnan(lag_values)  # record BEFORE imputation
            if was_missing.any():
                fill_yr = lag_year if lag_year is not None else current_year
                for ci, c in enumerate(components):
                    if was_missing[ci]:
                        lag_values[ci] = self._component_year_medians_.get(
                            (fill_yr, c), 0.0
                        )

            features.extend(lag_values)
            feature_names.extend([f"{c}_lag{lag}" for c in components])
            # Missingness indicator flags
            features.extend(was_missing.astype(float))
            feature_names.extend([f"{c}_lag{lag}_was_missing" for c in components])

        # ==================================================================
        # Block 3: Rolling statistics (mean, std, min, max)
        # Windows from self.rolling_windows (default: [2, 3, 5]).
        # ==================================================================
        for window in self.rolling_windows:
            if len(available_years) >= window:
                window_years = available_years[-window:]
                window_data = entity_data.loc[window_years, components]

                features.extend(window_data.mean().values)
                feature_names.extend([f"{c}_roll{window}_mean" for c in components])

                std_vals = window_data.std().fillna(0).values
                features.extend(std_vals)
                feature_names.extend([f"{c}_roll{window}_std" for c in components])

                features.extend(window_data.min().values)
                feature_names.extend([f"{c}_roll{window}_min" for c in components])

                features.extend(window_data.max().values)
                feature_names.extend([f"{c}_roll{window}_max" for c in components])
            else:
                # Insufficient history: padding with NaN triggers Tiered Median Imputation (Phase 4.5)
                features.extend(np.full(len(components), np.nan))
                feature_names.extend([f"{c}_roll{window}_mean" for c in components])
                features.extend(np.full(len(components), np.nan))
                feature_names.extend([f"{c}_roll{window}_std" for c in components])
                features.extend(np.full(len(components), np.nan))
                feature_names.extend([f"{c}_roll{window}_min" for c in components])
                features.extend(np.full(len(components), np.nan))
                feature_names.extend([f"{c}_roll{window}_max" for c in components])

        # ==================================================================
        # Block 4: Momentum and acceleration (first and second-order changes)
        # ==================================================================
        if self.include_momentum and year_idx > 0:
            prev_year = years[year_idx - 1]
            prev_values = entity_data.loc[prev_year, components].values.astype(float)
            momentum = current_values - prev_values
            features.extend(momentum)
            feature_names.extend([f"{c}_momentum" for c in components])

            if year_idx > 1:
                prev_prev_year = years[year_idx - 2]
                prev_prev_values = entity_data.loc[prev_prev_year, components].values.astype(float)
                prev_momentum = prev_values - prev_prev_values
                acceleration = momentum - prev_momentum
                features.extend(acceleration)
                feature_names.extend([f"{c}_acceleration" for c in components])
            else:
                features.extend(np.full(len(components), np.nan))
                feature_names.extend([f"{c}_acceleration" for c in components])
        else:
            features.extend(np.full(len(components), np.nan))
            feature_names.extend([f"{c}_momentum" for c in components])
            features.extend(np.full(len(components), np.nan))
            feature_names.extend([f"{c}_acceleration" for c in components])

        # ==================================================================
        # Block 5: Stationarity features
        #
        # Block 5-B — Entity-demeaned level:   y_t − ȳ_entity
        # Block 5-C — Entity-demeaned momentum: (y_t − y_{t-1}) − mean_Δ_entity
        # Block 5-D — Lagged first-difference:  delta2 = y_{t-1} − y_{t-2}
        #
        # NOTE: delta1 = y_t − y_{t-1} is identical to _momentum (Block 4)
        # and is intentionally omitted here (Fix D-02) to eliminate the
        # redundant feature dimension that wasted PCA variance budget and
        # inflated momentum feature-importance scores in threshold_only mode.
        # ==================================================================
        _st_lag1 = (
            entity_data.loc[years[year_idx - 1], components].values.astype(float)
            if year_idx >= 1 else np.full(len(components), np.nan)
        )
        _st_lag2 = (
            entity_data.loc[years[year_idx - 2], components].values.astype(float)
            if year_idx >= 2 else np.full(len(components), np.nan)
        )
        # delta1 as local variable only — feeds demeaned_momentum; not a feature.
        _delta1 = current_values - _st_lag1  # may contain NaN

        # ── Block 5-B: Entity-demeaned level ───────────────────────────────
        _entity_mean_vec = np.array(
            [self._entity_means_.get((entity, c), 0.0) for c in components],
            dtype=float,
        )
        _demeaned = current_values - _entity_mean_vec
        features.extend(_demeaned)
        feature_names.extend([f"{c}_demeaned" for c in components])

        # ── Block 5-C: Entity-demeaned momentum ────────────────────────────
        _entity_mean_delta_vec = np.array(
            [self._entity_mean_deltas_.get((entity, c), 0.0) for c in components],
            dtype=float,
        )
        _demeaned_momentum = _delta1 - _entity_mean_delta_vec
        features.extend(_demeaned_momentum)
        feature_names.extend([f"{c}_demeaned_momentum" for c in components])

        # ── Block 5-D: Lagged first-difference (delta2) ────────────────────
        _delta2 = _st_lag1 - _st_lag2  # may contain NaN
        features.extend(_delta2)
        feature_names.extend([f"{c}_delta2" for c in components])

        # ==================================================================
        # Block 6: Trend (linear slope via polyfit)
        #
        # Requires ≥ 3 valid (non-NaN) data points for a meaningful slope
        # estimate (Fix G-08: previously ≥ 2 allowed two-point fits that
        # amplify noise on short histories; 3-point minimum is the smallest
        # sample with one degree of freedom after including the intercept).
        # ==================================================================
        if len(available_years) >= 2:
            for c in components:
                y_vals_raw = entity_data.loc[available_years, c].values.astype(float)
                x_vals_raw = np.arange(len(y_vals_raw), dtype=float)
                valid_mask = ~np.isnan(y_vals_raw)
                if valid_mask.sum() >= 3:
                    slope = np.polyfit(
                        x_vals_raw[valid_mask], y_vals_raw[valid_mask], 1
                    )[0]
                else:
                    slope = np.nan
                features.append(slope)
                feature_names.append(f"{c}_trend")
        else:
            features.extend(np.full(len(components), np.nan))
            feature_names.extend([f"{c}_trend" for c in components])

        # ==================================================================
        # Block 7: EWMA features — exponentially weighted moving averages
        # (Enhancement G-01)
        #
        # Spans {2, 3, 5} give effective decay half-lives of ≈1.4, 2.1, and
        # 3.6 years, providing recency-sensitive level signals at short,
        # medium, and longer horizons.  ``ewm(min_periods=1)`` ensures a
        # value is always returned even on the first year of history; NaN
        # entries in ``available_years`` are skipped automatically by pandas.
        # ==================================================================
        if self.include_ewma:
            for c in components:
                c_series = pd.Series(
                    entity_data.loc[available_years, c].values.astype(float)
                )
                for span in [2, 3, 5]:
                    ewma_val = float(
                        c_series.ewm(span=span, min_periods=1).mean().iloc[-1]
                    )
                    features.append(ewma_val)
                    feature_names.append(f"{c}_ewma{span}")

        # ==================================================================
        # Block 8: Expanding window mean — unconditional historical baseline
        # (Enhancement G-03)
        #
        # Cumulative mean over all available years captures the province's
        # unconditional long-run baseline.  Orthogonal to _demeaned (which
        # subtracts this mean) and to rolling means (which are window-limited).
        # Particularly informative for slow-moving criteria such as C04
        # (anti-corruption) and C07 (environmental governance).
        # ==================================================================
        if self.include_expanding:
            for c in components:
                exp_vals = entity_data.loc[available_years, c].values.astype(float)
                exp_mean = (
                    float(np.nanmean(exp_vals))
                    if not np.all(np.isnan(exp_vals))
                    else np.nan
                )
                features.append(exp_mean)
                feature_names.append(f"{c}_expanding_mean")

        # ==================================================================
        # Block 9: Inter-component diversity (Enhancement G-04)
        #
        # std and range of current_values across components capture governance
        # imbalance: a province racing ahead on e-governance while lagging on
        # participation exhibits high diversity; a uniform improver exhibits
        # near-zero diversity.  Both are scale-free relative to SAW targets.
        # ==================================================================
        if self.include_diversity:
            valid_curr = current_values[~np.isnan(current_values)]
            diversity_std = float(np.std(valid_curr)) if len(valid_curr) >= 2 else np.nan
            diversity_rng = float(np.ptp(valid_curr)) if len(valid_curr) >= 2 else np.nan
            features.extend([diversity_std, diversity_rng])
            feature_names.extend(["component_diversity_std", "component_diversity_range"])

        # ==================================================================
        # Block 10: Rolling skewness (Enhancement G-07, window=5)
        #
        # Skewness of each component's 5-year distribution detects whether
        # the province is breaking out (positive skew: recent acceleration
        # above history) or regressing to mean (negative skew: post-peak).
        # min_periods=3 is the minimum sample for meaningful skew estimation.
        # ==================================================================
        if self.include_rolling_skewness:
            for c in components:
                if len(available_years) >= 5:
                    window_years_sk = available_years[-5:]
                    sk_vals = entity_data.loc[window_years_sk, c].values.astype(float)
                    valid_sk = sk_vals[~np.isnan(sk_vals)]
                    sk = float(_scipy_stats.skew(valid_sk)) if len(valid_sk) >= 3 else np.nan
                else:
                    sk = np.nan
                features.append(sk)
                feature_names.append(f"{c}_roll5_skewness")

        # ==================================================================
        # Block 11: Panel-relative features
        #   11-A  Cross-entity percentile rank and z-score (Fix D-03 applied)
        #   11-B  Rank-change: Δpercentile = pct_t − pct_{t-1} (G-05)
        #
        # Cross-sections are filtered to active provinces only (Fix D-03):
        # provinces absent from year_contexts[year].active_provinces are
        # excluded from the reference distribution, preventing measurement
        # bias from excluded entities inflating or deflating relative ranks.
        # ==================================================================
        if self.include_cross_entity or self.include_rank_change:
            year_cross_section = self._get_active_cross_section(
                panel_data, current_year
            )
            prev_year_rk = years[year_idx - 1] if year_idx > 0 else None
            prev_cross_section = (
                self._get_active_cross_section(panel_data, prev_year_rk)
                if prev_year_rk is not None
                else pd.DataFrame()
            )

            for c in components:
                entity_value = float(current_values[components.index(c)])

                # ── 11-A: Percentile and z-score ───────────────────────────
                if self.include_cross_entity:
                    if (
                        not year_cross_section.empty
                        and c in year_cross_section.columns
                    ):
                        _active_cols = [col for col in components if col in year_cross_section.columns and not year_cross_section[col].isna().all()]
                        _valid_cohort = year_cross_section.dropna(subset=_active_cols).index
                        if entity in _valid_cohort:
                            col_clean = year_cross_section.loc[_valid_cohort, c]
                        else:
                            col_clean = year_cross_section[c].dropna()
                            
                        if len(col_clean) > 0 and not np.isnan(entity_value):
                            percentile = float((col_clean < entity_value).mean())
                            mean_val   = float(col_clean.mean())
                            std_val    = float(col_clean.std())
                            z_score    = (entity_value - mean_val) / (std_val + 1e-10)
                        else:
                            percentile, z_score = 0.5, 0.0
                    else:
                        percentile, z_score = 0.5, 0.0
                    features.extend([percentile, z_score])
                    feature_names.extend([f"{c}_percentile", f"{c}_zscore"])
                else:
                    percentile = 0.5  # neutral fallback for rank-change when off

                # ── 11-B: Rank-change (Δpercentile) ────────────────────────
                if self.include_rank_change:
                    if (
                        prev_year_rk is not None
                        and not prev_cross_section.empty
                        and c in prev_cross_section.columns
                    ):
                        prev_entity_val = float(
                            entity_data.loc[prev_year_rk, c]
                            if prev_year_rk in entity_data.index
                            else np.nan
                        )
                        _prev_active_cols = [col for col in components if col in prev_cross_section.columns and not prev_cross_section[col].isna().all()]
                        _prev_valid_cohort = prev_cross_section.dropna(subset=_prev_active_cols).index
                        if entity in _prev_valid_cohort:
                            prev_col_clean = prev_cross_section.loc[_prev_valid_cohort, c]
                        else:
                            prev_col_clean = prev_cross_section[c].dropna()

                        if len(prev_col_clean) > 0 and not np.isnan(prev_entity_val):
                            # Recompute current percentile if cross_entity off
                            if not self.include_cross_entity:
                                if (
                                    not year_cross_section.empty
                                    and c in year_cross_section.columns
                                ):
                                    _curr_active_cols = [col for col in components if col in year_cross_section.columns and not year_cross_section[col].isna().all()]
                                    _curr_valid_cohort = year_cross_section.dropna(subset=_curr_active_cols).index
                                    if entity in _curr_valid_cohort:
                                        _curr_col = year_cross_section.loc[_curr_valid_cohort, c]
                                    else:
                                        _curr_col = year_cross_section[c].dropna()
                                    percentile = (
                                        float((_curr_col < entity_value).mean())
                                        if len(_curr_col) > 0
                                        and not np.isnan(entity_value)
                                        else 0.5
                                    )
                            pct_prev = float(
                                (prev_col_clean < prev_entity_val).mean()
                            )
                            rank_change = percentile - pct_prev
                        else:
                            rank_change = 0.0
                    else:
                        rank_change = 0.0
                    features.append(rank_change)
                    feature_names.append(f"{c}_rank_change")

        # ==================================================================
        # Block 12: Regional cluster one-hot dummies (Enhancement G-06)
        #
        # Five geographic regions of Vietnam:
        #   0 = Northern Mountains & Midlands
        #   1 = Red River Delta
        #   2 = Central (North-Central + South-Central coastal)
        #   3 = Central Highlands
        #   4 = Southern (South-East + Mekong Delta)
        #
        # All 5 dummies are retained (no drop-one convention): tree-based
        # models handle collinearity natively, and the PCA track for linear
        # models absorbs any redundancy.
        # Unknown province names → all-zero vector (no information injected).
        # ==================================================================
        if self.include_region_dummies:
            region_idx = _VIETNAM_PROVINCE_REGIONS.get(entity, -1)
            region_vec = np.zeros(_N_REGIONS, dtype=float)
            if 0 <= region_idx < _N_REGIONS:
                region_vec[region_idx] = 1.0
            features.extend(region_vec)
            feature_names.extend([f"region_{i}" for i in range(_N_REGIONS)])

        # ==================================================================
        # Dimension validation
        # ==================================================================
        if self.feature_names_ and len(self.feature_names_) != len(feature_names):
            raise ValueError(
                f"Feature dimension mismatch: previous call produced "
                f"{len(self.feature_names_)} features, this call produced "
                f"{len(feature_names)}. Check cross_section data for "
                f"entity={entity}, year={current_year}."
            )
        self.feature_names_ = feature_names
        return np.array(features, dtype=float)

    def _get_active_cross_section(
        self,
        panel_data,
        year: Optional[int],
    ) -> pd.DataFrame:
        """Return the cross-section for *year* filtered to active provinces.

        Implements Fix D-03: provinces absent from
        ``year_contexts[year].active_provinces`` are excluded from the
        reference distribution before computing percentile ranks and z-scores,
        preventing excluded entities from biasing relative-position statistics.

        Parameters
        ----------
        panel_data :
            Panel data object with ``criteria_cross_section`` /
            ``cross_section`` and optionally ``year_contexts``.
        year : int or None
            Calendar year.  Returns an empty DataFrame when ``year`` is None
            or the year is not present in the cross-section dict.

        Returns
        -------
        pd.DataFrame
            Province-indexed cross-section filtered to active entities.
            Empty DataFrame when data is unavailable.
        """
        if year is None:
            return pd.DataFrame()

        if self.target_level == "criteria":
            cs = panel_data.criteria_cross_section.get(year)
        else:
            # Sub-criteria mode: try legacy 'cross_section' first, then the
            # canonical 'subcriteria_cross_section'.
            cs = getattr(panel_data, "cross_section", {}).get(year)
            if cs is None:
                cs = panel_data.subcriteria_cross_section.get(year)

        if cs is None:
            return pd.DataFrame()

        if "Province" in cs.columns:
            cs = cs.set_index("Province")

        # Filter to active provinces only (Fix D-03)
        ctx = getattr(panel_data, "year_contexts", {}).get(year)
        if ctx is not None:
            active = [p for p in ctx.active_provinces if p in cs.index]
            if active:
                cs = cs.loc[active]

        return cs

    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self.feature_names_
