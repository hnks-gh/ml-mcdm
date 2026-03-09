# -*- coding: utf-8 -*-
"""
Temporal Feature Engineering
============================

Advanced feature engineering for time series panel data, creating
rich feature sets for ML forecasting models.

Features include:
- Lag features (historical values at t-1, t-2)
- Rolling statistics (mean, std, min, max over 2–3 year windows)
- Momentum and acceleration (first/second differences)
- Trend features (linear slope via polyfit)
- Cross-entity features (percentile rank, z-score relative to panel)

Dynamic exclusion via ``year_contexts``
---------------------------------------
When ``panel_data.year_contexts`` is present, ``fit_transform`` removes:

* **Training rows** whose *target* year excludes the entity (province absent
  from ``year_contexts[t+1].active_provinces``) or whose target sub-criterion
  is NaN for that year.  No imputation is performed on target values.

* **Prediction rows** limited to entities in the *forecast year*'s
  ``active_provinces`` set; excluded entities are simply absent from output.

NaN values in *feature* (input) vectors — e.g. missing lag values for
entities with short histories — are filled with 0.0 ("no prior information")
after all filters are applied.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from data.missing_data import fill_missing_features, has_complete_target


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
    
    Creates rich feature set including:
    - Lag features (t-1, t-2, ...)
    - Rolling statistics (mean, std, min, max, trend)
    - Seasonal features
    - Cross-entity features (relative position)
    - Momentum and acceleration features
    
    Parameters:
        lag_periods: List of lag periods to include [1, 2]
        rolling_windows: Window sizes for rolling statistics [2, 3]
        include_momentum: Whether to include rate of change features
        include_cross_entity: Whether to include relative position features
    
    Example:
        >>> engineer = TemporalFeatureEngineer(lag_periods=[1, 2])
        >>> X_train, y_train, X_pred, _ = engineer.fit_transform(panel_data, 2025)
    """
    
    def __init__(self,
                 lag_periods: List[int] = [1, 2],
                 rolling_windows: List[int] = [2, 3],
                 include_momentum: bool = True,
                 include_cross_entity: bool = True,
                 target_level: str = "criteria"):
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows
        self.include_momentum = include_momentum
        self.include_cross_entity = include_cross_entity
        self.target_level = target_level
        self.feature_names_: List[str] = []
    
    def fit_transform(self,
                      panel_data,
                      target_year: int,
                      use_saw_normalization: bool = False,
                      holdout_year: Optional[int] = None,
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
                    print(
                        f"    {sc}: {count}/{total} valid target years "
                        f"({total - count} missing → excluded from training)"
                    )

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

                if self.target_level == "criteria":
                    # Criteria mode: skip only if the entire target vector is NaN
                    if np.all(np.isnan(target)):
                        n_skipped_train += 1
                        continue
                else:
                    # Sub-criteria mode: strict — exclude any row with any NaN
                    if not has_complete_target(target):
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

            # ---- Prediction sample ----
            if ctx_pred_year is not None and entity not in ctx_pred_year.active_provinces:
                n_skipped_pred += 1
                continue  # Entity absent from target year → no prediction

            if last_year not in entity_data.index:
                n_skipped_pred += 1
                continue

            pred_features = self._create_features_safe(
                entity_data, entity, years, last_year, entities, panel_data
            )
            pred_feature_list.append(pred_features)
            pred_entities.append(entity)

        if n_skipped_train > 0:
            print(
                f"    Forecasting: {n_skipped_train} training observations "
                f"excluded (missing target values or inactive province-year pairs)"
            )
        if n_skipped_pred > 0:
            print(
                f"    Forecasting: {n_skipped_pred} province(s) excluded "
                f"from prediction (not in target-year active set)"
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

        # In criteria mode, fill partial NaN targets with per-column median.
        # When use_saw_normalization=True the median is also in [0, 1] since
        # all non-NaN SAW targets are bounded; imputation preserves the domain.
        if self.target_level == "criteria":
            nan_mask = np.isnan(y_train)
            if nan_mask.any():
                col_medians = np.nanmedian(y_train, axis=0)
                inds = np.where(nan_mask)
                y_train[inds] = np.take(col_medians, inds[1])

        # ---- Assemble holdout arrays ----
        if X_holdout_list:
            X_holdout = np.vstack(X_holdout_list)
            y_holdout = np.vstack(y_holdout_list)
            # Apply same partial-NaN imputation as training (criteria mode)
            if self.target_level == "criteria":
                ho_nan_mask = np.isnan(y_holdout)
                if ho_nan_mask.any():
                    # Use training medians — avoids leakage from holdout data
                    ho_col_medians = np.nanmedian(y_train, axis=0)
                    ho_inds = np.where(ho_nan_mask)
                    y_holdout[ho_inds] = np.take(ho_col_medians, ho_inds[1])
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
            {'entity_index': np.array(entity_indices, dtype=int)}
        )

        return X_train_df, y_train_df, X_pred_df, entity_index_df, X_holdout_df, y_holdout_df
    
    def _create_features_safe(
        self,
        entity_data: 'pd.DataFrame',
        entity: str,
        years: List[int],
        current_year: int,
        all_entities: List[str],
        panel_data,
    ) -> np.ndarray:
        """NaN-safe wrapper around :meth:`_create_features`.

        Calls ``_create_features`` and replaces any remaining NaN values
        (arising from missing lag years or absent cross-entity data) with
        ``0.0`` via :func:`data.missing_data.fill_missing_features`, encoding
        "no prior information" for the ML model without fabricating data.
        """
        features = self._create_features(
            entity_data, entity, years, current_year, all_entities, panel_data
        )
        return fill_missing_features(features)

    def _create_features(self,
                         entity_data: pd.DataFrame,
                         entity: str,
                         years: List[int],
                         current_year: int,
                         all_entities: List[str],
                         panel_data) -> np.ndarray:
        """Create feature vector for a single entity-year combination."""
        components = list(entity_data.columns)
        # Reindex to the full year sequence so that years when this entity was
        # absent from the source CSV produce NaN rows instead of KeyError in
        # subsequent .loc lookups.  NaN values in the resulting feature vector
        # are replaced with 0.0 by the caller (_create_features_safe).
        entity_data = entity_data.reindex(years)
        features = []
        feature_names = []
        
        # Current values
        current_values = entity_data.loc[current_year, components].values
        features.extend(current_values)
        feature_names.extend([f"{c}_current" for c in components])
        
        # Lag features
        # When there is insufficient history (year_idx < lag), use NaN rather
        # than current_values.  The previous fallback ``lag_values = current_values``
        # created a false "no-change" signal (lag feature == current feature) that
        # biased all models toward predicting zero change for early-history entities.
        # NaN is replaced with 0.0 by the _create_features_safe wrapper, encoding
        # "no prior information" without fabricating a spurious signal.
        year_idx = years.index(current_year)
        for lag in self.lag_periods:
            if year_idx - lag >= 0:
                lag_year = years[year_idx - lag]
                lag_values = entity_data.loc[lag_year, components].values
            else:
                lag_values = np.full(len(components), np.nan)  # no history → missing
            features.extend(lag_values)
            feature_names.extend([f"{c}_lag{lag}" for c in components])
        
        # Rolling statistics
        available_years = [y for y in years if y <= current_year]
        for window in self.rolling_windows:
            if len(available_years) >= window:
                window_years = available_years[-window:]
                window_data = entity_data.loc[window_years, components]
                
                # Mean
                features.extend(window_data.mean().values)
                feature_names.extend([f"{c}_roll{window}_mean" for c in components])
                
                # Std
                std_vals = window_data.std().fillna(0).values
                features.extend(std_vals)
                feature_names.extend([f"{c}_roll{window}_std" for c in components])
                
                # Min/Max
                features.extend(window_data.min().values)
                feature_names.extend([f"{c}_roll{window}_min" for c in components])
                features.extend(window_data.max().values)
                feature_names.extend([f"{c}_roll{window}_max" for c in components])
            else:
                # Pad with current values
                features.extend(current_values)
                feature_names.extend([f"{c}_roll{window}_mean" for c in components])
                features.extend(np.zeros(len(components)))
                feature_names.extend([f"{c}_roll{window}_std" for c in components])
                features.extend(current_values)
                feature_names.extend([f"{c}_roll{window}_min" for c in components])
                features.extend(current_values)
                feature_names.extend([f"{c}_roll{window}_max" for c in components])
        
        # Momentum features (rate of change)
        if self.include_momentum and year_idx > 0:
            prev_year = years[year_idx - 1]
            prev_values = entity_data.loc[prev_year, components].values
            momentum = current_values - prev_values
            features.extend(momentum)
            feature_names.extend([f"{c}_momentum" for c in components])
            
            # Acceleration (change in momentum)
            if year_idx > 1:
                prev_prev_year = years[year_idx - 2]
                prev_prev_values = entity_data.loc[prev_prev_year, components].values
                prev_momentum = prev_values - prev_prev_values
                acceleration = momentum - prev_momentum
                features.extend(acceleration)
                feature_names.extend([f"{c}_acceleration" for c in components])
            else:
                features.extend(np.zeros(len(components)))
                feature_names.extend([f"{c}_acceleration" for c in components])
        else:
            features.extend(np.zeros(len(components)))
            feature_names.extend([f"{c}_momentum" for c in components])
            features.extend(np.zeros(len(components)))
            feature_names.extend([f"{c}_acceleration" for c in components])
        
        # Trend feature (slope of linear fit over **valid** (non-NaN) years)
        if len(available_years) >= 2:
            for c in components:
                y_vals_raw = entity_data.loc[available_years, c].values.astype(float)
                x_vals_raw = np.arange(len(y_vals_raw), dtype=float)
                # Remove NaN years so polyfit receives a clean array
                valid_mask = ~np.isnan(y_vals_raw)
                if valid_mask.sum() >= 2:
                    slope = np.polyfit(
                        x_vals_raw[valid_mask], y_vals_raw[valid_mask], 1
                    )[0]
                else:
                    slope = 0.0
                features.append(slope)
                feature_names.append(f"{c}_trend")
        else:
            features.extend(np.zeros(len(components)))
            feature_names.extend([f"{c}_trend" for c in components])
        
        # Cross-entity features (relative position)
        if self.include_cross_entity:
            if self.target_level == "criteria":
                year_cross_section = panel_data.criteria_cross_section[current_year]
            else:
                year_cross_section = panel_data.cross_section[current_year]
            if 'Province' in year_cross_section.columns:
                year_cross_section = year_cross_section.set_index('Province')

            for c in components:
                if c in year_cross_section.columns:
                    entity_value = current_values[components.index(c)]
                    # Drop NaN from cross-section and guard against NaN entity value
                    col_values_clean = year_cross_section[c].dropna()
                    if len(col_values_clean) == 0 or np.isnan(entity_value):
                        # No cross-section data or entity value missing → neutral
                        features.extend([0.5, 0.0])
                    else:
                        percentile = float((col_values_clean < entity_value).mean())
                        mean_val   = float(col_values_clean.mean())
                        std_val    = float(col_values_clean.std())
                        z_score    = (entity_value - mean_val) / (std_val + 1e-10)
                        features.append(percentile)
                        features.append(z_score)
                    feature_names.append(f"{c}_percentile")
                    feature_names.append(f"{c}_zscore")
                else:
                    features.extend([0.5, 0.0])
                    feature_names.extend([f"{c}_percentile", f"{c}_zscore"])
        
        # Validate that every entity-year produces the same feature vector
        if self.feature_names_ and len(self.feature_names_) != len(feature_names):
            raise ValueError(
                f"Feature dimension mismatch: previous call produced "
                f"{len(self.feature_names_)} features, this call produced "
                f"{len(feature_names)}. Check cross_section data for "
                f"entity={entity}, year={current_year}."
            )
        self.feature_names_ = feature_names
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self.feature_names_
