# -*- coding: utf-8 -*-
"""Panel data loading with hierarchical structure and dynamic missing-data exclusion.

This module handles:
1. Loading multiple CSV files (one per year) from the data folder
2. Hierarchical structure: Subcriteria → Criteria → Final Score
3. **Dynamic exclusion**: missing provinces / sub-criteria are completely
   removed from each year's analysis rather than filled or imputed.
   - Province excluded for year Y  → not an alternative in MCDM or ML for Y
   - Sub-criterion excluded for Y  → not a column in the year's hierarchy
   - Criterion excluded for Y      → not a node in the ER aggregation tree
4. ``YearContext`` objects record exactly what was active vs. excluded per year.
5. Composite calculation at each hierarchy level (using only active sub-criteria)

Notes
-----
The dataset uses NaN (not zero) to represent missing observations.  A value of
exactly 0.0 is treated as a legitimate governance score.  All missing-data
detection therefore relies on ``pd.notna()`` / ``pd.isnull()`` rather than
comparisons against zero.

Downstream consumers **must** use ``panel_data.year_contexts[year]`` to
discover the active set of provinces / sub-criteria for each year instead of
assuming a fixed panel dimension.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

try:
    from .config import Config, get_config
    from .loggers import get_logger
except ImportError:
    from config import Config, get_config
    from logging import getLogger as get_logger


@dataclass
class HierarchyMapping:
    """Mapping between subcriteria and criteria."""
    subcriteria_to_criteria: Dict[str, str]  # SC11 -> C01
    criteria_to_subcriteria: Dict[str, List[str]]  # C01 -> [SC11, SC12, SC13, SC14]
    criteria_names: Dict[str, str]  # C01 -> "Participation"
    subcriteria_names: Dict[str, str]  # SC11 -> "Civic Knowledge"
    
    @property
    def all_subcriteria(self) -> List[str]:
        return sorted(self.subcriteria_to_criteria.keys())
    
    @property
    def all_criteria(self) -> List[str]:
        return sorted(self.criteria_to_subcriteria.keys())


@dataclass
class YearContext:
    """Per-year data availability for dynamic exclusion.

    Tracks which provinces and sub-criteria have *valid* (non-NaN) data for a
    specific year so that MCDM, weighting and ML components can exclude missing
    entities entirely instead of filling or imputing values.

    Attributes
    ----------
    year : int
        The calendar year this context describes.
    active_provinces : list of str
        Provinces with **at least one** valid sub-criterion entry.  These are
        the only alternatives used in MCDM ranking and ML forecasting for this
        year.
    active_subcriteria : list of str
        Sub-criteria columns with **at least one** valid province value.
        Completely missing SCs are dropped from the hierarchy for this year.
    active_criteria : list of str
        Criteria that have **at least one** active sub-criterion.
    excluded_provinces : list of str
        Provinces whose every sub-criterion is NaN this year.
    excluded_subcriteria : list of str
        Sub-criteria whose every province value is NaN this year.
    excluded_criteria : list of str
        Criteria where every sub-criterion is excluded this year.
    criterion_alternatives : dict
        ``{criterion_id: [province_list]}`` — per-criterion province sets that
        have **complete** valid data across **all** active SCs for that
        criterion.  This guarantees a NaN-free decision matrix per criterion.
    criterion_subcriteria : dict
        ``{criterion_id: [sc_list]}`` — per-criterion active (non-missing) SCs
        after global SC exclusion.
    valid_pairs : set of (province, sc) tuples
        Fine-grained per-cell availability used by ML forecasting to build
        per-sub-criterion temporal series without missing-year gaps.
    """
    year: int
    active_provinces: List[str]
    active_subcriteria: List[str]
    active_criteria: List[str]
    excluded_provinces: List[str]
    excluded_subcriteria: List[str]
    excluded_criteria: List[str]
    criterion_alternatives: Dict[str, List[str]]
    criterion_subcriteria: Dict[str, List[str]]
    valid_pairs: Set[Tuple[str, str]]

    def is_valid(self, province: str, sc: str) -> bool:
        """Return True if ``(province, sc)`` has valid data this year."""
        return (province, sc) in self.valid_pairs

    def describe(self) -> str:
        """Human-readable summary of this year's data availability."""
        n_total_prov = len(self.active_provinces) + len(self.excluded_provinces)
        n_total_sc   = len(self.active_subcriteria) + len(self.excluded_subcriteria)
        lines = [
            f"Year {self.year} — Data Availability:",
            f"  Provinces   : {len(self.active_provinces)}/{n_total_prov}"
            + (f"  [excluded: {', '.join(self.excluded_provinces)}]"
               if self.excluded_provinces else "  [all present]"),
            f"  Subcriteria : {len(self.active_subcriteria)}/{n_total_sc}"
            + (f"  [excluded: {', '.join(self.excluded_subcriteria)}]"
               if self.excluded_subcriteria else "  [all present]"),
            f"  Criteria    : {len(self.active_criteria)}/"
            f"{len(self.active_criteria) + len(self.excluded_criteria)}"
            + (f"  [excluded: {', '.join(self.excluded_criteria)}]"
               if self.excluded_criteria else "  [all present]"),
        ]
        return "\n".join(lines)


@dataclass
class PanelData:
    """Container for hierarchical panel data with multiple views."""
    # Raw subcriteria data
    subcriteria_long: pd.DataFrame  # Long format: (n*T) × (2 + K_sub)
    subcriteria_cross_section: Dict[int, pd.DataFrame]  # Year → subcriteria data

    # Aggregated criteria data (calculated from subcriteria)
    criteria_long: pd.DataFrame  # Long format: (n*T) × (2 + K_criteria)
    criteria_cross_section: Dict[int, pd.DataFrame]  # Year → criteria data

    # Final scores (calculated from criteria)
    final_long: pd.DataFrame  # Long format: (n*T) × 3 (Year, Province, FinalScore)
    final_cross_section: Dict[int, pd.DataFrame]  # Year → final scores

    # Metadata
    provinces: List[str]
    years: List[int]
    hierarchy: HierarchyMapping

    # --- Dynamic exclusion context (one per year) ---
    # Each YearContext records exactly which provinces / sub-criteria / criteria
    # are active vs. excluded for that year.  All downstream components MUST
    # consult this context rather than assuming a fixed panel size.
    year_contexts: Dict[int, 'YearContext'] = field(default_factory=dict)

    # Legacy availability dict (kept for backward-compat; prefer year_contexts)
    availability: Dict = field(default_factory=dict)

    @property
    def n_provinces(self) -> int:
        return len(self.provinces)

    @property
    def n_years(self) -> int:
        return len(self.years)

    @property
    def n_subcriteria(self) -> int:
        return len(self.hierarchy.all_subcriteria)

    @property
    def n_criteria(self) -> int:
        return len(self.hierarchy.all_criteria)

    @property
    def subcriteria_names(self) -> List[str]:
        """List of all subcriteria names from the hierarchy."""
        return self.hierarchy.all_subcriteria

    @property
    def cross_section(self) -> Dict:
        """Alias for subcriteria_cross_section (used by MCDM/forecasting)."""
        return self.subcriteria_cross_section

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_province(self, province: str) -> pd.DataFrame:
        """Get a province's subcriteria data across all years, indexed by year.

        Returns a DataFrame indexed by year.  Cells may be NaN for years where
        that (province, sub-criterion) pair had no data.  ML components should
        use :meth:`get_valid_temporal_series` or consult
        ``year_contexts[year].is_valid(province, sc)`` to skip missing cells.
        """
        long = self.subcriteria_long
        prov_data = long[long['Province'] == province].copy()
        prov_data = prov_data.set_index('Year')
        cols = [c for c in prov_data.columns if c != 'Province']
        return prov_data[cols]

    def get_valid_temporal_series(self, province: str, sc: str) -> pd.Series:
        """Return the temporal series for *one* (province, SC) pair.

        Only includes years where that cell has valid data (as recorded in
        ``year_contexts``).  Years with NaN are silently excluded so the ML
        model for this SC sees only its true temporal observation count.

        Returns a :class:`pd.Series` indexed by year (may be shorter than
        ``self.years`` if data is missing).
        """
        result = {}
        for year in self.years:
            ctx = self.year_contexts.get(year)
            if ctx is not None:
                if ctx.is_valid(province, sc):
                    cs = self.subcriteria_cross_section[year]
                    if province in cs.index and sc in cs.columns:
                        val = cs.loc[province, sc]
                        if pd.notna(val):
                            result[year] = float(val)
            else:
                # No context: fall back to NaN check
                cs = self.subcriteria_cross_section.get(year)
                if cs is not None and province in cs.index and sc in cs.columns:
                    val = cs.loc[province, sc]
                    if pd.notna(val):
                        result[year] = float(val)
        return pd.Series(result, dtype=float)

    def get_criterion_matrix(self, year: int, criterion: str) -> pd.DataFrame:
        """Return the **clean** (zero-NaN) decision matrix for one criterion-year.

        Uses the :class:`YearContext` to select only the provinces and
        sub-criteria that have *complete* valid data for this criterion in the
        requested year.  The returned DataFrame is guaranteed to be NaN-free
        and ready for MCDM/weighting computation.

        Parameters
        ----------
        year : int
        criterion : str
            Criterion code, e.g. ``'C01'``.

        Returns
        -------
        pd.DataFrame
            Shape ``(n_active_alternatives, n_active_subcriteria)``; may be
            empty if no provinces or SCs are available.
        """
        ctx = self.year_contexts.get(year)
        if ctx is None:
            # Fallback: old-style NaN filtering
            cs = self.subcriteria_cross_section.get(year, pd.DataFrame())
            sc_cols = self.hierarchy.criteria_to_subcriteria.get(criterion, [])
            sc_cols = [c for c in sc_cols if c in cs.columns]
            if not sc_cols:
                return pd.DataFrame()
            sub = cs[sc_cols].copy()
            sub = sub[sub.notna().any(axis=1)]  # drop all-NaN rows
            sub = sub.loc[:, sub.notna().any(axis=0)]  # drop all-NaN cols
            return sub.dropna()  # drop remaining NaN rows

        sc_cols  = ctx.criterion_subcriteria.get(criterion, [])
        provinces = ctx.criterion_alternatives.get(criterion, [])
        if not sc_cols or not provinces:
            return pd.DataFrame()

        cs = self.subcriteria_cross_section.get(year, pd.DataFrame())
        avail_provs = [p for p in provinces if p in cs.index]
        avail_scs   = [c for c in sc_cols  if c in cs.columns]
        if not avail_provs or not avail_scs:
            return pd.DataFrame()

        mat = cs.loc[avail_provs, avail_scs]
        # Defensive: drop any rows that are still NaN (should not happen
        # given the YearContext guarantees, but safety-nets never hurt)
        mat = mat.dropna()
        return mat

    def get_subcriteria_year(self, year: int) -> pd.DataFrame:
        """Get subcriteria data for specific year."""
        return self.subcriteria_cross_section[year]

    def get_criteria_year(self, year: int) -> pd.DataFrame:
        """Get criteria data for specific year."""
        return self.criteria_cross_section[year]

    def get_final_year(self, year: int) -> pd.DataFrame:
        """Get final scores for specific year."""
        return self.final_cross_section[year]

    def get_latest(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get latest year data (subcriteria, criteria, final)."""
        latest_year = max(self.years)
        return (
            self.subcriteria_cross_section[latest_year],
            self.criteria_cross_section[latest_year],
            self.final_cross_section[latest_year]
        )


class DataLoader:
    """Loads panel data from year-based CSV files with hierarchical structure."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.logger = get_logger()
    
    def load(self) -> PanelData:
        """Load panel data from data folder."""
        data_dir = self.config.paths.data_dir
        
        self.logger.info(f"Loading data from {data_dir}")
        
        # Load hierarchy mapping from codebook
        hierarchy = self._load_hierarchy_mapping(data_dir)
        
        # Load yearly data files
        yearly_data = self._load_yearly_files(data_dir, hierarchy)
        
        # Create hierarchical views
        panel_data = self._create_hierarchical_views(yearly_data, hierarchy)
        
        self.logger.info(f"[OK] Loaded: {panel_data.n_provinces} provinces (global), "
                        f"{panel_data.n_years} years, "
                        f"{panel_data.n_subcriteria} subcriteria, "
                        f"{panel_data.n_criteria} criteria")
        # Log per-year active counts
        for year in panel_data.years:
            ctx = panel_data.year_contexts.get(year)
            if ctx and (ctx.excluded_provinces or ctx.excluded_subcriteria):
                self.logger.info(
                    f"  {year}: {len(ctx.active_provinces)} provinces, "
                    f"{len(ctx.active_subcriteria)} subcriteria "
                    f"(excl. {len(ctx.excluded_provinces)} prov, "
                    f"{len(ctx.excluded_subcriteria)} SC)"
                )
        
        return panel_data
    
    def _load_hierarchy_mapping(self, data_dir: Path) -> HierarchyMapping:
        """Load hierarchy mapping from codebook files."""
        codebook_dir = data_dir / "codebook"
        
        # Load subcriteria codebook
        subcriteria_file = codebook_dir / "codebook_subcriteria.csv"
        if not subcriteria_file.exists():
            raise FileNotFoundError(f"Subcriteria codebook not found: {subcriteria_file}")
        
        sub_df = pd.read_csv(subcriteria_file)
        
        # Load criteria codebook
        criteria_file = codebook_dir / "codebook_criteria.csv"
        if not criteria_file.exists():
            raise FileNotFoundError(f"Criteria codebook not found: {criteria_file}")
        
        crit_df = pd.read_csv(criteria_file)
        
        # Build mappings
        subcriteria_to_criteria = dict(zip(sub_df['Variable_Code'], sub_df['Criteria_Code']))
        subcriteria_names = dict(zip(sub_df['Variable_Code'], sub_df['Variable_Name']))
        criteria_names = dict(zip(crit_df['Variable_Code'], crit_df['Variable_Name']))
        
        # Build reverse mapping
        criteria_to_subcriteria = {}
        for sc, c in subcriteria_to_criteria.items():
            if c not in criteria_to_subcriteria:
                criteria_to_subcriteria[c] = []
            criteria_to_subcriteria[c].append(sc)
        
        # Sort subcriteria within each criterion
        for c in criteria_to_subcriteria:
            criteria_to_subcriteria[c] = sorted(criteria_to_subcriteria[c])
        
        self.logger.info(f"[OK] Loaded hierarchy: {len(criteria_to_subcriteria)} criteria, "
                        f"{len(subcriteria_to_criteria)} subcriteria")
        
        return HierarchyMapping(
            subcriteria_to_criteria=subcriteria_to_criteria,
            criteria_to_subcriteria=criteria_to_subcriteria,
            criteria_names=criteria_names,
            subcriteria_names=subcriteria_names
        )
    
    def _load_yearly_files(self, data_dir: Path, hierarchy: HierarchyMapping) -> Dict[int, pd.DataFrame]:
        """Load all yearly CSV files from data directory."""
        yearly_data = {}
        
        # Find all year CSV files
        year_files = sorted(data_dir.glob("[0-9][0-9][0-9][0-9].csv"))
        
        if not year_files:
            raise FileNotFoundError(f"No yearly CSV files found in {data_dir}")
        
        for year_file in year_files:
            # Extract year from filename
            year = int(year_file.stem)
            
            # Load CSV
            df = pd.read_csv(year_file)
            
            # Validate structure
            if 'Province' not in df.columns:
                raise ValueError(f"'Province' column not found in {year_file}")
            
            # Check for subcriteria columns
            expected_subcriteria = hierarchy.all_subcriteria
            missing_cols = set(expected_subcriteria) - set(df.columns)
            if missing_cols:
                self.logger.warning(f"Year {year}: Missing subcriteria columns: {missing_cols}")
            
            # Add Year column
            df.insert(0, 'Year', year)
            
            yearly_data[year] = df
            self.logger.info(f"  Loaded {year}: {len(df)} provinces, {len(df.columns)-2} subcriteria")
        
        return yearly_data
    
    def _create_hierarchical_views(
        self,
        yearly_data: Dict[int, pd.DataFrame],
        hierarchy: HierarchyMapping
    ) -> PanelData:
        """Create hierarchical panel data with dynamic exclusion via YearContext.

        For each year:
        1. Build a ``YearContext`` that records *exactly* which provinces and
           sub-criteria are active (have at least one valid value) versus
           completely absent (all-NaN).
        2. Per criterion, further restricts to provinces that have **complete**
           valid data for all active SCs in that criterion — guaranteeing a
           NaN-free decision matrix for MCDM.
        3. Cross-sections and long-format frames preserve raw NaN values so
           that the full temporal history is queryable; consumers must call
           ``get_criterion_matrix()`` or consult ``year_contexts`` to obtain
           clean subsets.
        """
        years = sorted(yearly_data.keys())

        # Collect all provinces across all years (for the global province list)
        all_provinces: set = set()
        for df in yearly_data.values():
            all_provinces.update(df['Province'].unique())
        provinces = sorted(all_provinces)

        # ---- Containers ----
        subcriteria_data: List[pd.DataFrame] = []
        criteria_data:    List[pd.DataFrame] = []
        final_data:       List[pd.DataFrame] = []

        subcriteria_cross: Dict[int, pd.DataFrame] = {}
        criteria_cross:    Dict[int, pd.DataFrame] = {}
        final_cross:       Dict[int, pd.DataFrame] = {}

        year_contexts: Dict[int, YearContext] = {}

        # Legacy availability dict (kept for backward-compat)
        availability: Dict = {
            'province_by_year': {},
            'subcriteria_by_year': {},
            'criteria_by_year': {},
        }

        for year in years:
            df_year = yearly_data[year]
            subcriteria_cols = [c for c in hierarchy.all_subcriteria
                                if c in df_year.columns]
            df_sub = df_year[['Year', 'Province'] + subcriteria_cols].copy()

            # ==============================================================
            # Build YearContext
            # ==============================================================

            # Step 1 — Province exclusion: ALL sub-criteria NaN → excluded
            province_any_valid = df_sub[subcriteria_cols].notna().any(axis=1)
            active_provs   = df_sub.loc[province_any_valid,  'Province'].tolist()
            excluded_provs = df_sub.loc[~province_any_valid, 'Province'].tolist()

            # Step 2 — SC exclusion: ALL provinces NaN → SC excluded for year
            df_sub_idx = df_sub.set_index('Province')[subcriteria_cols]
            sc_any_valid  = df_sub_idx.notna().any(axis=0)
            active_scs    = sc_any_valid[sc_any_valid].index.tolist()
            excluded_scs  = sc_any_valid[~sc_any_valid].index.tolist()

            # Step 3 — Per-criterion: derive active SCs and clean province sets
            criterion_alts_map: Dict[str, List[str]] = {}
            criterion_scs_map:  Dict[str, List[str]] = {}

            df_active_prov = df_sub_idx.loc[
                [p for p in active_provs if p in df_sub_idx.index]
            ]

            for crit_id in hierarchy.all_criteria:
                all_scs_in_crit = hierarchy.criteria_to_subcriteria[crit_id]
                # Keep only globally-active SCs for this criterion
                crit_active_scs = [sc for sc in all_scs_in_crit
                                   if sc in active_scs]
                criterion_scs_map[crit_id] = crit_active_scs

                if not crit_active_scs:
                    criterion_alts_map[crit_id] = []
                    continue

                # Province participates in this criterion iff it has valid
                # data for *all* active SCs → guarantees NaN-free decision matrix
                avail_scs_in_df = [sc for sc in crit_active_scs
                                   if sc in df_active_prov.columns]
                if not avail_scs_in_df:
                    criterion_alts_map[crit_id] = []
                    continue

                prov_complete = (
                    df_active_prov[avail_scs_in_df].notna().all(axis=1)
                )
                criterion_alts_map[crit_id] = (
                    prov_complete[prov_complete].index.tolist()
                )

            # Step 4 — Active criteria: has at least one active SC
            active_criteria_list = [
                c for c in hierarchy.all_criteria
                if criterion_scs_map.get(c)
            ]
            excluded_criteria_list = [
                c for c in hierarchy.all_criteria
                if not criterion_scs_map.get(c)
            ]

            # Step 5 — Fine-grained valid pairs for ML temporal filtering
            valid_pairs: Set[Tuple[str, str]] = set()
            for prov in active_provs:
                if prov not in df_sub_idx.index:
                    continue
                for sc in active_scs:
                    if sc in df_sub_idx.columns:
                        val = df_sub_idx.loc[prov, sc]
                        if pd.notna(val):
                            valid_pairs.add((prov, sc))

            ctx = YearContext(
                year=year,
                active_provinces=active_provs,
                active_subcriteria=active_scs,
                active_criteria=active_criteria_list,
                excluded_provinces=excluded_provs,
                excluded_subcriteria=excluded_scs,
                excluded_criteria=excluded_criteria_list,
                criterion_alternatives=criterion_alts_map,
                criterion_subcriteria=criterion_scs_map,
                valid_pairs=valid_pairs,
            )
            year_contexts[year] = ctx

            # Log significant exclusions
            if excluded_provs:
                self.logger.info(
                    f"  Year {year}: excluded {len(excluded_provs)} province(s) "
                    f"(all-NaN): {excluded_provs}"
                )
            if excluded_scs:
                self.logger.info(
                    f"  Year {year}: excluded {len(excluded_scs)} sub-criterion(a) "
                    f"(all-NaN): {excluded_scs}"
                )
            if excluded_criteria_list:
                self.logger.info(
                    f"  Year {year}: excluded {len(excluded_criteria_list)} "
                    f"criteria (no valid SCs): {excluded_criteria_list}"
                )

            # Legacy availability
            availability['province_by_year'][year]   = active_provs
            availability['subcriteria_by_year'][year] = active_scs

            # ==============================================================
            # Criteria composite (mean of active SCs per province per criterion)
            # ==============================================================
            criteria_values: Dict[str, pd.Series] = {}
            criteria_availability_yr: Dict[str, List[str]] = {}

            for crit_id, crit_active_scs in criterion_scs_map.items():
                if not crit_active_scs:
                    criteria_values[crit_id] = pd.Series(
                        np.nan, index=df_sub.index)
                    criteria_availability_yr[crit_id] = []
                    continue

                avail_scs_in_df = [sc for sc in crit_active_scs
                                   if sc in df_sub.columns]
                criterion_scores = []
                provinces_with_crit = []

                for idx, row in df_sub.iterrows():
                    prov = row['Province']
                    sub_vals = [row[sc] for sc in avail_scs_in_df]
                    valid_vals = [v for v in sub_vals if pd.notna(v)]
                    if valid_vals:
                        criterion_scores.append(float(np.mean(valid_vals)))
                        provinces_with_crit.append(prov)
                    else:
                        criterion_scores.append(np.nan)

                criteria_values[crit_id] = pd.Series(
                    criterion_scores, index=df_sub.index)
                criteria_availability_yr[crit_id] = provinces_with_crit

            availability['criteria_by_year'][year] = criteria_availability_yr

            df_criteria = pd.DataFrame({
                'Year': df_sub['Year'],
                'Province': df_sub['Province'],
                **criteria_values,
            })

            # ==============================================================
            # Final composite (mean of criteria composites, skip NaN)
            # ==============================================================
            final_scores_list = []
            for _, row in df_criteria.iterrows():
                crit_vals = [
                    row[c] for c in hierarchy.all_criteria
                    if c in df_criteria.columns and pd.notna(row[c])
                ]
                final_scores_list.append(
                    float(np.mean(crit_vals)) if crit_vals else np.nan
                )

            df_final = pd.DataFrame({
                'Year': df_criteria['Year'],
                'Province': df_criteria['Province'],
                'FinalScore': final_scores_list,
            })

            # ---- Append to long format ----
            subcriteria_data.append(df_sub)
            criteria_data.append(df_criteria)
            final_data.append(df_final)

            # ---- Cross-sections (NaN cells preserved for history) ----
            subcriteria_cross[year] = df_sub_idx  # already indexed by Province
            criteria_cross[year]    = (
                df_criteria.set_index('Province')[hierarchy.all_criteria]
            )
            final_cross[year] = df_final.set_index('Province')[['FinalScore']]

        # ------------------------------------------------------------------
        # Concatenate long-format frames
        # ------------------------------------------------------------------
        subcriteria_long = pd.concat(subcriteria_data, ignore_index=True)
        criteria_long    = pd.concat(criteria_data,    ignore_index=True)
        final_long       = pd.concat(final_data,       ignore_index=True)

        return PanelData(
            subcriteria_long=subcriteria_long,
            subcriteria_cross_section=subcriteria_cross,
            criteria_long=criteria_long,
            criteria_cross_section=criteria_cross,
            final_long=final_long,
            final_cross_section=final_cross,
            provinces=provinces,
            years=years,
            hierarchy=hierarchy,
            year_contexts=year_contexts,
            availability=availability,
        )


def load_data() -> PanelData:
    """Convenience function to load panel data."""
    loader = DataLoader()
    return loader.load()
