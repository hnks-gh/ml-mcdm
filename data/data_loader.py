# -*- coding: utf-8 -*-
"""
Hierarchical Panel Data Loader
==============================

This module implements the loading and structural initialization of the 
ML-MCDM panel dataset. It coordinates the ingestion of multi-year CSV files 
and the associated hierarchical mapping between sub-criteria and criteria.

Key Capabilities:
-----------------
1.  CSV Ingestion: Loads yearly observations from tabular sources.
2.  Hierarchy Mapping: Links sub-criteria to parent criteria groups.
3.  Dynamic Exclusion: Calibrates active analytical scopes per year via YearContext.
4.  View Generation: Assembles long-form and cross-sectional data views.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

try:
    from config import Config, get_config
    from loggers import get_logger
except ImportError:
    from logging import getLogger as get_logger  # type: ignore[assignment]

    def get_config():  # type: ignore[misc]
        raise RuntimeError(
            "Config is not available.  Ensure the project root is on sys.path."
        )


@dataclass
class HierarchyMapping:
    """
    Mapping between sub-criteria and criteria levels of the hierarchy.

    Attributes
    ----------
    subcriteria_to_criteria : Dict[str, str]
        Direct mapping from sub-criterion code (e.g., 'SC11') to parent 
        criterion code (e.g., 'C01').
    criteria_to_subcriteria : Dict[str, List[str]]
        Inverse mapping from criterion code to its child sub-criteria.
    criteria_names : Dict[str, str]
        Human-readable labels for each criterion.
    subcriteria_names : Dict[str, str]
        Human-readable labels for each sub-criterion.
    """

    @property
    def all_subcriteria(self) -> List[str]:
        return sorted(self.subcriteria_to_criteria.keys())

    @property
    def all_criteria(self) -> List[str]:
        return sorted(self.criteria_to_subcriteria.keys())


@dataclass
class YearContext:
    """
    Year-specific data availability and exclusion contexts.

    Tracks active provinces, sub-criteria, and criteria for a specific year 
    in the panel. Enables dynamic exclusion to handle structural gaps 
    and missing data without biasing calculations.

    Attributes
    ----------
    year : int
        The calendar year.
    active_provinces : List[str]
        Provinces with at least one valid observation.
    active_subcriteria : List[str]
        Sub-criteria with at least one valid observation.
    active_criteria : List[str]
        Criteria with at least one active sub-criterion.
    excluded_provinces : List[str]
        Provinces entirely missing data for this year.
    excluded_subcriteria : List[str]
        Sub-criteria entirely missing data for this year.
    excluded_criteria : List[str]
        Criteria with no active child sub-criteria.
    criterion_alternatives : Dict[str, List[str]]
        Map of criteria to provinces that have complete data for all active 
        sub-criteria in that group.
    criterion_subcriteria : Dict[str, List[str]]
        Map of criteria to their active child sub-criteria.
    valid_pairs : Set[Tuple[str, str]]
        Set of (province, sub-criterion) pairs with valid data.
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
    """
    Comprehensive container for hierarchical panel data.

    Maintains multiple views of the dataset:
    - Long-form DataFrames for sub-criteria, criteria, and final scores.
    - Yearly cross-sections for matrix-style analysis.
    - Dynamic YearContext objects for NaN-aware processing.
    """
    # Raw subcriteria data
    subcriteria_long: pd.DataFrame              # Long format: (n*T) × (2 + K_sub)
    subcriteria_cross_section: Dict[int, pd.DataFrame]  # Year → subcriteria data

    # Aggregated criteria data (calculated from subcriteria)
    criteria_long: pd.DataFrame                 # Long format: (n*T) × (2 + K_criteria)
    criteria_cross_section: Dict[int, pd.DataFrame]     # Year → criteria data

    # Final scores (calculated from criteria)
    final_long: pd.DataFrame                    # Long format: (n*T) × 3
    final_cross_section: Dict[int, pd.DataFrame]        # Year → final scores

    # Metadata
    provinces: List[str]
    years: List[int]
    hierarchy: HierarchyMapping

    # Dynamic exclusion context (one per year)
    # All downstream components MUST consult this instead of assuming a fixed
    # panel size.
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
    def criteria_names(self) -> List[str]:
        """List of all criteria codes (C01–C08) from the hierarchy."""
        return self.hierarchy.all_criteria

    @property
    def cross_section(self) -> Dict:
        """Alias for subcriteria_cross_section (used by MCDM/forecasting)."""
        return self.subcriteria_cross_section

    @property
    def subcriteria(self) -> pd.DataFrame:
        """Alias for subcriteria_long (used by forecasting features)."""
        return self.subcriteria_long

    @property
    def criteria(self) -> pd.DataFrame:
        """Alias for criteria_long (used by forecasting features)."""
        return self.criteria_long

    @property
    def final(self) -> pd.DataFrame:
        """Alias for final_long (used by forecasting features)."""
        return self.final_long

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_province(self, province: str) -> pd.DataFrame:
        """
        Retrieve sub-criteria data for a specific province across all years.

        Parameters
        ----------
        province : str
            The name of the province.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by Year with sub-criteria columns.
        """
        long = self.subcriteria_long
        prov_data = long[long['Province'] == province].copy()
        prov_data = prov_data.set_index('Year')
        cols = [c for c in prov_data.columns if c != 'Province']
        return prov_data[cols]

    def get_province_criteria(self, province: str) -> pd.DataFrame:
        """Get a province's criteria composite scores across all years.

        Returns a DataFrame indexed by year with columns C01–C08.
        Cells may be NaN for years where the criterion had no active SCs.
        """
        long = self.criteria_long
        prov_data = long[long['Province'] == province].copy()
        prov_data = prov_data.set_index('Year')
        cols = [c for c in prov_data.columns if c not in ('Province', 'Year')]
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
        """
        Extract a clean (NaN-free) decision matrix for a specific criterion-year.

        Consults the YearContext to identify the intersection of active 
        provinces and sub-criteria that provide complete data for the 
        analytical phase.

        Parameters
        ----------
        year : int
            The analysis year.
        criterion : str
            The criterion code (e.g., 'C01').

        Returns
        -------
        pd.DataFrame
            A clean decision matrix ready for weighting or ranking.
        """
        ctx = self.year_contexts.get(year)
        if ctx is None:
            # Fallback: old-style NaN filtering (no YearContext available)
            cs = self.subcriteria_cross_section.get(year, pd.DataFrame())
            sc_cols = self.hierarchy.criteria_to_subcriteria.get(criterion, [])
            sc_cols = [c for c in sc_cols if c in cs.columns]
            if not sc_cols:
                return pd.DataFrame()
            sub = cs[sc_cols].copy()
            sub = sub[sub.notna().any(axis=1)]      # drop all-NaN rows
            sub = sub.loc[:, sub.notna().any(axis=0)]  # drop all-NaN cols
            return sub.dropna()                     # drop remaining partial-NaN rows

        sc_cols   = ctx.criterion_subcriteria.get(criterion, [])
        provinces = ctx.criterion_alternatives.get(criterion, [])
        if not sc_cols or not provinces:
            import logging
            logging.getLogger('ml_mcdm').warning(
                f"[DEBUG] get_criterion_matrix({criterion}, {year}): "
                f"EMPTY (sc_cols={len(sc_cols)}, provinces={len(provinces)})"
            )
            return pd.DataFrame()

        cs = self.subcriteria_cross_section.get(year, pd.DataFrame())
        avail_provs = [p for p in provinces if p in cs.index]
        avail_scs   = [c for c in sc_cols   if c in cs.columns]
        if not avail_provs or not avail_scs:
            import logging
            logging.getLogger('ml_mcdm').warning(
                f"[DEBUG] get_criterion_matrix({criterion}, {year}): "
                f"No data found (avail_provs={len(avail_provs)}, avail_scs={avail_scs})"
            )
            return pd.DataFrame()

        mat = cs.loc[avail_provs, avail_scs]
        # Defensive: drop any rows that still contain NaN (should not occur
        # given the YearContext guarantees, but guards against edge cases).
        mat = mat.dropna()
        if mat.empty:
            import logging
            logging.getLogger('ml_mcdm').warning(
                f"[DEBUG] get_criterion_matrix({criterion}, {year}): "
                f"Matrix was empty after NaN removal"
            )
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
            self.final_cross_section[latest_year],
        )


class DataLoader:
    """Load panel data from year-based CSV files with hierarchical structure.

    Reads ``data/csv/YYYY.csv`` files and the codebook from ``data/codebook/``
    and assembles a :class:`PanelData` object with per-year
    :class:`YearContext` objects for downstream NaN-aware processing.
    """

    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = get_logger('ml_mcdm.data_loader')

    def load(self) -> PanelData:
        """
        Execute the full data loading and assembly process.

        Loads hierarchy mappings, imports yearly CSV files, and creates 
        hierarchical views with associated YearContexts.

        Returns
        -------
        PanelData
            The complete, structured panel dataset.
        """
        data_dir = self.config.paths.data_dir
        csv_dir  = self.config.paths.data_csv_dir

        self.logger.info(f"Loading data from {csv_dir}")

        hierarchy    = self._load_hierarchy_mapping(data_dir)
        yearly_data  = self._load_yearly_files(csv_dir, hierarchy)
        panel_data   = self._create_hierarchical_views(yearly_data, hierarchy)

        self.logger.info(
            f"[OK] Loaded: {panel_data.n_provinces} provinces (global), "
            f"{panel_data.n_years} years, "
            f"{panel_data.n_subcriteria} subcriteria, "
            f"{panel_data.n_criteria} criteria"
        )
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_hierarchy_mapping(self, data_dir: Path) -> HierarchyMapping:
        """
        Extract the hierarchical structure from codebook CSV files.

        Parameters
        ----------
        data_dir : Path
            The base directory containing the codebook/ folder.

        Returns
        -------
        HierarchyMapping
            The linked mapping of sub-criteria and criteria.
        """
        codebook_dir = data_dir / "codebook"

        subcriteria_file = codebook_dir / "codebook_subcriteria.csv"
        if not subcriteria_file.exists():
            raise FileNotFoundError(
                f"Subcriteria codebook not found: {subcriteria_file}\n"
                "Expected layout: data/codebook/codebook_subcriteria.csv"
            )

        criteria_file = codebook_dir / "codebook_criteria.csv"
        if not criteria_file.exists():
            raise FileNotFoundError(
                f"Criteria codebook not found: {criteria_file}\n"
                "Expected layout: data/codebook/codebook_criteria.csv"
            )

        sub_df  = pd.read_csv(subcriteria_file, dtype=str).fillna("")
        crit_df = pd.read_csv(criteria_file,    dtype=str).fillna("")

        # Validate required columns
        for col in ("Variable_Code", "Criteria_Code", "Variable_Name"):
            if col not in sub_df.columns:
                raise ValueError(
                    f"Column '{col}' missing from {subcriteria_file}. "
                    f"Available: {sub_df.columns.tolist()}"
                )
        for col in ("Variable_Code", "Variable_Name"):
            if col not in crit_df.columns:
                raise ValueError(
                    f"Column '{col}' missing from {criteria_file}. "
                    f"Available: {crit_df.columns.tolist()}"
                )

        # Strip whitespace from all code/name columns
        for df in (sub_df, crit_df):
            for col in df.select_dtypes("object").columns:
                df[col] = df[col].str.strip()

        subcriteria_to_criteria = dict(
            zip(sub_df['Variable_Code'], sub_df['Criteria_Code'])
        )
        subcriteria_names = dict(
            zip(sub_df['Variable_Code'], sub_df['Variable_Name'])
        )
        criteria_names = dict(
            zip(crit_df['Variable_Code'], crit_df['Variable_Name'])
        )

        # ── SC52 Handling (Phase 1): Year-Interactive Inclusion ──
        # SC52 was discontinued as of 2021 but has valid data for 2011-2020.
        # We keep SC52 in the hierarchy and let YearContext handle year-by-year
        # exclusion based on data availability. This ensures:
        # - SC52 participates in MCDM for years 2011-2020 (data-driven)
        # - SC52 is excluded for years 2021-2024 (no data)
        # - Visualizations show SC52 with historical data + missing recent years
        self.logger.info(
            f"[PHASE 1] SC52 included in hierarchy; will be year-actively "
            f"excluded for 2021-2024 (data-driven by YearContext)"
        )

        criteria_to_subcriteria: Dict[str, List[str]] = {}
        for sc, c in subcriteria_to_criteria.items():
            if not sc or not c:
                continue  # skip blank rows
            criteria_to_subcriteria.setdefault(c, []).append(sc)
        for c in criteria_to_subcriteria:
            criteria_to_subcriteria[c] = sorted(criteria_to_subcriteria[c])

        # Validate: C05 should have 4 SCs (SC51, SC52, SC53, SC54)
        c05_scs = criteria_to_subcriteria.get("C05", [])
        if len(c05_scs) != 4 or "SC52" not in c05_scs:
            raise ValueError(
                f"[PHASE 1 VALIDATION FAILED] C05 should have 4 SCs (including SC52), "
                f"but got {len(c05_scs)}: {c05_scs}"
            )

        self.logger.info(
            f"[OK] Loaded hierarchy: {len(criteria_to_subcriteria)} criteria, "
            f"{len(subcriteria_to_criteria)} subcriteria (SC52 included, year-active)"
        )

        return HierarchyMapping(
            subcriteria_to_criteria=subcriteria_to_criteria,
            criteria_to_subcriteria=criteria_to_subcriteria,
            criteria_names=criteria_names,
            subcriteria_names=subcriteria_names,
        )

    def _load_yearly_files(
        self,
        csv_dir: Path,
        hierarchy: HierarchyMapping,
    ) -> Dict[int, pd.DataFrame]:
        """
        Import and validate annual CSV data files.

        Collects files matching 'YYYY.csv' and enforces structural consistency 
        checks for province names and sub-criteria columns.

        Parameters
        ----------
        csv_dir : Path
            Directory containing the yearly CSV files.
        hierarchy : HierarchyMapping
            The structure to validate against.

        Returns
        -------
        Dict[int, pd.DataFrame]
            A dictionary of DataFrames keyed by year.
        """
        year_files = sorted(csv_dir.glob("[0-9][0-9][0-9][0-9].csv"))

        # Backward-compat: if csv_dir is empty, try the parent (old layout)
        if not year_files:
            parent_files = sorted(
                csv_dir.parent.glob("[0-9][0-9][0-9][0-9].csv")
            )
            if parent_files:
                warnings.warn(
                    f"No CSV files found in '{csv_dir}'. "
                    f"Found {len(parent_files)} file(s) in '{csv_dir.parent}' "
                    f"(pre-reorganisation layout). "
                    "Move CSV files into the 'csv/' sub-directory.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                year_files = parent_files

        if not year_files:
            raise FileNotFoundError(
                f"No yearly CSV files (YYYY.csv) found in '{csv_dir}'.\n"
                "Expected layout: data/csv/2011.csv … data/csv/2024.csv"
            )

        expected_subcriteria = hierarchy.all_subcriteria
        yearly_data: Dict[int, pd.DataFrame] = {}

        for year_file in year_files:
            year = int(year_file.stem)

            df = pd.read_csv(year_file)

            # ----- Structural validation -------------------------------------
            if 'Province' not in df.columns:
                raise ValueError(
                    f"Required column 'Province' not found in '{year_file}'.\n"
                    f"Available columns: {df.columns.tolist()}"
                )

            # Normalise Province strings (strip whitespace, no empty entries)
            df['Province'] = df['Province'].astype(str).str.strip()
            blank_mask = df['Province'].eq('')
            if blank_mask.any():
                self.logger.warning(
                    f"Year {year}: {blank_mask.sum()} row(s) with blank "
                    "Province name — dropped."
                )
                df = df[~blank_mask].copy()

            # Duplicate province check
            dup = df['Province'][df['Province'].duplicated()]
            if not dup.empty:
                raise ValueError(
                    f"Duplicate province name(s) in '{year_file}': "
                    f"{dup.tolist()}"
                )

            # ----- Sub-criterion column audit --------------------------------
            missing_cols = set(expected_subcriteria) - set(df.columns)
            if missing_cols:
                self.logger.warning(
                    f"Year {year}: {len(missing_cols)} sub-criterion column(s) "
                    f"absent (structural gap, not an error): "
                    f"{sorted(missing_cols)}"
                )

            # Coerce SC columns to float64 — any non-numeric value becomes NaN
            sc_cols_present = [c for c in expected_subcriteria if c in df.columns]
            df[sc_cols_present] = df[sc_cols_present].apply(
                pd.to_numeric, errors='coerce'
            )

            # Insert Year column (position 0)
            df.insert(0, 'Year', year)

            yearly_data[year] = df
            self.logger.info(
                f"  Loaded {year}: {len(df)} provinces, "
                f"{len(sc_cols_present)} sub-criteria present"
            )

        return yearly_data

    def _create_hierarchical_views(
        self,
        yearly_data: Dict[int, pd.DataFrame],
        hierarchy: HierarchyMapping,
    ) -> PanelData:
        """Create hierarchical panel data with dynamic exclusion via YearContext.

        For each year:
        1. Build a ``YearContext`` that records *exactly* which provinces and
           sub-criteria are active (have at least one valid value) versus
           completely absent (all-NaN).
        2. Per criterion, further restrict to provinces that have **complete**
           valid data for all active SCs in that criterion — guaranteeing a
           NaN-free decision matrix for MCDM.
        3. Cross-sections and long-format frames preserve raw NaN values so
           that the full temporal history is queryable; consumers must call
           ``get_criterion_matrix()`` or consult ``year_contexts`` to obtain
           clean subsets.
        """
        years = sorted(yearly_data.keys())

        # Global province list: union of all provinces across all years
        all_provinces: set = set()
        for df in yearly_data.values():
            all_provinces.update(df['Province'].unique())
        provinces = sorted(all_provinces)

        # Containers
        subcriteria_data: List[pd.DataFrame] = []
        criteria_data:    List[pd.DataFrame] = []
        final_data:       List[pd.DataFrame] = []

        subcriteria_cross: Dict[int, pd.DataFrame] = {}
        criteria_cross:    Dict[int, pd.DataFrame] = {}
        final_cross:       Dict[int, pd.DataFrame] = {}

        year_contexts: Dict[int, YearContext] = {}

        # Legacy availability dict (kept for backward-compat)
        availability: Dict = {
            'province_by_year':   {},
            'subcriteria_by_year': {},
            'criteria_by_year':   {},
        }

        for year in years:
            df_year = yearly_data[year]
            subcriteria_cols = [
                c for c in hierarchy.all_subcriteria if c in df_year.columns
            ]
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
            sc_any_valid = df_sub_idx.notna().any(axis=0)
            active_scs   = sc_any_valid[sc_any_valid].index.tolist()
            excluded_scs = sc_any_valid[~sc_any_valid].index.tolist()

            # Step 3 — Per-criterion: derive active SCs and clean province sets
            criterion_alts_map: Dict[str, List[str]] = {}
            criterion_scs_map:  Dict[str, List[str]] = {}

            df_active_prov = df_sub_idx.loc[
                [p for p in active_provs if p in df_sub_idx.index]
            ]

            for crit_id in hierarchy.all_criteria:
                all_scs_in_crit = hierarchy.criteria_to_subcriteria[crit_id]
                crit_active_scs = [
                    sc for sc in all_scs_in_crit if sc in active_scs
                ]
                criterion_scs_map[crit_id] = crit_active_scs

                if not crit_active_scs:
                    criterion_alts_map[crit_id] = []
                    continue

                avail_scs_in_df = [
                    sc for sc in crit_active_scs if sc in df_active_prov.columns
                ]
                if not avail_scs_in_df:
                    criterion_alts_map[crit_id] = []
                    continue

                # Province participates iff it has valid data for ALL active SCs
                prov_complete = df_active_prov[avail_scs_in_df].notna().all(axis=1)
                criterion_alts_map[crit_id] = (
                    prov_complete[prov_complete].index.tolist()
                )

            # Step 4 — Active criteria: has at least one active SC
            active_criteria_list = [
                c for c in hierarchy.all_criteria if criterion_scs_map.get(c)
            ]
            excluded_criteria_list = [
                c for c in hierarchy.all_criteria if not criterion_scs_map.get(c)
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
            availability['province_by_year'][year]    = active_provs
            availability['subcriteria_by_year'][year] = active_scs

            # ==============================================================
            # Criteria composite (mean of active SCs per province per criterion)
            # ==============================================================
            criteria_values: Dict[str, pd.Series] = {}
            criteria_availability_yr: Dict[str, List[str]] = {}

            for crit_id, crit_active_scs in criterion_scs_map.items():
                if not crit_active_scs:
                    criteria_values[crit_id] = pd.Series(np.nan, index=df_sub.index)
                    criteria_availability_yr[crit_id] = []
                    continue

                avail_scs_in_df = [
                    sc for sc in crit_active_scs if sc in df_sub.columns
                ]
                if not avail_scs_in_df:
                    criteria_values[crit_id] = pd.Series(np.nan, index=df_sub.index)
                    criteria_availability_yr[crit_id] = []
                    continue

                # Vectorised row-mean — pandas skips NaN cells automatically
                col_scores = df_sub[avail_scs_in_df].mean(axis=1)
                criteria_values[crit_id] = col_scores
                has_score = df_sub[avail_scs_in_df].notna().any(axis=1)
                criteria_availability_yr[crit_id] = (
                    df_sub.loc[has_score, 'Province'].tolist()
                )

            availability['criteria_by_year'][year] = criteria_availability_yr

            df_criteria = pd.DataFrame({
                'Year':     df_sub['Year'],
                'Province': df_sub['Province'],
                **criteria_values,
            })

            # ==============================================================
            # Final composite (mean of criteria composites, skip NaN)
            # ==============================================================
            active_crit_cols = [
                c for c in hierarchy.all_criteria if c in df_criteria.columns
            ]
            final_scores = df_criteria[active_crit_cols].mean(axis=1)

            df_final = pd.DataFrame({
                'Year':       df_criteria['Year'],
                'Province':   df_criteria['Province'],
                'FinalScore': final_scores.values,
            })

            subcriteria_data.append(df_sub)
            criteria_data.append(df_criteria)
            final_data.append(df_final)

            subcriteria_cross[year] = df_sub_idx
            criteria_cross[year]    = (
                df_criteria.set_index('Province')[hierarchy.all_criteria]
            )
            final_cross[year] = df_final.set_index('Province')[['FinalScore']]

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

    def create_forecast_year_context(
        self,
        panel_data: PanelData,
        forecast_year: int,
    ) -> None:
        """Create a YearContext for a future forecast year (e.g., 2025).
        
        Creates a YearContext for the forecast year that:
        - **Mirrors sub-criteria and criteria structure** from the latest year (e.g., 2024)
          → 29 active SCs (SC52 included, year-active), 8 active criteria
        - **Uses ALL 63 global provinces** for prediction (not limited to latest year's available data)
        
        This ensures ML forecasting has a complete, consistent scope:
        **29 SCs × 8 criteria × 63 provinces** for all forecast year predictions.
        
        **PHASE 1 (2025 Planning)**: Used for 2025 forecast context creation.
        
        Parameters
        ----------
        panel_data : PanelData
            The loaded panel (2011–2024 or latest available).
        forecast_year : int
            The target future year (e.g., 2025).
        
        Side Effect:
            Adds ``panel_data.year_contexts[forecast_year]`` with structure mirrored
            from latest year but province scope set to ALL global provinces.
        
        Notes
        -----
        - Sub-criteria active set is copied from the latest year.
        - Criteria active set is copied from the latest year.
        - Province set is the GLOBAL province list (all 63 from the panel).
        - Per-criterion alternatives are built for all 63 provinces.
        - Valid pairs are constructed for all (province, SC) combinations.
        """
        latest_year = max(panel_data.years)
        ctx_latest = panel_data.year_contexts.get(latest_year)
        
        if ctx_latest is None:
            raise ValueError(
                f"YearContext for latest year {latest_year} not found. "
                f"Cannot create forecast context for {forecast_year}."
            )
        
        # ────────────────────────────────────────────────────────────────────
        # Mirror sub-criteria and criteria structure from latest year
        # ────────────────────────────────────────────────────────────────────
        active_scs = ctx_latest.active_subcriteria.copy()  # 29 SCs (SC52 year-active)
        active_criteria = ctx_latest.active_criteria.copy()  # 8 criteria
        
        # ────────────────────────────────────────────────────────────────────
        # Use ALL 63 global provinces for forecast year
        # ────────────────────────────────────────────────────────────────────
        all_global_provinces = panel_data.provinces  # All 63
        active_provinces = all_global_provinces.copy()  # Use all for prediction
        excluded_provinces = []  # No provinces excluded for forecast year
        
        # ────────────────────────────────────────────────────────────────────
        # Build per-criterion alternatives: all 63 provinces for each criterion
        # ────────────────────────────────────────────────────────────────────
        criterion_alternatives_map: Dict[str, List[str]] = {}
        criterion_scs_map: Dict[str, List[str]] = {}
        
        for crit_id in active_criteria:
            # All provinces participate in each criterion for forecast
            criterion_alternatives_map[crit_id] = active_provinces.copy()
            # SCs per criterion mirrored from latest year
            criterion_scs_map[crit_id] = (
                ctx_latest.criterion_subcriteria.get(crit_id, []).copy()
            )
        
        # ────────────────────────────────────────────────────────────────────
        # Build valid pairs: all (province, SC) combinations for 29 active SCs
        # ────────────────────────────────────────────────────────────────────
        valid_pairs: Set[Tuple[str, str]] = set()
        for prov in active_provinces:
            for sc in active_scs:
                valid_pairs.add((prov, sc))
        
        # ────────────────────────────────────────────────────────────────────
        # Create forecast YearContext
        # ────────────────────────────────────────────────────────────────────
        ctx_forecast = YearContext(
            year=forecast_year,
            active_provinces=active_provinces,
            active_subcriteria=active_scs,
            active_criteria=active_criteria,
            excluded_provinces=excluded_provinces,
            excluded_subcriteria=[],  # No SCs excluded (29 active, SC52 year-active)
            excluded_criteria=[],  # No criteria excluded (8 active)
            criterion_alternatives=criterion_alternatives_map,
            criterion_subcriteria=criterion_scs_map,
            valid_pairs=valid_pairs,
        )
        
        panel_data.year_contexts[forecast_year] = ctx_forecast
        
        self.logger.info(
            f"[PHASE 1] Created YearContext for forecast year {forecast_year}:\n"
            f"  Structure mirrored from {latest_year}\n"
            f"  Active: {len(ctx_forecast.active_provinces)} provinces [ALL GLOBAL],\n"
            f"          {len(ctx_forecast.active_subcriteria)} subcriteria,\n"
            f"          {len(ctx_forecast.active_criteria)} criteria\n"
            f"  ✓ Scope: 29 SCs × 8 criteria × 63 provinces"
        )
        
        # ────────────────────────────────────────────────────────────────────
        # Validation: enforce 29 SCs, 8 criteria, 63 provinces
        # ────────────────────────────────────────────────────────────────────
        if len(ctx_forecast.active_subcriteria) != 29:
            raise ValueError(
                f"[PHASE 1 VALIDATION FAILED] Forecast year {forecast_year} "
                f"should have 29 active subcriteria (SC52 year-active), "
                f"but got {len(ctx_forecast.active_subcriteria)}: "
                f"{ctx_forecast.active_subcriteria}"
            )
        
        if len(ctx_forecast.active_criteria) != 8:
            raise ValueError(
                f"[PHASE 1 VALIDATION FAILED] Forecast year {forecast_year} "
                f"should have 8 active criteria, "
                f"but got {len(ctx_forecast.active_criteria)}: "
                f"{ctx_forecast.active_criteria}"
            )
        
        if len(ctx_forecast.active_provinces) != 63:
            raise ValueError(
                f"[PHASE 1 VALIDATION FAILED] Forecast year {forecast_year} "
                f"should have 63 active provinces (ALL GLOBAL), "
                f"but got {len(ctx_forecast.active_provinces)}"
            )
        
        if len(ctx_forecast.valid_pairs) != 29 * 63:
            raise ValueError(
                f"[PHASE 1 VALIDATION FAILED] Forecast year {forecast_year} "
                f"should have 29×63=1827 valid pairs (SC52 year-active), "
                f"but got {len(ctx_forecast.valid_pairs)}"
            )


def load_data() -> PanelData:
    """Convenience function to load panel data with default config."""
    return DataLoader().load()
