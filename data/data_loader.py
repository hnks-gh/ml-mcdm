# -*- coding: utf-8 -*-
"""Panel data loading with hierarchical structure and dynamic missing-data exclusion.

This module handles:
1. Loading multiple CSV files from ``data/csv/`` (one file per year, YYYY.csv)
2. Hierarchical structure: Subcriteria → Criteria → Final Score
3. **Dynamic exclusion** via :class:`YearContext`: missing provinces/sub-criteria
   are completely removed from each year's analysis rather than filled or imputed.
   - Province excluded for year Y  → not an alternative in MCDM or ML for Y
   - Sub-criterion excluded for Y  → not a column in the year's hierarchy
   - Criterion excluded for Y      → not a node in the ER aggregation tree
4. ``YearContext`` objects record exactly what was active vs. excluded per year.
5. Composite calculation at each hierarchy level (using only active sub-criteria)

Directory layout expected by :class:`DataLoader`::

    data/
    ├── csv/
    │   ├── 2011.csv
    │   ├── 2012.csv
    │   │   …
    │   └── 2024.csv
    └── codebook/
        ├── codebook_criteria.csv
        ├── codebook_subcriteria.csv
        └── codebook_provinces.csv

Notes
-----
The dataset uses NaN (not zero) to represent missing observations.  A value of
exactly 0.0 is treated as a legitimate governance score.  All missing-data
detection relies on ``pd.notna()`` / ``pd.isnull()`` rather than comparisons
against zero.

Downstream consumers **must** use ``panel_data.year_contexts[year]`` to
discover the active set of provinces / sub-criteria for each year instead of
assuming a fixed panel dimension.

See Also
--------
data.missing_data : centralised NaN-filtering utilities shared with the
    weighting, ranking, and forecasting phases.
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
    """Mapping between subcriteria and criteria."""
    subcriteria_to_criteria: Dict[str, str]   # SC11 → C01
    criteria_to_subcriteria: Dict[str, List[str]]  # C01 → [SC11, SC12, …]
    criteria_names: Dict[str, str]            # C01 → "Participation"
    subcriteria_names: Dict[str, str]         # SC11 → "Civic Knowledge"

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
            return pd.DataFrame()

        cs = self.subcriteria_cross_section.get(year, pd.DataFrame())
        avail_provs = [p for p in provinces if p in cs.index]
        avail_scs   = [c for c in sc_cols   if c in cs.columns]
        if not avail_provs or not avail_scs:
            return pd.DataFrame()

        mat = cs.loc[avail_provs, avail_scs]
        # Defensive: drop any rows that still contain NaN (should not occur
        # given the YearContext guarantees, but guards against edge cases).
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
        """Load and assemble the full hierarchical panel dataset."""
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
        """Load hierarchy mapping from codebook files.
        
        NOTE: Permanently excludes SC52 (discontinued from 2021 onward) to ensure
        consistent 28-SC structure across all analysis phases.
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

        # ── CRITICAL FIX (Phase 1): Exclude SC52 globally from hierarchy ──
        # SC52 was discontinued as of 2021. For end-to-end integrity,
        # we permanently exclude it from the hierarchy structure.
        # This ensures all analysis phases (MCDM, forecasting) work with a
        # consistent 28-SC structure, not 29.
        if "SC52" in subcriteria_to_criteria:
            self.logger.info(
                f"[PHASE 1] Excluding SC52 from hierarchy (discontinued from 2021)"
            )
            del subcriteria_to_criteria["SC52"]
            if "SC52" in subcriteria_names:
                del subcriteria_names["SC52"]

        criteria_to_subcriteria: Dict[str, List[str]] = {}
        for sc, c in subcriteria_to_criteria.items():
            if not sc or not c:
                continue  # skip blank rows
            criteria_to_subcriteria.setdefault(c, []).append(sc)
        for c in criteria_to_subcriteria:
            criteria_to_subcriteria[c] = sorted(criteria_to_subcriteria[c])

        # Validate: C05 should have 3 SCs (SC51, SC53, SC54), not 4
        c05_scs = criteria_to_subcriteria.get("C05", [])
        if len(c05_scs) != 3 or "SC52" in c05_scs:
            raise ValueError(
                f"[PHASE 1 VALIDATION FAILED] C05 should have 3 SCs (excluding SC52), "
                f"but got {len(c05_scs)}: {c05_scs}"
            )

        self.logger.info(
            f"[OK] Loaded hierarchy: {len(criteria_to_subcriteria)} criteria, "
            f"{len(subcriteria_to_criteria)} subcriteria (SC52 excluded)"
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
        """Load all yearly CSV files from *csv_dir*.

        Expects files named ``YYYY.csv`` (four-digit year).  If *csv_dir* is
        empty the method falls back to ``csv_dir.parent`` and emits a
        :class:`DeprecationWarning` so operators are prompted to migrate to the
        new layout.

        Input data audit
        ----------------
        * Sub-criterion columns are coerced to ``float64`` via
          ``pd.to_numeric(errors='coerce')``: stray string values become NaN
          rather than crashing the pipeline.
        * The ``Province`` column is stripped of leading/trailing whitespace.
        * Missing sub-criterion columns are warned about (not raised), because
          earlier years legitimately lack later-introduced indicators.
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
          → 28 active SCs (SC52 excluded), 8 active criteria
        - **Uses ALL 63 global provinces** for prediction (not limited to latest year's available data)
        
        This ensures ML forecasting has a complete, consistent scope:
        **28 SCs × 8 criteria × 63 provinces** for all forecast year predictions.
        
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
        active_scs = ctx_latest.active_subcriteria.copy()  # 28 SCs
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
        # Build valid pairs: all (province, SC) combinations for 28 active SCs
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
            excluded_subcriteria=[],  # No SCs excluded (28 active)
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
            f"  ✓ Scope: 28 SCs × 8 criteria × 63 provinces"
        )
        
        # ────────────────────────────────────────────────────────────────────
        # Validation: enforce 28 SCs, 8 criteria, 63 provinces
        # ────────────────────────────────────────────────────────────────────
        if len(ctx_forecast.active_subcriteria) != 28:
            raise ValueError(
                f"[PHASE 1 VALIDATION FAILED] Forecast year {forecast_year} "
                f"should have 28 active subcriteria, "
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
        
        if len(ctx_forecast.valid_pairs) != 28 * 63:
            raise ValueError(
                f"[PHASE 1 VALIDATION FAILED] Forecast year {forecast_year} "
                f"should have 28×63=1764 valid pairs, "
                f"but got {len(ctx_forecast.valid_pairs)}"
            )


def load_data() -> PanelData:
    """Convenience function to load panel data with default config."""
    return DataLoader().load()
