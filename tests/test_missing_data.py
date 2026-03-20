# -*- coding: utf-8 -*-
"""Unit tests for data/missing_data.py — missing data handling.

Tests verify:
  • Partial NaN cells are preserved (not imputed).
  • All-NaN rows are excluded (structural filter).
  • All-NaN columns are excluded (structural filter).
  • ValueError raised when thresholds not met.
  • MatrixFilterReport tracks included/excluded metadata.

As of 2026-03-20, backward compatibility tests removed.
Deprecated functions (impute_neutral_score, impute_panel_temporal,
impute_column_mean) are no longer available.
"""

import numpy as np
import pandas as pd
import pytest

from data.missing_data import (
    prepare_decision_matrix,
    MatrixFilterReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(values, cols=None, index=None):
    """Build a small DataFrame from a 2-D list-of-lists."""
    cols  = cols  or [f"C{i}" for i in range(len(values[0]))]
    index = index or [f"P{i}" for i in range(len(values))]
    return pd.DataFrame(values, columns=cols, index=index)


# ---------------------------------------------------------------------------
# TestPrepareDecisionMatrix
# ---------------------------------------------------------------------------

class TestPrepareDecisionMatrix:
    """Behavioural tests — especially that partial NaN is no longer imputed."""

    # ── Core F-05 behaviour ───────────────────────────────────────────

    def test_partial_nan_cells_preserved_not_imputed(self):
        """
        Partial NaN cells (a row that has *some* valid data) must survive
        the call unchanged.  Before F-05, impute_column_mean() filled them
        with the column mean; after F-05 they should remain NaN.
        """
        df = _make_df([
            [0.8, 0.6, 0.4],
            [0.5, np.nan, 0.9],   # partial NaN in C1
            [0.3, 0.7, 0.2],
        ])
        result, _ = prepare_decision_matrix(df)

        assert pd.isna(result.at['P1', 'C1']), (
            "Partial NaN cell must be preserved (not column-mean imputed)"
        )
        # Non-NaN cells must be unchanged
        assert result.at['P1', 'C0'] == pytest.approx(0.5)
        assert result.at['P1', 'C2'] == pytest.approx(0.9)

    def test_partial_nan_not_equal_to_mean_imputation(self):
        """
        Confirm the result is different from what mean-imputation would have
        produced (guard against silent regression).
        """
        df = _make_df([
            [1.0, 2.0],
            [3.0, np.nan],
            [5.0, 6.0],
        ])
        mean_imputed_val = df['C1'].mean()   # = (2.0 + 6.0) / 2 = 4.0

        result, _ = prepare_decision_matrix(df)
        assert pd.isna(result.at['P1', 'C1']), (
            f"Expected NaN preserved; mean-imputation would have set {mean_imputed_val}"
        )

    # ── Structural filtering still works ─────────────────────────────

    def test_all_nan_row_excluded(self):
        """Province with ALL-NaN criteria must be dropped."""
        df = _make_df([
            [0.5, 0.6],
            [np.nan, np.nan],   # all-NaN row → must be excluded
            [0.3, 0.8],
        ])
        result, report = prepare_decision_matrix(df)

        assert 'P1' not in result.index, "All-NaN province must be excluded"
        assert 'P1' in report.excluded_rows
        assert report.n_included_rows == 2
        assert report.n_excluded_rows == 1

    def test_all_nan_column_excluded(self):
        """Sub-criterion with ALL-NaN values must be dropped."""
        df = _make_df([
            [0.5, np.nan],
            [0.3, np.nan],
            [0.7, np.nan],
        ])
        result, report = prepare_decision_matrix(df, min_cols=1)

        assert 'C1' not in result.columns, "All-NaN column must be excluded"
        assert 'C1' in report.excluded_columns
        assert report.n_included_columns == 1

    def test_clean_input_returned_as_is(self):
        """NaN-free input must pass through with values unchanged."""
        df = _make_df([
            [0.2, 0.8, 0.5],
            [0.6, 0.4, 0.9],
            [0.1, 0.7, 0.3],
        ])
        result, report = prepare_decision_matrix(df)

        pd.testing.assert_frame_equal(result, df)
        assert report.n_excluded_rows == 0
        assert report.n_excluded_columns == 0

    def test_entity_col_stripped(self):
        """The entity identifier column must not appear in the returned matrix."""
        df = pd.DataFrame({
            'Province': ['P1', 'P2', 'P3'],
            'C1': [0.5, 0.6, 0.7],
            'C2': [0.3, 0.4, 0.5],
        })
        result, _ = prepare_decision_matrix(df, entity_col='Province')

        assert 'Province' not in result.columns
        assert list(result.columns) == ['C1', 'C2']

    # ── ValueError thresholds still enforced ─────────────────────────

    def test_raises_insufficient_rows_after_filter(self):
        """ValueError raised when fewer than min_rows rows remain."""
        df = _make_df([
            [np.nan, np.nan],   # all-NaN → excluded
            [0.5,    0.6],      # 1 remaining < min_rows=2
        ])
        with pytest.raises(ValueError, match="Insufficient rows"):
            prepare_decision_matrix(df, min_rows=2)

    def test_raises_insufficient_cols_after_filter(self):
        """ValueError raised when fewer than min_cols columns remain."""
        df = _make_df([
            [0.5, np.nan],
            [0.6, np.nan],
            [0.7, np.nan],
        ])                      # C1 all-NaN → 1 col remaining < min_cols=2
        with pytest.raises(ValueError, match="Insufficient columns"):
            prepare_decision_matrix(df, min_cols=2)

    # ── MatrixFilterReport note updated ──────────────────────────────

    def test_report_note_does_not_mention_imputation(self):
        """MatrixFilterReport.to_dict() note must reflect the NaN-preserving strategy."""
        df = _make_df([[0.5, 0.6], [0.3, np.nan], [0.7, 0.8]])
        _, report = prepare_decision_matrix(df)
        note = report.to_dict()['note']

        assert 'imputed' not in note.lower(), (
            f"Note must not mention imputation but got: {note!r}"
        )
        assert 'preserved' in note.lower() or 'no imputation' in note.lower(), (
            f"Note should confirm NaN preservation but got: {note!r}"
        )

    def test_report_to_dict_structure(self):
        """MatrixFilterReport.to_dict() has the expected keys."""
        df = _make_df([[0.5, 0.6], [0.3, 0.4]])
        _, report = prepare_decision_matrix(df)
        d = report.to_dict()

        for key in ('included_rows', 'excluded_rows', 'included_columns',
                    'excluded_columns', 'note'):
            assert key in d, f"Expected key {key!r} in to_dict()"


# ---------------------------------------------------------------------------
# TestBuildMlPanelData
# Verify that build_ml_panel_data() produces a fully-imputed PanelData copy
# while leaving the original panel_data untouched.
# ---------------------------------------------------------------------------

class TestBuildMlPanelData:
    """Tests for build_ml_panel_data() and its helper _build_ml_year_contexts().

    Uses a minimal synthetic PanelData (3 provinces × 3 years × 1 criterion ×
    2 sub-criteria) where one province (P2) has all-NaN values in the last year
    — matching the real P17/P52 scenario that motivated the implementation.
    """

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_panel(last_year_all_nan_prov: str = "P2"):
        """Build a minimal PanelData for testing.

        Layout
        ------
        Provinces : P0, P1, P2
        Years     : 2011, 2012, 2013
        Subcriteria : s0, s1  (both in criterion C0)
        Criteria    : C0

        P2 has all-NaN in year 2013 (the last year) to simulate the
        all-NaN-last-year scenario.  P0 and P1 have complete valid data.
        """
        from data.data_loader import (
            PanelData, HierarchyMapping, YearContext,
        )

        provinces = ["P0", "P1", "P2"]
        years = [2011, 2012, 2013]
        scs = ["s0", "s1"]
        crits = ["C0"]

        hierarchy = HierarchyMapping(
            subcriteria_to_criteria={"s0": "C0", "s1": "C0"},
            criteria_to_subcriteria={"C0": ["s0", "s1"]},
            criteria_names={"C0": "Test Criterion"},
            subcriteria_names={"s0": "Sub0", "s1": "Sub1"},
        )

        rng = np.random.RandomState(99)

        # Build subcriteria cross-sections
        sub_cs: dict = {}
        for yr in years:
            data = rng.rand(3, 2) * 3.0   # governance scale [0, 3]
            cs = pd.DataFrame(data, index=provinces, columns=scs)
            cs.index.name = "Province"
            if last_year_all_nan_prov and yr == years[-1]:
                cs.loc[last_year_all_nan_prov, :] = np.nan
            sub_cs[yr] = cs

        # Build criteria cross-sections (mean of scs)
        crit_cs: dict = {}
        for yr, cs in sub_cs.items():
            crit_cs[yr] = pd.DataFrame(
                {"C0": cs[["s0", "s1"]].mean(axis=1)}, index=cs.index
            )

        # Build final cross-sections (only one criterion = criteria value)
        final_cs: dict = {}
        for yr, ccs in crit_cs.items():
            final_cs[yr] = pd.DataFrame(
                {"FinalScore": ccs["C0"]}, index=ccs.index
            )

        # Build long-format DataFrames
        sub_frames, crit_frames, final_frames = [], [], []
        for yr in years:
            sf = sub_cs[yr].reset_index()
            sf.insert(0, "Year", yr)
            sub_frames.append(sf)

            cf = crit_cs[yr].reset_index()
            cf.insert(0, "Year", yr)
            crit_frames.append(cf)

            ff = final_cs[yr].reset_index()
            ff.insert(0, "Year", yr)
            final_frames.append(ff)

        sub_long   = pd.concat(sub_frames,   ignore_index=True)
        crit_long  = pd.concat(crit_frames,  ignore_index=True)
        final_long = pd.concat(final_frames, ignore_index=True)

        # Build year_contexts (P2 excluded from last year due to all-NaN)
        year_contexts: dict = {}
        for yr in years:
            cs = sub_cs[yr]
            active = [p for p in provinces
                      if cs.loc[p, scs].notna().any()]
            excluded = [p for p in provinces if p not in active]
            yc = YearContext(
                year=yr,
                active_provinces=active,
                active_subcriteria=scs,
                active_criteria=crits,
                excluded_provinces=excluded,
                excluded_subcriteria=[],
                excluded_criteria=[],
                criterion_alternatives={"C0": active},
                criterion_subcriteria={"C0": scs},
                valid_pairs={
                    (p, s) for p in active for s in scs
                    if pd.notna(cs.loc[p, s])
                },
            )
            year_contexts[yr] = yc

        return PanelData(
            subcriteria_long=sub_long,
            subcriteria_cross_section=sub_cs,
            criteria_long=crit_long,
            criteria_cross_section=crit_cs,
            final_long=final_long,
            final_cross_section=final_cs,
            provinces=provinces,
            years=years,
            hierarchy=hierarchy,
            year_contexts=year_contexts,
            availability={},
        )

    # -----------------------------------------------------------------------
    # Importability
    # -----------------------------------------------------------------------

    def test_importable_from_data_package(self):
        """build_ml_panel_data must be importable from the data package."""
        from data import build_ml_panel_data as bmpd  # noqa: F401
        assert callable(bmpd)

    # -----------------------------------------------------------------------
    # Core correctness
    # -----------------------------------------------------------------------

    def test_no_nan_in_imputed_subcriteria(self):
        """Every cell in the imputed subcriteria_cross_section must be non-NaN."""
        from data.missing_data import build_ml_panel_data

        raw = self._make_panel()
        ml  = build_ml_panel_data(raw)

        for yr, cs in ml.subcriteria_cross_section.items():
            assert not cs.isna().any(axis=None), (
                f"NaN found in imputed subcriteria_cross_section[{yr}]"
            )

    def test_no_nan_in_imputed_criteria(self):
        """Every cell in the imputed criteria_cross_section must be non-NaN."""
        from data.missing_data import build_ml_panel_data

        raw = self._make_panel()
        ml  = build_ml_panel_data(raw)

        for yr, cs in ml.criteria_cross_section.items():
            assert not cs.isna().any(axis=None), (
                f"NaN found in imputed criteria_cross_section[{yr}]"
            )

    def test_all_nan_last_year_province_now_active(self):
        """Province with all-NaN in last year must appear in active_provinces
        after imputation (the primary fix for P17 / P52)."""
        from data.missing_data import build_ml_panel_data

        raw = self._make_panel(last_year_all_nan_prov="P2")
        ml  = build_ml_panel_data(raw)

        last_yr = raw.years[-1]

        # Original: P2 should NOT be active in the last raw year
        assert "P2" not in raw.year_contexts[last_yr].active_provinces, (
            "Test setup error: P2 should be excluded from raw year_contexts."
        )

        # After imputation: P2 MUST be active
        assert "P2" in ml.year_contexts[last_yr].active_provinces, (
            "P2 must be in active_provinces after imputation."
        )
        assert "P2" not in ml.year_contexts[last_yr].excluded_provinces

    def test_original_panel_data_unmodified(self):
        """build_ml_panel_data must not modify the original panel_data."""
        from data.missing_data import build_ml_panel_data

        raw = self._make_panel()
        last_yr = raw.years[-1]

        # Record original NaN count for P2 in last year
        original_nan_count = int(
            raw.subcriteria_cross_section[last_yr].loc["P2"].isna().sum()
        )

        _ = build_ml_panel_data(raw)   # call the function

        after_nan_count = int(
            raw.subcriteria_cross_section[last_yr].loc["P2"].isna().sum()
        )

        assert original_nan_count == after_nan_count, (
            "build_ml_panel_data must not mutate the original subcriteria "
            "cross-sections (deep copy required before imputation)."
        )
        # P2 should still be excluded from raw year_contexts
        assert "P2" not in raw.year_contexts[last_yr].active_provinces

    def test_provinces_and_years_unchanged(self):
        """provinces and years metadata must be identical to the original."""
        from data.missing_data import build_ml_panel_data

        raw = self._make_panel()
        ml  = build_ml_panel_data(raw)

        assert ml.provinces == raw.provinces
        assert ml.years     == raw.years

    def test_criteria_values_consistent_with_imputed_subcriteria(self):
        """criteria_cross_section[year][criterion] must equal the row-mean
        of the corresponding imputed subcriteria columns."""
        from data.missing_data import build_ml_panel_data

        raw = self._make_panel()
        ml  = build_ml_panel_data(raw)

        for yr in raw.years:
            sc_cs   = ml.subcriteria_cross_section[yr]
            crit_cs = ml.criteria_cross_section[yr]

            # C0 is mean of s0 and s1
            expected = sc_cs[["s0", "s1"]].mean(axis=1)
            actual   = crit_cs["C0"]

            pd.testing.assert_series_equal(
                actual.rename(None), expected.rename(None),
                check_names=False, rtol=1e-10,
                obj=f"criteria_cross_section[{yr}]['C0']",
            )

    def test_valid_pairs_include_formerly_excluded_province(self):
        """valid_pairs in the imputed year_contexts must include (P2, s0) and
        (P2, s1) for the last year — these were absent in the raw context."""
        from data.missing_data import build_ml_panel_data

        raw = self._make_panel(last_year_all_nan_prov="P2")
        ml  = build_ml_panel_data(raw)

        last_yr = raw.years[-1]
        imp_ctx = ml.year_contexts[last_yr]

        assert ("P2", "s0") in imp_ctx.valid_pairs
        assert ("P2", "s1") in imp_ctx.valid_pairs
