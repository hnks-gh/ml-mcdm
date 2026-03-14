# -*- coding: utf-8 -*-
"""
Unit tests for ranking/hierarchical_pipeline.py — focused on F-04 changes.

F-04 removed the `impute_neutral_score(result)` call from `_minmax_normalize`
so that partial NaN cells are preserved rather than replaced with 0.5.  These
tests verify:

  • NaN cells survive min-max normalisation unchanged.
  • Valid cells are correctly normalised to [0, 1].
  • Cost-criterion inversion still works.
  • The degenerate case (constant / all-NaN column) sets the column to 0.5 as
    an undefined-normalisation fallback — this is intentional behaviour, not
    imputation, and should be preserved.
  • Importing HierarchicalRankingPipeline no longer requires impute_neutral_score
    (the import was removed).
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Access _minmax_normalize (static method) without constructing the full
# pipeline to keep tests lightweight and dependency-free.
# ---------------------------------------------------------------------------

from ranking.hierarchical_pipeline import HierarchicalRankingPipeline

_norm = HierarchicalRankingPipeline._minmax_normalize


# ---------------------------------------------------------------------------
# TestMinMaxNormalizeNaN
# ---------------------------------------------------------------------------

class TestMinMaxNormalizeNaN:
    """Core F-04 tests: NaN preserved, not filled with 0.5."""

    # ── NaN preservation ─────────────────────────────────────────────

    def test_partial_nan_preserved_not_replaced_with_0_5(self):
        """
        A NaN cell in a column with valid data must survive normalisation
        unchanged.  Before F-04, impute_neutral_score filled it with 0.5.
        """
        df = pd.DataFrame({
            'C1': [0.2, 0.5, np.nan],
            'C2': [1.0, 2.0, 3.0],
        }, index=['P1', 'P2', 'P3'])

        result = _norm(df)

        assert pd.isna(result.at['P3', 'C1']), (
            "Partial NaN must be preserved — impute_neutral_score was removed"
        )

    def test_nan_not_equal_to_0_5_after_normalization(self):
        """
        Confirm directly that NaN != 0.5 after the call (regression guard).
        """
        df = pd.DataFrame({'A': [0.0, 1.0, np.nan]})
        result = _norm(df)

        cell = result.iloc[2, 0]
        assert pd.isna(cell), (
            f"Expected NaN preserved; got {cell!r}"
        )

    def test_multiple_nan_cells_all_preserved(self):
        """Multiple independent NaN cells in different rows/columns all survive."""
        df = pd.DataFrame({
            'X': [1.0,    np.nan, 3.0   ],
            'Y': [np.nan, 2.0,    4.0   ],
            'Z': [0.5,    1.5,    np.nan],
        })
        result = _norm(df)

        assert pd.isna(result.iloc[1, 0])  # X, row 1
        assert pd.isna(result.iloc[0, 1])  # Y, row 0
        assert pd.isna(result.iloc[2, 2])  # Z, row 2

    # ── Correct normalisation for valid values ────────────────────────

    def test_benefit_column_normalised_to_0_1(self):
        """Min value → 0.0, max value → 1.0, proportional in between."""
        df = pd.DataFrame({'C1': [0.0, 5.0, 10.0]})
        result = _norm(df)

        assert result['C1'].iloc[0] == pytest.approx(0.0)
        assert result['C1'].iloc[1] == pytest.approx(0.5)
        assert result['C1'].iloc[2] == pytest.approx(1.0)

    def test_normalised_range_with_nan_skipped(self):
        """
        NaN is skipped by pandas min/max, so the normalisation range must
        be computed from the non-NaN values only.
        """
        df = pd.DataFrame({'C1': [2.0, np.nan, 6.0]})
        result = _norm(df)

        # min=2, max=6, range=4 → (2-2)/4=0.0, NaN stays NaN, (6-2)/4=1.0
        assert result['C1'].iloc[0] == pytest.approx(0.0)
        assert pd.isna(result['C1'].iloc[1])
        assert result['C1'].iloc[2] == pytest.approx(1.0)

    def test_cost_criterion_inverted_correctly(self):
        """Cost criterion: lower raw value → higher normalised score."""
        df = pd.DataFrame({'C1': [0.0, 5.0, 10.0]})
        result = _norm(df, cost_criteria=['C1'])

        assert result['C1'].iloc[0] == pytest.approx(1.0)   # lowest raw → highest norm
        assert result['C1'].iloc[1] == pytest.approx(0.5)
        assert result['C1'].iloc[2] == pytest.approx(0.0)   # highest raw → lowest norm

    # ── Degenerate constant column ─────────────────────────────────────

    def test_constant_column_set_to_0_5_degenerate(self):
        """
        A constant column (zero range) is set to 0.5 as an
        undefined-normalisation fallback.  This is NOT imputation of NaN —
        it indicates that CRITIC has no discriminating information for that
        sub-criterion.  The behaviour must be preserved after F-04.
        """
        df = pd.DataFrame({'C1': [3.0, 3.0, 3.0], 'C2': [1.0, 2.0, 3.0]})
        result = _norm(df)

        assert result['C1'].sub(0.5).abs().max() < 1e-9, (
            "Constant column must be set to 0.5 (undefined normalisation)"
        )
        # C2 is unaffected
        assert result['C2'].iloc[0] == pytest.approx(0.0)
        assert result['C2'].iloc[2] == pytest.approx(1.0)

    def test_all_nan_column_set_to_0_5_degenerate(self):
        """All-NaN column: range is NaN (undefined) → entire column becomes 0.5."""
        df = pd.DataFrame({'C1': [np.nan, np.nan, np.nan], 'C2': [1.0, 2.0, 3.0]})
        result = _norm(df)
        assert result['C1'].sub(0.5).abs().max() < 1e-9

    # ── No dependency on impute_neutral_score ─────────────────────────

    def test_hierarchical_pipeline_import_no_longer_requires_impute_neutral_score(self):
        """
        Importing HierarchicalRankingPipeline must succeed without importing
        impute_neutral_score (the import was removed in F-04b).
        """
        import importlib, sys

        # If already imported (cached), force a fresh import check by
        # verifying the module's globals don't contain the removed symbol.
        import ranking.hierarchical_pipeline as _mod
        assert 'impute_neutral_score' not in dir(_mod), (
            "impute_neutral_score should not be in hierarchical_pipeline's namespace"
        )
