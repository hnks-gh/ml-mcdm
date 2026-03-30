# -*- coding: utf-8 -*-
"""Unit tests for ranking/hierarchical_pipeline.py — NaN handling.

Tests verify:
  • NaN cells survive min-max normalization unchanged.
  • Valid cells are correctly normalized to [0, 1].
  • Cost-criterion inversion still works.
  • Degenerate case (constant/all-NaN column) sets column to 0.5 fallback
    (undefined-normalization behavior, intentional, not imputation).

As of 2026-03-20, no imputation occurs in ranking phase.
Deprecated functions (impute_neutral_score) fully removed.
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


# ---------------------------------------------------------------------------
# TestERToggle — T-01: use_evidential_reasoning config toggle
#
# Validates that:
#   1. HierarchicalRankingPipeline defaults to use_evidential_reasoning=True.
#   2. rank_fast() returns None immediately when ranking aggregation is disabled.
#   3. HierarchicalRankingResult with er_result=None returns None from all
#      Ranking aggregation properties (final_ranking, final_scores, kendall_w, top_n).
#   4. summary() returns a meaningful string even without er_result.
# ---------------------------------------------------------------------------

from ranking.hierarchical_pipeline import HierarchicalRankingResult


class TestERToggle:
    """T-01: use_evidential_reasoning config toggle."""

    def test_pipeline_default_er_is_true(self):
        """Pipeline must default to use_evidential_reasoning=True (backward compat)."""
        pl = HierarchicalRankingPipeline()
        assert pl.use_evidential_reasoning is True

    def test_pipeline_er_false_init(self):
        """Pipeline can be created with use_evidential_reasoning=False."""
        pl = HierarchicalRankingPipeline(use_evidential_reasoning=False)
        assert pl.use_evidential_reasoning is False

    def test_rank_fast_returns_none_when_er_disabled(self):
        """rank_fast() must return None immediately when ranking aggregation is disabled."""
        pl = HierarchicalRankingPipeline(use_evidential_reasoning=False)
        result = pl.rank_fast(
            precomputed_scores={},
            subcriteria_weights={},
            hierarchy=None,   # not reached — returns None before using it
            alternatives=[],
        )
        assert result is None

    def _make_result_no_er(self):
        """Helper: build a HierarchicalRankingResult with er_result=None."""
        return HierarchicalRankingResult(
            er_result=None,
            criterion_method_scores={'C01': {'TOPSIS': pd.Series({'P1': 0.8})}},
            criterion_method_ranks={'C01': {'TOPSIS': pd.Series({'P1': 1})}},
            criterion_weights_used={'C01': 1.0},
            subcriteria_weights_used={'C01': {'SC11': 1.0}},
            methods_used=['TOPSIS', 'VIKOR', 'PROMETHEE', 'COPRAS', 'EDAS', 'Base'],
            target_year=2024,
        )

    def test_final_ranking_is_none_when_er_disabled(self):
        result = self._make_result_no_er()
        assert result.final_ranking is None

    def test_final_scores_is_none_when_er_disabled(self):
        result = self._make_result_no_er()
        assert result.final_scores is None

    def test_kendall_w_is_nan_when_er_disabled(self):
        result = self._make_result_no_er()
        kw = result.kendall_w
        # When ranking aggregation is disabled, kendall_w is either None or NaN (no ranking consensus)
        assert kw is None or (kw != kw)  # NaN != NaN is True

    def test_top_n_is_none_when_er_disabled(self):
        result = self._make_result_no_er()
        assert result.top_n(5) is None

    def test_summary_returns_string_when_er_disabled(self):
        result = self._make_result_no_er()
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 0
