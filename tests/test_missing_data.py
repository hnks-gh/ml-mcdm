# -*- coding: utf-8 -*-
"""
Unit tests for data/missing_data.py — focused on F-05 changes.

F-05 removed the `impute_column_mean` call from `prepare_decision_matrix` so
that partial NaN cells are preserved rather than silently synthesised.  These
tests verify:

  • Partial NaN cells survive the call unchanged.
  • All-NaN rows are still excluded (the key structural filter).
  • All-NaN columns are still excluded.
  • ValueError is still raised when min_rows / min_cols thresholds are not met.
  • MatrixFilterReport.to_dict()['note'] reflects the new strategy.
  • `impute_neutral_score` is still importable/callable (backward compat).
  • `impute_column_mean` is still importable/callable (backward compat).
"""

import numpy as np
import pandas as pd
import pytest

from data.missing_data import (
    prepare_decision_matrix,
    MatrixFilterReport,
    impute_neutral_score,
    impute_column_mean,
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
# TestImputeNeutralScoreBackwardCompat
# Verify the function is still importable and functional (F-04 removed its
# call from the ranking pipeline but did not delete the function).
# ---------------------------------------------------------------------------

class TestImputeNeutralScoreBackwardCompat:

    def test_function_still_exists_and_fills_nan(self):
        """impute_neutral_score must still exist and fill NaN with 0.5."""
        df = pd.DataFrame({'A': [0.2, np.nan, 0.8], 'B': [np.nan, 0.5, 0.3]})
        result = impute_neutral_score(df)

        assert result.isna().sum().sum() == 0, "All NaN must be filled"
        assert result.at[1, 'A'] == pytest.approx(0.5)
        assert result.at[0, 'B'] == pytest.approx(0.5)

    def test_custom_neutral_value(self):
        """Custom neutral fill value must be honored."""
        df = pd.DataFrame({'A': [np.nan], 'B': [0.3]})
        result = impute_neutral_score(df, neutral=0.99)
        assert result.at[0, 'A'] == pytest.approx(0.99)

    def test_exported_from_data_package(self):
        """impute_neutral_score must still be importable from the data package."""
        from data import impute_neutral_score as ins  # noqa: F401
        assert callable(ins)

    def test_impute_column_mean_still_exported(self):
        """impute_column_mean must still be importable from the data package."""
        from data import impute_column_mean as icm  # noqa: F401
        assert callable(icm)

    def test_impute_column_mean_still_fills_nan(self):
        """impute_column_mean must still work as documented (backward compat)."""
        df = pd.DataFrame({'A': [1.0, np.nan, 3.0], 'B': [4.0, 2.0, np.nan]})
        result = impute_column_mean(df)
        assert result.at[1, 'A'] == pytest.approx(2.0)   # mean(1, 3) = 2
        assert result.at[2, 'B'] == pytest.approx(3.0)   # mean(4, 2) = 3
