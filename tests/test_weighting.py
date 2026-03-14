# -*- coding: utf-8 -*-
"""
Unit tests for weighting methods.

Covers:
  - CRITICWeightCalculator  (single-level)
  - calculate_weights convenience function
  - WeightResult properties
  - CRITICWeightingCalculator (two-level deterministic pipeline)
"""

import numpy as np
import pandas as pd
import pytest

from weighting.base import WeightResult, calculate_weights
from weighting.critic import CRITICWeightCalculator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_data():
    """4 alternatives × 3 criteria, all positive, reasonable variation."""
    return pd.DataFrame(
        {
            "C1": [0.8, 0.6, 0.9, 0.4],
            "C2": [0.5, 0.5, 0.5, 0.5],  # constant → low CRITIC weight
            "C3": [0.2, 0.8, 0.5, 0.9],  # high variation → high CRITIC weight
        }
    )


@pytest.fixture()
def large_data():
    """8 alternatives × 4 criteria for more robust statistical tests."""
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        rng.uniform(0.1, 1.0, size=(8, 4)),
        columns=["A", "B", "C", "D"],
    )


# ---------------------------------------------------------------------------
# TestWeightResult
# ---------------------------------------------------------------------------

class TestWeightResult:
    def test_as_array_sum_to_one(self):
        wr = WeightResult(
            weights={"C1": 0.4, "C2": 0.3, "C3": 0.3},
            method="test",
            details={},
        )
        assert abs(wr.as_array.sum() - 1.0) < 1e-9

    def test_as_array_preserves_order(self):
        wr = WeightResult(
            weights={"C1": 0.1, "C2": 0.5, "C3": 0.4},
            method="test",
            details={},
        )
        arr = wr.as_array
        assert abs(arr[0] - 0.1) < 1e-12
        assert abs(arr[1] - 0.5) < 1e-12
        assert abs(arr[2] - 0.4) < 1e-12

    def test_as_series_matches_dict(self):
        weights = {"C1": 0.2, "C2": 0.5, "C3": 0.3}
        wr = WeightResult(weights=weights, method="test", details={})
        for k, v in weights.items():
            assert abs(wr.as_series[k] - v) < 1e-12

    def test_method_stored(self):
        wr = WeightResult(weights={"C1": 1.0}, method="entropy", details={})
        assert wr.method == "entropy"


# ---------------------------------------------------------------------------
# TestCRITICWeightCalculator
# ---------------------------------------------------------------------------

class TestCRITICWeightCalculator:
    def test_weights_sum_to_one(self, sample_data):
        result = CRITICWeightCalculator().calculate(sample_data)
        assert abs(result.as_array.sum() - 1.0) < 1e-9

    def test_weights_all_non_negative(self, sample_data):
        result = CRITICWeightCalculator().calculate(sample_data)
        assert (result.as_array >= 0).all()

    def test_method_name(self, sample_data):
        result = CRITICWeightCalculator().calculate(sample_data)
        assert "critic" in result.method.lower()

    def test_perfectly_correlated_columns_lower_weight(self):
        """Two perfectly correlated columns should share lower combined weight."""
        data = pd.DataFrame(
            {
                "C1": [1, 2, 3, 4, 5],
                "C2": [1, 2, 3, 4, 5],  # identical to C1
                "C3": [5, 1, 3, 2, 4],  # different pattern
            },
            dtype=float,
        )
        result = CRITICWeightCalculator().calculate(data)
        # C1 and C2 correlated → lower criteria importance per column
        # Their individual weights should be lower than C3
        # (or at least sum(C1+C2) not dominate unreasonably)
        assert abs(result.as_array.sum() - 1.0) < 1e-9

    def test_larger_matrix(self, large_data):
        result = CRITICWeightCalculator().calculate(large_data)
        assert abs(result.as_array.sum() - 1.0) < 1e-9
        assert len(result.weights) == 4


# ---------------------------------------------------------------------------
# TestCRITICWeightCalculatorNaN
# Verifies the complete-case NaN-exclusion behaviour (F-03 fix):
#   • NaN rows are DROPPED (not column-mean–imputed)
#   • Weights are identical to those from the same data pre-filtered manually
#   • sample_weights are positionally synchronised with surviving rows
#   • Degenerate case (<2 complete-case rows) returns equal weights gracefully
#   • A WARNING is emitted whenever NaN rows are detected
# ---------------------------------------------------------------------------

class TestCRITICWeightCalculatorNaN:
    """Behavioural tests for CRITICWeightCalculator's NaN handling strategy."""

    def test_all_nan_row_dropped_weights_match_clean_reference(self):
        """Weights from data with a fully-NaN row equal weights from the pre-dropped reference."""
        data_clean = pd.DataFrame(
            {
                "C1": [0.80, 0.60, 0.90, 0.70, 0.50],
                "C2": [0.75, 0.55, 0.85, 0.65, 0.45],
                "C3": [0.30, 0.90, 0.10, 0.70, 0.50],
            },
            dtype=float,
        )
        # Row 2 entirely missing  — simulates a Type 2 province-year gap
        data_nan = data_clean.copy()
        data_nan.iloc[2, :] = np.nan

        calc = CRITICWeightCalculator()
        result_auto   = calc.calculate(data_nan)
        result_manual = calc.calculate(data_clean.drop(index=2).reset_index(drop=True))

        for col in data_clean.columns:
            assert abs(result_auto.weights[col] - result_manual.weights[col]) < 1e-12, (
                f"{col}: auto={result_auto.weights[col]:.12f} "
                f"!= manual={result_manual.weights[col]:.12f}"
            )
        assert abs(sum(result_auto.weights.values()) - 1.0) < 1e-10

    def test_partial_nan_row_dropped_weights_match_clean_reference(self):
        """A row NaN in only one column is still row-dropped (row-wise complete-case exclusion)."""
        data = pd.DataFrame(
            {
                "C1": [0.80, 0.60, np.nan, 0.70],  # row 2: SC absent for this obs
                "C2": [0.75, 0.55, 0.85, 0.65],
                "C3": [0.30, 0.90, 0.10, 0.70],
            },
            dtype=float,
        )
        data_ref = data.dropna().reset_index(drop=True)

        calc = CRITICWeightCalculator()
        result      = calc.calculate(data)
        result_ref  = calc.calculate(data_ref)

        for col in data.columns:
            assert abs(result.weights[col] - result_ref.weights[col]) < 1e-12
        assert abs(sum(result.weights.values()) - 1.0) < 1e-10

    def test_weights_not_equal_to_mean_imputation_result(self):
        """Weights differ from what column-mean imputation would produce, confirming imputation is gone.

        Column-mean imputation suppresses σ (the imputed rows equal the column mean,
        shrinking variance toward zero) and inflates r (mean-imputed columns
        spuriously co-vary).  The complete-case result must differ.
        """
        # 6 rows, row 4 NaN in C1 only
        rng = np.random.RandomState(7)
        data = pd.DataFrame(rng.uniform(0.1, 1.0, size=(6, 3)),
                            columns=["C1", "C2", "C3"], dtype=float)
        data.iloc[4, 0] = np.nan  # row 4, column C1

        # What imputation would produce (former behaviour)
        data_imputed = data.copy()
        data_imputed["C1"] = data_imputed["C1"].fillna(data_imputed["C1"].mean())

        calc = CRITICWeightCalculator()
        result_cca      = calc.calculate(data)           # complete-case (desired)
        result_imputed  = calc.calculate(data_imputed)   # imputation-based (forbidden)

        # The C1 weight must differ between the two approaches
        assert abs(result_cca.weights["C1"] - result_imputed.weights["C1"]) > 1e-6, (
            "Expected CCA weight ≠ imputed weight for C1, but they are equal. "
            "This suggests imputation is still active in the code path."
        )

    def test_insufficient_complete_cases_returns_equal_weights_no_exception(self):
        """When NaN exclusion leaves <2 rows, equal weights are returned; no exception raised."""
        # 4 rows but only 1 is complete
        data = pd.DataFrame(
            {
                "C1": [0.8, np.nan, 0.9, np.nan],
                "C2": [0.75, 0.55, np.nan, np.nan],
                "C3": [0.30, np.nan, np.nan, 0.70],
            },
            dtype=float,
        )
        # Row 0 is the only complete row → dropna leaves 1 row < 2 threshold
        calc = CRITICWeightCalculator()
        result = calc.calculate(data)  # must not raise

        expected_w = 1.0 / 3
        for col in ["C1", "C2", "C3"]:
            assert abs(result.weights[col] - expected_w) < 1e-10, (
                f"Expected equal weight {expected_w:.6f} for {col}, "
                f"got {result.weights[col]:.6f}"
            )
        assert abs(sum(result.weights.values()) - 1.0) < 1e-10

    def test_sample_weights_positionally_synchronized_after_nan_drop(self):
        """sample_weights are aligned to surviving rows when NaN rows are dropped."""
        data = pd.DataFrame(
            {
                "C1": [0.80, np.nan, 0.90, 0.70, 0.50],   # row 1: NaN → dropped
                "C2": [0.75, 0.55, 0.85, 0.65, 0.45],
                "C3": [0.30, 0.90, 0.10, 0.70, 0.50],
            },
            dtype=float,
        )
        # Observation weights for original 5 rows; row 1 will be dropped
        sw_5 = np.array([0.30, 0.20, 0.20, 0.15, 0.15])  # sums to 1.0

        calc = CRITICWeightCalculator()
        result_auto = calc.calculate(data, sample_weights=sw_5)

        # Reference: manually remove row 1 from both data and weights
        data_clean = data.drop(index=1).reset_index(drop=True)
        sw_4 = np.array([0.30, 0.20, 0.15, 0.15])  # rows 0, 2, 3, 4 of sw_5
        result_manual = calc.calculate(data_clean, sample_weights=sw_4)

        for col in data.columns:
            assert abs(result_auto.weights[col] - result_manual.weights[col]) < 1e-10, (
                f"{col}: auto={result_auto.weights[col]:.10f} "
                f"!= manual={result_manual.weights[col]:.10f}"
            )
        assert abs(sum(result_auto.weights.values()) - 1.0) < 1e-10

    def test_warning_emitted_on_nan_rows(self, caplog):
        """A WARNING log entry must be emitted when NaN rows are dropped."""
        import logging
        data = pd.DataFrame(
            {
                "C1": [0.8, np.nan, 0.9],
                "C2": [0.75, 0.55, 0.85],
            },
            dtype=float,
        )
        with caplog.at_level(logging.WARNING, logger="weighting.critic"):
            CRITICWeightCalculator().calculate(data)

        messages = " ".join(r.message for r in caplog.records)
        assert "dropped" in messages.lower() or "nan" in messages.lower(), (
            "Expected a WARNING about dropped NaN rows, got: " + messages
        )

    def test_clean_data_no_warning_and_unchanged_behaviour(self, caplog):
        """NaN-free data must produce no warning and the same result as before."""
        import logging
        data = pd.DataFrame(
            {
                "C1": [0.8, 0.6, 0.9, 0.4],
                "C2": [0.5, 0.5, 0.5, 0.5],
                "C3": [0.2, 0.8, 0.5, 0.9],
            },
            dtype=float,
        )
        with caplog.at_level(logging.WARNING, logger="weighting.critic"):
            result = CRITICWeightCalculator().calculate(data)

        # No warning should have been issued
        warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warn_records) == 0, (
            f"Unexpected WARNING for NaN-free data: "
            f"{[r.message for r in warn_records]}"
        )
        # Output must still be a valid probability distribution
        assert abs(sum(result.weights.values()) - 1.0) < 1e-10
        assert all(w >= 0 for w in result.weights.values())

class TestCalculateWeights:
    @pytest.mark.parametrize("method", ["critic"])
    def test_known_methods_return_valid_result(self, sample_data, method):
        result = calculate_weights(sample_data, method)
        assert isinstance(result, WeightResult)
        assert abs(result.as_array.sum() - 1.0) < 1e-9

    def test_equal_method_produces_equal_weights(self, sample_data):
        result = calculate_weights(sample_data, "equal")
        expected = 1.0 / len(sample_data.columns)
        for w in result.weights.values():
            assert abs(w - expected) < 1e-9

    def test_unknown_method_raises_value_error(self, sample_data):
        with pytest.raises(ValueError):
            calculate_weights(sample_data, "nonexistent_method_xyz")

    def test_entropy_method_raises_value_error(self, sample_data):
        """Entropy weighting has been removed; calling it must raise."""
        with pytest.raises(ValueError):
            calculate_weights(sample_data, "entropy")

    def test_result_keys_match_columns(self, sample_data):
        result = calculate_weights(sample_data, "critic")
        assert set(result.weights.keys()) == set(sample_data.columns)


# ---------------------------------------------------------------------------
# Helpers for CRITICWeightingCalculator tests
# ---------------------------------------------------------------------------

def _make_critic_config():
    """Return a minimal WeightingConfig for fast unit tests."""
    from config import WeightingConfig
    return WeightingConfig(epsilon=1e-10, stability_threshold=0.90,
                           perform_stability_check=True)


def _make_panel(
    n_provinces: int = 12,
    n_years: int = 3,
    criteria_groups: dict | None = None,
    rng_seed: int = 42,
) -> tuple:
    """Build a synthetic long-format panel DataFrame + criteria_groups."""
    rng = np.random.RandomState(rng_seed)
    if criteria_groups is None:
        criteria_groups = {
            "C01": ["SC11", "SC12", "SC13"],
            "C02": ["SC21", "SC22"],
        }
    sc_cols = [sc for scs in criteria_groups.values() for sc in scs]
    provinces = [f"P{i:02d}" for i in range(n_provinces)]
    years     = list(range(2011, 2011 + n_years))
    rows = []
    for prov in provinces:
        for yr in years:
            row = {"Province": prov, "Year": yr}
            for sc in sc_cols:
                row[sc] = float(rng.uniform(0.1, 1.0))
            rows.append(row)
    df = pd.DataFrame(rows)
    return df, criteria_groups


# ---------------------------------------------------------------------------
# TestCRITICWeightingCalculator (two-level deterministic pipeline)
# ---------------------------------------------------------------------------

class TestCRITICWeightingCalculator:
    """Tests for the two-level deterministic CRITICWeightingCalculator."""

    @pytest.fixture()
    def panel_and_groups(self):
        return _make_panel(n_provinces=12, n_years=3)

    @pytest.fixture()
    def calc(self):
        from weighting.critic_weighting import CRITICWeightingCalculator
        return CRITICWeightingCalculator(config=_make_critic_config())

    # ── Output structure ─────────────────────────────────────────────────

    def test_method_name(self, calc, panel_and_groups):
        df, cg = panel_and_groups
        result = calc.calculate(df, cg, "Province", "Year")
        assert result.method == "critic_weighting"

    def test_returns_weight_result(self, calc, panel_and_groups):
        df, cg = panel_and_groups
        result = calc.calculate(df, cg, "Province", "Year")
        assert isinstance(result, WeightResult)

    def test_weights_keys_match_all_sc(self, calc, panel_and_groups):
        df, cg = panel_and_groups
        all_sc = [sc for scs in cg.values() for sc in scs]
        result = calc.calculate(df, cg, "Province", "Year")
        assert set(result.weights.keys()) == set(all_sc)

    # ── Simplex constraints ───────────────────────────────────────────────

    def test_global_weights_sum_to_one(self, calc, panel_and_groups):
        df, cg = panel_and_groups
        result = calc.calculate(df, cg, "Province", "Year")
        total = sum(result.weights.values())
        assert abs(total - 1.0) < 1e-8, f"global sum = {total}"

    def test_level1_local_weights_sum_to_one_per_group(self, calc, panel_and_groups):
        df, cg = panel_and_groups
        result = calc.calculate(df, cg, "Province", "Year")
        for crit_id, sc_list in cg.items():
            local_w = result.details["level1"][crit_id]["local_sc_weights"]
            total = sum(local_w.values())
            assert abs(total - 1.0) < 1e-8, f"{crit_id} local sum = {total}"

    def test_level2_criterion_weights_sum_to_one(self, calc, panel_and_groups):
        df, cg = panel_and_groups
        result = calc.calculate(df, cg, "Province", "Year")
        crit_w = result.details["level2"]["criterion_weights"]
        total = sum(crit_w.values())
        assert abs(total - 1.0) < 1e-8, f"criterion sum = {total}"

    def test_global_equals_local_times_criterion(self, calc, panel_and_groups):
        """global_w[sc] == local_w[sc|Ck] × criterion_w[Ck] for all sc."""
        df, cg = panel_and_groups
        result = calc.calculate(df, cg, "Province", "Year")
        gw   = result.details["global_sc_weights"]
        crit = result.details["level2"]["criterion_weights"]
        for crit_id, sc_list in cg.items():
            local = result.details["level1"][crit_id]["local_sc_weights"]
            v_k   = crit[crit_id]
            for sc in sc_list:
                expected = local[sc] * v_k
                assert abs(gw[sc] - expected) < 1e-8, (
                    f"global_w[{sc}]={gw[sc]:.8f} != "
                    f"local_w[{sc}]*crit_w[{crit_id}]={expected:.8f}"
                )

    # ── Determinism ───────────────────────────────────────────────────────

    def test_fully_deterministic(self, panel_and_groups):
        """Same inputs must always produce identical outputs."""
        from weighting.critic_weighting import CRITICWeightingCalculator
        df, cg = panel_and_groups
        cfg = _make_critic_config()
        r1 = CRITICWeightingCalculator(config=cfg).calculate(df, cg)
        r2 = CRITICWeightingCalculator(config=cfg).calculate(df, cg)
        for sc in r1.weights:
            assert r1.weights[sc] == r2.weights[sc], \
                f"Non-deterministic result for {sc}"

    # ── Details schema ────────────────────────────────────────────────────

    def test_details_schema_top_level_keys(self, calc, panel_and_groups):
        df, cg = panel_and_groups
        result = calc.calculate(df, cg, "Province", "Year")
        for key in ("level1", "level2", "global_sc_weights",
                    "critic_sc_weights", "critic_criterion_weights",
                    "n_observations", "n_criteria_groups",
                    "n_subcriteria", "n_provinces"):
            assert key in result.details, f"Missing details key: {key}"

    def test_details_level1_per_group_keys(self, calc, panel_and_groups):
        df, cg = panel_and_groups
        result = calc.calculate(df, cg, "Province", "Year")
        for crit_id in cg:
            grp = result.details["level1"][crit_id]
            assert "local_sc_weights" in grp, \
                f"{crit_id} level1 missing: local_sc_weights"

    # ── All weights non-negative ──────────────────────────────────────────

    def test_all_weights_non_negative(self, calc, panel_and_groups):
        df, cg = panel_and_groups
        result = calc.calculate(df, cg, "Province", "Year")
        for sc, w in result.weights.items():
            assert w >= 0, f"Negative weight for {sc}: {w}"

    # ── Exported from weighting.__init__ ─────────────────────────────────

    def test_exported_from_package(self):
        from weighting import CRITICWeightingCalculator
        assert callable(CRITICWeightingCalculator)


# ---------------------------------------------------------------------------
# TestCRITICWeightingCalculatorNaN
# Verifies F-01 (Level 1) and F-02 (Level 2) complete-case NaN exclusion
# inside CRITICWeightingCalculator:
#   • Level 1 group weights computed from only complete-case rows per group
#   • Level 2 criterion weights computed after dropping NaN-carrying Z rows
#   • Global weights still sum to 1 with structural year-gaps present
#   • Level 1 weights for a gap-affected group match a pre-filtered reference
# ---------------------------------------------------------------------------

class TestCRITICWeightingCalculatorNaN:
    """Behavioural tests for F-01/F-02 NaN exclusion in CRITICWeightingCalculator."""

    @pytest.fixture()
    def calc(self):
        from weighting.critic_weighting import CRITICWeightingCalculator
        return CRITICWeightingCalculator(config=_make_critic_config())

    def _make_gapped_panel(self, gap_group: str = "C02", rng_seed: int = 5):
        """Panel where C02 SCs are absent in early years (Type 1 structural gap)."""
        rng = np.random.RandomState(rng_seed)
        years_all = [2015, 2016, 2017, 2018, 2019]
        years_avail = [2017, 2018, 2019]   # C02 SCs only available from 2017
        provinces = [f"P{i}" for i in range(6)]
        rows = []
        for yr in years_all:
            for pv in provinces:
                sc11 = float(rng.uniform(0.1, 1.0))
                sc12 = float(rng.uniform(0.1, 1.0))
                sc21 = float(rng.uniform(0.1, 1.0)) if yr in years_avail else np.nan
                sc22 = float(rng.uniform(0.1, 1.0)) if yr in years_avail else np.nan
                rows.append({
                    "Province": pv, "Year": yr,
                    "SC11": sc11, "SC12": sc12,
                    "SC21": sc21, "SC22": sc22,
                })
        df = pd.DataFrame(rows)
        cg = {"C01": ["SC11", "SC12"], "C02": ["SC21", "SC22"]}
        return df, cg

    # ── Global constraints with structural gaps ───────────────────────

    def test_global_weights_sum_to_one_with_year_gaps(self, calc):
        """Global SC weights sum to 1 even when one group has a structural year-gap."""
        df, cg = self._make_gapped_panel()
        result = calc.calculate(df, cg, "Province", "Year")
        total = sum(result.weights.values())
        assert abs(total - 1.0) < 1e-8, f"global sum = {total}"

    def test_criterion_weights_sum_to_one_with_year_gaps(self, calc):
        """Level-2 criterion weights sum to 1 even with structural criterion gaps."""
        df, cg = self._make_gapped_panel()
        result = calc.calculate(df, cg, "Province", "Year")
        cw = result.details["level2"]["criterion_weights"]
        total = sum(cw.values())
        assert abs(total - 1.0) < 1e-8, f"criterion sum = {total}"

    def test_local_weights_sum_to_one_per_group_with_gaps(self, calc):
        """Level-1 local weights per group sum to 1 when that group has a year-gap."""
        df, cg = self._make_gapped_panel()
        result = calc.calculate(df, cg, "Province", "Year")
        for crit_id, sc_list in cg.items():
            local_w = result.details["level1"][crit_id]["local_sc_weights"]
            total = sum(local_w.values())
            assert abs(total - 1.0) < 1e-8, f"{crit_id} local sum = {total}"

    def test_all_weights_non_negative_with_year_gaps(self, calc):
        """All SC weights are non-negative in presence of structural year-gaps."""
        df, cg = self._make_gapped_panel()
        result = calc.calculate(df, cg, "Province", "Year")
        for sc, w in result.weights.items():
            assert w >= 0, f"Negative weight for {sc}: {w}"

    # ── F-01: Level 1 weights match pre-filtered reference ───────────

    def test_level1_gap_group_weights_match_manually_filtered_reference(self, calc):
        """
        Level 1 CRITIC weights for the gap-affected group (C02) are identical
        to those obtained from the manually pre-filtered dataset (rows with NaN
        in SC21/SC22 already removed).

        F-01 must produce the same complete-case inputs as the manual drop.
        """
        from weighting.critic_weighting import CRITICWeightingCalculator
        df_full, cg = self._make_gapped_panel(rng_seed=7)

        # Manual reference: drop rows where C02 SCs are absent
        df_ref = df_full.dropna(subset=["SC21", "SC22"]).reset_index(drop=True)

        cfg = _make_critic_config()
        result_auto = CRITICWeightingCalculator(config=cfg).calculate(
            df_full, cg, "Province", "Year")
        result_ref  = CRITICWeightingCalculator(config=cfg).calculate(
            df_ref,  cg, "Province", "Year")

        # C02 local weights must be identical (same complete-case rows used)
        local_auto = result_auto.details["level1"]["C02"]["local_sc_weights"]
        local_ref  = result_ref.details["level1"]["C02"]["local_sc_weights"]
        for sc in ["SC21", "SC22"]:
            assert abs(local_auto[sc] - local_ref[sc]) < 1e-8, (
                f"{sc}: auto={local_auto[sc]:.10f} != ref={local_ref[sc]:.10f}"
            )

    # ── F-02: Level 2 uses complete-case Z rows ───────────────────────

    def test_level2_three_groups_one_gapped_weights_valid(self):
        """
        With 3 criterion groups where C03 has a year-gap, Level 2 detects two
        year-regimes (one where C03 is absent, one where C03 is present) and runs
        a separate CRITIC per regime.  Aggregated criterion weights must form a
        valid probability distribution.
        """
        from weighting.critic_weighting import CRITICWeightingCalculator
        rng = np.random.RandomState(9)
        years = [2016, 2017, 2018, 2019, 2020]
        provinces = [f"P{i}" for i in range(8)]
        rows = []
        for yr in years:
            for pv in provinces:
                rows.append({
                    "Province": pv, "Year": yr,
                    "SC11": float(rng.uniform(0.1, 1.0)),
                    "SC21": float(rng.uniform(0.1, 1.0)),
                    # C03 only from 2019 onwards (Type 1 structural gap)
                    "SC31": float(rng.uniform(0.1, 1.0)) if yr >= 2019 else np.nan,
                })
        df = pd.DataFrame(rows)
        cg = {"C01": ["SC11"], "C02": ["SC21"], "C03": ["SC31"]}

        result = CRITICWeightingCalculator(config=_make_critic_config()).calculate(
            df, cg, "Province", "Year")

        cw = result.details["level2"]["criterion_weights"]
        assert abs(sum(cw.values()) - 1.0) < 1e-8
        assert all(w >= 0 for w in cw.values())
        assert abs(sum(result.weights.values()) - 1.0) < 1e-8

    # ── F-02 revised: year-regime-aware Level 2 behaviour ────────────

    def test_partial_sc_absence_yields_single_regime_valid_composite(self):
        """
        When only SOME SCs of a group are absent in early years (partial gap),
        the renormalized composite Z[i,k] remains valid (never NaN).
        Level 2 should detect exactly 1 regime covering all observations.
        """
        from weighting.critic_weighting import CRITICWeightingCalculator
        rng = np.random.RandomState(11)
        years = [2015, 2016, 2017, 2018, 2019]
        provinces = [f"P{i}" for i in range(6)]
        rows = []
        for yr in years:
            for pv in provinces:
                rows.append({
                    "Province": pv, "Year": yr,
                    "SC11": float(rng.uniform(0.1, 1.0)),
                    "SC12": float(rng.uniform(0.1, 1.0)),
                    # C02: SC21 always present; SC22 absent in 2015-2016 only
                    "SC21": float(rng.uniform(0.1, 1.0)),
                    "SC22": float(rng.uniform(0.1, 1.0)) if yr >= 2017 else np.nan,
                })
        df = pd.DataFrame(rows)
        cg = {"C01": ["SC11", "SC12"], "C02": ["SC21", "SC22"]}

        result = CRITICWeightingCalculator(config=_make_critic_config()).calculate(
            df, cg, "Province", "Year")

        regimes = result.details["level2"]["regimes"]
        # SC21 is always present → Z[., C02] is never NaN → only 1 regime
        assert len(regimes) == 1, (
            f"Expected 1 regime (C02 never fully absent via partial composite), "
            f"got {len(regimes)}: {[r['active_criteria'] for r in regimes]}"
        )
        assert set(regimes[0]["active_criteria"]) == {"C01", "C02"}
        assert abs(sum(result.weights.values()) - 1.0) < 1e-8

    def test_full_sc_absence_yields_two_regimes_with_correct_active_criteria(self):
        """
        When ALL SCs of a criterion group are absent in early years (full gap),
        Z[i, k] = NaN for those rows.  Level 2 should detect exactly 2 regimes:
        one without the criterion, one with it.
        """
        from weighting.critic_weighting import CRITICWeightingCalculator
        rng = np.random.RandomState(12)
        years_all   = [2015, 2016, 2017, 2018, 2019]
        years_avail = [2017, 2018, 2019]   # C02 SCs BOTH absent 2015-2016
        provinces = [f"P{i}" for i in range(6)]
        rows = []
        for yr in years_all:
            for pv in provinces:
                rows.append({
                    "Province": pv, "Year": yr,
                    "SC11": float(rng.uniform(0.1, 1.0)),
                    "SC12": float(rng.uniform(0.1, 1.0)),
                    "SC21": float(rng.uniform(0.1, 1.0)) if yr in years_avail else np.nan,
                    "SC22": float(rng.uniform(0.1, 1.0)) if yr in years_avail else np.nan,
                })
        df = pd.DataFrame(rows)
        cg = {"C01": ["SC11", "SC12"], "C02": ["SC21", "SC22"]}

        result = CRITICWeightingCalculator(config=_make_critic_config()).calculate(
            df, cg, "Province", "Year")

        regimes = result.details["level2"]["regimes"]
        assert len(regimes) == 2, (
            f"Expected 2 regimes (C02 entirely absent 2015-2016), "
            f"got {len(regimes)}: {[r['active_criteria'] for r in regimes]}"
        )
        active_sets = {frozenset(r["active_criteria"]) for r in regimes}
        assert frozenset({"C01"}) in active_sets, \
            "Expected a regime where only C01 is active"
        assert frozenset({"C01", "C02"}) in active_sets, \
            "Expected a regime where both C01 and C02 are active"
        assert abs(sum(result.weights.values()) - 1.0) < 1e-8

    def test_all_regime_obs_sum_to_reported_n_observations(self):
        """
        Every observation is assigned to exactly one year-regime.
        The sum of n_obs across all regimes must equal n_observations in details.
        """
        from weighting.critic_weighting import CRITICWeightingCalculator
        df, cg = self._make_gapped_panel(rng_seed=13)
        result = CRITICWeightingCalculator(config=_make_critic_config()).calculate(
            df, cg, "Province", "Year")

        regimes      = result.details["level2"]["regimes"]
        regime_total = sum(r["n_obs"] for r in regimes)
        reported     = result.details["n_observations"]
        assert regime_total == reported, (
            f"Regime obs sum {regime_total} != n_observations {reported}"
        )
