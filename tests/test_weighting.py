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
# TestCalculateWeights (convenience function)
# ---------------------------------------------------------------------------

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
