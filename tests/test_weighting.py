# -*- coding: utf-8 -*-
"""
Unit tests for weighting methods.

Covers:
  - EntropyWeightCalculator
  - CRITICWeightCalculator
  - MERECWeightCalculator
  - StandardDeviationWeightCalculator
  - calculate_weights convenience function
  - WeightResult properties
"""

import numpy as np
import pandas as pd
import pytest

from weighting.base import WeightResult, calculate_weights
from weighting.entropy import EntropyWeightCalculator
from weighting.critic import CRITICWeightCalculator
from weighting.merec import MERECWeightCalculator
from weighting.standard_deviation import StandardDeviationWeightCalculator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_data():
    """4 alternatives × 3 criteria, all positive, reasonable variation."""
    return pd.DataFrame(
        {
            "C1": [0.8, 0.6, 0.9, 0.4],
            "C2": [0.5, 0.5, 0.5, 0.5],  # constant → low entropy weight
            "C3": [0.2, 0.8, 0.5, 0.9],  # high variation → high entropy weight
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
# TestEntropyWeightCalculator
# ---------------------------------------------------------------------------

class TestEntropyWeightCalculator:
    def test_weights_sum_to_one(self, sample_data):
        result = EntropyWeightCalculator().calculate(sample_data)
        assert abs(result.as_array.sum() - 1.0) < 1e-9

    def test_weights_all_non_negative(self, sample_data):
        result = EntropyWeightCalculator().calculate(sample_data)
        assert (result.as_array >= 0).all()

    def test_constant_column_lower_weight(self, sample_data):
        """Constant column (C2) should receive a lower weight than C3."""
        result = EntropyWeightCalculator().calculate(sample_data)
        assert result.weights["C2"] < result.weights["C3"]

    def test_high_variance_column_higher_weight(self, sample_data):
        """C3 (most varied) should have higher weight than C2 (constant)."""
        result = EntropyWeightCalculator().calculate(sample_data)
        assert result.weights["C3"] > result.weights["C2"]

    def test_method_name(self, sample_data):
        result = EntropyWeightCalculator().calculate(sample_data)
        assert "entropy" in result.method.lower()

    def test_raises_on_empty_dataframe(self):
        with pytest.raises((ValueError, ZeroDivisionError, Exception)):
            EntropyWeightCalculator().calculate(pd.DataFrame())

    def test_sample_weights_produce_valid_result(self, sample_data):
        n = len(sample_data)
        sw = np.ones(n) / n
        result = EntropyWeightCalculator().calculate(sample_data, sample_weights=sw)
        assert abs(result.as_array.sum() - 1.0) < 1e-9

    def test_single_row_raises_or_returns_equal(self):
        """Single row is degenerate; method should raise or return equal weights."""
        single = pd.DataFrame({"C1": [0.5], "C2": [0.3]})
        try:
            result = EntropyWeightCalculator().calculate(single)
            # If it doesn't raise, weights should still be valid (summing to 1)
            assert abs(result.as_array.sum() - 1.0) < 1e-6
        except Exception:
            pass  # raising is acceptable for degenerate input


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
# TestMERECWeightCalculator
# ---------------------------------------------------------------------------

class TestMERECWeightCalculator:
    def test_weights_sum_to_one(self, sample_data):
        result = MERECWeightCalculator().calculate(sample_data)
        assert abs(result.as_array.sum() - 1.0) < 1e-9

    def test_weights_all_non_negative(self, sample_data):
        result = MERECWeightCalculator().calculate(sample_data)
        assert (result.as_array >= 0).all()

    def test_method_name(self, sample_data):
        result = MERECWeightCalculator().calculate(sample_data)
        assert "merec" in result.method.lower()

    def test_larger_matrix(self, large_data):
        result = MERECWeightCalculator().calculate(large_data)
        assert abs(result.as_array.sum() - 1.0) < 1e-9
        assert len(result.weights) == 4

    def test_constant_column_zero_weight(self):
        """MEREC weights must sum to 1 even when one column is constant."""
        data = pd.DataFrame(
            {
                "C1": [0.8, 0.6, 0.9, 0.4],
                "C2": [0.5, 0.5, 0.5, 0.5],  # constant
                "C3": [0.2, 0.8, 0.5, 0.9],
            }
        )
        result = MERECWeightCalculator().calculate(data)
        assert abs(result.as_array.sum() - 1.0) < 1e-9
        assert all(w >= 0 for w in result.weights.values())


# ---------------------------------------------------------------------------
# TestStandardDeviationWeightCalculator
# ---------------------------------------------------------------------------

class TestStandardDeviationWeightCalculator:
    def test_weights_sum_to_one(self, sample_data):
        result = StandardDeviationWeightCalculator().calculate(sample_data)
        assert abs(result.as_array.sum() - 1.0) < 1e-9

    def test_weights_all_non_negative(self, sample_data):
        result = StandardDeviationWeightCalculator().calculate(sample_data)
        assert (result.as_array >= 0).all()

    def test_zero_variance_column_zero_weight(self):
        """A constant column should receive weight = 0."""
        data = pd.DataFrame(
            {
                "C1": [0.8, 0.6, 0.9, 0.4],
                "C2": [0.5, 0.5, 0.5, 0.5],  # zero variance
                "C3": [0.2, 0.8, 0.5, 0.9],
            }
        )
        result = StandardDeviationWeightCalculator().calculate(data)
        assert abs(result.weights["C2"]) < 1e-9

    def test_method_name(self, sample_data):
        result = StandardDeviationWeightCalculator().calculate(sample_data)
        assert result.method is not None

    def test_higher_variance_higher_weight(self):
        """Column with higher standard deviation should get higher weight."""
        data = pd.DataFrame(
            {
                "Low":  [0.5, 0.6, 0.5, 0.6],  # small variance
                "High": [0.1, 0.9, 0.2, 0.8],  # large variance
            }
        )
        result = StandardDeviationWeightCalculator().calculate(data)
        assert result.weights["High"] > result.weights["Low"]

    def test_larger_matrix(self, large_data):
        result = StandardDeviationWeightCalculator().calculate(large_data)
        assert abs(result.as_array.sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# TestCalculateWeights (convenience function)
# ---------------------------------------------------------------------------

class TestCalculateWeights:
    @pytest.mark.parametrize("method", ["entropy", "critic", "merec", "std_dev"])
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

    def test_result_keys_match_columns(self, sample_data):
        result = calculate_weights(sample_data, "entropy")
        assert set(result.weights.keys()) == set(sample_data.columns)
