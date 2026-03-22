# -*- coding: utf-8 -*-
"""
Test: Forecast Visualization Metrics and Formulas

Tests for deterministic mathematical functions used across all chart modules:
R², RMSE, MAE, MAPE, conformal quantiles, bootstrap CI helpers.
Ensures metric consistency and statistical correctness.
"""

import pytest
import numpy as np
from typing import Tuple


# ============================================================================
# Metric Formula Reference (Phase 1 Target)
# ============================================================================

def r2_score_reference(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Reference implementation of R² for test validation.
    
    R² = 1 - (SS_res / SS_tot)
    where SS_res = Σ(y_true - y_pred)²
          SS_tot = Σ(y_true - mean(y_true))²
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    
    return float(1 - ss_res / (ss_tot + eps))


def rmse_reference(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Reference RMSE = sqrt(mean((y_true - y_pred)²))"""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_reference(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Reference MAE = mean(|y_true - y_pred|)"""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    return float(np.mean(np.abs(y_true - y_pred)))


def mape_reference(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """Reference MAPE = 100 * mean(|y_true - y_pred| / |y_true|)
    
    Clips denominator to eps to avoid division by zero.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    denom = np.where(np.abs(y_true) < eps, 1.0, np.abs(y_true))
    return float(100.0 * np.mean(np.abs(y_true - y_pred) / denom))


# ============================================================================
# Tests: R² Formula (Phase 1)
# ============================================================================

class TestR2Metric:
    """Tests for R² score calculation and edge cases."""

    def test_r2_perfect_prediction(self):
        """R² = 1.0 when predictions are perfect."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r2 = r2_score_reference(y_true, y_pred)
        assert np.isclose(r2, 1.0), f"Expected R²=1.0, got {r2}"

    def test_r2_zero_prediction(self):
        """R² = 0.0 when model predicts mean (baseline)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y_true, y_true.mean())  # Always predict mean
        r2 = r2_score_reference(y_true, y_pred)
        assert np.isclose(r2, 0.0), f"Expected R²=0.0, got {r2}"

    def test_r2_negative_prediction(self):
        """R² < 0.0 when model worse than baseline."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Reversed
        r2 = r2_score_reference(y_true, y_pred)
        assert r2 < 0.0, f"Expected R²<0.0, got {r2}"

    def test_r2_constant_target(self):
        """R² with constant target handled gracefully (with epsilon guard)."""
        y_true = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        y_pred = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        r2 = r2_score_reference(y_true, y_pred)
        # SS_tot = 0, formula uses epsilon to avoid division by zero
        assert np.isfinite(r2), f"R² should be finite, got {r2}"

    def test_r2_with_nan_handling(self):
        """R² ignores NaN values (implementation responsibility)."""
        y_true = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Reference assumes clean data; validators should filter NaN upstream
        # This test documents the assumption
        pytest.skip("Validators filter NaN before metrics.py is called")


# ============================================================================
# Tests: RMSE and MAE Metrics (Phase 1)
# ============================================================================

class TestRMSEMetric:
    """Tests for RMSE (Root Mean Squared Error)."""

    def test_rmse_zero_error(self):
        """RMSE = 0 when predictions perfect."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        rmse = rmse_reference(y_true, y_pred)
        assert np.isclose(rmse, 0.0)

    def test_rmse_calculation(self):
        """RMSE = sqrt(mean([0.1², 0.2², 0.3²])) ≈ 0.1886."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 3.3])
        rmse = rmse_reference(y_true, y_pred)
        expected = np.sqrt(np.mean([0.01, 0.04, 0.09]))
        assert np.isclose(rmse, expected)

    def test_rmse_large_error(self):
        """RMSE is always non-negative."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([5.0, 6.0, 7.0])
        rmse = rmse_reference(y_true, y_pred)
        assert rmse > 0


class TestMAEMetric:
    """Tests for MAE (Mean Absolute Error)."""

    def test_mae_zero_error(self):
        """MAE = 0 when predictions perfect."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        mae = mae_reference(y_true, y_pred)
        assert np.isclose(mae, 0.0)

    def test_mae_calculation(self):
        """MAE = mean([0.1, 0.2, 0.3]) = 0.2."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 3.3])
        mae = mae_reference(y_true, y_pred)
        expected = 0.2
        assert np.isclose(mae, expected)

    def test_mae_vs_rmse(self):
        """MAE <= RMSE (MAE less sensitive to outliers)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 100.1])
        mae = mae_reference(y_true, y_pred)
        rmse = rmse_reference(y_true, y_pred)
        assert mae <= rmse


class TestMAPEMetric:
    """Tests for MAPE (Mean Absolute Percentage Error)."""

    def test_mape_perfect_prediction(self):
        """MAPE = 0 when perfect."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        mape = mape_reference(y_true, y_pred)
        assert np.isclose(mape, 0.0)

    def test_mape_10_percent_error(self):
        """MAPE = 10% when all predictions 10% off."""
        y_true = np.array([100.0, 100.0, 100.0])
        y_pred = np.array([110.0, 110.0, 110.0])
        mape = mape_reference(y_true, y_pred)
        assert np.isclose(mape, 10.0)

    def test_mape_handles_zero_denominator(self):
        """MAPE handles zero denominator without dividing by zero."""
        y_true = np.array([0.0, 10.0, 20.0])
        y_pred = np.array([0.1, 10.1, 20.1])
        mape = mape_reference(y_true, y_pred)
        # Should not raise or produce inf/nan
        assert np.isfinite(mape)


# ============================================================================
# Tests: Conformal Prediction Metrics (Phase 1)
# ============================================================================

class TestConformalQuantile:
    """Tests for conformal prediction quantile calculation."""

    def test_conformal_quantile_computation(self):
        """Conformal quantile = ceiling((n+1) * (1-α)) / n -th order statistic."""
        # With n=100 samples and α=0.05 (95% confidence):
        # quantile_idx = ceil((100+1) * 0.95) - 1 = ceil(95.95) - 1 = 95
        # This ensures finite-sample coverage guarantee
        
        n = 100
        residuals = np.abs(np.random.randn(n))
        alpha = 0.05
        
        # Compute following conformal prediction finite-sample convention
        q_idx = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
        q_idx = min(q_idx, n - 1)  # Clamp to valid range
        
        q_val = float(np.sort(residuals)[q_idx])
        assert np.isfinite(q_val)
        assert q_val >= 0.0

    def test_coverage_property(self):
        """Conformal interval contains at least (1-α) fraction of test data."""
        np.random.seed(42)
        y_true = np.random.randn(200)
        y_pred = y_true + np.random.randn(200) * 0.3
        residuals = np.abs(y_true - y_pred)
        
        alpha = 0.05
        n = len(residuals)
        q_idx = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
        q_idx = min(q_idx, n - 1)
        q_val = float(np.sort(residuals)[q_idx])
        
        # Empirical coverage
        empirical_coverage = float(np.mean(residuals <= q_val))
        assert empirical_coverage >= (1.0 - alpha - 0.05)  # Allow 5% tolerance


# ============================================================================
# Tests: Bootstrap CI Helper (Phase 1)
# ============================================================================

class TestBootstrapCI:
    """Tests for bootstrap confidence interval calculation."""

    def test_bootstrap_ci_reproducibility(self):
        """Bootstrap CI with same seed produces same result."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        
        def compute_ci(seed):
            rng = np.random.RandomState(seed)
            boot_r2 = []
            for _ in range(100):
                idx = rng.choice(len(y_true), size=len(y_true), replace=True)
                r2 = r2_score_reference(y_true[idx], y_pred[idx])
                boot_r2.append(r2)
            boot_r2 = np.array(boot_r2)
            return (np.mean(boot_r2), np.percentile(boot_r2, 2.5), np.percentile(boot_r2, 97.5))
        
        ci1 = compute_ci(42)
        ci2 = compute_ci(42)
        assert np.allclose(ci1, ci2)

    def test_bootstrap_ci_bounds(self):
        """Bootstrap CI: lower < mean < upper."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.05, 2.05, 2.95, 4.05, 4.95])
        
        rng = np.random.RandomState(42)
        boot_r2 = []
        for _ in range(100):
            idx = rng.choice(len(y_true), size=len(y_true), replace=True)
            r2 = r2_score_reference(y_true[idx], y_pred[idx])
            boot_r2.append(r2)
        boot_r2 = np.array(boot_r2)
        
        lo = np.percentile(boot_r2, 2.5)
        mean = np.mean(boot_r2)
        hi = np.percentile(boot_r2, 97.5)
        
        assert lo <= mean <= hi


# ============================================================================
# Tests: Phase 1 Implementation Target
# ============================================================================

class TestMetricsPhase1:
    """Placeholder for Phase 1 metrics.py module implementation."""

    def test_metrics_module_exists(self):
        """metrics.py module can be imported."""
        # TODO: Once Phase 1 contracts/validators/metrics are created:
        # from output.visualization.forecast.metrics import (
        #     r2_score, rmse_score, mae_score, conformal_quantile, bootstrap_ci
        # )
        pytest.skip("Awaiting Phase 1 metrics.py implementation")

    def test_all_metric_formulas_consistent(self):
        """All metric implementations match reference formulas."""
        # TODO: Test metrics.r2_score matches r2_score_reference
        pytest.skip("Awaiting Phase 1 metrics.py implementation")

    def test_metric_edge_case_handling(self):
        """Metrics handle edge cases (constant target, all NaN, single value)."""
        # TODO: Test guard conditions and graceful degradation
        pytest.skip("Awaiting Phase 1 metrics.py implementation")


__all__ = [
    'TestR2Metric',
    'TestRMSEMetric',
    'TestMAEMetric',
    'TestMAPEMetric',
    'TestConformalQuantile',
    'TestBootstrapCI',
    'TestMetricsPhase1',
]
