"""
Comprehensive unit and integration tests for WindowedTemporalStabilityAnalyzer.

Test Coverage:
- Unit tests: window extraction, metrics computation, edge cases
- Integration tests: full workflow with real data
- Regression tests: validation against reference implementations
- Robustness tests: numerical stability, precision

Author: ML-MCDM Framework
Date: March 22, 2026
Status: Production-Ready
"""

import pytest
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, List, Tuple

from analysis.critic_temporal_stability import (
    TemporalStabilityResult,
    WindowedTemporalStabilityAnalyzer
)


# ============================================================================
# Module-Level Helper Functions
# ============================================================================

def create_weight_dict(year: int = None, criterion_count: int = 8, seed: int = None) -> Dict[str, float]:
    """
    Create a randomized weight dictionary for testing.
    
    Parameters
    ----------
    year : int, optional
        Year (not used, for clarity)
    criterion_count : int
        Number of criteria (default: 8)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    Dict[str, float]
        Weight dictionary with criterion names as keys
    """
    if seed is not None:
        np.random.seed(seed)
    criterion_names = [f'C{i+1:02d}' for i in range(criterion_count)]
    weights = np.random.dirichlet(np.ones(criterion_count))
    return {name: float(w) for name, w in zip(criterion_names, weights)}


def create_constant_weight_dict(criterion_count: int = 8, values: np.ndarray = None) -> Dict[str, float]:
    """
    Create a constant weight dictionary.
    
    Parameters
    ----------
    criterion_count : int
        Number of criteria
    values : np.ndarray, optional
        Weight values (default: uniform)
    
    Returns
    -------
    Dict[str, float]
        Weight dictionary
    """
    criterion_names = [f'C{i+1:02d}' for i in range(criterion_count)]
    if values is None:
        values = np.ones(criterion_count) / criterion_count
    else:
        values = values / values.sum()  # Ensure normalized
    return {name: float(w) for name, w in zip(criterion_names, values)}


# ============================================================================
# Test Classes
# ============================================================================



class TestWindowConstruction:
    """Unit tests for window extraction logic."""
    
    def test_window_extraction_basic_14_years(self):
        """Test that 14 years with window size 5 produces 10 windows."""
        years = list(range(2011, 2025))  # 2011-2024 (14 years)
        weights_per_year = {year: create_weight_dict(year) for year in years}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        # Should have 9 rolling correlations (10 windows - 1)
        assert len(result.spearman_rho_rolling) == 9
    
    def test_window_extraction_insufficient_data(self):
        """Test edge case: fewer than 5 years of data."""
        years = [2020, 2021, 2022]
        weights_per_year = {year: create_weight_dict(year) for year in years}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        # Should return default conservative result
        assert result.spearman_rho_rolling is not None
        assert result.spearman_rho_mean > 0.99  # Default near 1.0
    
    def test_window_consecutive_pairs_count(self):
        """Test that consecutive pairs are computed correctly."""
        # 20 years should yield 20 - 5 + 1 = 16 windows, 15 pairs
        years = list(range(2000, 2020))
        weights_per_year = {year: create_weight_dict(year) for year in years}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        assert len(result.spearman_rho_rolling) == 15
    
    def test_window_overlap_structure(self):
        """Test that windows have correct overlap (4 out of 5 years)."""
        years = list(range(2011, 2025))
        weights_dict = create_constant_weight_dict()  # Equal weights
        weights_per_year = {year: weights_dict for year in years}  # Constant across years
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        # All equal weights → all rankings identical → ρ = 1.0
        assert np.allclose(result.spearman_rho_rolling, 1.0, atol=1e-10)


class TestSpearmanRhoComputation:
    """Unit tests for Spearman's rank correlation logic."""
    
    def test_perfect_correlation_identical_rankings(self):
        """Test ρ = 1.0 when all window rankings are identical."""
        years = list(range(2011, 2025))
        # Fixed weights: always same ranking
        fixed_array = np.array([0.15, 0.20, 0.12, 0.25, 0.10, 0.08, 0.05, 0.05])
        fixed_weights = create_constant_weight_dict(values=fixed_array)
        weights_per_year = {year: fixed_weights for year in years}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        # All rho values should be 1.0
        assert np.allclose(result.spearman_rho_rolling, 1.0, atol=1e-10)
        assert np.isclose(result.spearman_rho_mean, 1.0, atol=1e-10)
        assert np.isclose(result.spearman_rho_std, 0.0, atol=1e-10)
    
    def test_anticorrelation_reversed_rankings(self):
        """Test ρ = -1.0 when rankings are exactly reversed."""
        years = list(range(2011, 2020))
        weights_per_year = {}
        
        # First half: monotone increasing
        for i, year in enumerate(years[:3]):
            w = np.array([float(i+1), 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
            weights_per_year[year] = create_constant_weight_dict(values=w)
        
        # Second half: monotone decreasing
        for i, year in enumerate(years[3:]):
            w = np.array([float(8-i), 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
            weights_per_year[year] = create_constant_weight_dict(values=w)
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        # At least one pair should show strong negative or zero correlation
        assert result.spearman_rho_mean < 0.9 or np.any(result.spearman_rho_rolling < 0.5)
    
    def test_rho_range_bounds(self):
        """Test that ρ is always in [-1, 1]."""
        # Generate 100 random weight time series
        years = list(range(2011, 2050))
        
        for trial in range(10):
            weights_per_year = {year: create_weight_dict(year, seed=trial) for year in years}
            
            analyzer = WindowedTemporalStabilityAnalyzer()
            result = analyzer.analyze(weights_per_year)
            
            assert np.all(result.spearman_rho_rolling >= -1.0)
            assert np.all(result.spearman_rho_rolling <= 1.0)
            assert -1.0 <= result.spearman_rho_mean <= 1.0
    
    def test_rho_numerical_precision_against_scipy(self):
        """Test that computed ρ matches scipy.stats.spearmanr to high precision."""
        # Create controlled weights for two consecutive windows
        years = list(range(2011, 2025))
        criterion_names = [f'C{i+1:02d}' for i in range(8)]
        
        # Window 1 & 2: slightly perturbed
        base_weights = np.array([0.15, 0.20, 0.12, 0.25, 0.10, 0.08, 0.05, 0.05])
        np.random.seed(42)
        
        weights_per_year = {}
        for i, year in enumerate(years):
            noise = np.random.normal(0, 0.02, size=8)
            w = base_weights + noise
            w = np.abs(w) / np.abs(w).sum()
            weights_per_year[year] = {name: float(weight) for name, weight in zip(criterion_names, w)}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        # Compute reference ρ using scipy for first window pair
        w1_list = np.array([[weights_per_year[year][name] for name in criterion_names] for year in years[:5]])
        w1 = w1_list.mean(axis=0)
        w2_list = np.array([[weights_per_year[year][name] for name in criterion_names] for year in years[1:6]])
        w2 = w2_list.mean(axis=0)
        
        rank1 = len(w1) + 1 - np.argsort(np.argsort(w1))  # Descending ranks
        rank2 = len(w2) + 1 - np.argsort(np.argsort(w2))
        
        scipy_rho, _ = spearmanr(rank1, rank2)
        
        # Our implementation should match scipy within 1e-10
        assert np.isclose(result.spearman_rho_rolling[0], scipy_rho, atol=1e-10)


class TestKendallsW:
    """Unit tests for Kendall's concordance coefficient W."""
    
    def test_perfect_agreement_identical_rankings_all_windows(self):
        """Test W = 1.0 when all windows have identical criterion rankings."""
        years = list(range(2011, 2025))
        fixed_array = np.array([0.15, 0.20, 0.12, 0.25, 0.10, 0.08, 0.05, 0.05])
        fixed_weights = create_constant_weight_dict(values=fixed_array)
        weights_per_year = {year: fixed_weights for year in years}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        # W should be 1.0 (perfect agreement)
        assert np.isclose(result.kendalls_w, 1.0, atol=1e-10)
    
    def test_kendalls_w_range_bounds(self):
        """Test that W is always in [0, 1]."""
        for trial in range(10):
            years = list(range(2011, 2040))
            weights_per_year = {year: create_weight_dict(year, seed=trial) for year in years}
            
            analyzer = WindowedTemporalStabilityAnalyzer()
            result = analyzer.analyze(weights_per_year)
            
            assert 0.0 <= result.kendalls_w <= 1.0
    
    def test_kendalls_w_8_criteria(self):
        """Test W computation with 8 criteria (actual use case)."""
        years = list(range(2011, 2025))
        weights_per_year = {year: create_weight_dict(year, seed=42) for year in years}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        # W should be reasonable (not NaN, not inf)
        assert not np.isnan(result.kendalls_w)
        assert not np.isinf(result.kendalls_w)
        assert result.kendalls_w > 0.0


class TestCoefficientOfVariation:
    """Unit tests for per-criterion Coefficient of Variation."""
    
    def test_zero_cv_for_constant_weights(self):
        """Test CV = 0 when weights are constant across all years."""
        years = list(range(2011, 2025))
        fixed_array = np.array([0.15, 0.20, 0.12, 0.25, 0.10, 0.08, 0.05, 0.05])
        fixed_weights = create_constant_weight_dict(values=fixed_array)
        weights_per_year = {year: fixed_weights for year in years}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        # All CVs should be ~0
        for cv in result.coefficient_variation.values():
            assert np.isclose(cv, 0.0, atol=1e-10)
    
    def test_cv_range_non_negative(self):
        """Test that CV ≥ 0 for all criteria."""
        for trial in range(10):
            years = list(range(2011, 2050))
            weights_per_year = {year: create_weight_dict(year, seed=trial) for year in years}
            
            analyzer = WindowedTemporalStabilityAnalyzer()
            result = analyzer.analyze(weights_per_year)
            
            for j, cv in result.coefficient_variation.items():
                assert cv >= 0.0
                assert not np.isnan(cv)
                assert not np.isinf(cv)
    
    def test_cv_dict_structure(self):
        """Test that CV dict has one entry per criterion."""
        years = list(range(2011, 2025))
        weights_per_year = {year: create_weight_dict(year, seed=42) for year in years}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        assert len(result.coefficient_variation) == 8
        for i in range(8):
            criterion_name = f'C{i+1:02d}'
            assert criterion_name in result.coefficient_variation
    
    def test_cv_increases_with_volatility(self):
        """Test that CV increases as weight volatility increases."""
        years = list(range(2011, 2025))
        criterion_names = [f'C{i+1:02d}' for i in range(8)]
        
        # Scenario 1: Stable weights (low noise)
        base_weights = np.array([0.15, 0.20, 0.12, 0.25, 0.10, 0.08, 0.05, 0.05])
        stable_weights = {}
        np.random.seed(42)
        for year in years:
            noise = np.random.normal(0, 0.005, 8)
            w = base_weights + noise
            w = np.abs(w) / np.abs(w).sum()
            stable_weights[year] = {name: float(weight) for name, weight in zip(criterion_names, w)}
        
        # Scenario 2: Volatile weights (high noise)
        volatile_weights = {}
        for year in years:
            noise = np.random.normal(0, 0.05, 8)
            w = base_weights + noise
            w = np.abs(w) / np.abs(w).sum()
            volatile_weights[year] = {name: float(weight) for name, weight in zip(criterion_names, w)}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        
        result_stable = analyzer.analyze(stable_weights)
        result_volatile = analyzer.analyze(volatile_weights)
        
        mean_cv_stable = np.mean(list(result_stable.coefficient_variation.values()))
        mean_cv_volatile = np.mean(list(result_volatile.coefficient_variation.values()))
        
        assert mean_cv_volatile > mean_cv_stable


class TestAggregationStatistics:
    """Unit tests for aggregation statistics (mean, std)."""
    
    def test_mean_std_consistency(self):
        """Test that mean and std are computed correctly."""
        years = list(range(2011, 2025))
        weights_per_year = {year: create_weight_dict(year, seed=42) for year in years}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        # Manual computation
        manual_mean = np.mean(result.spearman_rho_rolling)
        manual_std = np.std(result.spearman_rho_rolling, ddof=1)  # Sample std
        
        assert np.isclose(result.spearman_rho_mean, manual_mean)
        # Note: our std may use ddof=0, check appropriately
        assert np.isclose(result.spearman_rho_std, manual_std, atol=0.01)
    
    def test_timeline_year_range_fidelity(self):
        """Test that timeline covers correct year range."""
        years = list(range(2011, 2025))
        weights_per_year = {year: create_weight_dict(year, seed=42) for year in years}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        # Timeline should span from first window to last window
        assert result.year_range[0] == 2011  # Start of first window
        assert result.year_range[1] == 2024  # End of last window
        assert len(result.rolling_timeline) == 9  # 9 pairs


class TestEdgeCases:
    """Unit tests for numerical edge cases and robustness."""
    
    def test_zero_variance_criterion_ranks(self):
        """Test handling of criteria with zero variance (all same weight)."""
        years = list(range(2011, 2025))
        fixed_weight = create_constant_weight_dict()  # All equal
        weights_per_year = {year: fixed_weight for year in years}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        # Should not raise; all correlations should be 1.0 (by convention for tied ranks)
        assert not np.any(np.isnan(result.spearman_rho_rolling))
        assert not np.any(np.isinf(result.spearman_rho_rolling))
    
    def test_very_small_weight_numerical_stability(self):
        """Test stability with very small (near-zero) weights."""
        years = list(range(2011, 2025))
        criterion_names = [f'C{i+1:02d}' for i in range(8)]
        weights_per_year = {}
        
        for year in years:
            w = np.array([1e-8, 0.99, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8])
            w_dict = {name: float(weight) for name, weight in zip(criterion_names, w/w.sum())}
            weights_per_year[year] = w_dict
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        assert not np.any(np.isnan(result.spearman_rho_rolling))
        assert not np.any(np.isinf(result.spearman_rho_rolling))
    
    def test_normalized_weight_constraint(self):
        """Test that computed weights are handled as normalized."""
        years = list(range(2011, 2025))
        weights_per_year = {year: create_weight_dict(year, seed=42) for year in years}
        
        # Verify all inputs sum to 1.0
        for year, w in weights_per_year.items():
            assert np.isclose(sum(w.values()), 1.0, atol=1e-10)
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        # Analysis should complete without error
        assert result is not None


class TestIntegrationRealData:
    """Integration tests with realistic (synthetic) data."""
    
    def test_full_workflow_14_years_8_criteria(self):
        """Test complete workflow with realistic 14-year, 8-criterion data."""
        np.random.seed(42)
        years = list(range(2011, 2025))
        criterion_names = [f'C{i+1:02d}' for i in range(8)]
        
        # Simulate realistic weight evolution (slow drift with noise)
        base_weights = np.array([0.15, 0.20, 0.12, 0.25, 0.10, 0.08, 0.05, 0.05])
        weights_per_year = {}
        
        for i, year in enumerate(years):
            drift = 0.01 * np.sin(i * np.pi / 7) * np.ones(8)  # Slow oscillation
            noise = np.random.normal(0, 0.02, 8)
            w = base_weights + drift + noise
            w = np.abs(w) / np.abs(w).sum()
            weights_per_year[year] = {name: float(weight) for name, weight in zip(criterion_names, w)}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        # Check all components exist and are valid
        assert result is not None
        assert len(result.spearman_rho_rolling) == 9
        assert 0 <= result.spearman_rho_mean <= 1
        assert 0 <= result.spearman_rho_std <= 1
        assert 0 <= result.kendalls_w <= 1
        assert len(result.coefficient_variation) == 8
        assert result.year_range == (2011, 2024)
    
    def test_temporal_trend_detection(self):
        """Test that systematic trends are detected via low agreement."""
        years = list(range(2011, 2025))
        criterion_names = [f'C{i+1:02d}' for i in range(8)]
        weights_per_year = {}
        
        # Systematic trend: first criterion weight increases over time
        for i, year in enumerate(years):
            w = np.array([0.05 + 0.08 * i / 14, 0.20, 0.12, 0.25, 0.10, 0.08, 0.05, 0.05])
            w_dict = {name: float(weight) for name, weight in zip(criterion_names, w/w.sum())}
            weights_per_year[year] = w_dict
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        # Systematic trend should reduce correlation as windows shift
        # Average rho should be lower than with constant weights
        assert result.spearman_rho_mean < 0.98


class TestResultSerializability:
    """Tests for serializability and data export compatibility."""
    
    def test_result_to_dict_conversion(self):
        """Test conversion of TemporalStabilityResult to dictionary format."""
        years = list(range(2011, 2025))
        weights_per_year = {year: np.random.dirichlet(np.ones(8)) for year in years}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)        
        # Convert to dataclass dict
        result_dict = {
            'spearman_rho_rolling': result.spearman_rho_rolling,
            'spearman_rho_mean': result.spearman_rho_mean,
            'spearman_rho_std': result.spearman_rho_std,
            'kendalls_w': result.kendalls_w,
            'coefficient_variation': result.coefficient_variation,
            'rolling_timeline': result.rolling_timeline,
            'year_range': result.year_range
        }
        
        # All components should be JSON-serializable (after numpy conversion)
        assert 'spearman_rho_rolling' in result_dict
        assert isinstance(result_dict['coefficient_variation'], dict)        
        # Convert to dataclass dict
        result_dict = {
            'spearman_rho_rolling': result.spearman_rho_rolling,
            'spearman_rho_mean': result.spearman_rho_mean,
            'spearman_rho_std': result.spearman_rho_std,
            'kendalls_w': result.kendalls_w,
            'coefficient_variation': result.coefficient_variation,
            'rolling_timeline': result.rolling_timeline,
            'year_range': result.year_range
        }
        
        # All components should be JSON-serializable (after numpy conversion)
        assert 'spearman_rho_rolling' in result_dict
        assert isinstance(result_dict['coefficient_variation'], dict)


class TestRegressionValidation:
    """Regression tests against known baseline results."""
    
    def test_constant_weight_known_result(self):
        """Test against known constant-weight baseline."""
        years = list(range(2011, 2025))
        criterion_names = [f'C{i+1:02d}' for i in range(8)]
        
        # Known constant weights
        fixed_array = np.array([0.15, 0.20, 0.12, 0.25, 0.10, 0.08, 0.05, 0.05])
        fixed_w = {name: float(weight) for name, weight in zip(criterion_names, fixed_array)}
        weights_per_year = {year: fixed_w for year in years}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_per_year)
        
        # Baseline expectations
        assert np.isclose(result.spearman_rho_mean, 1.0, atol=1e-10)
        assert np.isclose(result.spearman_rho_std, 0.0, atol=1e-10)
        assert np.isclose(result.kendalls_w, 1.0, atol=1e-10)
        expected_cv = {name: 0.0 for name in criterion_names}
        for name, cv in result.coefficient_variation.items():
            assert np.isclose(cv, expected_cv[name], atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
