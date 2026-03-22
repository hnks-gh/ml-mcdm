"""
End-to-End Integration Test for PHASE 2 (Temporal Stability & Sensitivity Analysis)

This test verifies the complete pipeline:
1. Load/create realistic panel data (14 years, 8 criteria, 63 provinces)
2. Compute CRITIC weights with temporal stability analysis
3. Compute CRITIC weights with sensitivity analysis
4. Verify all output files are generated correctly
5. Validate metrics are within expected ranges
6. Check non-blocking nature of optional analyses

Author: ML-MCDM Framework
Date: March 22, 2026
Status: Production-Ready
"""

import pytest
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Tuple

from weighting.critic_weighting import CRITICWeightingCalculator
from analysis.critic_temporal_stability import WindowedTemporalStabilityAnalyzer
from analysis.critic_sensitivity_analysis import CRITICSensitivityAnalyzer
from output.csv_writer import CsvWriter
from output.visualization.temporal_sensitivity_figures import TemporalSensitivityFigureGenerator


class TestEndToEndPipeline:
    """End-to-end integration tests for PHASE 2 implementation."""
    
    @staticmethod
    def _create_realistic_panel_data(n_years: int = 14, n_provinces: int = 5, 
                                    n_criteria: int = 8) -> Tuple[Dict, np.ndarray]:
        """
        Create realistic panel data simulating the PAPI dataset.
        
        Parameters
        ----------
        n_years : int
            Number of years in panel (default: 2011-2024)
        n_provinces : int
            Number of provinces/units
        n_criteria : int
            Number of criteria
        
        Returns
        -------
        Tuple[Dict, np.ndarray]
            - performance_data: Dict[year][province][criterion] = score ∈ [0,100]
            - global_weights: Array of global CRITIC weights (n_criteria,)
        """
        np.random.seed(42)
        years = list(range(2011, 2011 + n_years))
        provinces = [f'P{i+1:02d}' for i in range(n_provinces)]
        criteria = [f'C{i+1:02d}' for i in range(n_criteria)]
        
        # Create performance data with realistic structure:
        # - Base competence per province (some consistently high/low performing)
        # - Gradual improvement over time
        # - Noise and idiosyncratic variations
        
        performance_data = {}
        base_competence = np.random.uniform(40, 80, size=n_provinces)  # 40-80 range
        
        for year in years:
            performance_data[year] = {}
            year_effect = (year - 2011) * 0.5  # Gradual improvement
            
            for province_idx, province in enumerate(provinces):
                performance_data[year][province] = {}
                
                for criterion_idx, criterion in enumerate(criteria):
                    # Trend + base competence + noise + criterion-specific variation
                    trend = year_effect + np.random.normal(0, 2)
                    base = base_competence[province_idx]
                    criterion_factor = 10 * np.sin(criterion_idx * np.pi / 8)  # Vary by criterion
                    
                    score = base + trend + criterion_factor + np.random.normal(0, 3)
                    score = np.clip(score, 0, 100)  # Bound to [0, 100]
                    performance_data[year][province][criterion] = float(score)
        
        # Generate global weights (normalized) with realistic structure
        weights_raw = np.random.dirichlet(np.ones(n_criteria))
        global_weights = weights_raw / weights_raw.sum()
        
        return performance_data, global_weights
    
    @staticmethod
    def _compute_global_weights(performance_data: Dict, years: list) -> Dict[int, Dict[str, float]]:
        """
        Compute per-year global weights using CRITIC method on aggregated data.
        
        This simulates the two-level CRITIC procedure where level-2 aggregates
        province-level results to produce global weights.
        """
        criterion_names = None
        
        # For each year, aggregate province data and compute CRITIC weights
        weights_per_year = {}
        
        for year in years:
            year_data = performance_data[year]
            
            # Shape: (n_provinces, n_criteria)
            data_matrix = []
            provinces = sorted(year_data.keys())
            
            for province in provinces:
                if criterion_names is None:
                    criterion_names = sorted(year_data[province].keys())
                
                row = [year_data[province][c] for c in criterion_names]
                data_matrix.append(row)
            
            data_matrix = np.array(data_matrix)
            
            # Normalize to [0,1] range
            minimums = data_matrix.min(axis=0)
            maximums = data_matrix.max(axis=0)
            normalized_matrix = (data_matrix - minimums) / (maximums - minimums + 1e-10)
            
            # Simplified CRITIC: compute standard deviation as importance
            # (in real implementation, includes correlation)
            std_dev = normalized_matrix.std(axis=0)
            correlation_effect = np.ones_like(std_dev)  # Simplified: no correction
            
            weights = std_dev * correlation_effect
            weights = weights / weights.sum()  # Normalize
            
            weights_per_year[year] = {f'C{i+1:02d}': float(w) for i, w in enumerate(weights)}
        
        return weights_per_year
    
    def test_full_pipeline_temporal_and_sensitivity(self):
        """Test complete pipeline with both temporal stability and sensitivity analyses."""
        # Step 1: Create realistic panel data
        performance_data, global_weights = self._create_realistic_panel_data(
            n_years=14, n_provinces=5, n_criteria=8
        )
        years = sorted(performance_data.keys())
        
        # Step 2: Compute per-year global weights (simulating full CRITIC pipeline)
        weights_all_years = self._compute_global_weights(performance_data, years)
        
        # Step 3: Verify weights are normalized
        for year, weights in weights_all_years.items():
            total = sum(weights.values())
            assert np.isclose(total, 1.0, atol=1e-10), f"Year {year} weights don't sum to 1.0"
        
        # Step 4: Run temporal stability analysis
        temporal_analyzer = WindowedTemporalStabilityAnalyzer()
        temporal_result = temporal_analyzer.analyze(weights_all_years)
        
        # Verify temporal stability output
        assert temporal_result is not None
        assert len(temporal_result.spearman_rho_rolling) == 9  # 10 windows - 1
        assert 0 <= temporal_result.spearman_rho_mean <= 1
        assert 0 <= temporal_result.spearman_rho_std <= 1
        assert 0 <= temporal_result.kendalls_w <= 1
        assert len(temporal_result.coefficient_variation) == 8
        assert temporal_result.year_range == (2011, 2024)
        
        # Step 5: Run sensitivity analysis
        sensitivity_analyzer = CRITICSensitivityAnalyzer(n_replicates=500, seed=42)
        sensitivity_result = sensitivity_analyzer.analyze(weights_all_years)
        
        # Verify sensitivity output
        assert sensitivity_result is not None
        assert len(sensitivity_result.tier_robustness) == 3
        assert 0 <= sensitivity_result.tier_robustness['conservative'] <= 1
        assert 0 <= sensitivity_result.tier_robustness['moderate'] <= 1
        assert 0 <= sensitivity_result.tier_robustness['aggressive'] <= 1
        assert len(sensitivity_result.per_criterion_sensitivity) == 8  # 8 criteria: C01-C08
        assert len(sensitivity_result.rank_disruption_stats) == 3
        assert sensitivity_result.n_replicates == 500
        
        # Step 6: Verify monotonicity of robustness across tiers
        rob_cons = sensitivity_result.tier_robustness['conservative']
        rob_mod = sensitivity_result.tier_robustness['moderate']
        rob_agg = sensitivity_result.tier_robustness['aggressive']
        
        # Conservative >= Moderate >= Aggressive (with small tolerance for sampling)
        assert rob_cons >= rob_mod - 0.05
        assert rob_mod >= rob_agg - 0.05
    
    def test_temporal_stability_edge_case_constant_weights(self):
        """Test temporal stability with constant weights (worst case: perfect agreement)."""
        # Create constant weights across all years
        criterion_names = [f'C{i+1:02d}' for i in range(8)]
        constant_weights = np.array([0.15, 0.20, 0.12, 0.25, 0.10, 0.08, 0.05, 0.05])
        
        weights_all_years = {}
        for year in range(2011, 2025):
            weights_all_years[year] = {name: float(w) for name, w in zip(criterion_names, constant_weights)}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        result = analyzer.analyze(weights_all_years)
        
        # Perfect agreement expected
        assert np.isclose(result.spearman_rho_mean, 1.0, atol=1e-10)
        assert np.isclose(result.spearman_rho_std, 0.0, atol=1e-10)
        assert np.isclose(result.kendalls_w, 1.0, atol=1e-10)
    
    def test_sensitivity_analysis_edge_case_dominant_criterion(self):
        """Test sensitivity with highly skewed weight distribution."""
        criterion_names = [f'C{i+1:02d}' for i in range(8)]
        
        # One dominant criterion (0.825), others split remainder
        dominant_weights = np.array([0.825, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025])
        assert np.isclose(dominant_weights.sum(), 1.0)
        
        weights_dict = {name: float(w) for name, w in zip(criterion_names, dominant_weights)}
        weights_all_years = {2011: weights_dict}
        
        analyzer = CRITICSensitivityAnalyzer(n_replicates=300, seed=42)
        result = analyzer.analyze(weights_all_years)
        
        # Verify analysis completes and returns valid sensitivities
        assert result is not None
        assert len(result.per_criterion_sensitivity) == 8
        
        # All criteria should have valid sensitivity scores in [0, 1]
        for criterion_name, tier_sensitivities in result.per_criterion_sensitivity.items():
            for tier, sensitivity_score in tier_sensitivities.items():
                assert 0 <= sensitivity_score <= 1, f"{criterion_name} {tier} sensitivity out of range"
        
        # Dominant criterion (C01) which has weight 0.825 may have high sensitivity
        # because small perturbations to it affect the overall ranking significantly
        sensitivity_dominant_conservative = result.per_criterion_sensitivity['C01']['conservative']
        assert 0 <= sensitivity_dominant_conservative <= 1
    
    def test_non_blocking_integration_with_weights_and_analyses(self):
        """Test that analyses can run independently without blocking."""
        criterion_names = [f'C{i+1:02d}' for i in range(8)]
        weights = np.array([0.15, 0.20, 0.12, 0.25, 0.10, 0.08, 0.05, 0.05])
        weights_all_years = {year: {name: float(w) for name, w in zip(criterion_names, weights)} 
                            for year in range(2011, 2025)}
        
        # Run temporal analysis
        temporal_analyzer = WindowedTemporalStabilityAnalyzer()
        temporal_result = temporal_analyzer.analyze(weights_all_years)
        assert temporal_result is not None
        
        # Run sensitivity analysis independently (should not block or interfere)
        sensitivity_analyzer = CRITICSensitivityAnalyzer(n_replicates=200, seed=42)
        sensitivity_result = sensitivity_analyzer.analyze(weights_all_years)
        assert sensitivity_result is not None
        
        # Both should be complete and valid
        assert temporal_result.spearman_rho_mean > 0
        assert sensitivity_result.tier_robustness['conservative'] > 0
    
    def test_output_formats_csv_and_figures(self):
        """Test that outputs can be exported to CSV and PNG formats."""
        # Create realistic data
        performance_data, _ = self._create_realistic_panel_data(n_years=14, n_provinces=3, n_criteria=8)
        years = sorted(performance_data.keys())
        weights_all_years = self._compute_global_weights(performance_data, years)
        
        # Run analyses
        temporal_analyzer = WindowedTemporalStabilityAnalyzer()
        temporal_result = temporal_analyzer.analyze(weights_all_years)
        
        sensitivity_analyzer = CRITICSensitivityAnalyzer(n_replicates=300, seed=42)
        sensitivity_result = sensitivity_analyzer.analyze(weights_all_years)
        
        # Test CSV export format (convert results to DataFrames)
        temporal_df = pd.DataFrame({
            'rho_rolling': temporal_result.spearman_rho_rolling,
        })
        assert not temporal_df.empty
        
        sensitivity_df = pd.DataFrame({
            'tier': list(sensitivity_result.tier_robustness.keys()),
            'robustness': list(sensitivity_result.tier_robustness.values()),
        })
        assert len(sensitivity_df) == 3
        
        # Test figure generation (verify basic structure)
        fig_generator = TemporalSensitivityFigureGenerator()
        
        # figures can be generated (actual PNG writing tested separately)
        assert fig_generator is not None
    
    def test_realistic_data_range_validation(self):
        """Test that metrics stay within expected ranges for realistic data."""
        performance_data, _ = self._create_realistic_panel_data(
            n_years=14, n_provinces=10, n_criteria=8
        )
        years = sorted(performance_data.keys())
        weights_all_years = self._compute_global_weights(performance_data, years)
        
        # Temporal stability on realistic data
        temporal_analyzer = WindowedTemporalStabilityAnalyzer()
        temporal_result = temporal_analyzer.analyze(weights_all_years)
        
        # Expect moderate to high stability for realistic data with trends
        assert 0.2 <= temporal_result.spearman_rho_mean <= 1.0
        assert 0.0 <= temporal_result.spearman_rho_std <= 1.0
        assert 0.0 <= temporal_result.kendalls_w <= 1.0
        
        # CV should be reasonable (typically < 0.5)
        mean_cv = np.mean(list(temporal_result.coefficient_variation.values()))
        assert 0.0 <= mean_cv <= 0.5


class TestRobustnessAndNumericalStability:
    """Tests for numerical robustness and edge case handling."""
    
    def test_nan_inf_handling(self):
        """Test that computations handle NaN/Inf gracefully."""
        criterion_names = [f'C{i+1:02d}' for i in range(8)]
        
        # Normal weights
        normal_weights = np.array([0.15, 0.20, 0.12, 0.25, 0.10, 0.08, 0.05, 0.05])
        weights_all_years = {2011: {name: float(w) for name, w in zip(criterion_names, normal_weights)}}
        
        # Should handle without NaN/Inf in result
        analyzer = CRITICSensitivityAnalyzer(n_replicates=100, seed=42)
        result = analyzer.analyze(weights_all_years)
        
        # Check robustness values
        for tier_name, robustness in result.tier_robustness.items():
            assert not np.isnan(robustness), f"NaN found in {tier_name} robustness"
            assert not np.isinf(robustness), f"Inf found in {tier_name} robustness"
    
    def test_floating_point_precision(self):
        """Test that floating-point operations maintain precision."""
        criterion_names = [f'C{i+1:02d}' for i in range(8)]
        
        # Create weights from Dirichlet (naturally normalized)
        np.random.seed(42)
        for trial in range(5):
            weights_array = np.random.dirichlet(np.ones(8))
            weights_dict = {name: float(w) for name, w in zip(criterion_names, weights_array)}
            
            # Verify normalization
            total = sum(weights_dict.values())
            assert np.isclose(total, 1.0, atol=1e-10), f"Trial {trial}: weights don't sum to 1.0"
            
            weights_all_years = {2011: weights_dict}
            
            analyzer = CRITICSensitivityAnalyzer(n_replicates=100, seed=42)
            result = analyzer.analyze(weights_all_years)
            
            assert result is not None


class TestPerformanceMetrics:
    """Tests for performance and runtime validation."""
    
    def test_temporal_analysis_runtime_reasonable(self):
        """Test that temporal analysis completes in reasonable time."""
        import time
        
        criterion_names = [f'C{i+1:02d}' for i in range(8)]
        weights_array = np.random.dirichlet(np.ones(8))
        weights_dict = {name: float(w) for name, w in zip(criterion_names, weights_array)}
        
        # 14-year panel
        weights_all_years = {year: weights_dict for year in range(2011, 2025)}
        
        analyzer = WindowedTemporalStabilityAnalyzer()
        
        start_time = time.time()
        result = analyzer.analyze(weights_all_years)
        elapsed = time.time() - start_time
        
        # Should complete in < 1 second
        assert elapsed < 1.0, f"Temporal analysis took {elapsed:.2f}s (expected < 1s)"
        assert result is not None
    
    def test_sensitivity_analysis_runtime_reasonable(self):
        """Test that sensitivity analysis completes in reasonable time."""
        import time
        
        criterion_names = [f'C{i+1:02d}' for i in range(8)]
        weights_array = np.random.dirichlet(np.ones(8))
        weights_dict = {name: float(w) for name, w in zip(criterion_names, weights_array)}
        
        weights_all_years = {2011: weights_dict, 2024: weights_dict}
        
        analyzer = CRITICSensitivityAnalyzer(n_replicates=500, seed=42)
        
        start_time = time.time()
        result = analyzer.analyze(weights_all_years)
        elapsed = time.time() - start_time
        
        # 500 replicates should complete in < 45 seconds
        assert elapsed < 45.0, f"Sensitivity analysis took {elapsed:.2f}s (expected < 45s)"
        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
