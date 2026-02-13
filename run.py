#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry point for ML-MCDM analysis pipeline.

Usage:
    python run.py                      # Run with data/ (default)
"""

import sys
from pathlib import Path

# Configuration
CONFIG = {
    'data_path': None,  # Uses yearly CSV files from data/ folder automatically
    'n_provinces': 63,  # Actual provinces in data
    'n_years': 14,
    'output_dir': 'outputs',
}


def main():
    """Run the ML-MCDM analysis pipeline."""
    
    # Parse command line argument if provided
    data_path = CONFIG['data_path']
    if len(sys.argv) > 1:
        data_path = sys.argv[1]

    # Import here to avoid slow startup for --help
    from src import MLMCDMPipeline, get_default_config
    
    # Configure  
    config = get_default_config()
    config.panel.n_provinces = CONFIG['n_provinces']
    # Use last N years from available data
    end_year = 2024
    start_year = end_year - CONFIG['n_years'] + 1
    config.panel.years = list(range(start_year, end_year + 1))
    # n_components will be read from actual data
    
    # Reduce computational burden for testing
    config.weighting.bootstrap_iterations = 9  # Very quick testing (default: 999)
    config.random_forest.n_estimators = 20  # Faster testing (default: 200)
    
    print(f"{'─'*70}")
    print(f"  CONFIGURATION")
    print(f"{'─'*70}")
    print(f"\n  Data source: {data_path if data_path else 'Synthetic generation'}")
    print(f"  Entities: {CONFIG['n_provinces']}")
    print(f"  Time periods: {CONFIG['n_years']}")
    print(f"  Criteria: Auto-detected from data")
    print(f"  Output: {CONFIG['output_dir']}/")
    print(f"  MCDM methods: 11 (6 traditional + 5 fuzzy)\n")
    
    print(f"\n{'─'*70}")
    print(f"  RUNNING ANALYSIS PIPELINE")
    print(f"{'─'*70}\n")
    
    # Run pipeline
    pipeline = MLMCDMPipeline(config)
    
    try:
        result = pipeline.run(data_path)
        
        # Print results
        print_results(result)
        
        # Results are already saved by the pipeline to organized subfolders:
        # - outputs/figures/    - High-resolution charts
        # - outputs/results/    - CSV numerical data
        # - outputs/reports/    - Comprehensive text reports
        
        print(f"\n{'─'*70}")
        print(f"  ANALYSIS COMPLETE")
        print(f"{'─'*70}")
        print("\n  ✅ All analyses completed successfully!")
        print(f"  📊 Results saved to '{CONFIG['output_dir']}/':")
        print(f"     • figures/  - High-resolution charts (300 DPI)")
        print(f"     • results/  - Complete numerical data (CSV)")
        print(f"     • reports/  - Comprehensive analysis report\n")
        
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_results(result):
    """Print analysis results summary."""
    import numpy as np
    
    print(f"\n{'─'*70}")
    print(f"  ANALYSIS RESULTS")
    print(f"{'─'*70}")
    
    # Overview
    print("\n  📊 DATA OVERVIEW")
    print(f"     Entities: {len(result.panel_data.entities)}")
    print(f"     Time periods: {len(result.panel_data.time_periods)}")
    print(f"     Components: {len(result.panel_data.components)}")
    
    # Top rankings
    print("\n  🏆 TOP 10 RANKINGS")
    ranking_df = result.get_final_ranking_df()
    print(f"     {'Rank':<6} {'Entity':<20} {'Score':<12} {'Kendall W'}")
    print(f"     {'-'*50}")
    for _, row in ranking_df.head(10).iterrows():
        print(f"     {int(row['final_rank']):<6} {row['province']:<20} "
              f"{row['final_score']:.4f}      {row['kendall_w']:.4f}")
    
    # MCDM Method Comparison - All 11 Methods
    print("\n  📈 MCDM METHOD COMPARISON (11 Methods)")
    print(f"     {'Method':<25} {'Top Score':>12} {'Metric Type'}")
    print(f"     {'-'*55}")
    
    # Traditional Methods (6)
    print("     \033[1mTraditional Methods:\033[0m")
    print(f"     {'  TOPSIS':<25} {result.topsis_scores.max():>12.4f} {'Higher=Better'}")
    print(f"     {'  Dynamic TOPSIS':<25} {result.dynamic_topsis_scores.max():>12.4f} {'Higher=Better'}")
    print(f"     {'  VIKOR':<25} {result.vikor_results['Q'].min():>12.4f} {'Lower=Better'}")
    
    # Get PROMETHEE, COPRAS, EDAS from pipeline result
    if hasattr(result, 'config'):
        # Access from saved results if available
        import pandas as pd
        try:
            mcdm_scores = pd.read_csv(f"{CONFIG['output_dir']}/results/mcdm_scores_detailed.csv")
            promethee_max = mcdm_scores['PROMETHEE_Phi_Net'].max()
            copras_max = mcdm_scores['COPRAS_Utility'].max()
            edas_max = mcdm_scores['EDAS_AS'].max()
            print(f"     {'  PROMETHEE':<25} {promethee_max:>12.4f} {'Higher=Better'}")
            print(f"     {'  COPRAS':<25} {copras_max:>12.2f} {'% Utility'}")
            print(f"     {'  EDAS':<25} {edas_max:>12.4f} {'Higher=Better'}")
        except:
            pass
    
    # Fuzzy Methods (5)
    print("\n     \033[1mFuzzy Methods:\033[0m")
    print(f"     {'  Fuzzy TOPSIS':<25} {result.fuzzy_topsis_scores.max():>12.4f} {'Higher=Better'}")
    
    if hasattr(result, 'config'):
        try:
            fuzzy_vikor_min = mcdm_scores['Fuzzy_VIKOR_Q'].min()
            fuzzy_prom_max = mcdm_scores['Fuzzy_PROMETHEE_Phi_Net'].max()
            fuzzy_copras_max = mcdm_scores['Fuzzy_COPRAS_Utility'].max()
            fuzzy_edas_max = mcdm_scores['Fuzzy_EDAS_AS'].max()
            print(f"     {'  Fuzzy VIKOR':<25} {fuzzy_vikor_min:>12.4f} {'Lower=Better'}")
            print(f"     {'  Fuzzy PROMETHEE':<25} {fuzzy_prom_max:>12.4f} {'Higher=Better'}")
            print(f"     {'  Fuzzy COPRAS':<25} {fuzzy_copras_max:>12.2f} {'% Utility'}")
            print(f"     {'  Fuzzy EDAS':<25} {fuzzy_edas_max:>12.4f} {'Higher=Better'}")
        except:
            pass
    
    # Key metrics
    print("\n  🔗 KEY METRICS")
    print(f"     Rank Agreement (Kendall's W): {result.aggregated_ranking.kendall_w:.4f}")
    print(f"     Stacking Meta-Model R²: {result.stacking_result.meta_model_r2:.4f}")
    
    if hasattr(result, 'convergence_result') and result.convergence_result:
        status = "CONVERGING ✓" if result.convergence_result.beta_converging else "DIVERGING ✗"
        print(f"     Convergence: {status} (β={result.convergence_result.beta_coefficient:.4f})")
    
    if result.sensitivity_result:
        print(f"     Robustness: {result.sensitivity_result.overall_robustness:.4f}")
    
    print(f"\n  ⏱️  Execution Time: {result.execution_time:.2f} seconds")


if __name__ == '__main__':
    main()
