#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML-MCDM Panel Data Analysis
==============================

Run this file to execute the complete analysis pipeline.

Usage:
    python main.py                              # Run with synthetic data
    python main.py data/panel_data.csv          # Run with custom data file
"""

import sys
from pathlib import Path

# Configuration - modify these as needed
CONFIG = {
    'data_path': None,           # Set to CSV path or None for synthetic data
    'n_provinces': 64,           # Number of entities
    'n_years': 5,                # Number of time periods  
    'n_components': 20,          # Number of criteria
    'output_dir': 'outputs',     # Output directory
}


def main():
    """Run the ML-MCDM analysis pipeline."""
    
    # Parse command line argument if provided
    data_path = CONFIG['data_path']
    if len(sys.argv) > 1:
        data_path = sys.argv[1]

    # Import here to avoid slow startup for --help
    from src import MLTOPSISPipeline, get_default_config
    
    # Configure
    config = get_default_config()
    config.panel.n_provinces = CONFIG['n_provinces']
    config.panel.years = list(range(2020, 2020 + CONFIG['n_years']))
    config.panel.n_components = CONFIG['n_components']
    
    print(f"{'─'*70}")
    print(f"  CONFIGURATION")
    print(f"{'─'*70}")
    print(f"\n  Data source: {data_path if data_path else 'Synthetic generation'}")
    print(f"  Entities: {CONFIG['n_provinces']}")
    print(f"  Time periods: {CONFIG['n_years']}")
    print(f"  Components: {CONFIG['n_components']}")
    print(f"  Output: {CONFIG['output_dir']}/")
    
    print(f"\n{'─'*70}")
    print(f"  RUNNING ANALYSIS PIPELINE")
    print(f"{'─'*70}\n")
    
    # Run pipeline
    pipeline = MLTOPSISPipeline(config)
    
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
    
    # Method comparison
    print("\n  📈 MCDM METHOD COMPARISON")
    print(f"     {'Method':<20} Top Score")
    print(f"     {'-'*35}")
    print(f"     {'TOPSIS':<20} {result.topsis_scores.max():.4f}")
    print(f"     {'Dynamic TOPSIS':<20} {result.dynamic_topsis_scores.max():.4f}")
    print(f"     {'Fuzzy TOPSIS':<20} {result.fuzzy_topsis_scores.max():.4f}")
    print(f"     {'VIKOR (1-Q)':<20} {1 - result.vikor_results['Q'].min():.4f}")
    
    # Key metrics
    print("\n  🔗 KEY METRICS")
    print(f"     Rank Agreement (Kendall's W): {result.aggregated_ranking.kendall_w:.4f}")
    print(f"     Stacking Meta-Model R²: {result.stacking_result.meta_model_r2:.4f}")
    
    if result.convergence_result:
        status = "CONVERGING ✓" if result.convergence_result.beta_converging else "DIVERGING ✗"
        print(f"     Convergence: {status} (β={result.convergence_result.beta_coefficient:.4f})")
    
    if result.sensitivity_result:
        print(f"     Robustness: {result.sensitivity_result.overall_robustness:.4f}")
    
    print(f"\n  ⏱️  Execution Time: {result.execution_time:.2f} seconds")


if __name__ == '__main__':
    main()
