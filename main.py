#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML-MCDM Analysis — Main Entry Point
=====================================

Usage
-----
    python main.py

Pipeline Phases
---------------
1. Data Loading        – yearly CSVs from data/
2. Weight Calculation  – GTWC (Entropy + CRITIC + MEREC + SD)
3. Hierarchical Ranking – 12 MCDM + two-stage ER
4. ML Forecasting       – 6-model ensemble + Super Learner + Conformal
5. Sensitivity Analysis  – Monte Carlo weight perturbation
6. Visualisation         – high-resolution PNGs
7. Result Export         – CSV / JSON / text report
"""

import sys
import time
from typing import Any


def main() -> None:
    """Configure and execute the ML-MCDM pipeline."""

    # ------------------------------------------------------------------
    # Lazy imports (avoids heavy loading on --help)
    # ------------------------------------------------------------------
    from ml_mcdm.pipeline import MLMCDMPipeline
    from ml_mcdm.config import get_default_config

    config = get_default_config()

    # Panel dimensions
    config.panel.n_provinces = 63
    config.panel.years = list(range(2011, 2025))

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  ML-MCDM: IFS + Evidential Reasoning Hierarchical Ranking")
    print(f"{'='*70}")
    print(f"  Provinces         : {config.panel.n_provinces}")
    print(f"  Years             : {config.panel.years[0]}-{config.panel.years[-1]} "
          f"({config.panel.n_years} years)")
    print(f"  Subcriteria       : {config.panel.n_subcriteria}")
    print(f"  Criteria          : {config.panel.n_criteria}")
    print(f"  MCDM methods      : 12 (6 traditional + 6 IFS)")
    print(f"  Bootstrap iters   : {config.weighting.bootstrap_iterations}")
    print(f"  Sensitivity sims  : {config.validation.n_simulations}")
    print(f"  Output            : outputs/")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    pipeline = MLMCDMPipeline(config)

    try:
        result = pipeline.run()
        print_results(result)

        print(f"\n{'='*70}")
        print("  ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print("  All outputs saved to outputs/:")
        print("    figures/  — high-resolution charts (300 DPI)")
        print("    results/  — numerical data (CSV / JSON)")
        print("    reports/  — comprehensive text report")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_results(result: Any) -> None:
    """Print concise results summary to console."""
    import numpy as np

    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")

    # Data overview
    pd_ = result.panel_data
    print(f"\n  DATA")
    print(f"    Provinces   : {len(pd_.provinces)}")
    print(f"    Years       : {min(pd_.years)}-{max(pd_.years)} ({len(pd_.years)} yr)")
    print(f"    Subcriteria : {pd_.n_subcriteria}")
    print(f"    Criteria    : {pd_.n_criteria}")

    # Top rankings
    print(f"\n  TOP 10 RANKINGS (Evidential Reasoning)")
    ranking_df = result.get_final_ranking_df()
    print(f"    {'Rank':<6} {'Province':<25} {'ER Score':>10}")
    print(f"    {'-'*42}")
    for _, row in ranking_df.head(10).iterrows():
        print(f"    {int(row['ER_Rank']):<6} {row['Province']:<25} "
              f"{row['ER_Score']:>10.4f}")

    # Kendall's W
    print(f"\n  CONCORDANCE")
    print(f"    Kendall's W : {result.ranking_result.kendall_w:.4f}")
    w = result.ranking_result.kendall_w
    if w > 0.7:
        interp = "Strong agreement"
    elif w > 0.5:
        interp = "Moderate agreement"
    else:
        interp = "Weak agreement"
    print(f"    Interpretation: {interp}")

    # Sensitivity
    if result.sensitivity_result:
        print(f"\n  SENSITIVITY")
        print(f"    Robustness  : {result.sensitivity_result.overall_robustness:.4f}")

    # ML Forecasting (if enabled)
    if result.forecast_result:
        print(f"\n  ML FORECASTING (Ensemble Performance)")
        cv_scores = result.forecast_result.cross_validation_scores
        all_cv = [s for scores in cv_scores.values() for s in scores]
        print(f"    Mean CV R²      : {np.mean(all_cv):.4f}")
        print(f"    Std  CV R²      : {np.std(all_cv):.4f}")
        print(f"    Prediction Year : {result.forecast_result.training_info.get('target_year', 'Latest+1')}")

    # Execution time
    print(f"\n  RUNTIME : {result.execution_time:.2f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
