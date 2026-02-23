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
    # Execute
    # ------------------------------------------------------------------
    pipeline = MLMCDMPipeline(config)

    try:
        result = pipeline.run()

        # Show run summary through the console logger
        pipeline.console.show_run_summary(result)
        pipeline.console.show_completion()

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
