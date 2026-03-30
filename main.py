#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML-MCDM Analysis Framework — Application Entry Point.

This script initializes and executes the complete hierarchical ML-MCDM 
analytical pipeline. It configures the panel dimensions, coordinates 
data processing, and manages the seven-phase workflow execution.

Usage
-----
$ python main.py
"""

import sys
import time
from typing import Any


def main() -> None:
    """
    Initialize and execute the ML-MCDM hierarchical pipeline.

    Sets up the default configuration for the 63-province/14-year panel 
    and triggers the orchestrated Seven-Phase analysis, including 
    forecasting and sensitivity reporting.
    """

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
