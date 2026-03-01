# conftest.py — pytest configuration for the tests/ package
import sys
import os

# Project root is one level above this conftest.py (tests/../)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Ensure the project root is on sys.path so that top-level packages
# (ranking, weighting, forecasting, etc.) are importable without
# an editable install.
sys.path.insert(0, _PROJECT_ROOT)

# Tell pytest not to collect __init__.py files as test modules.
collect_ignore_glob = ["__init__.py"]
collect_ignore = [os.path.join(_PROJECT_ROOT, "__init__.py")]

# ---------------------------------------------------------------------------
# Shared MCDM fixtures  (P4-29)
#
# dm3x3  — 3-alternative × 3-criterion all-benefit decision matrix
#           used across test_mcdm_textbook.py and test_mcdm_traditional.py
# w_equal_3 — uniform weights (1/3 each) for C1, C2, C3
#
# Property guaranteed:
#   A1 > A2 > A3  for every symmetric MCDM method because A1 dominates A3
#   on every criterion (row A1 strictly exceeds row A3 in all columns).
# ---------------------------------------------------------------------------

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def dm3x3() -> pd.DataFrame:
    """3-alternative × 3-criterion decision matrix — all benefit criteria."""
    return pd.DataFrame(
        {
            "C1": [0.9, 0.6, 0.3],
            "C2": [0.8, 0.5, 0.2],
            "C3": [0.7, 0.5, 0.1],
        },
        index=["A1", "A2", "A3"],
    )


@pytest.fixture
def w_equal_3() -> dict:
    """Uniform weights: each of the three criteria contributes equally."""
    return {"C1": 1 / 3, "C2": 1 / 3, "C3": 1 / 3}
