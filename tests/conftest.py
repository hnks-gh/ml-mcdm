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
