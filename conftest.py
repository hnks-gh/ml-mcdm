# conftest.py — root-level pytest configuration
import sys
import os

# Ensure the project root is on sys.path so that
# ``import forecasting`` etc. work without editable install.
sys.path.insert(0, os.path.dirname(__file__))

# Tell pytest not to try importing __init__.py from the root package
collect_ignore_glob = ["__init__.py"]
