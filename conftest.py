# conftest.py — root-level pytest configuration
import sys
import os

# Ensure the project root is on sys.path so that
# ``import forecasting`` etc. work without editable install.
sys.path.insert(0, os.path.dirname(__file__))

# Tell pytest not to try collecting tests from __init__.py files,
# and also not to import the root __init__.py as a test package.
collect_ignore_glob = ["__init__.py"]
collect_ignore = [os.path.join(os.path.dirname(__file__), "__init__.py")]
