"""
Compatibility shim for the renamed CatBoost forecasting module.

This module provides 1:1 aliasing for `CatBoostForecaster` and its 
helpers to maintain backward compatibility with earlier pipeline 
versions that imported from `forecasting.gradient_boosting`.

New code should import directly from `forecasting.catboost_forecaster`.
"""

from .catboost_forecaster import CatBoostForecaster, _chronological_es_split

__all__ = ["CatBoostForecaster", "_chronological_es_split"]
