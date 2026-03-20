# -*- coding: utf-8 -*-
"""
Compatibility shim for renamed CatBoost forecasting module.

Use forecasting.catboost_forecaster for all new imports.
"""

from .catboost_forecaster import CatBoostForecaster, _chronological_es_split

__all__ = ["CatBoostForecaster", "_chronological_es_split"]
