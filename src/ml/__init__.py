# -*- coding: utf-8 -*-
"""
Machine Learning Module
=======================

ML methods for panel data analysis and time series forecasting.

Submodules
----------
forecasting
    Comprehensive forecasting system with tree-based, linear, and neural models
    Feature engineering for temporal data
    Unified forecasting interface

Usage
-----
>>> from src.ml.forecasting import UnifiedForecaster, ForecastMode
>>> from src.ml.forecasting import GradientBoostingForecaster, NeuralForecaster
"""

# Import from forecasting submodule
from .forecasting import (
    # Base classes
    BaseForecaster, ForecastResult,
    
    # Feature engineering
    TemporalFeatureEngineer,
    
    # Tree ensemble forecasters
    GradientBoostingForecaster,
    RandomForestForecaster,
    ExtraTreesForecaster,
    
    # Linear forecasters
    BayesianForecaster,
    HuberForecaster,
    RidgeForecaster,
    
    # Neural forecasters
    NeuralForecaster,
    AttentionForecaster,
    DenseLayer,
    AttentionLayer,
    
    # Unified forecaster
    UnifiedForecaster,
    UnifiedForecastResult,
    ForecastMode,
    
    # Time-series RF for panel data (used by pipeline)
    RandomForestTS,
    RandomForestTSResult,
    TimeSeriesSplit,
    calculate_shap_importance,
)

# Backward compatibility aliases
BayesianRidgeForecaster = BayesianForecaster
AttentionTemporalForecaster = AttentionForecaster

__all__ = [
    # Feature engineering
    'TemporalFeatureEngineer',
    
    # Base classes  
    'BaseForecaster',
    'ForecastResult',
    
    # Tree ensemble forecasters
    'GradientBoostingForecaster',
    'RandomForestForecaster', 
    'ExtraTreesForecaster',
    
    # Linear forecasters
    'BayesianForecaster',
    'BayesianRidgeForecaster',  # Alias
    'HuberForecaster',
    'RidgeForecaster',
    
    # Neural forecasters
    'NeuralForecaster',
    'AttentionForecaster',
    'AttentionTemporalForecaster',  # Alias
    'DenseLayer',
    'AttentionLayer',
    
    # Unified system
    'UnifiedForecaster',
    'UnifiedForecastResult',
    'ForecastMode',
    
    # Time-series RF (panel data)
    'RandomForestTS',
    'RandomForestTSResult',
    'TimeSeriesSplit',
    'calculate_shap_importance',
]


def get_forecaster(mode: str = 'balanced'):
    """
    Get a configured forecaster instance.
    
    Parameters
    ----------
    mode : str, default='balanced'
        Forecasting mode: 'fast', 'balanced', 'accurate', 'neural', or 'ensemble'
    
    Returns
    -------
    UnifiedForecaster
        Configured forecaster instance
        
    Example
    -------
    >>> forecaster = get_forecaster('accurate')
    >>> result = forecaster.fit_predict(X, y, entity_ids, time_ids)
    """
    return UnifiedForecaster(mode=ForecastMode(mode.lower()))
