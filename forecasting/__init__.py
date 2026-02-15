# -*- coding: utf-8 -*-
"""
ML Forecasting Module
=====================

State-of-the-art ensemble forecasting system for multi-criteria
decision making with temporal/panel data.

Architecture:
    Tier 1 - Base Models:
        - tree_ensemble: Gradient Boosting, Random Forest, Extra Trees
        - linear: Bayesian Ridge, Huber, Ridge regression
        - neural: MLP, Attention-based networks
        - panel_var: Panel Vector Autoregression with fixed effects
        - quantile_forest: Distributional forecasting via quantile RF
        - hierarchical_bayes: Partial pooling via empirical Bayes
        - neural_additive: Neural Additive Models (interpretable)

    Tier 2 - Meta-Ensemble:
        - super_learner: Stacked generalization with meta-learning
        - auto_ensemble: Bayesian optimization model selection

    Tier 3 - Calibration:
        - conformal: Distribution-free prediction intervals
        - evaluation: Comprehensive evaluation and ablation

    Orchestration:
        - features: Temporal feature engineering
        - unified: Full pipeline orchestration

Example Usage:
    >>> from forecasting import UnifiedForecaster, ForecastMode
    >>>
    >>> # Standard mode (backward compatible)
    >>> forecaster = UnifiedForecaster(mode=ForecastMode.BALANCED)
    >>> result = forecaster.fit_predict(panel_data, target_year=2025)
    >>>
    >>> # Advanced mode with Super Learner + all SOTA models
    >>> forecaster = UnifiedForecaster(mode=ForecastMode.ADVANCED)
    >>> result = forecaster.fit_predict(panel_data, target_year=2025)
    >>>
    >>> # Auto mode with Bayesian optimization
    >>> forecaster = UnifiedForecaster(mode=ForecastMode.AUTO)
    >>> result = forecaster.fit_predict(panel_data, target_year=2025)
"""

# Feature engineering
from .features import TemporalFeatureEngineer

# Tree-based ensemble methods
from .tree_ensemble import (
    GradientBoostingForecaster,
    RandomForestForecaster,
    ExtraTreesForecaster,
)

# Linear methods
from .linear import (
    BayesianForecaster,
    HuberForecaster,
    RidgeForecaster,
)

# Neural network methods
from .neural import (
    NeuralForecaster,
    AttentionForecaster,
    DenseLayer,
    AttentionLayer,
)

# Advanced models (state-of-the-art)
from .panel_var import PanelVARForecaster
from .quantile_forest import QuantileRandomForestForecaster
from .hierarchical_bayes import HierarchicalBayesForecaster
from .neural_additive import NeuralAdditiveForecaster

# Meta-ensemble methods
from .super_learner import SuperLearner
from .auto_ensemble import AutoEnsembleSelector

# Calibration and evaluation
from .conformal import ConformalPredictor
from .evaluation import ForecastEvaluator, AblationStudy

# Unified orchestrator
from .unified import (
    UnifiedForecaster,
    UnifiedForecastResult,
    ForecastMode,
)

# Base classes and results
from .base import (
    BaseForecaster,
    ForecastResult,
)

# Time-series specific Random Forest (for panel data)
from .random_forest_ts import (
    RandomForestTS,
    RandomForestTSResult,
    TimeSeriesSplit,
    calculate_shap_importance,
)

__all__ = [
    # Feature engineering
    'TemporalFeatureEngineer',
    # Tree ensemble
    'GradientBoostingForecaster',
    'RandomForestForecaster',
    'ExtraTreesForecaster',
    # Linear
    'BayesianForecaster',
    'HuberForecaster',
    'RidgeForecaster',
    # Neural
    'NeuralForecaster',
    'AttentionForecaster',
    'DenseLayer',
    'AttentionLayer',
    # Advanced models (SOTA)
    'PanelVARForecaster',
    'QuantileRandomForestForecaster',
    'HierarchicalBayesForecaster',
    'NeuralAdditiveForecaster',
    # Meta-ensemble
    'SuperLearner',
    'AutoEnsembleSelector',
    # Calibration and evaluation
    'ConformalPredictor',
    'ForecastEvaluator',
    'AblationStudy',
    # Unified
    'UnifiedForecaster',
    'UnifiedForecastResult',
    'ForecastMode',
    # Base
    'BaseForecaster',
    'ForecastResult',
    # Time-series RF (panel data)
    'RandomForestTS',
    'RandomForestTSResult',
    'TimeSeriesSplit',
    'calculate_shap_importance',
]
