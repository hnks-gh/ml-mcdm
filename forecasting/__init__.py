# -*- coding: utf-8 -*-
"""
ML Forecasting Module
=====================

State-of-the-art ensemble forecasting system optimized for small-to-medium
panel data (N < 1000), emphasizing model diversity over quantity.

Architecture:
    Tier 1 - Base Models (6 diverse models):
        - gradient_boosting: Gradient Boosting (robust, sample-efficient)
        - bayesian: Bayesian Ridge regression (uncertainty quantification)
        - panel_var: Panel Vector Autoregression with fixed effects
        - quantile_forest: Distributional forecasting via quantile RF
        - hierarchical_bayes: Partial pooling via empirical Bayes
        - neural_additive: Neural Additive Models (interpretable)

    Tier 2 - Meta-Ensemble:
        - super_learner: Stacked generalization with automatic weighting

    Tier 3 - Calibration:
        - conformal: Distribution-free prediction intervals (95% coverage)
        - evaluation: Comprehensive evaluation and ablation

    Orchestration:
        - features: Temporal feature engineering
        - unified: Full pipeline orchestration

Design Philosophy:
    - Model diversity over quantity (6 diverse > 11 correlated)
    - Statistical appropriateness for N < 1000
    - Automatic optimal weighting (Super Learner)
    - Guaranteed uncertainty coverage (Conformal Prediction)

Example Usage:
    >>> from forecasting import UnifiedForecaster
    >>>
    >>> # State-of-the-art configuration (Super Learner + Conformal)
    >>> forecaster = UnifiedForecaster()
    >>> result = forecaster.fit_predict(panel_data, target_year=2025)
    >>>
    >>> # Custom conformal settings
    >>> forecaster = UnifiedForecaster(conformal_alpha=0.10, cv_folds=5)
    >>> result = forecaster.fit_predict(panel_data, target_year=2025)
"""

# Feature engineering
from .features import TemporalFeatureEngineer

# Tree-based ensemble methods
from .gradient_boosting import GradientBoostingForecaster

# Bayesian linear method
from .bayesian import BayesianForecaster

# Advanced models (state-of-the-art)
from .panel_var import PanelVARForecaster
from .quantile_forest import QuantileRandomForestForecaster
from .hierarchical_bayes import HierarchicalBayesForecaster
from .neural_additive import NeuralAdditiveForecaster

# Meta-ensemble methods
from .super_learner import SuperLearner

# Calibration and evaluation
from .conformal import ConformalPredictor
from .evaluation import ForecastEvaluator, AblationStudy

# Unified orchestrator
from .unified import (
    UnifiedForecaster,
    UnifiedForecastResult,
)

# Base classes and results
from .base import (
    BaseForecaster,
    ForecastResult,
)

__all__ = [
    # Feature engineering
    'TemporalFeatureEngineer',
    # Tree ensemble
    'GradientBoostingForecaster',
    # Bayesian linear
    'BayesianForecaster',
    # Advanced models (SOTA)
    'PanelVARForecaster',
    'QuantileRandomForestForecaster',
    'HierarchicalBayesForecaster',
    'NeuralAdditiveForecaster',
    # Meta-ensemble
    'SuperLearner',
    # Calibration and evaluation
    'ConformalPredictor',
    'ForecastEvaluator',
    'AblationStudy',
    # Unified
    'UnifiedForecaster',
    'UnifiedForecastResult',
    # Base
    'BaseForecaster',
    'ForecastResult',
]
