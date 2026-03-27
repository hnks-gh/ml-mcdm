# -*- coding: utf-8 -*-
"""
ML Forecasting Module
=====================

State-of-the-art ensemble forecasting system optimized for small-to-medium
panel data (N < 1000), emphasizing model diversity over quantity.

Architecture:
    Tier 1 - Base Models (5 diverse models):
        - catboost_forecaster: CatBoost (oblivious trees, MultiRMSE)
        - bayesian: Bayesian Ridge regression (uncertainty quantification)
        - quantile_forest: Distributional forecasting via quantile RF

    Tier 2 - Meta-Ensemble:
        - super_learner: Stacked generalization with automatic weighting
          (PanelWalkForwardCV; per-criterion RMSE tracking)

    Tier 3 - Calibration:
        - conformal: Distribution-free prediction intervals (95% coverage)
        - evaluation: Comprehensive evaluation and ablation

    Orchestration:
        - features: Temporal feature engineering
        - unified: Full pipeline orchestration (Phase 4: Optuna HP search,
                   Phase 5: reversible target transformation)

Design Philosophy:
    - Model diversity over quantity (5 diverse > many correlated)
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
from .features import TemporalFeatureEngineer, SAWNormalizer

# Tree-based ensemble methods
from .catboost_forecaster import CatBoostForecaster

# Bayesian linear method
from .bayesian import BayesianForecaster

# Kernel methods (T-03a, T-03b)
from .kernel_ridge import KernelRidgeForecaster
from .svr import SVRForecaster

# Linear methods with explicit feature selection
from .elasticnet_forecaster import ElasticNetForecaster

from .quantile_forest import QuantileRandomForestForecaster

# Meta-ensemble methods
from .super_learner import SuperLearner

# Calibration and evaluation
from .conformal import ConformalPredictor
from .evaluation import ForecastEvaluator, AblationStudy
from .evaluation_diagnostics import LeaveOneEntityOutCV

# Phase 3 — SOTA modules (E-05, E-06, E-08, E-10)
from .panel_mice import PanelSequentialMICE
from .augmentation import ConditionalPanelAugmenter, SyntheticAwareCV
from .shift_detection import PanelCovariateShiftDetector
from .incremental_update import IncrementalEnsembleUpdater

# Unified orchestrator
from .unified import (
    UnifiedForecaster,
    UnifiedForecastResult,
)

# Base classes
from .base import BaseForecaster

__all__ = [
    # Feature engineering
    'TemporalFeatureEngineer',
    'SAWNormalizer',
    # Tree ensemble
    'CatBoostForecaster',
    # Bayesian linear
    'BayesianForecaster',
    # Kernel methods (T-03a, T-03b)
    'KernelRidgeForecaster',
    'SVRForecaster',
    # Linear methods with explicit feature selection
    'ElasticNetForecaster',
    # Advanced models (SOTA)
    'QuantileRandomForestForecaster',
    # Meta-ensemble
    'SuperLearner',
    # Calibration and evaluation
    'ConformalPredictor',
    'ForecastEvaluator',
    'AblationStudy',
    'LeaveOneEntityOutCV',
    # Phase 3 — SOTA modules
    'PanelSequentialMICE',
    'ConditionalPanelAugmenter',
    'SyntheticAwareCV',
    'PanelCovariateShiftDetector',
    'IncrementalEnsembleUpdater',
    # Unified
    'UnifiedForecaster',
    'UnifiedForecastResult',
    # Base
    'BaseForecaster',
]
