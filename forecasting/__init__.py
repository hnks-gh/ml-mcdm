"""
Ensemble Forecasting System for Multi-Target Panel Data.

This package provides a state-of-the-art ensemble forecasting system optimized 
for small-to-medium panel data (N < 1000). It emphasizes model diversity, 
statistical rigor, and guaranteed uncertainty coverage through a multi-tier 
stacked generalization architecture.

Package Structure
-----------------
1. **Base Models (Tier 1)**: A diverse set of learners including gradient 
   boosting (CatBoost), Bayesian linear models, kernel-based regression (SVR, 
   Kernel Ridge), and distributional forests (Quantile RF).
2. **Meta-Ensemble (Tier 2)**: A `SuperLearner` that implements stacked 
   generalization with automatic per-criterion weight optimization using 
   temporal walk-forward cross-validation.
3. **Uncertainty & Calibration (Tier 3)**: A `ConformalPredictor` that 
   provides distribution-free prediction intervals with 95% coverage 
   guarantees, and comprehensive evaluation tools for ablation and diagnostics.

Orchestration
-------------
The `UnifiedForecaster` serves as the central entry point, managing the full 
pipeline from temporal feature engineering to final calibrated forecasts 
and diagnostic artifact generation.

Example Usage
-------------
>>> from forecasting import UnifiedForecaster
>>> forecaster = UnifiedForecaster()
>>> result = forecaster.fit_predict(df, target_year=2025)
>>> print(result.predictions)
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
