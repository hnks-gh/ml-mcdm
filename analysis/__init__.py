# -*- coding: utf-8 -*-
"""
Analysis Module
===============

Focused analysis layer for ML Forecasting and Evidential Reasoning.

Components
----------
Sensitivity Analysis (ML + ER)
    - ML: feature importance stability, LOO model impact, prediction sensitivity,
      temporal fold stability, conformal interval width sensitivity
    - ER: criterion belief sensitivity (OAT), grade threshold sensitivity,
      aggregation weight sensitivity, utility interval analysis, cross-level
      consistency, belief entropy diagnostics

Validation (ML + ER)
    - ML: CV fold diagnostics, conformal interval calibration, OOS evaluation,
      ensemble diversity, residual normality / autocorrelation / homoscedasticity
    - ER: belief admissibility, aggregation quality, utility interval widths,
      cross-level Spearman consistency, grade distribution diagnostics

Bootstrap (ML + ER)
    - ML: residual bootstrap for prediction uncertainty + feature importance stability
    - ER: belief distribution bootstrap via simplex-projected Gaussian perturbations

Stability (ML + ER)
    - ML: temporal fold-to-fold consistency, entity rank volatility, model agreement
    - ER: belief cosine similarity, utility rank correlation, grade consistency
"""

# Sensitivity Analysis
from .sensitivity import (
    MLSensitivityResult,
    ERSensitivityResult,
    CombinedSensitivityResult,
    MLSensitivityAnalysis,
    ERSensitivityAnalysis,
    run_ml_sensitivity_analysis,
    run_er_sensitivity_analysis,
    run_sensitivity_analysis,
)

# Validation
from .validation import (
    ERValidationResult,
    ForecastValidationResult,
    ValidationResult,
    ERValidator,
    ForecastValidator,
    Validator,
    run_validation,
)

# Bootstrap — ML + ER uncertainty quantification
from .ml_er_bootstrap import (
    ForecastBootstrapResult,
    ERBootstrapResult,
    ForecastBootstrap,
    ERBootstrap,
    forecast_bootstrap,
    er_bootstrap,
)

# Stability — ML + ER stability analysis
from .stability import (
    ForecastStabilityResult,
    ERStabilityResult,
    ForecastStabilityAnalyzer,
    ERStabilityAnalyzer,
    analyze_forecast_stability,
    analyze_er_stability,
)

__all__ = [
    # Sensitivity Analysis
    'SensitivityAnalysis',
    'SensitivityResult',
    'run_sensitivity_analysis',

    # Validation
    'Validator',
    'ValidationResult',
    'run_validation',

    # Bootstrap
    'bayesian_bootstrap_weights',
    'BayesianBootstrap',

    # Stability
    'StabilityResult',
    'temporal_stability_verification',
]
