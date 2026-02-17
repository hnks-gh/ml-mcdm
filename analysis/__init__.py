# -*- coding: utf-8 -*-
"""
Analysis Module
===============

Production-grade validation and sensitivity analysis for MCDM rankings and ML models.

Components
----------
Sensitivity Analysis (PRODUCTION-READY)
    State-of-the-art hierarchical sensitivity for IFS+ER+Forecasting pipeline
    - Multi-level analysis (subcriteria → criteria → final)
    - IFS membership/non-membership uncertainty
    - Temporal stability across years
    - Forecast robustness testing
    
Validation (PRODUCTION-READY)
    Comprehensive validation for hierarchical IFS+ER pipeline
    - Cross-level consistency checking
    - IFS parameter validation
    - Weight scheme robustness
    - Forecast quality metrics
    - End-to-end pipeline validation

Example
-------
>>> from analysis import (
...     SensitivityAnalysis, run_sensitivity_analysis,
...     Validator, run_validation
... )
>>> 
>>> # Production sensitivity analysis
>>> sens_result = run_sensitivity_analysis(
...     panel_data, ranking_pipeline, weights, ranking_result, forecast_result
... )
>>> print(f"Robustness: {sens_result.overall_robustness:.3f}")
>>> 
>>> # Production validation
>>> val_result = run_validation(
...     panel_data, weights, ranking_result, forecast_result
... )
>>> print(f"Validity: {val_result.overall_validity:.3f}")
"""

# Sensitivity Analysis (production-ready)
from .sensitivity import (
    SensitivityAnalysis,
    SensitivityResult,
    run_sensitivity_analysis
)

# Validation (production-ready)
from .validation import (
    Validator,
    ValidationResult,
    run_validation
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
]
