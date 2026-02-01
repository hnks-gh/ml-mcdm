# -*- coding: utf-8 -*-
"""
Analysis Module
===============

Validation and sensitivity analysis tools for MCDM rankings and ML models.

Components
----------
Sensitivity Analysis
    Monte Carlo sensitivity analysis for MCDM rankings
    Weight perturbation and critical weight analysis
    
Validation
    Cross-validation (K-Fold, Time Series)
    Bootstrap validation with confidence intervals
    MCDM ranking-specific validation

Example
-------
>>> from src.analysis import (
...     SensitivityAnalysis, run_sensitivity_analysis,
...     CrossValidator, BootstrapValidator, RankingValidator
... )
>>> 
>>> # Quick sensitivity analysis
>>> result = run_sensitivity_analysis(matrix, weights, ranking_func)
>>> print(f"Robustness: {result.overall_robustness:.3f}")
"""

from .sensitivity import (
    SensitivityAnalysis, 
    SensitivityResult,
    WeightPerturbation,
    run_sensitivity_analysis
)
from .validation import (
    CrossValidator, 
    ValidationResult, 
    BootstrapValidator,
    RankingValidator,
    bootstrap_validation, 
    r2_score, 
    mse_score, 
    mae_score
)

__all__ = [
    # Sensitivity Analysis
    'SensitivityAnalysis', 
    'SensitivityResult',
    'WeightPerturbation',
    'run_sensitivity_analysis',
    
    # Validation
    'CrossValidator', 
    'ValidationResult', 
    'BootstrapValidator',
    'RankingValidator',
    'bootstrap_validation', 
    
    # Metrics
    'r2_score', 
    'mse_score', 
    'mae_score',
]
