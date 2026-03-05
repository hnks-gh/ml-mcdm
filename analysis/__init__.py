# -*- coding: utf-8 -*-
"""
Analysis Module
===============

Centralised analysis layer for the ML-MCDM pipeline.

Components
----------
Sensitivity Analysis
    Hierarchical sensitivity for traditional MCDM + ER + Forecasting pipeline.
    - Multi-level analysis (subcriteria → criteria → final)
    - Temporal stability across years
    - Forecast robustness testing

Validation
    Comprehensive validation for hierarchical MCDM + ER pipeline.
    - Cross-level consistency checking
    - Weight scheme robustness
    - Forecast quality metrics
    - End-to-end pipeline validation

Bootstrap
    Bayesian Bootstrap (Rubin, 1981) for weight uncertainty quantification.
    - Dirichlet resampling via continuous observation weights
    - Posterior mean / std / 95 % credible intervals
    - Early-stopping convergence detection

Stability
    Split-half temporal stability validation for weight vectors.
    - Cosine similarity and Spearman rank correlation
    - Configurable stability threshold
"""

# Sensitivity Analysis
from .sensitivity import (
    SensitivityAnalysis,
    SensitivityResult,
    run_sensitivity_analysis,
)

# Validation
from .validation import (
    Validator,
    ValidationResult,
    run_validation,
)

# Bootstrap — weight uncertainty quantification
from .bootstrap import (
    bayesian_bootstrap_weights,
    BayesianBootstrap,
)

# Stability — temporal stability testing
from .stability import (
    StabilityResult,
    TemporalStabilityValidator,
    temporal_stability_verification,
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
    'TemporalStabilityValidator',
    'temporal_stability_verification',
]
