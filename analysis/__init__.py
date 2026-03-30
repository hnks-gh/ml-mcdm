"""
Analysis and Validation Suite for ML-MCDM.

This package provides a comprehensive set of diagnostic and validation tools 
for assessing the reliability, stability, and sensitivity of the forecasting 
and ranking pipelines.

Package Structure
-----------------
1. **Sensitivity Analysis**: Evaluates the impact of data perturbations and 
   model omissions on the final outputs (OOS performance, weight stability).
2. **Validation Framework**: Implements rigorous health checks for belief 
   distributions, forecast residuals, and cross-level ranking consistency.
3. **Bootstrap Uncertainty**: Quantifies confidence intervals for forecasts 
   and belief aggregations using residual and Dirichlet resampling.
4. **Temporal Stability**: Tracks the evolution of model performance and 
   weight rankings across sliding time windows.

Key Metrics
-----------
- **Overall Robustness**: Combined score reflecting sensitivity to noise.
- **Cross-Level Consistency**: Spearman correlation between hierarchical levels.
- **Interval Coverage**: Empirical calibration of conformal predictions.
- **Belief Entropy**: Quantification of ignorance and ambiguity in ER results.
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
