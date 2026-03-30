from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

# ============================================================================
# CORE: Production MICE Imputation
# ============================================================================

from .iterative import MICEImputer
from .validation import MICEValidation

@dataclass
class ImputationConfig:
    """
    Configuration for missing data handling in the ML forecasting pipeline.

    Defines the parameters for Multivariate Imputation by Chained Equations 
    (MICE) and diagnostic tests.

    Attributes
    ----------
    use_mice_imputation : bool
        Enable MICE imputation in preprocessing. Default True.
    n_imputations : int
        Number of stochastic MICE imputations for uncertainty quantification.
    mice_max_iter : int
        Maximum iterations for the iterative imputer to reach convergence.
    mice_n_nearest_features : int
        Number of most-correlated features used to estimate each missing value.
    mice_estimator : str
        The regression model used within MICE iterations.
    mice_add_indicator : bool
        Whether to append binary indicator columns for imputed features.
    random_state : int
        Seed for reproducibility.
    """
    
    # ======================================================================
    # CORE PARAMETERS (MICE-Only Strategy)
    # ======================================================================
    
    # Imputation method for decision matrix (MCDM phase, not ML forecasting)
    decision_matrix_method: str = "temporal_linear"
    ml_feature_method: str = "mice_missforest"
    
    # Enable/disable MICE imputation in preprocessing
    use_mice_imputation: bool = True
    """Apply MICE imputation to missing features. Default True."""
    
    # Multiple imputation for Rubin's Rules uncertainty quantification
    n_imputations: int = 5
    """Number of stochastic imputations (M) for multiple imputation.
    
    - M=1: single point estimate (faster, no uncertainty)
    - M=5: standard practice (recommended)
    - M≥10: for high missingness (>30%)
    
    When M>1, forecaster generates M imputed datasets, trains M models,
    pools predictions via Rubin's Rules for total variance estimation.
    """
    
    # MICE parameters (ExtraTreesRegressor-based IterativeImputer)
    mice_max_iter: int = 40
    """IterativeImputer convergence iterations. Increased from 20→40 for better convergence. Higher = more stable but slower."""
    
    mice_n_nearest_features: int = 30
    """Number of most-correlated features to estimate missing values from."""
    
    mice_estimator: Literal["random_forest", "extra_trees", "bayesian_ridge"] = "extra_trees"
    """Regression estimator for MICE imputation:
    
    - "extra_trees" (default): Fast, adaptive, good for nonlinear relationships
    - "random_forest": Stable, slower, for tabular data
    - "bayesian_ridge": Probabilistic, for uncertainty quantification
    """
    
    mice_add_indicator: bool = True
    """Append binary _was_missing flags for each imputed feature."""
    
    # Advanced options
    add_missingness_indicators: bool = True
    """Global flag for missingness indicators across all imputation stages."""
    
    enable_mcar_test: bool = True
    """Test missing-at-random (MCAR) assumption (diagnostic only)."""
    
    random_state: int = 42
    """Random seed for reproducibility."""
    
    # ======================================================================
    # DEPRECATED PARAMETERS (Backward Compatibility Only)
    # ======================================================================
    # These parameters are accepted but ignored. They are kept to avoid
    # breaking existing code that passes them to ImputationConfig.
    # See docstring above for migration guide.
    
    use_advanced_feature_imputation: bool = True
    """DEPRECATED: Per-block tier imputation. Now ignored (use MICE always)."""
    
    block_imputation_tiers: Dict[int, str] = field(default_factory=lambda: {
        1:   "training_mean",
        2:   "cross_sectional_median",
        3:   "temporal_median",
        4:   "temporal_median",
        5:   "cross_sectional_median",
        6:   "cross_sectional_median",
        7:   "temporal_median",
        8:   "temporal_median",
        9:   "cross_sectional_median",
        10:  "temporal_median",
        11:  "cross_sectional_median",
        12:  None,
    })
    """DEPRECATED: Per-block tier assignments. Now ignored (MICE used for all)."""
    
    temporal_imputation_window: int = 5
    """DEPRECATED: Rolling median window. Now ignored (MICE handles temporal)."""
    
    temporal_imputation_min_periods: int = 2
    """DEPRECATED: Minimum periods for rolling medians. Now ignored."""
    
    # ======================================================================
    # LEGACY PARAMETERS (Unused, kept for completeness)
    # ======================================================================
    
    # Parameters for CP-ALS tensor imputation (M-09, not currently used)
    cp_rank: int = 8
    cp_max_iter: int = 500
    cp_tol: float = 1e-5
    cp_lambda_reg: float = 0.01
    
    # Parameters for Gaussian Process imputation (M-06, not currently used)
    gp_length_scale_temporal: float = 3.0
    gp_length_scale_spatial: float = 2.0

@dataclass
class ImputationAudit:
    """Audit record for metadata tracking (M-14)."""
    n_cells_original: int
    n_imputed: int
    missingness_rate_initial: float
    method_applied: str
    blocks_imputed: Dict[int, str] = field(default_factory=dict)  # block_id -> method used
    mcar_p_value: Optional[float] = None
    mechanism_assessment: str = "unknown"
    imputed_locations: List[Tuple[int, int]] = field(default_factory=list)
    imputation_errors: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# PUBLIC API
# ============================================================================
# Only production-ready MICE imputation is exposed.
# Legacy methods (GAIN, multi-phase temporal, etc.) removed 2026-03-27.

__all__ = [
    'MICEImputer',           # Production MICE engine
    'ImputationConfig',      # Configuration dataclass
    'ImputationAudit',       # Audit trail
    'MICEValidation',        # Validation suite
]

