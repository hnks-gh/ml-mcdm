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
    Configuration for missing data handling in ML forecasting (MICE-only strategy).
    
    IMPUTATION STRATEGY (Simplified Phase B+):
    ─────────────────────────────────────────
    The forecasting pipeline uses a single unified MICE imputation strategy:
    
    1. **Feature Engineering**: TemporalFeatureEngineer outputs features with NaN
       when historical data is unavailable (e.g., lag-3 for year 1, rolling stats
       for short histories).
    
    2. **MICE Imputation (Preprocessing)**: The PanelFeatureReducer applies 
       IterativeImputer (ExtraTreesRegressor) to missing features BEFORE 
       dimensionality reduction. This captures multivariate feature correlations
       and produces realistic imputed values reflecting panel structure.
    
    3. **Multiple Imputation (Optional)**: When n_imputations > 1, the forecaster
       generates M stochastic imputations, trains M independent models, and pools
       predictions via Rubin's Rules for uncertainty quantification.
    
    RATIONALE: Tiered block-level imputation (PHASE A) was over-engineered. 
    MICE with ExtraTreesRegressor already captures:
    - ✓ Multivariate correlations (respects feature relationships)
    - ✓ Nonlinear patterns (ExtraTreesRegressor is adaptive)
    - ✓ Panel structure (when applied chronologically)
    - ✓ Uncertainty (via posterior sampling in multiple imputation)
    
    Key Parameters
    ──────────────
    use_mice_imputation : bool
        Enable MICE imputation in PanelFeatureReducer. Default True.
    n_imputations : int
        Number of stochastic MICE imputations (M). Default 5.
        M=1: single point estimate (faster)
        M=5: standard for uncertainty quantification
        M≥10: for high missingness rates (>30%)
    mice_max_iter : int
        IterativeImputer convergence iterations. Default 20.
    mice_n_nearest_features : int
        Number of nearest features for correlation weighting. Default 30.
    mice_estimator : str
        Regression model for MICE: "extra_trees" (default, fast, adaptive)
                                   "random_forest" (stable, slower)
    mice_add_indicator : bool
        Append binary _was_missing columns for each imputed feature. Default True.
        Allows models to learn imputation uncertainty representation.
    add_missingness_indicators : bool
        Global flag for missingness indicators. Default True.
    random_state : int
        Random seed for reproducibility. Default 42.
    
    DEPRECATED PARAMETERS (accepted for backward compatibility, but ignored):
    ───────────────────────────────────────────────────────────────────────
    The following parameters were used for per-block tier selection (PHASE A)
    and are NO LONGER USED. They are accepted to avoid breaking existing code,
    but have no effect:
    
    - use_advanced_feature_imputation (bool): DEPRECATED
      Previously controlled whether Tiers 2-4 caching was applied per block.
      Now ignored; all imputation uses MICE.
    
    - block_imputation_tiers (Dict[int, str]): DEPRECATED
      Previously mapped block numbers (1-12) to tier strategies.
      Example: {1: "training_mean", 3: "temporal_median", ...}
      Now ignored; MICE applied uniformly to all missing features.
    
    - temporal_imputation_window (int): DEPRECATED
      Previously used for per-entity annual medians (Tier 2).
      Now ignored; MICE handles temporal correlations automatically.
    
    - temporal_imputation_min_periods (int): DEPRECATED
      Previously controlled minimum samples for rolling medians.
      Now ignored.
    
    Migration Guide
    ───────────────
    If you have existing configs with deprecated parameters:
    
    OLD CONFIG:
        imputation_config = ImputationConfig(
            use_advanced_feature_imputation=True,
            block_imputation_tiers={1: "training_mean", 3: "temporal_median", ...},
            temporal_imputation_window=5,
            n_imputations=1,
        )
    
    NEW CONFIG (equivalent, recommended):
        imputation_config = ImputationConfig(
            use_mice_imputation=True,
            n_imputations=5,  # Enable uncertainty quantification
            mice_estimator="extra_trees",
            mice_max_iter=20,
        )
    
    The old config will **still work** (deprecated params ignored);
    the new config is preferred for clarity.
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

