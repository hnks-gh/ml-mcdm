from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class ImputationConfig:
    """
    Configuration for missing data handling architecture (M-12).
    
    TIER STRATEGY for Feature Blocks (PHASE A Enhancement):
    ────────────────────────────────────────────────────────
    Tier 1 (MICE):        Multivariate correlation imputation via ExtraTreesRegressor
    Tier 2 (Temporal):    Per-entity annual medians for time-series blocks
    Tier 3 (Sectional):   Cross-sectional medians for cross-entity features
    Tier 4 (Fallback):    Training means cache with 0.0 emergency sentinel
    
    Example block assignments:
    - Block 3 (rolling stats) → Tier 2 (temporal)
    - Block 5 (demeaned) → Tier 3 (sectional)
    - Block 7 (EWMA) → Tier 2 (temporal)
    - All blocks → Tier 1 (MICE) as first pass if enabled
    """
    # Imputation methods
    decision_matrix_method: str = "temporal_linear"  # "column_median", "temporal_linear", "cp_als", "gp_spatiotemporal"
    ml_feature_method: str = "mice_missforest"        # "training_mean", "mice_missforest", "multiple"
    
    # ========== PHASE A ENHANCEMENT: Block-level Imputation Strategy ==========
    
    # NEW: Use advanced tiered imputation for feature blocks (default: True)
    # Set to False to restore legacy 0.0-fallback behavior for regression testing
    use_advanced_feature_imputation: bool = True
    
    # NEW: Explicit block tier assignments (override defaults if needed)
    # Key: block_number (1-12), Value: "mice" | "temporal_median" | "cross_sectional_median" | "training_mean"
    block_imputation_tiers: Dict[int, str] = field(default_factory=lambda: {
        1:   "training_mean",           # Current values: numerical fallback
        2:   "cross_sectional_median",  # Lag features: already has _was_missing indicator
        3:   "temporal_median",         # Rolling stats: respect time-series continuity
        4:   "temporal_median",         # Momentum/acceleration: derivatives → temporal median
        5:   "cross_sectional_median",  # Entity-demeaned: cross-sectional relative
        6:   "cross_sectional_median",  # Polyfit trend: entity-level estimate
        7:   "temporal_median",         # EWMA levels: exponentially-weighted temporal
        8:   "temporal_median",         # Expanding mean: cumulative temporal
        9:   "cross_sectional_median",  # Inter-criterion diversity: cross-sectional spread
        10:  "temporal_median",         # Rolling skewness: temporal distribution
        11:  "cross_sectional_median",  # Panel percentile/rank: cross-sectional
        12:  None,                      # Regional dummies: categorical (no imputation)
    })
    
    # NEW: Parameters for temporal imputation (Tier 2)
    temporal_imputation_window: int = 5  # Years for rolling median window (robustness)
    temporal_imputation_min_periods: int = 2  # Minimum valid points required
    
    # ========================================================================
    
    # Parameters for MICE (M-02, M-08) — Tier 1
    n_imputations: int = 1
    mice_max_iter: int = 20  # Increased from 15 for higher-dim feature spaces
    mice_n_nearest_features: int = 30  # Increased from 20 for block-based application
    mice_estimator: Literal["random_forest", "extra_trees", "bayesian_ridge"] = "extra_trees"
    mice_add_indicator: bool = True  # Append _was_imputed binary columns
    
    # Parameters for CP-ALS (M-09)
    cp_rank: int = 8
    cp_max_iter: int = 500
    cp_tol: float = 1e-5
    cp_lambda_reg: float = 0.01
    
    # Parameters for GP (M-06)
    gp_length_scale_temporal: float = 3.0
    gp_length_scale_spatial: float = 2.0
    
    # Advanced Options
    enable_mcar_test: bool = True
    add_missingness_indicators: bool = True  # Global flag; per-block control via block_imputation_tiers
    random_state: int = 42

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

# Import specialized engines (to be populated later)
# from .iterative import MICEImputer, MissForestImputer
# from .tensor import CPALSImputer
# from .gp import GPImputer
