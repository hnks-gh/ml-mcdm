from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class ImputationConfig:
    """
    Configuration for missing data handling architecture.
    Tier 3 Enhancement M-12.
    """
    # Imputation methods
    decision_matrix_method: str = "temporal_linear"  # "column_median", "temporal_linear", "cp_als", "gp_spatiotemporal"
    ml_feature_method: str = "mice_missforest"        # "training_mean", "mice_missforest", "multiple"
    
    # Parameters for MICE (M-02, M-08)
    n_imputations: int = 1
    mice_max_iter: int = 15
    mice_n_nearest_features: int = 20
    mice_estimator: Literal["random_forest", "extra_trees", "bayesian_ridge"] = "extra_trees"
    
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
    add_missingness_indicators: bool = True
    random_state: int = 42

@dataclass
class ImputationAudit:
    """
    Audit record for metadata tracking (M-14).
    """
    n_cells_original: int
    n_imputed: int
    missingness_rate_initial: float
    method_applied: str
    mcar_p_value: Optional[float] = None
    mechanism_assessment: str = "unknown"
    imputed_locations: List[Tuple[int, int]] = field(default_factory=list)
    imputation_errors: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

# Import specialized engines (to be populated later)
# from .iterative import MICEImputer, MissForestImputer
# from .tensor import CPALSImputer
# from .gp import GPImputer
