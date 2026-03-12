from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import numpy as np
import pandas as pd
from .__init__ import ImputationConfig, ImputationAudit
from .mechanisms import test_missing_mechanism
from .iterative import MICEImputer

class UnifiedImputationOrchestrator:
    """
    Tier 3 Enhancement M-12: Unified Panel Imputation Architecture.
    
    Orchestrates missing data diagnostics (M-11), iterative imputation (MICE/M-02/M-08),
    and creates an audit trail (M-14).
    """
    def __init__(self, config: Optional[ImputationConfig] = None):
        self.config = config or ImputationConfig()
        self.mice_imputer = MICEImputer(self.config)
        self.audit_log: List[ImputationAudit] = []
        self._is_fitted = False
        
    def _create_audit(
        self, 
        df: pd.DataFrame, 
        method: str, 
        mechanism_p: Optional[float] = None,
        assessment: str = "unknown"
    ) -> ImputationAudit:
        """Create a snapshot of imputation state for M-14 audit."""
        mask = df.isna()
        n_nan = mask.sum().sum()
        rate = n_nan / (df.shape[0] * df.shape[1])
        
        # Get locations of 10 sample missing cells
        nan_indices = np.argwhere(mask.values)
        locations = [(int(r), int(c)) for r, c in nan_indices[:10]]
        
        return ImputationAudit(
            n_cells_original=df.size,
            n_imputed=int(n_nan),
            missingness_rate_initial=float(rate),
            method_applied=method,
            mcar_p_value=mechanism_p,
            mechanism_assessment=assessment,
            imputed_locations=locations
        )

    def process_panel(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Full imputation pipeline with diagnostics and nested strategies.
        """
        # 1. Mechanism Diagnostic (M-11)
        diag = {}
        if self.config.enable_mcar_test:
            try:
                diag = test_missing_mechanism(df)
            except Exception as e:
                diag = {"suggested_mechanism": "unknown", "mcar_p_value": None}
                
        # 2. Select Imputation Strategy
        method = self.config.ml_feature_method
        
        # Create initial audit (M-14)
        audit = self._create_audit(
            df, 
            method, 
            mechanism_p=diag.get("mcar_p_value"),
            assessment=diag.get("suggested_mechanism", "unknown")
        )
        
        # 3. Apply Imputation
        try:
            if is_training:
                imputed_arr = self.mice_imputer.fit_transform(df.values)
                self._is_fitted = True
            else:
                imputed_arr = self.mice_imputer.transform(df.values)
                
            # Convert back to DataFrame
            # MICE appends columns if add_indicator=True
            if self.config.add_missingness_indicators:
                # Need to construct new column names for mask indicators
                indicator_cols = [f"was_missing_{c}" for c in df.columns]
                all_cols = list(df.columns) + indicator_cols
                # IterativeImputer appends indicators at the end
                result_df = pd.DataFrame(imputed_arr, index=df.index, columns=all_cols)
            else:
                result_df = pd.DataFrame(imputed_arr, index=df.index, columns=df.columns)
                
        except Exception as e:
            # Fallback to mean (M-01 logic)
            audit.imputation_errors["strategy_failure"] = str(e)
            fallback = self.mice_imputer.get_fallback_values()
            mask = df.isna().values
            filled = np.where(mask, fallback[np.newaxis, :], df.values)
            result_df = pd.DataFrame(filled, index=df.index, columns=df.columns)
            audit.method_applied = "training_mean_fallback"
            
        self.audit_log.append(audit)
        return result_df

    def get_last_audit(self) -> Optional[ImputationAudit]:
        """Retrieve the most recent imputation audit record."""
        return self.audit_log[-1] if self.audit_log else None
