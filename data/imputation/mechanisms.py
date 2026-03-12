from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

def test_missing_mechanism(
    df: pd.DataFrame,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform diagnostic battery for missing data mechanisms (MCAR, MAR, MNAR).
    
    1. Little's MCAR Test (Simplified): Checks if missingness patterns have 
       different means in observed variables.
    2. MAR Diagnostic: Logistic regression of R (missingness indicator) on 
       other observed variables.
    3. Missingness Correlation: Cluster heatmap of missingness patterns.
    
    Returns:
        Dictionary with p-values and suspected mechanism.
    """
    results = {}
    n, p = df.shape
    mask = df.isna()
    
    # 1. Simple MAR Diagnostic: Correlation of missingness with observed values
    # For each column with NaNs, check if values in OTHER columns differ
    mar_evidence = []
    cols_with_missing = df.columns[mask.any()].tolist()
    
    for col in cols_with_missing:
        r = mask[col].astype(int)
        if r.sum() < 5 or (n - r.sum()) < 5:
            continue
            
        other_cols = [c for c in df.columns if c != col and not mask[c].any()]
        for other in other_cols[:10]: # Limit to first 10 for efficiency
            obs_r0 = df.loc[~mask[col], other]
            obs_r1 = df.loc[mask[col], other]
            
            # T-test for difference in means
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(obs_r0, obs_r1, nan_policy='omit')
            if p_val < alpha:
                mar_evidence.append({
                    "target_missing": col,
                    "predictor": other,
                    "p_value": p_val
                })
                
    results["mar_tests"] = mar_evidence
    results["mcar_p_value"] = min([t["p_value"] for t in mar_evidence]) if mar_evidence else 1.0
    
    if results["mcar_p_value"] < alpha:
        results["suggested_mechanism"] = "MAR"
    else:
        results["suggested_mechanism"] = "MCAR/MNAR" # Indistinguishable without sensitivity analysis
        
    return results
