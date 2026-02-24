# -*- coding: utf-8 -*-
"""
CRITIC Weight Calculator

Criteria Importance Through Inter-criteria Correlation method.
Considers both contrast intensity (standard deviation) and 
inter-criteria correlation to determine weights.

Mathematical Formula:
    w_j = C_j / Σ(C_k)
    
where:
    C_j = σ_j × Σ(1 - r_jk)  [information content]
    σ_j = standard deviation of criterion j
    r_jk = correlation between criteria j and k
"""

import numpy as np
import pandas as pd
from .base import WeightResult


class CRITICWeightCalculator:
    """
    CRITIC (Criteria Importance Through Inter-criteria Correlation) weights.
    
    The CRITIC method considers both:
    1. Contrast Intensity: Standard deviation of criterion values
    2. Conflicting Character: Correlation with other criteria
    
    Criteria with high variation AND low correlation with others
    receive higher weights as they provide unique information.
    
    Parameters
    ----------
    epsilon : float
        Small constant to avoid division by zero
    
    Attributes
    ----------
    epsilon : float
        Numerical stability constant
    
    Examples
    --------
    >>> import pandas as pd
    >>> from weighting import CRITICWeightCalculator
    >>> 
    >>> data = pd.DataFrame({
    ...     'C1': [0.8, 0.6, 0.9, 0.7],
    ...     'C2': [0.75, 0.55, 0.85, 0.65],  # Highly correlated with C1
    ...     'C3': [0.3, 0.9, 0.1, 0.7]       # Uncorrelated - higher weight
    ... })
    >>> 
    >>> calc = CRITICWeightCalculator()
    >>> result = calc.calculate(data)
    >>> print(result.weights)
    
    References
    ----------
    Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995).
    Determining objective weights in multiple criteria problems: 
    The CRITIC method. Computers & Operations Research.
    """
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
    
    def calculate(self, data: pd.DataFrame,
                  sample_weights: 'np.ndarray | None' = None) -> WeightResult:
        """
        Calculate CRITIC weights.
        
        Parameters
        ----------
        data : pd.DataFrame
            Decision matrix (alternatives × criteria)
        sample_weights : np.ndarray or None, optional
            Observation weights (length = n_alternatives).  When provided,
            the weighted covariance matrix is used to derive both the
            weighted standard deviations and the weighted Pearson
            correlation matrix.  Must sum to 1.
            If *None*, the unweighted formula is used.
        
        Returns
        -------
        WeightResult
            Calculated weights with standard deviation, conflict, 
            and correlation details
            
        Raises
        ------
        ValueError
            If data is empty or has less than 2 observations
        TypeError
            If data contains non-numeric columns
        """
        # Input validation
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        if len(data) < 2:
            raise ValueError("CRITIC calculation requires at least 2 observations")
        
        non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise TypeError(f"Non-numeric columns found: {non_numeric}")
        
        n = len(data)
        # Impute any NaN cells with the column mean before computing weights.
        # Upstream callers should pre-impute, but this is a defensive guard.
        # For wholly-missing columns fall back to epsilon (avoids NaN std/corr).
        data = data.copy()
        if data.isnull().any().any():
            _col_means = data.mean()
            _col_means = _col_means.fillna(self.epsilon)  # all-NaN col fallback
            data = data.fillna(_col_means)
        X = data.values  # (n, p)
        columns = data.columns.tolist()
        
        if sample_weights is not None:
            w = np.asarray(sample_weights, dtype=float)
            if w.shape[0] != n:
                raise ValueError(
                    f"sample_weights length ({w.shape[0]}) != n_observations ({n})")
            w = w / (w.sum() + self.epsilon)
            
            # Weighted covariance matrix via np.cov with aweights
            # np.cov returns bias-corrected estimate with aweights
            cov_matrix = np.cov(X.T, aweights=w)
            if cov_matrix.ndim == 0:
                cov_matrix = np.array([[float(cov_matrix)]])
            
            # Weighted std from diagonal of covariance
            std_arr = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))
            std_arr = np.where(std_arr < self.epsilon, self.epsilon, std_arr)
            
            # Weighted Pearson correlation from covariance
            # r_jk = cov_jk / (σ_j σ_k)
            outer_std = np.outer(std_arr, std_arr)
            outer_std = np.where(outer_std < self.epsilon, self.epsilon, outer_std)
            corr_arr = cov_matrix / outer_std
            # Clamp to [-1, 1] for numerical safety
            corr_arr = np.clip(corr_arr, -1.0, 1.0)
            
            corr_matrix = pd.DataFrame(corr_arr, index=columns, columns=columns)
            std = pd.Series(std_arr, index=columns)
        else:
            # Standard (unweighted) statistics
            std = data.std(axis=0)
            std = std.replace(0, self.epsilon)
            corr_matrix = data.corr()
        
        # Handle NaN values in correlation matrix (occurs with constant columns)
        corr_matrix = corr_matrix.fillna(1.0)
        
        # Conflict measure (sum of 1 - r_jk for all k)
        conflict = (1 - corr_matrix).sum(axis=0)
        conflict = conflict.clip(lower=self.epsilon)
        
        # Information content (std × conflict)
        C = std * conflict
        
        if C.sum() < self.epsilon:
            n_criteria = len(columns)
            weights = pd.Series(1.0 / n_criteria, index=columns)
        else:
            weights = C / C.sum()
        
        return WeightResult(
            weights=weights.to_dict(),
            method="critic",
            details={
                "std_values": std.to_dict(),
                "conflict_values": conflict.to_dict(),
                "information_content": C.to_dict(),
                "correlation_matrix": corr_matrix.to_dict()
            }
        )
