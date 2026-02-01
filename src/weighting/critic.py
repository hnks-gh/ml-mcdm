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
    >>> from src.weighting import CRITICWeightCalculator
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
    
    def calculate(self, data: pd.DataFrame) -> WeightResult:
        """
        Calculate CRITIC weights.
        
        Parameters
        ----------
        data : pd.DataFrame
            Decision matrix (alternatives × criteria)
        
        Returns
        -------
        WeightResult
            Calculated weights with standard deviation, conflict, 
            and correlation details
        """
        # Standard deviation (contrast intensity)
        std = data.std(axis=0)
        std = std.replace(0, self.epsilon)
        
        # Correlation matrix
        corr_matrix = data.corr()
        
        # Conflict measure
        conflict = (1 - corr_matrix).sum(axis=0)
        
        # Information content
        C = std * conflict
        
        # Normalize to weights
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
