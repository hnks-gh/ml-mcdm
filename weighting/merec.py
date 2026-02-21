# -*- coding: utf-8 -*-
"""
MEREC (Method based on Removal Effects of Criteria) weight calculator.

Measures criterion importance by impact of removal on overall performance.
Reference: Keshavarz-Ghorabaee et al. (2021), Symmetry, 13(4), 525.
"""

import numpy as np
import pandas as pd
from .base import WeightResult


class MERECWeightCalculator:
    """
    MEREC weight calculator.
    
    Parameters
    ----------
    epsilon : float, default=1e-10
        Numerical stability constant.
    """
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
    
    def calculate(self, data: pd.DataFrame,
                  sample_weights: 'np.ndarray | None' = None) -> WeightResult:
        """
        Calculate MEREC weights from decision matrix.
        
        Parameters
        ----------
        data : pd.DataFrame
            Decision matrix (alternatives × criteria)
        sample_weights : np.ndarray or None, optional
            Observation weights (length = n_alternatives).  When provided,
            the removal-effect sum is weighted:
            E_j = Σ_i  w_i |ln S_i^(j) − ln S_i|
            Must sum to 1.  If *None*, uniform weighting is used.
            
        Returns
        -------
        WeightResult
            Calculated weights with removal effects details
            
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
            raise ValueError("MEREC calculation requires at least 2 observations")
        
        non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise TypeError(f"Non-numeric columns found: {non_numeric}")
        
        n_alternatives, n_criteria = data.shape
        columns = data.columns.tolist()
        
        # Validate / default observation weights
        if sample_weights is not None:
            w = np.asarray(sample_weights, dtype=float)
            if w.shape[0] != n_alternatives:
                raise ValueError(
                    f"sample_weights length ({w.shape[0]}) "
                    f"!= n_alternatives ({n_alternatives})")
            w = w / (w.sum() + self.epsilon)
        else:
            w = np.full(n_alternatives, 1.0 / n_alternatives)
        
        # Step 1: Min-Max Normalization to [0, 1]
        X_norm = self._normalize(data.values)
        
        # Step 2: Calculate overall performance with all criteria
        S_overall = self._calculate_performance(X_norm)
        
        # Step 3-4: Calculate weighted removal effect for each criterion
        removal_effects = np.zeros(n_criteria)
        S_without = {}
        
        for j in range(n_criteria):
            X_removed = X_norm.copy()
            X_removed[:, j] = 1.0
            
            S_j = self._calculate_performance(X_removed)
            S_without[columns[j]] = S_j
            
            # Weighted removal effect: E_j = Σ w_i |ln(S_i^j) - ln(S_i)|
            abs_diff = np.abs(
                np.log(S_j + self.epsilon) - np.log(S_overall + self.epsilon)
            )
            removal_effects[j] = np.dot(w, abs_diff)
        
        # Step 5: Normalize to criterion weights
        removal_effects = np.clip(removal_effects, self.epsilon, None)
        weights = removal_effects / removal_effects.sum()
        
        return WeightResult(
            weights={col: float(weights[j]) for j, col in enumerate(columns)},
            method="merec",
            details={
                "removal_effects": {col: float(removal_effects[j]) 
                                   for j, col in enumerate(columns)},
                "overall_performance": S_overall.tolist(),
                "n_alternatives": n_alternatives,
                "n_criteria": n_criteria,
                "interpretation": "Higher weights indicate criteria whose removal "
                                "significantly impacts alternative performance rankings."
            }
        )
    
    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Min-max normalization to [epsilon, 1]."""
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        
        denominator = X_max - X_min
        denominator[denominator < self.epsilon] = self.epsilon
        
        # Normalize to [0, 1] then shift to [epsilon, 1]
        X_norm = (X - X_min) / denominator
        X_norm = X_norm * (1 - self.epsilon) + self.epsilon
        
        return X_norm
    
    def _calculate_performance(self, X_norm: np.ndarray) -> np.ndarray:
        """Calculate overall performance: S_i = ln(1 + (1/n)Σ|ln(x_ij)|)."""
        n_alternatives, n_criteria = X_norm.shape
        
        # Ensure all values are positive for logarithm
        X_safe = np.clip(X_norm, self.epsilon, None)
        
        # Calculate S_i for each alternative
        log_sum = np.sum(np.abs(np.log(X_safe)), axis=1) / n_criteria
        S = np.log(1 + log_sum)
        
        return S
