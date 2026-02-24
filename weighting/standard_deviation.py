# -*- coding: utf-8 -*-
"""
Standard Deviation weight calculator.

Simple variance-based weighting: w_j = σ_j / Σσ_k
Reference: Wang & Luo (2010), Math & Computer Modelling, 51(1-2), 1-12.
"""

import numpy as np
import pandas as pd
from .base import WeightResult


class StandardDeviationWeightCalculator:
    """
    Standard Deviation weight calculator.
    
    Parameters
    ----------
    epsilon : float, default=1e-10
        Numerical stability constant.
    ddof : int, default=1
        Degrees of freedom for std calculation.
    """
    
    def __init__(self, epsilon: float = 1e-10, ddof: int = 1):
        self.epsilon = epsilon
        self.ddof = ddof
    
    def calculate(self, data: pd.DataFrame,
                  sample_weights: 'np.ndarray | None' = None) -> WeightResult:
        """
        Calculate standard deviation weights from decision matrix.
        
        Parameters
        ----------
        data : pd.DataFrame
            Decision matrix (alternatives × criteria)
        sample_weights : np.ndarray or None, optional
            Observation weights (length = n_alternatives).  When provided,
            the weighted standard deviation is used:
            σ_j^w = sqrt( Σ w_i (x_ij − x̄_j^w)² )
            with x̄_j^w = Σ w_i x_ij.
            Must sum to 1.  If *None*, the unweighted formula is used.
            
        Returns
        -------
        WeightResult
            Calculated weights with std and CV details
            
        Raises
        ------
        ValueError
            If data is empty or has insufficient observations for ddof
        TypeError
            If data contains non-numeric columns
        """
        # Input validation
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        if len(data) <= self.ddof:
            raise ValueError(f"Need more than {self.ddof} observations for ddof={self.ddof}")
        
        non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise TypeError(f"Non-numeric columns found: {non_numeric}")
        
        n = len(data)
        # Impute any NaN cells with the column mean before computing weights.
        # Upstream callers should pre-impute, but this is a defensive guard.
        data = data.copy()
        if data.isnull().any().any():
            data = data.fillna(data.mean())
        X = data.values  # (n, p)
        columns = data.columns.tolist()
        
        if sample_weights is not None:
            w = np.asarray(sample_weights, dtype=float)
            if w.shape[0] != n:
                raise ValueError(
                    f"sample_weights length ({w.shape[0]}) != n_observations ({n})")
            w = w / (w.sum() + self.epsilon)
            
            # Weighted mean: x̄_j = Σ w_i x_ij
            wmean = w @ X  # (p,)
            # Weighted std: σ_j = sqrt(Σ w_i (x_ij - x̄_j)²)
            # with reliability-weights Bessel correction:
            #   V1 = Σw_i = 1,  V2 = Σw_i²
            #   σ² = (1 / (V1 - V2)) * Σ w_i (x - x̄)²
            deviations = X - wmean  # (n, p)
            var_biased = (w[:, None] * deviations ** 2).sum(axis=0)  # (p,)
            V2 = (w ** 2).sum()
            correction = 1.0 / max(1.0 - V2, self.epsilon)  # Bessel for weights
            std_arr = np.sqrt(var_biased * correction)
            mean_arr = wmean
        else:
            std_arr = data.std(axis=0, ddof=self.ddof).values
            mean_arr = data.mean(axis=0).values
        
        std_arr = np.where(std_arr < self.epsilon, self.epsilon, std_arr)
        mean_arr = np.where(np.abs(mean_arr) < self.epsilon, self.epsilon, mean_arr)
        
        # Normalize to weights
        weights_arr = std_arr / std_arr.sum()
        
        cv_arr = std_arr / np.abs(mean_arr)
        data_range = data.max(axis=0) - data.min(axis=0)
        
        std_dict = {col: float(std_arr[j]) for j, col in enumerate(columns)}
        cv_dict = {col: float(cv_arr[j]) for j, col in enumerate(columns)}
        weights_dict = {col: float(weights_arr[j]) for j, col in enumerate(columns)}
        mean_dict = {col: float(mean_arr[j]) for j, col in enumerate(columns)}
        
        return WeightResult(
            weights=weights_dict,
            method="standard_deviation",
            details={
                "std_values": std_dict,
                "coefficient_of_variation": cv_dict,
                "range_values": data_range.to_dict(),
                "mean_values": mean_dict,
                "n_samples": n,
                "ddof": self.ddof,
                "interpretation": "Higher weights indicate criteria with more "
                                "variation (dispersion) across alternatives."
            }
        )
