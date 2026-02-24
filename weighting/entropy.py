# -*- coding: utf-8 -*-
"""
Entropy Weight Calculator

Shannon entropy-based objective weight calculation method.
Assigns higher weights to criteria with more variation (information content).

Mathematical Formula:
    w_j = (1 - E_j) / Σ(1 - E_k)
    
where:
    E_j = -k × Σ(p_ij × ln(p_ij))  [entropy of criterion j]
    k = 1 / ln(m)  [normalization constant]
    p_ij = x_ij / Σx_ij  [proportion of value]
"""

import numpy as np
import pandas as pd
from .base import WeightResult


class EntropyWeightCalculator:
    """
    Shannon entropy-based objective weight calculation.
    
    The entropy method assigns higher weights to criteria that have 
    more variation across alternatives, as these provide more 
    information for distinguishing between options.
    
    Parameters
    ----------
    epsilon : float
        Small constant to avoid log(0) and division by zero
    
    Attributes
    ----------
    epsilon : float
        Numerical stability constant
    
    Examples
    --------
    >>> import pandas as pd
    >>> from weighting import EntropyWeightCalculator
    >>> 
    >>> data = pd.DataFrame({
    ...     'C1': [0.8, 0.6, 0.9, 0.7],
    ...     'C2': [0.5, 0.5, 0.5, 0.5],  # No variation - low weight
    ...     'C3': [0.3, 0.9, 0.1, 0.7]   # High variation - high weight
    ... })
    >>> 
    >>> calc = EntropyWeightCalculator()
    >>> result = calc.calculate(data)
    >>> print(result.weights)
    
    References
    ----------
    Shannon, C.E. (1948). A Mathematical Theory of Communication. 
    Bell System Technical Journal.
    """
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
    
    def calculate(self, data: pd.DataFrame,
                  sample_weights: 'np.ndarray | None' = None) -> WeightResult:
        """
        Calculate entropy weights.
        
        Parameters
        ----------
        data : pd.DataFrame
            Decision matrix (alternatives × criteria)
            All values should be positive
        sample_weights : np.ndarray or None, optional
            Observation weights (length = n_alternatives).  When provided,
            the proportion matrix and effective sample size are computed
            using these weights (continuous Bayesian Bootstrap support).
            Must sum to 1.  If *None*, all observations are weighted equally.
        
        Returns
        -------
        WeightResult
            Calculated weights with entropy and divergence details
            
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
            raise ValueError("Entropy calculation requires at least 2 observations")
        
        # Check for non-numeric columns
        non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise TypeError(f"Non-numeric columns found: {non_numeric}")
        
        n = len(data)
        
        # Validate / default observation weights
        if sample_weights is not None:
            w = np.asarray(sample_weights, dtype=float)
            if w.shape[0] != n:
                raise ValueError(
                    f"sample_weights length ({w.shape[0]}) != n_observations ({n})")
            w = w / (w.sum() + self.epsilon)  # ensure sums to 1
        else:
            w = np.full(n, 1.0 / n)
        
        # Validate input: entropy requires non-negative values
        # If negative values exist, shift data to positive range
        data_valid = data.copy()
        # Impute any NaN cells with the column mean before computing weights.
        # Upstream callers should pre-impute, but this is a defensive guard.
        # For columns that are entirely NaN (e.g., temporal split-half with
        # missing early-year data), data.mean() returns NaN and fillna(NaN)
        # is a no-op.  Fall back to epsilon so those columns carry no
        # information content (they receive zero weight via max entropy).
        if data_valid.isnull().any().any():
            _col_means = data_valid.mean()
            _col_means = _col_means.fillna(self.epsilon)  # all-NaN col fallback
            data_valid = data_valid.fillna(_col_means)
        for col in data_valid.columns:
            if data_valid[col].min() < 0:
                data_valid[col] = data_valid[col] - data_valid[col].min() + self.epsilon
        
        X = data_valid.values  # (n, p)
        
        # Weighted proportions: p_ij = w_i * x_ij / Σ_i(w_i * x_ij)
        wX = w[:, None] * X  # broadcast observation weights
        col_sums = wX.sum(axis=0)
        col_sums = np.clip(col_sums, self.epsilon, None)
        P = wX / col_sums
        P = np.clip(P, self.epsilon, None)
        
        # Effective sample size: N_eff = (Σw_i)² / Σw_i²  (Kish, 1965)
        n_eff = (w.sum() ** 2) / ((w ** 2).sum() + self.epsilon)
        n_eff = max(n_eff, 2.0)  # floor at 2 to keep log defined
        
        k = 1.0 / np.log(n_eff)
        E = -k * (P * np.log(P)).sum(axis=0)
        
        # Calculate divergence (information content)
        D = 1.0 - E
        D = np.clip(D, self.epsilon, None)
        
        # Normalize to criterion weights
        weights_arr = D / D.sum()
        
        columns = data.columns.tolist()
        weights_dict = {col: float(weights_arr[j]) for j, col in enumerate(columns)}
        E_dict = {col: float(E[j]) for j, col in enumerate(columns)}
        D_dict = {col: float(D[j]) for j, col in enumerate(columns)}
        
        return WeightResult(
            weights=weights_dict,
            method="entropy",
            details={
                "entropy_values": E_dict,
                "divergence_values": D_dict,
                "n_samples": n,
                "n_effective": float(n_eff),
            }
        )
