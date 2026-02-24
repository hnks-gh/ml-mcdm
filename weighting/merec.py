# -*- coding: utf-8 -*-
"""
MEREC (Method based on Removal Effects of Criteria) weight calculator.

Measures criterion importance by impact of removal on overall performance.
Reference: Keshavarz-Ghorabaee et al. (2021), Symmetry, 13(4), 525.

Normalization:
    Benefit criterion j:  n_ij = min_i(x_ij) / x_ij
    Cost    criterion j:  n_ij = x_ij         / max_i(x_ij)
Values are clipped to [epsilon, 1].  This ratio-based normalization
preserves rank-order relationships and matches the published method;
it differs from a plain min-max shift.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from .base import WeightResult


class MERECWeightCalculator:
    """
    MEREC weight calculator.

    Parameters
    ----------
    epsilon : float, default=1e-10
        Numerical stability constant.
    cost_criteria : list of str, optional
        Names of criteria where *lower* values are preferred (cost type).
        All other criteria are treated as benefit type.
        Defaults to an empty list (all benefit).
    """

    def __init__(
        self,
        epsilon: float = 1e-10,
        cost_criteria: Optional[List[str]] = None,
    ):
        self.epsilon = epsilon
        self.cost_criteria: List[str] = list(cost_criteria or [])
    
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

        # Impute any NaN cells with the column mean before computing weights.
        # Upstream callers should pre-impute, but this is a defensive guard.
        # For wholly-missing columns fall back to epsilon so that the ratio
        # normalisation in _normalize produces 1.0 (|ln 1| = 0), giving that
        # column zero removal effect and therefore zero weight — which is the
        # correct behaviour when no data exists for that criterion.
        data = data.copy()
        if data.isnull().any().any():
            _col_means = data.mean()
            _col_means = _col_means.fillna(self.epsilon)  # all-NaN col fallback
            data = data.fillna(_col_means)
        
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
        
        # Step 1: Ratio normalization per Keshavarz-Ghorabaee et al. (2021)
        X_norm = self._normalize(data.values, columns)
        
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
    
    def _normalize(self, X: np.ndarray, columns: list) -> np.ndarray:
        """
        Ratio-based normalization per Keshavarz-Ghorabaee et al. (2021).

        For each criterion j:
          - Benefit: n_ij = min_i(x_ij) / x_ij
          - Cost:    n_ij = x_ij        / max_i(x_ij)

        Values are clipped to [epsilon, 1] to stay in the valid log domain.
        """
        X_norm = np.empty_like(X, dtype=float)
        for j, col in enumerate(columns):
            col_vals = X[:, j]
            if col in self.cost_criteria:
                # cost: smaller is better → n_ij = x_ij / max
                col_max = col_vals.max()
                denom = col_max if col_max > self.epsilon else self.epsilon
                X_norm[:, j] = col_vals / denom
            else:
                # benefit: larger is better → n_ij = min / x_ij
                col_min = col_vals.min()
                safe_col = np.where(col_vals > self.epsilon, col_vals, self.epsilon)
                X_norm[:, j] = (
                    col_min / safe_col
                    if col_min > self.epsilon
                    else self.epsilon / safe_col
                )
        return np.clip(X_norm, self.epsilon, 1.0)
    
    def _calculate_performance(self, X_norm: np.ndarray) -> np.ndarray:
        """Calculate overall performance: S_i = ln(1 + (1/n)Σ|ln(x_ij)|)."""
        n_alternatives, n_criteria = X_norm.shape
        
        # Ensure all values are positive for logarithm
        X_safe = np.clip(X_norm, self.epsilon, None)
        
        # Calculate S_i for each alternative
        log_sum = np.sum(np.abs(np.log(X_safe)), axis=1) / n_criteria
        S = np.log(1 + log_sum)
        
        return S
