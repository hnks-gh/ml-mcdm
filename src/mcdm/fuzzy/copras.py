# -*- coding: utf-8 -*-
"""
Fuzzy COPRAS Method
===================

Fuzzy Complex Proportional Assessment method.

Extends COPRAS utility-based approach with Triangular Fuzzy Numbers
to handle uncertainty in benefit and cost assessments.

Mathematical Foundation:
    1. Normalize fuzzy decision matrix using sum normalization
    2. Calculate weighted normalized matrix: d̃_ij = w̃_j × r̃_ij
    3. Calculate fuzzy sums:
       - S̃⁺_i = Σ d̃_ij for benefit criteria
       - S̃⁻_i = Σ d̃_ij for cost criteria
    4. Calculate fuzzy relative significance:
       Q̃_i = S̃⁺_i + (S̃⁻_min × Σ S̃⁻) / (S̃⁻_i × Σ(S̃⁻/S̃⁻_i))
    5. Calculate utility degree: N_i = (Q_i / Q_max) × 100%

Reference:
    Zavadskas, E.K., Kaklauskas, A., & Sarka, V. (1994). The new method of 
    multicriteria complex proportional assessment of projects. 
    Technological and Economic Development of Economy, 1(3), 131-139.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .base import TriangularFuzzyNumber, FuzzyDecisionMatrix
from ...weighting import EnsembleWeightCalculator, WeightResult


@dataclass
class FuzzyCOPRASResult:
    """
    Result container for Fuzzy COPRAS calculation.
    
    Attributes:
        S_plus: Sum of weighted benefit criteria (defuzzified)
        S_minus: Sum of weighted cost criteria (defuzzified)
        Q: Relative significance (defuzzified)
        utility_degree: Utility degree N (percentage)
        fuzzy_S_plus: Fuzzy S+ values
        fuzzy_S_minus: Fuzzy S- values
        fuzzy_Q: Fuzzy Q values
        ranks: Final rankings
        weights: Criteria weights used
    """
    S_plus: pd.Series             # Sum of weighted benefit criteria (defuzzified)
    S_minus: pd.Series            # Sum of weighted cost criteria (defuzzified)
    Q: pd.Series                  # Relative significance (defuzzified)
    utility_degree: pd.Series     # Utility degree N (percentage)
    fuzzy_S_plus: Dict[str, TriangularFuzzyNumber]
    fuzzy_S_minus: Dict[str, TriangularFuzzyNumber]
    fuzzy_Q: Dict[str, TriangularFuzzyNumber]
    ranks: pd.Series
    weights: Dict[str, float]
    
    @property
    def final_ranks(self) -> pd.Series:
        """Get final ranking series."""
        return self.ranks
    
    def top_n(self, n: int = 10) -> pd.DataFrame:
        """Get top N alternatives with COPRAS metrics."""
        return pd.DataFrame({
            'S+': self.S_plus,
            'S-': self.S_minus,
            'Q': self.Q,
            'Utility_%': self.utility_degree,
            'Rank': self.ranks
        }).nsmallest(n, 'Rank')


class FuzzyCOPRAS:
    """
    Fuzzy COPRAS calculator using Triangular Fuzzy Numbers.
    
    Extends COPRAS utility-based method with fuzzy set theory to handle
    uncertainty in benefit and cost assessments. Particularly effective
    when criteria are clearly separable into benefit and cost categories.
    
    Parameters:
        defuzzification: Method for converting fuzzy to crisp
        benefit_criteria: List of benefit criteria names
        cost_criteria: List of cost criteria names (to be minimized)
    
    Example:
        >>> calculator = FuzzyCOPRAS(cost_criteria=['Cost', 'Risk'])
        >>> result = calculator.calculate_from_panel(panel_data)
        >>> print(f"Best alternative: {result.ranks.idxmin()}")
    """
    
    def __init__(self,
                 defuzzification: str = "centroid",
                 benefit_criteria: Optional[List[str]] = None,
                 cost_criteria: Optional[List[str]] = None):
        self.defuzzification = defuzzification
        self.benefit_criteria = benefit_criteria
        self.cost_criteria = cost_criteria or []
    
    def calculate(self,
                 data: pd.DataFrame,
                 weights: Union[Dict[str, float], WeightResult, None] = None,
                 uncertainty: Optional[pd.DataFrame] = None,
                 spread_ratio: float = 0.1
                 ) -> FuzzyCOPRASResult:
        """
        Calculate Fuzzy COPRAS from crisp data with uncertainty.
        
        Args:
            data: Decision matrix
            weights: Criteria weights
            uncertainty: Uncertainty matrix (optional)
            spread_ratio: Default spread ratio
        
        Returns:
            FuzzyCOPRASResult with utility assessments
        """
        fuzzy_matrix = FuzzyDecisionMatrix.from_crisp_with_uncertainty(
            data, uncertainty, spread_ratio
        )
        
        return self._calculate_fuzzy_copras(fuzzy_matrix, weights, data.columns.tolist())
    
    def calculate_from_panel(self,
                            panel_data,
                            weights: Union[Dict[str, float], WeightResult, None] = None,
                            spread_factor: float = 1.0
                            ) -> FuzzyCOPRASResult:
        """
        Calculate Fuzzy COPRAS using panel data temporal variance.
        
        Args:
            panel_data: Panel data object
            weights: Criteria weights
            spread_factor: Multiplier for std spread
        
        Returns:
            FuzzyCOPRASResult
        """
        fuzzy_matrix = FuzzyDecisionMatrix.from_panel_temporal_variance(
            panel_data, spread_factor
        )
        
        return self._calculate_fuzzy_copras(
            fuzzy_matrix, weights, panel_data.components
        )
    
    def _calculate_fuzzy_copras(self,
                                fuzzy_matrix: FuzzyDecisionMatrix,
                                weights: Union[Dict[str, float], WeightResult, None],
                                criteria: List[str]
                                ) -> FuzzyCOPRASResult:
        """Core Fuzzy COPRAS calculation."""
        alternatives = fuzzy_matrix.alternatives
        
        # Get weights
        if weights is None:
            crisp_data = fuzzy_matrix.to_crisp(self.defuzzification)
            weight_calc = EnsembleWeightCalculator()
            weight_result = weight_calc.calculate(crisp_data)
            weights = weight_result.weights
        elif isinstance(weights, WeightResult):
            weights = weights.weights
        
        weights = {col: weights.get(col, 1/len(criteria)) for col in criteria}
        
        # Determine benefit and cost criteria
        if self.benefit_criteria is None:
            self.benefit_criteria = [c for c in criteria if c not in self.cost_criteria]
        
        # Step 1: Normalize fuzzy decision matrix (sum normalization)
        normalized_matrix = self._normalize_fuzzy_matrix(fuzzy_matrix, criteria)
        
        # Step 2: Calculate weighted normalized fuzzy matrix
        weighted_matrix = self._apply_fuzzy_weights(normalized_matrix, weights, criteria)
        
        # Step 3: Calculate fuzzy S+ and S-
        fuzzy_S_plus = {}
        fuzzy_S_minus = {}
        
        benefit_cols = [c for c in criteria if c in self.benefit_criteria]
        cost_cols = [c for c in criteria if c in self.cost_criteria]
        
        for alt in alternatives:
            # Sum of benefit criteria
            s_plus = TriangularFuzzyNumber(0, 0, 0)
            for crit in benefit_cols:
                s_plus = s_plus + weighted_matrix.get(alt, crit)
            fuzzy_S_plus[alt] = s_plus
            
            # Sum of cost criteria
            s_minus = TriangularFuzzyNumber(0, 0, 0)
            for crit in cost_cols:
                s_minus = s_minus + weighted_matrix.get(alt, crit)
            fuzzy_S_minus[alt] = s_minus
        
        # Step 4: Calculate fuzzy Q (relative significance)
        fuzzy_Q = self._calculate_fuzzy_Q(fuzzy_S_plus, fuzzy_S_minus, alternatives)
        
        # Step 5: Defuzzify for ranking
        S_plus = pd.Series({alt: fuzzy_S_plus[alt].defuzzify(self.defuzzification) 
                          for alt in alternatives}, name='S_plus')
        S_minus = pd.Series({alt: fuzzy_S_minus[alt].defuzzify(self.defuzzification) 
                           for alt in alternatives}, name='S_minus')
        Q = pd.Series({alt: fuzzy_Q[alt].defuzzify(self.defuzzification) 
                      for alt in alternatives}, name='Q')
        
        # Step 6: Calculate utility degree
        Q_max = Q.max()
        utility_degree = (Q / Q_max * 100) if Q_max > 0 else pd.Series(0, index=alternatives)
        
        # Step 7: Rank alternatives
        ranks = Q.rank(ascending=False).astype(int)
        
        return FuzzyCOPRASResult(
            S_plus=S_plus,
            S_minus=S_minus,
            Q=Q,
            utility_degree=utility_degree,
            fuzzy_S_plus=fuzzy_S_plus,
            fuzzy_S_minus=fuzzy_S_minus,
            fuzzy_Q=fuzzy_Q,
            ranks=ranks,
            weights=weights
        )
    
    def _normalize_fuzzy_matrix(self,
                                fuzzy_matrix: FuzzyDecisionMatrix,
                                criteria: List[str]
                                ) -> FuzzyDecisionMatrix:
        """
        Normalize fuzzy matrix using sum normalization.
        
        r̃_ij = x̃_ij / Σ x̃_kj
        """
        normalized = FuzzyDecisionMatrix({}, fuzzy_matrix.alternatives, criteria)
        
        for crit in criteria:
            # Calculate sum for each criterion
            values = [fuzzy_matrix.get(alt, crit) for alt in fuzzy_matrix.alternatives]
            sum_l = sum(v.l for v in values)
            sum_m = sum(v.m for v in values)
            sum_u = sum(v.u for v in values)
            
            for alt in fuzzy_matrix.alternatives:
                v = fuzzy_matrix.get(alt, crit)
                
                # r_ij = x_ij / sum(x_ij)
                normalized.set(alt, crit, TriangularFuzzyNumber(
                    v.l / sum_u if sum_u > 0 else 0,
                    v.m / sum_m if sum_m > 0 else 0,
                    v.u / sum_l if sum_l > 0 else 0
                ))
        
        return normalized
    
    def _apply_fuzzy_weights(self,
                             fuzzy_matrix: FuzzyDecisionMatrix,
                             weights: Dict[str, float],
                             criteria: List[str]
                             ) -> FuzzyDecisionMatrix:
        """Apply weights to fuzzy matrix: d̃_ij = w_j × r̃_ij."""
        weighted = FuzzyDecisionMatrix({}, fuzzy_matrix.alternatives, criteria)
        
        for alt in fuzzy_matrix.alternatives:
            for crit in criteria:
                v = fuzzy_matrix.get(alt, crit)
                w = weights[crit]
                weighted.set(alt, crit, v * w)
        
        return weighted
    
    def _calculate_fuzzy_Q(self,
                          fuzzy_S_plus: Dict,
                          fuzzy_S_minus: Dict,
                          alternatives: List[str]
                          ) -> Dict:
        """
        Calculate fuzzy relative significance Q.
        
        Q_i = S+_i + (S-_min × Σ S-) / (S-_i × Σ(S-/S-_i))
        
        When no cost criteria: Q_i = S+_i
        """
        fuzzy_Q = {}
        
        # Check if we have any cost criteria
        has_cost = any(fuzzy_S_minus[alt].m > 0 for alt in alternatives)
        
        if not has_cost:
            # No cost criteria - Q = S+
            for alt in alternatives:
                fuzzy_Q[alt] = fuzzy_S_plus[alt]
        else:
            # Calculate S- min and sum
            S_minus_values = [fuzzy_S_minus[alt] for alt in alternatives]
            S_minus_min_m = min(v.m for v in S_minus_values) if S_minus_values else 0
            S_minus_sum_m = sum(v.m for v in S_minus_values) if S_minus_values else 0
            
            for alt in alternatives:
                S_plus = fuzzy_S_plus[alt]
                S_minus = fuzzy_S_minus[alt]
                
                # Calculate adjustment term
                if S_minus.m > 0 and S_minus_sum_m > 0:
                    # Sum of (S_minus_min / S_minus_i) for all i
                    ratio_sum = sum(S_minus_min_m / (v.m if v.m > 0 else 1e-10) 
                                   for v in S_minus_values)
                    
                    adjustment = (S_minus_min_m * S_minus_sum_m) / (S_minus.m * ratio_sum)
                    
                    fuzzy_Q[alt] = TriangularFuzzyNumber(
                        S_plus.l + adjustment * 0.8,  # Conservative lower
                        S_plus.m + adjustment,
                        S_plus.u + adjustment * 1.2   # Conservative upper
                    )
                else:
                    fuzzy_Q[alt] = S_plus
        
        return fuzzy_Q
