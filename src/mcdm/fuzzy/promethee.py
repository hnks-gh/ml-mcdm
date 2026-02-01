# -*- coding: utf-8 -*-
"""
Fuzzy PROMETHEE Method
======================

Fuzzy Preference Ranking Organization Method for Enrichment Evaluation.

Extends PROMETHEE outranking approach with Triangular Fuzzy Numbers
to handle uncertainty in pairwise preference comparisons.

Mathematical Foundation:
    1. For each pair (a, b), calculate fuzzy preference P̃(a, b)
    2. Aggregate using fuzzy weights: π̃(a, b) = Σ w̃_j × P̃_j(a, b)
    3. Calculate fuzzy outranking flows:
       - φ̃⁺(a) = (1/(n-1)) × Σ π̃(a, x)  (leaving flow)
       - φ̃⁻(a) = (1/(n-1)) × Σ π̃(x, a)  (entering flow)
    4. Net flow: φ̃(a) = φ̃⁺(a) - φ̃⁻(a)
    5. Rank by defuzzified net flow

Preference Functions:
    - usual: P(d) = 1 if d > 0, else 0
    - ushape: P(d) = 1 if d > q, else 0
    - vshape: P(d) = d/p if d < p, else 1
    - level: P(d) = 0.5 if q < d ≤ p, else usual
    - vshape_i: Linear with indifference threshold

Reference:
    Brans, J.P., & Vincke, P. (1985). A Preference Ranking Organisation Method.
    Management Science, 31(6), 647-656.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .base import TriangularFuzzyNumber, FuzzyDecisionMatrix
from ...weighting import EnsembleWeightCalculator, WeightResult


@dataclass
class FuzzyPROMETHEEResult:
    """
    Result container for Fuzzy PROMETHEE calculation.
    
    Attributes:
        phi_positive: Positive outranking flow (defuzzified)
        phi_negative: Negative outranking flow (defuzzified)
        phi_net: Net outranking flow (defuzzified)
        fuzzy_phi_positive: Fuzzy positive flows
        fuzzy_phi_negative: Fuzzy negative flows
        fuzzy_phi_net: Fuzzy net flows
        ranks: Final ranking by net flow
        preference_matrix: Aggregated preference matrix (defuzzified)
        weights: Criteria weights used
    """
    phi_positive: pd.Series       # Positive outranking flow (defuzzified)
    phi_negative: pd.Series       # Negative outranking flow (defuzzified)
    phi_net: pd.Series            # Net flow (defuzzified)
    fuzzy_phi_positive: Dict[str, TriangularFuzzyNumber]
    fuzzy_phi_negative: Dict[str, TriangularFuzzyNumber]
    fuzzy_phi_net: Dict[str, TriangularFuzzyNumber]
    ranks: pd.Series              # Final ranking by net flow
    preference_matrix: pd.DataFrame  # Aggregated preference (defuzzified)
    weights: Dict[str, float]
    
    @property
    def final_ranks(self) -> pd.Series:
        """Get final ranking series."""
        return self.ranks
    
    def top_n(self, n: int = 10) -> pd.DataFrame:
        """Get top N alternatives with PROMETHEE metrics."""
        return pd.DataFrame({
            'Phi+': self.phi_positive,
            'Phi-': self.phi_negative,
            'Phi_net': self.phi_net,
            'Rank': self.ranks
        }).nsmallest(n, 'Rank')


class FuzzyPROMETHEE:
    """
    Fuzzy PROMETHEE calculator using Triangular Fuzzy Numbers.
    
    Combines PROMETHEE outranking with fuzzy set theory to handle
    uncertainty in pairwise comparisons.
    
    Parameters:
        preference_function: Type of preference function
            ('usual', 'ushape', 'vshape', 'level', 'vshape_i')
        preference_threshold: Preference threshold p (for vshape)
        indifference_threshold: Indifference threshold q
        defuzzification: Method for converting fuzzy to crisp
        benefit_criteria: List of benefit criteria
        cost_criteria: List of cost criteria
    
    Example:
        >>> calculator = FuzzyPROMETHEE(preference_function='vshape', p=0.3, q=0.1)
        >>> result = calculator.calculate_from_panel(panel_data)
        >>> print(result.top_n(5))
    """
    
    def __init__(self,
                 preference_function: str = "vshape",
                 preference_threshold: float = 0.3,
                 indifference_threshold: float = 0.1,
                 defuzzification: str = "centroid",
                 benefit_criteria: Optional[List[str]] = None,
                 cost_criteria: Optional[List[str]] = None):
        self.preference_function = preference_function
        self.p = preference_threshold
        self.q = indifference_threshold
        self.defuzzification = defuzzification
        self.benefit_criteria = benefit_criteria
        self.cost_criteria = cost_criteria or []
    
    def calculate(self,
                 data: pd.DataFrame,
                 weights: Union[Dict[str, float], WeightResult, None] = None,
                 uncertainty: Optional[pd.DataFrame] = None,
                 spread_ratio: float = 0.1
                 ) -> FuzzyPROMETHEEResult:
        """
        Calculate Fuzzy PROMETHEE from crisp data with uncertainty.
        
        Args:
            data: Decision matrix
            weights: Criteria weights
            uncertainty: Uncertainty matrix (optional)
            spread_ratio: Default spread ratio
        
        Returns:
            FuzzyPROMETHEEResult with outranking flows
        """
        fuzzy_matrix = FuzzyDecisionMatrix.from_crisp_with_uncertainty(
            data, uncertainty, spread_ratio
        )
        
        return self._calculate_fuzzy_promethee(fuzzy_matrix, weights, data.columns.tolist())
    
    def calculate_from_panel(self,
                            panel_data,
                            weights: Union[Dict[str, float], WeightResult, None] = None,
                            spread_factor: float = 1.0
                            ) -> FuzzyPROMETHEEResult:
        """
        Calculate Fuzzy PROMETHEE using panel data temporal variance.
        
        Args:
            panel_data: Panel data object
            weights: Criteria weights
            spread_factor: Multiplier for std spread
        
        Returns:
            FuzzyPROMETHEEResult
        """
        fuzzy_matrix = FuzzyDecisionMatrix.from_panel_temporal_variance(
            panel_data, spread_factor
        )
        
        return self._calculate_fuzzy_promethee(
            fuzzy_matrix, weights, panel_data.components
        )
    
    def _calculate_fuzzy_promethee(self,
                                   fuzzy_matrix: FuzzyDecisionMatrix,
                                   weights: Union[Dict[str, float], WeightResult, None],
                                   criteria: List[str]
                                   ) -> FuzzyPROMETHEEResult:
        """Core Fuzzy PROMETHEE calculation."""
        alternatives = fuzzy_matrix.alternatives
        n = len(alternatives)
        
        # Get weights
        if weights is None:
            crisp_data = fuzzy_matrix.to_crisp(self.defuzzification)
            weight_calc = EnsembleWeightCalculator()
            weight_result = weight_calc.calculate(crisp_data)
            weights = weight_result.weights
        elif isinstance(weights, WeightResult):
            weights = weights.weights
        
        weights = {col: weights.get(col, 1/len(criteria)) for col in criteria}
        
        # Normalize fuzzy matrix
        normalized_matrix = self._normalize_fuzzy_matrix(fuzzy_matrix, criteria)
        
        # Calculate fuzzy preference matrix π(a,b)
        fuzzy_pref_matrix = {}
        crisp_pref_matrix = np.zeros((n, n))
        
        for i, a in enumerate(alternatives):
            fuzzy_pref_matrix[a] = {}
            for j, b in enumerate(alternatives):
                if i != j:
                    # Calculate aggregated preference π(a,b)
                    pref = self._calculate_fuzzy_preference(
                        normalized_matrix, a, b, weights, criteria
                    )
                    fuzzy_pref_matrix[a][b] = pref
                    crisp_pref_matrix[i, j] = pref.defuzzify(self.defuzzification)
                else:
                    fuzzy_pref_matrix[a][b] = TriangularFuzzyNumber(0, 0, 0)
        
        # Calculate outranking flows
        fuzzy_phi_positive = {}
        fuzzy_phi_negative = {}
        fuzzy_phi_net = {}
        
        for i, a in enumerate(alternatives):
            # Phi+ (leaving flow)
            phi_plus_sum = TriangularFuzzyNumber(0, 0, 0)
            for b in alternatives:
                if a != b:
                    phi_plus_sum = phi_plus_sum + fuzzy_pref_matrix[a][b]
            
            divisor = (n - 1) if n > 1 else 1
            fuzzy_phi_positive[a] = phi_plus_sum * (1.0 / divisor)
            
            # Phi- (entering flow)
            phi_minus_sum = TriangularFuzzyNumber(0, 0, 0)
            for b in alternatives:
                if a != b:
                    phi_minus_sum = phi_minus_sum + fuzzy_pref_matrix[b][a]
            
            fuzzy_phi_negative[a] = phi_minus_sum * (1.0 / divisor)
            
            # Net flow
            fuzzy_phi_net[a] = fuzzy_phi_positive[a] - fuzzy_phi_negative[a]
        
        # Defuzzify for ranking
        phi_positive = pd.Series({alt: fuzzy_phi_positive[alt].defuzzify(self.defuzzification) 
                                 for alt in alternatives}, name='Phi+')
        phi_negative = pd.Series({alt: fuzzy_phi_negative[alt].defuzzify(self.defuzzification) 
                                 for alt in alternatives}, name='Phi-')
        phi_net = pd.Series({alt: fuzzy_phi_net[alt].defuzzify(self.defuzzification) 
                           for alt in alternatives}, name='Phi_net')
        
        # Rank by net flow
        ranks = phi_net.rank(ascending=False).astype(int)
        
        pref_df = pd.DataFrame(crisp_pref_matrix, index=alternatives, columns=alternatives)
        
        return FuzzyPROMETHEEResult(
            phi_positive=phi_positive,
            phi_negative=phi_negative,
            phi_net=phi_net,
            fuzzy_phi_positive=fuzzy_phi_positive,
            fuzzy_phi_negative=fuzzy_phi_negative,
            fuzzy_phi_net=fuzzy_phi_net,
            ranks=ranks,
            preference_matrix=pref_df,
            weights=weights
        )
    
    def _normalize_fuzzy_matrix(self,
                                fuzzy_matrix: FuzzyDecisionMatrix,
                                criteria: List[str]
                                ) -> FuzzyDecisionMatrix:
        """Normalize fuzzy decision matrix to [0, 1] range."""
        normalized = FuzzyDecisionMatrix({}, fuzzy_matrix.alternatives, criteria)
        
        for crit in criteria:
            values = [fuzzy_matrix.get(alt, crit) for alt in fuzzy_matrix.alternatives]
            
            # Get max of upper bounds for normalization
            max_u = max(v.u for v in values) if values else 1
            min_l = min(v.l for v in values) if values else 0
            range_val = max_u - min_l if max_u - min_l > 0 else 1
            
            for alt in fuzzy_matrix.alternatives:
                v = fuzzy_matrix.get(alt, crit)
                
                if crit in self.cost_criteria:
                    # Cost criteria: invert
                    normalized.set(alt, crit, TriangularFuzzyNumber(
                        (max_u - v.u) / range_val,
                        (max_u - v.m) / range_val,
                        (max_u - v.l) / range_val
                    ))
                else:
                    # Benefit criteria
                    normalized.set(alt, crit, TriangularFuzzyNumber(
                        (v.l - min_l) / range_val,
                        (v.m - min_l) / range_val,
                        (v.u - min_l) / range_val
                    ))
        
        return normalized
    
    def _calculate_fuzzy_preference(self,
                                    fuzzy_matrix: FuzzyDecisionMatrix,
                                    a: str,
                                    b: str,
                                    weights: Dict[str, float],
                                    criteria: List[str]
                                    ) -> TriangularFuzzyNumber:
        """Calculate aggregated fuzzy preference π(a,b) = Σ w_j × P_j(a,b)."""
        pref_sum = TriangularFuzzyNumber(0, 0, 0)
        
        for crit in criteria:
            f_a = fuzzy_matrix.get(a, crit)
            f_b = fuzzy_matrix.get(b, crit)
            w = weights[crit]
            
            # Calculate fuzzy difference d(a,b)
            d = TriangularFuzzyNumber(
                f_a.l - f_b.u,
                f_a.m - f_b.m,
                f_a.u - f_b.l
            )
            
            # Apply preference function to fuzzy difference
            p = self._fuzzy_preference_value(d)
            
            # Weighted preference
            weighted_p = p * w
            pref_sum = pref_sum + weighted_p
        
        return pref_sum
    
    def _fuzzy_preference_value(self, d: TriangularFuzzyNumber) -> TriangularFuzzyNumber:
        """
        Calculate fuzzy preference value P(d) based on preference function.
        
        Returns fuzzy preference degree based on the selected preference function.
        """
        # Use defuzzified value for preference function application
        d_crisp = d.defuzzify(self.defuzzification)
        
        if d_crisp <= 0:
            return TriangularFuzzyNumber(0, 0, 0)
        
        if self.preference_function == "usual":
            # Usual: strict preference
            p = 1.0 if d_crisp > 0 else 0.0
        
        elif self.preference_function == "ushape":
            # U-shape: quasi-criterion
            p = 1.0 if d_crisp > self.q else 0.0
        
        elif self.preference_function == "vshape":
            # V-shape: linear preference
            if d_crisp >= self.p:
                p = 1.0
            else:
                p = d_crisp / self.p
        
        elif self.preference_function == "level":
            # Level criterion
            if d_crisp <= self.q:
                p = 0.0
            elif d_crisp > self.p:
                p = 1.0
            else:
                p = 0.5
        
        elif self.preference_function == "vshape_i":
            # V-shape with indifference
            if d_crisp <= self.q:
                p = 0.0
            elif d_crisp >= self.p:
                p = 1.0
            else:
                p = (d_crisp - self.q) / (self.p - self.q) if self.p > self.q else 1.0
        
        else:
            # Default: linear
            p = min(1.0, max(0.0, d_crisp))
        
        # Return as fuzzy number with spread based on input uncertainty
        spread = abs(d.u - d.l) / 6  # Reduced spread for preference
        return TriangularFuzzyNumber(
            max(0, p - spread),
            p,
            min(1, p + spread)
        )
