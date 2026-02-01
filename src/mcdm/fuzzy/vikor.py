# -*- coding: utf-8 -*-
"""
Fuzzy VIKOR Method
==================

Fuzzy VIseKriterijumska Optimizacija I Kompromisno Resenje
(Multicriteria Optimization and Compromise Solution).

Extends VIKOR with Triangular Fuzzy Numbers to find compromise solutions
under uncertainty, balancing maximum group utility and minimum regret.

Mathematical Foundation:
    1. Determine fuzzy best (f̃*) and worst (f̃⁻) for each criterion
    2. Calculate fuzzy S (group utility): S̃_i = Σ w̃_j × (f̃_j* - f̃_ij) / (f̃_j* - f̃_j⁻)
    3. Calculate fuzzy R (individual regret): R̃_i = max_j [w̃_j × (f̃_j* - f̃_ij) / (f̃_j* - f̃_j⁻)]
    4. Calculate fuzzy Q (compromise): Q̃_i = v(S̃_i - S̃*) / (S̃⁻ - S̃*) + (1-v)(R̃_i - R̃*) / (R̃⁻ - R̃*)
    5. Check conditions C1 (advantage) and C2 (stability)

Reference:
    Opricovic, S., & Tzeng, G. H. (2007). Extended VIKOR method in comparison 
    with outranking methods. European Journal of Operational Research, 178(2), 514-529.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .base import TriangularFuzzyNumber, FuzzyDecisionMatrix
from ...weighting import EnsembleWeightCalculator, WeightResult


@dataclass
class FuzzyVIKORResult:
    """
    Result container for Fuzzy VIKOR calculation.
    
    Attributes:
        S: Group utility values (defuzzified)
        R: Individual regret values (defuzzified)
        Q: Compromise index values (defuzzified)
        fuzzy_S: Fuzzy group utility values
        fuzzy_R: Fuzzy individual regret values
        fuzzy_Q: Fuzzy compromise index values
        ranks_Q: Rankings by Q (final ranking)
        compromise_solution: Best compromise alternative
        advantage_condition: Whether C1 is satisfied
        stability_condition: Whether C2 is satisfied
        compromise_set: Set of acceptable compromise solutions
        weights: Criteria weights used
        v: Strategy weight parameter
    """
    S: pd.Series                    # Group utility (defuzzified)
    R: pd.Series                    # Individual regret (defuzzified)
    Q: pd.Series                    # Compromise index (defuzzified)
    fuzzy_S: Dict[str, TriangularFuzzyNumber]  # Fuzzy group utility
    fuzzy_R: Dict[str, TriangularFuzzyNumber]  # Fuzzy individual regret
    fuzzy_Q: Dict[str, TriangularFuzzyNumber]  # Fuzzy compromise index
    ranks_Q: pd.Series              # Ranking by Q (final)
    compromise_solution: str
    advantage_condition: bool
    stability_condition: bool
    compromise_set: List[str]
    weights: Dict[str, float]
    v: float
    
    @property
    def final_ranks(self) -> pd.Series:
        """Get final ranking by Q."""
        return self.ranks_Q
    
    def top_n(self, n: int = 10) -> pd.DataFrame:
        """Get top N alternatives with VIKOR metrics."""
        return pd.DataFrame({
            'S': self.S,
            'R': self.R,
            'Q': self.Q,
            'Rank': self.ranks_Q
        }).nsmallest(n, 'Rank')


class FuzzyVIKOR:
    """
    Fuzzy VIKOR calculator using Triangular Fuzzy Numbers.
    
    Combines VIKOR compromise solution approach with fuzzy set theory
    to handle uncertainty in decision making. The parameter v controls
    the trade-off between maximum group utility and minimum regret.
    
    Parameters:
        v: Strategy weight for maximum group utility (0 to 1)
            - v = 0.5: Balanced (default)
            - v > 0.5: Emphasizes majority rule
            - v < 0.5: Emphasizes minimum regret
        defuzzification: Method for converting fuzzy to crisp values
        benefit_criteria: List of benefit criteria names
        cost_criteria: List of cost criteria names
    
    Example:
        >>> calculator = FuzzyVIKOR(v=0.5, cost_criteria=['Cost'])
        >>> result = calculator.calculate_from_panel(panel_data)
        >>> print(f"Compromise solution: {result.compromise_solution}")
    """
    
    def __init__(self,
                 v: float = 0.5,
                 defuzzification: str = "centroid",
                 benefit_criteria: Optional[List[str]] = None,
                 cost_criteria: Optional[List[str]] = None):
        if not 0 <= v <= 1:
            raise ValueError("v must be between 0 and 1")
        self.v = v
        self.defuzzification = defuzzification
        self.benefit_criteria = benefit_criteria
        self.cost_criteria = cost_criteria or []
    
    def calculate(self,
                 data: pd.DataFrame,
                 weights: Union[Dict[str, float], WeightResult, None] = None,
                 uncertainty: Optional[pd.DataFrame] = None,
                 spread_ratio: float = 0.1
                 ) -> FuzzyVIKORResult:
        """
        Calculate Fuzzy VIKOR from crisp data with uncertainty.
        
        Args:
            data: Decision matrix (alternatives × criteria)
            weights: Criteria weights
            uncertainty: Standard deviation for each cell (optional)
            spread_ratio: Default spread as ratio of value
        
        Returns:
            FuzzyVIKORResult with compromise analysis
        """
        fuzzy_matrix = FuzzyDecisionMatrix.from_crisp_with_uncertainty(
            data, uncertainty, spread_ratio
        )
        
        return self._calculate_fuzzy_vikor(fuzzy_matrix, weights, data.columns.tolist())
    
    def calculate_from_panel(self,
                            panel_data,
                            weights: Union[Dict[str, float], WeightResult, None] = None,
                            spread_factor: float = 1.0
                            ) -> FuzzyVIKORResult:
        """
        Calculate Fuzzy VIKOR using panel data temporal variance.
        
        Uses historical variance to generate fuzzy numbers.
        
        Args:
            panel_data: Panel data object
            weights: Criteria weights
            spread_factor: Multiplier for std spread
        
        Returns:
            FuzzyVIKORResult
        """
        fuzzy_matrix = FuzzyDecisionMatrix.from_panel_temporal_variance(
            panel_data, spread_factor
        )
        
        return self._calculate_fuzzy_vikor(
            fuzzy_matrix, weights, panel_data.components
        )
    
    def _calculate_fuzzy_vikor(self,
                               fuzzy_matrix: FuzzyDecisionMatrix,
                               weights: Union[Dict[str, float], WeightResult, None],
                               criteria: List[str]
                               ) -> FuzzyVIKORResult:
        """Core Fuzzy VIKOR calculation."""
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
        
        # Step 1: Determine fuzzy best (f*) and worst (f-) values
        f_best, f_worst = self._get_fuzzy_ideal_values(fuzzy_matrix, criteria)
        
        # Step 2: Calculate fuzzy S and R values
        fuzzy_S, fuzzy_R = self._calculate_fuzzy_S_R(
            fuzzy_matrix, weights, f_best, f_worst, criteria
        )
        
        # Step 3: Calculate fuzzy Q values
        fuzzy_Q = self._calculate_fuzzy_Q(fuzzy_S, fuzzy_R, alternatives)
        
        # Step 4: Defuzzify for ranking
        S = pd.Series({alt: fuzzy_S[alt].defuzzify(self.defuzzification) 
                      for alt in alternatives}, name='S')
        R = pd.Series({alt: fuzzy_R[alt].defuzzify(self.defuzzification) 
                      for alt in alternatives}, name='R')
        Q = pd.Series({alt: fuzzy_Q[alt].defuzzify(self.defuzzification) 
                      for alt in alternatives}, name='Q')
        
        # Step 5: Rank by Q
        ranks_Q = Q.rank(ascending=True).astype(int)
        
        # Step 6: Check acceptance conditions
        compromise_solution = Q.idxmin()
        advantage, stability, compromise_set = self._check_conditions(
            Q, S, R, ranks_Q, alternatives
        )
        
        return FuzzyVIKORResult(
            S=S,
            R=R,
            Q=Q,
            fuzzy_S=fuzzy_S,
            fuzzy_R=fuzzy_R,
            fuzzy_Q=fuzzy_Q,
            ranks_Q=ranks_Q,
            compromise_solution=compromise_solution,
            advantage_condition=advantage,
            stability_condition=stability,
            compromise_set=compromise_set,
            weights=weights,
            v=self.v
        )
    
    def _get_fuzzy_ideal_values(self,
                                fuzzy_matrix: FuzzyDecisionMatrix,
                                criteria: List[str]
                                ) -> Tuple[Dict, Dict]:
        """
        Determine fuzzy best (f*) and worst (f-) values for each criterion.
        
        For benefit: f* = max, f- = min
        For cost: f* = min, f- = max
        """
        f_best = {}
        f_worst = {}
        
        for crit in criteria:
            values = [fuzzy_matrix.get(alt, crit) for alt in fuzzy_matrix.alternatives]
            
            if crit in self.cost_criteria:
                # Cost: lower is better
                f_best[crit] = TriangularFuzzyNumber(
                    min(v.l for v in values),
                    min(v.m for v in values),
                    min(v.u for v in values)
                )
                f_worst[crit] = TriangularFuzzyNumber(
                    max(v.l for v in values),
                    max(v.m for v in values),
                    max(v.u for v in values)
                )
            else:
                # Benefit: higher is better
                f_best[crit] = TriangularFuzzyNumber(
                    max(v.l for v in values),
                    max(v.m for v in values),
                    max(v.u for v in values)
                )
                f_worst[crit] = TriangularFuzzyNumber(
                    min(v.l for v in values),
                    min(v.m for v in values),
                    min(v.u for v in values)
                )
        
        return f_best, f_worst
    
    def _calculate_fuzzy_S_R(self,
                             fuzzy_matrix: FuzzyDecisionMatrix,
                             weights: Dict[str, float],
                             f_best: Dict,
                             f_worst: Dict,
                             criteria: List[str]
                             ) -> Tuple[Dict, Dict]:
        """
        Calculate fuzzy S (group utility) and R (individual regret).
        
        S_i = Σ w_j × (f_j* - f_ij) / (f_j* - f_j-)
        R_i = max_j [w_j × (f_j* - f_ij) / (f_j* - f_j-)]
        """
        fuzzy_S = {}
        fuzzy_R = {}
        
        for alt in fuzzy_matrix.alternatives:
            S_terms = []
            R_terms = []
            
            for crit in criteria:
                f_ij = fuzzy_matrix.get(alt, crit)
                f_star = f_best[crit]
                f_minus = f_worst[crit]
                
                # Calculate normalized distance
                numerator = TriangularFuzzyNumber(
                    f_star.l - f_ij.u,
                    f_star.m - f_ij.m,
                    f_star.u - f_ij.l
                )
                
                denom_range = f_star.m - f_minus.m
                if abs(denom_range) < 1e-10:
                    denom_range = 1e-10
                
                normalized = TriangularFuzzyNumber(
                    numerator.l / denom_range if denom_range > 0 else 0,
                    numerator.m / denom_range if denom_range > 0 else 0,
                    numerator.u / denom_range if denom_range > 0 else 0
                )
                
                # Apply weight
                w = weights[crit]
                weighted = normalized * w
                
                S_terms.append(weighted)
                R_terms.append(weighted)
            
            # Sum for S
            S_sum = TriangularFuzzyNumber(0, 0, 0)
            for term in S_terms:
                S_sum = S_sum + term
            fuzzy_S[alt] = S_sum
            
            # Max for R
            R_max = TriangularFuzzyNumber(
                max(t.l for t in R_terms),
                max(t.m for t in R_terms),
                max(t.u for t in R_terms)
            )
            fuzzy_R[alt] = R_max
        
        return fuzzy_S, fuzzy_R
    
    def _calculate_fuzzy_Q(self,
                          fuzzy_S: Dict,
                          fuzzy_R: Dict,
                          alternatives: List[str]
                          ) -> Dict:
        """
        Calculate fuzzy compromise index Q.
        
        Q = v × (S - S*) / (S- - S*) + (1-v) × (R - R*) / (R- - R*)
        """
        # Get S* (min), S- (max), R* (min), R- (max)
        S_star = min(fuzzy_S[alt].m for alt in alternatives)
        S_minus = max(fuzzy_S[alt].m for alt in alternatives)
        R_star = min(fuzzy_R[alt].m for alt in alternatives)
        R_minus = max(fuzzy_R[alt].m for alt in alternatives)
        
        fuzzy_Q = {}
        
        for alt in alternatives:
            S = fuzzy_S[alt]
            R = fuzzy_R[alt]
            
            # Q = v * (S - S*) / (S- - S*) + (1-v) * (R - R*) / (R- - R*)
            S_range = S_minus - S_star if abs(S_minus - S_star) > 1e-10 else 1e-10
            R_range = R_minus - R_star if abs(R_minus - R_star) > 1e-10 else 1e-10
            
            S_term = TriangularFuzzyNumber(
                self.v * (S.l - S_star) / S_range,
                self.v * (S.m - S_star) / S_range,
                self.v * (S.u - S_star) / S_range
            )
            
            R_term = TriangularFuzzyNumber(
                (1 - self.v) * (R.l - R_star) / R_range,
                (1 - self.v) * (R.m - R_star) / R_range,
                (1 - self.v) * (R.u - R_star) / R_range
            )
            
            fuzzy_Q[alt] = S_term + R_term
        
        return fuzzy_Q
    
    def _check_conditions(self,
                         Q: pd.Series,
                         S: pd.Series,
                         R: pd.Series,
                         ranks_Q: pd.Series,
                         alternatives: List[str]
                         ) -> Tuple[bool, bool, List[str]]:
        """
        Check VIKOR acceptance conditions.
        
        C1 (Acceptable Advantage): Q(a'') - Q(a') ≥ DQ where DQ = 1/(m-1)
        C2 (Acceptable Stability): a' is also best in S or R ranking
        """
        sorted_alts = Q.sort_values().index.tolist()
        m = len(alternatives)
        
        if m <= 1:
            return True, True, sorted_alts[:1]
        
        # C1: Acceptable advantage
        Q_a1 = Q[sorted_alts[0]]
        Q_a2 = Q[sorted_alts[1]]
        DQ = 1 / (m - 1) if m > 1 else 0
        advantage = (Q_a2 - Q_a1) >= DQ
        
        # C2: Acceptable stability
        ranks_S = S.rank(ascending=True)
        ranks_R = R.rank(ascending=True)
        best_Q = sorted_alts[0]
        stability = (ranks_S[best_Q] == 1) or (ranks_R[best_Q] == 1)
        
        # Compromise set
        compromise_set = [sorted_alts[0]]
        if not advantage:
            for alt in sorted_alts[1:]:
                if Q[alt] - Q_a1 < DQ:
                    compromise_set.append(alt)
                else:
                    break
        if not stability:
            if sorted_alts[1] not in compromise_set:
                compromise_set.append(sorted_alts[1])
        
        return advantage, stability, compromise_set
