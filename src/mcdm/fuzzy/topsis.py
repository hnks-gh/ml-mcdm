# -*- coding: utf-8 -*-
"""
Fuzzy TOPSIS Method
===================

Fuzzy Technique for Order Preference by Similarity to Ideal Solution.

Extends classical TOPSIS with Triangular Fuzzy Numbers to handle uncertainty
in both decision matrix values and criteria weights.

Mathematical Foundation:
    1. Construct fuzzy decision matrix X̃ with TFN values
    2. Normalize: r̃_ij = x̃_ij / max(u_ij) for benefit criteria
    3. Calculate weighted normalized: ṽ_ij = w̃_j × r̃_ij
    4. Determine Fuzzy Ideal Solutions:
       - A* = {ṽ_1*, ..., ṽ_n*} where ṽ_j* = max(ṽ_ij) for benefit
       - A⁻ = {ṽ_1⁻, ..., ṽ_n⁻} where ṽ_j⁻ = min(ṽ_ij) for benefit
    5. Calculate distances: d_i* = Σ d(ṽ_ij, ṽ_j*), d_i⁻ = Σ d(ṽ_ij, ṽ_j⁻)
    6. Closeness coefficient: CC_i = d_i⁻ / (d_i* + d_i⁻)

Reference:
    Chen, C.T. (2000). Extensions of the TOPSIS for group decision-making
    under fuzzy environment. Fuzzy Sets and Systems, 114(1), 1-9.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .base import TriangularFuzzyNumber, FuzzyDecisionMatrix
from ...weighting import EnsembleWeightCalculator, WeightResult


@dataclass
class FuzzyTOPSISResult:
    """
    Result container for Fuzzy TOPSIS calculation.
    
    Attributes:
        scores: Closeness coefficient scores (defuzzified)
        ranks: Final rankings (1 = best)
        d_positive: Distance to fuzzy positive ideal
        d_negative: Distance to fuzzy negative ideal
        crisp_equivalent: Defuzzified decision matrix
        fuzzy_matrix: Original fuzzy decision matrix
        fuzzy_weights: Fuzzy criteria weights used
    """
    scores: pd.Series
    ranks: pd.Series
    d_positive: pd.Series
    d_negative: pd.Series
    crisp_equivalent: pd.DataFrame
    fuzzy_matrix: Dict
    fuzzy_weights: Dict[str, TriangularFuzzyNumber]
    
    @property
    def final_ranks(self) -> pd.Series:
        """Get final ranking series."""
        return self.ranks
    
    def top_n(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N alternatives.
        
        Args:
            n: Number of top alternatives to return
        
        Returns:
            DataFrame with scores, distances, and ranks
        """
        return pd.DataFrame({
            'Score': self.scores,
            'D+': self.d_positive,
            'D-': self.d_negative,
            'Rank': self.ranks
        }).nsmallest(n, 'Rank')


class FuzzyTOPSIS:
    """
    Fuzzy TOPSIS calculator using Triangular Fuzzy Numbers.
    
    Handles uncertainty in decision-making by representing values as
    fuzzy numbers, providing more robust rankings under uncertainty.
    
    Parameters:
        defuzzification: Method for converting fuzzy to crisp
            ('centroid', 'mom', 'bisector', 'graded_mean')
        benefit_criteria: List of benefit criteria names
        cost_criteria: List of cost criteria names (to be minimized)
    
    Example:
        >>> calculator = FuzzyTOPSIS(cost_criteria=['Cost', 'Risk'])
        >>> result = calculator.calculate_from_panel(panel_data, weights)
        >>> print(result.top_n(5))
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
                 ) -> FuzzyTOPSISResult:
        """
        Calculate Fuzzy TOPSIS from crisp data with uncertainty.
        
        Args:
            data: Decision matrix (alternatives × criteria)
            weights: Criteria weights (dict or WeightResult)
            uncertainty: Optional uncertainty matrix (std dev for each cell)
            spread_ratio: Default spread as ratio of value (if no uncertainty)
        
        Returns:
            FuzzyTOPSISResult with scores, ranks, and distances
        """
        fuzzy_matrix = FuzzyDecisionMatrix.from_crisp_with_uncertainty(
            data, uncertainty, spread_ratio
        )
        
        return self._calculate_from_fuzzy_matrix(
            fuzzy_matrix, weights, data.columns.tolist()
        )
    
    def calculate_from_panel(self,
                            panel_data,
                            weights: Union[Dict[str, float], WeightResult, None] = None,
                            spread_factor: float = 1.0
                            ) -> FuzzyTOPSISResult:
        """
        Calculate Fuzzy TOPSIS using panel data temporal variance.
        
        Uses historical variance across time periods to determine
        fuzzy number spreads, capturing temporal uncertainty.
        
        Args:
            panel_data: Panel data object with temporal data
            weights: Criteria weights
            spread_factor: Multiplier for temporal std spread
        
        Returns:
            FuzzyTOPSISResult with rankings
        """
        fuzzy_matrix = FuzzyDecisionMatrix.from_panel_temporal_variance(
            panel_data, spread_factor
        )
        
        return self._calculate_from_fuzzy_matrix(
            fuzzy_matrix, weights, panel_data.components
        )
    
    def calculate_from_crisp(self,
                            data: pd.DataFrame,
                            weights: Union[Dict[str, float], WeightResult, None] = None
                            ) -> FuzzyTOPSISResult:
        """
        Calculate Fuzzy TOPSIS with minimal fuzziness.
        
        Uses normalized data to generate fuzzy numbers based on
        data statistics (mean and std across alternatives).
        
        Args:
            data: Decision matrix
            weights: Criteria weights
        
        Returns:
            FuzzyTOPSISResult
        """
        # Normalize data to [0, 1]
        normalized = (data - data.min()) / (data.max() - data.min() + 1e-10)
        
        alternatives = data.index.tolist()
        columns = [c for c in data.columns if c not in ['Province', 'Year']]
        
        # Generate fuzzy matrix from statistics
        fuzzy_matrix = self._generate_fuzzy_from_statistics(
            normalized, alternatives, columns
        )
        
        # Get weights
        if weights is None:
            weight_calc = EnsembleWeightCalculator()
            weight_result = weight_calc.calculate(normalized[columns])
            weights = weight_result.weights
        elif isinstance(weights, WeightResult):
            weights = weights.weights
        
        # Create fuzzy weights
        fuzzy_weights = {}
        for col in columns:
            w = weights.get(col, 1/len(columns))
            fuzzy_weights[col] = TriangularFuzzyNumber(
                w * 0.9, w, w * 1.1
            )
        
        return self._calculate_fuzzy_topsis(
            fuzzy_matrix, fuzzy_weights, alternatives
        )
    
    def _calculate_from_fuzzy_matrix(self,
                                     fuzzy_matrix: FuzzyDecisionMatrix,
                                     weights: Union[Dict[str, float], WeightResult, None],
                                     criteria: List[str]
                                     ) -> FuzzyTOPSISResult:
        """Internal method to calculate from FuzzyDecisionMatrix."""
        alternatives = fuzzy_matrix.alternatives
        
        # Get weights
        if weights is None:
            crisp_data = fuzzy_matrix.to_crisp(self.defuzzification)
            weight_calc = EnsembleWeightCalculator()
            weight_result = weight_calc.calculate(crisp_data)
            weights = weight_result.weights
        elif isinstance(weights, WeightResult):
            weights = weights.weights
        
        # Create fuzzy weights with small spread
        fuzzy_weights = {}
        for col in criteria:
            w = weights.get(col, 1/len(criteria))
            fuzzy_weights[col] = TriangularFuzzyNumber(w * 0.95, w, w * 1.05)
        
        # Convert FuzzyDecisionMatrix to dict format
        matrix_dict = {}
        for alt in alternatives:
            matrix_dict[alt] = {}
            for crit in criteria:
                matrix_dict[alt][crit] = fuzzy_matrix.get(alt, crit)
        
        return self._calculate_fuzzy_topsis(
            matrix_dict, fuzzy_weights, alternatives
        )
    
    def _generate_fuzzy_from_statistics(self,
                                        data: pd.DataFrame,
                                        alternatives: List[str],
                                        columns: List[str]) -> Dict:
        """Generate fuzzy matrix based on data statistics."""
        fuzzy_matrix = {}
        
        for province in alternatives:
            if province not in data.index:
                continue
            fuzzy_matrix[province] = {}
            
            for col in columns:
                if col not in data.columns:
                    continue
                    
                mean_val = data.loc[province, col]
                std_val = data[col].std()
                
                # Create TFN from mean ± std
                l = max(0, mean_val - std_val)
                u = min(1, mean_val + std_val)
                
                fuzzy_matrix[province][col] = TriangularFuzzyNumber(l, mean_val, u)
        
        return fuzzy_matrix
    
    def _calculate_fuzzy_topsis(self,
                               fuzzy_matrix: Dict,
                               fuzzy_weights: Dict[str, TriangularFuzzyNumber],
                               alternatives: List[str]
                               ) -> FuzzyTOPSISResult:
        """Core Fuzzy TOPSIS calculation."""
        columns = list(fuzzy_weights.keys())
        
        # Step 1: Normalize fuzzy decision matrix
        normalized = self._normalize_fuzzy(fuzzy_matrix, alternatives, columns)
        
        # Step 2: Calculate weighted normalized fuzzy matrix
        weighted = self._apply_fuzzy_weights(normalized, fuzzy_weights, 
                                            alternatives, columns)
        
        # Step 3: Determine fuzzy ideal solutions
        f_ideal, f_anti_ideal = self._get_fuzzy_ideals(weighted, alternatives, columns)
        
        # Step 4: Calculate distances
        d_positive = {}
        d_negative = {}
        
        for alt in alternatives:
            d_pos = sum(
                weighted[alt][col].distance(f_ideal[col])
                for col in columns
            )
            d_neg = sum(
                weighted[alt][col].distance(f_anti_ideal[col])
                for col in columns
            )
            d_positive[alt] = d_pos
            d_negative[alt] = d_neg
        
        # Step 5: Calculate closeness coefficient
        scores = {}
        for alt in alternatives:
            d_neg = d_negative[alt]
            d_pos = d_positive[alt]
            scores[alt] = d_neg / (d_pos + d_neg + 1e-10)
        
        scores_series = pd.Series(scores, name='Fuzzy_TOPSIS_Score')
        ranks = scores_series.rank(ascending=False).astype(int)
        ranks.name = 'Fuzzy_TOPSIS_Rank'
        
        # Create crisp equivalent for reference
        crisp_data = {}
        for alt in alternatives:
            crisp_data[alt] = {
                col: fuzzy_matrix[alt][col].defuzzify(self.defuzzification)
                for col in columns
            }
        crisp_df = pd.DataFrame(crisp_data).T
        
        # Convert crisp weights for output
        crisp_weights = {col: w.m for col, w in fuzzy_weights.items()}
        
        return FuzzyTOPSISResult(
            scores=scores_series,
            ranks=ranks,
            d_positive=pd.Series(d_positive),
            d_negative=pd.Series(d_negative),
            crisp_equivalent=crisp_df,
            fuzzy_matrix=fuzzy_matrix,
            fuzzy_weights=fuzzy_weights
        )
    
    def _normalize_fuzzy(self, fuzzy_matrix: Dict, 
                        alternatives: List[str],
                        columns: List[str]) -> Dict:
        """
        Normalize fuzzy decision matrix.
        
        For benefit criteria: r̃_ij = x̃_ij / max(u_j)
        For cost criteria: r̃_ij = min(l_j) / x̃_ij
        """
        normalized = {alt: {} for alt in alternatives}
        
        for col in columns:
            # Get max upper bound for normalization
            if col in self.cost_criteria:
                # For cost: normalize by min lower bound
                min_l = min(fuzzy_matrix[alt][col].l for alt in alternatives)
                min_l = max(min_l, 1e-10)
                
                for alt in alternatives:
                    tfn = fuzzy_matrix[alt][col]
                    normalized[alt][col] = TriangularFuzzyNumber(
                        min_l / tfn.u if tfn.u > 0 else 0,
                        min_l / tfn.m if tfn.m > 0 else 0,
                        min_l / tfn.l if tfn.l > 0 else 1
                    )
            else:
                # For benefit: normalize by max upper bound
                max_u = max(fuzzy_matrix[alt][col].u for alt in alternatives)
                max_u = max(max_u, 1e-10)
                
                for alt in alternatives:
                    tfn = fuzzy_matrix[alt][col]
                    normalized[alt][col] = TriangularFuzzyNumber(
                        tfn.l / max_u,
                        tfn.m / max_u,
                        tfn.u / max_u
                    )
        
        return normalized
    
    def _apply_fuzzy_weights(self, normalized: Dict,
                            fuzzy_weights: Dict[str, TriangularFuzzyNumber],
                            alternatives: List[str],
                            columns: List[str]) -> Dict:
        """Apply fuzzy weights to normalized matrix: ṽ_ij = w̃_j × r̃_ij"""
        weighted = {alt: {} for alt in alternatives}
        
        for alt in alternatives:
            for col in columns:
                tfn = normalized[alt][col]
                w = fuzzy_weights[col]
                
                # Fuzzy multiplication
                weighted[alt][col] = TriangularFuzzyNumber(
                    tfn.l * w.l,
                    tfn.m * w.m,
                    tfn.u * w.u
                )
        
        return weighted
    
    def _get_fuzzy_ideals(self, weighted: Dict,
                         alternatives: List[str],
                         columns: List[str]
                         ) -> Tuple[Dict, Dict]:
        """
        Determine fuzzy ideal (A*) and anti-ideal (A⁻) solutions.
        
        For benefit criteria:
            A* = max values, A⁻ = min values
        For cost criteria:
            A* = min values, A⁻ = max values
        """
        f_ideal = {}
        f_anti_ideal = {}
        
        for col in columns:
            all_tfns = [weighted[alt][col] for alt in alternatives]
            
            if col in self.cost_criteria:
                f_ideal[col] = TriangularFuzzyNumber(
                    min(t.l for t in all_tfns),
                    min(t.m for t in all_tfns),
                    min(t.u for t in all_tfns)
                )
                f_anti_ideal[col] = TriangularFuzzyNumber(
                    max(t.l for t in all_tfns),
                    max(t.m for t in all_tfns),
                    max(t.u for t in all_tfns)
                )
            else:
                f_ideal[col] = TriangularFuzzyNumber(
                    max(t.l for t in all_tfns),
                    max(t.m for t in all_tfns),
                    max(t.u for t in all_tfns)
                )
                f_anti_ideal[col] = TriangularFuzzyNumber(
                    min(t.l for t in all_tfns),
                    min(t.m for t in all_tfns),
                    min(t.u for t in all_tfns)
                )
        
        return f_ideal, f_anti_ideal
