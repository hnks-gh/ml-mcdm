# -*- coding: utf-8 -*-
"""
TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
=======================================================================

A distance-based MCDM method that ranks alternatives based on their 
geometric distance from both the ideal (best) and anti-ideal (worst) 
solutions. Alternatives closer to the ideal and further from the 
anti-ideal receive higher scores.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from weighting import WeightResult, CRITICWeightCalculator


@dataclass
class TOPSISResult:
    """
    Container for TOPSIS calculation results and diagnostics.

    Attributes
    ----------
    scores : pd.Series
        The relative closeness coefficients (C_i) in range [0, 1].
    ranks : pd.Series
        Final preference rankings (1 = best).
    d_positive : pd.Series
        Euclidean distance to the ideal solution.
    d_negative : pd.Series
        Euclidean distance to the anti-ideal solution.
    weighted_matrix : pd.DataFrame
        The weighted normalized decision matrix.
    ideal_solution : pd.Series
        Coordinates of the ideal solution in the feature space.
    anti_ideal_solution : pd.Series
        Coordinates of the anti-ideal solution.
    weights : Dict[str, float]
        Criteria weights applied.
    """
    
    @property
    def final_ranks(self) -> pd.Series:
        """Get final rankings."""
        return self.ranks

    @property
    def final_scores(self) -> pd.Series:
        """Get final scores (closeness coefficients)."""
        return self.scores
    
    def top_n(self, n: int = 10) -> pd.DataFrame:
        """Get top n alternatives."""
        return pd.DataFrame({
            'Score': self.scores,
            'Rank': self.ranks
        }).nsmallest(n, 'Rank')
    
    def bottom_n(self, n: int = 10) -> pd.DataFrame:
        """Get bottom n alternatives."""
        return pd.DataFrame({
            'Score': self.scores,
            'Rank': self.ranks
        }).nlargest(n, 'Rank')


class TOPSISCalculator:
    """
    Calculator for the TOPSIS outranking method.

    Implements the standard Euclidean distance-based logic for cross-sectional 
    data, supporting multiple normalization strategies and benefit/cost 
    distinctions.
    """
    
    def __init__(self, 
                 normalization: str = "vector",
                 benefit_criteria: Optional[List[str]] = None,
                 cost_criteria: Optional[List[str]] = None):
        self.normalization = normalization
        self.benefit_criteria = benefit_criteria
        self.cost_criteria = cost_criteria or []
    
    def calculate(self, 
                  data: pd.DataFrame,
                  weights: Union[Dict[str, float], WeightResult, Optional[Any]] = None
                  ) -> TOPSISResult:
        """
        Execute the TOPSIS ranking algorithm.

        Parameters
        ----------
        data : pd.DataFrame
            The decision matrix with alternatives as rows.
        weights : Union[Dict[str, float], WeightResult], optional
            Weights for each criterion. If None, equal weights or 
            pre-calculated CRITIC weights are used.

        Returns
        -------
        TOPSISResult
            Object containing final scores, ranks, and distance metrics.
        """
        # Get weights
        if weights is None:
            weight_calc = CRITICWeightCalculator()
            weight_result = weight_calc.calculate(data)
            weights = weight_result.weights
        elif isinstance(weights, WeightResult):
            weights = weights.weights
        
        # Ensure weights match data columns
        weights = {col: weights.get(col, 1/len(data.columns)) 
                  for col in data.columns}
        
        # Step 1: Normalize
        norm_matrix = self._normalize(data)
        
        # Step 2: Apply weights
        weight_array = np.array([weights[col] for col in data.columns])
        weighted_matrix = norm_matrix * weight_array
        weighted_df = pd.DataFrame(weighted_matrix, 
                                  index=data.index, 
                                  columns=data.columns)
        
        # Step 3: Determine ideal solutions
        ideal, anti_ideal = self._get_ideal_solutions(weighted_df)
        
        # Step 4: Calculate distances
        d_pos = self._calculate_distance(weighted_df, ideal)
        d_neg = self._calculate_distance(weighted_df, anti_ideal)
        
        # Step 5: Calculate closeness coefficient
        # When d_pos = d_neg = 0 (alternative equals both ideal and anti-ideal),
        # assign 0.5 as a neutral score rather than 0 (the 1e-10 guard gives 0).
        denom = d_pos + d_neg
        scores = np.where(denom > 1e-14, d_neg / denom, 0.5)
        scores = pd.Series(scores, index=data.index, name='TOPSIS_Score')
        
        # Step 6: Rank
        ranks = scores.rank(ascending=False).astype(int)
        ranks.name = 'TOPSIS_Rank'
        
        return TOPSISResult(
            scores=scores,
            ranks=ranks,
            d_positive=pd.Series(d_pos, index=data.index),
            d_negative=pd.Series(d_neg, index=data.index),
            weighted_matrix=weighted_df,
            ideal_solution=ideal,
            anti_ideal_solution=anti_ideal,
            weights=weights
        )
    
    def _normalize(self, data: pd.DataFrame) -> np.ndarray:
        """Normalize decision matrix."""
        X = data.values.astype(float)
        
        if self.normalization == "vector":
            norm = np.sqrt((X ** 2).sum(axis=0))
            norm[norm == 0] = 1
            return X / norm
        
        elif self.normalization == "minmax":
            min_vals = X.min(axis=0)
            max_vals = X.max(axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1
            return (X - min_vals) / range_vals
        
        elif self.normalization == "max":
            max_vals = X.max(axis=0)
            max_vals[max_vals == 0] = 1
            return X / max_vals
        
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")
    
    def _get_ideal_solutions(self, weighted_df: pd.DataFrame
                            ) -> Tuple[pd.Series, pd.Series]:
        """Determine ideal and anti-ideal solutions."""
        ideal = pd.Series(index=weighted_df.columns, dtype=float)
        anti_ideal = pd.Series(index=weighted_df.columns, dtype=float)
        
        for col in weighted_df.columns:
            if col in self.cost_criteria:
                ideal[col] = weighted_df[col].min()
                anti_ideal[col] = weighted_df[col].max()
            else:  # Benefit criteria (default)
                ideal[col] = weighted_df[col].max()
                anti_ideal[col] = weighted_df[col].min()
        
        return ideal, anti_ideal
    
    def _calculate_distance(self, weighted_df: pd.DataFrame, 
                           reference: pd.Series) -> np.ndarray:
        """Calculate Euclidean distance to reference point."""
        diff = weighted_df - reference
        return np.sqrt((diff ** 2).sum(axis=1)).values


def calculate_topsis(data: pd.DataFrame,
                    weights: Optional[Dict[str, float]] = None,
                    normalization: str = "vector") -> TOPSISResult:
    """Convenience function for static TOPSIS."""
    calc = TOPSISCalculator(normalization=normalization)
    return calc.calculate(data, weights)
