# -*- coding: utf-8 -*-
"""
TOPSIS: Technique for Order Preference by Similarity to Ideal Solution

A distance-based MCDM method that ranks alternatives based on their
geometric distance from ideal and anti-ideal solutions.

Mathematical Steps:
1. Normalize the decision matrix
2. Calculate weighted normalized matrix  
3. Determine ideal (A+) and anti-ideal (A-) solutions
4. Calculate distances to ideal and anti-ideal
5. Calculate relative closeness: C_i = D_i^- / (D_i^+ + D_i^-)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from ...weighting import WeightResult, EntropyWeightCalculator


@dataclass
class TOPSISResult:
    """Result container for TOPSIS calculation."""
    scores: pd.Series                    # Closeness coefficients
    ranks: pd.Series                     # Final rankings
    d_positive: pd.Series                # Distance to ideal
    d_negative: pd.Series                # Distance to anti-ideal
    weighted_matrix: pd.DataFrame        # Weighted normalized matrix
    ideal_solution: pd.Series            # Ideal solution values
    anti_ideal_solution: pd.Series       # Anti-ideal solution values
    weights: Dict[str, float]            # Weights used
    
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
    Standard TOPSIS calculator for cross-sectional data.
    
    TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
    ranks alternatives by measuring their Euclidean distance from ideal
    and anti-ideal solutions.
    
    Parameters
    ----------
    normalization : str
        Normalization method: 'vector', 'minmax', or 'max'
    benefit_criteria : List[str], optional
        Criteria where higher values are better (default: all)
    cost_criteria : List[str], optional
        Criteria where lower values are better
    
    Examples
    --------
    >>> import pandas as pd
    >>> from mcdm.traditional import TOPSISCalculator
    >>> 
    >>> data = pd.DataFrame({
    ...     'Quality': [0.8, 0.6, 0.9, 0.7],
    ...     'Price': [100, 150, 120, 80],  # Cost criterion
    ...     'Speed': [5, 3, 4, 6]
    ... }, index=['A', 'B', 'C', 'D'])
    >>> 
    >>> weights = {'Quality': 0.4, 'Price': 0.3, 'Speed': 0.3}
    >>> calc = TOPSISCalculator(cost_criteria=['Price'])
    >>> result = calc.calculate(data, weights)
    >>> print(result.ranks)
    
    References
    ----------
    Hwang, C.L., & Yoon, K. (1981). Multiple Attribute Decision Making: 
    Methods and Applications. Springer-Verlag.
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
                 weights: Union[Dict[str, float], WeightResult, None] = None
                 ) -> TOPSISResult:
        """
        Calculate TOPSIS scores and rankings.
        
        Parameters
        ----------
        data : pd.DataFrame
            Decision matrix (alternatives × criteria)
        weights : Dict or WeightResult
            Criteria weights (if None, uses ensemble weights)
        
        Returns
        -------
        TOPSISResult
            Complete TOPSIS results
        """
        # Get weights
        if weights is None:
            weight_calc = EntropyWeightCalculator()
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
        scores = d_neg / (d_pos + d_neg + 1e-10)
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


@dataclass
class DynamicTOPSISResult(TOPSISResult):
    """Extended result for Dynamic TOPSIS."""
    trajectory_scores: pd.Series = None
    stability_scores: pd.Series = None
    yearly_ranks: pd.DataFrame = None
    composite_score: pd.Series = None


class DynamicTOPSIS:
    """
    Dynamic TOPSIS for panel data with trajectory analysis.
    
    Combines:
    - Static TOPSIS (final year performance)
    - Trajectory score (improvement over time)
    - Stability score (consistency across years)
    
    Parameters
    ----------
    temporal_discount : float
        Discount factor for time weighting (0-1)
    trajectory_weight : float
        Weight for improvement trajectory in composite score
    stability_weight : float
        Weight for consistency in composite score
    normalization : str
        Normalization method for underlying TOPSIS
    """
    
    def __init__(self,
                 temporal_discount: float = 0.9,
                 trajectory_weight: float = 0.3,
                 stability_weight: float = 0.2,
                 normalization: str = "vector"):
        self.temporal_discount = temporal_discount
        self.trajectory_weight = trajectory_weight
        self.stability_weight = stability_weight
        self.normalization = normalization
        self.static_topsis = TOPSISCalculator(normalization=normalization)
    
    def calculate(self,
                 panel_data,  # PanelData object
                 weights: Union[Dict[str, float], WeightResult, None] = None
                 ) -> DynamicTOPSISResult:
        """
        Calculate Dynamic TOPSIS scores for panel data.
        
        Parameters
        ----------
        panel_data : PanelData
            Panel data object with cross_section dict
        weights : Dict or WeightResult
            Criteria weights
        
        Returns
        -------
        DynamicTOPSISResult
            Complete dynamic TOPSIS results
        """
        years = panel_data.years
        provinces = panel_data.provinces
        
        # Calculate TOPSIS for each year
        yearly_results = {}
        yearly_scores = {}
        
        for year in years:
            data = panel_data.get_subcriteria_year(year)
            result = self.static_topsis.calculate(data, weights)
            yearly_results[year] = result
            yearly_scores[year] = result.scores
        
        # Create yearly ranks DataFrame
        yearly_ranks = pd.DataFrame({
            year: yearly_results[year].ranks
            for year in years
        })
        
        yearly_scores_df = pd.DataFrame(yearly_scores)
        
        # Calculate trajectory score (improvement trend)
        trajectory_scores = self._calculate_trajectory(yearly_scores_df)
        
        # Calculate stability score (consistency)
        stability_scores = self._calculate_stability(yearly_scores_df)
        
        # Calculate time-weighted average score
        time_weighted = self._time_weighted_average(yearly_scores_df, years)
        
        # Composite score
        level_weight = 1 - self.trajectory_weight - self.stability_weight
        composite = (
            level_weight * yearly_scores[years[-1]] +
            self.trajectory_weight * trajectory_scores +
            self.stability_weight * stability_scores
        )
        composite.name = 'Dynamic_TOPSIS_Score'
        
        # Final ranking
        final_ranks = composite.rank(ascending=False).astype(int)
        final_ranks.name = 'Dynamic_TOPSIS_Rank'
        
        # Get the latest year's detailed results for base info
        latest_result = yearly_results[years[-1]]
        
        return DynamicTOPSISResult(
            scores=composite,
            ranks=final_ranks,
            d_positive=latest_result.d_positive,
            d_negative=latest_result.d_negative,
            weighted_matrix=latest_result.weighted_matrix,
            ideal_solution=latest_result.ideal_solution,
            anti_ideal_solution=latest_result.anti_ideal_solution,
            weights=latest_result.weights,
            trajectory_scores=trajectory_scores,
            stability_scores=stability_scores,
            yearly_ranks=yearly_ranks,
            composite_score=composite
        )
    
    def _calculate_trajectory(self, scores_df: pd.DataFrame) -> pd.Series:
        """Calculate improvement trajectory score."""
        years = scores_df.columns.tolist()
        x = np.arange(len(years))
        
        slopes = []
        for province in scores_df.index:
            y = scores_df.loc[province].values
            slope = np.polyfit(x, y, 1)[0]
            slopes.append(slope)
        
        slopes = pd.Series(slopes, index=scores_df.index)
        
        min_slope = slopes.min()
        max_slope = slopes.max()
        if max_slope > min_slope:
            trajectory = (slopes - min_slope) / (max_slope - min_slope)
        else:
            trajectory = pd.Series(0.5, index=scores_df.index)
        
        trajectory.name = 'Trajectory_Score'
        return trajectory
    
    def _calculate_stability(self, scores_df: pd.DataFrame) -> pd.Series:
        """Calculate stability score (inverse of volatility)."""
        cv = scores_df.std(axis=1) / (scores_df.mean(axis=1) + 1e-10)
        stability = 1 / (1 + cv)
        
        min_stab = stability.min()
        max_stab = stability.max()
        if max_stab > min_stab:
            stability = (stability - min_stab) / (max_stab - min_stab)
        
        stability.name = 'Stability_Score'
        return stability
    
    def _time_weighted_average(self, scores_df: pd.DataFrame, 
                              years: List[int]) -> pd.Series:
        """Calculate time-weighted average score."""
        n_years = len(years)
        time_weights = np.array([self.temporal_discount ** (n_years - 1 - i) 
                                for i in range(n_years)])
        time_weights = time_weights / time_weights.sum()
        
        weighted_avg = (scores_df * time_weights).sum(axis=1)
        weighted_avg.name = 'Time_Weighted_Score'
        return weighted_avg


def calculate_topsis(data: pd.DataFrame, 
                    weights: Optional[Dict[str, float]] = None,
                    normalization: str = "vector") -> TOPSISResult:
    """Convenience function for static TOPSIS."""
    calc = TOPSISCalculator(normalization=normalization)
    return calc.calculate(data, weights)
