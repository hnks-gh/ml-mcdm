# -*- coding: utf-8 -*-
"""
COPRAS (Complex Proportional Assessment)
========================================

A multi-criteria decision-making method that evaluates alternatives 
based on their relative significance and utility degree. It accounts 
for both benefit and cost criteria through a proportional dependence 
model.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from weighting import WeightResult, CRITICWeightCalculator


@dataclass
class COPRASResult:
    """
    Container for COPRAS calculation results and diagnostics.

    Attributes
    ----------
    S_plus : pd.Series
        Sum of weighted normalized benefit criteria scores.
    S_minus : pd.Series
        Sum of weighted normalized cost criteria scores.
    Q : pd.Series
        Relative significance (priority) values.
    utility_degree : pd.Series
        Final utility scores as a percentage of the best alternative.
    ranks : pd.Series
        Final preference rankings (1 = best).
    weighted_matrix : pd.DataFrame
        The weighted normalized decision matrix.
    weights : Dict[str, float]
        Criteria weights applied.
    benefit_sum : float
        Aggregate significance of all benefit criteria.
    cost_sum : float
        Aggregate significance of all cost criteria.
    """
    
    @property
    def final_ranks(self) -> pd.Series:
        """Get final rankings."""
        return self.ranks

    @property
    def final_scores(self) -> pd.Series:
        """Get final scores (utility degree)."""
        return self.utility_degree
    
    def top_n(self, n: int = 10) -> pd.DataFrame:
        """Get top n alternatives."""
        return pd.DataFrame({
            'S+': self.S_plus,
            'S-': self.S_minus,
            'Q': self.Q,
            'Utility_%': self.utility_degree,
            'Rank': self.ranks
        }).nsmallest(n, 'Rank')
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "COPRAS RESULTS",
            f"{'='*60}",
            f"\nAlternatives: {len(self.ranks)}",
            f"Benefit contribution: {self.benefit_sum:.4f}",
            f"Cost contribution: {self.cost_sum:.4f}",
            f"\nTop 10 Alternatives:"
        ]
        top10 = self.top_n(10)
        for i, (idx, row) in enumerate(top10.iterrows(), 1):
            lines.append(f"  {i}. {idx}: Q={row['Q']:.4f}, Utility={row['Utility_%']:.1f}%")
        lines.append("=" * 60)
        return "\n".join(lines)


class COPRASCalculator:
    """
    Calculator for the COPRAS outranking method.

    Operates on sum-normalized data to determine the relative utility of 
    alternatives by balancing positive (benefit) and negative (cost) 
    criterion impacts.
    """
    
    def __init__(self,
                 benefit_criteria: Optional[List[str]] = None,
                 cost_criteria: Optional[List[str]] = None):
        self.benefit_criteria = benefit_criteria
        self.cost_criteria = cost_criteria or []
    
    def calculate(self,
                  data: pd.DataFrame,
                  weights: Union[Dict[str, float], WeightResult, Optional[Any]] = None
                  ) -> COPRASResult:
        """
        Execute the COPRAS ranking algorithm.

        Parameters
        ----------
        data : pd.DataFrame
            The decision matrix with alternatives as rows.
        weights : Union[Dict[str, float], WeightResult], optional
            Weights for each criterion. If None, defaults to equal weights 
            or pre-calculated CRITIC weights.

        Returns
        -------
        COPRASResult
            Object containing utility degrees, ranks, and significance 
            metrics.
        """
        # Get weights
        if weights is None:
            weight_calc = CRITICWeightCalculator()
            weight_result = weight_calc.calculate(data)
            weights = weight_result.weights
        elif isinstance(weights, WeightResult):
            weights = weights.weights
        
        weights = {col: weights.get(col, 1/len(data.columns)) 
                  for col in data.columns}
        
        # Determine benefit and cost criteria (use local to avoid stale state)
        benefit_criteria = (self.benefit_criteria if self.benefit_criteria is not None
                           else [col for col in data.columns
                                 if col not in self.cost_criteria])
        
        # Step 1: Normalize decision matrix (sum normalization)
        norm_matrix = self._normalize(data)
        
        # Step 2: Calculate weighted normalized matrix
        weighted_matrix = norm_matrix.copy()
        for col in data.columns:
            weighted_matrix[col] = norm_matrix[col] * weights[col]
        
        # Step 3: Calculate S+ and S- for each alternative
        benefit_cols = [col for col in data.columns if col in benefit_criteria]
        cost_cols = [col for col in data.columns if col in self.cost_criteria]
        
        S_plus = weighted_matrix[benefit_cols].sum(axis=1) if benefit_cols else pd.Series(0, index=data.index)
        S_minus = weighted_matrix[cost_cols].sum(axis=1) if cost_cols else pd.Series(0, index=data.index)
        
        S_plus.name = 'S_plus'
        S_minus.name = 'S_minus'
        
        # Step 4: Calculate relative significance Q
        # Q_i = S+_i + ΣS- / (S-_i × Σ(1/S-_j))  [Zavadskas et al., 1994]
        S_minus_sum = S_minus.sum()
        
        if len(cost_cols) > 0 and S_minus_sum > 0:
            S_minus_safe = S_minus.replace(0, 1e-10)
            inv_sum = (1.0 / S_minus_safe).sum()
            Q = S_plus + S_minus_sum / (S_minus_safe * inv_sum)
        else:
            Q = S_plus
        
        Q.name = 'Q'
        
        # Step 5: Calculate utility degree (percentage)
        Q_max = Q.max()
        utility_degree = (Q / Q_max * 100) if Q_max > 0 else pd.Series(0, index=data.index)
        utility_degree.name = 'Utility_Degree'
        
        # Step 6: Rank alternatives
        ranks = Q.rank(ascending=False).astype(int)
        ranks.name = 'Rank'
        
        return COPRASResult(
            S_plus=S_plus,
            S_minus=S_minus,
            Q=Q,
            utility_degree=utility_degree,
            ranks=ranks,
            weighted_matrix=weighted_matrix,
            weights=weights,
            benefit_sum=S_plus.sum(),
            cost_sum=S_minus.sum()
        )
    
    def _normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize using sum normalization.

        Uses ``abs(col_sum)`` so that columns with negative totals
        (rare, but possible with cost criteria) do not flip signs.
        """
        result = data.copy()
        for col in data.columns:
            col_sum = data[col].sum()
            if abs(col_sum) > 0:
                result[col] = data[col] / abs(col_sum)
            else:
                result[col] = 1 / len(data)
        return result


@dataclass
class MultiPeriodCOPRASResult:
    """Result container for Multi-Period COPRAS."""
    yearly_results: Dict[int, COPRASResult]
    aggregated_Q: pd.DataFrame
    utility_evolution: pd.DataFrame
    trend_scores: pd.Series
    stability_scores: pd.Series
    final_ranking: pd.Series
    composite_scores: pd.Series


class MultiPeriodCOPRAS:
    """
    Multi-Period COPRAS for panel data analysis.
    
    Extends COPRAS to handle temporal dynamics with:
    - Yearly COPRAS analysis
    - Temporal aggregation of utility scores
    - Trend and stability analysis
    """
    
    def __init__(self,
                 temporal_discount: float = 0.9,
                 trend_weight: float = 0.3,
                 stability_weight: float = 0.2,
                 benefit_criteria: Optional[List[str]] = None,
                 cost_criteria: Optional[List[str]] = None):
        self.temporal_discount = temporal_discount
        self.trend_weight = trend_weight
        self.stability_weight = stability_weight
        self.benefit_criteria = benefit_criteria
        self.cost_criteria = cost_criteria
        self.calculator = COPRASCalculator(
            benefit_criteria=benefit_criteria,
            cost_criteria=cost_criteria
        )
    
    def calculate(self,
                 panel_data,
                 weights: Union[Dict[str, float], WeightResult, None] = None
                 ) -> MultiPeriodCOPRASResult:
        """Calculate Multi-Period COPRAS rankings."""
        yearly_results = {}
        Q_matrix = {}
        utility_matrix = {}
        
        years = sorted(panel_data.years)
        
        for year in years:
            year_data = panel_data.cross_section[year]
            numeric_cols = [col for col in year_data.columns 
                          if col in panel_data.subcriteria_names]
            year_data = year_data[numeric_cols]
            
            benefit_cols = [col for col in numeric_cols 
                           if col not in (self.cost_criteria or [])]
            self.calculator.benefit_criteria = benefit_cols
            
            result = self.calculator.calculate(year_data, weights)
            yearly_results[year] = result
            
            Q_matrix[year] = result.Q
            utility_matrix[year] = result.utility_degree
        
        Q_df = pd.DataFrame(Q_matrix)
        utility_df = pd.DataFrame(utility_matrix)
        
        trend_scores = self._calculate_trend_scores(Q_df, years)
        stability_scores = self._calculate_stability_scores(utility_df, years)
        composite_scores = self._calculate_composite_scores(
            Q_df, trend_scores, stability_scores, years
        )
        
        final_ranking = composite_scores.rank(ascending=False).astype(int)
        final_ranking = final_ranking.sort_values()
        
        return MultiPeriodCOPRASResult(
            yearly_results=yearly_results,
            aggregated_Q=Q_df,
            utility_evolution=utility_df,
            trend_scores=trend_scores,
            stability_scores=stability_scores,
            final_ranking=final_ranking,
            composite_scores=composite_scores
        )
    
    def _calculate_trend_scores(self, Q_df: pd.DataFrame, years: List[int]) -> pd.Series:
        """Calculate improvement trends."""
        trends = {}
        for entity in Q_df.index:
            values = Q_df.loc[entity, years].values
            if len(values) > 1:
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                val_range = values.max() - values.min()
                trends[entity] = slope / (val_range + 1e-10)
            else:
                trends[entity] = 0.0
        return pd.Series(trends, name='Trend')
    
    def _calculate_stability_scores(self, utility_df: pd.DataFrame, years: List[int]) -> pd.Series:
        """Calculate utility stability over time."""
        utility_std = utility_df[years].std(axis=1)
        max_std = utility_std.max() if utility_std.max() > 0 else 1
        stability = 1 - (utility_std / max_std)
        stability.name = 'Stability'
        return stability
    
    def _calculate_composite_scores(self, Q_df: pd.DataFrame, trend_scores: pd.Series,
                                    stability_scores: pd.Series, years: List[int]) -> pd.Series:
        """Calculate composite scores."""
        n_years = len(years)
        temporal_weights = np.array([self.temporal_discount ** (n_years - 1 - i) 
                                     for i in range(n_years)])
        temporal_weights /= temporal_weights.sum()
        
        weighted_Q = Q_df[years].values @ temporal_weights
        weighted_Q_series = pd.Series(weighted_Q, index=Q_df.index)
        
        def normalize(s):
            min_val, max_val = s.min(), s.max()
            if max_val - min_val > 0:
                return (s - min_val) / (max_val - min_val)
            return pd.Series(0.5, index=s.index)
        
        Q_norm = normalize(weighted_Q_series)
        trend_norm = normalize(trend_scores)
        
        base_weight = 1 - self.trend_weight - self.stability_weight
        composite = (base_weight * Q_norm + 
                    self.trend_weight * trend_norm + 
                    self.stability_weight * stability_scores)
        composite.name = 'Composite_Score'
        
        return composite
