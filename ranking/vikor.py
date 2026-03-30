# -*- coding: utf-8 -*-
"""
VIKOR (Multi-criteria Optimization and Compromise Solution)
===========================================================

A compromise ranking method that focuses on ranking and selecting from 
a set of alternatives in the presence of conflicting criteria. It 
is based on an aggregating function representing 'closeness to the 
ideal' solution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.stats import spearmanr

from weighting import WeightResult, CRITICWeightCalculator


@dataclass
class VIKORResult:
    """
    Container for VIKOR calculation results and diagnostics.

    Attributes
    ----------
    S : pd.Series
        Group utility values (lower is better).
    R : pd.Series
        Individual regret values (lower is better).
    Q : pd.Series
        The VIKOR compromise index (lower is better).
    ranks_S : pd.Series
        Rankings based on group utility S.
    ranks_R : pd.Series
        Rankings based on individual regret R.
    ranks_Q : pd.Series
        The final compromise rankings based on Q (1 = best).
    compromise_solution : str
        The best candidate following the VIKOR decision rule.
    advantage_condition : bool
        Whether the 'Acceptable Advantage' condition (C1) is met.
    stability_condition : bool
        Whether the 'Acceptable Stability' condition (C2) is met.
    compromise_set : List[str]
        The set of alternatives belonging to the compromise solution space.
    weights : Dict[str, float]
        Criteria weights applied.
    v : float
        The 'weight of strategy' (majority of criteria) parameter.
    """
    
    @property
    def final_ranks(self) -> pd.Series:
        """Get final rankings (by Q)."""
        return self.ranks_Q

    @property
    def final_scores(self) -> pd.Series:
        """Get final scores (compromise index Q)."""
        return self.Q
    
    def top_n(self, n: int = 10) -> pd.DataFrame:
        """Get top n alternatives."""
        return pd.DataFrame({
            'S': self.S,
            'R': self.R,
            'Q': self.Q,
            'Rank': self.ranks_Q
        }).nsmallest(n, 'Rank')


class VIKORCalculator:
    """
    Calculator for the VIKOR outranking method.

    Focuses on ranking alternatives with conflicting criteria by providing 
    a solution that is closest to the ideal and furthest from the 
    anti-ideal, balancing group utility and individual regret.
    """
    
    def __init__(self, 
                 v: float = 0.5,
                 benefit_criteria: Optional[List[str]] = None,
                 cost_criteria: Optional[List[str]] = None):
        if not 0 <= v <= 1:
            raise ValueError("v must be between 0 and 1")
        self.v = v
        self.benefit_criteria = benefit_criteria
        self.cost_criteria = cost_criteria or []
    
    def calculate(self,
                  data: pd.DataFrame,
                  weights: Union[Dict[str, float], WeightResult, Optional[Any]] = None
                  ) -> VIKORResult:
        """
        Execute the VIKOR ranking algorithm.

        Parameters
        ----------
        data : pd.DataFrame
            The decision matrix with alternatives as rows.
        weights : Union[Dict[str, float], WeightResult], optional
            Weights for each criterion. If None, defaults to equal weights 
            or pre-calculated CRITIC weights.

        Returns
        -------
        VIKORResult
            Object containing final compromise index, rankings, and 
            decision-rule outcomes.
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
        
        # Step 1: Determine best (f*) and worst (f-) values
        f_best, f_worst = self._get_ideal_values(data)
        
        # Step 2: Calculate S and R values
        S, R = self._calculate_S_R(data, weights, f_best, f_worst)
        
        # Step 3: Calculate Q values
        Q = self._calculate_Q(S, R)
        
        # Step 4: Rank by Q, S, R
        ranks_Q = Q.rank(ascending=True).astype(int)
        ranks_S = S.rank(ascending=True).astype(int)
        ranks_R = R.rank(ascending=True).astype(int)
        
        # Step 5: Check acceptance conditions
        compromise_solution = Q.idxmin()
        advantage, stability, compromise_set = self._check_conditions(
            Q, S, R, ranks_Q, ranks_S, ranks_R
        )
        
        return VIKORResult(
            S=S,
            R=R,
            Q=Q,
            ranks_S=ranks_S,
            ranks_R=ranks_R,
            ranks_Q=ranks_Q,
            compromise_solution=compromise_solution,
            advantage_condition=advantage,
            stability_condition=stability,
            compromise_set=compromise_set,
            weights=weights,
            v=self.v
        )
    
    def _get_ideal_values(self, data: pd.DataFrame
                         ) -> Tuple[pd.Series, pd.Series]:
        """Determine best and worst values for each criterion."""
        f_best = pd.Series(index=data.columns, dtype=float)
        f_worst = pd.Series(index=data.columns, dtype=float)
        
        for col in data.columns:
            if col in self.cost_criteria:
                f_best[col] = data[col].min()
                f_worst[col] = data[col].max()
            else:  # Benefit criteria
                f_best[col] = data[col].max()
                f_worst[col] = data[col].min()
        
        return f_best, f_worst
    
    def _calculate_S_R(self, data: pd.DataFrame, weights: Dict[str, float],
                      f_best: pd.Series, f_worst: pd.Series
                      ) -> Tuple[pd.Series, pd.Series]:
        """Calculate S (group utility) and R (individual regret)."""
        n_alternatives = len(data)
        S = pd.Series(0.0, index=data.index)
        R = pd.Series(0.0, index=data.index)
        
        for col in data.columns:
            w = weights[col]
            best = f_best[col]
            worst = f_worst[col]
            
            # Avoid division by zero
            denominator = best - worst
            if abs(denominator) < 1e-10:
                denominator = 1e-10
            
            # Normalized regret
            regret = w * (best - data[col]) / denominator
            
            S += regret
            R = R.combine(regret, max)
        
        S.name = 'S'
        R.name = 'R'
        return S, R
    
    def _calculate_Q(self, S: pd.Series, R: pd.Series) -> pd.Series:
        """Calculate Q (compromise index)."""
        S_best, S_worst = S.min(), S.max()
        R_best, R_worst = R.min(), R.max()
        
        # Avoid division by zero
        S_range = S_worst - S_best if S_worst != S_best else 1e-10
        R_range = R_worst - R_best if R_worst != R_best else 1e-10
        
        Q = (
            self.v * (S - S_best) / S_range +
            (1 - self.v) * (R - R_best) / R_range
        )
        
        Q.name = 'Q'
        return Q
    
    def _check_conditions(self, Q: pd.Series, S: pd.Series, R: pd.Series,
                         ranks_Q: pd.Series, ranks_S: pd.Series, 
                         ranks_R: pd.Series) -> Tuple[bool, bool, List[str]]:
        """Check acceptance conditions for compromise solution."""
        n = len(Q)
        
        # Sort by Q
        sorted_alts = Q.sort_values().index.tolist()
        a1 = sorted_alts[0]  # Best by Q
        a2 = sorted_alts[1] if n > 1 else a1  # Second best
        
        # Condition 1: Acceptable advantage
        DQ = 1 / (n - 1) if n > 1 else 0
        advantage = (Q[a2] - Q[a1]) >= DQ
        
        # Condition 2: Acceptable stability
        stability = (ranks_S[a1] == 1) or (ranks_R[a1] == 1)
        
        # Determine compromise set
        if advantage and stability:
            compromise_set = [a1]
        elif not advantage:
            compromise_set = [alt for alt in sorted_alts 
                            if Q[alt] - Q[a1] < DQ]
        else:
            # C1 holds but C2 fails → compromise set is {a(1), a(2)}
            # per Opricovic & Tzeng (2004), Section 3, Condition C2.
            compromise_set = [a1, a2]
        
        return advantage, stability, compromise_set


class MultiPeriodVIKOR:
    """
    VIKOR analysis across multiple time periods.
    
    Extends VIKOR to panel data by calculating yearly rankings
    and aggregating with temporal weighting.
    """
    
    def __init__(self, v: float = 0.5, temporal_discount: float = 0.9):
        self.v = v
        self.temporal_discount = temporal_discount
        self.calculator = VIKORCalculator(v=v)
    
    def calculate(self, panel_data, weights: Optional[Dict] = None) -> Dict:
        """Calculate VIKOR for each year and aggregate."""
        years = panel_data.years
        yearly_results = {}
        
        for year in years:
            data = panel_data.get_subcriteria_year(year)
            result = self.calculator.calculate(data, weights)
            yearly_results[year] = result
        
        # Create Q scores matrix
        Q_matrix = pd.DataFrame({
            year: yearly_results[year].Q
            for year in years
        })
        
        # Time-weighted average Q
        n_years = len(years)
        time_weights = np.array([
            self.temporal_discount ** (n_years - 1 - i)
            for i in range(n_years)
        ])
        time_weights = time_weights / time_weights.sum()
        
        weighted_Q = (Q_matrix * time_weights).sum(axis=1)
        final_ranks = weighted_Q.rank(ascending=True).astype(int)
        
        return {
            'yearly_results': yearly_results,
            'Q_matrix': Q_matrix,
            'weighted_Q': weighted_Q,
            'final_ranks': final_ranks,
            'time_weights': dict(zip(years, time_weights))
        }


def compare_topsis_vikor(topsis_result, vikor_result) -> Dict:
    """Compare TOPSIS and VIKOR rankings."""
    topsis_ranks = topsis_result.ranks
    vikor_ranks = vikor_result.ranks_Q
    
    common_idx = topsis_ranks.index.intersection(vikor_ranks.index)
    t_ranks = topsis_ranks.loc[common_idx]
    v_ranks = vikor_ranks.loc[common_idx]
    
    correlation, p_value = spearmanr(t_ranks, v_ranks)
    rank_diff = (t_ranks - v_ranks).abs()
    disagreements = rank_diff[rank_diff > 5]
    
    return {
        'spearman_correlation': correlation,
        'p_value': p_value,
        'mean_rank_difference': rank_diff.mean(),
        'max_rank_difference': rank_diff.max(),
        'n_major_disagreements': len(disagreements),
        'major_disagreements': disagreements.to_dict(),
        'comparison_df': pd.DataFrame({
            'TOPSIS_Rank': t_ranks,
            'VIKOR_Rank': v_ranks,
            'Difference': rank_diff
        })
    }
