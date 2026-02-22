# -*- coding: utf-8 -*-
"""
PROMETHEE: Preference Ranking Organization Method for Enrichment Evaluations

An outranking method based on pairwise preference comparisons using
preference functions to model decision maker's preferences.

PROMETHEE I: Partial ranking (allows incomparability)
PROMETHEE II: Complete ranking (net flow)

Mathematical Steps:
1. Calculate pairwise preference degrees for each criterion
2. Calculate aggregated preference indices π(a,b)
3. Compute positive flow Φ+(a) and negative flow Φ-(a)
4. Calculate net flow: Φ(a) = Φ+(a) - Φ-(a)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from ...weighting import WeightResult, EntropyWeightCalculator


class PreferenceFunction(Enum):
    """Supported preference functions for PROMETHEE."""
    USUAL = "usual"           # Type I: Simple difference
    USHAPE = "ushape"         # Type II: U-shape with indifference threshold
    VSHAPE = "vshape"         # Type III: V-shape with preference threshold
    LEVEL = "level"           # Type IV: Level with both thresholds
    VSHAPE_I = "vshape_i"     # Type V: V-shape with indifference
    GAUSSIAN = "gaussian"     # Type VI: Gaussian distribution


@dataclass
class PROMETHEEResult:
    """Result container for PROMETHEE calculation."""
    phi_positive: pd.Series       # Positive outranking flow (leaving flow)
    phi_negative: pd.Series       # Negative outranking flow (entering flow)
    phi_net: pd.Series            # Net flow (Phi+ - Phi-)
    ranks_promethee_i: pd.DataFrame  # Partial ranking (I dominates J)
    ranks_promethee_ii: pd.Series    # Complete ranking by net flow
    preference_matrix: pd.DataFrame  # Aggregated preference matrix
    partial_preorder: Dict[str, List[str]]  # PROMETHEE I partial preorder
    weights: Dict[str, float]
    preference_functions: Dict[str, str]
    
    @property
    def final_ranks(self) -> pd.Series:
        """Get final rankings (PROMETHEE II)."""
        return self.ranks_promethee_ii

    @property
    def final_scores(self) -> pd.Series:
        """Get final scores (net flow Phi)."""
        return self.phi_net
    
    def top_n(self, n: int = 10) -> pd.DataFrame:
        """Get top n alternatives."""
        return pd.DataFrame({
            'Phi+': self.phi_positive,
            'Phi-': self.phi_negative,
            'Phi_net': self.phi_net,
            'Rank': self.ranks_promethee_ii
        }).nsmallest(n, 'Rank')
    
    def get_outranking_relations(self) -> pd.DataFrame:
        """Get pairwise outranking relations for PROMETHEE I."""
        alternatives = self.phi_positive.index
        relations = []
        
        for a in alternatives:
            for b in alternatives:
                if a != b:
                    phi_plus_a = self.phi_positive[a]
                    phi_plus_b = self.phi_positive[b]
                    phi_minus_a = self.phi_negative[a]
                    phi_minus_b = self.phi_negative[b]
                    
                    if (phi_plus_a >= phi_plus_b and phi_minus_a <= phi_minus_b and
                        (phi_plus_a > phi_plus_b or phi_minus_a < phi_minus_b)):
                        relations.append({'From': a, 'To': b, 'Relation': 'outranks'})
                    elif phi_plus_a == phi_plus_b and phi_minus_a == phi_minus_b:
                        relations.append({'From': a, 'To': b, 'Relation': 'indifferent'})
                    else:
                        relations.append({'From': a, 'To': b, 'Relation': 'incomparable'})
        
        return pd.DataFrame(relations)


class PROMETHEECalculator:
    """
    PROMETHEE method calculator with multiple preference functions.
    
    Implements both PROMETHEE I (partial ranking) and PROMETHEE II (complete ranking).
    
    Parameters
    ----------
    preference_function : str
        Default preference function for all criteria
    preference_threshold : float
        Preference threshold (p) - strict preference
    indifference_threshold : float
        Indifference threshold (q) - no preference
    sigma : float
        Standard deviation for Gaussian preference function
    benefit_criteria : List[str], optional
        Criteria where higher values are better
    cost_criteria : List[str], optional
        Criteria where lower values are better
    
    Examples
    --------
    >>> import pandas as pd
    >>> from mcdm.traditional import PROMETHEECalculator
    >>> 
    >>> data = pd.DataFrame({
    ...     'Quality': [0.8, 0.6, 0.9, 0.7],
    ...     'Price': [100, 150, 120, 80],
    ...     'Speed': [5, 3, 4, 6]
    ... }, index=['A', 'B', 'C', 'D'])
    >>> 
    >>> weights = {'Quality': 0.4, 'Price': 0.3, 'Speed': 0.3}
    >>> calc = PROMETHEECalculator(preference_function='vshape')
    >>> result = calc.calculate(data, weights)
    >>> print(result.phi_net)
    
    References
    ----------
    Brans, J.P., & Vincke, P. (1985). A preference ranking organisation method.
    Management Science.
    """
    
    def __init__(self,
                 preference_function: str = "vshape",
                 preference_threshold: float = 0.3,
                 indifference_threshold: float = 0.1,
                 sigma: float = 0.2,
                 benefit_criteria: Optional[List[str]] = None,
                 cost_criteria: Optional[List[str]] = None):
        self.default_pref_func = preference_function
        self.p = preference_threshold
        self.q = indifference_threshold
        self.sigma = sigma
        self.benefit_criteria = benefit_criteria
        self.cost_criteria = cost_criteria or []
        
        self.criterion_pref_funcs: Dict[str, str] = {}
        self.criterion_thresholds: Dict[str, Tuple[float, float]] = {}
    
    def set_criterion_preference(self, 
                                  criterion: str,
                                  pref_func: str,
                                  p: Optional[float] = None,
                                  q: Optional[float] = None) -> None:
        """Set preference function and thresholds for specific criterion."""
        self.criterion_pref_funcs[criterion] = pref_func
        if p is not None or q is not None:
            self.criterion_thresholds[criterion] = (
                q if q is not None else self.q,
                p if p is not None else self.p
            )
    
    def calculate(self,
                 data: pd.DataFrame,
                 weights: Union[Dict[str, float], WeightResult, None] = None
                 ) -> PROMETHEEResult:
        """
        Calculate PROMETHEE I and II rankings.
        
        Parameters
        ----------
        data : pd.DataFrame
            Decision matrix (alternatives × criteria)
        weights : Dict or WeightResult
            Criteria weights
        
        Returns
        -------
        PROMETHEEResult
            Complete PROMETHEE results with both partial and complete rankings
        """
        # Get weights
        if weights is None:
            weight_calc = EntropyWeightCalculator()
            weight_result = weight_calc.calculate(data)
            weights = weight_result.weights
        elif isinstance(weights, WeightResult):
            weights = weights.weights
        
        weights = {col: weights.get(col, 1/len(data.columns)) 
                  for col in data.columns}
        
        alternatives = data.index.tolist()
        n = len(alternatives)
        
        # Step 1: Normalize data
        data_norm = self._normalize_data(data)
        
        # Step 2: Calculate pairwise preferences for each criterion
        criterion_preferences = {}
        for col in data.columns:
            criterion_preferences[col] = self._calculate_criterion_preference(
                data_norm[col], col
            )
        
        # Step 3: Calculate aggregated preference matrix
        preference_matrix = np.zeros((n, n))
        for col in data.columns:
            w = weights[col]
            preference_matrix += w * criterion_preferences[col]
        
        pref_df = pd.DataFrame(preference_matrix, 
                               index=alternatives, columns=alternatives)
        
        # Step 4: Calculate outranking flows
        phi_positive = pd.Series(
            preference_matrix.sum(axis=1) / (n - 1) if n > 1 else preference_matrix.sum(axis=1),
            index=alternatives, name='Phi+'
        )
        phi_negative = pd.Series(
            preference_matrix.sum(axis=0) / (n - 1) if n > 1 else preference_matrix.sum(axis=0),
            index=alternatives, name='Phi-'
        )
        phi_net = phi_positive - phi_negative
        phi_net.name = 'Phi_net'
        
        # Step 5: Generate rankings
        ranks_ii = phi_net.rank(ascending=False).astype(int)
        ranks_ii.name = 'Rank_PROMETHEE_II'
        
        # Step 6: PROMETHEE I partial preorder
        partial_preorder = self._calculate_partial_preorder(phi_positive, phi_negative)
        ranks_i_df = pd.DataFrame({
            'Phi+': phi_positive,
            'Phi-': phi_negative,
            'Phi+_Rank': phi_positive.rank(ascending=False).astype(int),
            'Phi-_Rank': phi_negative.rank(ascending=True).astype(int)
        })
        
        pref_funcs_used = {col: self.criterion_pref_funcs.get(col, self.default_pref_func)
                         for col in data.columns}
        
        return PROMETHEEResult(
            phi_positive=phi_positive,
            phi_negative=phi_negative,
            phi_net=phi_net,
            ranks_promethee_i=ranks_i_df,
            ranks_promethee_ii=ranks_ii,
            preference_matrix=pref_df,
            partial_preorder=partial_preorder,
            weights=weights,
            preference_functions=pref_funcs_used
        )
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data to [0, 1] range."""
        result = data.copy()
        for col in data.columns:
            col_min = data[col].min()
            col_max = data[col].max()
            col_range = col_max - col_min
            if col_range > 0:
                if col in self.cost_criteria:
                    result[col] = (col_max - data[col]) / col_range
                else:
                    result[col] = (data[col] - col_min) / col_range
            else:
                result[col] = 0.5
        return result
    
    def _calculate_criterion_preference(self, 
                                        values: pd.Series,
                                        criterion: str) -> np.ndarray:
        """Calculate pairwise preference matrix for a single criterion."""
        n = len(values)
        pref_matrix = np.zeros((n, n))
        
        pref_func = self.criterion_pref_funcs.get(criterion, self.default_pref_func)
        q, p = self.criterion_thresholds.get(criterion, (self.q, self.p))
        
        vals = values.values
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = vals[i] - vals[j]
                    pref_matrix[i, j] = self._preference_value(d, pref_func, q, p)
        
        return pref_matrix
    
    def _preference_value(self, d: float, pref_func: str, q: float, p: float) -> float:
        """Calculate preference value P(d) based on preference function."""
        if d <= 0:
            return 0.0
        
        if pref_func == "usual":
            return 1.0 if d > 0 else 0.0
        elif pref_func == "ushape":
            return 1.0 if d > q else 0.0
        elif pref_func == "vshape":
            if d <= 0:
                return 0.0
            elif d >= p:
                return 1.0
            else:
                return d / p
        elif pref_func == "level":
            if d <= q:
                return 0.0
            elif d > p:
                return 1.0
            else:
                return 0.5
        elif pref_func == "vshape_i":
            if d <= q:
                return 0.0
            elif d >= p:
                return 1.0
            else:
                return (d - q) / (p - q) if p > q else 1.0
        elif pref_func == "gaussian":
            if d <= 0:
                return 0.0
            return 1.0 - np.exp(-(d ** 2) / (2 * self.sigma ** 2))
        else:
            raise ValueError(f"Unknown preference function: {pref_func}")
    
    def _calculate_partial_preorder(self,
                                    phi_plus: pd.Series,
                                    phi_minus: pd.Series) -> Dict[str, List[str]]:
        """Calculate PROMETHEE I partial preorder relations."""
        alternatives = phi_plus.index.tolist()
        preorder = {a: [] for a in alternatives}
        
        for a in alternatives:
            for b in alternatives:
                if a != b:
                    cond1 = phi_plus[a] >= phi_plus[b] and phi_minus[a] <= phi_minus[b]
                    cond2 = phi_plus[a] > phi_plus[b] or phi_minus[a] < phi_minus[b]
                    
                    if cond1 and cond2:
                        preorder[a].append(b)
        
        return preorder


@dataclass
class MultiPeriodPROMETHEEResult:
    """Result container for Multi-Period PROMETHEE."""
    yearly_results: Dict[int, PROMETHEEResult]
    aggregated_phi_net: pd.DataFrame
    trend_scores: pd.Series
    stability_scores: pd.Series
    final_ranking: pd.Series
    composite_scores: pd.Series
    flow_evolution: pd.DataFrame


class MultiPeriodPROMETHEE:
    """
    Multi-Period PROMETHEE for panel data analysis.
    
    Extends PROMETHEE to handle temporal dynamics with:
    - Yearly PROMETHEE II rankings
    - Temporal aggregation of net flows
    - Trend and stability analysis
    """
    
    def __init__(self,
                 temporal_discount: float = 0.9,
                 trend_weight: float = 0.3,
                 stability_weight: float = 0.2,
                 **promethee_kwargs):
        self.temporal_discount = temporal_discount
        self.trend_weight = trend_weight
        self.stability_weight = stability_weight
        self.promethee_kwargs = promethee_kwargs
        self.calculator = PROMETHEECalculator(**promethee_kwargs)
    
    def calculate(self,
                 panel_data,
                 weights: Union[Dict[str, float], WeightResult, None] = None
                 ) -> MultiPeriodPROMETHEEResult:
        """Calculate Multi-Period PROMETHEE rankings."""
        yearly_results = {}
        phi_net_matrix = {}
        phi_plus_matrix = {}
        phi_minus_matrix = {}
        
        years = sorted(panel_data.years)
        
        for year in years:
            year_data = panel_data.cross_section[year]
            numeric_cols = [col for col in year_data.columns 
                          if col in panel_data.subcriteria_names]
            year_data = year_data[numeric_cols]
            
            result = self.calculator.calculate(year_data, weights)
            yearly_results[year] = result
            
            phi_net_matrix[year] = result.phi_net
            phi_plus_matrix[year] = result.phi_positive
            phi_minus_matrix[year] = result.phi_negative
        
        phi_net_df = pd.DataFrame(phi_net_matrix)
        phi_plus_df = pd.DataFrame(phi_plus_matrix)
        phi_minus_df = pd.DataFrame(phi_minus_matrix)
        
        flow_evolution = pd.concat([
            phi_plus_df.mean(axis=1).rename('Avg_Phi+'),
            phi_minus_df.mean(axis=1).rename('Avg_Phi-'),
            phi_net_df.mean(axis=1).rename('Avg_Phi_net')
        ], axis=1)
        
        trend_scores = self._calculate_trend_scores(phi_net_df, years)
        stability_scores = self._calculate_stability_scores(yearly_results, years)
        composite_scores = self._calculate_composite_scores(
            phi_net_df, trend_scores, stability_scores, years
        )
        
        final_ranking = composite_scores.rank(ascending=False).astype(int)
        final_ranking = final_ranking.sort_values()
        
        return MultiPeriodPROMETHEEResult(
            yearly_results=yearly_results,
            aggregated_phi_net=phi_net_df,
            trend_scores=trend_scores,
            stability_scores=stability_scores,
            final_ranking=final_ranking,
            composite_scores=composite_scores,
            flow_evolution=flow_evolution
        )
    
    def _calculate_trend_scores(self, phi_net_df: pd.DataFrame, years: List[int]) -> pd.Series:
        """Calculate improvement trends."""
        trends = {}
        for entity in phi_net_df.index:
            values = phi_net_df.loc[entity, years].values
            if len(values) > 1:
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                val_range = values.max() - values.min()
                trends[entity] = slope / (val_range + 1e-10)
            else:
                trends[entity] = 0.0
        return pd.Series(trends, name='Trend')
    
    def _calculate_stability_scores(self, yearly_results: Dict, years: List[int]) -> pd.Series:
        """Calculate rank stability over time."""
        ranks_over_time = {year: yearly_results[year].ranks_promethee_ii for year in years}
        ranks_df = pd.DataFrame(ranks_over_time)
        rank_std = ranks_df.std(axis=1)
        max_std = rank_std.max() if rank_std.max() > 0 else 1
        stability = 1 - (rank_std / max_std)
        stability.name = 'Stability'
        return stability
    
    def _calculate_composite_scores(self, phi_net_df: pd.DataFrame, 
                                    trend_scores: pd.Series, 
                                    stability_scores: pd.Series,
                                    years: List[int]) -> pd.Series:
        """Calculate composite scores."""
        n_years = len(years)
        temporal_weights = np.array([self.temporal_discount ** (n_years - 1 - i) 
                                     for i in range(n_years)])
        temporal_weights /= temporal_weights.sum()
        
        weighted_phi = phi_net_df[years].values @ temporal_weights
        weighted_phi_series = pd.Series(weighted_phi, index=phi_net_df.index)
        
        def normalize(s):
            min_val, max_val = s.min(), s.max()
            if max_val - min_val > 0:
                return (s - min_val) / (max_val - min_val)
            return pd.Series(0.5, index=s.index)
        
        phi_norm = normalize(weighted_phi_series)
        trend_norm = normalize(trend_scores)
        
        base_weight = 1 - self.trend_weight - self.stability_weight
        composite = (base_weight * phi_norm + 
                    self.trend_weight * trend_norm + 
                    self.stability_weight * stability_scores)
        composite.name = 'Composite_Score'
        
        return composite
