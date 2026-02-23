# -*- coding: utf-8 -*-
"""
EDAS: Evaluation based on Distance from Average Solution

A method that uses the average solution as a reference point rather than
ideal solutions. More robust to outliers and extreme values.

Mathematical Steps:
1. Calculate Average Solution (AV)
2. Calculate Positive Distance from Average (PDA)
3. Calculate Negative Distance from Average (NDA)
4. Weighted sum: SP_i = Σw_j × PDA_ij, SN_i = Σw_j × NDA_ij
5. Normalize: NSP_i, NSN_i
6. Appraisal Score: AS_i = (NSP_i + (1 - NSN_i)) / 2
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from scipy import stats

from weighting import WeightResult, EntropyWeightCalculator


@dataclass
class EDASResult:
    """Result container for EDAS calculation."""
    PDA: pd.DataFrame              # Positive Distance from Average
    NDA: pd.DataFrame              # Negative Distance from Average
    SP: pd.Series                  # Weighted sum of PDA
    SN: pd.Series                  # Weighted sum of NDA
    NSP: pd.Series                 # Normalized SP
    NSN: pd.Series                 # Normalized SN
    AS: pd.Series                  # Appraisal Score
    ranks: pd.Series               # Final rankings
    average_solution: pd.Series    # Average solution values
    weights: Dict[str, float]
    
    @property
    def final_ranks(self) -> pd.Series:
        """Get final rankings."""
        return self.ranks

    @property
    def final_scores(self) -> pd.Series:
        """Get final scores (appraisal score)."""
        return self.AS
    
    def top_n(self, n: int = 10) -> pd.DataFrame:
        """Get top n alternatives."""
        return pd.DataFrame({
            'SP': self.SP,
            'SN': self.SN,
            'NSP': self.NSP,
            'NSN': self.NSN,
            'AS': self.AS,
            'Rank': self.ranks
        }).nsmallest(n, 'Rank')
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "EDAS RESULTS",
            f"{'='*60}",
            f"\nAlternatives: {len(self.ranks)}",
            f"Criteria: {len(self.average_solution)}",
            f"\nAverage Solution:",
        ]
        for crit, val in self.average_solution.items():
            lines.append(f"  {crit}: {val:.4f}")
        lines.append(f"\nTop 10 Alternatives:")
        top10 = self.top_n(10)
        for i, (idx, row) in enumerate(top10.iterrows(), 1):
            lines.append(f"  {i}. {idx}: AS={row['AS']:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)


class EDASCalculator:
    """
    EDAS (Evaluation based on Distance from Average Solution) calculator.
    
    Unlike TOPSIS which uses ideal solutions, EDAS uses the average solution
    as a reference point. This makes it more robust to outliers and extreme values.
    
    Parameters
    ----------
    benefit_criteria : List[str], optional
        Criteria where higher values are better
    cost_criteria : List[str], optional
        Criteria where lower values are better
    
    Examples
    --------
    >>> import pandas as pd
    >>> from mcdm.traditional import EDASCalculator
    >>> 
    >>> data = pd.DataFrame({
    ...     'Quality': [0.8, 0.6, 0.9, 0.7],
    ...     'Price': [100, 150, 120, 80],  # Cost criterion
    ...     'Speed': [5, 3, 4, 6]
    ... }, index=['A', 'B', 'C', 'D'])
    >>> 
    >>> weights = {'Quality': 0.4, 'Price': 0.3, 'Speed': 0.3}
    >>> calc = EDASCalculator(cost_criteria=['Price'])
    >>> result = calc.calculate(data, weights)
    >>> print(result.AS)
    
    References
    ----------
    Ghorabaee, M.K., Zavadskas, E.K., Olfat, L., & Turskis, Z. (2015).
    Multi-criteria inventory classification using a new method of evaluation 
    based on distance from average solution (EDAS). Informatica.
    """
    
    def __init__(self,
                 benefit_criteria: Optional[List[str]] = None,
                 cost_criteria: Optional[List[str]] = None):
        self.benefit_criteria = benefit_criteria
        self.cost_criteria = cost_criteria or []
    
    def calculate(self,
                 data: pd.DataFrame,
                 weights: Union[Dict[str, float], WeightResult, None] = None
                 ) -> EDASResult:
        """
        Calculate EDAS scores and rankings.
        
        Parameters
        ----------
        data : pd.DataFrame
            Decision matrix (alternatives × criteria)
        weights : Dict or WeightResult
            Criteria weights
        
        Returns
        -------
        EDASResult
            Complete EDAS results
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
        
        # Determine benefit and cost criteria (use local to avoid stale state)
        benefit_criteria = (self.benefit_criteria if self.benefit_criteria is not None
                           else [col for col in data.columns
                                 if col not in self.cost_criteria])
        
        # Step 1: Calculate Average Solution (AV)
        average_solution = data.mean()
        
        # Step 2: Calculate PDA and NDA
        PDA = pd.DataFrame(index=data.index, columns=data.columns, dtype=float)
        NDA = pd.DataFrame(index=data.index, columns=data.columns, dtype=float)
        
        for col in data.columns:
            av = average_solution[col]
            # Always normalise by |AV| (EDAS paper: divide by |average_solution|).
            # Using a small epsilon when av is near-zero keeps all PDA/NDA values
            # on a comparable scale regardless of the sign of the average.
            abs_av = max(abs(float(av)), 1e-10)

            if col in benefit_criteria:
                PDA[col] = np.maximum(0, data[col] - av) / abs_av
                NDA[col] = np.maximum(0, av - data[col]) / abs_av
            else:
                PDA[col] = np.maximum(0, av - data[col]) / abs_av
                NDA[col] = np.maximum(0, data[col] - av) / abs_av
        
        # Step 3: Calculate weighted sum of PDA and NDA
        SP = pd.Series(0.0, index=data.index)
        SN = pd.Series(0.0, index=data.index)
        
        for col in data.columns:
            w = weights[col]
            SP += w * PDA[col]
            SN += w * NDA[col]
        
        SP.name = 'SP'
        SN.name = 'SN'
        
        # Step 4: Normalize SP and SN
        SP_max = SP.max()
        SN_max = SN.max()
        
        NSP = SP / SP_max if SP_max > 0 else pd.Series(0, index=data.index)
        NSN = 1 - (SN / SN_max) if SN_max > 0 else pd.Series(1, index=data.index)
        
        NSP.name = 'NSP'
        NSN.name = 'NSN'
        
        # Step 5: Calculate Appraisal Score (AS)
        AS = (NSP + NSN) / 2
        AS.name = 'AS'
        
        # Step 6: Rank alternatives
        ranks = AS.rank(ascending=False).astype(int)
        ranks.name = 'Rank'
        
        return EDASResult(
            PDA=PDA,
            NDA=NDA,
            SP=SP,
            SN=SN,
            NSP=NSP,
            NSN=NSN,
            AS=AS,
            ranks=ranks,
            average_solution=average_solution,
            weights=weights
        )


@dataclass
class MultiPeriodEDASResult:
    """Result container for Multi-Period EDAS."""
    yearly_results: Dict[int, EDASResult]
    aggregated_AS: pd.DataFrame
    average_evolution: pd.DataFrame
    trend_scores: pd.Series
    stability_scores: pd.Series
    final_ranking: pd.Series
    composite_scores: pd.Series
    distance_from_avg_trend: pd.DataFrame


class MultiPeriodEDAS:
    """
    Multi-Period EDAS for panel data analysis.
    
    Extends EDAS to handle temporal dynamics with:
    - Yearly EDAS analysis with evolving average solutions
    - Temporal aggregation of appraisal scores
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
        self.calculator = EDASCalculator(
            benefit_criteria=benefit_criteria,
            cost_criteria=cost_criteria
        )
    
    def calculate(self,
                 panel_data,
                 weights: Union[Dict[str, float], WeightResult, None] = None
                 ) -> MultiPeriodEDASResult:
        """Calculate Multi-Period EDAS rankings."""
        yearly_results = {}
        AS_matrix = {}
        avg_matrix = {}
        
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
            
            AS_matrix[year] = result.AS
            avg_matrix[year] = result.average_solution
        
        AS_df = pd.DataFrame(AS_matrix)
        avg_evolution = pd.DataFrame(avg_matrix)
        
        distance_trend = self._calculate_distance_trend(panel_data, avg_evolution, years)
        trend_scores = self._calculate_trend_scores(AS_df, years)
        stability_scores = self._calculate_stability_scores(AS_df, years)
        composite_scores = self._calculate_composite_scores(
            AS_df, trend_scores, stability_scores, years
        )
        
        final_ranking = composite_scores.rank(ascending=False).astype(int)
        final_ranking = final_ranking.sort_values()
        
        return MultiPeriodEDASResult(
            yearly_results=yearly_results,
            aggregated_AS=AS_df,
            average_evolution=avg_evolution,
            trend_scores=trend_scores,
            stability_scores=stability_scores,
            final_ranking=final_ranking,
            composite_scores=composite_scores,
            distance_from_avg_trend=distance_trend
        )
    
    def _calculate_distance_trend(self, panel_data, avg_evolution: pd.DataFrame,
                                   years: List[int]) -> pd.DataFrame:
        """Calculate distance from average trend."""
        distance_data = {}
        for year in years:
            year_data = panel_data.cross_section[year]
            numeric_cols = [col for col in year_data.columns 
                          if col in panel_data.subcriteria_names]
            year_data = year_data[numeric_cols]
            avg = avg_evolution[year]
            
            distances = {}
            for entity in year_data.index:
                entity_values = year_data.loc[entity, numeric_cols]
                dist = np.sqrt(((entity_values - avg[numeric_cols]) ** 2).sum())
                distances[entity] = dist
            
            distance_data[year] = pd.Series(distances)
        
        return pd.DataFrame(distance_data)
    
    def _calculate_trend_scores(self, AS_df: pd.DataFrame, years: List[int]) -> pd.Series:
        """Calculate improvement trends."""
        trends = {}
        for entity in AS_df.index:
            values = AS_df.loc[entity, years].values
            if len(values) > 1:
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                val_range = values.max() - values.min()
                trends[entity] = slope / (val_range + 1e-10)
            else:
                trends[entity] = 0.0
        return pd.Series(trends, name='Trend')
    
    def _calculate_stability_scores(self, AS_df: pd.DataFrame, years: List[int]) -> pd.Series:
        """Calculate appraisal score stability."""
        AS_std = AS_df[years].std(axis=1)
        max_std = AS_std.max() if AS_std.max() > 0 else 1
        stability = 1 - (AS_std / max_std)
        stability.name = 'Stability'
        return stability
    
    def _calculate_composite_scores(self, AS_df: pd.DataFrame, trend_scores: pd.Series,
                                    stability_scores: pd.Series, years: List[int]) -> pd.Series:
        """Calculate composite scores."""
        n_years = len(years)
        temporal_weights = np.array([self.temporal_discount ** (n_years - 1 - i) 
                                     for i in range(n_years)])
        temporal_weights /= temporal_weights.sum()
        
        weighted_AS = AS_df[years].values @ temporal_weights
        weighted_AS_series = pd.Series(weighted_AS, index=AS_df.index)
        
        def normalize(s):
            min_val, max_val = s.min(), s.max()
            if max_val - min_val > 0:
                return (s - min_val) / (max_val - min_val)
            return pd.Series(0.5, index=s.index)
        
        AS_norm = normalize(weighted_AS_series)
        trend_norm = normalize(trend_scores)
        
        base_weight = 1 - self.trend_weight - self.stability_weight
        composite = (base_weight * AS_norm + 
                    self.trend_weight * trend_norm + 
                    self.stability_weight * stability_scores)
        composite.name = 'Composite_Score'
        
        return composite


class ModifiedEDAS(EDASCalculator):
    """
    Modified EDAS with trimmed mean or weighted average as reference.
    
    More robust to outliers than standard EDAS.
    """
    
    def __init__(self,
                 benefit_criteria: Optional[List[str]] = None,
                 cost_criteria: Optional[List[str]] = None,
                 use_trimmed_mean: bool = False,
                 trim_percentage: float = 0.1):
        super().__init__(benefit_criteria, cost_criteria)
        self.use_trimmed_mean = use_trimmed_mean
        self.trim_percentage = trim_percentage
    
    def calculate(self,
                 data: pd.DataFrame,
                 weights: Union[Dict[str, float], WeightResult, None] = None
                 ) -> EDASResult:
        """Calculate Modified EDAS with robust reference solution."""
        if self.use_trimmed_mean:
            # Override average solution with trimmed mean
            original_data = data.copy()
            result = super().calculate(original_data, weights)

            # Recalculate average solution using trimmed mean
            trimmed_avg = {}
            for col in data.columns:
                trimmed_avg[col] = stats.trim_mean(data[col].values, self.trim_percentage)
            result.average_solution = pd.Series(trimmed_avg)
            return result

        return super().calculate(data, weights)
