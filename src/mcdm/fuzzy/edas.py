# -*- coding: utf-8 -*-
"""
Fuzzy EDAS Method
=================

Fuzzy Evaluation based on Distance from Average Solution.

Extends EDAS average-based approach with Triangular Fuzzy Numbers
to handle uncertainty, providing robust rankings against outliers.

Mathematical Foundation:
    1. Calculate fuzzy Average Solution (AV): ÃV_j = (1/n) × Σ x̃_ij
    2. Calculate Positive Distance from Average (P̃DA):
       - Benefit: P̃DA_ij = max(0, x̃_ij - ÃV_j) / ÃV_j
       - Cost: P̃DA_ij = max(0, ÃV_j - x̃_ij) / ÃV_j
    3. Calculate Negative Distance from Average (ÑDA):
       - Benefit: ÑDA_ij = max(0, ÃV_j - x̃_ij) / ÃV_j
       - Cost: ÑDA_ij = max(0, x̃_ij - ÃV_j) / ÃV_j
    4. Weighted sums: S̃P_i = Σ w̃_j × P̃DA_ij, S̃N_i = Σ w̃_j × ÑDA_ij
    5. Normalize: NSP_i = SP_i / max(SP), NSN_i = 1 - SN_i / max(SN)
    6. Appraisal Score: AS_i = (NSP_i + NSN_i) / 2

Reference:
    Keshavarz Ghorabaee, M., et al. (2015). Multi-Criteria Inventory 
    Classification Using a New Method of Evaluation Based on Distance 
    from Average Solution (EDAS). Informatica, 26(3), 435-451.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .base import TriangularFuzzyNumber, FuzzyDecisionMatrix
from ...weighting import EnsembleWeightCalculator, WeightResult


@dataclass
class FuzzyEDASResult:
    """
    Result container for Fuzzy EDAS calculation.
    
    Attributes:
        PDA: Positive Distance from Average (defuzzified)
        NDA: Negative Distance from Average (defuzzified)
        SP: Weighted sum of PDA
        SN: Weighted sum of NDA
        NSP: Normalized SP
        NSN: Normalized SN (inverted)
        AS: Appraisal Score
        fuzzy_AS: Fuzzy appraisal scores
        ranks: Final rankings
        average_solution: Fuzzy average values per criterion
        weights: Criteria weights used
    """
    PDA: pd.DataFrame              # Positive Distance from Average (defuzzified)
    NDA: pd.DataFrame              # Negative Distance from Average (defuzzified)
    SP: pd.Series                  # Weighted sum of PDA
    SN: pd.Series                  # Weighted sum of NDA
    NSP: pd.Series                 # Normalized SP
    NSN: pd.Series                 # Normalized SN
    AS: pd.Series                  # Appraisal Score
    fuzzy_AS: Dict[str, TriangularFuzzyNumber]  # Fuzzy appraisal scores
    ranks: pd.Series
    average_solution: Dict[str, TriangularFuzzyNumber]  # Fuzzy average
    weights: Dict[str, float]
    
    @property
    def final_ranks(self) -> pd.Series:
        """Get final ranking series."""
        return self.ranks
    
    def top_n(self, n: int = 10) -> pd.DataFrame:
        """Get top N alternatives with EDAS metrics."""
        return pd.DataFrame({
            'SP': self.SP,
            'SN': self.SN,
            'NSP': self.NSP,
            'NSN': self.NSN,
            'AS': self.AS,
            'Rank': self.ranks
        }).nsmallest(n, 'Rank')


class FuzzyEDAS:
    """
    Fuzzy EDAS calculator using Triangular Fuzzy Numbers.
    
    Extends EDAS average-based method with fuzzy set theory to handle
    uncertainty. EDAS is particularly robust against outliers since it
    uses the average as reference point.
    
    Parameters:
        defuzzification: Method for converting fuzzy to crisp
        benefit_criteria: List of benefit criteria names
        cost_criteria: List of cost criteria names
    
    Example:
        >>> calculator = FuzzyEDAS(cost_criteria=['Cost'])
        >>> result = calculator.calculate_from_panel(panel_data)
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
                 ) -> FuzzyEDASResult:
        """
        Calculate Fuzzy EDAS from crisp data with uncertainty.
        
        Args:
            data: Decision matrix
            weights: Criteria weights
            uncertainty: Uncertainty matrix (optional)
            spread_ratio: Default spread ratio
        
        Returns:
            FuzzyEDASResult with appraisal scores
        """
        fuzzy_matrix = FuzzyDecisionMatrix.from_crisp_with_uncertainty(
            data, uncertainty, spread_ratio
        )
        
        return self._calculate_fuzzy_edas(fuzzy_matrix, weights, data.columns.tolist())
    
    def calculate_from_panel(self,
                            panel_data,
                            weights: Union[Dict[str, float], WeightResult, None] = None,
                            spread_factor: float = 1.0
                            ) -> FuzzyEDASResult:
        """
        Calculate Fuzzy EDAS using panel data temporal variance.
        
        Args:
            panel_data: Panel data object
            weights: Criteria weights
            spread_factor: Multiplier for std spread
        
        Returns:
            FuzzyEDASResult
        """
        fuzzy_matrix = FuzzyDecisionMatrix.from_panel_temporal_variance(
            panel_data, spread_factor
        )
        
        return self._calculate_fuzzy_edas(
            fuzzy_matrix, weights, panel_data.components
        )
    
    def _calculate_fuzzy_edas(self,
                              fuzzy_matrix: FuzzyDecisionMatrix,
                              weights: Union[Dict[str, float], WeightResult, None],
                              criteria: List[str]
                              ) -> FuzzyEDASResult:
        """Core Fuzzy EDAS calculation."""
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
        
        # Determine benefit and cost criteria
        if self.benefit_criteria is None:
            self.benefit_criteria = [c for c in criteria if c not in self.cost_criteria]
        
        # Step 1: Calculate Fuzzy Average Solution
        fuzzy_average = self._calculate_fuzzy_average(fuzzy_matrix, criteria)
        
        # Step 2: Calculate Fuzzy PDA and NDA
        fuzzy_PDA, fuzzy_NDA = self._calculate_fuzzy_distances(
            fuzzy_matrix, fuzzy_average, criteria
        )
        
        # Step 3: Calculate weighted sums SP and SN
        fuzzy_SP = {}
        fuzzy_SN = {}
        
        for alt in alternatives:
            sp = TriangularFuzzyNumber(0, 0, 0)
            sn = TriangularFuzzyNumber(0, 0, 0)
            
            for crit in criteria:
                w = weights[crit]
                sp = sp + (fuzzy_PDA[alt][crit] * w)
                sn = sn + (fuzzy_NDA[alt][crit] * w)
            
            fuzzy_SP[alt] = sp
            fuzzy_SN[alt] = sn
        
        # Step 4: Defuzzify SP and SN
        SP = pd.Series({alt: fuzzy_SP[alt].defuzzify(self.defuzzification) 
                       for alt in alternatives}, name='SP')
        SN = pd.Series({alt: fuzzy_SN[alt].defuzzify(self.defuzzification) 
                       for alt in alternatives}, name='SN')
        
        # Step 5: Normalize SP and SN
        SP_max = SP.max()
        SN_max = SN.max()
        
        NSP = SP / SP_max if SP_max > 0 else pd.Series(0, index=alternatives)
        NSN = 1 - (SN / SN_max) if SN_max > 0 else pd.Series(1, index=alternatives)
        
        # Step 6: Calculate Appraisal Score
        AS = (NSP + NSN) / 2
        
        # Create fuzzy AS
        fuzzy_AS = {}
        for alt in alternatives:
            nsp = NSP[alt]
            nsn = NSN[alt]
            as_val = AS[alt]
            
            # Spread based on difference between SP and SN
            spread = abs(nsp - nsn) / 4
            fuzzy_AS[alt] = TriangularFuzzyNumber(
                as_val - spread,
                as_val,
                as_val + spread
            )
        
        # Step 7: Rank alternatives
        ranks = AS.rank(ascending=False).astype(int)
        
        # Create crisp PDA and NDA DataFrames
        PDA_data = {}
        NDA_data = {}
        for alt in alternatives:
            PDA_data[alt] = {crit: fuzzy_PDA[alt][crit].defuzzify(self.defuzzification) 
                           for crit in criteria}
            NDA_data[alt] = {crit: fuzzy_NDA[alt][crit].defuzzify(self.defuzzification) 
                           for crit in criteria}
        
        PDA_df = pd.DataFrame(PDA_data).T
        NDA_df = pd.DataFrame(NDA_data).T
        
        return FuzzyEDASResult(
            PDA=PDA_df,
            NDA=NDA_df,
            SP=SP,
            SN=SN,
            NSP=NSP,
            NSN=NSN,
            AS=AS,
            fuzzy_AS=fuzzy_AS,
            ranks=ranks,
            average_solution=fuzzy_average,
            weights=weights
        )
    
    def _calculate_fuzzy_average(self,
                                 fuzzy_matrix: FuzzyDecisionMatrix,
                                 criteria: List[str]
                                 ) -> Dict[str, TriangularFuzzyNumber]:
        """
        Calculate fuzzy average solution for each criterion.
        
        AV_j = (1/n) × Σ x_ij
        """
        fuzzy_average = {}
        n = len(fuzzy_matrix.alternatives)
        
        for crit in criteria:
            values = [fuzzy_matrix.get(alt, crit) for alt in fuzzy_matrix.alternatives]
            
            avg_l = sum(v.l for v in values) / n
            avg_m = sum(v.m for v in values) / n
            avg_u = sum(v.u for v in values) / n
            
            fuzzy_average[crit] = TriangularFuzzyNumber(avg_l, avg_m, avg_u)
        
        return fuzzy_average
    
    def _calculate_fuzzy_distances(self,
                                   fuzzy_matrix: FuzzyDecisionMatrix,
                                   fuzzy_average: Dict[str, TriangularFuzzyNumber],
                                   criteria: List[str]
                                   ) -> Tuple[Dict, Dict]:
        """
        Calculate fuzzy PDA and NDA matrices.
        
        For benefit: PDA = max(0, x - AV) / AV, NDA = max(0, AV - x) / AV
        For cost: PDA = max(0, AV - x) / AV, NDA = max(0, x - AV) / AV
        """
        fuzzy_PDA = {}
        fuzzy_NDA = {}
        
        for alt in fuzzy_matrix.alternatives:
            fuzzy_PDA[alt] = {}
            fuzzy_NDA[alt] = {}
            
            for crit in criteria:
                x = fuzzy_matrix.get(alt, crit)
                av = fuzzy_average[crit]
                
                # Defuzzify for comparison
                x_crisp = x.defuzzify(self.defuzzification)
                av_crisp = av.defuzzify(self.defuzzification)
                
                if crit in self.benefit_criteria:
                    # Benefit: PDA = max(0, x - av) / av
                    if av_crisp > 0:
                        pda_val = max(0, x_crisp - av_crisp) / av_crisp
                        nda_val = max(0, av_crisp - x_crisp) / av_crisp
                    else:
                        pda_val = max(0, x_crisp - av_crisp)
                        nda_val = max(0, av_crisp - x_crisp)
                else:
                    # Cost: PDA = max(0, av - x) / av
                    if av_crisp > 0:
                        pda_val = max(0, av_crisp - x_crisp) / av_crisp
                        nda_val = max(0, x_crisp - av_crisp) / av_crisp
                    else:
                        pda_val = max(0, av_crisp - x_crisp)
                        nda_val = max(0, x_crisp - av_crisp)
                
                # Create fuzzy distance with spread
                spread_ratio = abs(x.u - x.l) / (x.m + 1e-10) / 3
                
                fuzzy_PDA[alt][crit] = TriangularFuzzyNumber(
                    max(0, pda_val * (1 - spread_ratio)),
                    pda_val,
                    pda_val * (1 + spread_ratio)
                )
                
                fuzzy_NDA[alt][crit] = TriangularFuzzyNumber(
                    max(0, nda_val * (1 - spread_ratio)),
                    nda_val,
                    nda_val * (1 + spread_ratio)
                )
        
        return fuzzy_PDA, fuzzy_NDA
