# -*- coding: utf-8 -*-
"""Adaptive weighting with zero handling.

Provides weight calculation that:
1. Excludes provinces with zero values from calculations
2. Excludes criteria/subcriteria with zero values from calculations
3. Recalculates weights dynamically based on available data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base import WeightResult
from .entropy import EntropyWeightCalculator
from .critic import CRITICWeightCalculator
from .merec import MERECWeightCalculator
from .standard_deviation import StandardDeviationWeightCalculator
from .fusion import GameTheoryWeightCombination


@dataclass
class AdaptiveWeightResult(WeightResult):
    """Extended weight result with adaptive calculation metadata."""
    included_alternatives: List[str]  # Provinces included in calculation
    excluded_alternatives: List[str]  # Provinces excluded (zeros)
    included_criteria: List[str]  # Criteria included in calculation
    excluded_criteria: List[str]  # Criteria excluded (all zeros)
    n_included: int  # Number of alternatives included
    n_excluded: int  # Number of alternatives excluded


class AdaptiveWeightCalculator:
    """
    Adaptive weight calculator that handles zeros in data.
    
    Key features:
    - Automatically excludes provinces with all-zero data
    - Excludes criteria where all provinces have zero
    - Recalculates weights based on available data only
    - Preserves weight normalization (sum to 1)
    
    Parameters
    ----------
    method : str
        Base weighting method: 'entropy', 'critic', 'merec', 'std_dev', 'hybrid'
    epsilon : float
        Small constant for numerical stability
    min_alternatives : int
        Minimum number of alternatives required for weight calculation
    min_criteria : int
        Minimum number of criteria required for weight calculation
    """
    
    def __init__(
        self, 
        method: str = "hybrid",
        epsilon: float = 1e-10,
        min_alternatives: int = 2,
        min_criteria: int = 2
    ):
        self.method = method
        self.epsilon = epsilon
        self.min_alternatives = min_alternatives
        self.min_criteria = min_criteria
        
        # Initialize base calculators
        self.entropy_calc = EntropyWeightCalculator(epsilon=epsilon)
        self.critic_calc = CRITICWeightCalculator(epsilon=epsilon)
        self.merec_calc = MERECWeightCalculator(epsilon=epsilon)
        self.sd_calc = StandardDeviationWeightCalculator(epsilon=epsilon)
    
    def calculate(
        self, 
        data: pd.DataFrame,
        alternative_col: str = "Province"
    ) -> AdaptiveWeightResult:
        """
        Calculate weights adaptively excluding zeros.
        
        Parameters
        ----------
        data : pd.DataFrame
            Decision matrix with alternatives and criteria
            If contains 'Province' or alternative_col, it's treated as index
        alternative_col : str
            Name of alternative identifier column (if present)
        
        Returns
        -------
        AdaptiveWeightResult
            Weights with adaptive calculation metadata
        """
        # Separate alternative column if present
        if alternative_col in data.columns:
            alternatives = data[alternative_col].tolist()
            criteria_data = data.drop(columns=[alternative_col])
        elif data.index.name == alternative_col or alternative_col == "index":
            alternatives = data.index.tolist()
            criteria_data = data
        else:
            alternatives = list(range(len(data)))
            criteria_data = data
        
        original_criteria = criteria_data.columns.tolist()
        
        # Step 1: Identify and exclude alternatives with all zeros
        row_sums = criteria_data.sum(axis=1)
        valid_rows = row_sums > 0
        
        included_alternatives = [alternatives[i] for i, v in enumerate(valid_rows) if v]
        excluded_alternatives = [alternatives[i] for i, v in enumerate(valid_rows) if not v]
        
        if sum(valid_rows) < self.min_alternatives:
            raise ValueError(
                f"Insufficient alternatives after zero exclusion: "
                f"{sum(valid_rows)} < {self.min_alternatives}"
            )
        
        # Filter data to valid rows
        filtered_data = criteria_data[valid_rows].copy()
        
        # Step 2: Identify and exclude criteria with all zeros
        col_sums = filtered_data.sum(axis=0)
        valid_cols = col_sums > 0
        
        included_criteria = [c for c, v in zip(original_criteria, valid_cols) if v]
        excluded_criteria = [c for c, v in zip(original_criteria, valid_cols) if not v]
        
        if sum(valid_cols) < self.min_criteria:
            raise ValueError(
                f"Insufficient criteria after zero exclusion: "
                f"{sum(valid_cols)} < {self.min_criteria}"
            )
        
        # Filter data to valid columns
        filtered_data = filtered_data.loc[:, valid_cols]
        
        # Step 3: Calculate weights on filtered data
        if self.method == "entropy":
            base_result = self.entropy_calc.calculate(filtered_data)
        elif self.method == "critic":
            base_result = self.critic_calc.calculate(filtered_data)
        elif self.method == "merec":
            base_result = self.merec_calc.calculate(filtered_data)
        elif self.method == "std_dev":
            base_result = self.sd_calc.calculate(filtered_data)
        elif self.method == "hybrid":
            # Use multiple methods and combine
            base_result = self._calculate_hybrid(filtered_data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Step 4: Expand weights back to include excluded criteria (with zero weight)
        full_weights = {}
        for criterion in original_criteria:
            if criterion in base_result.weights:
                full_weights[criterion] = base_result.weights[criterion]
            else:
                full_weights[criterion] = 0.0
        
        # Ensure weights sum to 1
        weight_sum = sum(full_weights.values())
        if weight_sum > self.epsilon:
            full_weights = {k: v / weight_sum for k, v in full_weights.items()}
        
        return AdaptiveWeightResult(
            weights=full_weights,
            method=f"adaptive_{self.method}",
            details={
                **base_result.details,
                "adaptive_filtering": {
                    "original_alternatives": len(alternatives),
                    "included_alternatives": len(included_alternatives),
                    "excluded_alternatives": len(excluded_alternatives),
                    "original_criteria": len(original_criteria),
                    "included_criteria": len(included_criteria),
                    "excluded_criteria": len(excluded_criteria),
                }
            },
            included_alternatives=included_alternatives,
            excluded_alternatives=excluded_alternatives,
            included_criteria=included_criteria,
            excluded_criteria=excluded_criteria,
            n_included=len(included_alternatives),
            n_excluded=len(excluded_alternatives)
        )
    
    def _calculate_hybrid(self, data: pd.DataFrame) -> WeightResult:
        """Calculate hybrid weights using multiple methods."""
        # Calculate individual weights
        w_entropy = self.entropy_calc.calculate(data)
        w_critic = self.critic_calc.calculate(data)
        w_merec = self.merec_calc.calculate(data)
        w_sd = self.sd_calc.calculate(data)
        
        # Extract weight arrays
        criteria = data.columns.tolist()
        weights_matrix = np.array([
            [w_entropy.weights[c] for c in criteria],
            [w_critic.weights[c] for c in criteria],
            [w_merec.weights[c] for c in criteria],
            [w_sd.weights[c] for c in criteria]
        ])
        
        # Simple averaging (can be enhanced with GTWC)
        # Using geometric mean for better stability
        fused_weights = np.power(np.prod(weights_matrix, axis=0), 1/4)
        
        # Normalize
        fused_weights = fused_weights / fused_weights.sum()
        
        return WeightResult(
            weights={c: w for c, w in zip(criteria, fused_weights)},
            method="hybrid",
            details={
                "entropy": w_entropy.weights,
                "critic": w_critic.weights,
                "merec": w_merec.weights,
                "std_dev": w_sd.weights
            }
        )


def calculate_adaptive_weights(
    data: pd.DataFrame,
    method: str = "hybrid",
    alternative_col: str = "Province"
) -> AdaptiveWeightResult:
    """
    Convenience function for adaptive weight calculation.
    
    Parameters
    ----------
    data : pd.DataFrame
        Decision matrix
    method : str
        Weighting method: 'entropy', 'critic', 'merec', 'std_dev', 'hybrid'
    alternative_col : str
        Name of alternative identifier column
    
    Returns
    -------
    AdaptiveWeightResult
        Calculated weights with adaptive metadata
    """
    calc = AdaptiveWeightCalculator(method=method)
    return calc.calculate(data, alternative_col=alternative_col)


class WeightCalculator:
    """
    Weight calculator for hierarchical data structure.
    
    Calculates weights at each level:
    - Subcriteria weights (for each criterion)
    - Criteria weights (for final aggregation)
    
    Both levels use adaptive zero handling.
    """
    
    def __init__(self, method: str = "hybrid", epsilon: float = 1e-10):
        self.method = method
        self.epsilon = epsilon
        self.adaptive_calc = AdaptiveWeightCalculator(method=method, epsilon=epsilon)
    
    def calculate_weights(
        self,
        subcriteria_data: pd.DataFrame,
        criteria_data: pd.DataFrame,
        hierarchy_mapping: Dict[str, List[str]]
    ) -> Dict[str, Dict]:
        """
        Calculate weights at all hierarchy levels.
        
        Parameters
        ----------
        subcriteria_data : pd.DataFrame
            Subcriteria decision matrix (provinces × subcriteria)
        criteria_data : pd.DataFrame
            Criteria decision matrix (provinces × criteria)
        hierarchy_mapping : Dict[str, List[str]]
            Mapping from criteria to their subcriteria
        
        Returns
        -------
        Dict
            {
                'criteria_weights': {...},
                'subcriteria_weights': {...},
                'subcriteria_by_criterion': {...}
            }
        """
        # Calculate criteria weights
        criteria_result = self.adaptive_calc.calculate(criteria_data)
        
        # Calculate subcriteria weights for each criterion
        subcriteria_by_criterion = {}
        
        for criterion, subcrit_list in hierarchy_mapping.items():
            # Get subcriteria data for this criterion
            available_subcrit = [sc for sc in subcrit_list if sc in subcriteria_data.columns]
            
            if not available_subcrit:
                continue
            
            sub_data = subcriteria_data[available_subcrit]
            
            # Calculate weights for these subcriteria
            try:
                sub_result = self.adaptive_calc.calculate(sub_data)
                subcriteria_by_criterion[criterion] = {
                    'weights': sub_result.weights,
                    'included': sub_result.included_criteria,
                    'excluded': sub_result.excluded_criteria
                }
            except ValueError:
                # Not enough data for this criterion
                subcriteria_by_criterion[criterion] = {
                    'weights': {sc: 1.0/len(available_subcrit) for sc in available_subcrit},
                    'included': available_subcrit,
                    'excluded': []
                }
        
        # Calculate global subcriteria weights (subcriteria weight × parent criteria weight)
        global_subcriteria_weights = {}
        
        for criterion, subcrit_info in subcriteria_by_criterion.items():
            criterion_weight = criteria_result.weights.get(criterion, 0.0)
            
            for subcrit, local_weight in subcrit_info['weights'].items():
                global_subcriteria_weights[subcrit] = local_weight * criterion_weight
        
        return {
            'criteria_weights': criteria_result.weights,
            'criteria_details': {
                'included': criteria_result.included_criteria,
                'excluded': criteria_result.excluded_criteria
            },
            'subcriteria_weights': global_subcriteria_weights,
            'subcriteria_by_criterion': subcriteria_by_criterion,
            'method': self.method
        }
