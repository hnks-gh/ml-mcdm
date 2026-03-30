# -*- coding: utf-8 -*-
"""
Adaptive Weighting Layer
========================

This module provides an adaptive weighting layer that respects the missing 
data structure of governance panels. It avoids imputation in the weighting 
phase, instead relying on complete-case analysis to preserve the statistical 
integrity of variance and correlation estimates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base import WeightResult
from .critic import CRITICWeightCalculator
from data.missing_data import prepare_decision_matrix, MatrixFilterReport


@dataclass
class AdaptiveWeightResult(WeightResult):
    """Extended weight result with adaptive calculation metadata."""
    included_alternatives: List[str] = None  # Provinces included in calculation
    excluded_alternatives: List[str] = None  # Provinces excluded (all NaN across all criteria)
    included_criteria: List[str] = None  # Criteria included in calculation
    excluded_criteria: List[str] = None  # Criteria excluded (all NaN across all provinces)
    n_included: int = 0  # Number of alternatives included
    n_excluded: int = 0  # Number of alternatives excluded


class AdaptiveWeightCalculator:
    """
    Adaptive weight calculator with automated missing data handling.

    Provides a robust interface for weight calculation that automatically 
    filters all-NaN entities and criteria while preserving partial 
    observations for complete-case statistical analysis.
    """

    def __init__(
        self,
        method: str = "critic",
        epsilon: float = 1e-10,
        min_alternatives: int = 2,
        min_criteria: int = 2
    ):
        self.method = method
        self.epsilon = epsilon
        self.min_alternatives = min_alternatives
        self.min_criteria = min_criteria

        self.critic_calc = CRITICWeightCalculator(epsilon=epsilon)
    
    def calculate(
        self,
        data: pd.DataFrame,
        entity_col: str = "Province"
    ) -> AdaptiveWeightResult:
        """
        Compute adaptive weights for the provided decision matrix.

        Parameters
        ----------
        data : pd.DataFrame
            The raw data matrix containing missing values.
        entity_col : str
            The column name identifying individual provinces or entities.

        Returns
        -------
        AdaptiveWeightResult
            The calculated weight results with inclusion/exclusion metadata.
        """
        original_criteria = [
            c for c in data.columns if c != entity_col
        ] if entity_col in data.columns else list(data.columns)

        # Delegate all-NaN row/column filtering to the centralised utility.
        # Partial NaN cells are preserved (no imputation); CRITICWeightCalculator
        # handles them via its own complete-case exclusion (F-03 guard).
        filtered_data, report = prepare_decision_matrix(
            data,
            entity_col=entity_col,
            min_rows=self.min_alternatives,
            min_cols=self.min_criteria,
        )

        # Calculate weights on the filtered sub-matrix (partial NaN handled downstream)
        if self.method == "critic":
            base_result = self.critic_calc.calculate(filtered_data)
        else:
            raise ValueError(
                f"Unknown method: '{self.method}'.  "
                "Supported: 'critic'."
            )

        # Expand weights back to include excluded criteria (weight = 0)
        full_weights = {
            criterion: base_result.weights.get(criterion, 0.0)
            for criterion in original_criteria
        }

        # Ensure weights sum to 1
        weight_sum = sum(full_weights.values())
        if weight_sum > self.epsilon:
            full_weights = {k: v / weight_sum for k, v in full_weights.items()}

        return AdaptiveWeightResult(
            weights=full_weights,
            method=f"adaptive_{self.method}",
            details={
                **base_result.details,
                "adaptive_filtering": report.to_dict(),
            },
            included_alternatives=report.included_rows,
            excluded_alternatives=report.excluded_rows,
            included_criteria=report.included_columns,
            excluded_criteria=report.excluded_columns,
            n_included=report.n_included_rows,
            n_excluded=report.n_excluded_rows,
        )
    

def calculate_adaptive_weights(
    data: pd.DataFrame,
    method: str = "critic",
    entity_col: str = "Province"
) -> AdaptiveWeightResult:
    """
    Convenience function for adaptive weight calculation.

    Parameters
    ----------
    data : pd.DataFrame
        Decision matrix.
    method : str
        Weighting method: ``'critic'``.
    entity_col : str
        Name of the entity identifier column (used only when *data* contains
        a row-label column instead of using the DataFrame index).

    Returns
    -------
    AdaptiveWeightResult
        Calculated weights with adaptive NaN-filtering metadata.
    """
    calc = AdaptiveWeightCalculator(method=method)
    return calc.calculate(data, entity_col=entity_col)


class WeightCalculator:
    """
    Weight calculator for hierarchical data structure.

    Calculates weights at each level:
    - Subcriteria weights (for each criterion)
    - Criteria weights (for final aggregation)

    Both levels use adaptive NaN-aware handling:
    - All-NaN rows/columns are excluded
    - Partial NaN cells are imputed with the column mean
    """
    
    def __init__(self, method: str = "critic", epsilon: float = 1e-10):
        self.method = method
        self.epsilon = epsilon
        self.adaptive_calc = AdaptiveWeightCalculator(method=method, epsilon=epsilon)
    
    def calculate_weights(
        self,
        subcriteria_data: pd.DataFrame,
        criteria_data: pd.DataFrame,
        hierarchy_mapping: Dict[str, List[str]],
        entity_col: str = "Province",
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
        entity_col : str
            Name of entity identifier column forwarded to the
            adaptive calculator (default ``"Province"``).
        
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
        criteria_result = self.adaptive_calc.calculate(criteria_data, entity_col=entity_col)
        
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
                sub_result = self.adaptive_calc.calculate(sub_data, entity_col=entity_col)
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
