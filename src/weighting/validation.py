# -*- coding: utf-8 -*-
"""Temporal stability validation for weight calculations."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StabilityResult:
    """Result container for temporal stability validation."""
    is_stable: bool
    cosine_similarity: float
    correlation: float
    split_point: int
    details: Dict
    
    @property
    def summary(self) -> str:
        status = "STABLE" if self.is_stable else "UNSTABLE"
        return (f"Temporal Stability: {status}\n"
                f"  Cosine Similarity: {self.cosine_similarity:.4f}\n"
                f"  Correlation: {self.correlation:.4f}\n"
                f"  Split Point: {self.split_point}")


class TemporalStabilityValidator:
    """
    Validates temporal stability of weights using split-half testing.
    
    Parameters
    ----------
    threshold : float, default=0.95
        Minimum cosine similarity for stability
    """
    
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
    
    def validate(
        self, 
        weights_1: np.ndarray, 
        weights_2: np.ndarray,
        split_point: Optional[int] = None
    ) -> StabilityResult:
        """
        Validate stability between two weight vectors.
        
        Parameters
        ----------
        weights_1 : np.ndarray
            First weight vector
        weights_2 : np.ndarray
            Second weight vector
        split_point : int, optional
            Time point where data was split
        
        Returns
        -------
        StabilityResult
            Validation result with stability metrics
        """
        # Ensure arrays are 1D
        w1 = np.atleast_1d(weights_1).flatten()
        w2 = np.atleast_1d(weights_2).flatten()
        
        # Calculate cosine similarity
        cos_sim = self._cosine_similarity(w1, w2)
        
        # Calculate Pearson correlation
        corr = np.corrcoef(w1, w2)[0, 1] if len(w1) > 1 else 1.0
        
        # Determine stability
        is_stable = cos_sim >= self.threshold
        
        return StabilityResult(
            is_stable=is_stable,
            cosine_similarity=cos_sim,
            correlation=corr,
            split_point=split_point or -1,
            details={
                'threshold': self.threshold,
                'weight_vector_1': w1.tolist(),
                'weight_vector_2': w2.tolist(),
                'l2_distance': np.linalg.norm(w1 - w2),
                'max_difference': np.abs(w1 - w2).max()
            }
        )
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)


def temporal_stability_verification(
    panel_df: pd.DataFrame,
    weight_calculator,
    entity_col: str = "Province",
    time_col: str = "Year",
    criteria_cols: Optional[List[str]] = None,
    threshold: float = 0.95
) -> StabilityResult:
    """
    Verify temporal stability of weights using split-half testing.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data in long format
    weight_calculator : callable
        Weight calculation function or object with calculate() method
    entity_col : str
        Name of entity column
    time_col : str
        Name of time column
    criteria_cols : List[str], optional
        Names of criteria columns
    threshold : float
        Minimum cosine similarity for stability
    
    Returns
    -------
    StabilityResult
        Validation result with stability metrics
    """
    # Auto-detect criteria columns if not provided
    if criteria_cols is None:
        criteria_cols = [c for c in panel_df.columns
                        if c not in (entity_col, time_col)
                        and pd.api.types.is_numeric_dtype(panel_df[c])]
    
    # Sort by time and split in half
    sorted_df = panel_df.sort_values(time_col).copy()
    time_periods = sorted(sorted_df[time_col].unique())
    split_idx = len(time_periods) // 2
    split_point = time_periods[split_idx]
    
    # Split data
    first_half = sorted_df[sorted_df[time_col] <= split_point]
    second_half = sorted_df[sorted_df[time_col] > split_point]
    
    # Calculate weights for each half
    if hasattr(weight_calculator, 'calculate'):
        # Object with calculate method
        result_1 = weight_calculator.calculate(
            first_half[criteria_cols + [entity_col, time_col]]
            if entity_col in first_half.columns else first_half[criteria_cols]
        )
        result_2 = weight_calculator.calculate(
            second_half[criteria_cols + [entity_col, time_col]]
            if entity_col in second_half.columns else second_half[criteria_cols]
        )
        weights_1 = np.array([result_1.weights[c] for c in criteria_cols])
        weights_2 = np.array([result_2.weights[c] for c in criteria_cols])
    else:
        # Callable function
        weights_1 = weight_calculator(first_half[criteria_cols])
        weights_2 = weight_calculator(second_half[criteria_cols])
    
    # Validate stability
    validator = TemporalStabilityValidator(threshold=threshold)
    return validator.validate(weights_1, weights_2, split_point=split_point)
