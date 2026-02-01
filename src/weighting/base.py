# -*- coding: utf-8 -*-
"""Base classes and utilities for weight calculation."""

import numpy as np
import pandas as pd
from typing import Dict
from dataclasses import dataclass


@dataclass
class WeightResult:
    """Result container for weight calculations."""
    weights: Dict[str, float]
    method: str
    details: Dict
    
    @property
    def as_array(self) -> np.ndarray:
        return np.array(list(self.weights.values()))
    
    @property
    def as_series(self) -> pd.Series:
        return pd.Series(self.weights)


def calculate_weights(data: pd.DataFrame, method: str = "ensemble") -> WeightResult:
    """
    Convenience function to calculate weights.
    
    Parameters
    ----------
    data : pd.DataFrame
        Decision matrix (alternatives × criteria)
    method : str
        Weight calculation method: 'entropy', 'critic', 'ensemble', or 'equal'
    
    Returns
    -------
    WeightResult
        Calculated weights with metadata
    """
    from .entropy import EntropyWeightCalculator
    from .critic import CRITICWeightCalculator
    from .ensemble import EnsembleWeightCalculator
    
    if method == "entropy":
        return EntropyWeightCalculator().calculate(data)
    elif method == "critic":
        return CRITICWeightCalculator().calculate(data)
    elif method == "ensemble":
        return EnsembleWeightCalculator().calculate(data)
    elif method == "equal":
        cols = data.columns.tolist()
        w = 1.0 / len(cols)
        return WeightResult(
            weights={c: w for c in cols},
            method="equal",
            details={}
        )
    else:
        raise ValueError(f"Unknown method: {method}")
