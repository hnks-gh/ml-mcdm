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
        """Return weights as numpy array preserving insertion (column) order.

        Weight calculators build the dict by iterating over ``data.columns``,
        so insertion order matches column order — guaranteed in Python 3.7+.
        Callers that need a specific order should index into ``self.weights``
        directly rather than relying on this property.
        """
        return np.array(list(self.weights.values()))
    
    @property
    def as_series(self) -> pd.Series:
        return pd.Series(self.weights)


def calculate_weights(data: pd.DataFrame, method: str = "robust_global") -> WeightResult:
    """
    Convenience function to calculate weights.
    
    Parameters
    ----------
    data : pd.DataFrame
        Decision matrix (alternatives × criteria)
    method : str
        Weight calculation method: 'entropy', 'critic', 'merec', 'std_dev',
        'robust_global', or 'equal'
    
    Returns
    -------
    WeightResult
        Calculated weights with metadata
    """
    from .entropy import EntropyWeightCalculator
    from .critic import CRITICWeightCalculator
    from .merec import MERECWeightCalculator
    from .standard_deviation import StandardDeviationWeightCalculator
    
    if method == "entropy":
        return EntropyWeightCalculator().calculate(data)
    elif method == "critic":
        return CRITICWeightCalculator().calculate(data)
    elif method == "merec":
        return MERECWeightCalculator().calculate(data)
    elif method == "std_dev":
        return StandardDeviationWeightCalculator().calculate(data)
    elif method in ("robust_global", "ensemble", "hybrid"):
        from .hybrid_weighting import HybridWeightingPipeline
        calc = HybridWeightingPipeline()
        return calc.calculate(data)
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
