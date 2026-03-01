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


def calculate_weights(data: pd.DataFrame, method: str = "entropy") -> WeightResult:
    """
    Convenience function to calculate weights using a single base method.

    For the full two-level MC ensemble used by the main pipeline, use
    ``HybridWeightingCalculator`` directly.

    Parameters
    ----------
    data : pd.DataFrame
        Decision matrix (alternatives × criteria).
    method : str
        Weight calculation method.  Supported values:

        ``'entropy'``
            Shannon entropy weighting — assigns higher weight to criteria
            with greater discriminating power across alternatives.
        ``'critic'``
            CRITIC weighting — rewards high variance *and* low correlation
            with other criteria (Diakoulaki et al., 1995).
        ``'equal'``
            Uniform weights, each equal to 1/p.

    Returns
    -------
    WeightResult
        Calculated weights with metadata.

    Raises
    ------
    ValueError
        If *method* is ``'hybrid'`` / ``'ensemble'`` (requires the full
        ``HybridWeightingCalculator`` pipeline), or an unknown string.
    """
    from .entropy import EntropyWeightCalculator
    from .critic import CRITICWeightCalculator

    if method == "entropy":
        return EntropyWeightCalculator().calculate(data)
    elif method == "critic":
        return CRITICWeightCalculator().calculate(data)
    elif method in ("robust_global", "ensemble", "hybrid"):
        raise ValueError(
            f"Method '{method}' requires panel data and criteria_groups. "
            "Use HybridWeightingCalculator directly."
        )
    elif method == "equal":
        cols = data.columns.tolist()
        w = 1.0 / len(cols)
        return WeightResult(
            weights={c: w for c in cols},
            method="equal",
            details={},
        )
    else:
        raise ValueError(
            f"Unknown method: '{method}'.  "
            "Supported: 'entropy', 'critic', 'equal'."
        )
