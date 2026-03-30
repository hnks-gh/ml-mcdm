# -*- coding: utf-8 -*-
"""Base classes and utilities for weight calculation."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class WeightResult:
    """
    Container for criterion weighting results and diagnostics.

    Attributes
    ----------
    weights : Dict[str, float]
        Mapping of criteria/sub-criteria codes to their calculated weights.
        Weights sum to 1.0.
    method : str
        The identification string of the weighting method used.
    details : Dict
        Supplemental diagnostics, including standard deviations and 
        correlation matrices.
    temporal_stability : Optional[Any]
        Rolling-window stability metrics, if enabled.
    sensitivity_analysis : Optional[Any]
        Perturbation-based robustness metrics, if enabled.
    """
    weights: Dict[str, float]
    method: str
    details: Dict
    temporal_stability: Optional[Any] = None
    sensitivity_analysis: Optional[Any] = None
    
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


def calculate_weights(data: pd.DataFrame, method: str = "critic") -> WeightResult:
    """
    Convenience function to calculate weights using a single base method.

    For the full two-level deterministic CRITIC pipeline used by the main
    pipeline, use ``CRITICWeightingCalculator`` directly.

    Parameters
    ----------
    data : pd.DataFrame
        Decision matrix (alternatives × criteria).
    method : str
        Weight calculation method.  Supported values:

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
        If *method* is no longer supported or unknown.
    """
    from .critic import CRITICWeightCalculator

    if method == "critic":
        return CRITICWeightCalculator().calculate(data)
    elif method in ("robust_global", "ensemble", "hybrid", "entropy"):
        raise ValueError(
            f"Method '{method}' is no longer supported.  "
            "Use CRITICWeightingCalculator directly."
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
            "Supported: 'critic', 'equal'."
        )
