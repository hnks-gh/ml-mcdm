# -*- coding: utf-8 -*-
"""
Weighting Methods Module

Objective weight calculation methods for MCDM:
- Entropy Weights: Information theory-based weighting
- CRITIC Weights: Criteria Importance Through Inter-criteria Correlation
- Ensemble Weights: Combined weighting approaches
"""

from .entropy import EntropyWeightCalculator
from .critic import CRITICWeightCalculator
from .ensemble import EnsembleWeightCalculator
from .base import WeightResult, calculate_weights

__all__ = [
    'WeightResult',
    'EntropyWeightCalculator',
    'CRITICWeightCalculator',
    'EnsembleWeightCalculator',
    'calculate_weights'
]
