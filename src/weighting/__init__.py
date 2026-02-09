# -*- coding: utf-8 -*-
"""
Weighting Methods Module

Objective weight calculation methods for MCDM:
- Entropy Weights: Information theory-based weighting
- CRITIC Weights: Criteria Importance Through Inter-criteria Correlation
- PCA Weights: Principal Component Analysis-based multivariate weighting
- Ensemble Weights: Advanced combined weighting (hybrid, game theory, Bayesian)
"""

from .entropy import EntropyWeightCalculator
from .critic import CRITICWeightCalculator
from .pca import PCAWeightCalculator
from .ensemble import EnsembleWeightCalculator
from .base import WeightResult, calculate_weights

__all__ = [
    'WeightResult',
    'EntropyWeightCalculator',
    'CRITICWeightCalculator',
    'PCAWeightCalculator',
    'EnsembleWeightCalculator',
    'calculate_weights'
]
