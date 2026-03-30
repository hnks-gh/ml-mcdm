# -*- coding: utf-8 -*-
"""
Weighted Model Package
======================

This package provides objective weighting methods for Multi-Criteria 
Decision Making (MCDM), focusing on the CRITIC (Criteria Importance 
Through Inter-criteria Correlation) approach.

Key Components:
---------------
- :class:`CRITICWeightingCalculator`: The primary two-level hierarchical 
  weighting engine.
- :class:`CRITICWeightCalculator`: Base implementation of the CRITIC 
  algorithm.
- :mod:`normalization`: Global min-max normalization utilities.
- :mod:`adaptive`: NaN-aware weighting wrappers.
"""

from .critic_weighting import CRITICWeightingCalculator
from .critic import CRITICWeightCalculator
from .normalization import global_min_max_normalize, GlobalNormalizer
from .base import WeightResult, calculate_weights
from .adaptive import (
    AdaptiveWeightCalculator,
    WeightCalculator,
    AdaptiveWeightResult,
    calculate_adaptive_weights,
)

__all__ = [
    # Core result types
    'WeightResult',
    'AdaptiveWeightResult',

    # Primary pipeline
    'CRITICWeightingCalculator',

    # Individual base calculator
    'CRITICWeightCalculator',

    # Adaptive weighting utilities
    'AdaptiveWeightCalculator',
    'WeightCalculator',
    'calculate_adaptive_weights',

    # Normalization utilities
    'global_min_max_normalize',
    'GlobalNormalizer',

    # Convenience dispatcher
    'calculate_weights',
]
