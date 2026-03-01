# -*- coding: utf-8 -*-
"""
Weighting Methods Module

Objective weight calculation methods for MCDM:

**Primary Pipeline:**
- HybridWeightingCalculator: Two-level hierarchical MC ensemble
  (Entropy + CRITIC blend, Beta-distributed) with tuning + stability check.

**Individual Base Methods (used internally by the MC ensemble):**
- EntropyWeightCalculator: Information theory-based weighting
- CRITICWeightCalculator: Contrast intensity + inter-criteria correlation

**Adaptive Weighting (NaN-aware utility):**
- AdaptiveWeightCalculator: NaN-aware weight calculation (excludes all-NaN
  rows/columns; imputes partial NaN cells with column mean) using
  Entropy, CRITIC, or a blended hybrid of the two.
- WeightCalculator: Hierarchical (two-level) weight calculation built on
  AdaptiveWeightCalculator.
- calculate_adaptive_weights: Convenience function for adaptive weights.

**Utilities:**
- global_min_max_normalize: Global min-max normalization function
- GlobalNormalizer: Stateful normalizer (fit/transform pattern)
- bayesian_bootstrap_weights: Bayesian bootstrap weight sampling
- temporal_stability_verification: Temporal stability validation
"""

from .hybrid_weighting import HybridWeightingCalculator
from .entropy import EntropyWeightCalculator
from .critic import CRITICWeightCalculator
from .normalization import global_min_max_normalize, GlobalNormalizer
from .bootstrap import bayesian_bootstrap_weights, BayesianBootstrap
from .validation import temporal_stability_verification, TemporalStabilityValidator, StabilityResult
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
    'HybridWeightingCalculator',

    # Individual base calculators
    'EntropyWeightCalculator',
    'CRITICWeightCalculator',

    # Adaptive weighting utilities
    'AdaptiveWeightCalculator',
    'WeightCalculator',
    'calculate_adaptive_weights',

    # Normalization & bootstrap utilities
    'global_min_max_normalize',
    'GlobalNormalizer',
    'bayesian_bootstrap_weights',
    'BayesianBootstrap',

    # Stability validation
    'temporal_stability_verification',
    'TemporalStabilityValidator',
    'StabilityResult',

    # Convenience dispatcher
    'calculate_weights',
]
