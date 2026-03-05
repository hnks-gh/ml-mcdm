# -*- coding: utf-8 -*-
"""
Weighting Methods Module

Objective weight calculation methods for MCDM:

**Primary Pipeline:**
- CRITICWeightingCalculator: Two-level deterministic CRITIC weighting.
  Level 1 — local SC weights per criterion group (sum to 1 within group).
  Level 2 — criterion weights over composite matrix (sum to 1 globally).
  Global  — global_w[SC_j] = local_w[SC_j | C_k] × criterion_w[C_k].

**Individual Base Method:**
- CRITICWeightCalculator: Contrast intensity + inter-criteria correlation

**Adaptive Weighting (NaN-aware utility):**
- AdaptiveWeightCalculator: NaN-aware CRITIC weight calculation (excludes
  all-NaN rows/columns; imputes partial NaN cells with column mean).
- WeightCalculator: Hierarchical (two-level) weight calculation built on
  AdaptiveWeightCalculator.
- calculate_adaptive_weights: Convenience function for adaptive weights.

**Utilities:**
- global_min_max_normalize: Global min-max normalization function
- GlobalNormalizer: Stateful normalizer (fit/transform pattern)
- bayesian_bootstrap_weights: Bayesian bootstrap weight sampling
- temporal_stability_verification: Temporal stability validation
"""

from .critic_weighting import CRITICWeightingCalculator
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
    'CRITICWeightingCalculator',

    # Individual base calculator
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
