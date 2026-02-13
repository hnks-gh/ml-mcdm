# -*- coding: utf-8 -*-
"""
Weighting Methods Module

Objective weight calculation methods for MCDM:

**Primary Pipeline:**
- HybridWeightingPipeline: 4-method hybrid (Entropy + CRITIC + MEREC + SD)
  → Game Theory Weight Combination (GTWC) + Bayesian Bootstrap + Stability

**Adaptive Weighting:**
- AdaptiveWeightCalculator: Adaptive weight calculation with zero handling
- WeightCalculator: Hierarchical weight calculation for multi-level data
- calculate_adaptive_weights: Convenience function for adaptive weights

**Fusion Method:**
- GameTheoryWeightCombination: Cooperative game-theoretic fusion

**Individual Methods:**
- EntropyWeightCalculator: Information theory-based weighting
- CRITICWeightCalculator: Contrast intensity + inter-criteria correlation
- MERECWeightCalculator: Method based on Removal Effects of Criteria
- StandardDeviationWeightCalculator: Variance-based weighting

**Utilities:**
- global_min_max_normalize: Global normalization function
- GlobalNormalizer: Stateful normalizer (fit/transform pattern)
- bayesian_bootstrap_weights: Uncertainty quantification
- temporal_stability_verification: Temporal stability validation
"""

from .hybrid_weighting import HybridWeightingPipeline
from .entropy import EntropyWeightCalculator
from .critic import CRITICWeightCalculator
from .merec import MERECWeightCalculator
from .standard_deviation import StandardDeviationWeightCalculator
from .fusion import GameTheoryWeightCombination
from .normalization import global_min_max_normalize, GlobalNormalizer
from .bootstrap import bayesian_bootstrap_weights, BayesianBootstrap
from .validation import temporal_stability_verification, TemporalStabilityValidator
from .base import WeightResult, calculate_weights
from .adaptive import (
    AdaptiveWeightCalculator, 
    WeightCalculator, 
    AdaptiveWeightResult,
    calculate_adaptive_weights
)

# Alias for common usage
RobustGlobalWeighting = HybridWeightingPipeline

__all__ = [
    # Core result types
    'WeightResult',
    'AdaptiveWeightResult',
    
    # Primary pipeline
    'HybridWeightingPipeline',
    'RobustGlobalWeighting',
    
    # Adaptive weighting
    'AdaptiveWeightCalculator',
    'WeightCalculator',
    'calculate_adaptive_weights',
    
    # Individual calculators
    'EntropyWeightCalculator',
    'CRITICWeightCalculator',
    'MERECWeightCalculator',
    'StandardDeviationWeightCalculator',
    
    # Fusion method
    'GameTheoryWeightCombination',
    'global_min_max_normalize',
    'GlobalNormalizer',
    'bayesian_bootstrap_weights',
    'BayesianBootstrap',
    'temporal_stability_verification',
    'TemporalStabilityValidator',
    'calculate_weights',
]
