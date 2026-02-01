# -*- coding: utf-8 -*-
"""
Ensemble Aggregation Module
===========================

This module provides rank aggregation methods and ensemble techniques
for combining results from multiple MCDM methods or ML models.

Rank Aggregation Methods
------------------------
- BordaCount: Positional voting method
- CopelandMethod: Pairwise comparison method
- KemenyYoung: Optimal consensus ranking
- MedianRank: Median-based aggregation

Stacking Ensemble
-----------------
- StackingEnsemble: Meta-learner for combining predictions
- TemporalStackingEnsemble: Time-aware stacking for panel data

Example
-------
>>> from src.ensemble.aggregation import (
...     BordaCount, CopelandMethod, StackingEnsemble,
...     aggregate_rankings, AggregatedRanking
... )
>>> 
>>> # Aggregate MCDM rankings
>>> rankings = {
...     'TOPSIS': topsis_ranks,
...     'VIKOR': vikor_ranks,
...     'PROMETHEE': promethee_ranks
... }
>>> 
>>> result = aggregate_rankings(rankings, method='borda')
>>> print(result.summary())
"""

from .base import BaseRankAggregator, AggregatedRanking
from .borda import BordaCount, borda_count
from .copeland import CopelandMethod, copeland_method
from .kemeny import KemenyYoung, MedianRank, kemeny_young, median_rank
from .stacking import (
    StackingEnsemble, 
    TemporalStackingEnsemble, 
    StackingResult,
    stacking_ensemble
)

# Convenience aggregation function
from typing import Dict, Optional
import numpy as np


def aggregate_rankings(rankings: Dict[str, np.ndarray],
                      method: str = 'borda',
                      weights: Optional[Dict[str, float]] = None) -> AggregatedRanking:
    """
    Aggregate multiple rankings into a consensus ranking.
    
    Parameters
    ----------
    rankings : Dict[str, np.ndarray]
        Rankings from different methods {method_name: ranks}
        Ranks should be 1-indexed (1 = best)
    method : str, default='borda'
        Aggregation method:
        - 'borda': Borda Count (positional voting)
        - 'copeland': Copeland method (pairwise comparison)
        - 'kemeny': Kemeny-Young (optimal consensus)
        - 'median': Median rank
    weights : Dict[str, float], optional
        Weights for each method. If None, equal weights used.
        
    Returns
    -------
    AggregatedRanking
        Aggregation result containing:
        - final_ranking: Consensus ranks
        - final_scores: Aggregated scores
        - kendall_w: Agreement coefficient
        - agreement_matrix: Pairwise method correlations
        
    Example
    -------
    >>> rankings = {
    ...     'TOPSIS': np.array([1, 3, 2, 4, 5]),
    ...     'VIKOR': np.array([2, 1, 3, 4, 5]),
    ...     'PROMETHEE': np.array([1, 2, 3, 5, 4])
    ... }
    >>> 
    >>> result = aggregate_rankings(rankings, method='borda')
    >>> print(f"Final ranking: {result.final_ranking}")
    >>> print(f"Agreement (Kendall W): {result.kendall_w:.3f}")
    """
    aggregators = {
        'borda': BordaCount(weights),
        'copeland': CopelandMethod(),
        'kemeny': KemenyYoung(),
        'median': MedianRank()
    }
    
    if method not in aggregators:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Choose from: {list(aggregators.keys())}"
        )
    
    aggregator = aggregators[method]
    return aggregator.aggregate(rankings, weights)


__all__ = [
    # Base classes
    'BaseRankAggregator',
    'AggregatedRanking',
    
    # Rank aggregation methods
    'BordaCount',
    'CopelandMethod', 
    'KemenyYoung',
    'MedianRank',
    
    # Stacking ensemble
    'StackingEnsemble',
    'TemporalStackingEnsemble',
    'StackingResult',
    
    # Convenience functions
    'aggregate_rankings',
    'borda_count',
    'copeland_method',
    'kemeny_young',
    'median_rank',
    'stacking_ensemble',
]
