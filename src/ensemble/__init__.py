# -*- coding: utf-8 -*-
"""
Ensemble Module
===============

Ensemble methods for combining MCDM rankings and ML predictions.

Submodules
----------
aggregation
    Rank aggregation methods (Borda, Copeland, Kemeny-Young, Median)
    Stacking ensemble for prediction combination

Legacy Imports
--------------
For backward compatibility, classes are also available at the
package level. New code should import from submodules:

>>> from src.ensemble.aggregation import BordaCount, StackingEnsemble
"""

# Import from new aggregation submodule
from .aggregation import (
    # Base classes
    BaseRankAggregator,
    AggregatedRanking,
    
    # Rank aggregation methods
    BordaCount,
    CopelandMethod,
    KemenyYoung,
    MedianRank,
    
    # Stacking ensemble
    StackingEnsemble,
    TemporalStackingEnsemble,
    StackingResult,
    
    # Convenience functions
    aggregate_rankings,
    borda_count,
    copeland_method,
    kemeny_young,
    median_rank,
    stacking_ensemble,
)

# Backward compatibility alias
RankAggregator = BaseRankAggregator

__all__ = [
    # Base classes
    'BaseRankAggregator',
    'RankAggregator',  # Backward compatibility
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
