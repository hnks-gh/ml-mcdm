# -*- coding: utf-8 -*-
"""Borda Count rank aggregation method."""

import numpy as np
from typing import Dict, Optional

from .base import BaseRankAggregator, AggregatedRanking


class BordaCount(BaseRankAggregator):
    """
    Borda Count Rank Aggregation.
    
    The Borda Count is a single-winner election method in which voters 
    rank options. Each alternative receives points based on its ranking 
    position from each voter (method). Lower ranks receive more points.
    
    Mathematical Formulation
    ------------------------
    For n alternatives and m methods:
    
    Borda Score for alternative i:
        B(i) = Σⱼ wⱼ × (n - rⱼ(i))
        
    where:
    - wⱼ = weight of method j
    - rⱼ(i) = rank of alternative i by method j (1 = best)
    - n = number of alternatives
    
    Properties
    ----------
    - Satisfies monotonicity: improving an alternative's position 
      cannot hurt its final ranking
    - Does NOT satisfy independence of irrelevant alternatives
    - Computationally efficient: O(m × n)
    
    Example
    -------
    >>> from src.ensemble.aggregation import BordaCount
    >>> 
    >>> rankings = {
    ...     'TOPSIS': np.array([1, 3, 2, 4, 5]),
    ...     'VIKOR': np.array([2, 1, 3, 4, 5]),
    ...     'PROMETHEE': np.array([1, 2, 3, 5, 4])
    ... }
    >>> 
    >>> borda = BordaCount()
    >>> result = borda.aggregate(rankings)
    >>> print(result.final_ranking)
    
    References
    ----------
    [1] de Borda, J.C. (1781). "Mémoire sur les élections au scrutin"
    [2] Emerson, P. (2013). "The original Borda count and partial voting"
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize Borda Count aggregator.
        
        Parameters
        ----------
        weights : Dict[str, float], optional
            Default weights for each ranking method.
            If None, equal weights are used.
        """
        self.weights = weights
    
    def aggregate(self,
                 rankings: Dict[str, np.ndarray],
                 weights: Optional[Dict[str, float]] = None) -> AggregatedRanking:
        """
        Aggregate rankings using Borda Count.
        
        Parameters
        ----------
        rankings : Dict[str, np.ndarray]
            Dictionary of rankings {method_name: ranks}
            Ranks should be 1-indexed (1 = best)
        weights : Dict[str, float], optional
            Weights for each method (overrides default weights)
            
        Returns
        -------
        AggregatedRanking
            Aggregation result with final ranking and scores
        """
        method_names = list(rankings.keys())
        n_alternatives = len(list(rankings.values())[0])
        
        # Get normalized weights
        if weights is None:
            weights = self.weights
        weights = self._normalize_weights(weights, method_names)
        
        # Calculate Borda scores
        borda_scores = np.zeros(n_alternatives)
        
        for method_name, ranks in rankings.items():
            # Borda score = n - rank (so rank 1 gets n-1 points)
            method_scores = n_alternatives - ranks
            borda_scores += weights[method_name] * method_scores
        
        # Convert to final ranking
        final_ranking = self.scores_to_ranks(borda_scores, higher_is_better=True)
        
        # Create ranking matrix for Kendall's W
        ranking_matrix = np.array([rankings[name] for name in method_names])
        kendall = self.kendall_w(ranking_matrix)
        
        # Agreement matrix
        agreement = self._compute_agreement_matrix(rankings)
        
        return AggregatedRanking(
            final_ranking=final_ranking,
            final_scores=borda_scores,
            method_rankings=rankings,
            method_weights=weights,
            agreement_matrix=agreement,
            kendall_w=kendall
        )
    
    def calculate_positional_scores(self,
                                   rankings: Dict[str, np.ndarray],
                                   score_function: str = 'linear') -> np.ndarray:
        """
        Calculate Borda scores with different scoring functions.
        
        Parameters
        ----------
        rankings : Dict[str, np.ndarray]
            Method rankings
        score_function : str
            'linear': n-r (classic Borda)
            'exponential': 2^(n-r)
            'logarithmic': 1/log(r+1)
            
        Returns
        -------
        np.ndarray
            Positional scores
        """
        n_alternatives = len(list(rankings.values())[0])
        scores = np.zeros(n_alternatives)
        
        for ranks in rankings.values():
            if score_function == 'linear':
                scores += n_alternatives - ranks
            elif score_function == 'exponential':
                scores += 2 ** (n_alternatives - ranks)
            elif score_function == 'logarithmic':
                scores += 1 / np.log2(ranks + 1)
            else:
                raise ValueError(f"Unknown score function: {score_function}")
        
        return scores


def borda_count(rankings: Dict[str, np.ndarray],
               weights: Optional[Dict[str, float]] = None) -> AggregatedRanking:
    """
    Convenience function for Borda Count aggregation.
    
    Parameters
    ----------
    rankings : Dict[str, np.ndarray]
        Rankings from different methods
    weights : Dict[str, float], optional
        Method weights
        
    Returns
    -------
    AggregatedRanking
        Aggregation result
    """
    aggregator = BordaCount(weights)
    return aggregator.aggregate(rankings, weights)
