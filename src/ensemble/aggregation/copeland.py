# -*- coding: utf-8 -*-
"""Copeland rank aggregation method."""

import numpy as np
from typing import Dict, Optional

from .base import BaseRankAggregator, AggregatedRanking


class CopelandMethod(BaseRankAggregator):
    """
    Copeland Rank Aggregation Method.
    
    The Copeland method is based on pairwise comparisons. For each pair 
    of alternatives, the method counts how many methods (voters) prefer 
    one over the other. The Copeland score is wins minus losses.
    
    Mathematical Formulation
    ------------------------
    For each pair of alternatives (i, j):
    
    P(i, j) = Σₖ wₖ × I[rₖ(i) < rₖ(j)]
    
    where:
    - wₖ = weight of method k
    - I[·] = indicator function (1 if true, 0 otherwise)
    - rₖ(i) = rank of alternative i by method k
    
    Copeland Score for alternative i:
        C(i) = Σⱼ≠ᵢ [I(P(i,j) > P(j,i)) - I(P(i,j) < P(j,i))]
    
    Properties
    ----------
    - Condorcet winner selection: if a Condorcet winner exists, 
      Copeland will select it
    - More robust to extreme rankings than Borda
    - Computational complexity: O(m × n²) where m = methods, n = alternatives
    
    Example
    -------
    >>> from src.ensemble.aggregation import CopelandMethod
    >>> 
    >>> rankings = {
    ...     'TOPSIS': np.array([1, 3, 2, 4, 5]),
    ...     'VIKOR': np.array([2, 1, 3, 4, 5]),
    ...     'PROMETHEE': np.array([1, 2, 3, 5, 4])
    ... }
    >>> 
    >>> copeland = CopelandMethod()
    >>> result = copeland.aggregate(rankings)
    >>> print(result.final_ranking)
    
    References
    ----------
    [1] Copeland, A.H. (1951). "A 'reasonable' social welfare function"
    [2] Saari, D.G. (2000). "Mathematical structure of voting paradoxes"
    """
    
    def aggregate(self,
                 rankings: Dict[str, np.ndarray],
                 weights: Optional[Dict[str, float]] = None) -> AggregatedRanking:
        """
        Aggregate rankings using Copeland method.
        
        Parameters
        ----------
        rankings : Dict[str, np.ndarray]
            Dictionary of rankings {method_name: ranks}
        weights : Dict[str, float], optional
            Weights for each method
            
        Returns
        -------
        AggregatedRanking
            Aggregation result
        """
        method_names = list(rankings.keys())
        n_alternatives = len(list(rankings.values())[0])
        
        # Normalize weights
        weights = self._normalize_weights(weights, method_names)
        
        # Calculate weighted pairwise preference matrix
        pairwise_wins = np.zeros((n_alternatives, n_alternatives))
        
        for method_name, ranks in rankings.items():
            w = weights[method_name]
            for i in range(n_alternatives):
                for j in range(n_alternatives):
                    if i != j:
                        # i beats j if i has lower rank (better)
                        if ranks[i] < ranks[j]:
                            pairwise_wins[i, j] += w
        
        # Copeland scores: wins - losses
        copeland_scores = np.zeros(n_alternatives)
        for i in range(n_alternatives):
            for j in range(n_alternatives):
                if i != j:
                    if pairwise_wins[i, j] > pairwise_wins[j, i]:
                        copeland_scores[i] += 1
                    elif pairwise_wins[i, j] < pairwise_wins[j, i]:
                        copeland_scores[i] -= 1
        
        # Convert to ranking
        final_ranking = self.scores_to_ranks(copeland_scores, higher_is_better=True)
        
        # Kendall's W
        ranking_matrix = np.array([rankings[name] for name in method_names])
        kendall = self.kendall_w(ranking_matrix)
        
        # Agreement matrix
        agreement = self._compute_agreement_matrix(rankings)
        
        return AggregatedRanking(
            final_ranking=final_ranking,
            final_scores=copeland_scores,
            method_rankings=rankings,
            method_weights=weights,
            agreement_matrix=agreement,
            kendall_w=kendall
        )
    
    def get_pairwise_matrix(self, 
                           rankings: Dict[str, np.ndarray],
                           weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Get the pairwise preference matrix.
        
        Parameters
        ----------
        rankings : Dict[str, np.ndarray]
            Method rankings
        weights : Dict[str, float], optional
            Method weights
            
        Returns
        -------
        np.ndarray
            Matrix P where P[i,j] = weighted sum of methods preferring i to j
        """
        method_names = list(rankings.keys())
        n_alternatives = len(list(rankings.values())[0])
        weights = self._normalize_weights(weights, method_names)
        
        pairwise = np.zeros((n_alternatives, n_alternatives))
        
        for method_name, ranks in rankings.items():
            w = weights[method_name]
            for i in range(n_alternatives):
                for j in range(n_alternatives):
                    if ranks[i] < ranks[j]:
                        pairwise[i, j] += w
        
        return pairwise
    
    def find_condorcet_winner(self, 
                              rankings: Dict[str, np.ndarray],
                              weights: Optional[Dict[str, float]] = None) -> Optional[int]:
        """
        Find the Condorcet winner if one exists.
        
        A Condorcet winner beats all other alternatives in pairwise comparison.
        
        Parameters
        ----------
        rankings : Dict[str, np.ndarray]
            Method rankings
        weights : Dict[str, float], optional
            Method weights
            
        Returns
        -------
        int or None
            Index of Condorcet winner, or None if none exists
        """
        pairwise = self.get_pairwise_matrix(rankings, weights)
        n = pairwise.shape[0]
        
        for i in range(n):
            is_winner = True
            for j in range(n):
                if i != j and pairwise[i, j] <= pairwise[j, i]:
                    is_winner = False
                    break
            if is_winner:
                return i
        
        return None


def copeland_method(rankings: Dict[str, np.ndarray],
                   weights: Optional[Dict[str, float]] = None) -> AggregatedRanking:
    """
    Convenience function for Copeland aggregation.
    
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
    aggregator = CopelandMethod()
    return aggregator.aggregate(rankings, weights)
