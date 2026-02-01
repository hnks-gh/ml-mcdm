# -*- coding: utf-8 -*-
"""Kemeny-Young optimal rank aggregation method."""

import numpy as np
from typing import Dict, Optional, List, Tuple
from itertools import permutations

from .base import BaseRankAggregator, AggregatedRanking
from .borda import BordaCount


class KemenyYoung(BaseRankAggregator):
    """
    Kemeny-Young Optimal Rank Aggregation.
    
    The Kemeny-Young method finds the ranking that minimizes the total 
    Kendall tau distance to all input rankings. It produces the 
    "consensus ranking" that is closest to all input rankings.
    
    Mathematical Formulation
    ------------------------
    Find ranking π* that minimizes:
    
        K(π) = Σⱼ wⱼ × τ(π, πⱼ)
        
    where:
    - wⱼ = weight of method j
    - τ(π, πⱼ) = Kendall tau distance between π and πⱼ
    
    Kendall tau distance counts pairwise disagreements:
        τ(π₁, π₂) = |{(i,j): π₁(i) < π₁(j) and π₂(i) > π₂(j)}|
    
    Properties
    ----------
    - Produces Condorcet winner when one exists
    - Minimizes total disagreement (consensus)
    - NP-hard for exact solution
    - Uses approximation for large problems (n > max_exact)
    
    Computational Complexity
    ------------------------
    - Exact: O(n! × m × n²) - exponential in n
    - Approximate: O(m × n²) - polynomial
    
    Example
    -------
    >>> from src.ensemble.aggregation import KemenyYoung
    >>> 
    >>> rankings = {
    ...     'TOPSIS': np.array([1, 3, 2, 4, 5]),
    ...     'VIKOR': np.array([2, 1, 3, 4, 5]),
    ...     'PROMETHEE': np.array([1, 2, 3, 5, 4])
    ... }
    >>> 
    >>> kemeny = KemenyYoung(max_exact=8)
    >>> result = kemeny.aggregate(rankings)
    >>> print(result.final_ranking)
    
    References
    ----------
    [1] Kemeny, J.G. (1959). "Mathematics without numbers"
    [2] Young, H.P. (1988). "Condorcet's theory of voting"
    [3] Dwork et al. (2001). "Rank aggregation revisited"
    """
    
    def __init__(self, max_exact: int = 8):
        """
        Initialize Kemeny-Young aggregator.
        
        Parameters
        ----------
        max_exact : int, default=8
            Maximum number of alternatives for exact solution.
            For n > max_exact, uses approximation algorithms.
        """
        self.max_exact = max_exact
    
    def aggregate(self,
                 rankings: Dict[str, np.ndarray],
                 weights: Optional[Dict[str, float]] = None) -> AggregatedRanking:
        """
        Aggregate rankings using Kemeny-Young method.
        
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
        
        weights = self._normalize_weights(weights, method_names)
        
        if n_alternatives <= self.max_exact:
            # Exact solution
            final_ranking, kemeny_distance = self._exact_kemeny(rankings, weights)
            kemeny_scores = -final_ranking.astype(float)  # Lower rank = higher score
        else:
            # Approximation using Borda + local search
            final_ranking, kemeny_scores = self._approximate_kemeny(rankings, weights)
        
        # Kendall's W
        ranking_matrix = np.array([rankings[name] for name in method_names])
        kendall = self.kendall_w(ranking_matrix)
        
        # Agreement matrix
        agreement = self._compute_agreement_matrix(rankings)
        
        return AggregatedRanking(
            final_ranking=final_ranking,
            final_scores=kemeny_scores,
            method_rankings=rankings,
            method_weights=weights,
            agreement_matrix=agreement,
            kendall_w=kendall
        )
    
    def _exact_kemeny(self,
                     rankings: Dict[str, np.ndarray],
                     weights: Dict[str, float]) -> Tuple[np.ndarray, float]:
        """
        Find exact Kemeny optimal ranking.
        
        Enumerates all permutations and finds minimum distance.
        Only feasible for small n.
        """
        method_names = list(rankings.keys())
        n = len(list(rankings.values())[0])
        
        best_ranking = None
        best_distance = float('inf')
        
        # Try all permutations
        for perm in permutations(range(n)):
            # Convert permutation to ranking
            candidate_ranking = np.zeros(n, dtype=int)
            for rank, idx in enumerate(perm):
                candidate_ranking[idx] = rank + 1
            
            # Calculate weighted Kendall tau distance to all rankings
            total_distance = 0
            for method_name, method_ranks in rankings.items():
                w = weights[method_name]
                tau_dist = self.kendall_tau_distance(candidate_ranking, method_ranks)
                total_distance += w * tau_dist
            
            if total_distance < best_distance:
                best_distance = total_distance
                best_ranking = candidate_ranking
        
        return best_ranking, best_distance
    
    def _approximate_kemeny(self,
                           rankings: Dict[str, np.ndarray],
                           weights: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate Kemeny using Borda initialization + local search.
        
        1. Start with Borda count ranking
        2. Apply local search (adjacent swaps) to improve
        """
        # Start with Borda as initial solution
        borda = BordaCount()
        borda_result = borda.aggregate(rankings, weights)
        
        current_ranking = borda_result.final_ranking.copy()
        current_distance = self._total_kendall_distance(current_ranking, rankings, weights)
        
        # Local search: try adjacent swaps
        improved = True
        max_iterations = 100
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            n = len(current_ranking)
            sorted_indices = np.argsort(current_ranking)
            
            # Try swapping adjacent pairs in the current order
            for i in range(n - 1):
                # Swap positions i and i+1
                test_ranking = current_ranking.copy()
                idx1, idx2 = sorted_indices[i], sorted_indices[i + 1]
                test_ranking[idx1], test_ranking[idx2] = test_ranking[idx2], test_ranking[idx1]
                
                test_distance = self._total_kendall_distance(test_ranking, rankings, weights)
                
                if test_distance < current_distance:
                    current_ranking = test_ranking
                    current_distance = test_distance
                    sorted_indices = np.argsort(current_ranking)
                    improved = True
        
        # Scores: negative distance (lower distance = higher score)
        scores = -current_ranking.astype(float)
        
        return current_ranking, scores
    
    def _total_kendall_distance(self,
                               ranking: np.ndarray,
                               rankings: Dict[str, np.ndarray],
                               weights: Dict[str, float]) -> float:
        """Calculate total weighted Kendall tau distance."""
        total = 0
        for method_name, method_ranks in rankings.items():
            w = weights[method_name]
            total += w * self.kendall_tau_distance(ranking, method_ranks)
        return total


class MedianRank(BaseRankAggregator):
    """
    Median Rank Aggregation.
    
    Uses the median rank across all methods for each alternative.
    Robust to outlier rankings from individual methods.
    
    Mathematical Formulation
    ------------------------
    For alternative i:
        Median_Rank(i) = median({rⱼ(i) : j = 1, ..., m})
    
    Properties
    ----------
    - Robust to outliers (single extreme rankings)
    - Simple and interpretable
    - May produce ties
    - O(m × n log m) complexity
    
    Example
    -------
    >>> from src.ensemble.aggregation import MedianRank
    >>> 
    >>> rankings = {
    ...     'TOPSIS': np.array([1, 3, 2, 4, 5]),
    ...     'VIKOR': np.array([2, 1, 3, 4, 5]),
    ...     'PROMETHEE': np.array([1, 2, 3, 5, 4])
    ... }
    >>> 
    >>> median = MedianRank()
    >>> result = median.aggregate(rankings)
    """
    
    def aggregate(self,
                 rankings: Dict[str, np.ndarray],
                 weights: Optional[Dict[str, float]] = None) -> AggregatedRanking:
        """
        Aggregate rankings using median rank.
        
        Parameters
        ----------
        rankings : Dict[str, np.ndarray]
            Dictionary of rankings {method_name: ranks}
        weights : Dict[str, float], optional
            Weights (not used, included for API consistency)
            
        Returns
        -------
        AggregatedRanking
            Aggregation result
        """
        method_names = list(rankings.keys())
        n_alternatives = len(list(rankings.values())[0])
        n_methods = len(method_names)
        
        if weights is None:
            weights = {name: 1.0 / n_methods for name in method_names}
        
        # Stack rankings
        ranking_matrix = np.array([rankings[name] for name in method_names])
        
        # Calculate median ranks
        median_ranks = np.median(ranking_matrix, axis=0)
        
        # Scores: negative median (lower median = better = higher score)
        scores = -median_ranks
        
        # Final ranking
        final_ranking = self.scores_to_ranks(scores, higher_is_better=True)
        
        kendall = self.kendall_w(ranking_matrix)
        agreement = self._compute_agreement_matrix(rankings)
        
        return AggregatedRanking(
            final_ranking=final_ranking,
            final_scores=scores,
            method_rankings=rankings,
            method_weights=weights,
            agreement_matrix=agreement,
            kendall_w=kendall
        )


def kemeny_young(rankings: Dict[str, np.ndarray],
                weights: Optional[Dict[str, float]] = None,
                max_exact: int = 8) -> AggregatedRanking:
    """
    Convenience function for Kemeny-Young aggregation.
    
    Parameters
    ----------
    rankings : Dict[str, np.ndarray]
        Rankings from different methods
    weights : Dict[str, float], optional
        Method weights
    max_exact : int, default=8
        Maximum alternatives for exact solution
        
    Returns
    -------
    AggregatedRanking
        Aggregation result
    """
    aggregator = KemenyYoung(max_exact=max_exact)
    return aggregator.aggregate(rankings, weights)


def median_rank(rankings: Dict[str, np.ndarray]) -> AggregatedRanking:
    """
    Convenience function for median rank aggregation.
    
    Parameters
    ----------
    rankings : Dict[str, np.ndarray]
        Rankings from different methods
        
    Returns
    -------
    AggregatedRanking
        Aggregation result
    """
    aggregator = MedianRank()
    return aggregator.aggregate(rankings)
