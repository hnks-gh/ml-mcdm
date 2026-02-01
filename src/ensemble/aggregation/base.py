# -*- coding: utf-8 -*-
"""Base classes for rank aggregation methods."""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class AggregatedRanking:
    """Result container for rank aggregation.
    
    Attributes
    ----------
    final_ranking : np.ndarray
        Final aggregated ranks (1 = best)
    final_scores : np.ndarray
        Final aggregated scores
    method_rankings : Dict[str, np.ndarray]
        Input rankings from each method
    method_weights : Dict[str, float]
        Weights assigned to each method
    agreement_matrix : np.ndarray
        Pairwise agreement between methods (Spearman correlation)
    kendall_w : float
        Kendall's coefficient of concordance (0-1)
    """
    final_ranking: np.ndarray
    final_scores: np.ndarray
    method_rankings: Dict[str, np.ndarray]
    method_weights: Dict[str, float]
    agreement_matrix: np.ndarray
    kendall_w: float
    
    def summary(self) -> str:
        """Generate summary string of aggregation results."""
        lines = [
            f"\n{'='*60}",
            "RANK AGGREGATION RESULTS",
            f"{'='*60}",
            f"\nKendall's W (agreement): {self.kendall_w:.4f}",
            f"\nMethod Weights:"
        ]
        for method, weight in self.method_weights.items():
            lines.append(f"  {method}: {weight:.4f}")
        lines.append(f"\nTop 10 Final Ranking:")
        top_10_idx = np.argsort(self.final_ranking)[:10]
        for i, idx in enumerate(top_10_idx):
            lines.append(f"  {i+1}. Index {idx} (Score: {self.final_scores[idx]:.4f})")
        lines.append("=" * 60)
        return "\n".join(lines)


class BaseRankAggregator:
    """
    Base class for rank aggregation methods.
    
    Provides common utilities for converting scores to ranks
    and calculating agreement metrics.
    """
    
    @staticmethod
    def scores_to_ranks(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
        """Convert scores to ranks (1 = best).
        
        Parameters
        ----------
        scores : np.ndarray
            Raw scores
        higher_is_better : bool, default=True
            If True, higher scores get lower (better) ranks
            
        Returns
        -------
        np.ndarray
            Ranks (1 = best)
        """
        if higher_is_better:
            return len(scores) - np.argsort(np.argsort(scores))
        else:
            return np.argsort(np.argsort(scores)) + 1
    
    @staticmethod
    def kendall_w(rankings: np.ndarray) -> float:
        """Calculate Kendall's W coefficient of concordance.
        
        Measures agreement between multiple rankers.
        
        Parameters
        ----------
        rankings : np.ndarray
            Matrix of rankings (methods x alternatives)
            
        Returns
        -------
        float
            Kendall's W (0 = no agreement, 1 = perfect agreement)
        """
        n_methods, n_alternatives = rankings.shape
        
        # Sum of ranks for each alternative
        rank_sums = rankings.sum(axis=0)
        
        # Mean rank sum
        mean_rank_sum = rank_sums.mean()
        
        # S = sum of squared deviations
        S = np.sum((rank_sums - mean_rank_sum) ** 2)
        
        # Maximum possible S
        S_max = (n_methods ** 2 * (n_alternatives ** 3 - n_alternatives)) / 12
        
        if S_max == 0:
            return 1.0
        
        return S / S_max
    
    @staticmethod
    def spearman_correlation(rank1: np.ndarray, rank2: np.ndarray) -> float:
        """Calculate Spearman rank correlation coefficient.
        
        Parameters
        ----------
        rank1, rank2 : np.ndarray
            Two ranking arrays
            
        Returns
        -------
        float
            Spearman correlation (-1 to 1)
        """
        n = len(rank1)
        d_squared = np.sum((rank1 - rank2) ** 2)
        return 1 - (6 * d_squared) / (n * (n ** 2 - 1))
    
    @staticmethod
    def kendall_tau_distance(rank1: np.ndarray, rank2: np.ndarray) -> int:
        """Calculate Kendall tau distance between two rankings.
        
        Counts number of pairwise disagreements.
        
        Parameters
        ----------
        rank1, rank2 : np.ndarray
            Two ranking arrays
            
        Returns
        -------
        int
            Number of pairwise disagreements
        """
        n = len(rank1)
        distance = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check if pair is concordant or discordant
                if (rank1[i] < rank1[j]) != (rank2[i] < rank2[j]):
                    distance += 1
        
        return distance
    
    def _normalize_weights(self, 
                          weights: Optional[Dict[str, float]], 
                          method_names: List[str]) -> Dict[str, float]:
        """Normalize method weights to sum to 1.
        
        Parameters
        ----------
        weights : Dict[str, float] or None
            Input weights (can be None for equal weights)
        method_names : List[str]
            Names of methods
            
        Returns
        -------
        Dict[str, float]
            Normalized weights summing to 1
        """
        n_methods = len(method_names)
        
        if weights is None:
            return {name: 1.0 / n_methods for name in method_names}
        
        # Ensure all methods have weights
        weights = {name: weights.get(name, 1.0 / n_methods) for name in method_names}
        
        # Normalize to sum to 1
        total_weight = sum(weights.values())
        return {k: v / total_weight for k, v in weights.items()}
    
    def _compute_agreement_matrix(self, 
                                 rankings: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute pairwise agreement matrix between methods.
        
        Parameters
        ----------
        rankings : Dict[str, np.ndarray]
            Rankings from each method
            
        Returns
        -------
        np.ndarray
            Symmetric matrix of Spearman correlations
        """
        method_names = list(rankings.keys())
        n_methods = len(method_names)
        
        agreement = np.zeros((n_methods, n_methods))
        for i, name_i in enumerate(method_names):
            for j, name_j in enumerate(method_names):
                agreement[i, j] = self.spearman_correlation(
                    rankings[name_i], rankings[name_j]
                )
        
        return agreement
    
    def aggregate(self,
                 rankings: Dict[str, np.ndarray],
                 weights: Optional[Dict[str, float]] = None) -> AggregatedRanking:
        """Aggregate multiple rankings into a final ranking.
        
        Must be implemented by subclasses.
        
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
        raise NotImplementedError("Subclasses must implement aggregate()")
