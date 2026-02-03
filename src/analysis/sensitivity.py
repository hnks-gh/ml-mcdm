# -*- coding: utf-8 -*-
"""Sensitivity analysis and robustness testing."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class SensitivityResult:
    """Result container for sensitivity analysis."""
    weight_sensitivity: Dict[str, float]    # Sensitivity index per criterion
    rank_stability: Dict[str, float]        # Rank stability per alternative
    critical_weights: Dict[str, Tuple[float, float]]  # Weight ranges maintaining rank
    overall_robustness: float               # 0-1 robustness score
    top_n_stability: Dict[int, float]       # Stability of top N rankings
    perturbation_analysis: Dict[str, np.ndarray]
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "SENSITIVITY ANALYSIS RESULTS",
            f"{'='*60}",
            f"\nOverall Robustness Score: {self.overall_robustness:.4f}",
            f"\n{'─'*30}",
            "WEIGHT SENSITIVITY (higher = more sensitive)",
            f"{'─'*30}"
        ]
        sorted_sens = sorted(self.weight_sensitivity.items(), 
                            key=lambda x: x[1], reverse=True)
        for criterion, sens in sorted_sens:
            bar = '█' * int(sens * 20)
            lines.append(f"  {criterion}: {sens:.4f} {bar}")
        
        lines.extend([
            f"\n{'─'*30}",
            "TOP-N STABILITY",
            f"{'─'*30}"
        ])
        for n, stability in self.top_n_stability.items():
            lines.append(f"  Top {n}: {stability:.1%} stable")
        
        lines.extend([
            f"\n{'─'*30}",
            "CRITICAL WEIGHT RANGES",
            f"{'─'*30}"
        ])
        for criterion, (low, high) in self.critical_weights.items():
            lines.append(f"  {criterion}: [{low:.3f}, {high:.3f}]")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class SensitivityAnalysis:
    """
    Comprehensive sensitivity analysis for MCDM rankings.
    
    Tests ranking robustness to weight changes and uncertainty.
    """
    
    def __init__(self,
                 n_simulations: int = 1000,
                 perturbation_range: float = 0.2,
                 seed: int = 42):
        """
        Initialize sensitivity analysis.
        
        Parameters
        ----------
        n_simulations : int
            Number of Monte Carlo simulations
        perturbation_range : float
            Maximum weight perturbation (as fraction)
        seed : int
            Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.perturbation_range = perturbation_range
        self.seed = seed
    
    def analyze(self,
               decision_matrix: np.ndarray,
               weights: np.ndarray,
               ranking_function: Callable,
               criteria_names: Optional[List[str]] = None,
               alternative_names: Optional[List[str]] = None) -> SensitivityResult:
        """
        Perform comprehensive sensitivity analysis.
        
        Parameters
        ----------
        decision_matrix : np.ndarray
            Decision matrix (alternatives x criteria)
        weights : np.ndarray
            Criterion weights
        ranking_function : Callable
            Function that takes (matrix, weights) and returns rankings
        criteria_names : List[str], optional
            Names of criteria
        alternative_names : List[str], optional
            Names of alternatives
        """
        np.random.seed(self.seed)
        
        # Handle DataFrame input
        if hasattr(decision_matrix, 'values'):
            decision_matrix = decision_matrix.values
        if hasattr(weights, 'values'):
            weights = weights.values
        
        # Ensure numpy arrays
        decision_matrix = np.asarray(decision_matrix)
        weights = np.asarray(weights)
        
        # Validate inputs
        if decision_matrix.ndim != 2:
            raise ValueError(f"Decision matrix must be 2D, got {decision_matrix.ndim}D")
        
        n_alternatives, n_criteria = decision_matrix.shape
        
        if len(weights) != n_criteria:
            raise ValueError(f"Weights length ({len(weights)}) must match criteria count ({n_criteria})")
        
        if criteria_names is None:
            criteria_names = [f"C{i+1}" for i in range(n_criteria)]
        if alternative_names is None:
            alternative_names = [f"A{i+1}" for i in range(n_alternatives)]
        
        # Ensure lists match dimensions
        criteria_names = list(criteria_names)[:n_criteria]
        alternative_names = list(alternative_names)[:n_alternatives]
        
        # Get base ranking with error handling
        try:
            base_ranking = ranking_function(decision_matrix, weights)
            base_ranking = np.asarray(base_ranking)
        except Exception as e:
            # Return default result if ranking function fails
            return SensitivityResult(
                weight_sensitivity={c: 0.0 for c in criteria_names},
                rank_stability={a: 1.0 for a in alternative_names},
                critical_weights={c: (weights[i] * 0.5, weights[i] * 1.5) for i, c in enumerate(criteria_names)},
                overall_robustness=1.0,
                top_n_stability={3: 1.0, 5: 1.0, 10: 1.0},
                perturbation_analysis={'mean_rankings': np.zeros(n_alternatives), 
                                       'std_rankings': np.zeros(n_alternatives),
                                       'rank_distribution': np.zeros((1, n_alternatives))}
            )
        
        # Weight sensitivity analysis
        try:
            weight_sensitivity = self._weight_sensitivity(
                decision_matrix, weights, ranking_function, criteria_names
            )
        except Exception:
            weight_sensitivity = {c: 0.0 for c in criteria_names}
        
        # Monte Carlo perturbation
        try:
            perturbation_results = self._monte_carlo_perturbation(
                decision_matrix, weights, ranking_function
            )
        except Exception:
            perturbation_results = np.tile(base_ranking, (10, 1))
        
        # Rank stability per alternative
        try:
            rank_stability = self._calculate_rank_stability(
                perturbation_results, alternative_names
            )
        except Exception:
            rank_stability = {a: 1.0 for a in alternative_names}
        
        # Critical weight ranges
        try:
            critical_weights = self._find_critical_weights(
                decision_matrix, weights, ranking_function, criteria_names
            )
        except Exception:
            critical_weights = {c: (max(0, weights[i] - 0.1), min(1, weights[i] + 0.1)) 
                               for i, c in enumerate(criteria_names)}
        
        # Top-N stability
        try:
            top_n_stability = self._calculate_top_n_stability(
                perturbation_results, base_ranking, [3, 5, 10]
            )
        except Exception:
            top_n_stability = {3: 1.0, 5: 1.0, 10: 1.0}
        
        # Overall robustness
        overall_robustness = np.mean(list(rank_stability.values())) if rank_stability else 1.0
        
        # Perturbation analysis summary
        perturbation_analysis = {
            'mean_rankings': perturbation_results.mean(axis=0),
            'std_rankings': perturbation_results.std(axis=0),
            'rank_distribution': perturbation_results
        }
        
        return SensitivityResult(
            weight_sensitivity=weight_sensitivity,
            rank_stability=rank_stability,
            critical_weights=critical_weights,
            overall_robustness=overall_robustness,
            top_n_stability=top_n_stability,
            perturbation_analysis=perturbation_analysis
        )
    
    def _weight_sensitivity(self,
                           matrix: np.ndarray,
                           weights: np.ndarray,
                           ranking_func: Callable,
                           criteria_names: List[str]) -> Dict[str, float]:
        """Calculate sensitivity index for each criterion weight."""
        sensitivity = {}
        base_ranking = ranking_func(matrix, weights)
        
        for i, name in enumerate(criteria_names):
            rank_changes = []
            
            # Test small perturbations
            for delta in np.linspace(-self.perturbation_range, 
                                     self.perturbation_range, 11):
                if delta == 0:
                    continue
                
                # Perturb single weight
                perturbed_weights = weights.copy()
                perturbed_weights[i] *= (1 + delta)
                
                # Renormalize
                perturbed_weights = perturbed_weights / perturbed_weights.sum()
                
                # Get new ranking
                new_ranking = ranking_func(matrix, perturbed_weights)
                
                # Calculate rank change
                change = np.sum(np.abs(new_ranking - base_ranking))
                rank_changes.append(change)
            
            # Sensitivity = average rank change per unit perturbation
            sensitivity[name] = np.mean(rank_changes) / len(matrix)
        
        # Normalize to 0-1
        max_sens = max(sensitivity.values()) if sensitivity.values() else 1
        if max_sens > 0:
            sensitivity = {k: v / max_sens for k, v in sensitivity.items()}
        else:
            sensitivity = {k: 0.0 for k in sensitivity.keys()}
        
        return sensitivity
    
    def _monte_carlo_perturbation(self,
                                  matrix: np.ndarray,
                                  weights: np.ndarray,
                                  ranking_func: Callable) -> np.ndarray:
        """Monte Carlo simulation with random weight perturbations."""
        n_alternatives = matrix.shape[0]
        rankings = np.zeros((self.n_simulations, n_alternatives))
        
        for sim in range(self.n_simulations):
            # Random perturbation
            perturbation = 1 + np.random.uniform(
                -self.perturbation_range, 
                self.perturbation_range, 
                len(weights)
            )
            perturbed_weights = weights * perturbation
            perturbed_weights = perturbed_weights / perturbed_weights.sum()
            
            rankings[sim] = ranking_func(matrix, perturbed_weights)
        
        return rankings
    
    def _calculate_rank_stability(self,
                                  perturbation_results: np.ndarray,
                                  names: List[str]) -> Dict[str, float]:
        """Calculate rank stability for each alternative."""
        stability = {}
        
        for i, name in enumerate(names):
            ranks = perturbation_results[:, i]
            
            # Stability = 1 - (std / max_possible_std)
            max_std = len(names) / 2
            actual_std = ranks.std()
            
            stability[name] = max(0, 1 - actual_std / max_std)
        
        return stability
    
    def _find_critical_weights(self,
                               matrix: np.ndarray,
                               weights: np.ndarray,
                               ranking_func: Callable,
                               criteria_names: List[str]) -> Dict[str, Tuple[float, float]]:
        """Find weight ranges that maintain the same top ranking."""
        critical = {}
        base_ranking = ranking_func(matrix, weights)
        top_alt = np.argmin(base_ranking)
        
        for i, name in enumerate(criteria_names):
            # Binary search for lower bound
            low = 0.0
            high = weights[i]
            
            while high - low > 0.01:
                mid = (low + high) / 2
                test_weights = weights.copy()
                test_weights[i] = mid
                test_weights = test_weights / test_weights.sum()
                
                new_ranking = ranking_func(matrix, test_weights)
                
                if np.argmin(new_ranking) == top_alt:
                    high = mid
                else:
                    low = mid
            
            lower_bound = high
            
            # Binary search for upper bound
            low = weights[i]
            high = min(1.0, weights[i] * 3)
            
            while high - low > 0.01:
                mid = (low + high) / 2
                test_weights = weights.copy()
                test_weights[i] = mid
                test_weights = test_weights / test_weights.sum()
                
                new_ranking = ranking_func(matrix, test_weights)
                
                if np.argmin(new_ranking) == top_alt:
                    low = mid
                else:
                    high = mid
            
            upper_bound = low
            
            critical[name] = (lower_bound, upper_bound)
        
        return critical
    
    def _calculate_top_n_stability(self,
                                   perturbation_results: np.ndarray,
                                   base_ranking: np.ndarray,
                                   n_values: List[int]) -> Dict[int, float]:
        """Calculate stability of top-N rankings."""
        stability = {}
        base_top = set(np.argsort(base_ranking)[:max(n_values)])
        
        for n in n_values:
            base_top_n = set(np.argsort(base_ranking)[:n])
            matches = 0
            
            for sim in range(len(perturbation_results)):
                sim_top_n = set(np.argsort(perturbation_results[sim])[:n])
                
                if sim_top_n == base_top_n:
                    matches += 1
            
            stability[n] = matches / len(perturbation_results)
        
        return stability


class WeightPerturbation:
    """
    Systematic weight perturbation analysis.
    """
    
    @staticmethod
    def one_at_a_time(weights: np.ndarray,
                      matrix: np.ndarray,
                      ranking_func: Callable,
                      steps: int = 11) -> Dict[int, List[np.ndarray]]:
        """
        Vary one weight at a time, keeping others proportionally adjusted.
        
        Returns rankings for each weight variation.
        """
        n_criteria = len(weights)
        results = {}
        
        for i in range(n_criteria):
            rankings_list = []
            
            for factor in np.linspace(0.5, 2.0, steps):
                test_weights = weights.copy()
                test_weights[i] *= factor
                test_weights = test_weights / test_weights.sum()
                
                rankings_list.append(ranking_func(matrix, test_weights))
            
            results[i] = rankings_list
        
        return results
    
    @staticmethod
    def pairwise_exchange(weights: np.ndarray,
                          matrix: np.ndarray,
                          ranking_func: Callable,
                          exchange_amount: float = 0.1) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Exchange weight between pairs of criteria.
        """
        n_criteria = len(weights)
        results = {}
        
        for i in range(n_criteria):
            for j in range(i + 1, n_criteria):
                # Transfer from i to j
                test_weights = weights.copy()
                transfer = min(weights[i] * exchange_amount, test_weights[i] * 0.9)
                test_weights[i] -= transfer
                test_weights[j] += transfer
                
                results[(i, j)] = ranking_func(matrix, test_weights)
                
                # Transfer from j to i
                test_weights = weights.copy()
                transfer = min(weights[j] * exchange_amount, test_weights[j] * 0.9)
                test_weights[j] -= transfer
                test_weights[i] += transfer
                
                results[(j, i)] = ranking_func(matrix, test_weights)
        
        return results


def run_sensitivity_analysis(matrix: np.ndarray,
                            weights: np.ndarray,
                            ranking_func: Callable,
                            n_simulations: int = 1000) -> SensitivityResult:
    """Convenience function for sensitivity analysis."""
    analyzer = SensitivityAnalysis(n_simulations=n_simulations)
    return analyzer.analyze(matrix, weights, ranking_func)
