# -*- coding: utf-8 -*-
"""
Ensemble Weight Calculator

Combines multiple weighting methods for more robust weight determination.
Supports arithmetic, geometric, and harmonic mean aggregation.

Mathematical Formulas:
    Arithmetic Mean: w_j = Σ(α_m × w_mj)
    Geometric Mean:  w_j = (Π w_mj)^(1/M)
    Harmonic Mean:   w_j = M / Σ(1/w_mj)
    
where:
    w_mj = weight from method m for criterion j
    α_m = weight of method m
    M = number of methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.stats import spearmanr
from .base import WeightResult
from .entropy import EntropyWeightCalculator
from .critic import CRITICWeightCalculator


class EnsembleWeightCalculator:
    """
    Ensemble weight calculation combining multiple methods.
    
    Aggregates weights from Entropy and CRITIC methods using
    arithmetic, geometric, or harmonic mean to produce more
    robust final weights.
    
    Parameters
    ----------
    methods : List[str], optional
        Weighting methods to combine. Default: ['entropy', 'critic']
    aggregation : str
        Aggregation method: 'arithmetic', 'geometric', or 'harmonic'
    
    Attributes
    ----------
    methods : List[str]
        Weighting methods being combined
    aggregation : str
        Selected aggregation method
    entropy_calc : EntropyWeightCalculator
        Entropy calculator instance
    critic_calc : CRITICWeightCalculator
        CRITIC calculator instance
    
    Examples
    --------
    >>> import pandas as pd
    >>> from src.weighting import EnsembleWeightCalculator
    >>> 
    >>> data = pd.DataFrame({
    ...     'C1': [0.8, 0.6, 0.9, 0.7],
    ...     'C2': [0.5, 0.5, 0.5, 0.5],
    ...     'C3': [0.3, 0.9, 0.1, 0.7]
    ... })
    >>> 
    >>> # Default: geometric mean of entropy and CRITIC
    >>> calc = EnsembleWeightCalculator()
    >>> result = calc.calculate(data)
    >>> print(result.weights)
    >>> 
    >>> # Arithmetic mean with custom method weights
    >>> calc = EnsembleWeightCalculator(aggregation='arithmetic')
    >>> result = calc.calculate(data, method_weights={'entropy': 0.6, 'critic': 0.4})
    
    References
    ----------
    Wang, Y.M., & Luo, Y. (2010). Integration of correlations with 
    standard deviations for determining attribute weights in multiple 
    attribute decision making. Mathematical and Computer Modelling.
    """
    
    def __init__(self, 
                 methods: Optional[List[str]] = None, 
                 aggregation: str = "geometric"):
        self.methods = methods or ["entropy", "critic"]
        self.aggregation = aggregation
        self.entropy_calc = EntropyWeightCalculator()
        self.critic_calc = CRITICWeightCalculator()
    
    def calculate(self, 
                 data: pd.DataFrame, 
                 method_weights: Optional[Dict[str, float]] = None) -> WeightResult:
        """
        Calculate ensemble weights.
        
        Parameters
        ----------
        data : pd.DataFrame
            Decision matrix (alternatives × criteria)
        method_weights : Dict[str, float], optional
            Importance weights for each method (used in arithmetic mean)
        
        Returns
        -------
        WeightResult
            Ensemble weights with individual method weights and 
            correlation analysis
        """
        # Get individual weights
        weight_results = {}
        
        if "entropy" in self.methods:
            weight_results["entropy"] = self.entropy_calc.calculate(data)
        
        if "critic" in self.methods:
            weight_results["critic"] = self.critic_calc.calculate(data)
        
        # Default method weights
        if method_weights is None:
            method_weights = {m: 1.0 / len(self.methods) for m in self.methods}
        
        # Aggregate weights
        columns = data.columns.tolist()
        
        if self.aggregation == "arithmetic":
            ensemble_weights = self._arithmetic_mean(weight_results, method_weights, columns)
        elif self.aggregation == "geometric":
            ensemble_weights = self._geometric_mean(weight_results, columns)
        elif self.aggregation == "harmonic":
            ensemble_weights = self._harmonic_mean(weight_results, columns)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Calculate correlation between methods
        weight_correlation = {}
        methods_list = list(weight_results.keys())
        for i, m1 in enumerate(methods_list):
            for m2 in methods_list[i+1:]:
                w1 = np.array([weight_results[m1].weights[c] for c in columns])
                w2 = np.array([weight_results[m2].weights[c] for c in columns])
                corr, _ = spearmanr(w1, w2)
                weight_correlation[f"{m1}_vs_{m2}"] = corr
        
        return WeightResult(
            weights=ensemble_weights,
            method=f"ensemble_{self.aggregation}",
            details={
                "individual_weights": {m: r.weights for m, r in weight_results.items()},
                "method_weights": method_weights,
                "aggregation": self.aggregation,
                "weight_correlation": weight_correlation
            }
        )
    
    def _arithmetic_mean(self, 
                        results: Dict, 
                        method_weights: Dict, 
                        columns: List[str]) -> Dict[str, float]:
        """Weighted arithmetic mean."""
        ensemble = {}
        for col in columns:
            weighted_sum = sum(
                method_weights.get(m, 1/len(results)) * r.weights[col]
                for m, r in results.items()
            )
            ensemble[col] = weighted_sum
        
        # Normalize
        total = sum(ensemble.values())
        return {k: v/total for k, v in ensemble.items()}
    
    def _geometric_mean(self, 
                       results: Dict, 
                       columns: List[str]) -> Dict[str, float]:
        """Geometric mean (equal importance)."""
        ensemble = {}
        n_methods = len(results)
        
        for col in columns:
            product = 1.0
            for r in results.values():
                product *= r.weights[col] ** (1/n_methods)
            ensemble[col] = product
        
        total = sum(ensemble.values())
        return {k: v/total for k, v in ensemble.items()}
    
    def _harmonic_mean(self, 
                      results: Dict, 
                      columns: List[str]) -> Dict[str, float]:
        """Harmonic mean."""
        ensemble = {}
        n_methods = len(results)
        epsilon = 1e-10
        
        for col in columns:
            harmonic_sum = sum(1/(r.weights[col] + epsilon) for r in results.values())
            ensemble[col] = n_methods / harmonic_sum
        
        total = sum(ensemble.values())
        return {k: v/total for k, v in ensemble.items()}
