# -*- coding: utf-8 -*-
"""
Ensemble Weight Calculator

Combines multiple weighting methods for more robust weight determination.
Supports 6 aggregation strategies from simple means to advanced hybrid methods.

Aggregation Strategies:
    Simple:
        Arithmetic Mean: w_j = Σ(α_m × w_mj)
        Geometric Mean:  w_j = (Π w_mj)^(1/M)   [min KL-divergence]
        Harmonic Mean:   w_j = M / Σ(1/w_mj)
    
    Advanced:
        Game Theory:          Min-deviation with entropy-confidence weighting
        Bayesian Bootstrap:   Inverse-variance weighting via resampling
        Integrated Hybrid:    Three-stage PCA→CRITIC→Entropy integration

References:
    Wang, Y.M., & Luo, Y. (2010). Integration of correlations with
    standard deviations for determining attribute weights in multiple
    attribute decision making. Mathematical and Computer Modelling.
    
    Yan, H.B., & Ma, T. (2015). A game theory-based approach for combining
    multiple sets of weights. Expert Systems with Applications.
    
    Deng, H., Yeh, C.H., & Willis, R.J. (2000). Inter-company comparison
    using modified TOPSIS with objective weights. Computers & Ops Research.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.stats import spearmanr
from .base import WeightResult
from .entropy import EntropyWeightCalculator
from .critic import CRITICWeightCalculator
from .pca import PCAWeightCalculator


class EnsembleWeightCalculator:
    """
    Ensemble weight calculation combining multiple objective methods.
    
    Aggregates weights from Entropy, CRITIC, and PCA methods using one
    of 6 strategies ranging from simple statistical means to advanced
    hybrid approaches.
    
    Parameters
    ----------
    methods : List[str], optional
        Weighting methods to combine. Default: ['entropy', 'critic', 'pca']
    aggregation : str
        Aggregation strategy. Options:
        - 'arithmetic': Weighted arithmetic mean
        - 'geometric': Geometric mean (min KL-divergence)
        - 'harmonic': Harmonic mean
        - 'game_theory': Min-deviation optimization with entropy-based
          confidence weighting
        - 'bayesian_bootstrap': Inverse-variance weighting via bootstrap
          resampling (auto-downweights unstable methods)
        - 'integrated_hybrid': Three-stage PCA→CRITIC→Entropy integration
          where methods structurally inform each other (recommended)
    pca_variance_threshold : float
        Cumulative variance threshold for PCA. Default 0.85.
    bootstrap_samples : int
        Number of bootstrap resamples for bayesian_bootstrap. Default 500.
    
    Attributes
    ----------
    methods : List[str]
        Weighting methods being combined
    aggregation : str
        Selected aggregation strategy
    
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
    >>> # Default: integrated hybrid of entropy + CRITIC + PCA
    >>> calc = EnsembleWeightCalculator()
    >>> result = calc.calculate(data)
    >>> 
    >>> # Game theory combination
    >>> calc = EnsembleWeightCalculator(aggregation='game_theory')
    >>> result = calc.calculate(data)
    >>> 
    >>> # Bayesian bootstrap (auto-downweights unstable methods)
    >>> calc = EnsembleWeightCalculator(aggregation='bayesian_bootstrap')
    >>> result = calc.calculate(data)
    >>> 
    >>> # Legacy: geometric mean of entropy + CRITIC only
    >>> calc = EnsembleWeightCalculator(
    ...     methods=['entropy', 'critic'], aggregation='geometric')
    >>> result = calc.calculate(data)
    
    References
    ----------
    Wang, Y.M., & Luo, Y. (2010). Integration of correlations with
    standard deviations for determining attribute weights.
    
    Yan, H.B., & Ma, T. (2015). A game theory-based approach for combining
    multiple sets of weights. Expert Systems with Applications.
    """
    
    VALID_AGGREGATIONS = {
        "arithmetic", "geometric", "harmonic",
        "game_theory", "bayesian_bootstrap", "integrated_hybrid"
    }
    
    def __init__(self, 
                 methods: Optional[List[str]] = None, 
                 aggregation: str = "integrated_hybrid",
                 pca_variance_threshold: float = 0.85,
                 bootstrap_samples: int = 500):
        self.methods = methods or ["entropy", "critic", "pca"]
        if aggregation not in self.VALID_AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. "
                f"Valid options: {sorted(self.VALID_AGGREGATIONS)}")
        self.aggregation = aggregation
        self.pca_variance_threshold = pca_variance_threshold
        self.bootstrap_samples = bootstrap_samples
        
        # Initialize calculators
        self.entropy_calc = EntropyWeightCalculator()
        self.critic_calc = CRITICWeightCalculator()
        self.pca_calc = PCAWeightCalculator(variance_threshold=pca_variance_threshold)
    
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
            Importance weights for each method (used in arithmetic mean).
            For game_theory and bayesian_bootstrap, these are computed
            automatically and this parameter is ignored.
        
        Returns
        -------
        WeightResult
            Ensemble weights with individual method details,
            correlation analysis, and strategy-specific metadata.
        """
        columns = data.columns.tolist()
        
        # Dispatch to the integrated hybrid (special path — deeply coupled)
        if self.aggregation == "integrated_hybrid":
            return self._integrated_hybrid(data, columns)
        
        # For all other strategies: compute individual weights independently
        weight_results = self._compute_individual_weights(data)
        
        # Default method weights for arithmetic mean
        if method_weights is None:
            method_weights = {m: 1.0 / len(self.methods) for m in self.methods}
        
        # Dispatch to aggregation strategy
        if self.aggregation == "arithmetic":
            ensemble_weights = self._arithmetic_mean(weight_results, method_weights, columns)
            strategy_details = {}
        elif self.aggregation == "geometric":
            ensemble_weights = self._geometric_mean(weight_results, columns)
            strategy_details = {}
        elif self.aggregation == "harmonic":
            ensemble_weights = self._harmonic_mean(weight_results, columns)
            strategy_details = {}
        elif self.aggregation == "game_theory":
            ensemble_weights, strategy_details = self._game_theory_combination(
                weight_results, columns)
        elif self.aggregation == "bayesian_bootstrap":
            ensemble_weights, strategy_details = self._bayesian_bootstrap_combination(
                data, weight_results, columns)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Calculate correlation between individual methods
        weight_correlation = self._compute_weight_correlations(weight_results, columns)
        
        details = {
            "individual_weights": {m: r.weights for m, r in weight_results.items()},
            "method_weights": method_weights,
            "aggregation": self.aggregation,
            "weight_correlation": weight_correlation,
            **strategy_details
        }
        
        return WeightResult(
            weights=ensemble_weights,
            method=f"ensemble_{self.aggregation}",
            details=details
        )
    
    # =========================================================================
    # Individual Weight Computation
    # =========================================================================
    
    def _compute_individual_weights(self, data: pd.DataFrame) -> Dict[str, WeightResult]:
        """Compute weights from each configured method independently."""
        weight_results = {}
        
        if "entropy" in self.methods:
            weight_results["entropy"] = self.entropy_calc.calculate(data)
        if "critic" in self.methods:
            weight_results["critic"] = self.critic_calc.calculate(data)
        if "pca" in self.methods:
            weight_results["pca"] = self.pca_calc.calculate(data)
        
        return weight_results
    
    def _compute_weight_correlations(self, weight_results: Dict[str, WeightResult],
                                      columns: List[str]) -> Dict[str, float]:
        """Compute pairwise Spearman rank correlations between weight vectors."""
        weight_correlation = {}
        methods_list = list(weight_results.keys())
        for i, m1 in enumerate(methods_list):
            for m2 in methods_list[i+1:]:
                w1 = np.array([weight_results[m1].weights[c] for c in columns])
                w2 = np.array([weight_results[m2].weights[c] for c in columns])
                if len(w1) >= 3:
                    corr, _ = spearmanr(w1, w2)
                else:
                    corr = np.corrcoef(w1, w2)[0, 1]
                weight_correlation[f"{m1}_vs_{m2}"] = float(corr)
        return weight_correlation
    
    # =========================================================================
    # Simple Aggregation Methods (preserved from original)
    # =========================================================================
    
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
        
        total = sum(ensemble.values())
        return {k: v/total for k, v in ensemble.items()}
    
    def _geometric_mean(self, 
                       results: Dict, 
                       columns: List[str]) -> Dict[str, float]:
        """Geometric mean (equal importance). Equivalent to min KL-divergence."""
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
    
    # =========================================================================
    # Advanced Strategy: Game Theory (Min-Deviation Optimization)
    # =========================================================================
    
    def _game_theory_combination(self, 
                                  weight_results: Dict[str, WeightResult],
                                  columns: List[str]
                                  ) -> tuple:
        """
        Game theory-based weight combination via min-deviation optimization.
        
        Each method's contribution is weighted by its entropy-based confidence:
        methods producing more differentiated (lower entropy) weight vectors
        are considered more "decisive" and get higher influence.
        
        The optimization minimizes:
            min_w Σ_m α_m × ||w - w^(m)||² 
            s.t. Σ w_j = 1, w_j ≥ 0
        
        where α_m = (1 - H(w^(m))) / Σ(1 - H(w^(l)))
        and H(w) = -Σ w_j ln(w_j) / ln(n) is the normalized entropy.
        
        Returns
        -------
        tuple
            (ensemble_weights_dict, strategy_details_dict)
        """
        n_criteria = len(columns)
        
        # Collect weight vectors and compute entropy-based confidence
        weight_vectors = {}
        method_entropies = {}
        confidence_scores = {}
        
        for method, result in weight_results.items():
            w = np.array([result.weights[c] for c in columns])
            weight_vectors[method] = w
            
            # Normalized Shannon entropy of the weight vector
            # H = 0 means one criterion has all weight (maximally decisive)
            # H = 1 means uniform weights (minimally decisive)
            w_safe = np.clip(w, 1e-15, None)
            H = -np.sum(w_safe * np.log(w_safe)) / (np.log(n_criteria) + 1e-15)
            H = min(H, 1.0)  # Clip for safety
            method_entropies[method] = float(H)
            confidence_scores[method] = max(1.0 - H, 1e-10)
        
        # Normalize confidence to get method-level weights α_m
        total_confidence = sum(confidence_scores.values())
        alpha = {m: c / total_confidence for m, c in confidence_scores.items()}
        
        # Closed-form solution: weighted average, then project onto simplex
        combined = np.zeros(n_criteria)
        for method, w in weight_vectors.items():
            combined += alpha[method] * w
        
        # Project onto probability simplex (non-negativity + sum-to-1)
        combined = np.clip(combined, 1e-15, None)
        combined = combined / combined.sum()
        
        ensemble_weights = {col: float(combined[j]) for j, col in enumerate(columns)}
        
        strategy_details = {
            "strategy": "game_theory",
            "method_entropies": method_entropies,
            "confidence_scores": confidence_scores,
            "game_theory_alpha": alpha,
            "description": ("Min-deviation optimization with entropy-based confidence. "
                          "Methods with more differentiated weights get higher influence.")
        }
        
        return ensemble_weights, strategy_details
    
    # =========================================================================
    # Advanced Strategy: Bayesian Bootstrap Combination
    # =========================================================================
    
    def _bayesian_bootstrap_combination(self,
                                         data: pd.DataFrame,
                                         weight_results: Dict[str, WeightResult],
                                         columns: List[str]
                                         ) -> tuple:
        """
        Bayesian bootstrap-based weight combination with inverse-variance weighting.
        
        Resamples the decision matrix to estimate each method's weight variance.
        Methods producing more stable weights across resamples get higher
        effective influence. This naturally down-weights unreliable methods.
        
        Process:
            1. Bootstrap-resample the decision matrix B times
            2. Recompute each method's weights on each resample
            3. Estimate per-criterion variance for each method
            4. Combine using inverse-variance weighting:
               w_j = (Σ_m σ²_mj^{-1})^{-1} × Σ_m σ²_mj^{-1} × w_mj
        
        Returns
        -------
        tuple
            (ensemble_weights_dict, strategy_details_dict)
        """
        m, n = data.shape
        n_boot = min(self.bootstrap_samples, 500)  # Cap for speed
        
        # Collect bootstrap weight samples for each method
        bootstrap_weights = {method: [] for method in weight_results.keys()}
        
        rng = np.random.RandomState(42)
        for _ in range(n_boot):
            # Resample rows with replacement
            idx = rng.choice(m, size=m, replace=True)
            boot_data = data.iloc[idx].reset_index(drop=True)
            
            # Handle degenerate resamples (zero-variance columns)
            col_stds = boot_data.std()
            if (col_stds < 1e-10).any():
                # Add tiny noise to zero-variance columns
                for col in col_stds[col_stds < 1e-10].index:
                    boot_data[col] += rng.normal(0, 1e-8, size=m)
            
            # Recompute weights for each method
            for method in weight_results.keys():
                try:
                    if method == "entropy":
                        result = self.entropy_calc.calculate(boot_data)
                    elif method == "critic":
                        result = self.critic_calc.calculate(boot_data)
                    elif method == "pca":
                        result = self.pca_calc.calculate(boot_data)
                    else:
                        continue
                    
                    w = np.array([result.weights[c] for c in columns])
                    bootstrap_weights[method].append(w)
                except Exception:
                    # Skip failed bootstraps (e.g., singular matrices)
                    continue
        
        # Compute original weights and bootstrap variance per criterion
        original_weights = {}
        weight_variances = {}
        bootstrap_stds = {}
        
        epsilon = 1e-15
        
        for method, result in weight_results.items():
            original_weights[method] = np.array([result.weights[c] for c in columns])
            
            if len(bootstrap_weights[method]) >= 10:
                boot_matrix = np.array(bootstrap_weights[method])  # (B, n)
                var = boot_matrix.var(axis=0)  # (n,)
                std = boot_matrix.std(axis=0)
            else:
                # Fallback: assume equal variance if bootstrap failed
                var = np.ones(n) * 0.01
                std = np.ones(n) * 0.1
            
            weight_variances[method] = np.clip(var, epsilon, None)
            bootstrap_stds[method] = std
        
        # Inverse-variance weighted combination per criterion
        combined = np.zeros(n)
        total_precision = np.zeros(n)
        
        for method in weight_results.keys():
            precision = 1.0 / weight_variances[method]  # (n,)
            combined += precision * original_weights[method]
            total_precision += precision
        
        combined = combined / (total_precision + epsilon)
        
        # Normalize to unit sum
        combined = np.clip(combined, epsilon, None)
        combined = combined / combined.sum()
        
        # Compute confidence intervals
        confidence_intervals = {}
        for method in weight_results.keys():
            if len(bootstrap_weights[method]) >= 10:
                boot_matrix = np.array(bootstrap_weights[method])
                ci_lower = np.percentile(boot_matrix, 2.5, axis=0)
                ci_upper = np.percentile(boot_matrix, 97.5, axis=0)
                confidence_intervals[method] = {
                    col: {"lower": float(ci_lower[j]), "upper": float(ci_upper[j])}
                    for j, col in enumerate(columns)
                }
        
        # Method-level stability scores (inverse of mean variance)
        stability_scores = {}
        for method, var in weight_variances.items():
            stability_scores[method] = float(1.0 / (var.mean() + epsilon))
        total_stability = sum(stability_scores.values())
        effective_weights = {m: s / total_stability for m, s in stability_scores.items()}
        
        ensemble_weights = {col: float(combined[j]) for j, col in enumerate(columns)}
        
        strategy_details = {
            "strategy": "bayesian_bootstrap",
            "n_bootstrap_samples": n_boot,
            "bootstrap_std": {
                method: {col: float(bootstrap_stds[method][j]) 
                        for j, col in enumerate(columns)}
                for method in weight_results.keys()
            },
            "confidence_intervals_95": confidence_intervals,
            "stability_scores": stability_scores,
            "effective_method_weights": effective_weights,
            "description": ("Inverse-variance weighted combination via bootstrap resampling. "
                          "Methods producing stable weights get higher influence.")
        }
        
        return ensemble_weights, strategy_details
    
    # =========================================================================
    # Advanced Strategy: Integrated Hybrid (Three-Stage)
    # =========================================================================
    
    def _integrated_hybrid(self, data: pd.DataFrame, 
                           columns: List[str]) -> WeightResult:
        """
        Three-stage deeply integrated hybrid: PCA → Modified CRITIC → Entropy-weighted.
        
        Unlike simple averaging, the three methods are structurally interdependent:
        
        Stage 1 — PCA Structural Analysis:
            Run PCA to get factor structure and compute PCA weights.
            Compute PCA-residualized correlation matrix.
        
        Stage 2 — Modified CRITIC with PCA-Informed Correlation:
            Use PCA-residualized correlations instead of raw Pearson correlations
            in the CRITIC conflict measure. This focuses CRITIC on *unique
            information* not captured by the dominant latent factors:
                C_j^hybrid = σ_j × Σ_k(1 - r_jk^residual)
        
        Stage 3 — Entropy-Weighted Integration:
            Compute entropy of each weight vector. Use divergence (1 - H)
            as integration coefficient. Methods with more differentiated
            weights are more "decisive" and get higher influence:
                w_j^final = Σ_m α_m × w_j^(m) / Σ_m α_m
                where α_m = 1 - H(w^(m))
        
        Returns
        -------
        WeightResult
            Integrated hybrid weights with full stage-by-stage details.
        """
        n_criteria = len(columns)
        epsilon = 1e-15
        
        # =====================================================================
        # Stage 1: PCA — Structural Analysis
        # =====================================================================
        
        # Standard PCA weights
        pca_result = self.pca_calc.calculate(data)
        pca_weights = np.array([pca_result.weights[c] for c in columns])
        
        # PCA-residualized correlation matrix
        residual_corr = self.pca_calc.get_residual_correlation(data)
        
        # Standard entropy weights
        entropy_result = self.entropy_calc.calculate(data)
        entropy_weights = np.array([entropy_result.weights[c] for c in columns])
        
        # =====================================================================
        # Stage 2: Modified CRITIC with PCA-Residualized Correlation
        # =====================================================================
        
        # Standard deviation (contrast intensity) — same as regular CRITIC
        std = data.std(axis=0)
        std = std.replace(0, epsilon)
        
        # Conflict measure using PCA-residualized correlation
        # This focuses on unique information not explained by dominant factors
        residual_conflict = (1 - residual_corr).sum(axis=0)
        
        # Hybrid information content
        C_hybrid = std * residual_conflict
        
        # Normalize to weights
        critic_hybrid_weights_series = C_hybrid / (C_hybrid.sum() + epsilon)
        critic_hybrid_weights = np.array([critic_hybrid_weights_series[c] for c in columns])
        
        # Also compute standard CRITIC for comparison
        critic_result = self.critic_calc.calculate(data)
        critic_standard_weights = np.array([critic_result.weights[c] for c in columns])
        
        # =====================================================================
        # Stage 3: Entropy-Weighted Integration
        # =====================================================================
        
        # Collect the three weight vectors
        weight_vectors = {
            "entropy": entropy_weights,
            "critic_pca_hybrid": critic_hybrid_weights,
            "pca": pca_weights
        }
        
        # Compute normalized entropy of each weight vector
        method_entropies = {}
        integration_coefficients = {}
        
        for method, w in weight_vectors.items():
            w_safe = np.clip(w, epsilon, None)
            w_norm = w_safe / w_safe.sum()  # Ensure normalization
            H = -np.sum(w_norm * np.log(w_norm)) / (np.log(n_criteria) + epsilon)
            H = min(H, 1.0)
            method_entropies[method] = float(H)
            integration_coefficients[method] = max(1.0 - H, epsilon)
        
        # Normalize integration coefficients
        total_coeff = sum(integration_coefficients.values())
        alpha = {m: c / total_coeff for m, c in integration_coefficients.items()}
        
        # Final weighted combination
        combined = np.zeros(n_criteria)
        for method, w in weight_vectors.items():
            combined += alpha[method] * w
        
        # Normalize
        combined = np.clip(combined, epsilon, None)
        combined = combined / combined.sum()
        
        ensemble_weights = {col: float(combined[j]) for j, col in enumerate(columns)}
        
        # Compute correlations between all weight vectors
        all_vectors = {
            "entropy": entropy_weights,
            "critic_standard": critic_standard_weights,
            "critic_pca_hybrid": critic_hybrid_weights,
            "pca": pca_weights,
            "integrated_hybrid": combined
        }
        
        correlations = {}
        method_names = list(all_vectors.keys())
        for i, m1 in enumerate(method_names):
            for m2 in method_names[i+1:]:
                if len(all_vectors[m1]) >= 3:
                    corr, _ = spearmanr(all_vectors[m1], all_vectors[m2])
                else:
                    corr = float(np.corrcoef(all_vectors[m1], all_vectors[m2])[0, 1])
                correlations[f"{m1}_vs_{m2}"] = float(corr)
        
        return WeightResult(
            weights=ensemble_weights,
            method="ensemble_integrated_hybrid",
            details={
                "aggregation": "integrated_hybrid",
                "individual_weights": {
                    "entropy": entropy_result.weights,
                    "critic_standard": critic_result.weights,
                    "critic_pca_hybrid": {col: float(critic_hybrid_weights[j]) 
                                          for j, col in enumerate(columns)},
                    "pca": pca_result.weights,
                },
                "method_weights": alpha,
                "method_entropies": method_entropies,
                "integration_coefficients": integration_coefficients,
                "pca_details": {
                    "n_components_retained": pca_result.details["n_components_retained"],
                    "total_variance_explained": pca_result.details["total_variance_explained"],
                },
                "hybrid_critic_vs_standard_critic": correlations.get(
                    "critic_standard_vs_critic_pca_hybrid", None),
                "weight_correlation": correlations,
                "stages": {
                    "stage_1": "PCA structural analysis + residualized correlation",
                    "stage_2": "Modified CRITIC with PCA-informed conflict measure",
                    "stage_3": "Entropy-weighted integration (decisive methods get more influence)"
                },
                "description": (
                    "Three-stage integrated hybrid: (1) PCA extracts factor structure and "
                    "residual correlations, (2) Modified CRITIC uses PCA-residualized "
                    "correlations for conflict measure focusing on unique information, "
                    "(3) Entropy of weight vectors determines integration coefficients "
                    "so more decisive methods contribute more to the final weights."
                )
            }
        )
