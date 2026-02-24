# -*- coding: utf-8 -*-
"""
Bayesian Bootstrap for Weight Uncertainty Quantification

Implements the Bayesian Bootstrap (Rubin, 1981) using Dirichlet resampling
to quantify uncertainty in weight estimates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional
import logging

logger = logging.getLogger(__name__)


def bayesian_bootstrap_weights(
    X_norm: np.ndarray,
    criteria_cols: List[str],
    weight_calculator: Callable[..., np.ndarray],
    n_iterations: int = 1000,
    seed: int = 42,
    epsilon: float = 1e-10
) -> Dict:
    """
    Perform Bayesian Bootstrap for weight uncertainty quantification.
    
    Uses continuous Dirichlet observation weights (Rubin, 1981) passed
    directly to the weight calculators, avoiding lossy discrete resampling.
    Each underlying method (Entropy, CRITIC, MEREC, SD) accepts the
    observation weight vector and computes weighted statistics internally.
    
    Parameters
    ----------
    X_norm : np.ndarray, shape (n_observations, n_criteria)
        Normalized criteria matrix.
    criteria_cols : List[str]
        Names of criteria columns.
    weight_calculator : Callable
        Function with signature
        ``(X_df, criteria_cols, sample_weights=None) -> np.ndarray``.
        *sample_weights* is a 1-D array of observation weights summing to 1.
    n_iterations : int, default=1000
        Number of bootstrap iterations. Odd number to avoid interpolation
        at percentiles (2.5%, 97.5%).
    seed : int, default=42
        Random seed for reproducibility.
    epsilon : float, default=1e-10
        Numerical stability constant.
    
    Returns
    -------
    results : Dict
        Dictionary containing:
        - 'mean_weights': np.ndarray, posterior mean (final weights)
        - 'std_weights': np.ndarray, posterior standard deviation
        - 'ci_lower': np.ndarray, 2.5th percentile (lower bound of 95% CI)
        - 'ci_upper': np.ndarray, 97.5th percentile (upper bound of 95% CI)
        - 'all_weights': np.ndarray, shape (n_iterations, n_criteria)
        - 'convergence_rate': float, proportion of successful iterations
    
    Notes
    -----
    **Bayesian Bootstrap Algorithm (Rubin, 1981):**
    
    For each iteration b = 1, ..., B:
    1. Draw observation weights from Dirichlet(1, ..., 1):
       - Sample g_i ~ Exponential(1) for i = 1, ..., N
       - Compute w_i = g_i / Σ_k g_k
    2. Pass continuous weights to each weight calculator which computes
       weighted proportions / weighted covariance / weighted removal
       effects internally.
    3. Fuse the four weight vectors via GTWC.
    4. Store the fused weight vector.
    
    **Why continuous weights instead of discrete resampling?**
    - Preserves the full information in the Dirichlet draw.
    - Discrete ``rng.choice()`` collapses the continuous weight vector
      into a multinomial count vector, discarding weight precision and
      potentially duplicating / dropping observations.
    - All four weight methods (Entropy, CRITIC, MEREC, SD) now accept
      ``sample_weights`` for exact weighted computation.
    
    **Why B=999?**
    - Odd number avoids interpolation at 2.5th and 97.5th percentiles
    - Standard practice for percentile-based credible intervals
    - Provides stable posterior statistics (Davison & Hinkley, 1997)
    
    References
    ----------
    1. Rubin, D.B. (1981). The Bayesian Bootstrap. The Annals of Statistics,
       9(1), 130-134.
    2. Davison, A.C. & Hinkley, D.V. (1997). Bootstrap Methods and Their
       Application. Cambridge University Press.
    3. Efron, B. & Tibshirani, R.J. (1993). An Introduction to the Bootstrap.
       Chapman & Hall.
    """
    N, p = X_norm.shape
    B = n_iterations
    rng = np.random.RandomState(seed)
    
    # Full dataset (used in every iteration — no resampling)
    X_df_full = pd.DataFrame(X_norm, columns=criteria_cols)
    
    # Storage for bootstrap samples
    all_weights = np.zeros((B, p))
    failed_iterations = 0
    
    logger.info(f"Starting Bayesian Bootstrap: {B} iterations on "
                f"{N} observations × {p} criteria (continuous weights)")
    
    for b in range(B):
        try:
            # Step 1: Draw Dirichlet(1,...,1) weights via exponential trick
            g = rng.exponential(1.0, size=N)
            obs_weights = g / g.sum()
            
            # Step 2: Pass continuous observation weights to the weight
            #         calculator (no discrete resampling).
            W_boot = weight_calculator(X_df_full, criteria_cols,
                                       sample_weights=obs_weights)
            
            # Validate and normalize
            if np.any(np.isnan(W_boot)) or np.any(np.isinf(W_boot)):
                raise ValueError("NaN or Inf in bootstrap weights")
            
            W_boot = W_boot / (W_boot.sum() + epsilon)
            all_weights[b, :] = W_boot
            
        except Exception as e:
            # Fallback: use previous iteration or uniform weights
            failed_iterations += 1
            if b > 0:
                all_weights[b, :] = all_weights[b-1, :]
            else:
                all_weights[b, :] = 1.0 / p
            
            if failed_iterations <= 5:  # Only log first few failures
                logger.warning(f"Bootstrap iteration {b} failed: {e}")
    
    if failed_iterations > 0:
        logger.warning(f"Bootstrap: {failed_iterations}/{B} iterations failed "
                      f"({100*failed_iterations/B:.1f}%)")
    
    # Calculate posterior statistics
    mean_weights = all_weights.mean(axis=0)
    mean_weights = mean_weights / (mean_weights.sum() + epsilon)  # Renormalize
    
    std_weights = all_weights.std(axis=0, ddof=1)
    
    ci_lower = np.percentile(all_weights, 2.5, axis=0)
    ci_upper = np.percentile(all_weights, 97.5, axis=0)
    
    convergence_rate = 1.0 - (failed_iterations / B)
    
    logger.info(f"Bootstrap complete: convergence rate = {convergence_rate:.3f}, "
                f"mean std = {std_weights.mean():.6f}")
    
    return {
        'mean_weights': mean_weights,
        'std_weights': std_weights,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'all_weights': all_weights,
        'convergence_rate': convergence_rate,
    }


class BayesianBootstrap:
    """
    Stateful Bayesian Bootstrap for repeated use with same configuration.
    
    Parameters
    ----------
    n_iterations : int, default=1000
        Number of bootstrap iterations.
    seed : int, default=42
        Random seed for reproducibility.
    epsilon : float, default=1e-10
        Numerical stability constant.
    
    Examples
    --------
    >>> bootstrap = BayesianBootstrap(n_iterations=1000, seed=42)
    >>> results = bootstrap.run(X_norm, criteria_cols, weight_calculator)
    """
    
    def __init__(
        self,
        n_iterations: int = 1000,
        seed: int = 42,
        epsilon: float = 1e-10
    ):
        self.n_iterations = n_iterations
        self.seed = seed
        self.epsilon = epsilon
    
    def run(
        self,
        X_norm: np.ndarray,
        criteria_cols: List[str],
        weight_calculator: Callable[..., np.ndarray]
    ) -> Dict:
        """
        Execute Bayesian Bootstrap.
        
        See bayesian_bootstrap_weights() for parameter details.
        """
        return bayesian_bootstrap_weights(
            X_norm=X_norm,
            criteria_cols=criteria_cols,
            weight_calculator=weight_calculator,
            n_iterations=self.n_iterations,
            seed=self.seed,
            epsilon=self.epsilon
        )
