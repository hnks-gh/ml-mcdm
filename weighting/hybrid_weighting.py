# -*- coding: utf-8 -*-
"""
Hybrid Weighting Pipeline for Panel MCDM Data

Combines four objective weighting methods (Entropy, CRITIC, MEREC, Standard Deviation)
through Game Theory Weight Combination (GTWC) with intra-group hybridization,
validated via Bayesian Bootstrap and temporal stability verification.

This is the primary weighting system for the MCDM forecasting pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

from .base import WeightResult
from .entropy import EntropyWeightCalculator
from .critic import CRITICWeightCalculator
from .merec import MERECWeightCalculator
from .standard_deviation import StandardDeviationWeightCalculator
from .fusion import GameTheoryWeightCombination
from .normalization import global_min_max_normalize
from .bootstrap import bayesian_bootstrap_weights
from .validation import temporal_stability_verification

logger = logging.getLogger(__name__)


class HybridWeightingPipeline:
    """
    Four-method hybrid weighting pipeline with Bayesian Bootstrap validation.
    
    Combines Entropy, CRITIC, MEREC, and Standard Deviation methods via
    game-theoretic weight combination (GTWC) or reliability-weighted fusion.
    Quantifies uncertainty through Bayesian Bootstrap and validates temporal stability.
    
    Parameters
    ----------
    bootstrap_iterations : int, default=999
        Number of Bayesian Bootstrap iterations for uncertainty quantification.
    stability_threshold : float, default=0.95
        Minimum cosine similarity for temporal stability (split-half test).
    epsilon : float, default=1e-10
        Numerical stability constant for avoiding division by zero.
    seed : int, default=42
        Random seed for reproducible bootstrap sampling.
    
    Attributes
    ----------
    entropy_calc : EntropyWeightCalculator
        Shannon entropy-based weighting.
    critic_calc : CRITICWeightCalculator
        CRITIC (contrast intensity + correlation) weighting.
    merec_calc : MERECWeightCalculator
        MEREC (removal effects) weighting.
    sd_calc : StandardDeviationWeightCalculator
        Standard deviation-based weighting.
    gtwc : GameTheoryWeightCombination
        Game Theory Weight Combination for Nash equilibrium fusion.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from weighting import HybridWeightingPipeline
    >>> 
    >>> # Panel data: long format with entity, time, and criteria
    >>> panel_data = pd.DataFrame({
    >>>     'Year': [2020, 2020, 2021, 2021],
    >>>     'Province': ['A', 'B', 'A', 'B'],
    >>>     'C1': [1.0, 2.0, 1.5, 2.5],
    >>>     'C2': [10.0, 20.0, 15.0, 25.0],
    >>> })
    >>> 
    >>> # Initialize pipeline with GTWC fusion
    >>> pipeline = HybridWeightingPipeline(
    >>>     bootstrap_iterations=999
    >>> )
    >>> 
    >>> # Calculate weights
    >>> result = pipeline.calculate(
    >>>     panel_data,
    >>>     entity_col='Province',
    >>>     time_col='Year',
    >>>     criteria_cols=['C1', 'C2']
    >>> )
    >>> 
    >>> # Access final weights
    >>> print(result.weights)  # {'C1': 0.xx, 'C2': 0.xx}
    >>> 
    >>> # Check GTWC coefficients
    >>> print(result.details['fusion']['game_theory_coefficients'])
    >>> 
    >>> # Check uncertainty
    >>> print(result.details['bootstrap']['std_weights'])
    >>> 
    >>> # Verify stability
    >>> print(result.details['stability']['is_stable'])
    
    Notes
    -----
    **Pipeline Workflow:**
    
    1. **Global Normalization:** Min-max normalize entire panel (preserves
       temporal trends)
    
    2. **Calculate Four Weight Vectors:**
       - Entropy: information-theoretic dispersion
       - CRITIC: variance + independence
       - MEREC: criterion removal impact
       - Standard Deviation: variance-based
    
    3. **Weight Fusion (GTWC - Default):**
       
       a. **Logical Clustering:**
          - Group A (Dispersion): Entropy + SD → Geometric Mean
          - Group B (Interaction): CRITIC + MEREC → Harmonic Mean
       
       b. **Game-Theoretic Optimization:**
          - Find Nash equilibrium coefficients (α₁, α₂)
          - Minimize L2-distance to both groups simultaneously
          - Solve: A @ α = b where A is dot product matrix
       
       c. **Final Aggregation:**
          - W* = α₁·W_GroupA + α₂·W_GroupB
    
    4. **Bayesian Bootstrap:** Quantify uncertainty via Dirichlet resampling
       (999 iterations)
    
    5. **Temporal Stability:** Split-half validation ensures weights are
       structural (not time-dependent)
    
    **Output:** Final weights with comprehensive uncertainty bounds and
    validation metrics.
    
    References
    ----------
    1. Shannon (1948). A Mathematical Theory of Communication.
    2. Diakoulaki et al. (1995). The CRITIC method. Computers & OR.
    3. Keshavarz-Ghorabaee et al. (2021). MEREC method. Symmetry, 13(4).
    4. Wang & Luo (2010). Standard deviation weighting. Math & Comp Modelling.
    5. Rubin (1981). The Bayesian Bootstrap. Annals of Statistics.
    6. Nash (1950). Equilibrium points in n-person games. PNAS, 36(1).
    7. Yager (1988). OWA operators. IEEE Trans Systems, Man, Cybernetics.
    8. Saaty (1980). The Analytic Hierarchy Process. McGraw-Hill.
    """

    def __init__(
        self,
        bootstrap_iterations: int = 999,
        stability_threshold: float = 0.95,
        epsilon: float = 1e-10,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        bootstrap_iterations : int, default=999
            Number of Bayesian bootstrap iterations for uncertainty quantification.
        stability_threshold : float, default=0.95
            Minimum required stability score (0-1) for validation.
        epsilon : float, default=1e-10
            Numerical stability constant.
        seed : int, default=42
            Random seed for reproducibility.
        """
        self.bootstrap_iterations = bootstrap_iterations
        self.stability_threshold = stability_threshold
        self.epsilon = epsilon
        self.seed = seed
        
        # Initialize individual weight calculators
        self.entropy_calc = EntropyWeightCalculator(epsilon=epsilon)
        self.critic_calc = CRITICWeightCalculator(epsilon=epsilon)
        self.merec_calc = MERECWeightCalculator(epsilon=epsilon)
        self.sd_calc = StandardDeviationWeightCalculator(epsilon=epsilon)
        
        # Initialize GTWC fusion system
        self.gtwc = GameTheoryWeightCombination(epsilon=epsilon)

    def calculate(
        self,
        panel_data: pd.DataFrame,
        entity_col: str = "Province",
        time_col: str = "Year",
        criteria_cols: Optional[List[str]] = None,
    ) -> WeightResult:
        """
        Execute hybrid weighting pipeline on panel data.
        
        Parameters
        ----------
        panel_data : pd.DataFrame
            Panel data in long format with entity, time, and criteria columns.
        entity_col : str, default='Province'
            Name of entity identifier column.
        time_col : str, default='Year'
            Name of time period column.
        criteria_cols : List[str], optional
            Names of criteria columns. If None, auto-detects numeric columns
            excluding entity_col and time_col.
        
        Returns
        -------
        WeightResult
            Result object containing:
            - weights: Dict[str, float] - final weights (posterior mean)
            - method: str - 'hybrid_weighting_pipeline'
            - details: Dict - comprehensive statistics and metadata
        """
        # Validate and prepare data
        panel_data = panel_data.copy()
        if criteria_cols is None:
            criteria_cols = [c for c in panel_data.columns
                            if c not in (entity_col, time_col)
                            and pd.api.types.is_numeric_dtype(panel_data[c])]

        n_obs = len(panel_data)
        n_criteria = len(criteria_cols)
        logger.info(f"Hybrid Weighting Pipeline: {n_obs} observations × "
                    f"{n_criteria} criteria")

        # Extract criteria matrix
        X_raw = panel_data[criteria_cols].values.astype(np.float64)

        # ── Step 1: Global Min-Max Normalization ──
        X_norm = global_min_max_normalize(X_raw, epsilon=self.epsilon)
        logger.info("Step 1: Global normalization complete")

        # ── Step 2: Calculate Individual Weight Vectors ──
        X_df = pd.DataFrame(X_norm, columns=criteria_cols)
        
        # 2a. Entropy weights
        entropy_result = self.entropy_calc.calculate(X_df)
        W_e = np.array([entropy_result.weights[c] for c in criteria_cols])
        logger.info(f"Step 2a: Entropy — range [{W_e.min():.4f}, {W_e.max():.4f}]")
        
        # 2b. CRITIC weights
        critic_result = self.critic_calc.calculate(X_df)
        W_c = np.array([critic_result.weights[c] for c in criteria_cols])
        logger.info(f"Step 2b: CRITIC — range [{W_c.min():.4f}, {W_c.max():.4f}]")
        
        # 2c. MEREC weights
        merec_result = self.merec_calc.calculate(X_df)
        W_m = np.array([merec_result.weights[c] for c in criteria_cols])
        logger.info(f"Step 2c: MEREC — range [{W_m.min():.4f}, {W_m.max():.4f}]")
        
        # 2d. Standard Deviation weights
        sd_result = self.sd_calc.calculate(X_df)
        W_s = np.array([sd_result.weights[c] for c in criteria_cols])
        logger.info(f"Step 2d: Std Dev — range [{W_s.min():.4f}, {W_s.max():.4f}]")

        # ── Step 3: Weight Fusion ──
        weight_vectors = {
            'entropy': W_e,
            'critic': W_c,
            'merec': W_m,
            'std_dev': W_s
        }
        
        # Game Theory Weight Combination (GTWC)
        W_fused, fusion_details = self.gtwc.combine(weight_vectors)
        
        logger.info(f"Step 3: GTWC Fusion — range [{W_fused.min():.4f}, {W_fused.max():.4f}]")
        logger.info(
            f"  α_dispersion={fusion_details['phase_3']['alpha_dispersion']:.4f}, "
            f"α_interaction={fusion_details['phase_3']['alpha_interaction']:.4f}"
        )
        logger.info(
            f"  Group cosine similarity: "
            f"{fusion_details['phase_2']['group_cosine_similarity']:.4f}"
        )

        # ── Step 4: Bayesian Bootstrap Validation ──
        def compute_fused_weights(X_df: pd.DataFrame, cols: List[str],
                                  sample_weights: 'np.ndarray | None' = None) -> np.ndarray:
            """Helper: compute fused weights for bootstrap using GTWC.
            
            Parameters
            ----------
            X_df : pd.DataFrame
                Full (unsampled) data matrix.
            cols : list[str]
                Criterion column names.
            sample_weights : np.ndarray or None
                Dirichlet observation weights forwarded from the Bayesian
                Bootstrap loop.  Passed through to each weight calculator
                so that weighted statistics are computed without discrete
                resampling.
            """
            W_e = np.array([self.entropy_calc.calculate(X_df, sample_weights=sample_weights).weights[c] for c in cols])
            W_c = np.array([self.critic_calc.calculate(X_df, sample_weights=sample_weights).weights[c] for c in cols])
            W_m = np.array([self.merec_calc.calculate(X_df, sample_weights=sample_weights).weights[c] for c in cols])
            W_s = np.array([self.sd_calc.calculate(X_df, sample_weights=sample_weights).weights[c] for c in cols])
            
            wv = {'entropy': W_e, 'critic': W_c, 'merec': W_m, 'std_dev': W_s}
            W_fused, _ = self.gtwc.combine(wv)
            return W_fused
        
        bootstrap_results = bayesian_bootstrap_weights(
            X_norm=X_norm,
            criteria_cols=criteria_cols,
            weight_calculator=compute_fused_weights,
            n_iterations=self.bootstrap_iterations,
            seed=self.seed,
            epsilon=self.epsilon
        )
        
        W_final = bootstrap_results["mean_weights"]
        logger.info(f"Step 4: Bootstrap ({self.bootstrap_iterations} iter) — "
                    f"mean std: {bootstrap_results['std_weights'].mean():.6f}")

        # ── Step 5: Temporal Stability Verification ──
        if time_col not in panel_data.columns:
            # Cross-sectional data has no time dimension — skip stability check
            logger.warning(
                "Step 5: time_col '%s' not found in panel_data; "
                "temporal stability verification skipped.", time_col
            )
            stability = {
                'cosine_similarity': None,
                'pearson_correlation': None,
                'is_stable': None,
                'split_point': None,
                'note': f"time_col '{time_col}' not present — cross-sectional data",
            }
        else:
            # Build a temporary DataFrame for split-half verification
            stability_df = panel_data[[time_col] + criteria_cols].copy()

            def compute_weights_from_df(df_half):
                """Helper: normalize subset and compute fused weights."""
                X_half = df_half[criteria_cols].values.astype(float)
                X_half_norm = global_min_max_normalize(X_half, epsilon=self.epsilon)
                X_half_df = pd.DataFrame(X_half_norm, columns=criteria_cols)
                return compute_fused_weights(X_half_df, criteria_cols)

            _stab = temporal_stability_verification(
                panel_df=stability_df,
                weight_calculator=compute_weights_from_df,
                entity_col=entity_col,
                time_col=time_col,
                criteria_cols=criteria_cols,
                threshold=self.stability_threshold,
            )

            # Convert StabilityResult dataclass → dict for downstream consumption
            stability = {
                'cosine_similarity': _stab.cosine_similarity,
                'pearson_correlation': _stab.correlation,
                'is_stable': _stab.is_stable,
                'split_point': _stab.split_point,
            }
        
        cos_v  = stability.get('cosine_similarity')
        pear_v = stability.get('pearson_correlation')
        cos_str  = f"{cos_v:.4f}"  if cos_v  is not None else "N/A"
        pear_str = f"{pear_v:.4f}" if pear_v is not None else "N/A"
        logger.info(f"Step 5: Stability — cosine={cos_str}, pearson={pear_str}")

        # Build result dictionary
        weights_dict = {col: float(W_final[j])
                        for j, col in enumerate(criteria_cols)}

        details = {
            # Individual weight vectors
            "individual_weights": {
                "entropy": {col: float(W_e[j]) for j, col in enumerate(criteria_cols)},
                "critic": {col: float(W_c[j]) for j, col in enumerate(criteria_cols)},
                "merec": {col: float(W_m[j]) for j, col in enumerate(criteria_cols)},
                "std_dev": {col: float(W_s[j]) for j, col in enumerate(criteria_cols)},
                "fused": {col: float(W_fused[j]) for j, col in enumerate(criteria_cols)},
            },
            
            # Fusion details
            "fusion": fusion_details,
            "fusion_method": "gtwc",
            
            # Bootstrap statistics
            "bootstrap": {
                "iterations": self.bootstrap_iterations,
                "mean_weights": {col: float(W_final[j]) for j, col in enumerate(criteria_cols)},
                "std_weights": {col: float(bootstrap_results["std_weights"][j])
                                for j, col in enumerate(criteria_cols)},
                "ci_lower_2_5": {col: float(bootstrap_results["ci_lower"][j])
                                 for j, col in enumerate(criteria_cols)},
                "ci_upper_97_5": {col: float(bootstrap_results["ci_upper"][j])
                                  for j, col in enumerate(criteria_cols)},
                "convergence_rate": float(bootstrap_results["convergence_rate"]),
            },
            
            # Stability verification
            "stability": stability,
            
            # Metadata
            "n_observations": n_obs,
            "n_criteria": n_criteria,
        }

        return WeightResult(
            weights=weights_dict,
            method="hybrid_weighting_pipeline",
            details=details,
        )
