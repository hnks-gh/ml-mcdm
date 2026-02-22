# -*- coding: utf-8 -*-
"""
Advanced weight fusion methods for combining multiple objective weighting methods.

Implements:
- GameTheoryWeightCombination (GTWC): State-of-the-art cooperative game-theoretic
  fusion with intra-group hybridization and Nash equilibrium optimization.

See technical documentation for detailed formulas and references.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class GameTheoryWeightCombination:
    """
    Game Theory Weight Combination (GTWC) for cooperative weight fusion.
    
    Combines four objective weighting methods through a principled 3-phase
    pipeline that prevents variance bias and achieves Nash equilibrium:
    
    **Phase 1 (external):** Calculate base weights (Entropy, SD, CRITIC, MEREC)
    
    **Phase 2 — Intra-Group Hybridization:**
    Clusters methods by measurement philosophy to prevent redundancy:
    
    - *Group A (Dispersion Camp):* Geometric mean of Entropy + SD.
      Both measure within-criterion variance; geometric mean amplifies
      shared signals while preventing zero-dominance.
    - *Group B (Interaction Camp):* Harmonic mean of CRITIC + MEREC.
      Both measure between-criterion relationships; harmonic mean
      ensures a criterion is favored only if BOTH methods agree.
    
    **Phase 3 — Game Theory Optimization:**
    Treats Group A and Group B as cooperative game players. Finds
    optimal coefficients (α₁, α₂) that minimize the combined L2-distance
    to both groups, solved via Lagrange multipliers.
    
    **Phase 4 — Final Aggregation:**
    W* = α₁ · W_GroupA + α₂ · W_GroupB
    
    Parameters
    ----------
    epsilon : float, default=1e-10
        Numerical stability constant for avoiding division by zero.
    
    References
    ----------
    1. Nash (1950). The Bargaining Problem. Econometrica, 18(2), 155-162.
    2. Shapley (1953). A Value for n-Person Games.
       Contributions to the Theory of Games, 2, 307-317.
    3. Ding & Shi (2005). Game theory approach to discrete multicriteria
       analysis. European Journal of Operational Research, 166(3), 838-848.
    
    Examples
    --------
    >>> import numpy as np
    >>> from weighting.fusion import GameTheoryWeightCombination
    >>> 
    >>> gtwc = GameTheoryWeightCombination()
    >>> weight_vectors = {
    ...     'entropy': np.array([0.20, 0.30, 0.50]),
    ...     'std_dev': np.array([0.25, 0.35, 0.40]),
    ...     'critic':  np.array([0.30, 0.40, 0.30]),
    ...     'merec':   np.array([0.40, 0.30, 0.30]),
    ... }
    >>> W_final, details = gtwc.combine(weight_vectors)
    >>> print(f"Final weights: {W_final}")
    >>> print(f"α_dispersion={details['phase_3']['alpha_dispersion']:.3f}")
    >>> print(f"α_interaction={details['phase_3']['alpha_interaction']:.3f}")
    """
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
    
    def _normalize(self, w: np.ndarray) -> np.ndarray:
        """Normalize weight vector to sum to 1 with positivity guarantee."""
        w = np.clip(w, self.epsilon, None)
        return w / w.sum()
    
    def _geometric_mean_fusion(
        self, w1: np.ndarray, w2: np.ndarray
    ) -> np.ndarray:
        """
        Geometric mean fusion for Dispersion Camp (Group A).
        
        Uses √(w₁ · w₂) to amplify shared signals while being more
        balanced than raw element-wise product — prevents a near-zero
        weight in one method from destroying importance in the other.
        
        Parameters
        ----------
        w1 : np.ndarray
            First weight vector (e.g., Entropy weights).
        w2 : np.ndarray
            Second weight vector (e.g., SD weights).
        
        Returns
        -------
        np.ndarray
            Normalized geometric mean weight vector.
        """
        w1_safe = np.clip(w1, self.epsilon, None)
        w2_safe = np.clip(w2, self.epsilon, None)
        return self._normalize(np.sqrt(w1_safe * w2_safe))
    
    def _harmonic_mean_fusion(
        self, w1: np.ndarray, w2: np.ndarray
    ) -> np.ndarray:
        """
        Harmonic mean fusion for Interaction Camp (Group B).
        
        H = 2 / (1/w₁ + 1/w₂)
        
        The harmonic mean is conservative: a criterion receives high
        weight only if BOTH methods agree on its importance. A low
        weight from either method pulls the result down.
        
        Parameters
        ----------
        w1 : np.ndarray
            First weight vector (e.g., CRITIC weights).
        w2 : np.ndarray
            Second weight vector (e.g., MEREC weights).
        
        Returns
        -------
        np.ndarray
            Normalized harmonic mean weight vector.
        """
        w1_safe = np.clip(w1, self.epsilon, None)
        w2_safe = np.clip(w2, self.epsilon, None)
        harmonic = 2.0 / (1.0 / w1_safe + 1.0 / w2_safe)
        return self._normalize(harmonic)
    
    def _solve_game_theory(
        self, W_A: np.ndarray, W_B: np.ndarray
    ) -> tuple:
        """
        Solve cooperative game via Lagrange multipliers to find Nash
        equilibrium coefficients.
        
        Minimizes the combined L2-distance objective:
            L = ‖α₁W_A + α₂W_B − W_A‖² + ‖α₁W_A + α₂W_B − W_B‖²
        
        This yields the linear system:
            ⎡ W_A·W_A   W_A·W_B ⎤ ⎡α₁⎤   ⎡ W_A·W_A ⎤
            ⎣ W_B·W_A   W_B·W_B ⎦ ⎣α₂⎦ = ⎣ W_B·W_B ⎦
        
        where · denotes the dot product (scalar).
        
        Parameters
        ----------
        W_A : np.ndarray
            Group A super-weight vector (Dispersion Camp).
        W_B : np.ndarray
            Group B super-weight vector (Interaction Camp).
        
        Returns
        -------
        alpha_final : np.ndarray, shape (2,)
            Normalized non-negative equilibrium coefficients [α₁, α₂].
        alpha_raw : np.ndarray, shape (2,)
            Raw solution before clipping/normalization.
        details : dict
            Comprehensive solution diagnostics.
        """
        # Construct the 2×2 linear system
        # From ∇L = 0 where L = ||α₁W_A + α₂W_B - W_A||² + ||α₁W_A + α₂W_B - W_B||²
        # Gradient equations:
        #   2α₁(W_A·W_A) + 2α₂(W_A·W_B) = (W_A·W_A) + (W_A·W_B)
        #   2α₁(W_A·W_B) + 2α₂(W_B·W_B) = (W_A·W_B) + (W_B·W_B)
        # Or equivalently (dividing by 2):
        #   α₁(W_A·W_A) + α₂(W_A·W_B) = (W_A·W_A + W_A·W_B) / 2
        #   α₁(W_A·W_B) + α₂(W_B·W_B) = (W_A·W_B + W_B·W_B) / 2
        
        dot_AA = np.dot(W_A, W_A)
        dot_AB = np.dot(W_A, W_B)
        dot_BB = np.dot(W_B, W_B)
        
        A_matrix = np.array([
            [dot_AA, dot_AB],
            [dot_AB, dot_BB]
        ])
        
        # Corrected RHS: b = [(dot_AA + dot_AB)/2, (dot_AB + dot_BB)/2]
        # Previously was: b = [dot_AA, dot_BB] which is WRONG
        b_vector = np.array([dot_AA + dot_AB, dot_AB + dot_BB]) / 2.0
        
        # Solve A @ α = b
        cond_number = np.linalg.cond(A_matrix)
        
        try:
            alpha_raw = np.linalg.solve(A_matrix, b_vector)
        except np.linalg.LinAlgError:
            logger.warning(
                "GTWC linear system is singular (cond=%.2e). "
                "Falling back to equal coefficients.",
                cond_number
            )
            alpha_raw = np.array([0.5, 0.5])
        
        # Ensure non-negative and normalize to sum to 1
        alpha_clipped = np.clip(alpha_raw, 0, None)
        alpha_sum = alpha_clipped.sum()
        
        if alpha_sum < self.epsilon:
            logger.warning(
                "All GTWC coefficients clipped to zero. "
                "Using equal weights as fallback."
            )
            alpha_final = np.array([0.5, 0.5])
        else:
            alpha_final = alpha_clipped / alpha_sum
        
        details = {
            "dot_products": {
                "W_A·W_A": float(dot_AA),
                "W_A·W_B": float(dot_AB),
                "W_B·W_B": float(dot_BB),
            },
            "system_matrix": A_matrix.tolist(),
            "system_rhs": b_vector.tolist(),
            "alpha_raw": alpha_raw.tolist(),
            "alpha_final": alpha_final.tolist(),
            "condition_number": float(cond_number),
        }
        
        return alpha_final, alpha_raw, details
    
    def combine(
        self,
        weight_vectors: Dict[str, np.ndarray],
    ) -> tuple:
        """
        Execute the full GTWC fusion pipeline (Phases 2–4).
        
        Parameters
        ----------
        weight_vectors : Dict[str, np.ndarray]
            Must contain keys: ``'entropy'``, ``'std_dev'``, ``'critic'``,
            ``'merec'``. Each value is a 1-D array of criterion weights
            that sums to 1.
        
        Returns
        -------
        W_final : np.ndarray
            Game-theoretic integrated weight vector (sums to 1).
        gtwc_details : dict
            Comprehensive details for all phases:
            - ``phase_2``: group compositions and super-weight vectors
            - ``phase_3``: game-theory solution diagnostics and α values
            - ``phase_4``: final aggregated weight vector
        
        Raises
        ------
        KeyError
            If any of the four required weight vectors is missing.
        """
        # Extract the four base weight vectors
        required_keys = {'entropy', 'std_dev', 'critic', 'merec'}
        missing = required_keys - set(weight_vectors.keys())
        if missing:
            raise KeyError(
                f"Missing required weight vectors: {missing}. "
                f"GTWC requires: {required_keys}"
            )
        
        W_entropy = weight_vectors['entropy']
        W_sd = weight_vectors['std_dev']
        W_critic = weight_vectors['critic']
        W_merec = weight_vectors['merec']
        
        n_criteria = len(W_entropy)
        logger.info(
            "GTWC Phase 2: Intra-group hybridization (%d criteria)", n_criteria
        )
        
        # ── Phase 2: Intra-Group Hybridization ──
        # Group A (Dispersion): Geometric mean of Entropy + SD
        W_GroupA = self._geometric_mean_fusion(W_entropy, W_sd)
        
        # Group B (Interaction): Harmonic mean of CRITIC + MEREC
        W_GroupB = self._harmonic_mean_fusion(W_critic, W_merec)
        
        logger.info(
            "  Group A (Dispersion): range [%.4f, %.4f]",
            W_GroupA.min(), W_GroupA.max()
        )
        logger.info(
            "  Group B (Interaction): range [%.4f, %.4f]",
            W_GroupB.min(), W_GroupB.max()
        )
        
        # ── Phase 3: Game Theory Weight Combination ──
        logger.info("GTWC Phase 3: Solving cooperative game for α coefficients")
        alpha_final, alpha_raw, gt_details = self._solve_game_theory(
            W_GroupA, W_GroupB
        )
        
        logger.info(
            "  α_dispersion=%.4f, α_interaction=%.4f (cond=%.2e)",
            alpha_final[0], alpha_final[1], gt_details['condition_number']
        )
        
        # ── Phase 4: Final Aggregation ──
        W_final = alpha_final[0] * W_GroupA + alpha_final[1] * W_GroupB
        W_final = self._normalize(W_final)
        
        logger.info(
            "GTWC Phase 4: Final weights range [%.4f, %.4f]",
            W_final.min(), W_final.max()
        )
        
        # Compile comprehensive details
        gtwc_details = {
            "method": "game_theory_weight_combination",
            "phase_2": {
                "group_A_method": "geometric_mean(entropy, std_dev)",
                "group_A_rationale": "Amplifies shared dispersion signals "
                                    "via sqrt(W_entropy * W_sd)",
                "group_B_method": "harmonic_mean(critic, merec)",
                "group_B_rationale": "Conservative fusion — criterion favored "
                                    "only if both interaction methods agree",
                "W_GroupA": W_GroupA.tolist(),
                "W_GroupB": W_GroupB.tolist(),
                "group_cosine_similarity": float(
                    np.dot(W_GroupA, W_GroupB) /
                    (np.linalg.norm(W_GroupA) * np.linalg.norm(W_GroupB) + self.epsilon)
                ),
            },
            "phase_3": {
                "alpha_dispersion": float(alpha_final[0]),
                "alpha_interaction": float(alpha_final[1]),
                **gt_details,
            },
            "phase_4": {
                "W_final": W_final.tolist(),
            },
        }
        
        return W_final, gtwc_details
