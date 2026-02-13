# -*- coding: utf-8 -*-
"""
Evidential Reasoning Base Classes
==================================

Implements the ER analytical algorithm (Yang & Singh, 1994;
Yang & Xu, 2002) for combining multiple sources of evidence in
multi-attribute decision analysis.

Mathematical Framework
---------------------
A belief distribution S(e_i) = {(H_n, β_n,i), n = 1..N}
expresses the assessor's degree of belief β that the alternative is
evaluated to grade H_n on attribute e_i.

The ER combination rule aggregates L attributes with weights w_i:

    m̂_I(H_n) = K × [∏_{i=1}^{L} (w_i β_{n,i} + 1 - w_i Σ_j β_{j,i})
                      - ∏_{i=1}^{L} (1 - w_i Σ_j β_{j,i})]

    K = [1 - ∏_{i=1}^{L} (1 - w_i Σ_j β_{j,i})]⁻¹       (1)

Unassigned belief mass:
    m̂_I(H) = K × ∏_{i=1}^{L} (1 - w_i)                   (2a)
    m̃_I(H) = K × [∏_{i=1}^{L} (1 - w_i Σ_j β_{j,i})
                   - ∏_{i=1}^{L} (1 - w_i)]                (2b)

Final belief:
    β_n  = m̂_I(H_n) / (1 - m̃_I(H))                       (3)
    β_H  = m̂_I(H)   / (1 - m̃_I(H))                       residual

References
----------
[1] Yang, J.B. & Singh, M.G. (1994). "An Evidential Reasoning Approach
    for Multiple-Attribute Decision Making with Uncertainty."
    IEEE Trans. Systems, Man, and Cybernetics, 24(1), 1–18.
[2] Yang, J.B. & Xu, D.L. (2002). "On the evidential reasoning algorithm
    for multiple attribute decision analysis under uncertainty."
    IEEE Trans. Systems, Man, and Cybernetics – Part A, 32(3), 289–304.
[3] Yang, J.B. & Xu, D.L. (2013). "Evidential reasoning rule for evidence
    combination." Artificial Intelligence, 205, 1–29.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# =========================================================================
# Data structures
# =========================================================================

@dataclass
class BeliefDistribution:
    """
    Belief distribution over N evaluation grades.

    Attributes
    ----------
    grades : list of str
        Ordered grade labels (best → worst), e.g.
        ['Excellent', 'Good', 'Fair', 'Poor', 'Bad'].
    beliefs : np.ndarray  (length N)
        β_n  for n = 1..N.   Σ β_n ≤ 1.
    """

    grades: List[str]
    beliefs: np.ndarray

    def __post_init__(self):
        self.beliefs = np.asarray(self.beliefs, dtype=float)
        self.beliefs = np.clip(self.beliefs, 0, 1)
        total = self.beliefs.sum()
        if total > 1.0 + 1e-9:
            self.beliefs /= total

    @property
    def unassigned(self) -> float:
        """Residual ignorance: β_H = 1 − Σβ_n."""
        return max(0.0, 1.0 - self.beliefs.sum())

    @property
    def n_grades(self) -> int:
        return len(self.grades)

    # ------------------------------------------------------------------
    # Utility scoring
    # ------------------------------------------------------------------
    def expected_utility(self, utilities: Optional[np.ndarray] = None) -> float:
        """
        Expected utility  E[u] = Σ β_n × u(H_n).

        Parameters
        ----------
        utilities : array, optional
            Utility of each grade ordered best→worst.
            Defaults to linspace(1, 0, N).
        """
        if utilities is None:
            utilities = np.linspace(1.0, 0.0, self.n_grades)
        return float(np.dot(self.beliefs, utilities))

    def utility_interval(self, utilities: Optional[np.ndarray] = None
                         ) -> Tuple[float, float]:
        """
        [u_min, u_max] interval accounting for unassigned belief.

        u_min: unassigned mass assigned to worst grade.
        u_max: unassigned mass assigned to best grade.
        """
        if utilities is None:
            utilities = np.linspace(1.0, 0.0, self.n_grades)
        base = float(np.dot(self.beliefs, utilities))
        h = self.unassigned
        u_min = base + h * utilities[-1]   # assign to worst
        u_max = base + h * utilities[0]    # assign to best
        return u_min, u_max

    def average_utility(self, utilities: Optional[np.ndarray] = None) -> float:
        """Mid-point of the utility interval (default ranking metric)."""
        lo, hi = self.utility_interval(utilities)
        return (lo + hi) / 2.0

    def belief_entropy(self) -> float:
        """Shannon entropy of the belief distribution (uncertainty)."""
        p = self.beliefs[self.beliefs > 0]
        return -float(np.sum(p * np.log2(p))) if len(p) > 0 else 0.0

    def __repr__(self) -> str:
        parts = ", ".join(f"{g}={b:.3f}" for g, b in zip(self.grades, self.beliefs))
        return f"Belief({parts}, H={self.unassigned:.3f})"


# =========================================================================
# ER combination engine
# =========================================================================

class EvidentialReasoningEngine:
    """
    Analytical ER algorithm (Yang & Xu, 2002).

    Combines L evidence sources (attributes / methods) each characterised
    by a belief distribution over N grades, using reliability weights.
    """

    def __init__(self, grades: Optional[List[str]] = None):
        self.grades = grades or ['Excellent', 'Good', 'Fair', 'Poor', 'Bad']

    # ------------------------------------------------------------------
    # Core ER combination
    # ------------------------------------------------------------------
    def combine(self,
                belief_distributions: List[BeliefDistribution],
                weights: np.ndarray,
                ) -> BeliefDistribution:
        """
        ER analytical combination.

        Parameters
        ----------
        belief_distributions : list of BeliefDistribution
            One per attribute / evidence source.
        weights : np.ndarray  (length L)
            Relative importance weights (need not sum to 1; will be
            normalised internally).

        Returns
        -------
        BeliefDistribution
            Combined belief distribution.
        """
        L = len(belief_distributions)
        N = len(self.grades)
        weights = np.asarray(weights, dtype=float)

        # Normalise weights to sum to 1
        w = weights / weights.sum() if weights.sum() > 0 else np.ones(L) / L

        # Build  β matrix: (L × N)
        beta = np.zeros((L, N))
        for i, bd in enumerate(belief_distributions):
            beta[i, :N] = bd.beliefs[:N]

        # Σ_j β_{j,i}  per source
        beta_sum = beta.sum(axis=1)                 # (L,)

        # Products in ER formula (Eqs 1, 2a, 2b)
        # For numerical stability, accumulate in log-space when possible.
        # But small L (≤ 20) is fine with direct multiplication.

        # Pre-compute per-source quantities
        # A_{n,i} = w_i β_{n,i} + 1 - w_i Σ_j β_{j,i}
        # B_i     = 1 - w_i Σ_j β_{j,i}
        # C_i     = 1 - w_i

        A = np.zeros((N, L))          # A[n, i]
        B = np.zeros(L)
        C = np.zeros(L)

        for i in range(L):
            for n in range(N):
                A[n, i] = w[i] * beta[i, n] + 1 - w[i] * beta_sum[i]
            B[i] = 1 - w[i] * beta_sum[i]
            C[i] = 1 - w[i]

        # Product terms
        prod_A = np.prod(A, axis=1)    # (N,)  — ∏_i A_{n,i}
        prod_B = np.prod(B)            # scalar — ∏_i B_i
        prod_C = np.prod(C)            # scalar — ∏_i C_i

        # K  (normalisation constant)
        denom = 1.0 - prod_B
        if abs(denom) < 1e-15:
            # Degenerate case: all evidence is fully unassigned
            return BeliefDistribution(self.grades, np.zeros(N))
        K = 1.0 / denom

        # m̂_I(H_n)  — mass of grade n   (Eq 1)
        m_hat = K * (prod_A - prod_B)                  # (N,)

        # m̂_I(H)   — residual from weight incompleteness (Eq 2a)
        m_hat_H = K * prod_C

        # m̃_I(H)   — residual from belief incompleteness (Eq 2b)
        m_tilde_H = K * (prod_B - prod_C)

        # Final belief degrees  (Eq 3)
        denom_final = 1.0 - m_tilde_H
        if abs(denom_final) < 1e-15:
            denom_final = 1e-15
        final_beta = m_hat / denom_final
        # Residual belief (ignorance due to weight incompleteness)
        # beta_H = m_hat_H / denom_final  (available but not stored separately)

        final_beta = np.clip(final_beta, 0, 1)
        # Renormalise if slightly > 1 due to numerical drift
        total = final_beta.sum()
        if total > 1.0:
            final_beta /= total

        return BeliefDistribution(self.grades, final_beta)

    # ------------------------------------------------------------------
    # Score-to-belief conversion
    # ------------------------------------------------------------------
    def score_to_belief(self,
                        score: float,
                        method: str = 'linear_interpolation',
                        ) -> BeliefDistribution:
        """
        Convert a crisp score in [0, 1] to a belief distribution over grades.

        Method: linear_interpolation
        ----------------------------
        Grade utilities (u_k) are evenly spaced from 1 (best) to 0 (worst).
        The score falls between two adjacent grades k, k+1.
        Belief is linearly split between them:
            β_k   = (score − u_{k+1}) / (u_k − u_{k+1})
            β_{k+1} = 1 − β_k
        All other grades receive zero belief.

        Parameters
        ----------
        score : float
            Value in [0, 1].
        method : str
            Conversion method (only 'linear_interpolation' implemented).

        Returns
        -------
        BeliefDistribution
        """
        N = len(self.grades)
        utilities = np.linspace(1.0, 0.0, N)
        score = float(np.clip(score, 0.0, 1.0))

        beliefs = np.zeros(N)

        if score >= utilities[0]:           # above best grade
            beliefs[0] = 1.0
        elif score <= utilities[-1]:        # below worst grade
            beliefs[-1] = 1.0
        else:
            for k in range(N - 1):
                if utilities[k] >= score >= utilities[k + 1]:
                    span = utilities[k] - utilities[k + 1]
                    if span > 0:
                        beliefs[k] = (score - utilities[k + 1]) / span
                        beliefs[k + 1] = 1.0 - beliefs[k]
                    else:
                        beliefs[k] = 1.0
                    break

        return BeliefDistribution(self.grades, beliefs)

    def rank_to_score(self,
                      rank: int,
                      n_alternatives: int) -> float:
        """Convert a 1-based rank to a utility score in [0, 1]."""
        if n_alternatives <= 1:
            return 1.0
        return 1.0 - (rank - 1) / (n_alternatives - 1)
