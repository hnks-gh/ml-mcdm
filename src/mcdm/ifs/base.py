# -*- coding: utf-8 -*-
"""
Intuitionistic Fuzzy Set (IFS) Base Classes
============================================

Mathematical Foundation (Atanassov, 1986):
    An Intuitionistic Fuzzy Set A on a universe X is defined as:
        A = {⟨x, μ_A(x), ν_A(x)⟩ | x ∈ X}

    where:
        μ_A(x) ∈ [0, 1]  — membership degree (degree of belongingness)
        ν_A(x) ∈ [0, 1]  — non-membership degree (degree of non-belongingness)
        π_A(x) = 1 - μ_A(x) - ν_A(x)  — hesitancy degree (indeterminacy)

    Constraint: 0 ≤ μ_A(x) + ν_A(x) ≤ 1

References
----------
[1] Atanassov, K.T. (1986). "Intuitionistic Fuzzy Sets."
    Fuzzy Sets and Systems, 20(1), 87–96.
[2] Xu, Z. (2007). "Intuitionistic Fuzzy Aggregation Operators."
    IEEE Transactions on Fuzzy Systems, 15(6), 1179–1187.
[3] Szmidt, E. & Kacprzyk, J. (2000). "Distances Between Intuitionistic
    Fuzzy Sets." Fuzzy Sets and Systems, 114(3), 505–518.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class IFN:
    """
    Intuitionistic Fuzzy Number.

    Represents a value with membership (μ), non-membership (ν), and
    hesitancy (π = 1 - μ - ν) degrees.

    Parameters
    ----------
    mu : float
        Membership degree ∈ [0, 1].
    nu : float
        Non-membership degree ∈ [0, 1].

    Properties
    ----------
    pi : float
        Hesitancy degree π = 1 − μ − ν.

    Raises
    ------
    ValueError
        If μ or ν not in [0, 1] or μ + ν > 1 + ε.
    """

    mu: float
    nu: float

    def __post_init__(self):
        eps = 1e-9
        self.mu = float(np.clip(self.mu, 0.0, 1.0))
        self.nu = float(np.clip(self.nu, 0.0, 1.0))
        if self.mu + self.nu > 1.0 + eps:
            # Normalize to satisfy constraint
            total = self.mu + self.nu
            self.mu /= total
            self.nu /= total

    @property
    def pi(self) -> float:
        """Hesitancy degree: π = 1 − μ − ν."""
        return max(0.0, 1.0 - self.mu - self.nu)

    # ------------------------------------------------------------------
    # Scoring & ranking helpers
    # ------------------------------------------------------------------
    def score(self) -> float:
        """Score function S(A) = μ − ν  ∈ [−1, 1]. (Chen & Tan, 1994)"""
        return self.mu - self.nu

    def accuracy(self) -> float:
        """Accuracy function H(A) = μ + ν  ∈ [0, 1]. (Hong & Choi, 2000)"""
        return self.mu + self.nu

    def score_xu(self) -> float:
        """Xu & Yager (2006) score: S(A) = μ − ν."""
        return self.score()

    # ------------------------------------------------------------------
    # Arithmetic operators  (Xu, 2007)
    # ------------------------------------------------------------------
    def __add__(self, other: 'IFN') -> 'IFN':
        """IFS addition: μ₁+μ₂−μ₁μ₂ , ν₁ν₂."""
        return IFN(
            mu=self.mu + other.mu - self.mu * other.mu,
            nu=self.nu * other.nu,
        )

    def __mul__(self, other: Union['IFN', float]) -> 'IFN':
        if isinstance(other, IFN):
            return IFN(mu=self.mu * other.mu,
                       nu=self.nu + other.nu - self.nu * other.nu)
        lam = float(other)
        if lam < 0:
            raise ValueError("Scalar multiplier must be ≥ 0 for IFS.")
        return IFN(mu=1 - (1 - self.mu) ** lam,
                   nu=self.nu ** lam)

    def __rmul__(self, scalar: float) -> 'IFN':
        return self.__mul__(scalar)

    def power(self, n: float) -> 'IFN':
        """IFS power: Aⁿ = (μⁿ, 1−(1−ν)ⁿ)."""
        return IFN(mu=self.mu ** n, nu=1 - (1 - self.nu) ** n)

    # ------------------------------------------------------------------
    # Distance measures  (Szmidt & Kacprzyk, 2000)
    # ------------------------------------------------------------------
    @staticmethod
    def euclidean_distance(a: 'IFN', b: 'IFN') -> float:
        """Normalized Euclidean distance between two IFNs."""
        return np.sqrt(((a.mu - b.mu) ** 2
                        + (a.nu - b.nu) ** 2
                        + (a.pi - b.pi) ** 2) / 2)

    @staticmethod
    def hamming_distance(a: 'IFN', b: 'IFN') -> float:
        """Normalized Hamming distance between two IFNs."""
        return (abs(a.mu - b.mu) + abs(a.nu - b.nu) + abs(a.pi - b.pi)) / 2

    @staticmethod
    def normalized_euclidean(a: 'IFN', b: 'IFN') -> float:
        """Szmidt–Kacprzyk normalized Euclidean distance."""
        return np.sqrt((
            (a.mu - b.mu) ** 2
            + (a.nu - b.nu) ** 2
            + (a.pi - b.pi) ** 2
        ) / 6)

    # ------------------------------------------------------------------
    # Comparison (lexicographic: score first, then accuracy)
    # ------------------------------------------------------------------
    def __lt__(self, other: 'IFN') -> bool:
        if abs(self.score() - other.score()) > 1e-12:
            return self.score() < other.score()
        return self.accuracy() < other.accuracy()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IFN):
            return NotImplemented
        return (abs(self.mu - other.mu) < 1e-12
                and abs(self.nu - other.nu) < 1e-12)

    def __repr__(self) -> str:
        return f"IFN(μ={self.mu:.4f}, ν={self.nu:.4f}, π={self.pi:.4f})"


# =========================================================================
# IFS Decision Matrix
# =========================================================================
class IFSDecisionMatrix:
    """
    Decision matrix where each cell is an Intuitionistic Fuzzy Number.

    Stores an (m alternatives × n criteria) matrix of IFNs together with
    alternative labels and criterion labels.

    Parameters
    ----------
    matrix : Dict[str, Dict[str, IFN]]
        Nested dict  alternative → criterion → IFN.
    alternatives : List[str]
    criteria : List[str]
    """

    def __init__(self,
                 matrix: Dict[str, Dict[str, IFN]],
                 alternatives: List[str],
                 criteria: List[str]):
        self.matrix = matrix
        self.alternatives = alternatives
        self.criteria = criteria

    def get(self, alt: str, crit: str) -> IFN:
        return self.matrix[alt][crit]

    def to_score_matrix(self) -> pd.DataFrame:
        """Defuzzify every cell using the score function S = μ − ν."""
        data = {alt: {crit: self.matrix[alt][crit].score()
                      for crit in self.criteria}
                for alt in self.alternatives}
        return pd.DataFrame(data).T[self.criteria]

    def to_mu_matrix(self) -> pd.DataFrame:
        """Extract membership‑degree matrix."""
        data = {alt: {crit: self.matrix[alt][crit].mu
                      for crit in self.criteria}
                for alt in self.alternatives}
        return pd.DataFrame(data).T[self.criteria]

    def to_nu_matrix(self) -> pd.DataFrame:
        """Extract non‑membership‑degree matrix."""
        data = {alt: {crit: self.matrix[alt][crit].nu
                      for crit in self.criteria}
                for alt in self.alternatives}
        return pd.DataFrame(data).T[self.criteria]

    def to_pi_matrix(self) -> pd.DataFrame:
        """Extract hesitancy‑degree matrix."""
        data = {alt: {crit: self.matrix[alt][crit].pi
                      for crit in self.criteria}
                for alt in self.alternatives}
        return pd.DataFrame(data).T[self.criteria]

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @staticmethod
    def from_temporal_variance(
        current_data: pd.DataFrame,
        historical_std: pd.DataFrame,
        global_range: Optional[pd.Series] = None,
        spread_factor: float = 1.0,
    ) -> 'IFSDecisionMatrix':
        """
        Build an IFS decision matrix from crisp data + temporal uncertainty.

        Construction logic (per cell):
            μ_ij = x_ij  (already normalised to [0, 1])
            π_ij = min(σ_ij × spread_factor / R_j,  1 − μ_ij)
            ν_ij = 1 − μ_ij − π_ij

        where σ_ij is the historical standard deviation and R_j is the
        global range for criterion j.

        Parameters
        ----------
        current_data : pd.DataFrame
            Normalised crisp matrix (alternatives × criteria), values in [0, 1].
        historical_std : pd.DataFrame
            Standard deviations computed across years (same shape).
        global_range : pd.Series, optional
            Range per criterion for normalising the spread.  Defaults to 1.
        spread_factor : float
            Multiplier on σ to control hesitancy magnitude.

        Returns
        -------
        IFSDecisionMatrix
        """
        alternatives = current_data.index.tolist()
        criteria = current_data.columns.tolist()

        if global_range is None:
            global_range = pd.Series(1.0, index=criteria)

        matrix: Dict[str, Dict[str, IFN]] = {}
        for alt in alternatives:
            matrix[alt] = {}
            for crit in criteria:
                mu = float(np.clip(current_data.loc[alt, crit], 0, 1))
                sigma = float(historical_std.loc[alt, crit]) if alt in historical_std.index else 0.0
                rng = float(global_range[crit]) if global_range[crit] > 0 else 1.0

                pi = min(sigma * spread_factor / rng, 1.0 - mu)
                pi = max(pi, 0.0)
                nu = 1.0 - mu - pi

                matrix[alt][crit] = IFN(mu=mu, nu=nu)

        return IFSDecisionMatrix(matrix, alternatives, criteria)

    @staticmethod
    def from_crisp(data: pd.DataFrame, default_pi: float = 0.0) -> 'IFSDecisionMatrix':
        """
        Convert a crisp [0, 1] matrix with fixed hesitancy.

        Parameters
        ----------
        data : pd.DataFrame
            Normalised crisp matrix (values in [0, 1]).
        default_pi : float
            Fixed hesitancy degree assigned to every cell.
        """
        alternatives = data.index.tolist()
        criteria = data.columns.tolist()
        matrix: Dict[str, Dict[str, IFN]] = {}
        for alt in alternatives:
            matrix[alt] = {}
            for crit in criteria:
                mu = float(np.clip(data.loc[alt, crit], 0, 1))
                pi = min(default_pi, 1.0 - mu)
                nu = 1.0 - mu - pi
                matrix[alt][crit] = IFN(mu=mu, nu=nu)
        return IFSDecisionMatrix(matrix, alternatives, criteria)
