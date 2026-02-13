# -*- coding: utf-8 -*-
"""
IFS-COPRAS: Intuitionistic Fuzzy COPRAS

Extends COPRAS using IFS score functions to separate benefit and cost
contributions under uncertainty.

Steps
-----
1. Weighted IFS score matrix:  V_ij = w_j × S(IFN_ij).
2. S⁺_i = sum of V_ij over benefit criteria.
3. S⁻_i = sum of |V_ij| over cost criteria.
4. Relative significance
       Q_i = S⁺_i + (S⁻_min × ΣS⁻) / (S⁻_i × Σ(S⁻_min/S⁻_k))
5. Utility degree  N_i = (Q_i / Q_max) × 100%.

References
----------
[1] Razavi Hajiagha, S.H. et al. (2013). "Extensions of COPRAS method
    with interval-valued intuitionistic fuzzy numbers."
    Journal of Applied Mathematics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from .base import IFN, IFSDecisionMatrix
from ...weighting import WeightResult


@dataclass
class IFS_COPRASResult:
    """Result container for IFS-COPRAS."""
    S_plus: pd.Series
    S_minus: pd.Series
    Q: pd.Series
    utility_degree: pd.Series
    ranks: pd.Series
    weights: Dict[str, float]

    @property
    def final_ranks(self) -> pd.Series:
        return self.ranks

    def top_n(self, n: int = 10) -> pd.DataFrame:
        return pd.DataFrame({
            'S+': self.S_plus, 'S-': self.S_minus,
            'Q': self.Q, 'Utility_%': self.utility_degree,
            'Rank': self.ranks,
        }).nsmallest(n, 'Rank')


class IFS_COPRAS:
    """
    IFS-COPRAS calculator.

    Parameters
    ----------
    cost_criteria : list, optional
        Criteria where lower values are preferred.
    """

    def __init__(self, cost_criteria: Optional[List[str]] = None):
        self.cost_criteria = cost_criteria or []

    def calculate(self,
                  ifs_matrix: IFSDecisionMatrix,
                  weights: Union[Dict[str, float], WeightResult, None] = None,
                  ) -> IFS_COPRASResult:

        criteria = ifs_matrix.criteria
        alternatives = ifs_matrix.alternatives
        w = self._resolve_weights(weights, criteria)

        # Score matrix
        score_df = ifs_matrix.to_score_matrix()

        # Normalise scores to [0, 1] for COPRAS sum-normalisation
        col_sums = score_df.abs().sum(axis=0)
        col_sums[col_sums == 0] = 1
        norm = score_df / col_sums

        # Weighted normalised
        w_arr = np.array([w[c] for c in criteria])
        weighted = norm.values * w_arr

        benefit_mask = np.array([c not in self.cost_criteria for c in criteria])
        cost_mask = ~benefit_mask

        s_plus = weighted[:, benefit_mask].sum(axis=1) if benefit_mask.any() else np.zeros(len(alternatives))
        s_minus = np.abs(weighted[:, cost_mask]).sum(axis=1) if cost_mask.any() else np.zeros(len(alternatives))

        # Avoid zero division
        s_minus_safe = np.where(s_minus == 0, 1e-12, s_minus)
        s_minus_min = s_minus_safe.min()
        inv_sum = np.sum(s_minus_min / s_minus_safe)

        Q = s_plus + (s_minus_min * s_minus_safe.sum()) / (s_minus_safe * inv_sum + 1e-12)

        q_max = Q.max() if Q.max() > 0 else 1.0
        utility = (Q / q_max) * 100

        S_plus_s = pd.Series(s_plus, index=alternatives, name='S+')
        S_minus_s = pd.Series(s_minus, index=alternatives, name='S-')
        Q_s = pd.Series(Q, index=alternatives, name='Q')
        util_s = pd.Series(utility, index=alternatives, name='Utility_%')
        ranks = Q_s.rank(ascending=False).astype(int)
        ranks.name = 'IFS_COPRAS_Rank'

        return IFS_COPRASResult(
            S_plus=S_plus_s, S_minus=S_minus_s,
            Q=Q_s, utility_degree=util_s,
            ranks=ranks, weights=w,
        )

    @staticmethod
    def _resolve_weights(weights, criteria):
        if weights is None:
            return {c: 1.0 / len(criteria) for c in criteria}
        if isinstance(weights, WeightResult):
            weights = weights.weights
        return {c: weights.get(c, 1.0 / len(criteria)) for c in criteria}
