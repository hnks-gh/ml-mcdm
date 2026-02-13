# -*- coding: utf-8 -*-
"""
IFS-PROMETHEE II: Intuitionistic Fuzzy PROMETHEE

Extends PROMETHEE II by defining pairwise preference degrees on
IFS score values and incorporating hesitancy-weighted preference
functions.

Steps
-----
1. Compute score differences  d_j(a, b) = S(a_j) − S(b_j).
2. Apply V-shape preference function  P_j(a, b) = max(0, d_j) / p_j.
3. Aggregated preference  π(a, b) = Σ_j w_j × P_j(a, b).
4. Positive flow  Φ⁺(a) = (1/(n−1)) Σ_b π(a, b).
5. Negative flow  Φ⁻(a) = (1/(n−1)) Σ_b π(b, a).
6. Net flow  Φ(a) = Φ⁺(a) − Φ⁻(a).

References
----------
[1] Liao, H. & Xu, Z. (2014). "Multi-criteria decision making with
    intuitionistic fuzzy PROMETHEE." Journal of Intelligent & Fuzzy
    Systems, 27(4), 1703–1717.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from .base import IFN, IFSDecisionMatrix
from ...weighting import WeightResult


@dataclass
class IFS_PROMETHEEResult:
    """Result container for IFS-PROMETHEE."""
    phi_positive: pd.Series
    phi_negative: pd.Series
    phi_net: pd.Series
    ranks: pd.Series
    preference_matrix: pd.DataFrame
    weights: Dict[str, float]

    @property
    def final_ranks(self) -> pd.Series:
        return self.ranks

    def top_n(self, n: int = 10) -> pd.DataFrame:
        return pd.DataFrame({
            'Phi+': self.phi_positive,
            'Phi-': self.phi_negative,
            'Phi_net': self.phi_net,
            'Rank': self.ranks,
        }).nsmallest(n, 'Rank')


class IFS_PROMETHEE:
    """
    IFS-PROMETHEE II calculator.

    Parameters
    ----------
    preference_threshold : float
        Strict preference threshold (p) for V-shape function.
    cost_criteria : list, optional
        Criteria where lower values are preferred.
    """

    def __init__(self,
                 preference_threshold: float = 0.3,
                 cost_criteria: Optional[List[str]] = None):
        self.p = preference_threshold
        self.cost_criteria = cost_criteria or []

    def calculate(self,
                  ifs_matrix: IFSDecisionMatrix,
                  weights: Union[Dict[str, float], WeightResult, None] = None,
                  ) -> IFS_PROMETHEEResult:

        criteria = ifs_matrix.criteria
        alternatives = ifs_matrix.alternatives
        n = len(alternatives)
        w = self._resolve_weights(weights, criteria)

        # Score matrix: S_ij = μ − ν
        score_mat = ifs_matrix.to_score_matrix()

        # Aggregated preference matrix  π(a, b)
        pi_mat = np.zeros((n, n))
        for i, a in enumerate(alternatives):
            for j, b in enumerate(alternatives):
                if i == j:
                    continue
                agg = 0.0
                for crit in criteria:
                    sa = score_mat.loc[a, crit]
                    sb = score_mat.loc[b, crit]
                    diff = sa - sb
                    if crit in self.cost_criteria:
                        diff = -diff
                    # V-shape preference function
                    if diff <= 0:
                        pref = 0.0
                    elif diff >= self.p:
                        pref = 1.0
                    else:
                        pref = diff / self.p
                    agg += w[crit] * pref
                pi_mat[i, j] = agg

        # Flows
        phi_pos = pi_mat.sum(axis=1) / max(n - 1, 1)
        phi_neg = pi_mat.sum(axis=0) / max(n - 1, 1)
        phi_net_vals = phi_pos - phi_neg

        phi_positive = pd.Series(phi_pos, index=alternatives, name='Phi+')
        phi_negative = pd.Series(phi_neg, index=alternatives, name='Phi-')
        phi_net = pd.Series(phi_net_vals, index=alternatives, name='Phi_net')

        ranks = phi_net.rank(ascending=False).astype(int)
        ranks.name = 'IFS_PROMETHEE_Rank'

        pref_df = pd.DataFrame(pi_mat, index=alternatives, columns=alternatives)

        return IFS_PROMETHEEResult(
            phi_positive=phi_positive, phi_negative=phi_negative,
            phi_net=phi_net, ranks=ranks,
            preference_matrix=pref_df, weights=w,
        )

    @staticmethod
    def _resolve_weights(weights, criteria):
        if weights is None:
            return {c: 1.0 / len(criteria) for c in criteria}
        if isinstance(weights, WeightResult):
            weights = weights.weights
        return {c: weights.get(c, 1.0 / len(criteria)) for c in criteria}
