# -*- coding: utf-8 -*-
"""
IFS-VIKOR: Intuitionistic Fuzzy VIKOR

Compromise ranking under IFS uncertainty.  Uses IFS score values and
distance operators to compute group utility (S), individual regret (R),
and the compromise index (Q).

Steps
-----
1. Determine IFS best (f*) and worst (f⁻) for each criterion.
2. S_i = Σ_j w_j × d(f*_j, f_ij) / d(f*_j, f⁻_j)   (group utility).
3. R_i = max_j  w_j × d(f*_j, f_ij) / d(f*_j, f⁻_j)  (individual regret).
4. Q_i = v × (S_i − S*) / (S⁻ − S*) + (1−v) × (R_i − R*) / (R⁻ − R*).

References
----------
[1] Devi, K. (2011). "Extension of VIKOR method in intuitionistic fuzzy
    environment for robot selection." Expert Systems with Applications,
    38(11), 14163–14168.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from .base import IFN, IFSDecisionMatrix
from ...weighting import WeightResult


@dataclass
class IFS_VIKORResult:
    """Result container for IFS-VIKOR."""
    S: pd.Series
    R: pd.Series
    Q: pd.Series
    ranks_S: pd.Series
    ranks_R: pd.Series
    ranks_Q: pd.Series
    compromise_set: List[str]
    weights: Dict[str, float]
    v: float

    @property
    def final_ranks(self) -> pd.Series:
        return self.ranks_Q

    def top_n(self, n: int = 10) -> pd.DataFrame:
        return pd.DataFrame({
            'S': self.S, 'R': self.R, 'Q': self.Q, 'Rank': self.ranks_Q
        }).nsmallest(n, 'Rank')


class IFS_VIKOR:
    """
    IFS-VIKOR calculator.

    Parameters
    ----------
    v : float
        Strategy weight for group utility vs individual regret.
    cost_criteria : list, optional
        Criteria where lower values are preferred.
    """

    def __init__(self, v: float = 0.5,
                 cost_criteria: Optional[List[str]] = None):
        self.v = v
        self.cost_criteria = cost_criteria or []

    def calculate(self,
                  ifs_matrix: IFSDecisionMatrix,
                  weights: Union[Dict[str, float], WeightResult, None] = None,
                  ) -> IFS_VIKORResult:

        criteria = ifs_matrix.criteria
        alternatives = ifs_matrix.alternatives
        w = self._resolve_weights(weights, criteria)
        dist_fn = IFN.normalized_euclidean

        # Step 1 — IFS best / worst per criterion
        best: Dict[str, IFN] = {}
        worst: Dict[str, IFN] = {}
        for crit in criteria:
            ifns = [ifs_matrix.get(a, crit) for a in alternatives]
            scores = [ifn.score() for ifn in ifns]
            if crit in self.cost_criteria:
                best[crit] = ifns[int(np.argmin(scores))]
                worst[crit] = ifns[int(np.argmax(scores))]
            else:
                best[crit] = ifns[int(np.argmax(scores))]
                worst[crit] = ifns[int(np.argmin(scores))]

        # Steps 2–3 — S and R
        S_vals, R_vals = [], []
        for alt in alternatives:
            s_i, r_i = 0.0, 0.0
            for crit in criteria:
                d_star = dist_fn(best[crit], ifs_matrix.get(alt, crit))
                d_range = dist_fn(best[crit], worst[crit])
                if d_range < 1e-12:
                    d_range = 1e-12
                val = w[crit] * d_star / d_range
                s_i += val
                r_i = max(r_i, val)
            S_vals.append(s_i)
            R_vals.append(r_i)

        S = pd.Series(S_vals, index=alternatives, name='S')
        R = pd.Series(R_vals, index=alternatives, name='R')

        # Step 4 — Q
        S_star, S_minus = S.min(), S.max()
        R_star, R_minus = R.min(), R.max()
        denom_s = S_minus - S_star if S_minus - S_star > 1e-12 else 1e-12
        denom_r = R_minus - R_star if R_minus - R_star > 1e-12 else 1e-12

        Q = self.v * (S - S_star) / denom_s + (1 - self.v) * (R - R_star) / denom_r
        Q.name = 'Q'

        ranks_S = S.rank(ascending=True).astype(int)
        ranks_R = R.rank(ascending=True).astype(int)
        ranks_Q = Q.rank(ascending=True).astype(int)
        ranks_S.name, ranks_R.name, ranks_Q.name = 'Rank_S', 'Rank_R', 'Rank_Q'

        # Compromise set
        n = len(alternatives)
        threshold = 1.0 / (n - 1) if n > 1 else 0
        sorted_q = Q.sort_values()
        compromise = [sorted_q.index[0]]
        for idx in sorted_q.index[1:]:
            if sorted_q[idx] - sorted_q.iloc[0] <= threshold:
                compromise.append(idx)
            else:
                break

        return IFS_VIKORResult(
            S=S, R=R, Q=Q,
            ranks_S=ranks_S, ranks_R=ranks_R, ranks_Q=ranks_Q,
            compromise_set=compromise, weights=w, v=self.v,
        )

    @staticmethod
    def _resolve_weights(weights, criteria):
        if weights is None:
            return {c: 1.0 / len(criteria) for c in criteria}
        if isinstance(weights, WeightResult):
            weights = weights.weights
        return {c: weights.get(c, 1.0 / len(criteria)) for c in criteria}
