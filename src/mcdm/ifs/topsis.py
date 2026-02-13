# -*- coding: utf-8 -*-
"""
IFS-TOPSIS: Intuitionistic Fuzzy TOPSIS

Extends classical TOPSIS using IFS distance measures to account for
hesitancy in the decision matrix.

Steps
-----
1. Build the IFS weighted decision matrix.
2. Determine IFS ideal (A⁺) and anti-ideal (A⁻) alternatives.
3. Calculate IFS distances d⁺ and d⁻ for each alternative.
4. Compute relative closeness  C_i = d⁻_i / (d⁺_i + d⁻_i).

References
----------
[1] Boran, F.E. et al. (2009). "A multi-criteria intuitionistic fuzzy
    group decision making for supplier selection with TOPSIS method."
    Expert Systems with Applications, 36, 11363–11368.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from .base import IFN, IFSDecisionMatrix
from ...weighting import WeightResult


@dataclass
class IFS_TOPSISResult:
    """Result container for IFS-TOPSIS."""
    scores: pd.Series
    ranks: pd.Series
    d_positive: pd.Series
    d_negative: pd.Series
    ideal: Dict[str, IFN]
    anti_ideal: Dict[str, IFN]
    weights: Dict[str, float]

    def top_n(self, n: int = 10) -> pd.DataFrame:
        return pd.DataFrame({
            'Score': self.scores,
            'Rank': self.ranks,
            'd+': self.d_positive,
            'd-': self.d_negative,
        }).nsmallest(n, 'Rank')


class IFS_TOPSIS:
    """
    IFS-TOPSIS calculator.

    Parameters
    ----------
    distance_metric : str
        'euclidean', 'hamming', or 'normalized_euclidean'.
    cost_criteria : list, optional
        Criteria where lower values are preferred.
    """

    def __init__(self,
                 distance_metric: str = 'euclidean',
                 cost_criteria: Optional[List[str]] = None):
        self.distance_metric = distance_metric
        self.cost_criteria = cost_criteria or []

    def calculate(self,
                  ifs_matrix: IFSDecisionMatrix,
                  weights: Union[Dict[str, float], WeightResult, None] = None,
                  ) -> IFS_TOPSISResult:

        criteria = ifs_matrix.criteria
        alternatives = ifs_matrix.alternatives
        w = self._resolve_weights(weights, criteria)

        # Step 1 — IFS weighted matrix (scalar weight × IFN)
        weighted: Dict[str, Dict[str, IFN]] = {}
        for alt in alternatives:
            weighted[alt] = {}
            for crit in criteria:
                ifn = ifs_matrix.get(alt, crit)
                weighted[alt][crit] = w[crit] * ifn   # scalar * IFN

        # Step 2 — Ideal and anti-ideal
        ideal: Dict[str, IFN] = {}
        anti_ideal: Dict[str, IFN] = {}
        for crit in criteria:
            mus = [weighted[a][crit].mu for a in alternatives]
            nus = [weighted[a][crit].nu for a in alternatives]
            if crit in self.cost_criteria:
                ideal[crit] = IFN(mu=min(mus), nu=max(nus))
                anti_ideal[crit] = IFN(mu=max(mus), nu=min(nus))
            else:
                ideal[crit] = IFN(mu=max(mus), nu=min(nus))
                anti_ideal[crit] = IFN(mu=min(mus), nu=max(nus))

        # Step 3 — Distances
        dist_fn = self._get_distance_fn()
        d_pos_vals, d_neg_vals = [], []
        for alt in alternatives:
            dp = np.sqrt(sum(dist_fn(weighted[alt][c], ideal[c]) ** 2
                             for c in criteria))
            dn = np.sqrt(sum(dist_fn(weighted[alt][c], anti_ideal[c]) ** 2
                             for c in criteria))
            d_pos_vals.append(dp)
            d_neg_vals.append(dn)

        d_pos = pd.Series(d_pos_vals, index=alternatives, name='d+')
        d_neg = pd.Series(d_neg_vals, index=alternatives, name='d-')

        # Step 4 — Closeness coefficient
        scores = d_neg / (d_pos + d_neg + 1e-12)
        scores.name = 'IFS_TOPSIS_Score'
        ranks = scores.rank(ascending=False).astype(int)
        ranks.name = 'IFS_TOPSIS_Rank'

        return IFS_TOPSISResult(
            scores=scores, ranks=ranks,
            d_positive=d_pos, d_negative=d_neg,
            ideal=ideal, anti_ideal=anti_ideal, weights=w,
        )

    def _get_distance_fn(self):
        if self.distance_metric == 'hamming':
            return IFN.hamming_distance
        if self.distance_metric == 'normalized_euclidean':
            return IFN.normalized_euclidean
        return IFN.euclidean_distance

    @staticmethod
    def _resolve_weights(weights, criteria):
        if weights is None:
            return {c: 1.0 / len(criteria) for c in criteria}
        if isinstance(weights, WeightResult):
            weights = weights.weights
        return {c: weights.get(c, 1.0 / len(criteria)) for c in criteria}
