# -*- coding: utf-8 -*-
"""
IFS-SAW: Intuitionistic Fuzzy Simple Additive Weighting

The simplest IFS-MCDM method — weighted aggregation of IFS score values.

    Score_i = Σ_j  w_j × S(IFN_ij)

where S(IFN) = μ − ν  is the Chen–Tan score function.

References
----------
[1] Abdullah, L. & Najib, L. (2016). "A new type-2 fuzzy set of
    linguistic variables for the fuzzy analytic hierarchy process."
    Expert Systems with Applications.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from .base import IFN, IFSDecisionMatrix
from ...weighting import WeightResult


@dataclass
class IFS_SAWResult:
    """Result container for IFS-SAW."""
    scores: pd.Series
    ranks: pd.Series
    weighted_scores: pd.DataFrame
    weights: Dict[str, float]

    def top_n(self, n: int = 10) -> pd.DataFrame:
        return pd.DataFrame({
            'Score': self.scores, 'Rank': self.ranks
        }).nsmallest(n, 'Rank')


class IFS_SAW:
    """
    IFS-based Simple Additive Weighting.

    Steps
    -----
    1. Compute score S_ij = μ_ij − ν_ij  for every cell.
    2. Weighted sum:  SAW_i = Σ_j w_j × S_ij.
    3. Rank alternatives by SAW score (higher → better).
    """

    def calculate(self,
                  ifs_matrix: IFSDecisionMatrix,
                  weights: Union[Dict[str, float], WeightResult, None] = None,
                  ) -> IFS_SAWResult:

        score_df = ifs_matrix.to_score_matrix()
        criteria = ifs_matrix.criteria
        alternatives = ifs_matrix.alternatives

        w = self._resolve_weights(weights, criteria)
        w_arr = np.array([w[c] for c in criteria])

        weighted = score_df.values * w_arr
        weighted_df = pd.DataFrame(weighted, index=alternatives, columns=criteria)
        saw_scores = pd.Series(weighted.sum(axis=1), index=alternatives, name='IFS_SAW_Score')

        ranks = saw_scores.rank(ascending=False).astype(int)
        ranks.name = 'IFS_SAW_Rank'

        return IFS_SAWResult(scores=saw_scores, ranks=ranks,
                             weighted_scores=weighted_df, weights=w)

    @staticmethod
    def _resolve_weights(weights, criteria):
        if weights is None:
            return {c: 1.0 / len(criteria) for c in criteria}
        if isinstance(weights, WeightResult):
            weights = weights.weights
        return {c: weights.get(c, 1.0 / len(criteria)) for c in criteria}
