# -*- coding: utf-8 -*-
"""
IFS-EDAS: Intuitionistic Fuzzy EDAS

Evaluation based on Distance from Average Solution under IFS uncertainty.

Steps
-----
1. Average IFS score per criterion:  AV_j = mean(S(IFN_ij)).
2. Positive distance  PDA_ij = max(0, S_ij − AV_j) / |AV_j|.
3. Negative distance  NDA_ij = max(0, AV_j − S_ij) / |AV_j|.
4. Weighted sums  SP_i, SN_i.
5. Normalise  NSP, NSN.
6. Appraisal score  AS_i = (NSP_i + 1 − NSN_i) / 2.

References
----------
[1] Kahraman, C. et al. (2017). "Intuitionistic fuzzy EDAS method."
    International Journal of Information Technology & Decision Making,
    16(6), 1511–1536.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from .base import IFN, IFSDecisionMatrix
from ...weighting import WeightResult


@dataclass
class IFS_EDASResult:
    """Result container for IFS-EDAS."""
    PDA: pd.DataFrame
    NDA: pd.DataFrame
    SP: pd.Series
    SN: pd.Series
    NSP: pd.Series
    NSN: pd.Series
    AS: pd.Series
    ranks: pd.Series
    average_solution: pd.Series
    weights: Dict[str, float]

    @property
    def final_ranks(self) -> pd.Series:
        return self.ranks

    def top_n(self, n: int = 10) -> pd.DataFrame:
        return pd.DataFrame({
            'SP': self.SP, 'SN': self.SN,
            'NSP': self.NSP, 'NSN': self.NSN,
            'AS': self.AS, 'Rank': self.ranks,
        }).nsmallest(n, 'Rank')


class IFS_EDAS:
    """
    IFS-EDAS calculator.

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
                  ) -> IFS_EDASResult:

        criteria = ifs_matrix.criteria
        alternatives = ifs_matrix.alternatives
        w = self._resolve_weights(weights, criteria)

        score_df = ifs_matrix.to_score_matrix()

        # Step 1 — Average solution
        av = score_df.mean(axis=0)

        # Steps 2–3 — PDA & NDA
        pda_data = np.zeros_like(score_df.values)
        nda_data = np.zeros_like(score_df.values)
        for j, crit in enumerate(criteria):
            av_j = av[crit]
            denom = abs(av_j) if abs(av_j) > 1e-12 else 1e-12
            for i, alt in enumerate(alternatives):
                val = score_df.iloc[i, j]
                if crit in self.cost_criteria:
                    pda_data[i, j] = max(0, av_j - val) / denom
                    nda_data[i, j] = max(0, val - av_j) / denom
                else:
                    pda_data[i, j] = max(0, val - av_j) / denom
                    nda_data[i, j] = max(0, av_j - val) / denom

        PDA = pd.DataFrame(pda_data, index=alternatives, columns=criteria)
        NDA = pd.DataFrame(nda_data, index=alternatives, columns=criteria)

        # Step 4 — Weighted sums
        w_arr = np.array([w[c] for c in criteria])
        SP = pd.Series((PDA.values * w_arr).sum(axis=1), index=alternatives, name='SP')
        SN = pd.Series((NDA.values * w_arr).sum(axis=1), index=alternatives, name='SN')

        # Step 5 — Normalise
        sp_max = SP.max() if SP.max() > 0 else 1.0
        sn_max = SN.max() if SN.max() > 0 else 1.0
        NSP = SP / sp_max
        NSN = SN / sn_max
        NSP.name, NSN.name = 'NSP', 'NSN'

        # Step 6 — Appraisal score
        AS = (NSP + 1 - NSN) / 2
        AS.name = 'IFS_EDAS_AS'

        ranks = AS.rank(ascending=False).astype(int)
        ranks.name = 'IFS_EDAS_Rank'

        return IFS_EDASResult(
            PDA=PDA, NDA=NDA, SP=SP, SN=SN,
            NSP=NSP, NSN=NSN, AS=AS,
            ranks=ranks, average_solution=av, weights=w,
        )

    @staticmethod
    def _resolve_weights(weights, criteria):
        if weights is None:
            return {c: 1.0 / len(criteria) for c in criteria}
        if isinstance(weights, WeightResult):
            weights = weights.weights
        return {c: weights.get(c, 1.0 / len(criteria)) for c in criteria}
