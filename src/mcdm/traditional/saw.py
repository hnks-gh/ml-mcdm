# -*- coding: utf-8 -*-
"""
SAW: Simple Additive Weighting (Weighted Sum Model)

The simplest MCDM method — serves as a transparent baseline.

    Score_i = Σ_j  w_j × r_ij

where r_ij is the normalised value and w_j is the criterion weight.

Properties
----------
- Linear aggregation (fully compensatory).
- Assumes preference independence among criteria.
- Computationally O(m × n).

References
----------
[1] Fishburn, P.C. (1967). "Additive Utilities with Incomplete Product
    Sets: Application to Priorities and Assignments."
    Operations Research, 15(3), 537–542.
[2] MacCrimmon, K.R. (1968). "Decision Making among Multiple-Attribute
    Alternatives." RAND Corporation, RM-4823-ARPA.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from ...weighting import WeightResult, EntropyWeightCalculator


@dataclass
class SAWResult:
    """Result container for SAW."""
    scores: pd.Series
    ranks: pd.Series
    weighted_matrix: pd.DataFrame
    weights: Dict[str, float]

    @property
    def final_ranks(self) -> pd.Series:
        return self.ranks

    def top_n(self, n: int = 10) -> pd.DataFrame:
        return pd.DataFrame({
            'Score': self.scores, 'Rank': self.ranks
        }).nsmallest(n, 'Rank')

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "SAW (Simple Additive Weighting) RESULTS",
            f"{'='*60}",
            f"\nAlternatives: {len(self.ranks)}",
            f"Criteria: {len(self.weights)}",
        ]
        top10 = self.top_n(10)
        lines.append("\nTop 10 Alternatives:")
        for i, (idx, row) in enumerate(top10.iterrows(), 1):
            lines.append(f"  {i}. {idx}: Score={row['Score']:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)


class SAWCalculator:
    """
    Simple Additive Weighting calculator.

    Parameters
    ----------
    normalization : str
        Normalisation method: 'minmax' (default), 'max', 'sum'.
    cost_criteria : list, optional
        Criteria where lower values are preferred.
    """

    def __init__(self,
                 normalization: str = 'minmax',
                 cost_criteria: Optional[List[str]] = None):
        self.normalization = normalization
        self.cost_criteria = cost_criteria or []

    def calculate(self,
                  data: pd.DataFrame,
                  weights: Union[Dict[str, float], WeightResult, None] = None,
                  ) -> SAWResult:
        """
        Calculate SAW scores and rankings.

        Parameters
        ----------
        data : pd.DataFrame
            Decision matrix (alternatives × criteria).
        weights : dict or WeightResult
            Criterion weights.

        Returns
        -------
        SAWResult
        """
        if weights is None:
            wc = EntropyWeightCalculator()
            weights = wc.calculate(data).weights
        elif isinstance(weights, WeightResult):
            weights = weights.weights

        weights = {c: weights.get(c, 1 / len(data.columns)) for c in data.columns}

        # Normalise
        norm = self._normalize(data)

        # Weighted sum
        w_arr = np.array([weights[c] for c in data.columns])
        weighted = norm * w_arr
        weighted_df = pd.DataFrame(weighted, index=data.index, columns=data.columns)

        scores = pd.Series(weighted.sum(axis=1), index=data.index, name='SAW_Score')
        ranks = scores.rank(ascending=False).astype(int)
        ranks.name = 'SAW_Rank'

        return SAWResult(scores=scores, ranks=ranks,
                         weighted_matrix=weighted_df, weights=weights)

    def _normalize(self, data: pd.DataFrame) -> np.ndarray:
        X = data.values.astype(float)

        if self.normalization == 'minmax':
            min_v = X.min(axis=0)
            max_v = X.max(axis=0)
            rng = max_v - min_v
            rng[rng == 0] = 1
            norm = np.zeros_like(X)
            for j, col in enumerate(data.columns):
                if col in self.cost_criteria:
                    norm[:, j] = (max_v[j] - X[:, j]) / rng[j]
                else:
                    norm[:, j] = (X[:, j] - min_v[j]) / rng[j]
            return norm

        elif self.normalization == 'max':
            max_v = X.max(axis=0)
            max_v[max_v == 0] = 1
            norm = np.zeros_like(X)
            for j, col in enumerate(data.columns):
                if col in self.cost_criteria:
                    min_v = X[:, j].min()
                    norm[:, j] = min_v / X[:, j] if min_v > 0 else 0
                else:
                    norm[:, j] = X[:, j] / max_v[j]
            return norm

        elif self.normalization == 'sum':
            col_sums = X.sum(axis=0)
            col_sums[col_sums == 0] = 1
            return X / col_sums

        raise ValueError(f"Unknown normalization: {self.normalization}")
