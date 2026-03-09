# -*- coding: utf-8 -*-
"""
Temporal Stability Validation for CRITIC Weight Vectors
========================================================

Split-half stability analysis: compare weight vectors derived from the
first half of years with those from the second half using cosine similarity
and Spearman rank correlation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


@dataclass
class StabilityResult:
    """Temporal stability result for a weight vector."""
    is_stable: bool
    cosine_similarity: float
    spearman_correlation: float
    threshold: float

    @property
    def summary(self) -> str:
        status = "STABLE" if self.is_stable else "UNSTABLE"
        return (
            f"Temporal Stability [{status}] | "
            f"Cosine={self.cosine_similarity:.4f} | "
            f"Spearman={self.spearman_correlation:.4f} | "
            f"Threshold={self.threshold:.4f}"
        )


class TemporalStabilityValidator:
    """
    Split-half validator for CRITIC weight temporal stability.

    Parameters
    ----------
    stability_threshold : float, default=0.85
        Minimum cosine similarity considered stable.
    """

    def __init__(self, stability_threshold: float = 0.85):
        self.stability_threshold = stability_threshold

    def validate(self, weight_history: pd.DataFrame) -> StabilityResult:
        """
        Split the weight history in half and compare halves.

        Parameters
        ----------
        weight_history : pd.DataFrame
            Shape (n_periods, n_criteria). Each row is a weight vector.

        Returns
        -------
        StabilityResult
        """
        n = len(weight_history)
        if n < 2:
            return StabilityResult(
                is_stable=True,
                cosine_similarity=1.0,
                spearman_correlation=1.0,
                threshold=self.stability_threshold,
            )
        mid = n // 2
        first_half = weight_history.iloc[:mid].mean(axis=0).values.astype(float)
        second_half = weight_history.iloc[mid:].mean(axis=0).values.astype(float)

        norm1 = np.linalg.norm(first_half)
        norm2 = np.linalg.norm(second_half)
        if norm1 < 1e-10 or norm2 < 1e-10:
            cosine_sim = 1.0
        else:
            cosine_sim = float(np.dot(first_half, second_half) / (norm1 * norm2))

        if len(first_half) >= 2:
            corr, _ = spearmanr(first_half, second_half)
            spearman_corr = float(np.nan_to_num(corr))
        else:
            spearman_corr = 1.0

        is_stable = cosine_sim >= self.stability_threshold

        return StabilityResult(
            is_stable=is_stable,
            cosine_similarity=cosine_sim,
            spearman_correlation=spearman_corr,
            threshold=self.stability_threshold,
        )


def temporal_stability_verification(
    weight_history: pd.DataFrame,
    stability_threshold: float = 0.85,
) -> StabilityResult:
    """Convenience function: temporal stability for weight history."""
    return TemporalStabilityValidator(stability_threshold).validate(weight_history)

