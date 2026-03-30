# -*- coding: utf-8 -*-
"""
CRITIC Weight Calculator

Criteria Importance Through Inter-criteria Correlation method.
Considers both contrast intensity (standard deviation) and 
inter-criteria correlation to determine weights.

Mathematical Formula:
    w_j = C_j / Σ(C_k)
    
where:
    C_j = σ_j × Σ(1 - r_jk)  [information content]
    σ_j = standard deviation of criterion j
    r_jk = correlation between criteria j and k
"""

import logging

import numpy as np
import pandas as pd

from .base import WeightResult

logger = logging.getLogger(__name__)


class CRITICWeightCalculator:
    """
    Criteria Importance Through Inter-criteria Correlation (CRITIC) calculator.

    Determines objective weights based on two fundamental dimensions:
    1.  Contrast Intensity: Measured by the standard deviation of each 
        criterion across alternatives.
    2.  Conflicting Character: Measured by the correlation between 
        criteria.

    Criteria with high variation and low correlation with others are 
    prioritized as providing more unique information.
    """
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
    
    def calculate(self, data: pd.DataFrame,
                  sample_weights: 'np.ndarray | None' = None) -> WeightResult:
        """
        Execute the CRITIC weighting algorithm.

        Applies complete-case exclusion to handle missing values, ensuring 
        that variance and correlation estimates are not biased by imputation 
        artifacts.

        Parameters
        ----------
        data : pd.DataFrame
            The normalized decision matrix (provinces × criteria).
        sample_weights : Optional[np.ndarray]
            Weights for observations to compute a weighted covariance matrix.

        Returns
        -------
        WeightResult
            Object containing normalized weights and detailed metrics.
        """
        # Input validation
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        if len(data) < 2:
            raise ValueError("CRITIC calculation requires at least 2 observations")
        
        non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise TypeError(f"Non-numeric columns found: {non_numeric}")
        
        # ── Complete-case exclusion ───────────────────────────────────────
        # Drop rows containing any NaN before computing CRITIC statistics.
        #
        # Rationale: imputing NaN with column means (the former behaviour)
        # causes two distinct statistical biases:
        #   1. Variance attenuation — imputed rows all equal the column mean,
        #      so σ_j is artificially reduced.
        #   2. Spurious correlation — columns with overlapping NaN positions
        #      are both pulled toward their respective means, inflating r_{jk}.
        # Both effects corrupt C_j = σ_j × Σ_k(1 − r_{jk}) and, therefore,
        # the final weights.  Complete-case analysis is the statistically
        # correct estimator when missingness is structural (e.g. a governance
        # indicator not yet collected for certain years), not random.
        #
        # Callers (CRITICWeightingCalculator F-01/F-02) already perform
        # per-group dropna before invoking CRITIC, so this guard fires only
        # for secondary callers that pass raw data.
        data = data.copy()
        if data.isnull().any().any():
            n_before = len(data)
            valid_mask = ~data.isnull().any(axis=1)
            data = data[valid_mask]
            n_dropped = n_before - len(data)
            logger.warning(
                "CRITICWeightCalculator: %d row(s) with NaN dropped "
                "(%d → %d rows). Callers should pre-filter via complete-case "
                "exclusion; imputation removed to prevent biased weight "
                "estimation (variance attenuation + spurious correlation).",
                n_dropped, n_before, len(data),
            )
            # Synchronise sample_weights: drop weights for excluded rows so
            # the positional alignment with the surviving rows is preserved.
            if sample_weights is not None:
                _sw_arr = np.asarray(sample_weights, dtype=float)
                if _sw_arr.shape[0] == n_before:
                    sample_weights = _sw_arr[valid_mask.values]
            if len(data) < 2:
                n_criteria = len(data.columns)
                logger.warning(
                    "CRITICWeightCalculator: fewer than 2 complete-case rows "
                    "remain after NaN exclusion — returning equal weights.",
                )
                return WeightResult(
                    weights={c: 1.0 / n_criteria for c in data.columns},
                    method="critic",
                    details={
                        "note": (
                            "equal_weights: fewer than 2 complete-case rows "
                            "remain after NaN exclusion"
                        )
                    },
                )

        n = len(data)
        X = data.values  # (n, p) — NaN-free beyond this point
        columns = data.columns.tolist()
        
        if sample_weights is not None:
            w = np.asarray(sample_weights, dtype=float)
            if w.shape[0] != n:
                raise ValueError(
                    f"sample_weights length ({w.shape[0]}) != n_observations ({n})")
            w_sum = w.sum()
            w = w / w_sum if w_sum > self.epsilon else np.full(n, 1.0 / n)
            
            # Weighted covariance matrix via np.cov with aweights.
            # Use ddof=0 (biased / population estimate) to avoid the Bessel
            # correction denominator V1 − V2/V1 approaching zero when one
            # Dirichlet-drawn weight dominates (audit fix M2).  The biased
            # estimator is always well-defined and still consistent.
            cov_matrix = np.cov(X.T, aweights=w, ddof=0)
            if cov_matrix.ndim == 0:
                cov_matrix = np.array([[float(cov_matrix)]])
            
            # Weighted std from diagonal of covariance
            std_arr = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))
            std_arr = np.where(std_arr < self.epsilon, self.epsilon, std_arr)
            
            # Weighted Pearson correlation from covariance
            # r_jk = cov_jk / (σ_j σ_k)
            outer_std = np.outer(std_arr, std_arr)
            outer_std = np.where(outer_std < self.epsilon, self.epsilon, outer_std)
            corr_arr = cov_matrix / outer_std
            # Clamp to [-1, 1] for numerical safety
            corr_arr = np.clip(corr_arr, -1.0, 1.0)
            
            corr_matrix = pd.DataFrame(corr_arr, index=columns, columns=columns)
            std = pd.Series(std_arr, index=columns)
        else:
            # Standard (unweighted) statistics
            std = data.std(axis=0)
            std = std.replace(0, self.epsilon)
            corr_matrix = data.corr()
        
        # Handle NaN values in correlation matrix (occurs with constant columns)
        corr_matrix = corr_matrix.fillna(1.0)
        
        # Conflict measure (sum of 1 - r_jk for all k)
        conflict = (1 - corr_matrix).sum(axis=0)
        conflict = conflict.clip(lower=self.epsilon)
        
        # Information content (std × conflict)
        C = std * conflict
        
        if C.sum() < self.epsilon:
            n_criteria = len(columns)
            weights = pd.Series(1.0 / n_criteria, index=columns)
        else:
            weights = C / C.sum()
        
        return WeightResult(
            weights=weights.to_dict(),
            method="critic",
            details={
                "std_values": std.to_dict(),
                "conflict_values": conflict.to_dict(),
                "information_content": C.to_dict(),
                "correlation_matrix": corr_matrix.to_dict()
            }
        )
