"""
Perturbation Sensitivity Analysis for CRITIC Weights.

This module provides tools for assessing the robustness of CRITIC weights 
to measurement noise and data revisions. It implements a three-tier 
perturbation strategy (conservative, moderate, aggressive) and measures 
the resulting rank disruption in the composite criteria scores.

Key Features
------------
- **Three-Tier Stress Testing**: Evaluates weight stability under ±5%, 
  ±15%, and ±50% perturbations.
- **Rank Disruption Metric**: Quantifies sensitivity using 1 - Spearman's 
  rho between original and perturbed rankings.
- **Criterion-Level Sensitivity**: Identifies which specific criteria are 
  most likely to change rank under noise.
- **Re-normalization Integrity**: Ensures that perturbed weight vectors 
  always satisfy the sum-to-one constraint and non-negativity.

References
----------
- Saltelli et al. (2008). "Global Sensitivity Analysis." Wiley.
- Sobol' (2001). "Global sensitivity indices for nonlinear models." 
  Mathematics and Computers in Simulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """
    Result container for three-tier perturbation sensitivity analysis.

    Attributes
    ----------
    tier_robustness : Dict[str, float]
        Robustness score per tier ∈ [0, 1].
        Keys: 'conservative', 'moderate', 'aggressive'.
        Robustness = fraction of perturbations NOT causing rank change.
        High robustness (>0.9): weights insensitive to perturbations.
        Low robustness (<0.5): weights sensitive, rankings easily disrupted.

    per_criterion_sensitivity : Dict[str, Dict[str, float]]
        Sensitivity per criterion and tier.
        Structure: {
            'C01': {'conservative': 0.05, 'moderate': 0.12, 'aggressive': 0.45},
            'C02': {...},
            ...
        }
        Values are ranks (0-based). Higher = more frequently changes rank.

    weight_delta_stats : Dict[str, Dict[str, Any]]
        Post-perturbation weight statistics per tier.
        Structure: {
            'conservative': {'mean': 0.001, 'max': 0.012, 'min': 0.0},
            ...
        }
        Captures magnitude of weight changes (before re-normalization).

    rank_disruption_stats : Dict[str, Dict[str, Any]]
        Rank disruption metrics per tier.
        Structure: {
            'conservative': {'mean': 0.02, 'max': 0.15, 'q95': 0.08},
            ...
        }
        Disruption = 1 - Spearman's ρ(before_rank, after_rank).

    top_sensitive_criteria : Dict[str, List[str]]
        Most sensitive criteria per tier (top 3).
        Keys: 'conservative', 'moderate', 'aggressive'.
        Values: ['C05', 'C07', 'C02'] (sorted by sensitivity descending).

    year_range : Tuple[int, int]
        (min_year, max_year) of the panel data.

    n_replicates : int
        Number of perturbation replicates per year-tier combination.

    perturbation_tiers : Dict[str, float]
        Magnitude mapping: tier name → epsilon.
        {'conservative': 0.05, 'moderate': 0.15, 'aggressive': 0.50}
    """
    tier_robustness: Dict[str, float]
    per_criterion_sensitivity: Dict[str, Dict[str, float]]
    weight_delta_stats: Dict[str, Dict[str, Any]]
    rank_disruption_stats: Dict[str, Dict[str, Any]]
    top_sensitive_criteria: Dict[str, List[str]]
    year_range: Tuple[int, int]
    n_replicates: int
    perturbation_tiers: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @property
    def is_robust(self) -> bool:
        """
        Check if the weights are robust at the conservative tier.

        Returns
        -------
        bool
            True if conservative robustness > 0.90, indicating high stability 
            under typical measurement noise.
        """
        return self.tier_robustness.get('conservative', 0.0) > 0.90


class CRITICSensitivityAnalyzer:
    """
    Analyze weight sensitivity to perturbations via rank disruption.

    Parameters
    ----------
    perturbation_tiers : Dict[str, float], optional
        Tier names and magnitudes. Default:
        {'conservative': 0.05, 'moderate': 0.15, 'aggressive': 0.50}
    n_replicates : int, default=50
        Number of perturbation replicates per year-tier combination.
    rank_disruption_threshold : float, default=0.1
        Threshold for "significant" rank disruption.
        Perturbations with disruption > threshold count against robustness.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    perturbation_tiers : Dict[str, float]
    n_replicates : int
    rank_disruption_threshold : float
    seed : Optional[int]
    rng : np.random.RandomState

    Notes
    -----
    Perturbation procedure (per year, per tier):
      1. Start with original weight vector w
      2. For each of n_replicates:
         a. Draw perturbations δ_i ~ U(-ε, +ε) for each criterion
         b. Perturbed weights: w'_i = w_i * (1 + δ_i)
         c. Re-normalize: w'_i ← w'_i / Σ w'_j (ensures sum=1.0)
         d. Compute disruption = 1 - Spearman's ρ(rank(w), rank(w'))
      3. Robustness per tier = fraction of replicates with disruption ≤ threshold

    Edge cases handled:
      - Identical weights: robustness = 1.0 (no rank change possible)
      - Single criterion: robustness = 1.0 (cannot rank)
      - No valid perturbations: return default result
    """

    def __init__(
        self,
        perturbation_tiers: Optional[Dict[str, float]] = None,
        n_replicates: int = 50,
        rank_disruption_threshold: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize the CRITIC sensitivity analyzer.

        Parameters
        ----------
        perturbation_tiers : Dict[str, float], optional
            Tier names and their corresponding perturbation magnitudes (as 
            fractions). Defaults to conservative (0.05), moderate (0.15), 
            and aggressive (0.50).
        n_replicates : int, default=50
            Number of random perturbations to generate per year per tier.
        rank_disruption_threshold : float, default=0.1
            The maximum '1 - Spearman rho' value allowed for a perturbation 
            to be considered "robust".
        seed : int, optional
            Seed for reproducible random perturbations.
        """
        if perturbation_tiers is None:
            self.perturbation_tiers = {
                'conservative': 0.05,
                'moderate': 0.15,
                'aggressive': 0.50,
            }
        else:
            self.perturbation_tiers = perturbation_tiers

        self.n_replicates = n_replicates
        self.rank_disruption_threshold = rank_disruption_threshold
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        if n_replicates <= 0:
            raise ValueError(f"n_replicates must be > 0, got {n_replicates}")

    def analyze(
        self, weight_all_years: Dict[int, Dict[str, float]]
    ) -> SensitivityResult:
        """
        Perform three-tier perturbation sensitivity analysis.

        Parameters
        ----------
        weight_all_years : Dict[int, Dict[str, float]]
            Year-indexed weights. Format:
            {
                2011: {'C01': 0.12, 'C02': 0.15, ..., 'C08': 0.08},
                ...
                2024: {'C01': 0.11, 'C02': 0.16, ..., 'C08': 0.07}
            }

        Returns
        -------
        SensitivityResult
            Sensitivity metrics across all tiers, criteria, and years.

        Raises
        ------
        ValueError
            If weight_all_years is empty or malformed.
        """
        if not weight_all_years:
            raise ValueError("weight_all_years cannot be empty")

        years = sorted(weight_all_years.keys())
        year_range = (years[0], years[-1])
        criteria = list(weight_all_years[years[0]].keys())

        # Aggregate results across years and tiers
        tier_disruptions = {t: [] for t in self.perturbation_tiers.keys()}
        tier_weight_deltas = {t: [] for t in self.perturbation_tiers.keys()}
        criterion_rank_changes = {
            c: {t: 0 for t in self.perturbation_tiers.keys()}
            for c in criteria
        }

        # For each year and tier, run perturbation replicates
        for year in years:
            weights = weight_all_years[year]
            w_vec = np.array([weights[c] for c in criteria])

            # Guard: identical or near-identical weights
            if np.allclose(w_vec, w_vec.mean()):
                logger.warning(f"Year {year}: all weights nearly identical, skipping")
                continue

            for tier_name, epsilon in self.perturbation_tiers.items():
                for rep in range(self.n_replicates):
                    # Generate perturbations
                    deltas = self.rng.uniform(-epsilon, epsilon, size=len(criteria))

                    # Perturbed weights (before re-normalization)
                    w_perturbed_prenorm = w_vec * (1 + deltas)
                    weight_delta = np.abs(w_perturbed_prenorm - w_vec).mean()
                    tier_weight_deltas[tier_name].append(weight_delta)

                    # Re-normalize to satisfy sum=1.0 constraint
                    w_perturbed = w_perturbed_prenorm / np.sum(w_perturbed_prenorm)

                    # Compute rank disruption
                    disruption = self._compute_rank_disruption(w_vec, w_perturbed)
                    tier_disruptions[tier_name].append(disruption)

                    # Track criterion rank changes
                    rank_before = np.argsort(w_vec)
                    rank_after = np.argsort(w_perturbed)
                    for i, c in enumerate(criteria):
                        if rank_before[i] != rank_after[i]:
                            criterion_rank_changes[c][tier_name] += 1

        # Compute robustness per tier (fraction not significantly disrupted)
        tier_robustness = {}
        for tier_name in self.perturbation_tiers.keys():
            if tier_disruptions[tier_name]:
                disruptions = np.array(tier_disruptions[tier_name])
                robustness = float(
                    np.mean(disruptions <= self.rank_disruption_threshold)
                )
                tier_robustness[tier_name] = robustness
            else:
                tier_robustness[tier_name] = 1.0

        # Compute per-criterion sensitivity (fraction of perturbs causing rank change)
        per_criterion_sensitivity = {}
        total_perturbs = len(years) * self.n_replicates

        for c in criteria:
            per_criterion_sensitivity[c] = {}
            for tier_name in self.perturbation_tiers.keys():
                rank_changes = criterion_rank_changes[c][tier_name]
                sensitivity = rank_changes / total_perturbs if total_perturbs > 0 else 0.0
                per_criterion_sensitivity[c][tier_name] = float(sensitivity)

        # Compute weight delta statistics per tier
        weight_delta_stats = {}
        for tier_name in self.perturbation_tiers.keys():
            if tier_weight_deltas[tier_name]:
                deltas = np.array(tier_weight_deltas[tier_name])
                weight_delta_stats[tier_name] = {
                    'mean': float(np.mean(deltas)),
                    'std': float(np.std(deltas)),
                    'min': float(np.min(deltas)),
                    'max': float(np.max(deltas)),
                    'q95': float(np.percentile(deltas, 95)),
                }
            else:
                weight_delta_stats[tier_name] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'q95': 0.0
                }

        # Compute rank disruption statistics per tier
        rank_disruption_stats = {}
        for tier_name in self.perturbation_tiers.keys():
            if tier_disruptions[tier_name]:
                disruptions = np.array(tier_disruptions[tier_name])
                rank_disruption_stats[tier_name] = {
                    'mean': float(np.mean(disruptions)),
                    'std': float(np.std(disruptions)),
                    'min': float(np.min(disruptions)),
                    'max': float(np.max(disruptions)),
                    'q95': float(np.percentile(disruptions, 95)),
                }
            else:
                rank_disruption_stats[tier_name] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'q95': 0.0
                }

        # Identify top sensitive criteria per tier
        top_sensitive_criteria = {}
        for tier_name in self.perturbation_tiers.keys():
            sensitivities = [
                (c, per_criterion_sensitivity[c][tier_name])
                for c in criteria
            ]
            sensitivities.sort(key=lambda x: x[1], reverse=True)
            top_sensitive_criteria[tier_name] = [c for c, _ in sensitivities[:3]]

        result = SensitivityResult(
            tier_robustness=tier_robustness,
            per_criterion_sensitivity=per_criterion_sensitivity,
            weight_delta_stats=weight_delta_stats,
            rank_disruption_stats=rank_disruption_stats,
            top_sensitive_criteria=top_sensitive_criteria,
            year_range=year_range,
            n_replicates=self.n_replicates,
            perturbation_tiers=self.perturbation_tiers,
        )

        logger.info(
            f"Sensitivity analysis complete: "
            f"Conservative robustness={tier_robustness.get('conservative', 0.0):.3f}, "
            f"Moderate={tier_robustness.get('moderate', 0.0):.3f}, "
            f"Aggressive={tier_robustness.get('aggressive', 0.0):.3f}"
        )

        return result

    def _compute_rank_disruption(self, w_before: np.ndarray, w_after: np.ndarray) -> float:
        """
        Compute rank disruption metric between weight vectors.

        Parameters
        ----------
        w_before : np.ndarray
            Original weight vector.
        w_after : np.ndarray
            Perturbed and re-normalized weight vector.

        Returns
        -------
        float
            Disruption ∈ [0, 1], computed as 1 - Spearman's ρ.
            0: identical ranking
            1: completely reversed ranking

        Notes
        -----
        Formula: Disruption = 1 - ρ_spearman(rank(w_before), rank(w_after))
          where ρ ∈ [-1, 1], so Disruption ∈ [0, 2] → clipped to [0, 1].

        Spearman's ρ is robust to weight magnitude changes, sensitive to
        ranking order. Perfect disruption (rank reversal) → ρ ≈ -1 → D ≈ 2 → clipped to 1.
        """
        rank_before = np.argsort(w_before)
        rank_after = np.argsort(w_after)

        # Compute Spearman's rho between rankings
        rho, _ = spearmanr(rank_before, rank_after)
        rho = float(np.nan_to_num(rho, nan=1.0))

        # Disruption = 1 - rho, clipped to [0, 1]
        disruption = 1.0 - rho
        return float(np.clip(disruption, 0.0, 1.0))
