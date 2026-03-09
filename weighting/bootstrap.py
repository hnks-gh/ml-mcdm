# -*- coding: utf-8 -*-
"""
Bayesian Bootstrap for CRITIC Weight Uncertainty Quantification
===============================================================

Rubin (1981) Bayesian Bootstrap applied to CRITIC-derived weights.
Each observation is assigned a Dirichlet random weight; the weighted
mean produces one bootstrap weight vector.  Convergence is measured
via the rolling coefficient of variation of the posterior mean.

References
----------
Rubin (1981). The Bayesian bootstrap. Ann. Stat. 9(1), 130-134.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WeightBootstrapResult:
    """Results from Bayesian weight bootstrap."""
    posterior_mean: pd.Series
    posterior_std: pd.Series
    credible_interval_lower: pd.Series
    credible_interval_upper: pd.Series
    n_iterations: int
    converged: bool
    convergence_history: List[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "\n" + "=" * 60,
            "WEIGHT BOOTSTRAP RESULTS",
            "=" * 60,
            f"Iterations:  {self.n_iterations}",
            f"Converged:   {self.converged}",
            f"\n{'Criterion':<30} {'Mean':>8} {'Std':>8} {'CI95_lo':>8} {'CI95_hi':>8}",
            "-" * 60,
        ]
        for crit in self.posterior_mean.index:
            lines.append(
                f"{str(crit):<30} "
                f"{self.posterior_mean[crit]:>8.4f} "
                f"{self.posterior_std[crit]:>8.4f} "
                f"{self.credible_interval_lower[crit]:>8.4f} "
                f"{self.credible_interval_upper[crit]:>8.4f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


class BayesianBootstrap:
    """
    Bayesian Bootstrap for CRITIC weight uncertainty quantification.

    Parameters
    ----------
    n_iterations : int, default=1000
    confidence_level : float, default=0.95
    convergence_threshold : float, default=0.01
    seed : int, default=42
    """

    def __init__(
        self,
        n_iterations: int = 1000,
        confidence_level: float = 0.95,
        convergence_threshold: float = 0.01,
        seed: int = 42,
    ):
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level
        self.convergence_threshold = convergence_threshold
        self.seed = seed

    def run(self, weight_matrix: pd.DataFrame) -> WeightBootstrapResult:
        """
        Run Bayesian bootstrap on a weight matrix.

        Parameters
        ----------
        weight_matrix : pd.DataFrame
            Shape (n_observations, n_criteria).
        """
        rng = np.random.default_rng(self.seed)
        n_obs, n_crit = weight_matrix.shape
        criteria = weight_matrix.columns.tolist()
        data = weight_matrix.values.astype(float)

        samples = np.zeros((self.n_iterations, n_crit))
        cv_history: List[float] = []
        converged = False

        for b in range(self.n_iterations):
            dirichlet_weights = rng.dirichlet(np.ones(n_obs))
            boot_weight = dirichlet_weights @ data
            if boot_weight.sum() > 1e-10:
                boot_weight /= boot_weight.sum()
            samples[b] = boot_weight

            if b >= 49 and (b + 1) % 10 == 0:
                cur_mean = samples[:b + 1].mean(axis=0)
                cur_std = samples[:b + 1].std(axis=0, ddof=1)
                cv = float(np.mean(cur_std / (np.abs(cur_mean) + 1e-10)))
                cv_history.append(cv)
                if cv < self.convergence_threshold:
                    samples = samples[:b + 1]
                    converged = True
                    break

        alpha = 1.0 - self.confidence_level
        lo = np.quantile(samples, alpha / 2, axis=0)
        hi = np.quantile(samples, 1.0 - alpha / 2, axis=0)
        post_mean = samples.mean(axis=0)
        post_std = samples.std(axis=0, ddof=1)

        return WeightBootstrapResult(
            posterior_mean=pd.Series(post_mean, index=criteria),
            posterior_std=pd.Series(post_std, index=criteria),
            credible_interval_lower=pd.Series(lo, index=criteria),
            credible_interval_upper=pd.Series(hi, index=criteria),
            n_iterations=len(samples),
            converged=converged,
            convergence_history=cv_history,
        )


def bayesian_bootstrap_weights(
    weight_matrix: pd.DataFrame,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    convergence_threshold: float = 0.01,
    seed: int = 42,
) -> WeightBootstrapResult:
    """Convenience function: Bayesian bootstrap on a CRITIC weight matrix."""
    return BayesianBootstrap(
        n_iterations=n_iterations,
        confidence_level=confidence_level,
        convergence_threshold=convergence_threshold,
        seed=seed,
    ).run(weight_matrix)
