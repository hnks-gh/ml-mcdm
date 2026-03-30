"""
Bootstrap Analysis for ML Forecasting and Evidential Reasoning.

This module provides bootstrap-based uncertainty quantification for both 
machine learning forecasts and Evidential Reasoning (ER) aggregations. It 
enables the estimation of confidence intervals, feature importance 
stability, and belief distribution sensitivity.

Key Features
------------
- **Forecast Residual Bootstrap**: Generates prediction confidence 
  intervals by perturbing forecasts based on historical error 
  distributions.
- **Feature Importance Stability**: Assesses the reliability of feature 
  rankings across resampled data folds.
- **Model Contribution Uncertainty**: Uses Dirichlet perturbation to 
  estimate variance in ensemble model weighting.
- **ER Belief Bootstrap**: Quantifies the stability of final utility 
  scores and dominant grade assignments under belief noise.

References
----------
- Efron & Tibshirani (1993). "An Introduction to the Bootstrap." Chapman & Hall.
- Davison & Hinkley (1997). "Bootstrap Methods and Their Application." 
  Cambridge University Press.
- Yang & Xu (2002). "On the evidential reasoning algorithm for multiple 
  attribute decision analysis under uncertainty." IEEE Transactions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# Result containers
# ============================================================================

@dataclass
class ForecastBootstrapResult:
    """Result container for forecast bootstrap uncertainty analysis."""
    mean_predictions: pd.DataFrame
    ci_lower: pd.DataFrame          # Lower bound at (1-confidence)/2
    ci_upper: pd.DataFrame          # Upper bound at 1 - (1-confidence)/2
    std_predictions: pd.DataFrame
    feature_importance_mean: pd.Series
    feature_importance_std: pd.Series
    feature_importance_ci_lower: pd.Series
    feature_importance_ci_upper: pd.Series
    model_contribution_mean: Dict[str, float]
    model_contribution_std: Dict[str, float]
    effective_iterations: int
    convergence_rate: float

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            "FORECAST BOOTSTRAP RESULTS",
            f"{'='*70}",
            f"Effective iterations: {self.effective_iterations}",
            f"Convergence rate:     {self.convergence_rate:.1%}",
            f"\nMean prediction range: [{self.mean_predictions.values.min():.4f}, "
            f"{self.mean_predictions.values.max():.4f}]",
            f"\nTop 10 Most Stable Features (lowest importance CV):",
        ]
        if not self.feature_importance_mean.empty:
            cv = self.feature_importance_std / (self.feature_importance_mean.abs() + 1e-10)
            for feat, cv_val in cv.nsmallest(10).items():
                lines.append(f"  {feat}: CV={cv_val:.4f}")
        lines.append("=" * 70)
        return "\n".join(lines)


@dataclass
class ERBootstrapResult:
    """Result container for ER belief distribution bootstrap analysis."""
    mean_beliefs: Dict[str, np.ndarray]        # entity → mean belief vector
    std_beliefs: Dict[str, np.ndarray]         # entity → std belief vector
    ci_lower_beliefs: Dict[str, np.ndarray]    # entity → lower CI per grade
    ci_upper_beliefs: Dict[str, np.ndarray]    # entity → upper CI per grade
    mean_utility: Dict[str, float]             # entity → mean final utility
    std_utility: Dict[str, float]              # entity → utility std
    utility_ci: Dict[str, Tuple[float, float]] # entity → (lower, upper) CI
    grade_stability: Dict[str, float]          # entity → dominant-grade stability
    grades: List[str]
    effective_iterations: int
    convergence_rate: float

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            "EVIDENTIAL REASONING BOOTSTRAP RESULTS",
            f"{'='*70}",
            f"Effective iterations: {self.effective_iterations}",
            f"Convergence rate:     {self.convergence_rate:.1%}",
        ]
        if self.mean_utility:
            u_vals = list(self.mean_utility.values())
            lines.append(
                f"\nMean utility range: [{min(u_vals):.4f}, {max(u_vals):.4f}]"
            )
        if self.grade_stability:
            lines.append("\nGrade stability (top 5 most stable entities):")
            for entity, stab in sorted(
                self.grade_stability.items(), key=lambda x: x[1], reverse=True
            )[:5]:
                lines.append(f"  {entity}: {stab:.4f}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# ML Forecast Bootstrap
# ============================================================================

class ForecastBootstrap:
    """
    Bootstrap uncertainty quantification for ML forecast predictions.

    Implements residual bootstrap to generate prediction confidence intervals
    and feature importance stability estimates.

    Parameters
    ----------
    n_iterations : int, default=500
        Number of bootstrap iterations.
    seed : int, default=42
        Random seed for reproducibility.
    confidence : float, default=0.95
        Confidence level for credible intervals.
    """

    def __init__(
        self,
        n_iterations: int = 500,
        seed: int = 42,
        confidence: float = 0.95,
    ):
        """
        Initialize the forecast bootstrap analyzer.

        Parameters
        ----------
        n_iterations : int, default=500
            Number of bootstrap replicates to generate.
        seed : int, default=42
            Seed for reproducible random sampling.
        confidence : float, default=0.95
            The width of the credible interval (e.g., 0.95 for 95% CI).
        """
        self.n_iterations = n_iterations
        self.seed = seed
        self.confidence = confidence
        self._alpha = (1 - confidence) / 2

    def run(self, forecast_result) -> ForecastBootstrapResult:
        """
        Execute bootstrap analysis on a UnifiedForecastResult.

        Parameters
        ----------
        forecast_result : UnifiedForecastResult

        Returns
        -------
        ForecastBootstrapResult
        """
        rng = np.random.default_rng(self.seed)
        B = self.n_iterations
        lo_pct = self._alpha * 100
        hi_pct = (1 - self._alpha) * 100

        predictions = forecast_result.predictions
        uncertainty = forecast_result.uncertainty
        n_entities, n_comps = predictions.shape
        feature_importance = forecast_result.feature_importance

        # Residual bootstrap: perturb predictions by uncertainty
        boot_preds = np.zeros((B, n_entities, n_comps))
        fi_samples: List[np.ndarray] = []
        model_contrib_samples: Dict[str, List[float]] = {
            m: [] for m in forecast_result.model_contributions
        }
        failed = 0

        for b in range(B):
            try:
                noise = rng.normal(0, 1, (n_entities, n_comps))
                boot_pred = predictions.values + noise * uncertainty.values
                boot_preds[b] = boot_pred

                # Feature importance: resample components (columns)
                if feature_importance is not None and not feature_importance.empty:
                    n_c = feature_importance.shape[1]
                    if n_c > 1:
                        comp_idx = rng.choice(n_c, size=n_c, replace=True)
                        fi_boot = feature_importance.iloc[:, comp_idx].mean(axis=1).values
                    else:
                        noise_fi = rng.normal(0, 0.01, feature_importance.shape[0])
                        fi_boot = feature_importance.iloc[:, 0].values + noise_fi
                    fi_samples.append(np.abs(fi_boot))

                # Model contributions: Dirichlet perturbation
                contrib_vals = np.array(
                    list(forecast_result.model_contributions.values()), dtype=float
                )
                g = rng.exponential(1.0, size=len(contrib_vals))
                perturbed = contrib_vals * g / (contrib_vals * g).sum()
                for m, v in zip(model_contrib_samples.keys(), perturbed):
                    model_contrib_samples[m].append(float(v))

            except Exception as exc:
                failed += 1
                logger.warning(f"Forecast bootstrap iteration {b} failed: {exc}")
                boot_preds[b] = predictions.values

        mean_preds = np.mean(boot_preds, axis=0)
        ci_lo = np.percentile(boot_preds, lo_pct, axis=0)
        ci_hi = np.percentile(boot_preds, hi_pct, axis=0)
        std_preds = np.std(boot_preds, axis=0, ddof=1)

        to_df = lambda arr: pd.DataFrame(
            arr, index=predictions.index, columns=predictions.columns
        )

        # Feature importance statistics
        if fi_samples:
            fi_matrix = np.array(fi_samples)
            fi_index = feature_importance.index
            fi_mean = pd.Series(fi_matrix.mean(axis=0), index=fi_index)
            fi_std = pd.Series(fi_matrix.std(axis=0, ddof=1), index=fi_index)
            fi_lo_s = pd.Series(np.percentile(fi_matrix, lo_pct, axis=0), index=fi_index)
            fi_hi_s = pd.Series(np.percentile(fi_matrix, hi_pct, axis=0), index=fi_index)
        else:
            empty = pd.Series(dtype=float)
            fi_mean = fi_std = fi_lo_s = fi_hi_s = empty

        contrib_mean = {m: float(np.mean(vs)) for m, vs in model_contrib_samples.items()}
        contrib_std = {
            m: float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0
            for m, vs in model_contrib_samples.items()
        }

        convergence_rate = 1.0 - (failed / B)
        logger.info(
            f"ForecastBootstrap: {B - failed}/{B} successful "
            f"(convergence={convergence_rate:.1%})"
        )

        return ForecastBootstrapResult(
            mean_predictions=to_df(mean_preds),
            ci_lower=to_df(ci_lo),
            ci_upper=to_df(ci_hi),
            std_predictions=to_df(std_preds),
            feature_importance_mean=fi_mean,
            feature_importance_std=fi_std,
            feature_importance_ci_lower=fi_lo_s,
            feature_importance_ci_upper=fi_hi_s,
            model_contribution_mean=contrib_mean,
            model_contribution_std=contrib_std,
            effective_iterations=B - failed,
            convergence_rate=convergence_rate,
        )


# ============================================================================
# ER Bootstrap
# ============================================================================

class ERBootstrap:
    """
    Bootstrap uncertainty quantification for Evidential Reasoning aggregations.

    Perturbs input belief distributions using Gaussian noise then projects
    back onto the probability simplex to estimate uncertainty in final ER scores.

    Parameters
    ----------
    n_iterations : int, default=300
        Bootstrap iterations.
    seed : int, default=42
        Random seed.
    confidence : float, default=0.95
        Confidence level for credible intervals.
    noise_scale : float, default=0.05
        Standard deviation of Gaussian noise added to belief distributions.
    """

    def __init__(
        self,
        n_iterations: int = 300,
        seed: int = 42,
        confidence: float = 0.95,
        noise_scale: float = 0.05,
    ):
        """
        Initialize the ER bootstrap analyzer.

        Parameters
        ----------
        n_iterations : int, default=300
            Number of bootstrap replicates to generate.
        seed : int, default=42
            Seed for reproducible random sampling.
        confidence : float, default=0.95
            The width of the utility confidence interval.
        noise_scale : float, default=0.05
            Standard deviation of the Gaussian noise added to belief 
            distributions during perturbation.
        """
        self.n_iterations = n_iterations
        self.seed = seed
        self.confidence = confidence
        self.noise_scale = noise_scale
        self._alpha = (1 - confidence) / 2

    def run(self, er_result, engine=None) -> ERBootstrapResult:
        """
        Bootstrap ER aggregation uncertainty.

        Parameters
        ----------
        er_result : ERResult
            Result from EvidentialReasoningEngine or HierarchicalER.
        engine : EvidentialReasoningEngine, optional
            ER engine (used to obtain grade labels). If None, inferred from
            er_result or defaults to 5-grade scale.

        Returns
        -------
        ERBootstrapResult
        """
        from ranking.evidential_reasoning.base import BeliefDistribution, EvidentialReasoningEngine

        rng = np.random.default_rng(self.seed)
        B = self.n_iterations
        lo_pct = self._alpha * 100
        hi_pct = (1 - self._alpha) * 100

        if engine is None:
            engine = EvidentialReasoningEngine()

        grades = engine.grades
        N_grades = len(grades)

        if not hasattr(er_result, 'final_scores'):
            logger.warning("ERBootstrap: er_result has no final_scores")
            return self._empty_result(grades)

        entities = list(er_result.final_scores.index)
        n_entities = len(entities)
        belief_dists = getattr(er_result, 'belief_distributions', {})

        # Storage: (B, n_entities, N_grades) and (B, n_entities)
        boot_beliefs = np.zeros((B, n_entities, N_grades))
        boot_utilities = np.zeros((B, n_entities))
        failed = 0

        for b in range(B):
            try:
                for i, entity in enumerate(entities):
                    if entity in belief_dists:
                        beliefs = np.array(belief_dists[entity].beliefs, dtype=float)
                    else:
                        beliefs = np.ones(N_grades) / N_grades

                    noise = rng.normal(0, self.noise_scale, N_grades)
                    perturbed = np.clip(beliefs + noise, 0, None)
                    total = perturbed.sum()
                    if total > 1.0:
                        perturbed /= total

                    bd_boot = BeliefDistribution(grades=grades, beliefs=perturbed)
                    boot_beliefs[b, i] = bd_boot.beliefs
                    boot_utilities[b, i] = bd_boot.average_utility()

            except Exception as exc:
                failed += 1
                logger.warning(f"ER bootstrap iteration {b} failed: {exc}")
                for i, entity in enumerate(entities):
                    boot_utilities[b, i] = float(
                        er_result.final_scores.get(entity, 0.5)
                        if hasattr(er_result.final_scores, 'get')
                        else 0.5
                    )

        # Aggregate statistics per entity
        mean_beliefs: Dict[str, np.ndarray] = {}
        std_beliefs: Dict[str, np.ndarray] = {}
        ci_lo_beliefs: Dict[str, np.ndarray] = {}
        ci_hi_beliefs: Dict[str, np.ndarray] = {}
        mean_utility: Dict[str, float] = {}
        std_utility: Dict[str, float] = {}
        utility_ci: Dict[str, Tuple[float, float]] = {}
        grade_stability: Dict[str, float] = {}

        for i, entity in enumerate(entities):
            b_arr = boot_beliefs[:, i, :]
            u_arr = boot_utilities[:, i]

            mean_beliefs[entity] = b_arr.mean(axis=0)
            std_beliefs[entity] = b_arr.std(axis=0, ddof=1)
            ci_lo_beliefs[entity] = np.percentile(b_arr, lo_pct, axis=0)
            ci_hi_beliefs[entity] = np.percentile(b_arr, hi_pct, axis=0)

            mean_utility[entity] = float(u_arr.mean())
            std_utility[entity] = float(u_arr.std(ddof=1)) if B > 1 else 0.0
            utility_ci[entity] = (
                float(np.percentile(u_arr, lo_pct)),
                float(np.percentile(u_arr, hi_pct)),
            )

            # Grade stability: fraction of iterations with same dominant grade
            dominant = b_arr.argmax(axis=1)
            if len(dominant) > 0:
                mode = np.bincount(dominant, minlength=N_grades).argmax()
                grade_stability[entity] = float((dominant == mode).mean())
            else:
                grade_stability[entity] = 0.0

        convergence_rate = 1.0 - (failed / B)
        logger.info(
            f"ERBootstrap: {B - failed}/{B} successful "
            f"(convergence={convergence_rate:.1%})"
        )

        return ERBootstrapResult(
            mean_beliefs=mean_beliefs,
            std_beliefs=std_beliefs,
            ci_lower_beliefs=ci_lo_beliefs,
            ci_upper_beliefs=ci_hi_beliefs,
            mean_utility=mean_utility,
            std_utility=std_utility,
            utility_ci=utility_ci,
            grade_stability=grade_stability,
            grades=grades,
            effective_iterations=B - failed,
            convergence_rate=convergence_rate,
        )

    def _empty_result(self, grades: List[str]) -> ERBootstrapResult:
        return ERBootstrapResult(
            mean_beliefs={}, std_beliefs={},
            ci_lower_beliefs={}, ci_upper_beliefs={},
            mean_utility={}, std_utility={},
            utility_ci={}, grade_stability={},
            grades=grades, effective_iterations=0, convergence_rate=0.0,
        )


# ============================================================================
# Convenience functions
# ============================================================================

def forecast_bootstrap(
    forecast_result,
    n_iterations: int = 500,
    seed: int = 42,
    confidence: float = 0.95,
) -> ForecastBootstrapResult:
    """Convenience function: bootstrap CI for ML forecast predictions."""
    return ForecastBootstrap(
        n_iterations=n_iterations, seed=seed, confidence=confidence
    ).run(forecast_result)


def er_bootstrap(
    er_result,
    n_iterations: int = 300,
    seed: int = 42,
    confidence: float = 0.95,
    noise_scale: float = 0.05,
) -> ERBootstrapResult:
    """Convenience function: bootstrap uncertainty for ER belief distributions."""
    return ERBootstrap(
        n_iterations=n_iterations, seed=seed,
        confidence=confidence, noise_scale=noise_scale,
    ).run(er_result)
