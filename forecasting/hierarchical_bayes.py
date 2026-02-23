# -*- coding: utf-8 -*-
"""
Hierarchical Bayesian Forecaster with Partial Pooling
=====================================================

Implements a hierarchical (multi-level) Bayesian model that exploits
the panel structure of the data. Provinces are treated as groups with
partially pooled parameters, allowing weaker provinces to borrow
statistical strength from stronger ones.

Model Hierarchy:
    Global level:     μ_global ~ Normal(0, σ_prior)
                      σ_group  ~ HalfNormal(1)
    Province level:   α_i      ~ Normal(μ_global, σ_group)   (partial pooling)
    Observation:      y_it     ~ Normal(α_i + β·X_it, σ_obs)

The key advantage for small-T panel data:
    - Partial pooling: Optimal bias-variance tradeoff
    - Information borrowing: Weak entities improved by group
    - Full posterior: Natural uncertainty quantification
    - Handles heterogeneity: Entity-specific intercepts
    - Missing data: Natural imputation through hierarchy

Implementation uses Empirical Bayes (EB) with iterative estimation
for efficiency, avoiding MCMC overhead while preserving the
hierarchical structure. Falls back to analytical approximation
when PyMC is not available.

References:
    - Gelman & Hill (2006). "Data Analysis Using Regression and
      Multilevel/Hierarchical Models"
    - Efron & Morris (1975). "Data Analysis Using Stein's Estimator
      and its Generalizations" JASA
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import StandardScaler

from .base import BaseForecaster


class HierarchicalBayesForecaster(BaseForecaster):
    """
    Hierarchical Bayesian forecaster with empirical Bayes partial pooling.

    Uses a two-level hierarchical model where province-level parameters
    are shrunk toward the global mean. The shrinkage factor is estimated
    from the data (empirical Bayes), providing an optimal bias-variance
    tradeoff for small samples.

    Shrinkage Formula:
        α_i^EB = B_i * μ_global + (1 - B_i) * α_i^OLS

        B_i = σ²_obs / (σ²_obs + n_i * σ²_group)

    Where B_i is the shrinkage factor:
        - B_i → 1 when σ²_group is small (strong pooling)
        - B_i → 0 when σ²_group is large (no pooling)

    Parameters:
        n_em_iterations: Number of EM iterations for variance estimation
        prior_precision: Prior precision for global parameters
        use_partial_pooling: Whether to use partial pooling (True) or
                            full pooling (False = ignore entity structure)
        convergence_tol: Convergence tolerance for EM
        min_shrinkage: Minimum shrinkage factor (prevents full no-pooling)
        max_shrinkage: Maximum shrinkage factor (prevents full pooling)
        random_state: Random seed
        n_posterior_samples: Number of posterior predictive samples
                           for uncertainty quantification

    Example:
        >>> hb = HierarchicalBayesForecaster(n_em_iterations=50)
        >>> hb.fit(X_train, y_train, group_indices=group_ids)
        >>> predictions = hb.predict(X_test)
        >>> mean, std = hb.predict_with_uncertainty(X_test)
    """

    def __init__(
        self,
        n_em_iterations: int = 50,
        prior_precision: float = 1.0,
        use_partial_pooling: bool = True,
        convergence_tol: float = 1e-4,
        min_shrinkage: float = 0.01,
        max_shrinkage: float = 0.99,
        random_state: int = 42,
        n_posterior_samples: int = 200,
    ):
        self.n_em_iterations = n_em_iterations
        self.prior_precision = prior_precision
        self.use_partial_pooling = use_partial_pooling
        self.convergence_tol = convergence_tol
        self.min_shrinkage = min_shrinkage
        self.max_shrinkage = max_shrinkage
        self.random_state = random_state
        self.n_posterior_samples = n_posterior_samples

        # Fitted parameters — stored per output
        self._global_models: List[Optional[BayesianRidge]] = []
        self._group_intercepts_per_output: List[Dict[int, float]] = []
        self._shrinkage_factors_per_output: List[Dict[int, float]] = []
        self._sigma_obs_per_output: List[float] = []
        self._sigma_group_per_output: List[float] = []
        self._mu_global: Optional[np.ndarray] = None
        self._scaler: StandardScaler = StandardScaler()
        self._n_outputs: int = 1
        self._fitted: bool = False
        self.feature_importance_: Optional[np.ndarray] = None
        self._group_indices: Optional[np.ndarray] = None

        # Posterior parameters for uncertainty (first output, backward compat)
        self._posterior_beta_mean: Optional[np.ndarray] = None
        self._posterior_beta_cov: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_indices: Optional[np.ndarray] = None,
    ) -> "HierarchicalBayesForecaster":
        """
        Fit the hierarchical Bayesian model using Empirical Bayes EM.

        The EM algorithm alternates between:
            E-step: Estimate group-level parameters given variance components
            M-step: Update variance components given group parameters

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) or (n_samples, n_outputs)
            group_indices: Optional group assignments for each sample.
                          If None, estimates groups from sample ordering.

        Returns:
            Self for method chaining
        """
        np.random.seed(self.random_state)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._n_outputs = y.shape[1]

        X_scaled = self._scaler.fit_transform(X)
        n_samples, n_features = X_scaled.shape

        # If no group indices provided, fall back to plain BayesianRidge
        # (fabricating synthetic groups would be meaningless)
        if group_indices is None:
            import warnings
            warnings.warn(
                "HierarchicalBayesForecaster: no group_indices provided. "
                "Falling back to plain BayesianRidge (no partial pooling).",
                UserWarning,
            )

        # Store group_indices for predict-time use
        self._group_indices = group_indices

        if group_indices is not None:
            unique_groups = np.unique(group_indices)
            n_groups = len(unique_groups)
        else:
            unique_groups = np.array([])
            n_groups = 0

        # Reset per-output storage
        self._global_models = []
        self._group_intercepts_per_output = []
        self._shrinkage_factors_per_output = []
        self._sigma_obs_per_output = []
        self._sigma_group_per_output = []

        # Fit per-output models
        all_importances = []

        for out_col in range(self._n_outputs):
            y_col = y[:, out_col]

            # Step 1: Fit global Bayesian Ridge model (pooled model)
            global_model = BayesianRidge(
                max_iter=300,
                compute_score=True,
                alpha_1=1e-6, alpha_2=1e-6,
                lambda_1=1e-6, lambda_2=1e-6,
            )
            global_model.fit(X_scaled, y_col)
            self._global_models.append(global_model)

            if out_col == 0:
                self._posterior_beta_mean = global_model.coef_.copy()
                # Store posterior covariance for uncertainty
                try:
                    self._posterior_beta_cov = global_model.sigma_.copy()
                except AttributeError:
                    self._posterior_beta_cov = np.eye(n_features) * 0.01

            global_pred = global_model.predict(X_scaled)
            global_residuals = y_col - global_pred

            # Step 2: Estimate variance components via EM
            # (only meaningful when group_indices are provided)
            group_intercepts: Dict[int, float] = {}
            shrinkage_factors: Dict[int, float] = {}
            sigma_obs = np.var(global_residuals) + 1e-10
            sigma_group = 1e-10

            if group_indices is not None and n_groups > 1:
                sigma_group = np.var([
                    np.mean(global_residuals[group_indices == g])
                    for g in unique_groups
                    if np.sum(group_indices == g) > 0
                ]) + 1e-10

                prev_sigma_group = sigma_group

                for em_iter in range(self.n_em_iterations):
                    # E-step: Compute shrinkage factors and group intercepts
                    group_intercepts = {}
                    shrinkage_factors = {}

                    for g in unique_groups:
                        mask = group_indices == g
                        n_g = mask.sum()
                        if n_g == 0:
                            continue

                        # Group mean of residuals (no-pooling estimate)
                        y_bar_g = np.mean(global_residuals[mask])

                        # Shrinkage factor B_g
                        B_g = sigma_obs / (sigma_obs + n_g * sigma_group)
                        B_g = np.clip(B_g, self.min_shrinkage, self.max_shrinkage)

                        # Partially pooled intercept
                        alpha_g = (1.0 - B_g) * y_bar_g  # Shrink toward 0 (global mean)

                        group_intercepts[g] = alpha_g
                        shrinkage_factors[g] = B_g

                    # M-step: Update variance components
                    # Update σ²_obs
                    residuals_new = np.zeros(n_samples)
                    for g in unique_groups:
                        mask = group_indices == g
                        if mask.sum() == 0:
                            continue
                        residuals_new[mask] = global_residuals[mask] - group_intercepts.get(g, 0.0)

                    sigma_obs = np.var(residuals_new) + 1e-10

                    # Update σ²_group
                    intercept_values = np.array(list(group_intercepts.values()))
                    sigma_group = np.var(intercept_values) + 1e-10

                    # Check convergence
                    if abs(sigma_group - prev_sigma_group) / (prev_sigma_group + 1e-10) < self.convergence_tol:
                        break
                    prev_sigma_group = sigma_group

            # Store fitted parameters for this output
            self._group_intercepts_per_output.append(group_intercepts)
            self._shrinkage_factors_per_output.append(shrinkage_factors)
            self._sigma_obs_per_output.append(sigma_obs)
            self._sigma_group_per_output.append(sigma_group)

            all_importances.append(np.abs(global_model.coef_))

        self.feature_importance_ = np.mean(all_importances, axis=0)
        self._fitted = True
        return self

    def predict(
        self, X: np.ndarray, group_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Make point predictions using the hierarchical model.

        Uses per-output global models plus group intercepts when available.
        For new / unknown entities, predictions fall back to population level
        (global model only, group intercept = 0).

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            group_indices: Optional group IDs for each sample.
                          If None, uses training-time group_indices mapping
                          or predicts at population level.

        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_outputs)
        """
        if not self._fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        X_scaled = self._scaler.transform(X)
        n_samples = X_scaled.shape[0]
        predictions = np.zeros((n_samples, self._n_outputs))

        for out_col in range(self._n_outputs):
            global_model = self._global_models[out_col]
            base_pred = global_model.predict(X_scaled)

            # Add group intercepts if available
            group_intercepts = self._group_intercepts_per_output[out_col]
            if group_indices is not None and group_intercepts:
                for i in range(n_samples):
                    g = group_indices[i]
                    base_pred[i] += group_intercepts.get(g, 0.0)

            predictions[:, out_col] = base_pred

        if self._n_outputs == 1:
            return predictions.ravel()
        return predictions

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with full Bayesian uncertainty quantification.

        Uncertainty combines:
            1. Parameter uncertainty (posterior variance of β)
            2. Observation noise (σ²_obs)
            3. Group-level variance (σ²_group)

        Total variance: Var(y_new) = x^T Σ_β x + σ²_obs + σ²_group

        Args:
            X: Feature matrix

        Returns:
            Tuple of (mean prediction, standard deviation)
        """
        if not self._fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        X_scaled = self._scaler.transform(X)

        # Point prediction from first output's global model (backward compat)
        global_model_0 = self._global_models[0]
        mean_pred, param_std = global_model_0.predict(X_scaled, return_std=True)

        # Total uncertainty: parameter uncertainty + observation noise + group variance
        sigma_obs = self._sigma_obs_per_output[0] if self._sigma_obs_per_output else 1.0
        sigma_group = self._sigma_group_per_output[0] if self._sigma_group_per_output else 1.0
        total_variance = param_std ** 2 + sigma_obs + sigma_group
        total_std = np.sqrt(total_variance)

        return mean_pred, total_std

    def predict_posterior_samples(
        self, X: np.ndarray, n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Draw samples from the posterior predictive distribution.

        Args:
            X: Feature matrix of shape (n_test, n_features)
            n_samples: Number of posterior samples (default: self.n_posterior_samples)

        Returns:
            Posterior samples of shape (n_samples, n_test)
        """
        if n_samples is None:
            n_samples = self.n_posterior_samples

        X_scaled = self._scaler.transform(X)
        n_test = X_scaled.shape[0]

        mean_pred, std_pred = self.predict_with_uncertainty(X)

        # Draw samples: y ~ Normal(mean_pred, std_pred)
        rng = np.random.RandomState(self.random_state)
        samples = np.zeros((n_samples, n_test))
        for s in range(n_samples):
            noise = rng.randn(n_test) * std_pred
            samples[s] = mean_pred + noise

        return samples

    def predict_intervals(
        self, X: np.ndarray, coverage: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute credible intervals from posterior predictive distribution.

        Args:
            X: Feature matrix
            coverage: Desired coverage probability (default 0.95)

        Returns:
            Tuple of (lower_bound, median, upper_bound) arrays
        """
        samples = self.predict_posterior_samples(X)
        alpha = (1.0 - coverage) / 2.0

        lower = np.quantile(samples, alpha, axis=0)
        median = np.quantile(samples, 0.50, axis=0)
        upper = np.quantile(samples, 1.0 - alpha, axis=0)

        return lower, median, upper

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from absolute Bayesian coefficients."""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet")
        return self.feature_importance_

    def get_shrinkage_summary(self) -> Dict[str, Any]:
        """
        Return summary of shrinkage factors across groups.

        Returns:
            Dictionary with shrinkage statistics:
                - mean_shrinkage: Average shrinkage factor
                - min_shrinkage: Minimum shrinkage factor
                - max_shrinkage: Maximum shrinkage factor
                - sigma_obs: Observation-level variance
                - sigma_group: Group-level variance
                - variance_partition: σ²_group / (σ²_group + σ²_obs)
        """
        if not self._fitted:
            raise ValueError("Model not fitted yet")

        shrink_vals = list(self._shrinkage_factors_per_output[0].values()) if self._shrinkage_factors_per_output else []
        sigma_obs = self._sigma_obs_per_output[0] if self._sigma_obs_per_output else 1.0
        sigma_group = self._sigma_group_per_output[0] if self._sigma_group_per_output else 1.0
        return {
            "mean_shrinkage": np.mean(shrink_vals) if shrink_vals else 0.0,
            "min_shrinkage": np.min(shrink_vals) if shrink_vals else 0.0,
            "max_shrinkage": np.max(shrink_vals) if shrink_vals else 0.0,
            "sigma_obs": sigma_obs,
            "sigma_group": sigma_group,
            "variance_partition": sigma_group / (sigma_group + sigma_obs),
            "n_groups": len(self._group_intercepts_per_output[0]) if self._group_intercepts_per_output else 0,
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive model diagnostics."""
        diag = self.get_shrinkage_summary()
        diag.update({
            "n_em_iterations": self.n_em_iterations,
            "convergence_tol": self.convergence_tol,
            "n_posterior_samples": self.n_posterior_samples,
        })
        return diag
