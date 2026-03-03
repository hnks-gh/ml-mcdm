# -*- coding: utf-8 -*-
"""
Neural Additive Models (NAMs) Forecaster
==========================================

Interpretable neural networks that learn individual shape functions
for each feature, then combine them additively:

    ŷ = β₀ + Σⱼ fⱼ(xⱼ)

Each fⱼ is a small neural network (2-3 hidden layers) that takes a
single feature as input and outputs a scalar contribution. This
architecture provides:

    - Interpretability: Can plot each fⱼ(xⱼ) to see feature effects
    - Non-linearity: Each fⱼ can model complex shape functions
    - Regularization: Additive constraint is a strong structural prior
    - Small-data friendly: Fewer parameters than fully-connected nets

The additive structure keeps the model interpretable for the base case.
Optional pairwise interactions (NAM², Bug N-2 fix) can be enabled via
``include_interactions=True``, which adds cross-feature shape functions
$g_{jk}(x_j, x_k)$ for the top-$K$ most important feature pairs:

    ŷ = β₀ + Σⱼ fⱼ(xⱼ) + Σⱼ<ₖ gⱼₖ(xⱼ, xₖ)

Each $g_{jk}$ is a ``_PairwiseFeatureNetwork`` mapping $(x_j, x_k)$
through a 2D Random Fourier basis and then Ridge regression onto the
additive-model residual.  The pairwise terms are re-centred to zero
mean before being added to the ensemble prediction, preserving feature
identifiability.

Temporal Extension:
    For time series data, we add temporal basis functions:
    f_j(x_j, t) = MLP([x_j, sin(2πt/T), cos(2πt/T)])

Implementation:
    Uses sklearn-compatible implementation via Explainable Boosting
    Machine (EBM) from interpret library as primary backend, with a
    pure numpy/sklearn fallback for environments without interpret.

References:
    - Agarwal et al. (2021). "Neural Additive Models: Interpretable
      Machine Learning with Neural Nets" NeurIPS
    - Lou, Caruana, Gehrke (2012). "Intelligible Models for
      Classification and Regression" KDD (original GAM² concept)
    - Nori et al. (2019). "InterpretML: A Unified Framework for
      Machine Learning Interpretability" arXiv:1909.09223
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from .base import BaseForecaster


class _SingleFeatureNetwork:
    """
    Small neural network for a single feature's shape function.

    Architecture: x_j → [hidden₁, ReLU] → [hidden₂, ReLU] → output(1)

    Uses gradient-free optimization (closed-form Ridge on random
    features) for efficiency and stability on small datasets.

    This is a Random Kitchen Sink (RKS) approximation to a
    single-feature neural network, providing non-linear
    transformations without backpropagation.

    Parameters:
        n_basis: Number of random Fourier basis functions
        random_state: Random seed
    """

    def __init__(self, n_basis: int = 50, random_state: int = 42):
        self.n_basis = n_basis
        self.random_state = random_state
        self._W: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None
        self._coef: Optional[np.ndarray] = None
        self._intercept: float = 0.0
        self._is_fitted: bool = False

    def _transform(self, x: np.ndarray) -> np.ndarray:
        """Map single feature to random Fourier basis."""
        # Random Fourier Features (Rahimi & Recht, 2007)
        z = x.reshape(-1, 1) @ self._W.T + self._b
        return np.concatenate([np.cos(z), np.sin(z)], axis=1)

    def fit(self, x: np.ndarray, residual: np.ndarray, alpha: float = 1.0):
        """Fit the shape function on one feature's contribution."""
        rng = np.random.RandomState(self.random_state)
        self._W = rng.randn(self.n_basis, 1) * 2.0
        self._b = rng.uniform(0, 2 * np.pi, self.n_basis)

        Z = self._transform(x)
        # Solve Ridge regression on basis-expanded features
        ZtZ = Z.T @ Z + alpha * np.eye(Z.shape[1])
        Zty = Z.T @ residual
        self._coef = np.linalg.solve(ZtZ, Zty)
        self._intercept = 0.0
        self._is_fitted = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the shape function's contribution.

        Raises:
            RuntimeError: If called before :meth:`fit`, with a clear message
                instead of the uninformative ``TypeError`` that would arise
                from operating on ``None`` weight matrices.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "_SingleFeatureNetwork.predict() called before fit(). "
                "Call fit(x, residual) first to initialise the random Fourier "
                "basis and solve for coefficients."
            )
        Z = self._transform(x)
        return Z @ self._coef + self._intercept


class _PairwiseFeatureNetwork:
    """
    Shape function for a pair of features (pairwise interaction term).

    Used by ``NeuralAdditiveForecaster`` when ``include_interactions=True``
    to implement NAM² — an extension of the Neural Additive Model that adds
    pairwise interaction shape functions gⱼₖ(xⱼ, xₖ) on top of the main
    additive terms.

    Architecture matches ``_SingleFeatureNetwork`` but maps two input
    features to a 2D random-Fourier basis before Ridge regression.

    Parameters:
        n_basis: Number of random Fourier basis functions
        random_state: Random seed
    """

    def __init__(self, n_basis: int = 50, random_state: int = 42):
        self.n_basis = n_basis
        self.random_state = random_state
        self._W: Optional[np.ndarray] = None   # shape (n_basis, 2)
        self._b: Optional[np.ndarray] = None   # shape (n_basis,)
        self._coef: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def _transform(self, X2: np.ndarray) -> np.ndarray:
        """Map a 2-feature matrix to random Fourier basis."""
        z = X2 @ self._W.T + self._b          # (n, n_basis)
        return np.concatenate([np.cos(z), np.sin(z)], axis=1)

    def fit(self, X2: np.ndarray, residual: np.ndarray, alpha: float = 1.0):
        """Fit on the 2-feature partial residual.

        Args:
            X2: Array of shape (n_samples, 2) — two scaled features
            residual: Target partial residual of shape (n_samples,)
            alpha: Ridge regularisation strength
        """
        rng = np.random.RandomState(self.random_state)
        self._W = rng.randn(self.n_basis, 2) * 2.0
        self._b = rng.uniform(0, 2 * np.pi, self.n_basis)
        Z = self._transform(X2)
        ZtZ = Z.T @ Z + alpha * np.eye(Z.shape[1])
        Zty = Z.T @ residual
        self._coef = np.linalg.solve(ZtZ, Zty)
        self._is_fitted = True

    def predict(self, X2: np.ndarray) -> np.ndarray:
        """Predict interaction term contribution.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "_PairwiseFeatureNetwork.predict() called before fit(). "
                "Call fit(X2, residual) first."
            )
        Z = self._transform(X2)
        return Z @ self._coef


class NeuralAdditiveForecaster(BaseForecaster):
    """
    Neural Additive Model (NAM) forecaster.

    Learns individual shape functions fⱼ(xⱼ) for each feature using
    Random Kitchen Sink approximation, then combines additively.

    For time series panel data, the model structure is:
        ŷ = β₀ + Σⱼ fⱼ(xⱼ) + g(t)

    Where g(t) captures temporal dynamics via Fourier basis.

    The implementation uses a backfitting algorithm:
    1. Initialize: ŷ⁰ = mean(y)
    2. For each iteration:
       a. For each feature j:
          - Compute partial residual: r_j = y - ŷ + fⱼ(xⱼ)
          - Re-fit fⱼ on (xⱼ, r_j)
       b. Update: ŷ = β₀ + Σⱼ fⱼ(xⱼ)
    3. Until convergence

    Parameters:
        n_basis_per_feature: Number of random basis functions per feature
        n_iterations: Number of backfitting iterations
        learning_rate: Step size for backfitting updates
        regularization: L2 regularization strength for each fⱼ
        include_interactions: Whether to include pairwise interactions
                            for top features (NAM²)
        max_interaction_features: Max features for pairwise interactions
        random_state: Random seed

    Example:
        >>> nam = NeuralAdditiveForecaster(n_basis_per_feature=50)
        >>> nam.fit(X_train, y_train)
        >>> predictions = nam.predict(X_test)
        >>> shapes = nam.get_shape_functions(X_test)
    """

    def __init__(
        self,
        n_basis_per_feature: int = 50,
        n_iterations: int = 10,
        learning_rate: float = 0.8,
        regularization: float = 1.0,
        include_interactions: bool = False,
        max_interaction_features: int = 5,
        random_state: int = 42,
    ):
        self.n_basis_per_feature = n_basis_per_feature
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.include_interactions = include_interactions
        self.max_interaction_features = max_interaction_features
        self.random_state = random_state

        self._shape_functions: Dict[int, _SingleFeatureNetwork] = {}
        self._intercept: float = 0.0
        self._scaler: StandardScaler = StandardScaler()
        self._n_features: int = 0
        self._n_outputs: int = 1
        self._fitted: bool = False
        self.feature_importance_: Optional[np.ndarray] = None

        # For multi-output: one set of shape functions per output
        self._multi_output_shapes: Dict[int, Dict[int, _SingleFeatureNetwork]] = {}
        self._multi_output_intercepts: Dict[int, float] = {}

        # Pairwise interaction networks, keyed by output_idx -> (j, k) -> network
        self._multi_output_interaction_shapes: Dict[
            int, Dict[Tuple[int, int], _PairwiseFeatureNetwork]
        ] = {}

        # Per-output feature importance: stored for unified.py U-4 extraction
        # Shape: (n_outputs, n_features)
        self._per_output_importance_: Optional[np.ndarray] = None

    def _fit_single_output(
        self, X: np.ndarray, y: np.ndarray, output_idx: int = 0
    ) -> Tuple[
        Dict[int, _SingleFeatureNetwork],
        float,
        np.ndarray,
        Dict[Tuple[int, int], "_PairwiseFeatureNetwork"],
    ]:
        """Fit NAM for a single output using backfitting.

        Returns:
            Tuple of (shape_functions, intercept, importances, interaction_shapes)
            where ``interaction_shapes`` is empty unless ``include_interactions``
            is True.
        """
        n_samples, n_features = X.shape

        # Initialize shape functions
        shape_functions = {}
        for j in range(n_features):
            sf = _SingleFeatureNetwork(
                n_basis=self.n_basis_per_feature,
                random_state=self.random_state + j + output_idx * 1000,
            )
            shape_functions[j] = sf

        # Initialize predictions
        intercept = np.mean(y)
        current_contributions = {j: np.zeros(n_samples) for j in range(n_features)}
        current_pred = np.full(n_samples, intercept)

        # Backfitting iterations
        for iteration in range(self.n_iterations):
            for j in range(n_features):
                # Partial residual: what's left after removing all other contributions
                partial_residual = y - current_pred + current_contributions[j]

                # Fit shape function on partial residual
                shape_functions[j].fit(
                    X[:, j], partial_residual, alpha=self.regularization
                )

                # Update contribution with learning rate
                new_contribution = shape_functions[j].predict(X[:, j])
                current_contributions[j] = (
                    (1 - self.learning_rate) * current_contributions[j]
                    + self.learning_rate * new_contribution
                )

                # Re-center contributions (identifiability constraint)
                mean_contrib = np.mean(current_contributions[j])
                current_contributions[j] -= mean_contrib
                intercept += mean_contrib

                # Update total prediction
                current_pred = intercept + sum(current_contributions.values())

        # Compute feature importance (variance of each shape function)
        importances = np.array([
            np.var(current_contributions[j]) for j in range(n_features)
        ])
        # Normalize
        imp_sum = importances.sum()
        if imp_sum > 0:
            importances /= imp_sum

        # ------------------------------------------------------------------
        # Pairwise interaction fitting (NAM²)
        # Fit gⱼₖ(xⱼ, xₖ) on the residual of the additive-only model for
        # the top-K most important features.
        # ------------------------------------------------------------------
        interaction_shapes: Dict[Tuple[int, int], _PairwiseFeatureNetwork] = {}
        if self.include_interactions and n_features >= 2:
            k = min(self.max_interaction_features, n_features)
            top_features = list(
                np.argsort(importances)[::-1][:k]
            )
            additive_pred = current_pred.copy()
            for fi in range(len(top_features)):
                for fj in range(fi + 1, len(top_features)):
                    j, kk = top_features[fi], top_features[fj]
                    pair_key = (j, kk)
                    # Partial residual = y minus current additive predictions
                    interaction_residual = y - additive_pred
                    X2 = X[:, [j, kk]]
                    net = _PairwiseFeatureNetwork(
                        n_basis=self.n_basis_per_feature,
                        random_state=self.random_state + j * 10000 + kk + output_idx * 100000,
                    )
                    net.fit(X2, interaction_residual, alpha=self.regularization)
                    interaction_shapes[pair_key] = net
                    # Re-center and update additive_pred to avoid double-counting
                    contrib_interaction = net.predict(X2)
                    contrib_interaction -= np.mean(contrib_interaction)
                    additive_pred += contrib_interaction

        return shape_functions, intercept, importances, interaction_shapes

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralAdditiveForecaster":
        """
        Fit the Neural Additive Model using backfitting.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) or (n_samples, n_outputs)

        Returns:
            Self for method chaining
        """
        X_scaled = self._scaler.fit_transform(X)
        self._n_features = X_scaled.shape[1]

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._n_outputs = y.shape[1]

        all_importances = []

        for out_col in range(self._n_outputs):
            shapes, intercept, importances, interaction_shapes = self._fit_single_output(
                X_scaled, y[:, out_col], output_idx=out_col
            )
            self._multi_output_shapes[out_col] = shapes
            self._multi_output_intercepts[out_col] = intercept
            self._multi_output_interaction_shapes[out_col] = interaction_shapes
            all_importances.append(importances)

        # Average feature importance across outputs
        self.feature_importance_ = np.mean(all_importances, axis=0)

        # Per-output importance: shape (n_outputs, n_features) — used by unified.py
        self._per_output_importance_ = np.array(all_importances)  # (n_outputs, n_features)

        # Store first output's shapes as default
        self._shape_functions = self._multi_output_shapes.get(0, {})
        self._intercept = self._multi_output_intercepts.get(0, 0.0)
        self._fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using fitted shape functions.

        ŷ = β₀ + Σⱼ fⱼ(xⱼ)

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_outputs)
        """
        if not self._fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        X_scaled = self._scaler.transform(X)
        n_samples = X_scaled.shape[0]
        predictions = np.zeros((n_samples, self._n_outputs))

        for out_col in range(self._n_outputs):
            shapes = self._multi_output_shapes[out_col]
            intercept = self._multi_output_intercepts[out_col]

            pred = np.full(n_samples, intercept)
            for j in range(self._n_features):
                if j in shapes:
                    pred += shapes[j].predict(X_scaled[:, j])

            # Add pairwise interaction contributions (NAM² extension)
            if self.include_interactions and out_col in self._multi_output_interaction_shapes:
                for (j, k), inet in self._multi_output_interaction_shapes[out_col].items():
                    if j < X_scaled.shape[1] and k < X_scaled.shape[1]:
                        interaction_contrib = inet.predict(X_scaled[:, [j, k]])
                        interaction_contrib -= np.mean(interaction_contrib)
                        pred += interaction_contrib

            predictions[:, out_col] = pred

        if self._n_outputs == 1:
            return predictions.ravel()
        return predictions

    def get_shape_functions(
        self, X: np.ndarray, output_idx: int = 0
    ) -> Dict[int, np.ndarray]:
        """
        Evaluate shape functions for each feature at given input points.

        Useful for plotting and interpreting individual feature effects.

        Args:
            X: Feature matrix
            output_idx: Which output's shape functions to evaluate

        Returns:
            Dictionary mapping feature index to contribution array
        """
        if not self._fitted:
            raise ValueError("Model not fitted yet")

        X_scaled = self._scaler.transform(X)
        shapes = self._multi_output_shapes.get(output_idx, {})

        contributions = {}
        for j in range(self._n_features):
            if j in shapes:
                contributions[j] = shapes[j].predict(X_scaled[:, j])
            else:
                contributions[j] = np.zeros(X_scaled.shape[0])

        return contributions

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from shape function variances."""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet")
        return self.feature_importance_

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get model diagnostics and configuration."""
        return {
            "n_features": self._n_features,
            "n_outputs": self._n_outputs,
            "n_basis_per_feature": self.n_basis_per_feature,
            "n_iterations": self.n_iterations,
            "regularization": self.regularization,
            "include_interactions": self.include_interactions,
            "feature_importance_top10": (
                dict(enumerate(np.argsort(self.feature_importance_)[::-1][:10]))
                if self.feature_importance_ is not None
                else {}
            ),
        }
