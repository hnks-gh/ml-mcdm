# -*- coding: utf-8 -*-
"""
Feature Preprocessing for Panel Forecasting
============================================

Provides dimensionality reduction to address the p >> n problem that
arises when ~400 temporal features are generated for panels with only
~100–700 training rows.

Pipeline:
    1. VarianceThreshold — removes near-constant columns (provinces
       with short histories produce many near-zero lag/rolling features).
    2. StandardScaler — centres and scales before PCA so that features
       with larger raw scales do not dominate the principal components.
    3. PCA — collapses highly correlated lag/rolling/cross-entity feature
       groups into a compact set of orthogonal components.

The component cap ``min(n_samples // 5, max_components)`` ensures that
every CV fold has at least 5 rows per principal component, keeping the
effective p/n ratio safely below 0.2.

Interpretability:
    ``inverse_importance()`` maps per-PC importance vectors back to the
    original feature space via the PCA loading matrix ``|components_|``,
    so downstream CSV reports retain original feature names.

References:
    - Bai & Ng (2002). "Determining the Number of Factors in Approximate
      Factor Models" Econometrica
"""

import numpy as np
from typing import Literal, Optional, List
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


class PanelFeatureReducer:
    """
    Two-stage dimensionality reduction for panel forecasting features.

    Stage A: VarianceThreshold removes near-constant columns.
    Stage B: PCA retains 95 % of variance, capped at
             ``min(n_samples // 5, max_components)`` components.

    Parameters
    ----------
    variance_threshold : float
        Minimum variance to keep a feature (default 0.01).
    pca_variance_ratio : float
        Cumulative variance ratio for PCA (default 0.95).
    max_components : int
        Hard cap on the number of PCA components (default 60).
    random_state : int
        Random seed for PCA solver.
    mode : {'pca', 'threshold_only'}
        Reduction mode.
        ``'pca'`` (default): VarianceThreshold → StandardScaler → PCA.
        ``'threshold_only'``: VarianceThreshold → StandardScaler, no PCA.
        Tree-based models work best with the original feature structure, so
        use ``'threshold_only'`` for them and ``'pca'`` for linear models.
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        pca_variance_ratio: float = 0.95,
        max_components: int = 60,
        random_state: int = 42,
        mode: Literal['pca', 'threshold_only'] = 'pca',
    ):
        self.variance_threshold = variance_threshold
        self.pca_variance_ratio = pca_variance_ratio
        self.max_components = max_components
        self.random_state = random_state
        self.mode = mode

        self._var_selector: Optional[VarianceThreshold] = None
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._original_feature_names: Optional[List[str]] = None
        self._kept_feature_mask: Optional[np.ndarray] = None
        self._n_components_fitted: int = 0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> "PanelFeatureReducer":
        """
        Fit the reduction pipeline on training features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        feature_names : list of str, optional
            Original feature names for inverse importance mapping.

        Returns
        -------
        self
        """
        n_samples, n_features = X.shape
        self._original_feature_names = (
            feature_names if feature_names is not None
            else [f"f{i}" for i in range(n_features)]
        )

        # Stage A: remove near-constant features
        self._var_selector = VarianceThreshold(threshold=self.variance_threshold)
        X_var = self._var_selector.fit_transform(X)
        self._kept_feature_mask = self._var_selector.get_support()

        # Stage B: standardise before PCA
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_var)

        if self.mode == 'threshold_only':
            # Skip PCA: keep all variance-filtered, standardised features.
            # n_components equals the count of features that survived Stage A.
            self._pca = None
            self._n_components_fitted = X_var.shape[1]
        else:
            # Stage C: PCA with adaptive component cap.
            #
            # The cap n_cap = min(n_samples // 5, max_components) enforces p/n <= 0.2
            # in every CV fold, preventing over-fitting in the PCA-compressed space.
            #
            # Two-step logic:
            #   Step 1 — fit using the variance-ratio criterion so sklearn
            #            automatically selects the minimal set of PCs that together
            #            explain >= pca_variance_ratio of variance.
            #   Step 2 — if the variance criterion selected more components than the
            #            p/n cap allows, refit with the integer cap.
            n_cap = min(n_samples // 5, self.max_components, X_var.shape[1])
            n_cap = max(n_cap, 2)  # always keep at least 2 components

            # Step 1: variance-ratio criterion
            self._pca = PCA(
                n_components=self.pca_variance_ratio,
                svd_solver="full",
                random_state=self.random_state,
            )
            self._pca.fit(X_scaled)

            # Step 2: enforce the p/n <= 0.2 hard cap
            if self._pca.n_components_ > n_cap:
                self._pca = PCA(
                    n_components=n_cap,
                    svd_solver="full",
                    random_state=self.random_state,
                )
                self._pca.fit(X_scaled)

            self._n_components_fitted = self._pca.n_components_

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features through the fitted pipeline.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_reduced : ndarray of shape (n_samples, n_components)
        """
        if not self._fitted:
            raise ValueError("PanelFeatureReducer not fitted. Call fit() first.")
        X_var = self._var_selector.transform(X)
        X_scaled = self._scaler.transform(X_var)
        if self.mode == 'threshold_only':
            return X_scaled
        return self._pca.transform(X_scaled)

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(X, feature_names=feature_names)
        return self.transform(X)

    def inverse_importance(
        self, pc_importance: np.ndarray
    ) -> np.ndarray:
        """
        Map per-PC importance back to the original feature space.

        For a vector ``w`` of shape ``(n_components,)`` or a matrix
        ``(n_components, n_outputs)``, returns a vector/matrix of shape
        ``(n_original_features,)`` / ``(n_original_features, n_outputs)``
        by distributing each PC's importance across original features
        proportionally to ``|PCA.components_|``.

        Parameters
        ----------
        pc_importance : ndarray
            Importance weights in PCA-space.

        Returns
        -------
        original_importance : ndarray
            Importance weights in original-feature-space.
        """
        if not self._fitted:
            raise ValueError("PanelFeatureReducer not fitted.")

        squeeze = False
        if pc_importance.ndim == 1:
            pc_importance = pc_importance[:, np.newaxis]
            squeeze = True

        n_original = len(self._original_feature_names)

        if self.mode == 'threshold_only':
            # Identity pass-through: importance is already in the kept-feature
            # space (no PCA backprojection needed); just expand to n_original.
            full_importance = np.zeros((n_original, pc_importance.shape[1]))
            full_importance[self._kept_feature_mask] = pc_importance
            if squeeze:
                return full_importance.ravel()
            return full_importance

        loadings = np.abs(self._pca.components_)  # (n_pc, n_kept_features)

        # Normalise each PC's loadings to sum to 1 so that larger-loading
        # features receive proportionally more of that PC's importance.
        row_sums = loadings.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        loadings_normed = loadings / row_sums

        # (n_kept_features, n_outputs) = loadings_normed.T @ pc_importance
        kept_importance = loadings_normed.T @ pc_importance

        # Expand back to the full original feature set (fill 0 for removed cols)
        full_importance = np.zeros((n_original, kept_importance.shape[1]))
        full_importance[self._kept_feature_mask] = kept_importance

        if squeeze:
            return full_importance.ravel()
        return full_importance

    @property
    def n_components(self) -> int:
        """Number of PCA components after fitting."""
        return self._n_components_fitted

    @property
    def explained_variance_ratio(self) -> float:
        """Total explained variance ratio of the fitted PCA."""
        if self._pca is not None:
            return float(np.sum(self._pca.explained_variance_ratio_))
        if self.mode == 'threshold_only' and self._fitted:
            return 1.0  # threshold-only retains all variance of kept features
        return 0.0

    def get_summary(self) -> str:
        """Human-readable summary of the reduction."""
        if not self._fitted:
            return "PanelFeatureReducer: not fitted"
        n_orig = len(self._original_feature_names)
        n_kept = int(self._kept_feature_mask.sum())
        if self.mode == 'threshold_only':
            return (
                f"PanelFeatureReducer (threshold_only): "
                f"{n_orig} → {n_kept} features (variance filter, no PCA)"
            )
        return (
            f"PanelFeatureReducer: {n_orig} → {n_kept} (variance filter) "
            f"→ {self._n_components_fitted} PCs "
            f"({self.explained_variance_ratio:.1%} variance retained)"
        )
