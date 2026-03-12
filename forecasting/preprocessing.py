# -*- coding: utf-8 -*-
"""
Feature Preprocessing for Panel Forecasting
============================================

Provides two complementary dimensionality-reduction tracks for the
dual-model architecture in UnifiedForecaster.

LINEAR TRACK — mode='pls'
    VarianceThreshold(0.005) → StandardScaler → [MI pre-filter] → PLSRegression
    Supervised compression: PLSRegression (PLS2) finds the linear combinations
    of X with maximum covariance with all 8 criterion targets simultaneously.
    Target-aware compression is strictly superior to PCA for forecasting tasks
    where the objective is prediction accuracy, not explained feature variance.
    n_components = min(n_samples // 10, 20) → p/n ≤ 0.024.

TREE TRACK — mode='threshold_only'
    VarianceThreshold(0.005) → raw features (no scaling, no compression).
    Tree-based models (CatBoost, QRF, NAM) are scale-invariant.  StandardScaler
    has been removed to prevent double-scaling with models that apply their own
    internal scaling (QRF: RobustScaler; PanelVAR: per-column StandardScaler;
    CatBoost: scale-invariant by design).

LEGACY — mode='pca'
    VarianceThreshold → StandardScaler → PCA.
    Retained for backward compatibility.  Prefer mode='pls' for new usage.

IMPUTATION — use_mice_imputation=True
    Prepends IterativeImputer(RandomForestRegressor) before VarianceThreshold
    in any mode.  Activated only when residual NaN values are detected after
    Phase-1 median-imputation (edge case; not active in normal operation).

MI Pre-filter (P-03) — mi_prefilter=True (default, active in 'pls' mode only)
    SelectKBest(mutual_info_regression) applied per output column; union of
    top-k features kept.  Removes ~50 % of low-signal features before PLS,
    improving component alignment with the prediction objective.

References
----------
- Mevik & Wehrens (2007). "The pls Package." JSS 18(2).
- Bair et al. (2006). "Prediction by Supervised Principal Components." JASA.
- Groen & Kapetanios (2016). "Revisiting Useful Approaches to Data-Rich
  Macroeconomic Forecasting." Computational Statistics & Data Analysis.
"""

import numpy as np
from typing import List, Literal, Optional, Union

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 — required before IterativeImputer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_regression
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler


class PanelFeatureReducer:
    """
    Multi-mode dimensionality reduction for panel forecasting features.

    Parameters
    ----------
    variance_threshold : float
        Minimum variance to keep a feature (default 0.005).
        Lowered from the previous 0.01 to avoid discarding stable but
        predictive governance criteria with legitimately low variance.
    pca_variance_ratio : float
        Cumulative variance ratio for PCA in ``'pca'`` legacy mode (default 0.95).
    max_components : int
        Hard cap on PCA components in ``'pca'`` legacy mode (default 60).
    n_pls_components : int or None
        Number of PLS components in ``'pls'`` mode.
        ``None`` (default): auto = ``min(n_samples // 10, 20)``.
    mi_prefilter : bool
        If ``True`` (default), apply a per-output mutual-information pre-filter
        before PLS.  The union of the top-``mi_k`` features across all outputs
        is kept, removing low-signal features before PLS fitting.
        Only active when ``mode='pls'`` and ``y`` is provided to ``fit()``.
    mi_k : int or 'half'
        Number of features to retain per MI selection.
        ``'half'`` (default): keep ``n_features // 2`` features.
    use_mice_imputation : bool
        If ``True``, apply ``IterativeImputer`` (RandomForestRegressor) before
        ``VarianceThreshold`` to handle residual NaN values.
    random_state : int
        Random seed for PCA, PLS, and imputation solvers.
    mode : {'pls', 'threshold_only', 'pca'}
        Reduction mode.
        ``'pls'``: Supervised PLS compression — use for linear models.
        ``'threshold_only'``: Variance filter only, no scaling/compression —
            use for tree-based models (CatBoost, QRF, NAM, PanelVAR).
        ``'pca'``: Legacy PCA compression — retained for backward compatibility.
    """

    def __init__(
        self,
        variance_threshold: float = 0.005,
        pca_variance_ratio: float = 0.95,
        max_components: int = 60,
        n_pls_components: Optional[int] = None,
        mi_prefilter: bool = True,
        mi_k: Union[int, str] = 'half',
        use_mice_imputation: bool = True,  # M-02: Activated by default
        random_state: int = 42,
        mode: Literal['pls', 'threshold_only', 'pca'] = 'pca',
    ):
        self.variance_threshold = variance_threshold
        self.pca_variance_ratio = pca_variance_ratio
        self.max_components = max_components
        self.n_pls_components = n_pls_components
        self.mi_prefilter = mi_prefilter
        self.mi_k = mi_k
        self.use_mice_imputation = use_mice_imputation
        self.random_state = random_state
        self.mode = mode

        self._var_selector: Optional[VarianceThreshold] = None
        self._scaler: Optional[StandardScaler] = None
        self._pls: Optional[PLSRegression] = None
        self._pca: Optional[PCA] = None
        self._imputer: Optional[IterativeImputer] = None
        self._mi_mask: Optional[np.ndarray] = None   # bool (n_kept_after_var,)
        self._pls_evr_: float = 0.0                  # estimated X-variance ratio for PLS
        self._original_feature_names: Optional[List[str]] = None
        self._kept_feature_mask: Optional[np.ndarray] = None
        self._n_components_fitted: int = 0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "PanelFeatureReducer":
        """
        Fit the reduction pipeline on training features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs), optional
            Training targets.  Required for ``mode='pls'`` to fit the
            supervised compression.  Ignored in ``'threshold_only'`` and
            ``'pca'`` modes.
        feature_names : list of str, optional
            Original feature names for ``inverse_importance()`` mapping.

        Returns
        -------
        self
        """
        n_samples, n_features = X.shape
        self._original_feature_names = (
            feature_names if feature_names is not None
            else [f"f{i}" for i in range(n_features)]
        )

        # M-02/M-08: Enhanced MICE imputation with MissForest configuration
        # Uses ExtraTreesRegressor (not RandomForest) for better handling of
        # correlated governance features. ExtraTrees provides lower variance
        # predictions due to extreme randomization of splits.
        # Enhanced parameters: max_iter=20 for convergence, add_indicator=True
        # for explicit missingness flags, n_nearest_features limits to most
        # informationally relevant neighbors.
        if self.use_mice_imputation and np.isnan(X).any():
            from sklearn.ensemble import ExtraTreesRegressor
            self._imputer = IterativeImputer(
                estimator=ExtraTreesRegressor(
                    n_estimators=100,      # stability: was 50 in original MICE
                    max_depth=6,           # avoid overfitting on N≈756
                    min_samples_leaf=3,    # M-08: conservative for N=63 provinces
                    max_features='sqrt',   # reduces tree correlation
                    bootstrap=True,
                    random_state=self.random_state,
                ),
                max_iter=20,               # M-08: convergence-based (was 5, Phase 1: 15)
                tol=1e-3,                  # convergence tolerance (sklearn default, now explicit)
                initial_strategy='median', # robust to outliers
                n_nearest_features=min(20, n_features),  # efficiency
                add_indicator=True,        # append missingness flags
                random_state=self.random_state,
            )
            X = self._imputer.fit_transform(X)
        else:
            self._imputer = None

        # Stage A: remove near-constant features (P-01: threshold=0.005)
        self._var_selector = VarianceThreshold(threshold=self.variance_threshold)
        X_var = self._var_selector.fit_transform(X)
        self._kept_feature_mask = self._var_selector.get_support()

        # ── TREE TRACK ────────────────────────────────────────────────────
        if self.mode == 'threshold_only':
            # No scaling, no compression.  Trees are scale-invariant; adding
            # StandardScaler here caused double-scaling with QRF's RobustScaler
            # (destroying its robust statistics) and was pure overhead for
            # scale-invariant CatBoost and PanelVAR (which scales per-column).
            self._scaler = None
            self._pls = None
            self._pca = None
            self._mi_mask = None
            self._n_components_fitted = X_var.shape[1]
            self._fitted = True
            return self

        # ── LINEAR TRACKS (PLS and legacy PCA) ───────────────────────────
        # Stage B: standardise before PLS/PCA
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_var)

        # ── PLS TRACK (primary, supervised) ──────────────────────────────
        if self.mode == 'pls':
            X_mi = X_scaled
            self._mi_mask = None

            # MI pre-filter (P-03): union of top-k features per output column
            if self.mi_prefilter and y is not None:
                y_2d = y if y.ndim == 2 else y[:, np.newaxis]
                n_keep = (
                    max(1, X_scaled.shape[1] // 2)
                    if self.mi_k == 'half'
                    else int(self.mi_k)
                )
                n_keep = min(n_keep, X_scaled.shape[1])
                union_mask = np.zeros(X_scaled.shape[1], dtype=bool)
                for col_idx in range(y_2d.shape[1]):
                    y_col = y_2d[:, col_idx]
                    valid = ~np.isnan(y_col)
                    if valid.sum() < 10:
                        continue
                    sel = SelectKBest(mutual_info_regression, k=n_keep)
                    sel.fit(X_scaled[valid], y_col[valid])
                    union_mask |= sel.get_support()
                if union_mask.sum() >= 2:
                    self._mi_mask = union_mask
                    X_mi = X_scaled[:, union_mask]
                # If union_mask has < 2 True entries keep all (fallback)

            # PLS: n_components = min(n_samples // 10, 20) enforces p/n ≤ 0.024
            n_pls = self.n_pls_components
            if n_pls is None:
                n_pls = min(n_samples // 10, 20, X_mi.shape[1])
            n_pls = max(n_pls, 2)

            if y is not None:
                y_fit = (y if y.ndim == 2 else y[:, np.newaxis]).copy().astype(float)
                # Replace NaN in targets with column median for PLS fitting
                for col in range(y_fit.shape[1]):
                    nan_mask = np.isnan(y_fit[:, col])
                    if nan_mask.any():
                        y_fit[nan_mask, col] = float(np.nanmedian(y_fit[:, col]))
                self._pls = PLSRegression(n_components=n_pls, scale=False)
                self._pls.fit(X_mi, y_fit)
                # Estimate proportion of X variance captured by PLS x-scores
                x_score_var = float(np.var(self._pls.x_scores_, axis=0).sum())
                x_total_var = float(np.var(X_mi, axis=0).sum()) + 1e-12
                self._pls_evr_ = min(x_score_var / x_total_var, 1.0)
            else:
                # y absent — supervised PLS not possible; fall back to PCA
                self._pls = None
                n_cap = min(n_samples // 5, self.max_components, X_mi.shape[1])
                n_cap = max(n_cap, 2)
                self._pca = PCA(
                    n_components=self.pca_variance_ratio,
                    svd_solver='full',
                    random_state=self.random_state,
                )
                self._pca.fit(X_mi)
                if self._pca.n_components_ > n_cap:
                    self._pca = PCA(
                        n_components=n_cap,
                        svd_solver='full',
                        random_state=self.random_state,
                    )
                    self._pca.fit(X_mi)
                self._n_components_fitted = self._pca.n_components_
                self._fitted = True
                return self

            self._n_components_fitted = n_pls
            self._pca = None
            self._fitted = True
            return self

        # ── LEGACY PCA TRACK ──────────────────────────────────────────────
        # mode == 'pca': retained for backward compatibility.
        self._mi_mask = None
        # The cap n_cap = min(n_samples // 5, max_components) enforces p/n ≤ 0.2
        n_cap = min(n_samples // 5, self.max_components, X_scaled.shape[1])
        n_cap = max(n_cap, 2)

        self._pca = PCA(
            n_components=self.pca_variance_ratio,
            svd_solver='full',
            random_state=self.random_state,
        )
        self._pca.fit(X_scaled)
        if self._pca.n_components_ > n_cap:
            self._pca = PCA(
                n_components=n_cap,
                svd_solver='full',
                random_state=self.random_state,
            )
            self._pca.fit(X_scaled)

        self._n_components_fitted = self._pca.n_components_
        self._pls = None
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

        if self._imputer is not None and np.isnan(X).any():
            X = self._imputer.transform(X)

        X_var = self._var_selector.transform(X)

        if self.mode == 'threshold_only':
            # Raw variance-filtered features — no scaling for tree models
            return X_var

        X_scaled = self._scaler.transform(X_var)

        if self.mode == 'pls':
            X_mi = X_scaled if self._mi_mask is None else X_scaled[:, self._mi_mask]
            if self._pls is not None:
                return self._pls.transform(X_mi)
            # PLS fallback to PCA when y was absent during fit
            return self._pca.transform(X_mi)

        # legacy pca mode
        return self._pca.transform(X_scaled)

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(X, y=y, feature_names=feature_names)
        return self.transform(X)

    def inverse_importance(self, pc_importance: np.ndarray) -> np.ndarray:
        """
        Map per-component importance back to the original feature space.

        For a vector of shape ``(n_components,)`` or a matrix of shape
        ``(n_components, n_outputs)``, returns the importance distributed
        back to original features proportionally to the loading magnitudes.

        For ``'pls'`` mode the PLS x-loadings matrix ``|x_loadings_|``
        (shape ``(n_features, n_components)``) is used; for ``'pca'`` mode
        ``|components_|`` is used.

        Parameters
        ----------
        pc_importance : ndarray
            Importance weights in compressed space.

        Returns
        -------
        original_importance : ndarray
            Importance weights mapped back to original-feature space.
        """
        if not self._fitted:
            raise ValueError("PanelFeatureReducer not fitted.")

        squeeze = False
        if pc_importance.ndim == 1:
            pc_importance = pc_importance[:, np.newaxis]
            squeeze = True

        n_original = len(self._original_feature_names)

        if self.mode == 'threshold_only':
            # Direct expansion: importance is already in kept-feature space.
            full_importance = np.zeros((n_original, pc_importance.shape[1]))
            full_importance[self._kept_feature_mask] = pc_importance
            if squeeze:
                return full_importance.ravel()
            return full_importance

        if self.mode == 'pls' and self._pls is not None:
            # x_loadings_ shape: (n_mi_features, n_components)
            # Backproject: importance in mi-space = abs(P) @ component_importance
            loadings = np.abs(self._pls.x_loadings_)  # (n_mi_feat, n_comp)
            row_sums = loadings.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            loadings_normed = loadings / row_sums
            # (n_mi_features, n_outputs)
            mi_importance = loadings_normed @ pc_importance

            # Expand from MI-filtered space → kept-features space
            n_kept = int(self._kept_feature_mask.sum())
            kept_importance = np.zeros((n_kept, pc_importance.shape[1]))
            if self._mi_mask is not None:
                kept_importance[self._mi_mask] = mi_importance
            else:
                kept_importance = mi_importance
        else:
            # PCA mode (legacy or PLS fallback when y was absent)
            _pca = self._pca
            loadings = np.abs(_pca.components_)  # (n_pc, n_kept_features)
            row_sums = loadings.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            loadings_normed = loadings / row_sums
            kept_importance = loadings_normed.T @ pc_importance

        # Expand from kept-features space → original feature space
        full_importance = np.zeros((n_original, kept_importance.shape[1]))
        full_importance[self._kept_feature_mask] = kept_importance

        if squeeze:
            return full_importance.ravel()
        return full_importance

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_components(self) -> int:
        """Number of output components after fitting."""
        return self._n_components_fitted

    @property
    def explained_variance_ratio(self) -> float:
        """
        Total explained variance ratio.

        For ``'pca'`` mode: cumulative PCA explained variance ratio.
        For ``'pls'`` mode: estimated proportion of X variance captured by
            the PLS x-scores (approximation; PLS optimises covariance with y,
            not X-variance).
        For ``'threshold_only'`` mode: 1.0 (no compression applied).
        """
        if self.mode == 'pca' and self._pca is not None:
            return float(np.sum(self._pca.explained_variance_ratio_))
        if self.mode == 'threshold_only' and self._fitted:
            return 1.0
        if self.mode == 'pls':
            if self._pls is not None:
                return self._pls_evr_
            if self._pca is not None:  # PLS fallback to PCA
                return float(np.sum(self._pca.explained_variance_ratio_))
        return 0.0

    def get_summary(self) -> str:
        """Human-readable summary of the reduction pipeline."""
        if not self._fitted:
            return "PanelFeatureReducer: not fitted"
        n_orig = len(self._original_feature_names)
        n_kept = int(self._kept_feature_mask.sum())
        if self.mode == 'threshold_only':
            return (
                f"PanelFeatureReducer (threshold_only): "
                f"{n_orig} → {n_kept} features (variance filter, no scaling)"
            )
        if self.mode == 'pls' and self._pls is not None:
            n_mi = int(self._mi_mask.sum()) if self._mi_mask is not None else n_kept
            return (
                f"PanelFeatureReducer (pls): "
                f"{n_orig} → {n_kept} (variance) "
                f"→ {n_mi} (MI filter) "
                f"→ {self._n_components_fitted} PLS components "
                f"(≈{self._pls_evr_:.1%} X-variance)"
            )
        evr = self.explained_variance_ratio
        evr_str = f"{evr:.1%}" if not (evr != evr) else "n/a"  # NaN check
        return (
            f"PanelFeatureReducer ({self.mode}): "
            f"{n_orig} → {n_kept} (variance filter) "
            f"→ {self._n_components_fitted} components "
            f"({evr_str} variance retained)"
        )
