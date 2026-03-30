"""
Feature Preprocessing for Panel Forecasting.

This module provides dimensionality reduction tracks optimized for different
model families in the `UnifiedForecaster` architecture.

Linear Track (mode='pls')
-------------------------
1. VarianceThreshold(0.005)
2. StandardScaler
3. MI pre-filter
4. PLSRegression (Supervised compression)

Finds linear combinations of features that maximize covariance with all 
targets simultaneously.

Tree Track (mode='threshold_only')
----------------------------------
1. VarianceThreshold(0.005)
2. MI pre-filter (capped at 200 features)
3. VIF filter (Collinearity removal)

Maintains raw features (no scaling/compression) to preserve tree-based 
model interpretability and performance.

Imputation (Tier 1 MICE)
------------------------
Applies `IterativeImputer` with `ExtraTreesRegressor` to handle residual 
missing data after variance filtering, ensuring a complete feature set 
for downstream models.

References
----------
- Mevik & Wehrens (2007). "The pls Package." JSS 18(2).
- Bair et al. (2006). "Prediction by Supervised Principal Components." JASA.
- Groen & Kapetanios (2016). "Revisiting Useful Approaches to Data-Rich 
  Macroeconomic Forecasting." Computational Statistics & Data Analysis.
"""

import numpy as np
from typing import List, Literal, Optional, Union
import logging
import warnings

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 — required before IterativeImputer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_regression
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

from data.imputation import ImputationConfig

logger = logging.getLogger('ml_mcdm')


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
            use for tree-based models (CatBoost, QRF).
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
        imputation_config: Optional[ImputationConfig] = None,  # PHASE A: Tier 1 MICE config
        max_vif_threshold: Optional[float] = None,  # Phase 4.3 VIF filter
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
        self.imputation_config = imputation_config or ImputationConfig()  # PHASE A
        self.max_vif_threshold = max_vif_threshold
        self.random_state = random_state
        self.mode = mode

        self._var_selector: Optional[VarianceThreshold] = None
        self._scaler: Optional[StandardScaler] = None
        self._pls: Optional[PLSRegression] = None
        self._pca: Optional[PCA] = None
        self._imputer: Optional[IterativeImputer] = None
        self._mice_imputer: Optional[IterativeImputer] = None  # PHASE A: Tier 1 MICE
        self._vif_drop_mask: Optional[np.ndarray] = None       # Phase 4.3
        self._mi_mask: Optional[np.ndarray] = None   # bool (n_kept_after_var,) — PLS track
        self._tree_mi_mask_: Optional[np.ndarray] = None  # Phase 2 §2.1 bool (n_kept_after_var,) tree track
        self._tree_vif_drop_mask_: Optional[np.ndarray] = None  # Phase 2 §2.3 tree-only VIF
        self._pls_evr_: float = 0.0                  # estimated X-variance ratio for PLS
        self._original_feature_names: Optional[List[str]] = None
        self._kept_feature_mask: Optional[np.ndarray] = None
        self._n_components_fitted: int = 0
        self._selected_feature_names_: Optional[List[str]] = None  # E-01
        self._mice_fitted: bool = False  # PHASE A: Track MICE imputer state
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

        # MICE imputation is applied AFTER variance threshold (see "Tier 1 MICE" below).
        # This allows MICE to focus on informative dimensions only.
        self._imputer = None

        # Stage A: remove near-constant features (P-01: threshold=0.005)
        self._var_selector = VarianceThreshold(threshold=self.variance_threshold)
        X_var = self._var_selector.fit_transform(X)
        self._kept_feature_mask = self._var_selector.get_support()

        # Store per-column training means of variance-filtered features.
        # Used as a fallback NaN fill in transform() when the MICE imputer
        # was not fitted (training was NaN-free) but holdout / prediction
        # features carry residual NaN values.  Column-mean imputation is the
        # correct strategy: it avoids the semantically-wrong "overall mean"
        # fill that fill_missing_features produces for 1-D row vectors, and
        # it is leakage-free because only training statistics are used.
        _col_means = np.nanmean(X_var, axis=0)
        self._var_col_means_ = np.where(np.isnan(_col_means), 0.0, _col_means)

        # ===== TIER 1: MICE Imputation (Production-Ready) =====
        # Apply unified MICE imputation to residual NaN after variance threshold.
        # This is the ONLY imputation pass in the pipeline (no redundant pre-filtering).
        #
        # Uses data.imputation.MICEImputer with ExtraTreesRegressor for:
        # - Multivariate feature correlations (learns relationships automatically)
        # - Nonlinear patterns (tree-based adaptive estimation)
        # - Panel structure (temporal/spatial correlations preserved)
        # - Uncertainty quantification (via posterior sampling in multiple imputation)
        if self.use_mice_imputation and np.isnan(X_var).any():
            from data.imputation import MICEImputer

            self._mice_imputer = MICEImputer(self.imputation_config)
            nm_before = np.isnan(X_var).sum()
            logger.info(
                f"[MICE] Fitting on variance-filtered features: "
                f"{X_var.shape[0]} samples × {X_var.shape[1]} features, "
                f"{nm_before} NaN cells ({100*nm_before/X_var.size:.2f}%)"
            )

            X_var = self._mice_imputer.fit_transform(X_var)
            nm_after = np.isnan(X_var).sum()
            logger.info(
                f"[MICE] Imputation complete: {nm_before} → {nm_after} NaN cells. "
                f"Missingness indicators appended: {self.imputation_config.mice_add_indicator}"
            )
            self._mice_fitted = True
        else:
            self._mice_imputer = None
            self._mice_fitted = False

        # ========================= End TIER 1 MICE ========================

        # ── TREE TRACK ────────────────────────────────────────────────────
        if self.mode == 'threshold_only':
            # No scaling, no compression.  Trees are scale-invariant; adding
            # StandardScaler here caused double-scaling with QRF's RobustScaler
            # (destroying its robust statistics) and was pure overhead for
            # scale-invariant CatBoost.
            self._scaler = None
            self._pls = None
            self._pca = None

            # ── Phase 2 §2.1: MI pre-filter for tree track (cap at 200) ──
            # After variance threshold + MICE imputation, apply a union-of-
            # per-output MI filter. SelectKBest(mutual_info_regression) selects
            # the top min(200, p) features for each output dimension; the union
            # across all outputs is kept. This addresses BUG-1: p/n ≥ 1 with
            # raw ~790 features causes CatBoost information-gain heuristics to
            # fail (split-finding picks noise splits in equally-informative
            # features when p/n ≥ 1). Target: p ≤ 200 → p/n ≤ 0.27.
            #
            # Union over outputs: consistent with CatBoost's MultiRMSE objective
            # (all output criteria optimised jointly). A feature with high MI to
            # any criterion is retained, preserving multi-output coverage.
            self._tree_mi_mask_ = None
            self._tree_vif_drop_mask_ = None
            X_tree = X_var.copy()

            if self.mi_prefilter and y is not None:
                y_2d = y if y.ndim == 2 else y[:, np.newaxis]
                n_keep_tree = min(200, X_tree.shape[1])  # hard cap at 200
                union_mask_tree = np.zeros(X_tree.shape[1], dtype=bool)
                valid_cols = 0
                for col_idx in range(y_2d.shape[1]):
                    y_col = y_2d[:, col_idx]
                    valid = ~np.isnan(y_col)
                    if valid.sum() < 10:
                        continue  # insufficient data for MI estimation
                    sel = SelectKBest(mutual_info_regression, k=n_keep_tree)
                    # MI regression requires non-NaN X rows too; use valid rows
                    valid_rows = valid & ~np.any(np.isnan(X_tree), axis=1)
                    if valid_rows.sum() < 10:
                        continue
                    sel.fit(X_tree[valid_rows], y_col[valid_rows])
                    union_mask_tree |= sel.get_support()
                    valid_cols += 1

                n_selected = int(union_mask_tree.sum())
                if n_selected >= 2 and valid_cols > 0:
                    self._tree_mi_mask_ = union_mask_tree
                    # Update the composite _kept_feature_mask so it accurately
                    # reflects which ORIGINAL features survive into the tree track.
                    # _kept_feature_mask is a bool array of shape (n_original_features,).
                    # Among the variance-kept features (indices = np.where(mask)[0]),
                    # keep only those also selected by MI.
                    kept_by_var_indices = np.where(self._kept_feature_mask)[0]
                    new_original_mask = np.zeros(len(self._kept_feature_mask), dtype=bool)
                    kept_by_var_and_mi = kept_by_var_indices[union_mask_tree]
                    new_original_mask[kept_by_var_and_mi] = True
                    self._kept_feature_mask = new_original_mask
                    X_tree = X_tree[:, union_mask_tree]
                    logger.info(
                        f"[Tree MI §2.1] MI pre-filter: {X_var.shape[1]} → {n_selected} "
                        f"features (cap={n_keep_tree}, n_outputs_used={valid_cols})"
                    )
                else:
                    # Degenerate: keep all post-variance features (safe fallback)
                    logger.info(
                        f"[Tree MI §2.1] MI pre-filter skipped: union selected "
                        f"{n_selected} features (< 2) or no valid output columns. "
                        f"Retaining all {X_tree.shape[1]} post-variance features."
                    )
            else:
                if self.mi_prefilter and y is None:
                    logger.debug(
                        "[Tree MI §2.1] MI pre-filter skipped: y=None "
                        "(tree MI requires targets; call fit(X, y=...) to enable)."
                    )

            # ── Phase 2 §2.3: VIF filter — TREE TRACK ONLY, post MI ──────
            # VIF is moved from the shared code path (which incorrectly applied
            # it to PLS/PCA modes) to inside the tree branch, sequenced AFTER
            # the MI filter so VIF acts on the already-reduced p ≤ 200 space.
            #
            # VIF_j = 1 / (1 − R²_j) where R²_j is the R² from regressing
            # feature j on all other features. VIF > 10 ⟹ R²_j > 0.90,
            # i.e., 90% of feature j's variance is explained by other features.
            # For tree information-gain splits, two features with VIF > 10
            # compete for the same split, inflating one's importance and
            # deflating the other's arbitrarily. Removing them stabilises
            # CatBoost's feature-importance estimates.
            self._tree_vif_drop_mask_ = np.zeros(X_tree.shape[1], dtype=bool)
            if self.max_vif_threshold is not None and X_tree.shape[1] > 2:
                logger.info(
                    f"[Tree VIF §2.3] Applying VIF filter (threshold={self.max_vif_threshold}). "
                    f"Input features={X_tree.shape[1]}"
                )
                # Step 1: Drop perfectly collinear (|r| > 0.98) to ensure
                # matrix invertibility. np.corrcoef is O(p²n) but p ≤ 200 here.
                corr = np.corrcoef(X_tree, rowvar=False)
                upper_tri = np.triu(np.abs(corr), k=1)
                drop_corr: List[int] = [
                    i for i in range(upper_tri.shape[1])
                    if np.any(upper_tri[:, i] > 0.98)
                ]
                kept: List[int] = [
                    i for i in range(X_tree.shape[1]) if i not in drop_corr
                ]

                # Step 2: Iterative VIF via inverse correlation matrix.
                # Uses StandardScaler for correlation computation only
                # (VIF is scale-invariant so scaling is just for numerical
                # stability in the inversion).
                X_vif_s = StandardScaler().fit_transform(X_tree)
                drop_vif: List[int] = []

                while len(kept) > 2:
                    c_mat = np.corrcoef(X_vif_s[:, kept], rowvar=False)
                    c_mat += np.eye(c_mat.shape[0]) * 1e-4  # ridge for stability
                    try:
                        vif = np.diag(np.linalg.inv(c_mat))
                    except np.linalg.LinAlgError:
                        logger.warning(
                            "[Tree VIF §2.3] Correlation matrix singular — "
                            "VIF iteration halted; retaining current feature set."
                        )
                        break

                    exceed_indices = np.where(vif > self.max_vif_threshold)[0]
                    if len(exceed_indices) == 0:
                        break  # All remaining features below threshold

                    # Drop top-5 highest-VIF features in batch (speeds convergence)
                    exceed_sorted = sorted(
                        [(vif[i], i) for i in exceed_indices], reverse=True
                    )
                    batch_drop = min(5, len(exceed_sorted))
                    indices_to_drop_local = sorted(
                        [exceed_sorted[k][1] for k in range(batch_drop)], reverse=True
                    )
                    for idx in indices_to_drop_local:
                        drop_vif.append(kept.pop(idx))

                all_dropped_vif = drop_corr + drop_vif
                if all_dropped_vif:
                    self._tree_vif_drop_mask_[all_dropped_vif] = True
                    # Back-propagate VIF drops into _kept_feature_mask
                    # (original-feature space). _kept_feature_mask currently
                    # reflects variance + MI filters. We now mark VIF-dropped
                    # features (in X_tree / post-MI space) as False.
                    kept_indices_in_original = np.where(self._kept_feature_mask)[0]
                    original_indices_to_drop = kept_indices_in_original[all_dropped_vif]
                    self._kept_feature_mask[original_indices_to_drop] = False
                    X_tree = X_tree[:, ~self._tree_vif_drop_mask_]
                    logger.info(
                        f"[Tree VIF §2.3] Removed {len(drop_corr)} near-collinear + "
                        f"{len(drop_vif)} high-VIF. Remaining features={X_tree.shape[1]}"
                    )
                else:
                    logger.info("[Tree VIF §2.3] VIF filter: no features exceeded threshold.")

            # ── Finalise tree track ───────────────────────────────────────
            self._mi_mask = None  # PLS-track MI mask; unused in tree mode
            self._vif_drop_mask = np.zeros(1, dtype=bool)  # unused in tree mode
            self._n_components_fitted = X_tree.shape[1]

            # E-01: store final post-MI+VIF feature names so that
            # ``get_demeaned_column_indices`` can identify which columns
            # in the reduced matrix correspond to ``_demeaned`` and
            # ``_demeaned_momentum`` features without a full feature-name scan.
            _selected_idxs = np.where(self._kept_feature_mask)[0]
            if self._original_feature_names is not None:
                self._selected_feature_names_ = [
                    self._original_feature_names[i] for i in _selected_idxs
                ]
            else:
                self._selected_feature_names_ = [
                    f"f{i}" for i in range(X_tree.shape[1])
                ]

            # Sanity check: selected names count must equal X_tree columns
            if len(self._selected_feature_names_) != X_tree.shape[1]:
                logger.warning(
                    f"[Tree track] Feature name count mismatch after MI+VIF: "
                    f"names={len(self._selected_feature_names_)}, "
                    f"X_tree.shape[1]={X_tree.shape[1]}. Using generic names."
                )
                self._selected_feature_names_ = [
                    f"f{i}" for i in range(X_tree.shape[1])
                ]

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

        # No pre-variance MICE (removed redundant Tier 0)
        X_var = self._var_selector.transform(X)

        # ===== TIER 1: MICE Transform (production-ready) =====
        if self._mice_fitted and self._mice_imputer is not None and np.isnan(X_var).any():
            X_var = self._mice_imputer.transform(X_var)

        # Fallback NaN fill with training column means when no MICE imputer
        # was fitted (training data was NaN-free) but holdout / prediction
        # features carry residual NaN. Without this guard, StandardScaler
        # raises "Input contains NaN" crashing downstream transforms.
        if np.isnan(X_var).any():
            _means = getattr(self, '_var_col_means_', np.zeros(X_var.shape[1]))
            X_var = np.where(np.isnan(X_var), _means[np.newaxis, :], X_var)

        if self.mode == 'threshold_only':
            # Apply tree-track MI mask (Phase 2 §2.1)
            if self._tree_mi_mask_ is not None:
                X_var = X_var[:, self._tree_mi_mask_]
            # Apply tree-track VIF mask (Phase 2 §2.3)
            if self._tree_vif_drop_mask_ is not None and self._tree_vif_drop_mask_.any():
                # _tree_vif_drop_mask_ is in post-MI space
                X_var = X_var[:, ~self._tree_vif_drop_mask_]
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
        """
        Fit and transform in one call.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray, shape (n_samples, n_outputs), optional
            Training target matrix.
        feature_names : List[str], optional
            Original feature names.

        Returns
        -------
        np.ndarray
            Reduced feature matrix.
        """
        return self.fit(X, y, feature_names).transform(X)

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
        """
        Get a human-readable summary of the reduction pipeline.

        Returns
        -------
        str
            Summary of feature counts and transformation steps.
        """
        if not self._fitted:
            return "PanelFeatureReducer: not fitted"
        n_orig = len(self._original_feature_names)
        _var_support = (
            self._var_selector.get_support()
            if self._var_selector is not None
            else np.ones(n_orig, dtype=bool)
        )
        n_kept_var = int(_var_support.sum())
        n_kept_final = int(self._kept_feature_mask.sum()) if self._kept_feature_mask is not None else 0
        if self.mode == 'threshold_only':
            n_mi = (
                int(self._tree_mi_mask_.sum())
                if self._tree_mi_mask_ is not None
                else n_kept_var
            )
            n_vif_dropped = (
                int(self._tree_vif_drop_mask_.sum())
                if self._tree_vif_drop_mask_ is not None
                else 0
            )
            return (
                f"PanelFeatureReducer (threshold_only): "
                f"{n_orig} → {n_kept_var} (variance) "
                f"→ {n_mi} (MI §2.1) "
                f"→ {n_kept_final} features (VIF §2.3 dropped {n_vif_dropped})"
            )
        if self.mode == 'pls' and self._pls is not None:
            n_mi = int(self._mi_mask.sum()) if self._mi_mask is not None else n_kept_final
            return (
                f"PanelFeatureReducer (pls): "
                f"{n_orig} → {n_kept_final} (variance) "
                f"→ {n_mi} (MI filter) "
                f"→ {self._n_components_fitted} PLS components "
                f"(≈{self._pls_evr_:.1%} X-variance)"
            )
        evr = self.explained_variance_ratio
        evr_str = f"{evr:.1%}" if not (evr != evr) else "n/a"  # NaN check
        return (
            f"PanelFeatureReducer ({self.mode}): "
            f"{n_orig} → {n_kept_final} (variance filter) "
            f"→ {self._n_components_fitted} components "
            f"({evr_str} variance retained)"
        )

    def get_demeaned_column_indices(self) -> tuple:
        """
        Return column indices of entity-demeaned features in the reduced
        (post-variance-threshold) feature matrix.

        Used by the E-01 fold-correction logic in ``SuperLearner.fit()``
        to identify which columns in ``X_train_tree_`` correspond to
        ``{comp}_demeaned`` and ``{comp}_demeaned_momentum`` features.
        These are the only columns whose values depend on entity-level
        means computed from ALL training years, and therefore the only
        columns that require fold-specific correction inside the CV loop.

        Only meaningful for ``mode='threshold_only'`` (tree track).

        Returns
        -------
        demeaned_indices : list[int]
            Column indices where feature name ends with ``'_demeaned'``
            (but NOT ``'_demeaned_momentum'``).
        demeaned_momentum_indices : list[int]
            Column indices where feature name ends with
            ``'_demeaned_momentum'``.

        Both lists are empty when ``_selected_feature_names_`` has not
        been populated (i.e. the reducer is not in ``threshold_only``
        mode, or ``fit()`` was called without feature names).
        """
        if (
            self._selected_feature_names_ is None
            or len(self._selected_feature_names_) == 0
        ):
            return [], []

        demeaned_idx: List[int] = []
        demeaned_mom_idx: List[int] = []
        for i, name in enumerate(self._selected_feature_names_):
            if name.endswith('_demeaned_momentum'):
                demeaned_mom_idx.append(i)
            elif name.endswith('_demeaned'):
                demeaned_idx.append(i)

        return demeaned_idx, demeaned_mom_idx
