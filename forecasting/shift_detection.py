# -*- coding: utf-8 -*-
"""
Panel Covariate Shift Detection (E-08)
=======================================

Detects and corrects for covariate distribution shifts between CV training
and validation folds in the panel walk-forward setting.

Algorithm
---------
1. Compute MMD² (Maximum Mean Discrepancy) with Gaussian RBF kernel:
       MMD²(P, Q) = E[k(x,x')] + E[k(z,z')] − 2·E[k(x,z)]
   Unbiased estimator with diagonal zeroing; median-heuristic bandwidth σ².

2. Bootstrap permutation null distribution (n_boot=200 resamples of pooled
   X_train ∪ X_val) → (1−α)-th quantile threshold δ.
   Shift is flagged when MMD² > δ.

3. Logistic-regression density ratio estimator (classifier approach):
       w(x) ≈ P̂(val|x) / P̂(train|x) × (n_train / n_val)
   Clipped to [1/max_weight_ratio, max_weight_ratio] and normalised so
   mean(w) = 1 (unbiased estimator for the training distribution).

References
----------
Gretton, Borgwardt, Rasch, Schölkopf & Smola (2012).
    "A Kernel Two-Sample Test." JMLR 13, 723–773.
Sugiyama, Suzuki & Kanamori (2008).
    "Direct importance estimation with model selection." Neural Computation 20(10).
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler


class PanelCovariateShiftDetector:
    """Detect and correct covariate shift between CV training/validation folds.

    For each fold in a panel walk-forward CV:

    1. Compute MMD² between ``X_train_fold`` and ``X_val_fold``.
    2. Bootstrap permutation test → threshold δ at significance ``alpha``.
    3. When MMD² > δ, compute per-sample importance weights via logistic
       regression density ratio estimation.  Return uniform weights otherwise.

    Parameters
    ----------
    alpha : float
        Significance level for the permutation test (default 0.05 → 95-th
        percentile null).  Smaller α is more conservative (fewer folds
        re-weighted).
    n_bootstrap : int
        Bootstrap resamples for the null distribution (default 200).
    max_weight_ratio : float
        Maximum clipping of importance weights (default 10.0).  Prevents
        extreme weights from inflating variance.
    min_train_for_shift : int
        Minimum training rows to enable detection; returns uniform weights
        below this threshold (default 30).
    reduce_dim : int or None
        Random-projection dimensionality before MMD² (for high-d X).
        ``None`` disables reduction.
    random_state : int
    verbose : bool
        Print per-fold diagnostic info.

    Attributes
    ----------
    shift_detected_history_ : list[bool]
    mmd2_history_ : list[float]
    threshold_history_ : list[float]
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 200,
        max_weight_ratio: float = 10.0,
        min_train_for_shift: int = 30,
        reduce_dim: Optional[int] = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.alpha                = alpha
        self.n_bootstrap          = n_bootstrap
        self.max_weight_ratio     = max_weight_ratio
        self.min_train_for_shift  = min_train_for_shift
        self.reduce_dim           = reduce_dim
        self.random_state         = random_state
        self.verbose              = verbose

        self.shift_detected_history_: list = []
        self.mmd2_history_:            list = []
        self.threshold_history_:        list = []
        self._rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def compute_fold_weights(
        self,
        X_train: np.ndarray,
        X_val:   np.ndarray,
    ) -> np.ndarray:
        """Return per-row importance weights for one CV fold's training set.

        Parameters
        ----------
        X_train : ndarray, shape (n_train, n_features)
            Training fold features.  May contain NaN — NaN rows are
            assigned weight 1.0.
        X_val : ndarray, shape (n_val, n_features)
            Validation fold features.

        Returns
        -------
        weights : ndarray, shape (n_train,)
            Non-negative importance weights.  Uniform (1.0) when no
            significant shift is detected.
        """
        n_train = X_train.shape[0]
        uniform_w = np.ones(n_train, dtype=np.float64)

        if n_train < self.min_train_for_shift or len(X_val) < 2:
            return uniform_w

        # NaN → column mean for kernel computation only
        X_tr_full = self._fill_nan(X_train)
        X_va_full = self._fill_nan(X_val)

        # Optional random projection to reduce kernel cost
        if (self.reduce_dim is not None
                and X_tr_full.shape[1] > self.reduce_dim):
            X_tr_k, X_va_k = self._random_project(X_tr_full, X_va_full)
        else:
            X_tr_k, X_va_k = X_tr_full, X_va_full

        mmd2  = self.detect_shift(X_tr_k, X_va_k)
        delta = self.bootstrap_threshold(X_tr_k, X_va_k)
        shift = bool(mmd2 > delta)

        self.mmd2_history_.append(mmd2)
        self.threshold_history_.append(delta)
        self.shift_detected_history_.append(shift)

        if self.verbose:
            marker = "SHIFT" if shift else "ok   "
            print(f"    [ShiftDetect] {marker} "
                  f"MMD²={mmd2:.4f} δ={delta:.4f}")

        if not shift:
            return uniform_w

        # Shift detected: logistic density-ratio weights
        weights = self.compute_importance_weights(X_tr_full, X_va_full)
        # Restore NaN-original rows to weight 1.0
        nan_rows = np.isnan(X_train).any(axis=1)
        weights[nan_rows] = 1.0
        return weights

    # ------------------------------------------------------------------
    # Core statistical methods
    # ------------------------------------------------------------------

    def detect_shift(
        self,
        X_train: np.ndarray,
        X_val:   np.ndarray,
    ) -> float:
        """Unbiased MMD² with Gaussian RBF kernel (median bandwidth).

        MMD²(P,Q) = E_{x,x'~P}[k(x,x')] + E_{z,z'~Q}[k(z,z')]
                    − 2·E_{x~P,z~Q}[k(x,z)]

        Unbiased: diagonal of K_tt and K_vv zeroed out.

        Parameters
        ----------
        X_train, X_val : ndarray (NaN-free)

        Returns
        -------
        mmd2 : float
        """
        sigma2 = self._median_bandwidth(X_train)
        gamma  = 1.0 / (2.0 * sigma2 + 1e-15)

        n_t = len(X_train)
        n_v = len(X_val)

        K_tt = rbf_kernel(X_train, X_train, gamma=gamma)
        K_vv = rbf_kernel(X_val,   X_val,   gamma=gamma)
        K_tv = rbf_kernel(X_train, X_val,   gamma=gamma)

        np.fill_diagonal(K_tt, 0.0)
        np.fill_diagonal(K_vv, 0.0)

        mmd2 = (
            K_tt.sum() / max(n_t * (n_t - 1), 1)
            + K_vv.sum() / max(n_v * (n_v - 1), 1)
            - 2.0 * K_tv.mean()
        )
        return float(mmd2)

    def bootstrap_threshold(
        self,
        X_train: np.ndarray,
        X_val:   np.ndarray,
    ) -> float:
        """Bootstrap permutation null (1−α)-th quantile for MMD².

        Pools X_train ∪ X_val, permutes ``n_bootstrap`` times, splits at
        n_train, and computes MMD² on each permuted split.

        Returns
        -------
        delta : float
            (1−α)-th quantile of the null MMD² distribution.
        """
        X_pool = np.vstack([X_train, X_val])
        n_t    = len(X_train)
        n_pool = len(X_pool)

        null_mmd2 = np.empty(self.n_bootstrap)
        for b in range(self.n_bootstrap):
            perm = self._rng.permutation(n_pool)
            X_p  = X_pool[perm]
            null_mmd2[b] = self.detect_shift(X_p[:n_t], X_p[n_t:])

        return float(np.quantile(null_mmd2, 1.0 - self.alpha))

    def compute_importance_weights(
        self,
        X_train: np.ndarray,
        X_val:   np.ndarray,
    ) -> np.ndarray:
        """Density ratio w(x) = P_val(x)/P_train(x) via logistic regression.

        Labels: X_train → 0 (source distribution), X_val → 1 (target).
        Logistic classifier estimates P̂(label=1|x) = P̂(val|x).

        Density ratio approximation (class-prior corrected):
            w(x) = P̂(val|x) / (1 − P̂(val|x)) × (n_train / n_val)

        Weights clipped to [1/max_weight_ratio, max_weight_ratio] and
        normalised to mean 1 (importance sampling identity).

        Parameters
        ----------
        X_train, X_val : ndarray (NaN-free)

        Returns
        -------
        weights : ndarray, shape (n_train,)
        """
        n_t = len(X_train)
        n_v = len(X_val)

        X_joint = np.vstack([X_train, X_val])
        y_joint = np.concatenate([
            np.zeros(n_t, dtype=int),
            np.ones(n_v,  dtype=int),
        ])

        scaler    = StandardScaler()
        X_joint_s = scaler.fit_transform(X_joint)

        try:
            clf = LogisticRegression(
                C=1.0, solver='lbfgs', max_iter=400,
                random_state=self.random_state,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_joint_s, y_joint)
            p_val = clf.predict_proba(X_joint_s[:n_t])[:, 1]
        except Exception:
            return np.ones(n_t, dtype=np.float64)

        eps   = 1e-8
        # Importance weight = density ratio × class-prior correction
        ratio = (p_val + eps) / (1.0 - p_val + eps) * (n_t / max(n_v, 1))
        ratio = np.clip(ratio, 1.0 / self.max_weight_ratio, self.max_weight_ratio)
        # Normalise: E_train[w(x)] = 1
        ratio = ratio / ratio.mean()
        return ratio.astype(np.float64)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def fold_summary(self) -> dict:
        """Return a summary dict of shift history across all processed folds."""
        if not self.mmd2_history_:
            return {}
        detected = np.array(self.shift_detected_history_, dtype=bool)
        return {
            'n_folds':            len(self.mmd2_history_),
            'n_shifted':          int(detected.sum()),
            'frac_shifted':       float(detected.mean()),
            'mean_mmd2':          float(np.mean(self.mmd2_history_)),
            'mean_threshold':     float(np.mean(self.threshold_history_)),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _median_bandwidth(X: np.ndarray) -> float:
        """Median pairwise squared Euclidean distance (bandwidth σ²)."""
        n = len(X)
        if n > 300:
            # Subsample for efficiency
            idx = np.random.RandomState(0).choice(n, 300, replace=False)
            X = X[idx]
        from sklearn.metrics import pairwise_distances
        dists  = pairwise_distances(X, metric='sqeuclidean')
        triu   = dists[np.triu_indices_from(dists, k=1)]
        sigma2 = float(np.median(triu)) if len(triu) > 0 else 1.0
        return max(sigma2, 1e-8)

    @staticmethod
    def _fill_nan(X: np.ndarray) -> np.ndarray:
        """Replace NaN with column means (for kernel evaluation only)."""
        if not np.isnan(X).any():
            return X
        X_out = X.copy()
        col_means = np.nanmean(X_out, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        row_idx, col_idx = np.where(np.isnan(X_out))
        X_out[row_idx, col_idx] = col_means[col_idx]
        return X_out

    def _random_project(
        self,
        X_tr: np.ndarray,
        X_va: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random Gaussian projection to ``reduce_dim`` dimensions."""
        d = X_tr.shape[1]
        k = min(self.reduce_dim, d)     # type: ignore[arg-type]
        P = self._rng.randn(d, k) / np.sqrt(k)
        return X_tr @ P, X_va @ P
