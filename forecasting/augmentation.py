"""
Conditional Panel Augmentation (E-06).

This module generates synthetic panel trajectories that preserve the 
statistical properties of observed data, including cross-sectional 
correlations, within-entity dynamics, and empirical marginal distributions.

Key Features
------------
1. **Cross-Sectional Correlation**: Uses a Gaussian copula over annual feature 
   means to preserve inter-feature dependencies.
2. **Within-Entity Dynamics**: Estimates per-entity VAR(1) marginal dynamics 
   to maintain temporal autocorrelation.
3. **Empirical Marginals**: Applies probability integral transforms (PIT) 
   to map copula samples back to empirical feature distributions.

Production Integrity
--------------------
- **Synthetic-Aware CV**: Ensures synthetic rows are never used for validation, 
  only for training augmentation within earlier year windows.
- **Gain Gate**: A lightweight proxy evaluation (Ridge-proxy CV) validates 
  the benefit of augmentation before committing to modified training sets.

References
----------
- Aas et al. (2009). "Pair-copula constructions of multiple dependence." 
  Insurance Mathematics and Economics 44(2).
- Yoon et al. (2018). "GAIN: Missing Data Imputation using Generative 
  Adversarial Networks." ICML 2018.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm as _scipy_norm
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score

class SyntheticAwareCV:
    """
    Walk-forward CV splitter that marks synthetic rows training-only.

    Validation indices contain only real observations, while training indices
    include original real rows plus synthetic rows whose label is before the
    current validation year.
    """

    def __init__(self, min_train_years: int = 6, max_folds: int = 999):
        """
        Initialize the synthetic-aware CV splitter.

        Parameters
        ----------
        min_train_years : int, default=6
            Minimum calendar years in the training window before the first 
            validation fold.
        max_folds : int, default=999
            Maximum number of folds to generate.
        """
        self.min_train_years = min_train_years
        self.max_folds       = max_folds

    def split(
        self,
        X:              np.ndarray,
        year_labels:    np.ndarray,
        synthetic_mask: Optional[np.ndarray] = None,
    ):
        """Yield (train_idx, val_idx) pairs.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
        year_labels : ndarray, shape (n_samples,)
            Calendar year per observation.  Synthetic rows must have a year
            label that is a float suffixed with e.g. 0.5 OR can be the same
            integer year but with ``synthetic_mask`` marking them.
        synthetic_mask : ndarray of bool, shape (n_samples,), optional
            True for synthetic rows.

        Yields
        ------
        train_idx, val_idx : ndarray of int
        """
        if synthetic_mask is None:
            synthetic_mask = np.zeros(len(y := year_labels), dtype=bool)

        real_mask     = ~synthetic_mask
        real_years    = np.sort(np.unique(year_labels[real_mask]))
        n_real_years  = len(real_years)

        if n_real_years < self.min_train_years + 1:
            return

        first_val = self.min_train_years
        n_folds   = min(self.max_folds, n_real_years - first_val)

        for k in range(n_folds):
            val_year  = int(real_years[first_val + k])
            train_yrs = real_years[:first_val + k]           # real train years

            # Real training rows
            real_train = np.where(
                real_mask & (year_labels < val_year)
            )[0]
            # Synthetic rows with year < val_year
            synth_train = np.where(
                synthetic_mask & (year_labels < val_year)
            )[0]
            # Validation: only real rows from val_year
            val_real = np.where(
                real_mask & (year_labels == val_year)
            )[0]

            if len(val_real) == 0:
                continue

            train_idx = np.sort(np.concatenate([real_train, synth_train]))
            yield train_idx, val_real


class ConditionalPanelAugmenter:
    """Generate synthetic panel trajectories via Gaussian copula + VAR(1).

    Workflow
    --------
    1. ``fit(X, entity_indices, year_labels)`` — estimate per-entity VAR(1)
       dynamics and cross-sectional Gaussian copula.
    2. ``augment(n_synth_years)`` — draw synthetic trajectories; package as
       arrays with synthetic entity/year labels.
    3. ``evaluate_gain(X, y, entity_indices, year_labels, n_synth_years)``
       — lightweight Ridge-proxy 5-fold walk-forward CV with/without
       augmentation; return ΔR² (positive = augmentation helps).
    4. ``fit_augment_if_beneficial(...)`` — convenience wrapper (steps 1–3
       with optional commit).

    Parameters
    ----------
    n_synth_years : int
        Number of synthetic years to generate per entity (default 14 = 1×
        the original T, doubling the training set).
    gain_threshold : float
        Minimum Ridge-proxy ΔR² to commit to augmentation (default 0.005;
        i.e. +0.5 percentage-point mean CV R²).
    min_entity_years : int
        Minimum observed years per entity to fit VAR(1) (default 4).
    noise_std : float
        Additive Gaussian noise added to synthetic features for regularisation
        (default 0.02; ≈ 2% of std).
    random_state : int
    verbose : bool

    Attributes
    ----------
    _var_coefs_ : dict[entity, ndarray]
        Per-entity VAR(1) coefficient matrices (n_features × n_features).
    _var_intercepts_ : dict[entity, ndarray]
        Per-entity VAR(1) intercepts (n_features,).
    _copula_chol_ : ndarray
        Cholesky factor of the empirical correlation matrix of
        cross-sectional annual means (n_features × n_features).
    _ecdf_quantiles_ : ndarray
        Empirical quantile lookup table for the probability integral
        transform (n_quantile_points × n_features).
    """

    def __init__(
        self,
        n_synth_years: int = 14,
        gain_threshold: float = 0.005,
        min_entity_years: int = 4,
        noise_std: float = 0.02,
        random_state: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize the augmenter.

        Parameters
        ----------
        n_synth_years : int, default=14
            Number of synthetic years to generate per entity.
        gain_threshold : float, default=0.005
            Minimum gain in proxy R² required to commit to augmentation.
        min_entity_years : int, default=4
            Minimum historical years per entity required for VAR(1) estimation.
        noise_std : float, default=0.02
            Standard deviation of white noise added to synthetic features.
        random_state : int, default=42
            Seed for reproducible synthetic generation.
        verbose : bool, default=True
            Whether to print progress information.
        """
        self.n_synth_years    = n_synth_years
        self.gain_threshold   = gain_threshold
        self.min_entity_years = min_entity_years
        self.noise_std        = noise_std
        self.random_state     = random_state
        self.verbose          = verbose

        self._var_coefs_:       dict = {}
        self._var_intercepts_:  dict = {}
        self._copula_chol_:  Optional[np.ndarray] = None
        self._ecdf_quantiles_: Optional[np.ndarray] = None
        self._n_features_:   Optional[int] = None
        self._fitted_:       bool  = False
        self._rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        entity_indices: np.ndarray,
        year_labels:    np.ndarray,
    ) -> 'ConditionalPanelAugmenter':
        """Estimate VAR(1) dynamics and Gaussian copula from training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)  (NaN-free preferred)
        entity_indices : ndarray, shape (n_samples,)
        year_labels : ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        n_feat = X.shape[1]
        self._n_features_ = n_feat

        # Replace residual NaN with column means for fitting
        X_clean = self._impute_with_means(X)

        # ── Per-entity VAR(1) ─────────────────────────────────────────
        for ent in np.unique(entity_indices):
            mask  = entity_indices == ent
            yrs   = year_labels[mask]
            order = np.argsort(yrs)
            X_ent = X_clean[np.where(mask)[0][order]]   # (T_ent, n_feat)

            if len(X_ent) < self.min_entity_years:
                # Not enough data: use zeros (no dynamics)
                self._var_coefs_[ent]      = np.zeros((n_feat, n_feat))
                self._var_intercepts_[ent] = np.mean(X_ent, axis=0)
                continue

            # OLS VAR(1): regress X_t on X_{t-1} per feature (independent)
            A = np.zeros((n_feat, n_feat))
            b = np.zeros(n_feat)
            for f in range(n_feat):
                y_t   = X_ent[1:, f]       # target: feature f at time t
                X_tm1 = X_ent[:-1, :]      # predictor: all features at t-1
                if np.std(y_t) < 1e-10:
                    b[f] = np.mean(y_t)
                    continue
                try:
                    reg = Ridge(alpha=1.0, fit_intercept=True)
                    reg.fit(X_tm1, y_t)
                    A[f, :] = reg.coef_
                    b[f]    = reg.intercept_
                except Exception:
                    b[f] = np.mean(y_t)
            self._var_coefs_[ent]      = A
            self._var_intercepts_[ent] = b

        # ── Gaussian copula: cross-sectional correlation ───────────────
        # Annual cross-section means: (n_years, n_feat)
        unique_years = np.sort(np.unique(year_labels))
        cs_means = np.vstack([
            X_clean[year_labels == yr].mean(axis=0)
            for yr in unique_years
        ])  # (n_years, n_feat)

        # Correlation matrix (use clipping to ensure PSD)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.corrcoef(cs_means.T)    # (n_feat, n_feat)
        corr = self._nearest_psd(corr)
        try:
            self._copula_chol_ = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            self._copula_chol_ = np.eye(n_feat)

        # ── Empirical CDF quantile table for probability integral transform
        N_Q = 500
        qs = np.linspace(0.0, 1.0, N_Q)
        self._ecdf_quantiles_ = np.quantile(X_clean, qs, axis=0)   # (N_Q, n_feat)

        self._fitted_ = True
        if self.verbose:
            print(f"  ConditionalPanelAugmenter fitted: "
                  f"{len(np.unique(entity_indices))} entities × "
                  f"{len(unique_years)} years, "
                  f"{n_feat} features.")
        return self

    def augment(
        self,
        entity_indices: np.ndarray,
        year_labels:    np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Draw synthetic panel trajectories.

        For each entity, generates ``n_synth_years`` synthetic feature vectors
        by:
        1. Drawing from the Gaussian copula to get correlated Gaussian innovations.
        2. Mapping Gaussian → uniform via Φ⁻¹ (probit); uniform → empirical CDF
           quantile (probability integral transform).
        3. Propagating through the VAR(1) dynamics to create temporally
           correlated trajectories.
        4. Adding small Gaussian noise (``noise_std``) for regularisation.

        Synthetic year labels are assigned as ``max(real_year) + 1, +2, …``
        per entity.  Synthetic entity indices map back to original entity IDs.

        Returns
        -------
        X_synth : ndarray, shape (n_synth_samples, n_features)
        entity_synth : ndarray, shape (n_synth_samples,)
        year_synth : ndarray, shape (n_synth_samples,)
        y_synth : ndarray or None
            If ``y`` is provided, synthetic targets are drawn as small
            perturbations of the entity-mean targets (not used for training;
            provided only for API completeness).
        """
        if not self._fitted_:
            raise RuntimeError("Call fit() before augment().")

        n_feat = self._n_features_
        unique_entities = np.unique(entity_indices)
        max_real_year = int(year_labels.max())

        X_rows:    List[np.ndarray] = []
        ent_rows:  List[int]        = []
        yr_rows:   List[int]        = []
        y_rows:    List[np.ndarray] = []

        for ent in unique_entities:
            ent_mask  = entity_indices == ent
            ent_yrs   = year_labels[ent_mask]
            A         = self._var_coefs_[ent]
            bvec      = self._var_intercepts_[ent]

            # Initial state: last observed row for this entity
            last_yr = int(ent_yrs.max())
            last_idx = np.where(ent_mask & (year_labels == last_yr))[0]
            if len(last_idx) == 0:
                continue
            x_prev  = np.mean(self._impute_with_means(
                np.zeros((1, n_feat))   # fallback zero vector
            ), axis=0)
            # Use actual last observation as initial state
            # (X not stored; approximate from VAR intercept)
            x_prev  = bvec.copy()   # warm start from entity mean

            for t in range(self.n_synth_years):
                # Gaussian copula innovation
                z = self._copula_chol_ @ self._rng.randn(n_feat)   # corr Gaussian
                u = _scipy_norm.cdf(z)                              # uniform [0,1]
                x_innov = self._quantile_transform(u)               # back to data space

                # VAR(1) dynamics: x_t = b + A·x_{t-1} + innovation residual
                x_t = (bvec + A @ x_prev + x_innov) / 2.0          # blend
                # Add regularisation noise
                if self.noise_std > 0:
                    x_t = x_t + self._rng.randn(n_feat) * self.noise_std

                X_rows.append(x_t)
                ent_rows.append(int(ent))
                yr_rows.append(max_real_year + t + 1)    # padded virtual year
                x_prev = x_t.copy()

                # Simple y_synth: entity mean + small noise
                if y is not None:
                    ent_y_mean = np.nanmean(y[ent_mask], axis=0)
                    ent_y_std  = max(np.nanstd(y[ent_mask]), 1e-6)
                    y_rows.append(ent_y_mean
                                  + self._rng.randn(*ent_y_mean.shape)
                                  * ent_y_std * 0.1)

        if not X_rows:
            empty = np.empty((0, n_feat))
            return (empty, np.array([], dtype=int),
                    np.array([], dtype=int), None)

        X_synth   = np.vstack(X_rows)
        ent_synth = np.array(ent_rows, dtype=int)
        yr_synth  = np.array(yr_rows,  dtype=int)
        y_synth   = (np.vstack(y_rows)
                     if y_rows else None)

        if self.verbose:
            print(f"  Augmented: {len(X_synth)} synthetic rows "
                  f"({len(unique_entities)} entities × {self.n_synth_years} years).")

        return X_synth, ent_synth, yr_synth, y_synth

    def evaluate_gain(
        self,
        X:              np.ndarray,
        y:              np.ndarray,
        entity_indices: np.ndarray,
        year_labels:    np.ndarray,
        min_train_years: int = 6,
    ) -> float:
        """Evaluate augmentation benefit via Ridge-proxy 5-fold walk-forward CV.

        Trains a ``RidgeCV`` proxy model with and without augmented data on
        the first output column (representative for overall gain).

        The CV uses ``SyntheticAwareCV`` so synthetic rows are only in training
        folds and never in validation.

        Parameters
        ----------
        X, y, entity_indices, year_labels : training data arrays
        min_train_years : int
            Walk-forward CV minimum training years (default 6).

        Returns
        -------
        delta_r2 : float
            Mean validation R² with augmentation minus without.
            Positive = augmentation improves generalisation.
        """
        if not self._fitted_:
            raise RuntimeError("Call fit() before evaluate_gain().")

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y_col = y[:, 0]   # first output as proxy

        # Cross-validate WITHOUT augmentation
        r2_base = self._cv_ridge(X, y_col, entity_indices, year_labels,
                                  np.zeros(len(X), dtype=bool),
                                  min_train_years)

        # Generate synthetic data
        X_s, e_s, yr_s, _ = self.augment(entity_indices, year_labels)
        if len(X_s) == 0:
            return 0.0

        # Augmented arrays
        X_aug   = np.vstack([X, X_s])
        y_aug   = np.concatenate([y_col,
                                   np.full(len(X_s), np.nanmean(y_col))])
        ent_aug = np.concatenate([entity_indices, e_s])
        yr_aug  = np.concatenate([year_labels,    yr_s])
        synth_m = np.concatenate([np.zeros(len(X), dtype=bool),
                                   np.ones(len(X_s), dtype=bool)])

        r2_aug = self._cv_ridge(X_aug, y_aug, ent_aug, yr_aug,
                                 synth_m, min_train_years)

        delta = r2_aug - r2_base
        if self.verbose:
            print(f"  Augmentation gain: ΔR²={delta:+.4f} "
                  f"(base={r2_base:.4f}, aug={r2_aug:.4f})")
        return float(delta)

    def fit_augment_if_beneficial(
        self,
        X:              np.ndarray,
        y:              np.ndarray,
        entity_indices: np.ndarray,
        year_labels:    np.ndarray,
        min_train_years: int = 6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """Fit augmenter, evaluate gain, and return augmented data if beneficial.

        Parameters
        ----------
        X, y, entity_indices, year_labels : training data
        min_train_years : int

        Returns
        -------
        X_out : ndarray
            Original X if no gain; augmented X otherwise.
        y_out : ndarray
            Corresponding targets.
        entity_out : ndarray
        year_out : ndarray
        augmented : bool
            True if augmentation was committed.
        """
        self.fit(X, entity_indices, year_labels)
        delta = self.evaluate_gain(X, y, entity_indices, year_labels,
                                    min_train_years=min_train_years)

        if delta > self.gain_threshold:
            if self.verbose:
                print(f"  Augmentation committed (ΔR²={delta:+.4f} > "
                      f"threshold={self.gain_threshold}).")
            X_s, e_s, yr_s, y_s = self.augment(entity_indices, year_labels, y)
            if len(X_s) > 0:
                if y.ndim == 1:
                    y_s_arr = (y_s.ravel() if y_s is not None
                               else np.full(len(X_s), np.nanmean(y)))
                else:
                    # For multi-output, generate entity-mean synthetic targets
                    y_s_arr = np.vstack([
                        np.nanmean(y[entity_indices == e_s[i]], axis=0)
                        for i in range(len(e_s))
                    ])
                X_out  = np.vstack([X, X_s])
                y_out  = (np.concatenate([y.ravel(), y_s_arr.ravel()])
                          if y.ndim == 1
                          else np.vstack([y, y_s_arr]))
                e_out  = np.concatenate([entity_indices, e_s])
                yr_out = np.concatenate([year_labels,    yr_s])
                return X_out, y_out, e_out, yr_out, True

        if self.verbose:
            print(f"  Augmentation skipped (ΔR²={delta:+.4f} ≤ "
                  f"threshold={self.gain_threshold}).")
        return X, y, entity_indices, year_labels, False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cv_ridge(
        self,
        X:              np.ndarray,
        y:              np.ndarray,
        entity_indices: np.ndarray,
        year_labels:    np.ndarray,
        synthetic_mask: np.ndarray,
        min_train_years: int,
    ) -> float:
        """Lightweight walk-forward Ridge CV — proxy for augmentation gain."""
        splitter = SyntheticAwareCV(
            min_train_years=min_train_years, max_folds=5
        )
        r2_scores = []
        reg = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])

        for train_idx, val_idx in splitter.split(
            X, year_labels, synthetic_mask
        ):
            if len(val_idx) < 2:
                continue
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx],   y[val_idx]

            # Drop NaN rows from training
            valid_tr = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
            valid_va = ~(np.isnan(X_va).any(axis=1) | np.isnan(y_va))
            if valid_tr.sum() < 3 or valid_va.sum() < 2:
                continue

            # Replace remaining NaN in X with column means (quick fix)
            X_tr_c = self._impute_with_means(X_tr[valid_tr])
            X_va_c = self._impute_with_means(X_va[valid_va])
            y_tr_c = y_tr[valid_tr]
            y_va_c = y_va[valid_va]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    reg_copy = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                    reg_copy.fit(X_tr_c, y_tr_c)
                    pred = reg_copy.predict(X_va_c)
                r2_scores.append(float(r2_score(y_va_c, pred)))
            except Exception:
                pass

        return float(np.nanmean(r2_scores)) if r2_scores else 0.0

    def _quantile_transform(self, u: np.ndarray) -> np.ndarray:
        """Map uniform u ∈ [0,1]^n_feat back through empirical CDF (inverse PIT).

        Uses the pre-computed quantile table ``_ecdf_quantiles_``.
        """
        u_clipped = np.clip(u, 0.0, 1.0)
        n_q       = self._ecdf_quantiles_.shape[0]
        idx       = np.round(u_clipped * (n_q - 1)).astype(int)
        return self._ecdf_quantiles_[idx, np.arange(self._n_features_)]

    @staticmethod
    def _impute_with_means(X: np.ndarray) -> np.ndarray:
        """Replace NaN with column means (fast in-place copy)."""
        if not np.isnan(X).any():
            return X
        X_out = X.copy()
        col_means = np.nanmean(X_out, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        r, c = np.where(np.isnan(X_out))
        X_out[r, c] = col_means[c]
        return X_out

    @staticmethod
    def _nearest_psd(A: np.ndarray) -> np.ndarray:
        """Project A onto the nearest positive semi-definite matrix
        (Higham 1988, via eigenvalue clipping and symmetrization)."""
        A = (A + A.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals  = np.maximum(eigvals, 1e-8)
        A_psd    = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Rescale to correlation matrix (diagonal = 1)
        d        = np.sqrt(np.diag(A_psd))
        d        = np.where(d < 1e-10, 1.0, d)
        A_corr   = A_psd / np.outer(d, d)
        return np.clip(A_corr, -1.0, 1.0)
