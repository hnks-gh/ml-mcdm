# -*- coding: utf-8 -*-
"""
Super Learner (Stacked Generalization) Meta-Ensemble
====================================================

Implements the Super Learner algorithm (van der Laan et al., 2007),
a principled approach to ensemble learning that uses a meta-learner
trained on out-of-fold predictions to optimally combine base models.

Architecture:
    Base Models → Out-of-Fold Predictions → Meta-Learner → Final Prediction
         ├─ Gradient Boosting  (ŷ₁)           ↓
         ├─ Random Forest      (ŷ₂)      ElasticNet / Ridge
         └─ Bayesian Ridge     (ŷ₃)      (learns α₁...αₙ)
                                              ↓
                                         ŷ_final = Σ αᵢŷᵢ

Key Properties:
    1. Oracle inequality: Super Learner performs asymptotically
       as well as the best weighted combination of base models
    2. Cross-validated meta-features prevent information leakage
    3. Non-negative weights (NNLS) ensure interpretability and
       per-entity weight clamping prevents boundary artefacts
    4. Panel-aware temporal CV (``_PanelTemporalSplit``) uses the
       *median* entity length to size fold windows, preserving data
       from longer-history entities that the old min-based splitter
       discarded (Bug S-2 fix)
    5. OOF predictions are cached and re-used for conformal calibration
       — the SuperLearner ensemble is never deep-copied (Bug U-2 fix)

Variants:
    - Standard: ElasticNet meta-learner with positive weights
    - Bayesian Stacking: Dirichlet-weighted meta-learner for uncertainty
    - Dynamic: Time-varying weights via exponential weighting

References:
    - van der Laan, Polley & Hubbard (2007). "Super Learner"
      Statistical Applications in Genetics and Molecular Biology
    - Naimi & Balzer (2018). "Stacked Generalization: An Introduction
      to Super Learning" European Journal of Epidemiology
    - Yao et al. (2018). "Using Stacking to Average Bayesian Predictive
      Distributions" Bayesian Analysis
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNetCV, RidgeCV, Ridge
from sklearn.metrics import r2_score, mean_squared_error

logger = logging.getLogger('ml_mcdm')
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls
import copy
import warnings
import functools

from .base import BaseForecaster
from .persistence import PersistenceForecaster


def _silence_warnings(func):
    """Scope all warning filters to the duration of *func* only."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return func(*args, **kwargs)
    return wrapper


class _PanelTemporalSplit:
    """
    Panel-aware temporal cross-validation splitter.

    Unlike ``TimeSeriesSplit`` (which splits on the absolute row position in
    the stacked panel), this splitter identifies the temporal position of
    each row *within its entity* and produces folds that respect temporal
    order for every entity simultaneously.

    For each fold k:
      * **train**: rows whose within-entity time position is in ``[0, cut_k)``
      * **val**  : rows whose within-entity time position is in
                   ``[cut_k, cut_{k+1})``

    This guarantees that no entity's future leaks into training data and
    that every entity contributes both train and validation rows.

    Parameters
    ----------
    n_splits : int
        Number of CV folds (≥ 2).
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(
        self,
        X: np.ndarray,
        entity_indices: Optional[np.ndarray] = None,
    ):
        """
        Yield ``(train_idx, val_idx)`` pairs.

        Parameters
        ----------
        X : ndarray  (n_samples, n_features)
        entity_indices : ndarray of shape (n_samples,), optional
            Entity/group ID for each row.  When *None* the splitter falls
            back to a standard ``TimeSeriesSplit``.
        """
        if entity_indices is None:
            yield from TimeSeriesSplit(n_splits=self.n_splits).split(X)
            return

        unique_entities = np.unique(entity_indices)
        # Rows for each entity in their original (assumed temporal) order
        entity_rows: Dict[Any, np.ndarray] = {
            ent: np.where(entity_indices == ent)[0]
            for ent in unique_entities
        }

        # S-2 FIX: Use the MEDIAN entity length to size the fold windows rather
        # than the MIN.  The old code used T = min(...) — any entity shorter
        # than the fold window would exclude ALL data from longer-history
        # entities, wasting up to 9 years of panel data per province.
        #
        # New behaviour:
        #   * Fold boundaries are derived from the median entity length.
        #   * For each fold, each entity contributes its OWN rows up to its
        #     actual T — longer entities provide full training + validation;
        #     shorter entities that fall below the validation window simply
        #     contribute 0 validation rows for that fold (acceptable).
        T_per_entity = {ent: len(rows) for ent, rows in entity_rows.items()}
        T_median = max(2, int(np.median(list(T_per_entity.values()))))

        if T_median < 2:
            yield from TimeSeriesSplit(n_splits=self.n_splits).split(X)
            return

        # Reserve at least half the timeline as the initial training window
        # so that even the first fold has enough data relative to the feature
        # dimensionality.  The old formula ``max(1, T // (K+1))`` gave only
        # 3 years for T=14, K=3 — far too small for 400+ features.
        min_train_T = max(T_median // 2, T_median // (self.n_splits + 1))
        fold_size = max(1, (T_median - min_train_T) // self.n_splits)

        _n_yielded = 0
        for fold in range(self.n_splits):
            cut = min_train_T + fold * fold_size
            val_end = cut + fold_size
            if cut >= T_median:
                break

            train_idx_parts: List[np.ndarray] = []
            val_idx_parts: List[np.ndarray] = []

            for ent, rows in entity_rows.items():
                T_ent = len(rows)
                # Training: all rows of this entity with within-entity
                # position < cut (clamped to the entity's actual length)
                train_cut_ent = min(cut, T_ent)
                if train_cut_ent > 0:
                    train_idx_parts.append(rows[:train_cut_ent])

                # Validation: rows in [cut, val_end], clamped to T_ent.
                # Entities shorter than `cut` contribute 0 val rows — that
                # is fine; they still contribute to training in later folds.
                val_start_ent = min(cut, T_ent)
                val_end_ent = min(val_end, T_ent)
                if val_end_ent > val_start_ent:
                    val_idx_parts.append(rows[val_start_ent:val_end_ent])

            if not train_idx_parts or not val_idx_parts:
                continue

            train_idx = np.sort(np.concatenate(train_idx_parts))
            val_idx   = np.sort(np.concatenate(val_idx_parts))

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            yield train_idx, val_idx
            _n_yielded += 1

        # Safety guard: yield at least one fold even in degenerate cases
        # (e.g. T_median=2 with n_splits=3 in sub-criteria mode).
        if _n_yielded == 0 and T_median >= 2:
            split_T = max(1, T_median // 2)
            train_parts, val_parts = [], []
            for ent, rows in entity_rows.items():
                T_ent = len(rows)
                sp = min(split_T, T_ent - 1)
                if sp > 0:
                    train_parts.append(rows[:sp])
                if sp < T_ent:
                    val_parts.append(rows[sp:])
            if train_parts and val_parts:
                yield (
                    np.sort(np.concatenate(train_parts)),
                    np.sort(np.concatenate(val_parts)),
                )


class _WalkForwardYearlySplit:
    """
    Walk-forward yearly cross-validation with 1-year validation steps.

    Unlike _PanelTemporalSplit (which uses median entity-length-based fold
    windows), this splitter uses explicit calendar year labels to define
    folds.  Each validation fold covers exactly one calendar year × all
    active entities — directly mirroring the production forecasting task
    (predict year T+1 using data through year T).

    Fold k (0-indexed):
      train : rows where year_label < val_year  (i.e. all prior years)
      val   : rows where year_label == val_year

    where val_year = unique_years[min(min_train_years, len-1) + k].

    Example with 13 target years (2012–2024), min_train_years=8, max_folds=5:
      Fold 0: train 2012–2019, validate 2020
      Fold 1: train 2012–2020, validate 2021
      Fold 2: train 2012–2021, validate 2022
      Fold 3: train 2012–2022, validate 2023
      Fold 4: train 2012–2023, validate 2024

    Parameters
    ----------
    min_train_years : int
        Minimum number of target-year cohorts in the first training fold.
        Default 8 ensures at least 8 years of history before the first
        validation year (first val year = 2020 for labels 2012–2024).
    max_folds : int
        Maximum number of folds to yield.  Prevents excessive folds for
        very long panels.
    """

    def __init__(self, min_train_years: int = 8, max_folds: int = 5):
        self.min_train_years = min_train_years
        self.max_folds = max_folds

    def split(self, X: np.ndarray, year_labels: np.ndarray):
        """
        Yield ``(train_idx, val_idx)`` pairs using calendar year labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Used for shape only; not inspected.
        year_labels : ndarray, shape (n_samples,)
            Integer calendar year for each training row (the *target* year,
            i.e. ``next_yr`` from :meth:`TemporalFeatureEngineer.fit_transform`).
        """
        unique_years = np.sort(np.unique(year_labels))
        n_years = len(unique_years)

        if n_years < 2:
            # Degenerate panel: single split at midpoint
            mid = len(X) // 2
            if mid > 0:
                yield np.arange(mid), np.arange(mid, len(X))
            return

        # Index of the first validation year inside unique_years
        first_val_pos = min(self.min_train_years, n_years - 1)

        n_yielded = 0
        for k in range(self.max_folds):
            val_pos = first_val_pos + k
            if val_pos >= n_years:
                break

            val_year = int(unique_years[val_pos])
            train_idx = np.where(year_labels < val_year)[0]
            val_idx   = np.where(year_labels == val_year)[0]

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            yield np.sort(train_idx), np.sort(val_idx)
            n_yielded += 1

        if n_yielded == 0:
            # Safety fallback: validate on the last available year
            val_year = int(unique_years[-1])
            train_idx = np.where(year_labels < val_year)[0]
            val_idx   = np.where(year_labels == val_year)[0]
            if len(train_idx) > 0 and len(val_idx) > 0:
                yield np.sort(train_idx), np.sort(val_idx)


# Public alias — exposed as `PanelWalkForwardCV` in the module API so downstream
# code (unified.py, tests) can reference it by a descriptive name without having
# to import the private `_WalkForwardYearlySplit` directly.
PanelWalkForwardCV = _WalkForwardYearlySplit


class SuperLearner:
    """
    Super Learner meta-ensemble combining multiple base forecasters.

    The algorithm:
    1. Generate out-of-fold (OOF) predictions using temporal CV
    2. Train a meta-learner on OOF predictions as features
    3. Re-train all base models on full training data
    4. Combine base model predictions using meta-learner weights

    Parameters:
        base_models: Dictionary of {name: BaseForecaster} instances
        meta_learner_type: Type of meta-learner ('elasticnet', 'ridge',
                          'bayesian_stacking')
        n_cv_folds: Number of CV folds for OOF predictions
        cv_min_train_years: Minimum year-label cohorts before first validation
            fold when using _WalkForwardYearlySplit.  Default 7 places the
            first validation at 2019 when year_labels run 2012–2024.
        positive_weights: If True, constrain meta-weights to be non-negative
        normalize_weights: If True, meta-weights sum to 1
        meta_alpha_range: Range of regularization values for meta-learner CV
        temperature: Temperature for the softmax-weighted stacking (higher
            temperature → flatter weights; lower → winner-takes-all). Only
            used when ``meta_learner_type='bayesian_stacking'``.
            Note: despite the name, the current implementation is a
            deterministic *softmax* weighting of OOF R² scores
            (temperature-scaled), not a full Bayesian Dirichlet-posterior
            stacking as in Yao et al. (2018).  The name is retained for
            backward compatibility.
        random_state: Random seed
        verbose: Print progress messages

    Example:
        >>> from forecasting.catboost_forecaster import CatBoostForecaster
        >>> from forecasting.bayesian import BayesianForecaster
        >>>
        >>> base = {
        ...     'catboost': CatBoostForecaster(),
        ...     'bayesian': BayesianForecaster(),
        ... }
        >>> sl = SuperLearner(base_models=base)
        >>> sl.fit(X_train, y_train)
        >>> predictions = sl.predict(X_test)
    """

    def __init__(
        self,
        base_models: Dict[str, BaseForecaster],
        meta_learner_type: str = "ridge",
        n_cv_folds: int = 5,
        cv_min_train_years: int = 8,
        conformal_min_train_years: int = 3,
        positive_weights: bool = True,
        normalize_weights: bool = True,
        meta_alpha_range: Optional[List[float]] = None,
        temperature: float = 5.0,
        random_state: int = 42,
        verbose: bool = True,
        meta_group_lasso_lambda: float = 0.0,
        max_total_stage3_minutes: Optional[float] = None,
        max_secondary_conformal_folds: int = 999,
        allow_skip_secondary_conformal_when_slow: bool = True,
    ):
        self.base_models = base_models
        self.meta_learner_type = meta_learner_type
        self.n_cv_folds = n_cv_folds
        self.cv_min_train_years = cv_min_train_years
        self.conformal_min_train_years = max(2, conformal_min_train_years)
        self.positive_weights = positive_weights
        self.normalize_weights = normalize_weights
        self.meta_alpha_range = meta_alpha_range or [
            0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0
        ]
        self.temperature = temperature
        self.random_state = random_state
        self.verbose = verbose
        # E-04: Group LASSO soft-sharing λ.  When > 0, each output criterion's
        # per-output NNLS weight vector is softly nudged toward the cross-output
        # mean, borrowing strength across correlated criteria.
        # '0.0' (default) = fully independent per-output NNLS (backward compat).
        self._meta_group_lasso_lambda: float = float(meta_group_lasso_lambda)
        self.max_total_stage3_minutes = max_total_stage3_minutes
        self.max_secondary_conformal_folds = max(1, int(max_secondary_conformal_folds))
        self.allow_skip_secondary_conformal_when_slow = bool(
            allow_skip_secondary_conformal_when_slow
        )
        self._max_stage3_seconds: Optional[float] = None
        if max_total_stage3_minutes is not None and float(max_total_stage3_minutes) > 0:
            self._max_stage3_seconds = float(max_total_stage3_minutes) * 60.0

        # Fitted components
        self._fitted_base_models: Dict[str, BaseForecaster] = {}
        self._meta_learner = None
        self._meta_weights: Dict[str, float] = {}
        self._cv_scores: Dict[str, List[float]] = {}
        self._fitted: bool = False
        self._n_outputs: int = 1
        self._oof_r2: Dict[str, float] = {}

        # F-02: Per-output meta-weight matrix.
        # Shape: (n_outputs, n_models) — each row is the NNLS-optimal weight
        # vector for one output criterion.  This supersedes the scalar
        # ``_meta_weights`` dict, which holds the column-mean for backward
        # compatibility only.  Populated by ``_fit_meta_learner()``.
        self._meta_weights_per_output_: Optional[np.ndarray] = None   # (n_outputs, n_models)
        self._meta_weights_col_names_: List[str] = []                 # model insertion order

        # F-03: entity indices captured at fit() time for entity-block
        # bootstrap in ``_compute_dirichlet_weight_std``.
        self._entity_indices_fit_: Optional[np.ndarray] = None

        # OOF ensemble predictions — stored at fit time so that
        # UnifiedForecaster can calibrate conformal intervals from pre-computed
        # residuals without deep-copying the full ensemble (U-2).
        self._oof_ensemble_predictions_: Optional[np.ndarray] = None  # (n_samples, n_outputs)
        self._oof_valid_mask_: Optional[np.ndarray] = None            # (n_samples,) bool — joint (all cols)
        # F-01: per-column valid mask: row i is True for col d if col d of the
        # ensemble OOF prediction at row i is not NaN.  Used by stage5 so that
        # each criterion's conformal predictor calibrates on the maximum number
        # of rows available for *that* criterion, not the joint all-outputs mask.
        self._oof_valid_mask_per_col_: Optional[np.ndarray] = None    # (n_samples, n_outputs)

        # E-02: extended conformal OOF residuals collected via secondary
        # walk-forward sweep (min_train_years = conformal_min_train_years).
        # Shape: (n_total, n_outputs) — covers ALL training years, not just
        # the primary CV window.  Used by stage5 to widen the conformal
        # calibration set for tighter, more reliable coverage.
        self._oof_conformal_residuals_: Optional[np.ndarray] = None   # (n_total, n_outputs)

        # Phase B item 3: Comprehensive diagnostics telemetry for post-mortem analysis.
        # Tracks planned vs completed folds, timing, and termination events per stage.
        self._stage3_diagnostics_: Dict[str, Any] = {
            'primary_cv': {
                'planned_folds': 0,
                'completed_folds': 0,
                'start_time': None,
                'end_time': None,
                'elapsed_seconds': None,
                'per_fold_times': [],  # list of (fold_idx, elapsed_seconds)
                'truncated': False,
                'truncation_reason': None,
            },
            'persistence_cv': {
                'planned_folds': 0,
                'completed_folds': 0,
                'start_time': None,
                'end_time': None,
                'elapsed_seconds': None,
                'per_fold_times': [],
                'truncated': False,
                'truncation_reason': None,
            },
            'secondary_conformal': {
                'planned_folds': 0,
                'completed_folds': 0,
                'start_time': None,
                'end_time': None,
                'elapsed_seconds': None,
                'per_fold_times': [],
                'truncated': False,
                'truncation_reason': None,
            },
            'meta_learner': {
                'start_time': None,
                'end_time': None,
                'elapsed_seconds': None,
            },
            'full_refit': {
                'start_time': None,
                'end_time': None,
                'elapsed_seconds': None,
            },
            'stage3_total': {
                'start_time': None,
                'end_time': None,
                'elapsed_seconds': None,
                'target_budget_seconds': self._max_stage3_seconds,
            },
        }

        # E-03: Dirichlet stacking weight uncertainty estimates.
        # Shape: {model_name: std_of_weight} — computed via bootstrap
        # resampling of OOF rows during meta-learner fit.  Only populated
        # when meta_learner_type == "dirichlet_stacking".
        self._meta_weight_std_: Optional[Dict[str, float]] = None

        # Per-criterion RMSE across CV folds (Phase 4).
        # Shape: {model_name: [ [rmse_c1, ..., rmse_cK]_fold1, ... ]}
        self._cv_scores_per_criterion_: Dict[str, List[List[float]]] = {}

    # ------------------------------------------------------------------
    # F-02: per-output weight accessor
    # ------------------------------------------------------------------
    def _get_weight(self, model_name: str, out_col: int) -> float:
        """Return the meta-weight for *model_name* on output column *out_col*.

        Uses ``_meta_weights_per_output_`` when available (populated after a
        full ``_fit_meta_learner`` call).  Falls back to the scalar
        ``_meta_weights`` dict for backward compatibility (e.g. unit tests
        that construct SuperLearner without calling fit:+).

        Parameters
        ----------
        model_name : str
        out_col    : int   0-based index into the n_outputs axis.
        """
        if (
            self._meta_weights_per_output_ is not None
            and self._meta_weights_col_names_
            and model_name in self._meta_weights_col_names_
        ):
            idx = self._meta_weights_col_names_.index(model_name)
            return float(self._meta_weights_per_output_[out_col, idx])
        return self._meta_weights.get(model_name, 0.0)

    # ------------------------------------------------------------------
    # E-04: Group LASSO soft-sharing for multi-output meta-weights
    # ------------------------------------------------------------------
    def _apply_group_lasso_sharing(
        self,
        W: np.ndarray,
        lambda_group: float,
    ) -> np.ndarray:
        """Proximal soft-sharing step — group LASSO regularisation.

        Given the per-output NNLS weight matrix ``W`` of shape
        ``(n_outputs, n_models)``, each row ``W[d, :]`` is the normalised
        weight vector for output criterion ``d``.  The soft-sharing step
        nudges every row toward the cross-output mean, borrowing strength
        across correlated criteria.

        Objective contribution (group LASSO penalty)::

            λ · Σ_k  ||W[:,k] - W_mean[k]||_2

        where ``W_mean[k] = mean_d W[d,k]`` is the average weight for
        model ``k`` across all output criteria.

        Proximal operator (closed form for this linear structure)::

            W_shared = (1 - γ) · W + γ · W_mean_broadcast
            γ = λ / (λ + spread + ε)           (adaptive shrinkage factor)
            spread = ||W - W_mean||_F / n_outputs   (per-model per-output deviation)

        After sharing, each row is L1-renormalised to ensure weights still
        sum to 1 for every output criterion.

        Parameters
        ----------
        W : ndarray, shape (n_outputs, n_models)
            Per-output NNLS weight matrix, each row summing to 1.
        lambda_group : float
            Soft-sharing strength λ > 0.  0 → no sharing.  1 → full
            equalisation toward the cross-output mean.

        Returns
        -------
        W_shared : ndarray, shape (n_outputs, n_models)
            Shared weight matrix, each row still summing to 1.

        Notes
        -----
        With λ = 0.01 and typical spread ≈ 0.05–0.15, γ ≈ 0.06–0.17 —
        a mild nudge that preserves per-criterion optimality while reducing
        noise for criteria with sparse calibration data.
        """
        if W.ndim != 2 or W.shape[0] < 2:
            # Single output or degenerate — sharing is a no-op.
            return W

        # Cross-output mean weight for each model: shape (1, n_models)
        W_mean = W.mean(axis=0, keepdims=True)

        # Frobenius spread (per-output deviation from the shared mean).
        diff = W - W_mean          # (n_outputs, n_models)
        spread = float(np.linalg.norm(diff, 'fro')) / max(W.shape[0], 1)

        # Adaptive shrinkage factor γ ∈ [0, 1).
        gamma = lambda_group / (lambda_group + spread + 1e-12)
        gamma = float(np.clip(gamma, 0.0, 1.0))

        # Soft-sharing interpolation.
        W_shared = (1.0 - gamma) * W + gamma * W_mean  # (n_outputs, n_models)

        # Renormalise each row so weights sum to 1 per output criterion.
        row_sums = W_shared.sum(axis=1, keepdims=True)  # (n_outputs, 1)
        # Avoid division by zero for degenerate rows.
        row_sums = np.where(row_sums < 1e-15, 1.0, row_sums)
        W_shared = W_shared / row_sums

        return W_shared

    @_silence_warnings
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        entity_indices: Optional[np.ndarray] = None,
        per_model_X: Optional[Dict[str, np.ndarray]] = None,
        year_labels: Optional[np.ndarray] = None,
        refit_X: Optional[np.ndarray] = None,
        refit_y: Optional[np.ndarray] = None,
        refit_per_model_X: Optional[Dict[str, np.ndarray]] = None,
        refit_entity_indices: Optional[np.ndarray] = None,
        fold_correction_fn=None,
        shift_detector=None,
    ) -> "SuperLearner":
        """
        Fit the Super Learner ensemble.

        Stage 1: Generate out-of-fold predictions via temporal CV
        Stage 2: Train meta-learner on OOF predictions
        Stage 3: Re-train base models on refit data (or full data)
        Stage 4: (E-02) Secondary conformal OOF sweep to extend calibration
                 residuals to ALL training years, not just the primary window.

        Args:
            X: Feature matrix used for the CV/OOF phase (n_samples, n_features).
            y: Target values for the CV/OOF phase.
            entity_indices: Optional entity group IDs for panel-aware models.
            per_model_X: Optional per-model feature matrices for the CV/OOF phase.
            year_labels: Optional integer array of shape (n_samples,) giving
                the calendar target year for each training row.  When provided,
                uses :class:`_WalkForwardYearlySplit` (1-year validation steps)
                instead of :class:`_PanelTemporalSplit`.
            refit_X: When not None, base models are re-trained on this matrix
                instead of ``X`` after the CV phase.  Use this to restrict
                retraining to the original training set (excluding a holdout
                year that was appended to ``X`` solely for the last CV fold).
            refit_y: Targets paired with ``refit_X``.
            refit_per_model_X: Per-model matrices paired with ``refit_X``.
            refit_entity_indices: Entity indices paired with ``refit_X``.
            fold_correction_fn: Optional callable ``(model_name, X_fold,
                train_idx, fold_entity_indices) -> X_fold_corrected`` that
                applies fold-aware entity-demean corrections to tree-track
                feature matrices inside the CV loop (E-01 fix).  Called
                once per (fold, model) pair for training and validation folds.
                When None, no correction is applied (default).
            shift_detector: Optional :class:`PanelCovariateShiftDetector`
                (E-08).  When not ``None``, per-fold MMD²-based covariate
                shift detection is run before base-model fitting.  Detected
                shifts yield per-sample importance weights that are forwarded
                to base models whose ``fit()`` accepts ``sample_weight``.
                Uniform weights (1.0) are used when no shift is detected.
                When ``None``, no shift detection is performed (default).

        Returns:
            Self for method chaining
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._n_outputs = y.shape[1]
        n_samples = X.shape[0]
        n_models = len(self.base_models)

        # F-03: capture entity_indices at fit time so that
        # _compute_dirichlet_weight_std can use entity-block bootstrap.
        self._entity_indices_fit_ = entity_indices

        logger.info(
            f"Super Learner: {n_models} base models, {self.n_cv_folds} CV folds"
        )

        stage3_start = time.perf_counter()
        stage3_deadline = (
            stage3_start + self._max_stage3_seconds
            if self._max_stage3_seconds is not None else None
        )
        if stage3_deadline is None:
            logger.info("Stage 3 guardrail: disabled (no time budget configured).")
        else:
            logger.info(
                "Stage 3 guardrail: max_total_stage3_minutes=%.2f (deadline active).",
                float(self.max_total_stage3_minutes),
            )

        def _time_left() -> Optional[float]:
            if stage3_deadline is None:
                return None
            return stage3_deadline - time.perf_counter()

        def _enforce_deadline(stage_name: str) -> None:
            rem = _time_left()
            if rem is not None and rem <= 0:
                raise TimeoutError(
                    "Stage 3 runtime budget exceeded during "
                    f"{stage_name}. Increase ForecastConfig.max_total_stage3_minutes "
                    "or reduce model/CV complexity."
                )

        # ============================================================
        # Stage 1: Generate out-of-fold predictions
        # ============================================================
        stage1_start = time.perf_counter()
        logger.info("Stage 3/1: Primary OOF CV started.")
        # Choose splitter based on whether calendar year labels are provided.
        # _WalkForwardYearlySplit validates on exactly one calendar year per
        # fold — matching the production task of predicting year T+1 from
        # data through year T.  Falls back to _PanelTemporalSplit when no
        # year labels are available (e.g., unit tests with synthetic data).
        if year_labels is not None:
            tscv = _WalkForwardYearlySplit(
                min_train_years=self.cv_min_train_years,
                max_folds=self.n_cv_folds,
            )
            cv_iter = tscv.split(X, year_labels)
        else:
            tscv = _PanelTemporalSplit(n_splits=self.n_cv_folds)
            cv_iter = tscv.split(X, entity_indices=entity_indices)

        # OOF prediction storage: (n_samples, n_models * n_outputs)
        oof_predictions = np.full(
            (n_samples, n_models * self._n_outputs), np.nan
        )
        self._cv_scores = {name: [] for name in self.base_models}
        self._cv_scores_per_criterion_ = {name: [] for name in self.base_models}

        cv_pairs = list(cv_iter)
        n_primary_folds = len(cv_pairs)
        logger.info("Stage 3/1: Planned primary folds = %d.", n_primary_folds)

        # Phase B item 3: Initialize primary CV diagnostics
        self._stage3_diagnostics_['primary_cv']['planned_folds'] = n_primary_folds
        self._stage3_diagnostics_['primary_cv']['start_time'] = time.perf_counter()

        for fold_idx, (train_idx, val_idx) in enumerate(cv_pairs):
            fold_start = time.perf_counter()
            _enforce_deadline("primary CV")
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

            # ── E-08: fold-level covariate shift detection ─────────────────
            # Run MMD² on the common X (tree-track).  Returns uniform (1.0)
            # weights when no significant shift is detected, so the extra
            # computation is the only cost in the no-shift case.
            _fold_weights: Optional[np.ndarray] = None
            if shift_detector is not None:
                _fold_weights = shift_detector.compute_fold_weights(
                    X[train_idx], X[val_idx]
                )

            if self.verbose and year_labels is not None:
                train_yrs = np.unique(year_labels[train_idx])
                val_yrs   = np.unique(year_labels[val_idx])
                # year_labels are target years (T+1); display training range
                # as source-data years by shifting the start back by 1 so
                # the log reads "train 2011–2018, validate 2019" rather than
                # the raw label range "2012–2018".
                train_start = int(train_yrs[0]) - 1
                train_end   = int(train_yrs[-1])
                yr_range = (f"{train_start}–{train_end}"
                            if len(train_yrs) > 1 else str(train_start))
                print(
                    f"    Fold {fold_idx + 1}: train {yr_range}, "
                    f"validate {int(val_yrs[0])} ({len(val_idx)} rows)"
                )

            for m_idx, (name, model) in enumerate(self.base_models.items()):
                # Select per-model feature matrix (two-track preprocessing)
                X_m = per_model_X[name] if (per_model_X and name in per_model_X) else X
                X_train_cv = X_m[train_idx]
                X_val_cv = X_m[val_idx]

                # ── E-01: Apply fold-aware entity demean correction ────────
                # The entity-demeaned features in X_train_tree_ were computed
                # using global entity means (all training years).  In early
                # CV folds some of those years are in the future (leakage).
                # fold_correction_fn adjusts only the _demeaned and
                # _demeaned_momentum columns to use fold-restricted means.
                # Applied only to tree-track matrices (the callable returns
                # the input unchanged for PLS-compressed matrices).
                if fold_correction_fn is not None:
                    _train_ent_fold = (
                        entity_indices[train_idx]
                        if entity_indices is not None else None
                    )
                    _val_ent_fold = (
                        entity_indices[val_idx]
                        if entity_indices is not None else None
                    )
                    X_train_cv = fold_correction_fn(
                        name, X_train_cv, train_idx, _train_ent_fold
                    )
                    X_val_cv = fold_correction_fn(
                        name, X_val_cv, train_idx, _val_ent_fold
                    )

                # Drop rows where any y target is NaN — partial-NaN rows are
                # preserved upstream for imputation reporting but base models
                # require complete target vectors.
                _y_nan = (np.isnan(y_train_cv).any(axis=1)
                          if y_train_cv.ndim > 1 else np.isnan(y_train_cv))
                if _y_nan.any():
                    _valid = ~_y_nan
                    _X_fit = X_train_cv[_valid]
                    _y_fit = y_train_cv[_valid]
                    _ent_fit = (entity_indices[train_idx][_valid]
                                if entity_indices is not None else None)
                else:
                    _X_fit, _y_fit = X_train_cv, y_train_cv
                    _ent_fit = (entity_indices[train_idx]
                                if entity_indices is not None else None)

                # ── E-08: slice importance weights to NaN-dropped rows ─────
                # _fold_weights covers the full train_idx; align to _X_fit.
                _sw: Optional[np.ndarray] = None
                if _fold_weights is not None:
                    _sw = _fold_weights[~_y_nan] if _y_nan.any() else _fold_weights

                try:
                    model_copy = copy.deepcopy(model)
                    # Forward entity_indices and sample_weight to models that
                    # accept them (inspect-based dispatch in _fit_model).
                    self._fit_model(model_copy, _X_fit, _y_fit, _ent_fit,
                                    sample_weight=_sw)

                    # S-1 fix: forward val entity_indices so panel models
                    # use the correct fixed effects for each
                    # validation entity rather than the reference entity.
                    _val_ent = (
                        entity_indices[val_idx]
                        if entity_indices is not None else None
                    )
                    pred = self._predict_model(model_copy, X_val_cv, _val_ent)
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, 1)
                    elif np.ndim(pred) == 0:
                        # Handle 0-D scalar case
                        pred = np.asarray([[pred[()]]] * X_val_cv.shape[0])

                    # Store OOF predictions
                    for out_col in range(self._n_outputs):
                        col_idx = m_idx * self._n_outputs + out_col
                        pred_col = min(out_col, pred.shape[1] - 1)
                        oof_predictions[val_idx, col_idx] = pred[:, pred_col]

                    # Compute CV score: mean R² across all output columns
                    # for this fold (one value per fold, not per output).
                    # Filter NaN validation-target rows per column — targets
                    # intentionally preserve NaN for missing governance data
                    # (features.py M-04) and r2_score / mean_squared_error
                    # raise ValueError on NaN inputs.
                    fold_r2s = []
                    fold_rmse_per_criterion = []
                    for out_col in range(y_val_cv.shape[1]):
                        pred_col = min(out_col, pred.shape[1] - 1)
                        y_col = y_val_cv[:, out_col]
                        p_col = pred[:, pred_col]
                        _valid_val = ~np.isnan(y_col)
                        if _valid_val.sum() < 2:
                            fold_r2s.append(np.nan)
                            fold_rmse_per_criterion.append(np.nan)
                            continue
                        fold_r2s.append(
                            r2_score(y_col[_valid_val], p_col[_valid_val])
                        )
                        fold_rmse_per_criterion.append(
                            float(np.sqrt(mean_squared_error(
                                y_col[_valid_val], p_col[_valid_val]
                            )))
                        )
                    _mean_r2 = (
                        float(np.nanmean(fold_r2s))
                        if not all(np.isnan(r) for r in fold_r2s)
                        else np.nan
                    )
                    self._cv_scores[name].append(_mean_r2)
                    self._cv_scores_per_criterion_[name].append(fold_rmse_per_criterion)

                except Exception as e:
                    logger.warning(f"{name} failed on fold {fold_idx}: {e}")
                    self._cv_scores[name].append(np.nan)

            fold_elapsed = time.perf_counter() - fold_start
            logger.info(
                "Stage 3/1: Primary fold %d/%d completed in %.2fs "
                "(train=%d, val=%d).",
                fold_idx + 1,
                n_primary_folds,
                fold_elapsed,
                len(train_idx),
                len(val_idx),
            )
            # Phase B item 3: Track per-fold completion
            self._stage3_diagnostics_['primary_cv']['completed_folds'] += 1
            self._stage3_diagnostics_['primary_cv']['per_fold_times'].append(
                {'fold_idx': fold_idx, 'elapsed_seconds': fold_elapsed}
            )

        logger.info(
            "Stage 3/1: Primary OOF CV finished in %.2fs.",
            time.perf_counter() - stage1_start,
        )
        # Phase B item 3: Finalize primary CV diagnostics
        self._stage3_diagnostics_['primary_cv']['end_time'] = time.perf_counter()
        self._stage3_diagnostics_['primary_cv']['elapsed_seconds'] = (
            self._stage3_diagnostics_['primary_cv']['end_time'] -
            self._stage3_diagnostics_['primary_cv']['start_time']
        )

        # Compute OOF R² for each model (per-model valid mask, not joint)
        # CRITICAL: filter NaN from y here — targets preserve NaN for missing
        # governance data (features.py M-04).  Without this filter r2_score
        # raises "Input contains NaN" and crashes the entire fit() call,
        # which propagates as "! Forecasting skipped: Input contains NaN."
        for m_idx, name in enumerate(self.base_models):
            model_cols = slice(
                m_idx * self._n_outputs, (m_idx + 1) * self._n_outputs
            )
            valid = ~np.isnan(oof_predictions[:, model_cols]).any(axis=1)
            if valid.sum() > 0:
                r2_vals = []
                for out_col in range(self._n_outputs):
                    col_idx = m_idx * self._n_outputs + out_col
                    y_col = y[valid, out_col]
                    p_col = oof_predictions[valid, col_idx]
                    # Additional filter: drop rows where y itself is NaN
                    _y_ok = ~np.isnan(y_col)
                    if _y_ok.sum() < 2:
                        r2_vals.append(-1.0)
                        continue
                    r2_vals.append(
                        r2_score(y_col[_y_ok], p_col[_y_ok])
                    )
                self._oof_r2[name] = float(np.mean(r2_vals))
            else:
                self._oof_r2[name] = -1.0

        # ────────────────────────────────────────────────────────────
        # Persistence baseline: Naive carry-forward for benchmarking (Phase A)
        # ────────────────────────────────────────────────────────────
        # Run the persistence forecaster through the same CV loop to compute
        # CV R² scores comparable to base models. This enables skill score
        # calculation: SS = (R²_ensemble - R²_persistence) / (1 - R²_persistence)
        if self.verbose:
            logger.info("  Evaluating persistence baseline...")

        stage_persist_start = time.perf_counter()
        logger.info("Stage 3/1b: Persistence baseline CV started.")

        persistence_model = PersistenceForecaster(verbose=False)
        persistence_cv_scores = []
        persistence_oof_preds = np.full((n_samples, self._n_outputs), np.nan)

        # Re-iterate through CV folds to evaluate persistence
        if year_labels is not None:
            tscv_persist = _WalkForwardYearlySplit(
                min_train_years=self.cv_min_train_years,
                max_folds=self.n_cv_folds,
            )
            cv_iter_persist = tscv_persist.split(X, year_labels)
        else:
            tscv_persist = _PanelTemporalSplit(n_splits=self.n_cv_folds)
            cv_iter_persist = tscv_persist.split(X, entity_indices=entity_indices)

        cv_pairs_persist = list(cv_iter_persist)
        n_persist_folds = len(cv_pairs_persist)
        logger.info("Stage 3/1b: Planned persistence folds = %d.", n_persist_folds)

        # Phase B item 3: Initialize persistence CV diagnostics
        self._stage3_diagnostics_['persistence_cv']['planned_folds'] = n_persist_folds
        self._stage3_diagnostics_['persistence_cv']['start_time'] = time.perf_counter()

        for fold_idx, (train_idx, val_idx) in enumerate(cv_pairs_persist):
            fold_start = time.perf_counter()
            _enforce_deadline("persistence CV")
            y_train_persist = y[train_idx]
            y_val_persist = y[val_idx]

            try:
                # Fit persistence on training fold
                persistence_model.fit(X[train_idx], y_train_persist)
                pred_persist = persistence_model.predict(X[val_idx])

                if pred_persist.ndim == 1:
                    pred_persist = pred_persist.reshape(-1, 1)

                # Compute per-fold R² (match base model logic: mean across outputs)
                fold_r2s_persist = []
                for out_col in range(y_val_persist.shape[1]):
                    y_col = y_val_persist[:, out_col]
                    p_col = pred_persist[:, out_col]
                    _valid = ~np.isnan(y_col)
                    if _valid.sum() >= 2:
                        fold_r2s_persist.append(
                            float(r2_score(y_col[_valid], p_col[_valid]))
                        )
                    else:
                        fold_r2s_persist.append(np.nan)

                mean_r2_persist = (
                    float(np.nanmean(fold_r2s_persist))
                    if not all(np.isnan(r) for r in fold_r2s_persist)
                    else np.nan
                )
                persistence_cv_scores.append(mean_r2_persist)

                # Store OOF predictions for conformal calibration (optional)
                persistence_oof_preds[val_idx, :] = pred_persist[:, :self._n_outputs]

            except Exception as e:
                logger.warning(
                    f"Persistence baseline failed on fold {fold_idx}: {e}"
                )
                persistence_cv_scores.append(np.nan)

            fold_elapsed = time.perf_counter() - fold_start
            logger.info(
                "Stage 3/1b: Persistence fold %d/%d completed in %.2fs "
                "(train=%d, val=%d).",
                fold_idx + 1,
                n_persist_folds,
                fold_elapsed,
                len(train_idx),
                len(val_idx),
            )
            # Phase B item 3: Track per-fold completion for persistence CV
            self._stage3_diagnostics_['persistence_cv']['completed_folds'] += 1
            self._stage3_diagnostics_['persistence_cv']['per_fold_times'].append(
                {'fold_idx': fold_idx, 'elapsed_seconds': fold_elapsed}
            )

        # Store persistence CV scores and OOF predictions
        self._cv_scores['Persistence'] = persistence_cv_scores
        self._cv_scores_per_criterion_['Persistence'] = [
            [np.nan] * self._n_outputs
        ] * len(persistence_cv_scores)  # Per-criterion breakdown not available

        # Compute persistence OOF R² (global across all outputs)
        valid_persist = ~np.isnan(persistence_oof_preds).any(axis=1)
        if valid_persist.sum() > 0:
            r2_vals_persist = []
            for out_col in range(self._n_outputs):
                y_col = y[valid_persist, out_col]
                p_col = persistence_oof_preds[valid_persist, out_col]
                _y_ok = ~np.isnan(y_col)
                if _y_ok.sum() >= 2:
                    r2_vals_persist.append(
                        r2_score(y_col[_y_ok], p_col[_y_ok])
                    )
                else:
                    r2_vals_persist.append(-1.0)
            self._oof_r2['Persistence'] = float(np.mean(r2_vals_persist))
        else:
            self._oof_r2['Persistence'] = -1.0

        if self.verbose:
            pers_mean_r2 = float(np.nanmean(persistence_cv_scores))
            logger.info(
                f"  Persistence baseline CV R²: {pers_mean_r2:.4f} "
                f"(OOF: {self._oof_r2['Persistence']:.4f})"
            )
        logger.info(
            "Stage 3/1b: Persistence baseline CV finished in %.2fs.",
            time.perf_counter() - stage_persist_start,
        )
        # Phase B item 3: Finalize persistence CV diagnostics
        self._stage3_diagnostics_['persistence_cv']['end_time'] = time.perf_counter()
        self._stage3_diagnostics_['persistence_cv']['elapsed_seconds'] = (
            self._stage3_diagnostics_['persistence_cv']['end_time'] -
            self._stage3_diagnostics_['persistence_cv']['start_time']
        )

        # ============================================================
        # Stage 2: Train meta-learner on OOF predictions
        # ============================================================
        stage2_start = time.perf_counter()
        logger.info("Stage 3/2: Meta-learner fitting started.")
        _enforce_deadline("meta-learner fitting")
        # Check if ANY model produced enough valid OOF rows (per-model,
        # not joint).  The old joint mask excluded every row when a
        # single model (e.g. QuantileRF) failed, poisoning
        # all rows even for models that succeeded.
        max_valid_per_model = max(
            (~np.isnan(
                oof_predictions[:, m * self._n_outputs:(m + 1) * self._n_outputs]
            ).any(axis=1)).sum()
            for m in range(n_models)
        )
        if max_valid_per_model < 5:
            # Fallback: use simple averaging if not enough OOF data
            logger.warning("Not enough OOF data, falling back to weighted avg")
            self._meta_weights = self._fallback_weights()
        else:
            # Pass full OOF matrix; _fit_meta_learner handles NaN
            # per-model via its own per-output valid filter.
            self._fit_meta_learner(oof_predictions, y)
        logger.info(
            "Stage 3/2: Meta-learner fitting finished in %.2fs.",
            time.perf_counter() - stage2_start,
        )

        # ----------------------------------------------------------
        # Cache ensemble OOF predictions for conformal calibration
        # (U-2): build the meta-weighted blend of per-model OOF preds
        # so UnifiedForecaster can calibrate from these residuals
        # without deep-copying the full ensemble inside cv_plus.
        # ----------------------------------------------------------
        oof_ensemble = np.full((n_samples, self._n_outputs), np.nan)
        for i_out in range(self._n_outputs):
            weighted_col = np.zeros(n_samples)
            weight_sum_col = np.zeros(n_samples)
            for m_idx, name in enumerate(self.base_models):
                # F-02: use per-output weight for this criterion column
                w = self._get_weight(name, i_out)
                if w == 0.0:
                    continue
                col_idx = m_idx * self._n_outputs + i_out
                oof_col = oof_predictions[:, col_idx]
                valid_col = ~np.isnan(oof_col)
                weighted_col[valid_col] += w * oof_col[valid_col]
                weight_sum_col[valid_col] += w
            valid_col_any = weight_sum_col > 0
            oof_ensemble[valid_col_any, i_out] = (
                weighted_col[valid_col_any] / weight_sum_col[valid_col_any]
            )
        self._oof_ensemble_predictions_ = oof_ensemble
        # F-01a: joint valid mask (all outputs non-NaN) — kept for backward
        # compatibility; diagnostics and _oof_predictions_per_model_ use it.
        self._oof_valid_mask_ = ~np.isnan(oof_ensemble).any(axis=1)
        # F-01b: per-column valid mask — used by stage5 conformal predictor so
        # each criterion calibrates on its own maximum available residual set.
        self._oof_valid_mask_per_col_ = ~np.isnan(oof_ensemble)   # (n_samples, n_outputs)

        # ── F-05c C2: per-model OOF predictions for downstream diagnostics ──
        # Slice each base model's columns out of the raw oof_predictions matrix
        # and keep only the valid (non-NaN ensemble) rows so callers get arrays
        # aligned with _oof_ensemble_predictions_[_oof_valid_mask_].
        self._oof_predictions_per_model_: Dict[str, np.ndarray] = {}
        _oof_valid_rows = self._oof_valid_mask_
        for _m_idx, _m_name in enumerate(self.base_models):
            _cols = slice(_m_idx * self._n_outputs, (_m_idx + 1) * self._n_outputs)
            _per_m = oof_predictions[:, _cols]          # (n_samples, n_outputs)
            self._oof_predictions_per_model_[_m_name] = _per_m[_oof_valid_rows]

        # ================================================================
        # Stage 4 (E-02): Extended conformal OOF sweep
        # ================================================================
        # When ``conformal_min_train_years < cv_min_train_years`` and
        # ``year_labels`` is available, run an additional walk-forward pass
        # starting from ``conformal_min_train_years`` to collect OOF
        # residuals for early training years not covered by the primary CV.
        #
        # Example: primary CV (min=8) → val years 2020–2024 (315 residuals)
        #          secondary sweep (min=3) → val years 2015–2019 (315 more)
        #          combined calibration set: 630 residuals → tighter q̂
        #
        # The secondary sweep uses the same base_models (unfitted templates),
        # re-fitting each per fold with the same entity_indices routing.
        # Meta-weights from Stage 2 are applied to produce ensemble OOF
        # predictions, which are compared against y to form residuals.
        #
        # Theoretical validity: conservative coverage guaranteed because
        # early-fold base models are trained on fewer years → slightly larger
        # residuals → q̂ may be slightly inflated → intervals are wider than
        # necessary but never under-covering (distribution-free guarantee).
        # ================================================================
        stage4_start = time.perf_counter()
        logger.info("Stage 3/4: Secondary conformal OOF sweep started.")
        self._oof_conformal_residuals_ = self._build_conformal_oof_residuals(
            X=X,
            y=y,
            entity_indices=entity_indices,
            per_model_X=per_model_X,
            year_labels=year_labels,
            primary_oof=oof_ensemble,
            fold_correction_fn=fold_correction_fn,
            stage3_deadline=stage3_deadline,
            max_secondary_folds=self.max_secondary_conformal_folds,
            allow_skip_when_slow=self.allow_skip_secondary_conformal_when_slow,
        )
        logger.info(
            "Stage 3/4: Secondary conformal OOF sweep finished in %.2fs.",
            time.perf_counter() - stage4_start,
        )

        # ============================================================
        # Stage 3: Re-train all base models on refit data
        # ============================================================
        stage3_refit_start = time.perf_counter()
        logger.info("Stage 3/3: Final full-data refit started.")
        _enforce_deadline("final full-data refit")
        # When refit_X / refit_y are provided (e.g. training-only data without
        # the holdout year that was appended to X for the CV phase), base models
        # are retrained on that subset so Stage 6 holdout evaluation remains
        # leakage-free.  Falls back to the CV data when not provided.
        _rx   = refit_X            if refit_X            is not None else X
        _ry   = refit_y            if refit_y            is not None else y
        _rpmX = refit_per_model_X  if refit_per_model_X  is not None else per_model_X
        _rei  = refit_entity_indices if refit_entity_indices is not None else entity_indices
        # Drop NaN-target rows once for the full refit data
        _ry_nan = (np.isnan(_ry).any(axis=1)
                   if _ry.ndim > 1 else np.isnan(_ry))
        if _ry_nan.any():
            _rv = ~_ry_nan
            _rx_clean   = _rx[_rv]
            _ry_clean   = _ry[_rv]
            _rpmX_clean = ({k: v[_rv] for k, v in _rpmX.items()}
                           if _rpmX else None)
            _rei_clean  = _rei[_rv] if _rei is not None else None
        else:
            _rx_clean, _ry_clean = _rx, _ry
            _rpmX_clean, _rei_clean = _rpmX, _rei
        for name, model in self.base_models.items():
            _enforce_deadline(f"final full-data refit ({name})")
            try:
                X_m = _rpmX_clean[name] if (_rpmX_clean and name in _rpmX_clean) else _rx_clean
                fitted_model = copy.deepcopy(model)
                self._fit_model(fitted_model, X_m, _ry_clean, _rei_clean)
                self._fitted_base_models[name] = fitted_model
            except Exception as e:
                logger.warning(f"{name} failed on full data: {e}")

        logger.info(
            "Stage 3/3: Final full-data refit finished in %.2fs.",
            time.perf_counter() - stage3_refit_start,
        )

        self._fitted = True

        logger.info("Super Learner meta-weights:")
        for name, w in sorted(
            self._meta_weights.items(), key=lambda x: x[1], reverse=True
        ):
            bar = "█" * int(w * 30)
            logger.info(f"  {name:25s}: {w:.4f} {bar}")

        total_stage3 = time.perf_counter() - stage3_start
        _rem = _time_left()
        logger.info(
            "Stage 3 summary: elapsed=%.2fs%s",
            total_stage3,
            (
                f", remaining_budget={_rem:.2f}s"
                if _rem is not None else ""
            ),
        )

        # Phase B item 3: Finalize Stage 3 total diagnostics
        self._stage3_diagnostics_['stage3_total']['end_time'] = time.perf_counter()
        self._stage3_diagnostics_['stage3_total']['start_time'] = stage3_start
        self._stage3_diagnostics_['stage3_total']['elapsed_seconds'] = total_stage3

        return self

    @staticmethod
    def _fit_model(model, X, y, entity_indices=None, sample_weight=None):
        """Fit a base model, forwarding entity_indices and sample_weight when supported.

        Uses ``inspect.signature`` to detect which keyword arguments the model's
        ``fit()`` accepts, then passes only the supported ones.  This avoids
        coupling the SuperLearner to any specific base-model interface.

        Parameters
        ----------
        model : BaseForecaster
            The model to fit (already deep-copied by the caller).
        X, y : ndarray
            Feature matrix and targets.
        entity_indices : ndarray, optional
            Panel entity IDs forwarded to models that accept ``entity_indices``
            or ``group_indices``.
        sample_weight : ndarray of shape (n_train,), optional
            Per-sample importance weights from E-08
            ``PanelCovariateShiftDetector``.  Forwarded only when the model's
            ``fit()`` declares a ``sample_weight`` parameter.
        """
        import inspect
        sig = inspect.signature(model.fit)
        fit_kwargs: dict = {}
        # Entity routing
        if 'entity_indices' in sig.parameters and entity_indices is not None:
            fit_kwargs['entity_indices'] = entity_indices
        elif 'group_indices' in sig.parameters and entity_indices is not None:
            fit_kwargs['group_indices'] = entity_indices
        # Importance weights routing
        if 'sample_weight' in sig.parameters and sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        model.fit(X, y, **fit_kwargs)

    @staticmethod
    def _predict_model(model, X, entity_indices=None):
        """Call predict on a base model, forwarding entity_indices when supported."""
        import inspect
        sig = inspect.signature(model.predict)
        if 'entity_indices' in sig.parameters and entity_indices is not None:
            return model.predict(X, entity_indices=entity_indices)
        return model.predict(X)

    def _build_conformal_oof_residuals(
        self,
        X: np.ndarray,
        y: np.ndarray,
        entity_indices: Optional[np.ndarray],
        per_model_X: Optional[Dict[str, np.ndarray]],
        year_labels: Optional[np.ndarray],
        primary_oof: np.ndarray,
        fold_correction_fn=None,
        stage3_deadline: Optional[float] = None,
        max_secondary_folds: int = 999,
        allow_skip_when_slow: bool = True,
    ) -> Optional[np.ndarray]:
        """
        E-02 — Extended conformal OOF residuals covering ALL training years.

        Runs a secondary walk-forward CV starting from
        ``self.conformal_min_train_years`` and collects ensemble OOF
        predictions for every year *not* covered by the primary CV.  The
        returned array is the element-wise difference ``y − ŷ_oof`` over
        the extended window, pooled with the primary OOF residuals so that
        conformal calibration uses the full training history.

        Returns
        -------
        combined_residuals : ndarray, shape (n_total_valid, n_outputs), or None
            Extended OOF residuals combining primary and secondary sweeps.
            ``None`` when ``year_labels`` is absent (non-panel mode) or when
            ``conformal_min_train_years >= cv_min_train_years`` (no secondary
            sweep needed because primary already covers all years).
        """
        n_samples, n_outputs = y.shape[0], y.shape[1] if y.ndim > 1 else 1

        # Guard: secondary sweep only makes sense with year_labels and when
        # the secondary window starts earlier than the primary.
        if (
            year_labels is None
            or self.conformal_min_train_years >= self.cv_min_train_years
        ):
            return None

        if stage3_deadline is not None and time.perf_counter() >= stage3_deadline:
            if allow_skip_when_slow:
                logger.warning(
                    "Stage 3/4: Skipping secondary conformal sweep because "
                    "the Stage-3 time budget is exhausted."
                )
                return None
            raise TimeoutError(
                "Stage 3 runtime budget exceeded before secondary conformal "
                "sweep."
            )

        # Determine which validation years are already in the primary OOF.
        primary_has_oof = ~np.isnan(primary_oof).all(axis=1)
        primary_val_years = (
            set(int(yy) for yy in np.unique(year_labels[primary_has_oof]))
            if primary_has_oof.any() else set()
        )

        # CRITICAL: Restrict secondary conformal to TRUE early-gap years only.
        #
        # The primary CV is capped by max_folds (e.g., 5 folds), so late years
        # may be missing from primary_OOF simply due to the fold cap, not
        # because they are early data.  Secondary sweep should only cover years
        # BEFORE the earliest year validated by primary CV — these are the
        # true "early-gap" years representing early historical data that primary
        # CV cannot access due to min_train_years constraint.
        #
        # Example: if primary validates on [2020, 2021, 2022, 2023, 2024] but
        # due to fold cap only covers [2022, 2023, 2024], secondary should still
        # only target years < 2020 (true early-gap), not re-validate on
        # [2020, 2021] which are just late-years missing due to cap.
        min_primary_val_year = min(primary_val_years) if primary_val_years else None

        # Build secondary splitter — starts earlier than primary.
        secondary_splitter = _WalkForwardYearlySplit(
            min_train_years=self.conformal_min_train_years,
            max_folds=999,  # exhaust all years
        )

        # Collect per-row secondary OOF predictions
        secondary_oof = np.full((n_samples, n_outputs), np.nan)

        candidate_folds: List[Tuple[np.ndarray, np.ndarray]] = []
        for _train_idx, _val_idx in secondary_splitter.split(X, year_labels):
            _val_years = set(int(yy) for yy in np.unique(year_labels[_val_idx]))
            # Skip folds whose val_year is already covered by primary OOF
            if _val_years.issubset(primary_val_years):
                continue
            # NEW: Also skip late-year folds that are NOT early-gap years.
            # Include only folds where ALL validation years < min(primary_val_years).
            if min_primary_val_year is not None and not all(
                yy < min_primary_val_year for yy in _val_years
            ):
                continue
            candidate_folds.append((_train_idx, _val_idx))

        if max_secondary_folds is not None and max_secondary_folds > 0:
            candidate_folds = candidate_folds[:int(max_secondary_folds)]

        logger.info(
            "Stage 3/4: Planned secondary conformal folds = %d "
            "(cap=%d).",
            len(candidate_folds),
            int(max_secondary_folds),
        )

        # Phase B item 3: Initialize secondary conformal diagnostics
        self._stage3_diagnostics_['secondary_conformal']['planned_folds'] = len(candidate_folds)
        self._stage3_diagnostics_['secondary_conformal']['start_time'] = time.perf_counter()

        for _fold_idx, (_train_idx, _val_idx) in enumerate(candidate_folds):
            _fold_start = time.perf_counter()
            if stage3_deadline is not None and time.perf_counter() >= stage3_deadline:
                if allow_skip_when_slow:
                    logger.warning(
                        "Stage 3/4: Secondary conformal sweep truncated at fold %d/%d "
                        "due to Stage-3 time budget.",
                        _fold_idx,
                        len(candidate_folds),
                    )
                    break
                raise TimeoutError(
                    "Stage 3 runtime budget exceeded during secondary conformal "
                    f"fold {_fold_idx + 1}."
                )

            _y_train = y[_train_idx]
            _y_nan = (
                np.isnan(_y_train).any(axis=1)
                if _y_train.ndim > 1 else np.isnan(_y_train)
            )
            _valid = ~_y_nan
            _ent_train = (
                entity_indices[_train_idx][_valid]
                if entity_indices is not None else None
            )
            _ent_val = (
                entity_indices[_val_idx]
                if entity_indices is not None else None
            )

            # Fit all base models and combine with per-output meta-weights (F-02)
            secondary_preds = []
            secondary_names = []   # track names for _get_weight() calls
            for m_idx, (name, model) in enumerate(self.base_models.items()):
                # Gate: if scalar (mean) weight is 0 all per-output weights are 0
                scalar_w = self._meta_weights.get(name, 0.0)
                if scalar_w == 0.0:
                    continue
                X_m = (
                    per_model_X[name]
                    if (per_model_X and name in per_model_X) else X
                )
                _X_train_cv = X_m[_train_idx][_valid]
                _X_val_cv   = X_m[_val_idx]

                # Apply fold correction for entity demean features (E-01)
                if fold_correction_fn is not None:
                    _X_train_cv = fold_correction_fn(
                        name, _X_train_cv, _train_idx[_valid], _ent_train
                    )
                    _X_val_cv = fold_correction_fn(
                        name, _X_val_cv, _train_idx, _ent_val
                    )

                try:
                    model_copy = copy.deepcopy(model)
                    self._fit_model(
                        model_copy, _X_train_cv, _y_train[_valid], _ent_train
                    )
                    pred = self._predict_model(model_copy, _X_val_cv, _ent_val)
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, 1)
                    elif np.ndim(pred) == 0:
                        pred = np.asarray([[float(pred)]] * len(_val_idx))
                    secondary_preds.append(pred)
                    secondary_names.append(name)
                except Exception:
                    continue

            if not secondary_preds:
                logger.info(
                    "Stage 3/4: Secondary fold %d/%d produced no valid base-model "
                    "predictions.",
                    _fold_idx + 1,
                    len(candidate_folds),
                )
                continue

            # F-02: per-output weighted ensemble for secondary fold.
            # Each output criterion uses its own optimal weight vector.
            n_val = len(_val_idx)
            ens_pred = np.zeros((n_val, n_outputs))
            for out_col in range(n_outputs):
                wt_sum = 0.0
                for pp, nm in zip(secondary_preds, secondary_names):
                    w = self._get_weight(nm, out_col)
                    if w < 1e-15:
                        continue
                    pred_col = min(out_col, pp.shape[1] - 1)
                    ens_pred[:, out_col] += w * pp[:, pred_col]
                    wt_sum += w
                if wt_sum > 1e-15:
                    ens_pred[:, out_col] /= wt_sum

            # ens_pred is already (n_val, n_outputs) — assign all columns at once
            secondary_oof[np.ix_(_val_idx, np.arange(n_outputs))] = ens_pred
            _fold_elapsed = time.perf_counter() - _fold_start
            logger.info(
                "Stage 3/4: Secondary fold %d/%d completed in %.2fs "
                "(train=%d, val=%d).",
                _fold_idx + 1,
                len(candidate_folds),
                _fold_elapsed,
                len(_train_idx),
                len(_val_idx),
            )
            # Phase B item 3: Track per-fold completion for secondary conformal
            self._stage3_diagnostics_['secondary_conformal']['completed_folds'] += 1
            self._stage3_diagnostics_['secondary_conformal']['per_fold_times'].append(
                {'fold_idx': _fold_idx, 'elapsed_seconds': _fold_elapsed}
            )

        # Phase B item 3: Finalize secondary conformal diagnostics (before return)
        self._stage3_diagnostics_['secondary_conformal']['end_time'] = time.perf_counter()
        self._stage3_diagnostics_['secondary_conformal']['elapsed_seconds'] = (
            self._stage3_diagnostics_['secondary_conformal']['end_time'] -
            self._stage3_diagnostics_['secondary_conformal']['start_time']
        )
        # Check if truncation occurred
        if self._stage3_diagnostics_['secondary_conformal']['completed_folds'] < \
           self._stage3_diagnostics_['secondary_conformal']['planned_folds']:
            self._stage3_diagnostics_['secondary_conformal']['truncated'] = True
            self._stage3_diagnostics_['secondary_conformal']['truncation_reason'] = (
                "Stage 3 time budget exhausted"
            )

        # Pool primary and secondary OOF residuals
        # primary_oof covers the later calibration years, secondary_oof the earlier
        combined_oof = np.where(
            np.isnan(primary_oof), secondary_oof, primary_oof
        )
        valid_rows = ~np.isnan(combined_oof).all(axis=1)

        if valid_rows.sum() == 0:
            return None

        # Return residuals (y - ŷ_oof) over the combined valid set
        y_valid  = y[valid_rows]
        oof_valid = combined_oof[valid_rows]

        # Zero-out NaN target cells before residual computation
        _y_ok = ~np.isnan(y_valid)
        residuals = np.where(_y_ok, y_valid - oof_valid, np.nan)
        return residuals

    def _fit_meta_learner(self, oof_X: np.ndarray, oof_y: np.ndarray):
        """Fit the second-level meta-learner on OOF predictions.

        Handles partial OOF data gracefully: models whose OOF columns are
        entirely NaN (because they failed on every fold) are excluded from
        the NNLS / meta-learner fit and receive weight 0.  Only models
        with valid predictions participate, so one crashed model no longer
        poisons the meta-weight estimation for the remaining models.
        """
        n_models = len(self.base_models)
        model_names = list(self.base_models.keys())

        # Fit per output column, average weights
        all_coefs = []

        for out_col in range(self._n_outputs):
            # Extract model predictions for this output
            model_preds = np.column_stack([
                oof_X[:, m_idx * self._n_outputs + out_col]
                for m_idx in range(n_models)
            ])

            y_col = oof_y[:, out_col]

            # Identify models that have *any* valid OOF predictions
            model_has_data = [
                (~np.isnan(model_preds[:, m])).sum() > 0
                for m in range(n_models)
            ]
            active_indices = [m for m in range(n_models) if model_has_data[m]]

            if not active_indices:
                all_coefs.append(np.ones(n_models) / n_models)
                continue

            # Build sub-matrix using only active (non-failed) models
            active_preds = model_preds[:, active_indices]
            valid = ~np.isnan(active_preds).any(axis=1) & ~np.isnan(y_col)
            if valid.sum() < 3:
                all_coefs.append(np.ones(n_models) / n_models)
                continue

            active_preds_valid = active_preds[valid]
            y_valid = y_col[valid]

            # Compute coefficients for active models only
            active_coefs = None

            if self.meta_learner_type == "elasticnet":
                meta = ElasticNetCV(
                    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                    alphas=self.meta_alpha_range,
                    positive=self.positive_weights,
                    cv=max(2, min(3, valid.sum() // 2)),
                    max_iter=5000,
                    random_state=self.random_state,
                )
            elif self.meta_learner_type == "bayesian_stacking":
                # Bayesian stacking: softmax-weighted by OOF R²
                scores = np.array([
                    float(np.nanmean(self._cv_scores[model_names[m]]))
                    if not np.all(np.isnan(self._cv_scores[model_names[m]]))
                    else 0.0
                    for m in active_indices
                ])
                scores = np.clip(scores, 0, None)
                exp_scores = np.exp(self.temperature * scores)
                active_coefs = exp_scores / exp_scores.sum()
            elif self.meta_learner_type == "dirichlet_stacking":
                # Yao et al. (2018) — true log-score Bayesian model stacking.
                # Maximises: w* = argmax_{w ∈ Δ_K} Σ_n log( Σ_k w_k · N(y_n; ŷ_kn, σ_k²) )
                # where σ_k² = OOF MSE of model k (Gaussian predictive density approx).
                # Optimisation: unconstrained L-BFGS-B in logit space, softmax → simplex.
                active_coefs = self._fit_dirichlet_stacking(
                    active_preds_valid, y_valid
                )
            else:  # ridge
                if self.positive_weights:
                    # Use NNLS for a proper non-negative least-squares
                    # solution instead of post-hoc clipping of Ridge coefs.
                    try:
                        coefs_nnls, _ = nnls(active_preds_valid, y_valid)
                        active_coefs = coefs_nnls
                        if out_col == 0:
                            self._meta_learner = None  # no sklearn meta object
                    except Exception:
                        pass  # fall through to RidgeCV

                if active_coefs is None:
                    meta = RidgeCV(
                        alphas=self.meta_alpha_range,
                        cv=TimeSeriesSplit(n_splits=max(2, min(3, valid.sum() // 2))),
                    )

            # Fit sklearn meta-learner if active_coefs not already set
            if active_coefs is None:
                try:
                    meta.fit(active_preds_valid, y_valid)
                    active_coefs = meta.coef_.copy()
                    if self.positive_weights:
                        active_coefs = np.maximum(active_coefs, 0)
                    if out_col == 0:
                        self._meta_learner = meta
                except Exception:
                    active_coefs = np.ones(len(active_indices)) / len(active_indices)

            # Scatter active coefficients back into full-model vector
            full_coefs = np.zeros(n_models)
            for i, m_idx in enumerate(active_indices):
                full_coefs[m_idx] = active_coefs[i]
            all_coefs.append(full_coefs)

        # Normalize each per-output coefficient vector to sum to 1.
        normed_coefs = []
        for c in all_coefs:
            s = float(np.sum(c))
            normed_coefs.append(c / s if s > 1e-15
                                else np.ones(len(c)) / len(c))
        normed_coefs_arr = np.array(normed_coefs)   # (n_outputs, n_models)

        # E-04: Group LASSO soft-sharing — borrow strength across correlated
        # output criteria.  When _meta_group_lasso_lambda > 0, each criterion's
        # per-output weight vector is nudged toward the cross-output mean.
        # λ = 0.0 (default) → pure per-output NNLS, backward-compatible.
        if self._meta_group_lasso_lambda > 1e-15:
            normed_coefs_arr = self._apply_group_lasso_sharing(
                normed_coefs_arr, self._meta_group_lasso_lambda
            )

        # F-02: Store per-output weight matrix.
        # _meta_weights_per_output_[d, k] = optimal weight for model k on
        # output criterion d.  This is the authoritative weight source used
        # by _get_weight(), OOF ensemble caching, and predict().
        self._meta_weights_per_output_ = normed_coefs_arr          # (n_outputs, n_models)
        self._meta_weights_col_names_  = list(self.base_models.keys())

        # Backward-compat scalar weights: unweighted mean across outputs so
        # that all callers of the old _meta_weights dict still work correctly.
        # These are used ONLY for diagnostics and logging — all ensemble
        # combination internally goes through _get_weight().
        avg_coefs = normed_coefs_arr.mean(axis=0)                  # (n_models,)
        if self.normalize_weights:
            coef_sum = avg_coefs.sum()
            if coef_sum > 1e-15:
                avg_coefs = avg_coefs / coef_sum
            else:
                avg_coefs = np.ones(n_models) / n_models

        self._meta_weights = dict(zip(self.base_models.keys(), avg_coefs))

        # E-03: For dirichlet stacking, estimate meta-weight uncertainty via
        # entity-block parametric bootstrap (F-03: entity_indices are stored
        # at fit() time in self._entity_indices_fit_).
        if self.meta_learner_type == "dirichlet_stacking":
            self._meta_weight_std_ = self._compute_dirichlet_weight_std(oof_X, oof_y)

    def _fit_dirichlet_stacking(
        self,
        preds: np.ndarray,   # (n_valid, K_active)  OOF predictions for active models
        y: np.ndarray,       # (n_valid,)            true targets for this output column
        max_iter: int = 500,
    ) -> np.ndarray:
        """Yao et al. (2018) log-score Bayesian stacking for one output column.

        Maximises the leave-one-fold-out log predictive score:

            w* = argmax_{w ∈ Δ_K} Σ_n log( Σ_k w_k · N(y_n; ŷ_kn, σ_k²) )

        where σ_k² = mean OOF squared error of model k (Gaussian predictive
        density approximation for point forecasters).

        Parametrisation: w = softmax(logits) — guarantees w ∈ Δ_K without
        constraint handling.  Optimised via L-BFGS-B in unconstrained logit
        space.  Falls back to NNLS if scipy is unavailable.

        Parameters
        ----------
        preds : ndarray, shape (n_valid, K_active)
            OOF predictions, one column per active base model.
        y : ndarray, shape (n_valid,)
            True target values (no NaNs expected).
        max_iter : int
            Maximum L-BFGS-B iterations.

        Returns
        -------
        weights : ndarray, shape (K_active,)
            Non-negative mixture weights summing to 1.
        """
        K = preds.shape[1]
        if K == 1:
            return np.ones(1)

        try:
            from scipy.optimize import minimize
            from scipy.special import softmax as _sfx
        except ImportError:
            # scipy not available — fall back to equal weights
            return np.ones(K) / K

        # Per-model predictive variance σ_k² = MSE on OOF rows.
        # Clamped to ≥ 1e-8 to avoid log(0) in the density.
        sigma2 = np.maximum(
            np.mean((y[:, np.newaxis] - preds) ** 2, axis=0),
            1e-8,
        )  # (K,)
        log_sigma   = 0.5 * np.log(sigma2)   # (K,)
        inv_2sigma2 = 0.5 / sigma2            # (K,)
        n_inner     = y.shape[0]

        # F-04: analytical gradient — avoids 2K finite-difference evals per
        # iteration (Yao et al. 2018, score-function identity for softmax mix).
        #
        # Derivation (kept here for auditability):
        #   L   = −Σ_n log(Σ_k w_k · p_k(y_n))
        #   r_nk = w_k · p_k(y_n) / Σ_j w_j·p_j(y_n)   (responsibility)
        #   ∂L/∂logit_j = n · w_j − Σ_n r_nj
        #               = n · (w_j − mean_n r_nj)
        # where w = softmax(logits).
        def _neg_log_score_and_grad(
            logits: np.ndarray,
        ) -> Tuple[float, np.ndarray]:
            w     = _sfx(logits)                        # (K,)  simplex
            log_w = np.log(w + 1e-30)

            diff2  = (y[:, np.newaxis] - preds) ** 2   # (n, K)
            log_pk = -inv_2sigma2 * diff2 - log_sigma   # (n, K)  Gaussian log-density (const omitted)

            # log-mixture via log-sum-exp for numerical stability
            lse_in  = log_w[np.newaxis, :] + log_pk    # (n, K)
            lse_max = lse_in.max(axis=1, keepdims=True) # (n, 1)
            exp_in  = np.exp(lse_in - lse_max)          # (n, K) — relative to max
            mix_sum = exp_in.sum(axis=1, keepdims=True)  # (n, 1)
            log_mix = lse_max[:, 0] + np.log(mix_sum[:, 0] + 1e-30)  # (n,)

            nll = float(-np.sum(log_mix))

            # Responsibility matrix: r[n, k] = w_k · p_k / Σ_j w_j·p_j
            responsibility = exp_in / (mix_sum + 1e-30)   # (n, K)

            # Analytical gradient ∂NLL/∂logit_j = n·w_j − Σ_n r_{nj}
            grad = n_inner * w - responsibility.sum(axis=0)  # (K,)

            return nll, grad.astype(np.float64)

        x0  = np.zeros(K)
        res = minimize(
            _neg_log_score_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,                                   # F-04: provide analytical gradient
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8},
        )
        return _sfx(res.x)

    def _compute_dirichlet_weight_std(
        self,
        oof_X: np.ndarray,    # (n_samples, n_models * n_outputs)
        oof_y: np.ndarray,    # (n_samples, n_outputs)
        n_boot: int = 200,
    ) -> Dict[str, float]:
        """Bootstrap estimate of Dirichlet stacking weight standard deviation.

        Resamples OOF rows with replacement ``n_boot`` times, refits
        ``_fit_dirichlet_stacking`` per output column, averages across output
        columns, then returns the per-model std across bootstrap replicates as
        a dict keyed by model name.

        Parameters
        ----------
        oof_X : ndarray, shape (n_samples, n_models × n_outputs)
            OOF predictions stacked column-wise (same layout as in
            ``_fit_meta_learner``).
        oof_y : ndarray, shape (n_samples, n_outputs)
        n_boot : int
            Number of bootstrap replicates (200 ≈ ±2% std error on std).

        Returns
        -------
        weight_std : dict  {model_name: float}
        """
        n_models     = len(self.base_models)
        model_names  = list(self.base_models.keys())
        n_samples    = oof_X.shape[0]
        rng          = np.random.RandomState(self.random_state)
        boot_weights: list = []   # list of (K_full,) arrays

        # F-03: entity-block bootstrap — resample whole entity time-series
        # to preserve within-entity temporal dependence and cross-entity
        # correlation in the panel.  Falls back to i.i.d. row bootstrap when
        # entity_indices are unavailable (e.g. unit-test path).
        entity_indices = getattr(self, '_entity_indices_fit_', None)
        if entity_indices is not None and len(entity_indices) == n_samples:
            unique_entities = np.unique(entity_indices)
            # Build lookup: entity → row positions in OOF arrays
            _ent_rows: Dict[Any, np.ndarray] = {
                e: np.where(entity_indices == e)[0]
                for e in unique_entities
            }
        else:
            unique_entities = None
            _ent_rows = {}

        for _ in range(n_boot):
            if unique_entities is not None and len(unique_entities) >= 2:
                # Entity-block bootstrap: resample entities with replacement,
                # then concatenate their row indices.  This preserves within-
                # entity temporal structure and cross-entity dependence.
                sampled_ents = rng.choice(
                    unique_entities, size=len(unique_entities), replace=True
                )
                idx = np.concatenate([_ent_rows[e] for e in sampled_ents])
            else:
                # Fallback: i.i.d. row resample (no entity info available)
                idx = rng.randint(0, n_samples, size=n_samples)

            b_X     = oof_X[idx]
            b_y     = oof_y[idx]

            per_out_coefs = []
            for out_col in range(self._n_outputs):
                model_preds = np.column_stack([
                    b_X[:, m * self._n_outputs + out_col]
                    for m in range(n_models)
                ])
                y_col = b_y[:, out_col] if b_y.ndim > 1 else b_y
                model_has_data = [(~np.isnan(model_preds[:, m])).sum() > 0
                                  for m in range(n_models)]
                active_idx = [m for m in range(n_models) if model_has_data[m]]
                if not active_idx:
                    per_out_coefs.append(np.ones(n_models) / n_models)
                    continue
                active_preds = model_preds[:, active_idx]
                valid = ~np.isnan(active_preds).any(axis=1) & ~np.isnan(y_col)
                if valid.sum() < 3:
                    per_out_coefs.append(np.ones(n_models) / n_models)
                    continue
                a_coefs = self._fit_dirichlet_stacking(
                    active_preds[valid], y_col[valid], max_iter=200
                )
                full = np.zeros(n_models)
                for i, mi in enumerate(active_idx):
                    full[mi] = a_coefs[i]
                s = full.sum()
                per_out_coefs.append(full / s if s > 1e-12 else full)

            # Average across output columns (mirrors backward-compat scalar weights)
            avg = np.mean(per_out_coefs, axis=0)
            s   = avg.sum()
            boot_weights.append(avg / s if s > 1e-12 else avg)

        std_arr = np.std(boot_weights, axis=0)   # (K_full,)
        return dict(zip(model_names, std_arr))

    def _fallback_weights(self) -> Dict[str, float]:
        """Compute fallback weights from CV scores when meta-learner fails.

        Failed folds are stored as ``np.nan`` and excluded via
        ``np.nanmean``.  Models whose folds all failed get weight 0.
        """
        scores = {}
        for name, s in self._cv_scores.items():
            arr = np.array(s, dtype=float)
            if np.all(np.isnan(arr)):
                scores[name] = 0.0
            else:
                scores[name] = max(0.0, float(np.nanmean(arr)))
        total = sum(scores.values())
        if total > 0:
            return {name: v / total for name, v in scores.items()}
        return {name: 1.0 / len(scores) for name in scores}

    def predict(
        self,
        X: np.ndarray,
        entity_indices: Optional[np.ndarray] = None,
        per_model_X_pred: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Make predictions using the Super Learner ensemble.

        Combines base model predictions using learned meta-weights.

        Args:
            X: Feature matrix of shape (n_samples, n_features). Default for
                all models unless overridden by ``per_model_X_pred``.
            entity_indices: Optional entity group IDs for panel-aware models
            per_model_X_pred: Optional dict mapping model name → prediction
                feature matrix (mirrors the ``per_model_X`` used in ``fit``).

        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_outputs)
        """
        if not self._fitted:
            raise ValueError("Super Learner not fitted. Call fit() first.")

        all_predictions = []
        model_names = []

        for name, model in self._fitted_base_models.items():
            X_m = per_model_X_pred[name] if (per_model_X_pred and name in per_model_X_pred) else X
            try:
                pred = self._predict_model(model, X_m, entity_indices)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                all_predictions.append(pred)
                model_names.append(name)
            except Exception:
                continue

        if not all_predictions:
            raise ValueError("No base models produced predictions")

        # F-02: per-output weighted combination.
        # For each output criterion d, use the weight vector for that specific
        # criterion rather than a single scalar averaged across all criteria.
        n_samples = X.shape[0]
        result = np.zeros((n_samples, self._n_outputs))

        for out_col in range(self._n_outputs):
            # Collect per-output weights for active models; renormalize in
            # case some models failed prediction on this call.
            col_weights = {
                name: self._get_weight(name, out_col)
                for name in model_names
            }
            w_sum = sum(col_weights.values())
            if w_sum > 1e-15:
                col_weights = {n: w / w_sum for n, w in col_weights.items()}
            else:
                # All weights zero — equal fallback
                col_weights = {n: 1.0 / len(model_names) for n in model_names}

            for pred, name in zip(all_predictions, model_names):
                w = col_weights.get(name, 0.0)
                pred_col = min(out_col, pred.shape[1] - 1)
                result[:, out_col] += w * pred[:, pred_col]

        if self._n_outputs == 1:
            return result.ravel()
        return result

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        entity_indices: Optional[np.ndarray] = None,
        per_model_X_pred: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty from model disagreement.

        Uncertainty = weighted standard deviation of base model predictions.

        Args:
            X: Feature matrix. Default for all models unless overridden by
                ``per_model_X_pred``.
            entity_indices: Optional entity group IDs, forwarded to panel-aware
                base models.
                When *None*, panel models fall back to population-level (zero-dummy)
                predictions, making the uncertainty estimate entity-wrong.
                Always pass the same ``entity_indices`` used during ``predict()``.
            per_model_X_pred: Optional dict mapping model name → prediction
                feature matrix (mirrors the ``per_model_X`` used in ``fit``).

        Returns:
            Tuple of (mean prediction, standard deviation)
        """
        all_predictions = []
        model_names = []

        for name, model in self._fitted_base_models.items():
            X_m = per_model_X_pred[name] if (per_model_X_pred and name in per_model_X_pred) else X
            try:
                # Use _predict_model so that panel-aware models receive
                # entity_indices (fixing audit issue S-1: the previous
                # model.predict(X) call omitted entity_indices, causing
                # wrong uncertainty for entity-level predictions).
                pred = self._predict_model(model, X_m, entity_indices)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                all_predictions.append(pred)
                model_names.append(name)
            except Exception:
                continue

        if not all_predictions:
            raise ValueError("No base models produced predictions")

        pred_stack = np.stack(
            [p[:, :self._n_outputs] if p.shape[1] >= self._n_outputs
             else np.column_stack([p] * self._n_outputs)
             for p in all_predictions],
            axis=0,
        )

        # Keep the predictive mean consistent with predict(): use per-output
        # meta-weights, then compute model-disagreement uncertainty on top.
        n_samples = pred_stack.shape[1]
        mean_pred = np.zeros((n_samples, self._n_outputs))
        std_pred = np.zeros((n_samples, self._n_outputs))

        for out_col in range(self._n_outputs):
            col_weights = np.array([
                self._get_weight(name, out_col)
                for name in model_names
            ], dtype=float)
            w_sum = float(col_weights.sum())
            if w_sum > 1e-15:
                col_weights = col_weights / w_sum
            else:
                col_weights = np.ones(len(model_names), dtype=float) / len(model_names)

            mean_pred[:, out_col] = np.average(
                pred_stack[:, :, out_col],
                weights=col_weights,
                axis=0,
            )
            diff_col = pred_stack[:, :, out_col] - mean_pred[:, out_col][np.newaxis, :]
            var_col = np.average(diff_col ** 2, weights=col_weights, axis=0)
            std_pred[:, out_col] = np.sqrt(var_col)

        if self._n_outputs == 1:
            return mean_pred.ravel(), std_pred.ravel()
        return mean_pred, std_pred

    def get_meta_weights(self) -> Dict[str, float]:
        """Get the learned meta-learner weights."""
        return self._meta_weights.copy()

    def get_cv_scores(self) -> Dict[str, List[float]]:
        """Get cross-validation R² scores for each base model."""
        return self._cv_scores.copy()

    def get_oof_performance(self) -> Dict[str, float]:
        """Get out-of-fold R² for each base model."""
        return self._oof_r2.copy()

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics."""
        return {
            "meta_learner_type": self.meta_learner_type,
            "n_cv_folds": self.n_cv_folds,
            "n_base_models": len(self.base_models),
            "meta_weights": self._meta_weights,
            "oof_r2": self._oof_r2,
            "mean_cv_scores": {
                name: float(np.nanmean(scores))
                for name, scores in self._cv_scores.items()
            },
            "positive_weights": self.positive_weights,
        }

    def get_stage3_diagnostics(self) -> Dict[str, Any]:
        """Phase B item 3: Get Stage 3 execution diagnostics.

        Returns a dictionary with planned vs completed fold counts,
        timing information, and truncation telemetry for each major
        stage (primary CV, persistence CV, secondary conformal, etc.).
        Useful for post-mortem analysis of runtime behavior and
        identifying compute bottlenecks.

        Returns
        -------
        diagnostics : dict
            Keys are stage names ('primary_cv', 'persistence_cv',
            'secondary_conformal', 'meta_learner', 'full_refit',
            'stage3_total'), each containing:
            - planned_folds / completed_folds (for CV loops)
            - start_time, end_time, elapsed_seconds (wall-clock)
            - per_fold_times: list of dicts with fold-level timings
            - truncated / truncation_reason (if applicable)
        """
        return self._stage3_diagnostics_.copy()

    def save_stage3_diagnostics(self, output_path: str) -> None:
        """Phase B item 3: Save Stage 3 diagnostics to JSON file.

        Parameters
        ----------
        output_path : str
            File path to save diagnostics JSON (recommend output/logs/stage3_diags.json).
            Parent directories are created if they don't exist.
        """
        import json
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert timestamps to ISO format strings for JSON serialization
        diags_json = {}
        for stage_name, stage_data in self._stage3_diagnostics_.items():
            if isinstance(stage_data, dict):
                diags_json[stage_name] = {}
                for key, value in stage_data.items():
                    if isinstance(value, float) and value > 1e6 and value < 1e10:
                        # Likely a timestamp (perf_counter in seconds since Start)
                        # Just keep as-is; it's relative so doesn't need ISO conversion
                        diags_json[stage_name][key] = value
                    else:
                        diags_json[stage_name][key] = value

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(diags_json, f, indent=2)
            logger.info(f"Saved Stage 3 execution diagnostics to {path}")
        except Exception as e:
            logger.warning(f"Failed to save diagnostics to {path}: {e}")

