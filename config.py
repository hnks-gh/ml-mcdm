# -*- coding: utf-8 -*-
"""
Centralised Configuration for ML-MCDM Pipeline
================================================

All configurable parameters are defined here as typed dataclasses.
The master ``Config`` class composes every sub-config and provides
serialisation, summary printing, and global singleton management.

Configuration Groups
--------------------
- PathConfig               — directory structure
- PanelDataConfig          — entity / time / hierarchy dimensions
- RandomConfig             — reproducibility seeds
- TOPSISConfig             — TOPSIS method parameters
- VIKORConfig              — VIKOR compromise parameter
- WeightingConfig          — Hybrid MC Ensemble + Bayesian Bootstrap
- EvidentialReasoningConfig— two-stage ER aggregation
- ForecastConfig           — ML forecasting ensemble settings
- ValidationConfig         — sensitivity / robustness analysis
- VisualizationConfig      — figure appearance defaults
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Literal
from enum import Enum
import json

from data.imputation import ImputationConfig


# =========================================================================
# Enumerations
# =========================================================================

class NormalizationType(Enum):
    """Supported normalization methods for traditional MCDM."""
    VECTOR = "vector"
    MINMAX = "minmax"
    ZSCORE = "zscore"
    MAX = "max"


class WeightMethod(Enum):
    """Supported weighting method families."""
    CRITIC = "critic"
    EQUAL = "equal"


class AggregationType(Enum):
    """Supported global rank aggregation."""
    EVIDENTIAL_REASONING = "evidential_reasoning"


# =========================================================================
# Path Configuration
# =========================================================================

@dataclass
class PathConfig:
    """File and directory paths, all derived from *base_dir*."""
    base_dir: Path = field(default_factory=lambda: Path.cwd())

    @property
    def data_dir(self) -> Path:
        """Root of the data/ directory (codebook, csv/ sub-dir)."""
        return self.base_dir / "data"

    @property
    def data_csv_dir(self) -> Path:
        """Input CSV files directory (``data/csv/YYYY.csv``)."""
        return self.data_dir / "csv"

    @property
    def output_dir(self) -> Path:
        return self.base_dir / "output" / "result"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def csv_dir(self) -> Path:
        return self.output_dir / "csv"

    # Keep old name as alias for backward compatibility
    @property
    def results_dir(self) -> Path:
        return self.csv_dir

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    def ensure_directories(self) -> None:
        """Create every output directory if missing."""
        phases = ["weighting", "ranking", "mcdm", "forecasting", "sensitivity", "summary"]
        dirs = [
            self.data_dir, self.output_dir,
            self.reports_dir, self.logs_dir,
        ]
        # figures phase subdirs
        for phase in phases:
            dirs.append(self.figures_dir / phase)
        # csv phase subdirs
        for phase in phases:
            dirs.append(self.csv_dir / phase)
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# =========================================================================
# Data Configuration
# =========================================================================

@dataclass
class PanelDataConfig:
    """Panel data dimensions and naming conventions."""
    n_provinces: int = 63
    years: List[int] = field(default_factory=lambda: list(range(2011, 2025)))
    province_col: str = "Province"
    year_col: str = "Year"

    # Hierarchy
    # NOTE: n_subcriteria = 28 (SC52 is permanently excluded as of 2021)
    # C01:4, C02:4, C03:3, C04:4, C05:3 (was 4, excluding SC52), C06:4, C07:3, C08:3  → 28 total
    n_subcriteria: int = 28
    n_criteria: int = 8
    subcriteria_prefix: str = "SC"
    criteria_prefix: str = "C"

    @property
    def n_years(self) -> int:
        return len(self.years)

    @property
    def n_observations(self) -> int:
        return self.n_provinces * self.n_years

    # Number of sub-criteria per criterion, in C01..C08 order.
    # C01:4, C02:4, C03:3, C04:4, C05:3, C06:4, C07:3, C08:3  → 28 total
    # NOTE: C05 reduced from 4 to 3 because SC52 is excluded globally
    _subcriteria_per_criterion: List[int] = field(
        default_factory=lambda: [4, 4, 3, 4, 3, 4, 3, 3],
        init=False,
        repr=False,
    )

    @property
    def subcriteria_cols(self) -> List[str]:
        """Return SC codes in dataset order: SC11–SC14, SC21–SC24, …, SC81–SC83.
        
        Permanently excludes SC52 (discontinued from 2021 onward).
        For C05: returns [SC51, SC53, SC54] (skips SC52).
        """
        codes: List[str] = []
        for crit_idx, n_sub in enumerate(self._subcriteria_per_criterion, start=1):
            for sub_idx in range(1, n_sub + 1):
                # Special handling for C05: skip SC52
                if crit_idx == 5:  # C05
                    # Map sub_idx 1→1, 2→3, 3→4 (to align with codebook)
                    actual_sub_idx = sub_idx if sub_idx < 2 else sub_idx + 1
                    codes.append(f"{self.subcriteria_prefix}{crit_idx}{actual_sub_idx}")
                else:
                    codes.append(f"{self.subcriteria_prefix}{crit_idx}{sub_idx}")
        return codes

    @property
    def criteria_cols(self) -> List[str]:
        return [f"{self.criteria_prefix}{i+1:02d}"
                for i in range(self.n_criteria)]

    @property
    def train_years(self) -> List[int]:
        return self.years[:-1]

    @property
    def test_year(self) -> int:
        return self.years[-1]


# =========================================================================
# Reproducibility
# =========================================================================

@dataclass
class RandomConfig:
    """Random-state and resampling defaults."""
    seed: int = 42


# =========================================================================
# MCDM Method Parameters
# =========================================================================

@dataclass
class TOPSISConfig:
    """TOPSIS configuration."""
    normalization: NormalizationType = NormalizationType.VECTOR
    benefit_criteria: Optional[List[str]] = None   # None → all benefit
    cost_criteria: Optional[List[str]] = None


@dataclass
class VIKORConfig:
    """VIKOR compromise parameter."""
    v: float = 0.5


# =========================================================================
# Weighting Configuration
# =========================================================================

@dataclass
class WeightingConfig:
    """CRITIC Weighting configuration.

    Two-level deterministic design
    ──────────────────────────────
    Level 1 : CRITIC on each criterion group (m × n_k matrices)
              → local SC weights summing to 1 within each group
    Level 2 : CRITIC on criterion composite matrix
              → criterion weights summing to 1 globally
    Global  : global_SC_weight = local_SC_weight × criterion_weight

    Fully deterministic — no Monte Carlo, no Beta blending, no tuning grid.
    Temporal stability analysis is handled separately in ``analysis/``.
    """
    # ── Numerics ────────────────────────────────────────────────────────
    epsilon: float = 1e-10

    # ── Stability (analytical, not MC-based) ────────────────────────────
    stability_threshold:     float = 0.95
    perform_stability_check: bool  = True


# =========================================================================
# Evidential Reasoning Configuration
# =========================================================================

@dataclass
class EvidentialReasoningConfig:
    """Two-stage ER aggregation (Yang & Xu, 2002).

    Stage 1 — Within each criterion, combine 6 traditional MCDM method scores.
    Stage 2 — Combine 8 criterion beliefs with criterion weights.
    """
    n_grades: int = 5
    method_weight_scheme: Literal["equal", "rank_based"] = "equal"
    base_methods: List[str] = field(default_factory=lambda: [
        # Traditional
        "topsis", "vikor", "promethee", "copras", "edas", "saw",
    ])


# =========================================================================
# ML Forecasting (State-of-the-Art Ensemble)
# =========================================================================

@dataclass
class ForecastConfig:
    """
    State-of-the-art forecasting configuration for UnifiedForecaster.

    Uses diverse base models + Meta-Learner ensemble + Conformal
    Prediction calibration.  Optimised for small-to-medium panel data (N < 1000).

    Model hyperparameters
    ---------------------
    gb_max_depth / gb_n_estimators
        CatBoostForecaster (oblivious symmetric trees, MultiRMSE objective).
        ``max_depth=5`` is the principled midpoint between the underfitting
        depth-4 and the overfitting depth-6 at n≈756: depth-5 yields
        32 leaves ≈ 24 samples/leaf, providing a healthy bias-variance
        trade-off.  ``n_estimators=200`` aligns the ensemble build with
        the standalone class default (previously hardcoded to 100 in
        ``_create_models``).
    """
    enabled: bool = True
    target_year: Optional[int] = None  # Auto-set to latest_year + 1

    # ===== PHASE 4: Feature Selection & Multicollinearity =====
    max_vif_threshold: float = 10.0
    target_max_features: int = 80

    # ===== PHASE B Enhancement: MICE-Only Imputation Configuration =====
    imputation_config: Optional[ImputationConfig] = field(default_factory=ImputationConfig)
    """Configuration for MICE imputation strategy (Phase B+).
    
    Uses single unified MICE imputation for all missing features (ExtraTreesRegressor).
    Replaces prior multi-tier block-level imputation (PHASE A) for simplicity and
    optimality — MICE automatically captures multivariate feature correlations without
    per-block tier configuration.
    
    Key parameters:
    - use_mice_imputation : bool = True  (enable MICE preprocessing)
    - n_imputations : int = 5             (multiple imputation for uncertainty)
    - mice_estimator : str = "extra_trees"
    - mice_max_iter : int = 20
    - mice_add_indicator : bool = True    (append _was_missing columns)
    
    DEPRECATED (kept for backward compatibility, ignored):
    - use_advanced_feature_imputation
    - block_imputation_tiers
    - temporal_imputation_window
    
    See ImputationConfig docstring in data/imputation/__init__.py for details.
    """
    
    use_multiple_imputation: bool = True
    """Enable multiple imputation with Rubin's Rules (Phase B+).
    
    When True, the forecaster wraps base models with MultipleImputationForecaster
    to generate M = n_imputations stochastic MICE imputations per training fold.
    Predictions are pooled via Rubin's Rules for:
    
    - **Pooled prediction**: (1/M) Σ ŷ^(m)
    - **Total variance**: Var_within + (1+1/M)·Var_between
      - Var_within = (1/M) Σ Var^(m)      [average within-imputation variance]
      - Var_between = (1/(M-1)) Σ (ŷ^(m) − ȳ)²  [between-imputation variance]
    
    Benefits:
    ✓ Prediction intervals reflect missingness-induced uncertainty
    ✓ FMI (Fraction of Missing Information) quantifies impact
    ✓ Properly propagates imputation uncertainty into final forecasts
    
    When False, uses single-imputation point estimates (faster, no uncertainty).
    Default True for production systems requiring uncertainty quantification.
    
    References:
    Rubin, D. B. (1987). Multiple Imputation for Nonresponse in Surveys.
    van Buuren, S., & Groothuis-Oudshoorn, K. (2011).
        mice: Multivariate Imputation by Chained Equations in R.
    """

    # ─────────────────────────────────────────────────────────────────────

    # ── Conformal prediction ─────────────────────────────────────────────
    conformal_method: str = 'cv_plus'  # 'split', 'cv_plus', 'adaptive'
    conformal_alpha: float = 0.05      # 95% joint coverage (Bonferroni across components)
    uncertainty_method: str = 'qrf_quantile'
    """Prediction interval method.

    ``'qrf_quantile'`` — Heteroscedastic intervals from the fitted Quantile
        Random Forest.  Each entity (province) receives interval widths that
        reflect the empirical spread of training labels in the same leaf nodes
        as that entity's feature vector.  Volatile entities get wider, stable
        ones narrower bands.  Bonferroni-corrected per-component quantiles
        (``lower_q = α / (2D)``) guarantee joint coverage ≥ 1 − α across all
        D criteria simultaneously.

    ``'conformal'`` — Distribution-free conformal prediction intervals
        calibrated from out-of-fold residuals.  Provides marginal coverage
        guarantee and homoscedastic widths (constant across entities).
    """

    # ── Cross-validation ─────────────────────────────────────────────────
    cv_folds: int = 5
    cv_min_train_years: int = 5
    """Minimum number of unique year-label cohorts before the first validation
    fold.  With year_labels = target years (2012–2024):

    *  ``5`` (Phase 2 default): first validation year is 2017 (index 5 in
       [2012…2024]).  Produces 7 validation folds × 63 entities = **441 OOF rows**
       for meta-learner NNLS, vs. 315 rows (5 folds) at the old default of 8.
       The +40% OOF data reduces meta-weight estimation variance; each of the
       8 output criteria now has ≈55 OOF rows (vs. ≈40 previously).
    *  ``8`` (previous default): first validation year 2020;
       fold design: Fold 1 → train 2011–2019, validate 2020.

    Phase 2.5 reduction: ``8 → 5``; conformal extended OOF sweep
    (``cv_conformal_min_train_years=3``) remains unchanged and continues
    to contribute additional residuals for uncertainty calibration only."""
    cv_conformal_min_train_years: int = 3
    """Minimum training cohorts for the **conformal-only** extended OOF sweep
    (E-02).  The SuperLearner runs a secondary walk-forward pass to generate
    OOF residuals for early years not covered by the primary CV
    (``cv_min_train_years=8`` leaves years 2012–2019 without OOF).

    Setting 3 yields early folds: val = 2015, 2016, 2017, 2018, 2019 on
    top of the primary 5 folds (val = 2020...2024), giving ~8 additional
    fold-years × 63 entities = 504 more calibration residuals.  These
    extended residuals are pooled with the primary OOF residuals for
    conformal calibration only — they do NOT influence meta-weights.

    Minimum 2 required (at least 2 training cohorts before first val year).
    Set equal to ``cv_min_train_years`` to disable the secondary sweep."""

    # ── Phase II SOTA Conformal Enhancements ─────────────────────────────
    # All flags default False / off for full backward compatibility.
    # Enable individually to activate the corresponding SOTA technique.

    # E-02: Conformalized Quantile Regression (Romano, Patterson & Candès 2019)
    use_cqr_calibration: bool = False
    """Enable CQR (Conformalized Quantile Regression) for prediction intervals.

    When True, stage5 uses the Quantile Random Forest's per-entity quantile
    predictions as a heteroscedastic base, calibrated by a single additive
    shift ``q̂_CQR`` (Romano, Patterson & Candès, NeurIPS 2019, Eq. 2):

        C(x) = [Q̂_{α/2}(x) − q̂_CQR,  Q̂_{1−α/2}(x) + q̂_CQR]

    Advantages over split conformal (homoscedastic):
    - Adaptive width: volatile provinces get wider, stable ones narrower.
    - Same marginal coverage guarantee (≥ 1 − α), achieved by calibrating
      the conformity score ``s_i = max(Q̂_lo(x_i) − y_i, y_i − Q̂_hi(x_i))``.
    - 15–25% sharper intervals on average vs. constant-width split conformal.

    Requires ``uncertainty_method = 'qrf_quantile'`` to be effective; falls
    back to standard split conformal when QRF fitting fails.
    Default False (backward-compatible; mirrors pre-Phase-II behaviour).
    """

    # E-03: Mondrian Conformal Prediction stratified by missingness rate
    use_mondrian_conformal: bool = False
    """Enable Mondrian conformal prediction stratified by feature missingness.

    When True, stage5 partitions calibration residuals into ``conformal_n_strata``
    strata based on each province's fraction of missing input features.
    A per-stratum Papadopoulos quantile is computed, yielding stratum-conditional
    coverage guarantees (Vovk et al. 2005, Chapter 3):

        P(Y ∈ C(X) | stratum(X) = s) ≥ 1 − α   for each stratum s.

    Strata use equal-frequency binning of the calibration missingness
    distribution.  Strata with fewer than 5 calibration samples fall back to
    the global quantile to avoid under-coverage.

    Default False. Typically combined with ``conformal_n_strata = 3``.
    """

    conformal_n_strata: int = 3
    """Number of missingness strata for Mondrian conformal (E-03).

    Equal-frequency bins of the calibration missingness distribution.
    - 2: binary low/high split.
    - 3 (default): low / medium / high — recommended for n_cal ≥ 300.
    - 4+: finer stratification; requires larger calibration sets.
    Only used when ``use_mondrian_conformal = True``.
    """

    # E-04: Multi-output group LASSO soft-sharing across output criteria
    meta_group_lasso_lambda: float = 0.0
    """Group LASSO soft-sharing strength λ for multi-output meta-weights (E-04).

    When > 0, each output criterion's per-output NNLS weight vector is softly
    nudged toward the cross-output mean via the proximal operator:

        W_shared[d, :] = (1 − γ) · W[d, :] + γ · mean_d W[d, :]
        γ = λ / (λ + spread + ε)   (adaptive shrinkage factor)

    This borrows strength across correlated criteria, reducing estimation
    variance for small-n criteria.  Expected gain: +2–5% RMSE.
    - 0.0 (default): fully independent per-output NNLS, backward-compatible.
    - 0.1: mild sharing — recommended starting point.
    - 0.5: moderate sharing; effective when criteria are highly correlated.
    - 1.0: full equalisation toward the cross-output mean weight.
    """

    # E-05: Locally Weighted Conformal Prediction (entity-aware kernel)
    use_locally_weighted_conformal: bool = False
    """Enable Locally Weighted Conformal Prediction (Tibshirani et al. 2019).

    When True, stage5 uses per-test-point RBF-kernel importance weights over
    the calibration set.  Same-entity calibration rows receive a multiplicative
    ``conformal_entity_weight`` boost, reflecting higher exchangeability between
    a province's own history and its future prediction.

    Weighted quantile (Tibshirani et al. 2019, Eq. 3):
        w_i ∝ K_h(‖x_i − x_test‖) × boost_i,   Σ w_i = 1

    Highest-priority conformal branch in stage5 when enabled.
    Default False.
    """

    conformal_entity_weight: float = 2.0
    """Same-entity calibration point weight boost for LWCP (E-05).

    Calibration residuals from the same entity (province) as the test
    prediction receive this multiplicative factor before RBF-kernel weighting.
    - 1.0: no entity boost (pure feature-space RBF weighting).
    - 2.0 (default): double-weight same-province history rows.
    - 5.0+: strongly entity-specific intervals.
    Only used when ``use_locally_weighted_conformal = True``.
    """

    # E-06: Student-t predictive distribution for small-n calibration
    conformal_studentt_small_n: bool = False
    """Enable Student-t predictive quantile for small-n conformal calibration (E-06).

    When True and a criterion's calibration set size is below
    ``conformal_studentt_threshold``, the conformal half-width is computed
    via maximum-likelihood Student-t fit to |OOF residuals|, using predictive
    degrees-of-freedom ``df_pred = df_MLE − 1`` (Meeker & Escobar 1998, §4.3).

    Falls back to Papadopoulos when n_cal > 100, scipy is unavailable, or
    MLE optimisation diverges.
    Default False (opt-in).
    """

    conformal_studentt_threshold: int = 50
    """Calibration set size threshold below which Student-t quantile is used (E-06).

    - 30: conservative — Student-t only when Papadopoulos is very noisy.
    - 50 (default): balanced — activates for early Mondrian strata.
    - 80: aggressive — Student-t even for moderate-sized calibration sets.
    Only used when ``conformal_studentt_small_n = True``.
    """

    meta_learner_type: str = "ridge"
    """Second-level meta-learner used to combine base-model OOF predictions.

    Options
    -------
    ``"ridge"`` (default)
        Non-negative least squares (NNLS) when ``positive_weights=True``
        (the production default), otherwise scikit-learn ``RidgeCV``.
        Fast, deterministic, low-variance despite small OOF count.
    ``"dirichlet_stacking"``
        Yao et al. (2018) approximate Bayesian model stacking.
        Maximises the leave-one-fold-out log predictive score via
        L-BFGS-B in logit (softmax) space with Gaussian predictive
        densities.  Naturally handles model uncertainty; avoids the
        NNLS hard zero-weight artefact.  Adds ~2–5 s fitting overhead
        and stores bootstrap weight std in ``_meta_weight_std_``.
    ``"bayesian_stacking"``
        Lightweight temperature-softmax over OOF R² scores.  Fast but
        ignores predictive density — kept for backwards compatibility.
    ``"elasticnet"``
        ElasticNetCV with positive constraint; useful when base models
        are highly collinear.
    """

    # ── Reproducibility ──────────────────────────────────────────────────
    random_state: int = 42
    verbose: bool = True

    # ── CatBoost hyperparameters ─────────────────────────────────────────
    gb_max_depth: int = 5
    """Tree depth. 5 ≈ 32 leaves → ~24 samples/leaf at n=756."""
    gb_n_estimators: int = 200
    """Number of boosting stages; aligned with CatBoostForecaster defaults."""

    # ── Hyperparameter auto-tuning (Phase 3) ─────────────────────────────
    auto_tune_gb: bool = False
    """When True, runs a one-time optuna TPE search to find optimal hyperparameters
    for CatBoostForecaster before ensemble training."""
    
    auto_tune_kernel: bool = False
    """When True, runs optuna TPE search for KernelRidge and SVR models."""
    
    auto_tune_qrf: bool = False
    """When True, runs optuna TPE search for QuantileRandomForest."""

    hp_tune_n_trials: int = 40
    """Number of optuna TPE trials per model during hyperparameter search."""
    
    hp_tune_timeout_seconds: int = 3600
    """Maximum timeout in seconds for tuning a single model."""

    # ── Phase 2: Early Stopping for Gradient Boosting (Phase 2.1) ────────
    gb_early_stopping_rounds: int = 20
    """Number of consecutive boosting rounds with no improvement on the
    internal chronological validation set before training is halted.

    **Why chronological?** — Panel CV folds are ordered by calendar year.
    We hold out the chronologically *last* ``gb_validation_fraction`` of
    each fold's training rows as the early-stopping monitor set.  This
    preserves temporal integrity: the monitor set always represents years
    *after* the model training window, avoiding any look-ahead bias.

    With ``learning_rate=0.05`` and ``n_train ≈ 200–500``, the loss
    typically plateaus at 50–120 iterations (vs. the 200–300 max), so
    early stopping recovers 50–75% of wasted compute and prevents
    overfitting on small CV folds (a known root cause of unstable
    negative CV R² episodes observed in audit runs).

    Set ``0`` to disable early stopping and recover old behaviour
    (full-iteration training regardless of validation loss).
    """

    gb_validation_fraction: float = 0.20
    """Fraction of each fold's training rows held out as the chronological
    early-stopping validation set (Phase 2.1).

    Rows are selected by index position (last ``round(n_train × frac)``
    rows), which corresponds to the most recent year-cohorts because the
    feature matrix is sorted ``(year, entity)`` by the feature engineer.

    * ``0.20`` (default): 20% holdout.
      - ``n_train=400`` → ``n_es_val=80`` — robust loss estimate.
      - ``n_train=200`` → ``n_es_val=40`` — marginal but sufficient.
    * Minimum of ``max(10, round(n_train × frac))`` is enforced so the
      validation set always has at least 10 samples even for tiny folds.
    * Early stopping is suppressed when ``n_train < 40`` to avoid leaving
      fewer than 30 actual training rows after the split.
    Only used when ``gb_early_stopping_rounds > 0``.
    """

    # ── Phase 2: QuantileRF stabilization (Phase 2.4) ────────────────────
    qrf_n_estimators: int = 300
    """Number of trees in the Quantile Random Forest (QuantileRF) ensemble.

    **Bug fix (Phase 2.4)**: ``_create_models()`` previously hardcoded
    ``n_estimators=100``, ignoring both the class default (200) and this
    config field.  The fix reads this value and passes it through, raising
    the effective tree count to 300.

    **Rationale for 300**: QRF's quantile estimates are empirical CDFs over
    leaf training samples.  At the extreme quantiles used for 95%
    prediction intervals (q=0.025, q=0.975), estimates require many trees
    to stabilise.  Breiman (2001) recommends ≥10 × √n_features for OOB
    error convergence; with ~700 tree-track features √700 ≈ 26, so
    ≥260 trees.  300 gives comfortable headroom.

    Typical fitting time: ~8–12 s per output column on CPU.
    """
    qrf_min_samples_leaf: int = 3
    """Default minimum number of training samples required at each leaf node
    in the Quantile Random Forest.

    ``fit()`` auto-scales this value when ``n_train < 400`` to prevent
    micro-leaves that memorise individual samples:

    +-------------+----------------------+
    | n_train     | effective_min_leaf   |
    +=============+======================+
    | < 200       | max(5, n//20)        |
    | 200 – 399   | max(3, n//30)        |
    | ≥ 400       | qrf_min_samples_leaf |
    +-------------+----------------------+

    The auto-scaling fires only when this field retains its default value
    (3); explicit non-default values are always honoured.
    """

    # ── Target transformation (Phase 5) ──────────────────────────────────
    use_target_transform: bool = True
    """Apply reversible target transformation before model training.

    When True (default):
    * ``use_saw_targets=True``  → logit transform: f(y) = log(y/(1-y)), maps
      SAW-normalized [0,1] scores to ℝ, stabilising variance near 0/1 boundaries
      critical for border-province scores.  Inverse: sigmoid (recovers [0,1]).
    * ``use_saw_targets=False`` → Yeo-Johnson transform (standardize=True) per
      column; normalises raw criterion composites toward N(0,1), improving
      Gaussian-assumption estimators (BayesianRidge, RidgeCV meta-learner).

    Predictions and prediction intervals are inverse-transformed back to
    original space at the end of Stage 5 before any evaluation or reporting.

    Set False to disable transformation entirely (debugging / ablation studies).
    """

    # ── Forecast target level ─────────────────────────────────────────────
    forecast_level: str = "subcriteria"
    """Forecast target granularity.
    'subcriteria' — predict all 28 raw sub-criterion values (SC11–SC83, excluding SC52). [DEFAULT]
    'criteria'    — predict 8 aggregated criterion composites (C01–C08). [DEPRECATED - see note below]
    
    DEPRECATION NOTICE (v2025.01):
    'criteria' mode is deprecated. Sub-criteria forecasting is required for
    end-to-end integrity. Forecasting at the sub-criteria level, then aggregating
    via CRITIC weighting, preserves information density and ensures consistency
    with the MCDM weighting and ranking phases.
    """

    # ── SAW target normalization & true holdout ───────────────────────────
    use_saw_targets: bool = False
    """Predict per-year SAW-normalized [0,1] criteria scores instead of raw
    composite means.

    DEPRECATED for sub-criteria forecasting: SAW normalization was designed
    for criteria-level targets bounded [0, 1]. Sub-criteria have different
    scale distributions and should use raw values.

    When True (legacy, criteria mode only):
    * Targets are per-year column-wise minmax-normalized over the full
      provincial cross-section (same formula as SAWCalculator._normalize
      with normalization='minmax', benefit criteria only).
    * Each year's cross-section is normalized independently, removing
      cross-year level shifts while preserving within-year ordinal structure.
    * After prediction, CRITICWeightCalculator is applied to the predicted
      cross-section to derive a single composite score per province.

    Rationale: raw criteria composites conflate level with temporal trend
    and are biased by the year-specific CRITIC weighting structure used
    during data preparation. SAW-normalized targets lie in [0, 1], are
    scale-invariant, and produce predictions that can be directly compared
    across forecast years.
    """

    holdout_year: Optional[int] = None
    """Year reserved as a true out-of-sample holdout for evaluation.

    When None (default), auto-set at runtime to ``max(training_years)``
    (i.e. the most recent year of available data before the forecast target
    year — typically 2024 when predicting 2025).

    Training samples whose *target* year equals ``holdout_year`` are withheld
    from model fitting and stored as ``UnifiedForecaster.X_holdout_`` and
    ``.y_holdout_`` for Phase 6 evaluation (true holdout performance vs. all
    base models + ensemble).

    Set to 0 or a year outside the data range to disable the holdout split
    (training uses all available consecutive year-pairs).
    """

    # ── Pipeline mode (Phase 8 — Pipeline Decoupling) ─────────────────────
    pipeline_mode: Literal['full', 'features_only', 'fit_only', 'evaluate_only'] = 'full'
    """Controls how many stages ``fit_predict()`` executes before returning.

    ``'full'`` (default)
        All 7 stages.  Returns a complete ``UnifiedForecastResult``.

    ``'features_only'``
        Stages 1–2 only (feature engineering + dimensionality reduction),
        then returns ``None``.  Enables rapid feature inspection without any
        model fitting; use ``forecaster.X_train_``, ``forecaster.y_train_``,
        ``forecaster.X_pred_`` and ``forecaster.get_stage_outputs()`` to
        audit the engineered feature matrix before committing to training.

    ``'fit_only'``
        Stages 1–4 (feature engineering → dimensionality reduction →
        base-model training → ensemble predictions), then returns ``None``.
        Omits interval estimation (Stage 5) and evaluation (Stage 6–7).
        Combine with a subsequent ``'evaluate_only'`` call to decouple
        training from evaluation.

    ``'evaluate_only'``
        Stages 5–7 only.  Requires that Stages 1–4 are already complete
        (i.e. a previous ``fit_predict()`` with mode ``'full'`` or
        ``'fit_only'`` has been run on this forecaster).  Useful for
        re-running interval estimation or evaluation with different
        configuration settings without re-fitting the ensemble.
    """

    # ── Phase 3 — SOTA modules (E-05, E-06, E-08, E-10) ──────────────────
    use_panel_mice: bool = False
    """Enable three-phase PanelSequentialMICE imputation of missing features
    before dimensionality reduction (E-05).

    When True, ``stage2_reduce_features()`` applies
    ``PanelSequentialMICE.fit_transform()`` to the raw training feature matrix
    before the PLS and threshold-only reducers.  The fitted imputer is then
    applied via ``transform()`` to the prediction and holdout matrices.

    Default False: uses the existing 0-fill strategy (faster; appropriate
    when structured missingness is low).  Set True for panels with >5% NaN
    to improve feature quality flowing into the ensemble.
    """

    use_shift_detection: bool = False
    """Enable MMD²-based covariate shift detection and importance-weighted CV
    (E-08).

    When True, ``stage3_fit_base_models()`` creates a
    ``PanelCovariateShiftDetector`` and passes it to ``SuperLearner.fit()``
    as ``shift_detector``.  For each CV fold the detector computes the
    MMD² between the training and validation cross-sections; when a
    statistically significant shift is detected (p < ``alpha=0.05``), a
    logistic-regression density ratio re-weights the training rows so the
    base-model CV fits are biased toward the validation distribution.

    Default False: standard uniform-weight CV (faster; no shift assumption).
    Set True for panels with suspected structural break between training
    and recent years.
    """

    use_data_augmentation: bool = False
    """Enable Gaussian-copula + VAR(1) synthetic data augmentation (E-06).

    When True, ``stage2b_augment_data()`` is called between stage2 and
    stage3.  ``ConditionalPanelAugmenter.fit_augment_if_beneficial()``
    generates synthetic entity-year rows via per-entity VAR(1) dynamics
    seeded by a Gaussian copula capturing cross-entity correlation.
    Augmented data is committed only when 5-fold walk-forward CV shows
    ΔR² > ``augment_gain_threshold`` (default 0.005).

    Default False: no augmentation (faster; avoids risk of distributional
    mismatch from synthetic data).  Set True when n < 500 and the ensemble
    shows high variance.
    """

    augment_gain_threshold: float = 0.005
    """Minimum 5-fold walk-forward ΔR² required to commit augmented data
    (E-06 gate).  Lower values commit augmentation more aggressively;
    set to 0.0 to always commit, or >1.0 to always skip.

    Only used when ``use_data_augmentation=True``.
    """

    use_incremental_update: bool = False
    """Enable ``IncrementalEnsembleUpdater`` for model continuation when new
    data becomes available (E-10).

    When True and ``stage3b_incremental_update()`` is called with
    ``(X_new, y_new)``, each base model is updated using its most
    efficient strategy (CatBoost gradient continuation, full retrain for
    non-CatBoost members) and the meta-weights are
    re-calibrated via γ-blending with the new data.

    Default False: the standard workflow re-runs the full pipeline when
    new data arrives.  Set True for 2024→2025 update scenarios where
    re-fitting all models from scratch is prohibitively slow.
    """

    incremental_update_strategy: str = "auto"
    """Strategy for ``IncrementalEnsembleUpdater`` (E-10).

    ``'auto'`` — use the most efficient per-model strategy (CatBoost
        continuation with full retrain fallback for non-CatBoost models).
    ``'full_retrain'`` — always retrain every base model from scratch on the
        combined (historical + new) data.
    """

    incremental_update_gamma: float = 0.3
    """Meta-weight blending factor γ for ``IncrementalEnsembleUpdater``
    (E-10).  ``w_final = (1-γ)·w_prev + γ·w_new_calib``.

    Smaller γ → more stable, historical-data-weighted meta-weights.
    Larger γ → faster adaptation to new-data calibration.
    Default 0.3 preserves historical stability while incorporating
    new-year information.

    Only used when ``use_incremental_update=True``.
    """

    # ── Model toggles (T-02) ─────────────────────────────────────────────

    # ── Kernel Ridge Regression hyperparameters (T-03a) ──────────────────
    krr_alpha: float = 1.0
    """Tikhonov regularisation strength for KernelRidgeForecaster.

    Corresponds to ``sklearn.kernel_ridge.KernelRidge(alpha=.)``.
    Larger values increase bias and reduce variance.
    """

    krr_gamma: str = "scale"
    """RBF kernel bandwidth for KernelRidgeForecaster.

    ``'scale'`` → γ = 1 / (n_features × Var[X]), adapts to data scale.
    ``'auto'``  → γ = 1 / n_features.
    Or pass a float string for a fixed bandwidth.
    """

    # ── SVR hyperparameters (T-03b) ───────────────────────────────────────
    svr_C: float = 1.0
    """Regularisation parameter C for SVRForecaster.

    Larger C → lower bias, higher variance (tighter fit, more support
    vectors); smaller C → higher bias, lower variance.
    """

    svr_epsilon: float = 0.1
    """ε-tube half-width for SVRForecaster.

    Training samples with residual |yᵢ − ŷᵢ| ≤ ε contribute zero loss;
    only boundary / violating samples become support vectors.
    """

    svr_gamma: str = "scale"
    """RBF kernel bandwidth for SVRForecaster.

    Shares the same semantics as ``krr_gamma``.
    """


# =========================================================================
# Validation & Sensitivity
# =========================================================================

@dataclass
class ValidationConfig:
    """Sensitivity analysis and robustness testing."""
    n_simulations: int = 1000
    weight_perturbation: float = 0.1
    n_sensitivity_scenarios: int = 100
    rank_correlation_threshold: float = 0.85


# =========================================================================
# Ranking Configuration
# =========================================================================

@dataclass
class RankingConfig:
    """
    Hierarchical ranking configuration.

    use_evidential_reasoning
        **Disabled by default** (= False).  When True, the ranking pipeline
        uses the full two-stage Evidential Reasoning aggregation (Yang & Xu,
        2002) with belief distributions and utility intervals.  When False
        (current default), ER is skipped entirely — the pipeline runs the
        6 MCDM methods per criterion (TOPSIS, VIKOR, PROMETHEE II, COPRAS,
        EDAS, SAW) and stores their scores/ranks as separate independent
        outputs in criterion_method_scores.  No ER fusion is applied.

    run_all_years
        When True, the ranking pipeline is executed for every year in the
        panel (not just the target year) to enable temporal charts.
        Execution is parallelised via ThreadPoolExecutor.

    max_parallel_years
        Maximum number of year-ranking threads to run concurrently.
        Defaults to min(n_years, cpu_count).  Set to 1 to force sequential
        execution (debugging / memory-constrained environments).

    expose_mc_province_stats
        When True, per-province Monte Carlo rank distribution statistics
        (mean_rank, std_rank, prob_top1, prob_topK) are carried through
        the details dict so CSV writers can persist them.
    """
    use_evidential_reasoning: bool = False
    run_all_years: bool = True
    max_parallel_years: Optional[int] = None   # None → auto (cpu_count)
    expose_mc_province_stats: bool = True


# =========================================================================
# Visualisation
# =========================================================================

@dataclass
class VisualizationConfig:
    """Figure appearance defaults."""
    figsize: tuple = field(default_factory=lambda: (12, 8))
    dpi: int = 300
    style: str = "seaborn-v0_8-whitegrid"
    palette: str = "viridis"
    heatmap_cmap: str = "RdYlGn"
    diverging_cmap: str = "RdBu_r"
    save_formats: List[str] = field(default_factory=lambda: ["png"])
    # Ranking chart options
    ranking_top_n: int = 20      # how many provinces to show in top-N ranking charts
    ranking_top_n_options: List[int] = field(
        default_factory=lambda: [5, 10, 15, 20]
    )   # additional cut-offs to generate separate charts for


# =========================================================================
# Master Configuration
# =========================================================================

@dataclass
class Config:
    """Master configuration composing every sub-config."""
    paths: PathConfig = field(default_factory=PathConfig)
    panel: PanelDataConfig = field(default_factory=PanelDataConfig)
    random: RandomConfig = field(default_factory=RandomConfig)

    # MCDM
    topsis: TOPSISConfig = field(default_factory=TOPSISConfig)
    vikor: VIKORConfig = field(default_factory=VIKORConfig)

    # Weighting
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    er: EvidentialReasoningConfig = field(default_factory=EvidentialReasoningConfig)

    # Ranking
    ranking: RankingConfig = field(default_factory=RankingConfig)

    # ML Forecasting
    forecast: ForecastConfig = field(default_factory=ForecastConfig)

    # Analysis
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self):
        self.paths.ensure_directories()

    # --- convenience properties ---

    @property
    def output_dir(self) -> str:
        return str(self.paths.output_dir)

    # --- serialisation ---

    def to_dict(self) -> Dict:
        def _cvt(obj):
            if hasattr(obj, '__dataclass_fields__'):
                d = {k: _cvt(v) for k, v in obj.__dict__.items()}
                # Include computed properties that are not stored in __dict__
                if isinstance(obj, PanelDataConfig):
                    d['subcriteria_cols'] = obj.subcriteria_cols
                    d['criteria_cols'] = obj.criteria_cols
                    d['train_years'] = obj.train_years
                    d['test_year'] = obj.test_year
                    d['n_observations'] = obj.n_observations
                return d
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, (list, tuple)):
                return [_cvt(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _cvt(v) for k, v in obj.items()}
            return obj
        return _cvt(self)

    def save(self, filepath: Path) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        return (
            f"\n{'='*72}\n"
            f"  ML-MCDM Configuration Summary\n"
            f"{'='*72}\n\n"
            f"  PANEL DATA\n"
            f"    Provinces       : {self.panel.n_provinces}\n"
            f"    Subcriteria     : {self.panel.n_subcriteria}  (SC11-SC83)\n"
            f"    Criteria        : {self.panel.n_criteria}  (C01-C08)\n"
            f"    Years           : {self.panel.years[0]}-{self.panel.years[-1]}"
            f"  ({self.panel.n_years} years)\n"
            f"    Observations    : {self.panel.n_observations}\n\n"
            f"  MCDM METHODS\n"
            f"    TOPSIS norm     : {self.topsis.normalization.value}\n"
            f"    VIKOR v         : {self.vikor.v}\n\n"
            f"  WEIGHTING\n"
            f"    Strategy        : Deterministic CRITIC\n"
            f"    Stability thr   : {self.weighting.stability_threshold}\n\n"
            f"  RANKING\n"
            f"    Run all years   : {self.ranking.run_all_years}\n"
            f"    Max workers     : {self.ranking.max_parallel_years or 'auto'}\n\n"
            f"  EVIDENTIAL REASONING\n"
            f"    Base methods    : {len(self.er.base_methods)}\n"
            f"    Method weights  : {self.er.method_weight_scheme}\n"
            f"    Grades          : {self.er.n_grades}\n\n"
            f"  VALIDATION\n"
            f"    Sensitivity sim : {self.validation.n_simulations}\n"
            f"    Weight perturb  : {self.validation.weight_perturbation}\n"
            f"{'='*72}\n"
        )


# =========================================================================
# Global Config Singleton
# =========================================================================

_config: Optional[Config] = None


def get_config() -> Config:
    """Return global config (create default on first call)."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def get_default_config() -> Config:
    """Return a *fresh* default Config instance."""
    return Config()


def set_config(config: Config) -> None:
    """Replace the global config singleton."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset to a fresh default Config."""
    global _config
    _config = Config()
