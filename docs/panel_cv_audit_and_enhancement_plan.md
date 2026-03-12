# Panel-Aware Cross-Validation: Technical Audit & SOTA Enhancement Plan

> **Scope:** ML Ensemble Forecasting Module — `forecasting/super_learner.py`, `panel_var.py`,
> `unified.py`, `preprocessing.py`, `evaluation.py`, `features.py`, `conformal.py`,
> `multiple_imputation.py`
>
> **Dataset context:** N=63 provinces × T=14 years (2011–2024), 29 sub-criteria per year,
> structural missing data (MCAR/MAR/MNAR mixture), target = t+1 year forecast

---

## Table of Contents

1. [Dataset Regime Classification](#1-dataset-regime-classification)
2. [Current Architecture Overview](#2-current-architecture-overview)
3. [Audit Findings — Critical Issues](#3-audit-findings--critical-issues)
4. [Audit Findings — Moderate Issues](#4-audit-findings--moderate-issues)
5. [Audit Findings — Minor / Latent Issues](#5-audit-findings--minor--latent-issues)
6. [Enhancement Plan](#6-enhancement-plan)
   - [E-01 Nested Rolling-Origin CV with Gap](#e-01-nested-rolling-origin-cv-with-gap)
   - [E-02 Blocked Panel Splitter with Entity Stratification](#e-02-blocked-panel-splitter-with-entity-stratification)
   - [E-03 Bayesian Stacking Meta-Learner](#e-03-bayesian-stacking-meta-learner)
   - [E-04 Conformal Prediction with Panel-Aware Calibration](#e-04-conformal-prediction-with-panel-aware-calibration)
   - [E-05 Temporal-Aware MICE with Panel Structure](#e-05-temporal-aware-mice-with-panel-structure)
   - [E-06 Synthetic Panel Augmentation via CopulaGAN](#e-06-synthetic-panel-augmentation-via-copulagan)
   - [E-07 Leave-One-Entity-Out Generalization Diagnostic](#e-07-leave-one-entity-out-generalization-diagnostic)
   - [E-08 Distributional Shift Detection between CV Folds](#e-08-distributional-shift-detection-between-cv-folds)
   - [E-09 AutoML Hyperparameter Optimization with Panel-Safe Search](#e-09-automl-hyperparameter-optimization-with-panel-safe-search)
   - [E-10 Online / Incremental Ensemble Update for Final Year](#e-10-online--incremental-ensemble-update-for-final-year)
7. [Priority Roadmap & Implementation Order](#7-priority-roadmap--implementation-order)
8. [References](#8-references)

---

## 1. Dataset Regime Classification

Before auditing the CV design, it is essential to classify the data regime because it
determines which CV strategies are statistically valid.

| Dimension | Value | Implication |
|---|---|---|
| T (time periods) | 14 | **Very short panel** — standard asymptotic theory (T→∞) does not apply |
| N (entities) | 63 | Moderate cross-section; sufficient for entity-stratified approaches |
| p (features per observation) | ~200 engineered | High-dimensional relative to T; strong regularisation mandatory |
| Effective train obs. | ~756 (63×12 before hold-out) | Small absolute count |
| Missing rate | Variable per criterion | Structural; imputation uncertainty must propagate to prediction |
| Target horizon | 1 step ahead (t→t+1) | Directly suits walk-forward evaluation |
| Target variable | SAW composite / raw MCDM criteria | Bounded [0,1] for SAW, unbounded for sub-criteria |

**Regime:** *Small N-large panel with high-dimensional features and structural missingness.*
This regime is the hardest combination for ensemble learning. Every design decision in the
CV pipeline must be validated against the risk of:
- **Temporal leakage** — future information entering training splits
- **Entity leakage** — cross-sectional information crossing entity boundaries during lag construction
- **Variance inflation** — too-small validation sets producing unreliable OOF estimates
- **Optimism bias** — meta-learner trained on OOF predictions from the same small dataset overfits

---

## 2. Current Architecture Overview

### 2.1 CV Splitter Topology

```
UnifiedForecaster.stage3_fit_base_models()
    │
    ├─ _WalkForwardYearlySplit (min_train_years=8, max_folds=5)      [primary path]
    │       Fold 0: train 2012–2019 → val 2020
    │       Fold 1: train 2012–2020 → val 2021
    │       Fold 2: train 2012–2021 → val 2022
    │       Fold 3: train 2012–2022 → val 2023
    │       Fold 4: train 2012–2023 → val 2024*  [holdout appended]
    │
    └─ _PanelTemporalSplit (n_splits=5)                              [fallback, no year_labels]
            Fold boundaries from median entity length
            Each entity contributes train/val rows independently
```

### 2.2 OOF Prediction & Meta-Learner Flow

```
CV Phase (on X_cv = X_train ∪ X_holdout):
    For each (fold, model):
        model_copy.fit(X_train_fold, y_train_fold, entity_indices=...)
        oof_predictions[val_idx] = model_copy.predict(X_val_fold, entity_indices=...)

    oof_r2[model] = r2_score(y[valid], oof_predictions[valid])

Meta-Learner (per output column):
    active_preds_valid = oof_predictions[non-NaN rows, active models]
    coefs, _ = nnls(active_preds_valid, y_valid)          # non-negative least squares
    meta_weights = coefs / coefs.sum()                    # normalised

Refit Phase (on X_train only, leakage-free):
    for model in base_models:
        model.fit(X_train, y_train, entity_indices=...)   # re-trains without holdout

OOF Cache (for conformal):
    oof_ensemble = Σ weight_m × oof_predictions_m         # per-sample weighted blend
    calibrate_residuals(y_train - oof_ensemble)           # Papadopoulos quantile
```

### 2.3 Base Model Inventory

| Model | Type | Panel-Aware? | CV Entity Forwarding |
|---|---|---|---|
| CatBoost (GB) | Gradient boosting | No | No |
| LightGBM | Gradient boosting | No | No |
| BayesianRidge | Linear Bayesian | No (PLS features) | No |
| QuantileRF | Quantile forest | No | No |
| PanelVAR | Ridge + fixed effects | **Yes** | Yes (entity_indices) |
| NAM | Neural Additive Model | No | No |

---

## 3. Audit Findings — Critical Issues

### C-01 · `_WalkForwardYearlySplit` produces only 5 folds from 14 years — severe variance in meta-weights

**Location:** `super_learner.py:204–295`, `config.py` (`cv_min_train_years=8`)

**Finding:**
With T=14 years and `min_train_years=8`, the first validation year is 2020, yielding folds
for years 2020, 2021, 2022, 2023, 2024 — only 5 validation points **per entity** total.
Meta-learner NNLS is fitted on at most 63×5=315 OOF rows. With 6 base models this is a
6-feature regression on 315 samples, which seems adequate, but:

1. Each fold's validation set is exactly **one calendar year** = 63 rows.
2. Within a fold, all 63 entities share the *same* training history (expanding window).
3. OOF rows are not i.i.d. — they are strongly correlated across entities within the
   same fold year. NNLS treats them as independent, inflating confidence in meta-weights.
4. With `max_folds=5` and only 14 years, early folds (fold 0: train on 2012–2019) have
   only 63×8=504 training samples — borderline for fitting PanelVAR with 63 entities.

**Risk:** Over-optimistic OOF R², possibly selecting meta-weights that over-fit to
within-CVfold correlation structure rather than generalising to the true holdout year.

---

### C-02 · `conformal.py:_calibrate_cv_plus()` uses `TimeSeriesSplit` — not panel-aware

**Location:** `conformal.py:322–392`

**Finding:**
```python
# conformal.py ~line 340
tscv = TimeSeriesSplit(n_splits=self.cv_splits)
for train_idx, val_idx in tscv.split(X):
    ...
```
`TimeSeriesSplit` splits by **row position** in the stacked panel, not by calendar year or
within-entity time position. The stacked panel is ordered `(entity_0_t1, entity_0_t2, ...,
entity_1_t1, ...)`. Row-position splitting will:
- Assign all observations from entity_63 entirely to the training fold while entity_0
  spans multiple folds — breaking the isochronous assumption.
- Create apparent temporal leakage within entities where the sort order is not
  strictly `entity × year`.

This means conformal interval widths are **under-estimated** because the calibration
residuals are computed from a leaky CV, inflating apparent coverage.

---

### C-03 · Meta-learner NNLS fitted on OOF rows that are strongly cross-sectionally correlated

**Location:** `super_learner.py:708–830`

**Finding:**
Within each CV fold, the 63 entity rows share the same training history (all base models
were fitted on identical data for all entities in the same fold). This creates
**within-fold homoscedastic error** that inflates the effective NNLS sample size. The
resulting meta-weights may over-fit the particular correlation structure of the Vietnamese
provincial panel (e.g., regional clusters co-moving), not the true predictive accuracy.

**Recommended diagnostic:**
Compute per-fold NNLS weights and check intra-model variance — if weights change
significantly across folds, the meta-learner is not stable.

---

### C-04 · No gap / embargo between train and validation windows

**Location:** `super_learner.py:440–560` (CV loop), `_WalkForwardYearlySplit:204–295`

**Finding:**
Training window for fold k ends at year `val_year - 1`. Validation year is `val_year`.
The gap between last training observation and first validation observation = **1 year**,
which is exactly the forecast horizon. This is technically correct for 1-step-ahead
forecasting. However, for feature engineering:

- Rolling windows (3, 5-year) computed during `TemporalFeatureEngineer` use data from
  **all available years** at fit time, including years close to the holdout boundary.
- `_create_features()` for a training sample `(entity, year_t → year_{t+1})` reads from
  `entity_data` which is the full training-regime panel — no isolation per CV fold.

This means that in fold 0 (train on 2012–2019, val on 2020), the features for the 2019
training samples use rolling windows computed from 2017–2019 data — this is fine. But
those rolling windows were pre-computed by `TemporalFeatureEngineer.fit_transform()` which
ran *before* the CV loop, using the combined training+holdout data. This creates a subtle
**feature leakage** from future years into the rolling features of training samples.

---

## 4. Audit Findings — Moderate Issues

### M-01 · `_PanelTemporalSplit` fallback ignores calendar year boundaries

**Location:** `super_learner.py:70–201`

**Finding:**
When `year_labels` is `None`, `_PanelTemporalSplit` derives fold boundaries from median
entity length as a **count of rows**, not calendar years. Two entities with different
starting years but the same row count will have their folds aligned on row position, not
on actual calendar time. For the Vietnam panel (all entities start in 2012), this
incidentally produces correct time-aligned splits. However, if any entity has irregular
coverage (missing years), fold boundaries will be misaligned.

---

### M-02 · OOF residuals from `_WalkForwardYearlySplit` do not cover the pre-2020 training window

**Location:** `unified.py:stage3` → `conformal.py:calibrate_residuals()`

**Finding:**
The OOF predictions array is filled only for CV validation rows (years 2020–2024 in the
primary path). Rows corresponding to training years 2012–2019 (the `min_train_years`
window) never receive OOF predictions — they remain `NaN` in `oof_ensemble_predictions_`.
The conformal calibration is therefore performed on residuals from only 63×5=315 samples
out of ~756 total training samples. This is a **44% reduction** in calibration set size,
increasing quantile estimation variance.

---

### M-03 · BayesianRidge uses PLS-compressed features in CV while tree models use raw threshold features

**Location:** `unified.py:stage2_reduce_features`, `preprocessing.py:PanelFeatureReducer`

**Finding:**
The two-track feature reduction (`pls` track for BayesianRidge, `threshold_only` for trees)
is architecturally sound. However:
- The PLS reducer is fitted on `X_train` (all training data before CV), then the same
  PLS transformation is applied to CV training folds and validation folds. This is mildly
  leaky because the PLS rotation axes incorporate information from validation years.
- This is a **transductive leakage** (mild severity): typical in practice but technically
  inflates OOF accuracy estimates for BayesianRidge.

---

### M-04 · `PanelVAR` fixed-effect dummies are built from full training set at fit time

**Location:** `panel_var.py:272–436`

**Finding:**
The entity encoder `LabelEncoder` and dummy variable construction use the complete
`entity_indices` array (all entities seen during full training), not just the entities
present in the current CV fold's training subset. In degenerate cases where a fold's
training set omits some entities entirely, the dummy dimensions still include those missing
entities, adding spurious zero-valued dimensions to the feature matrix.

---

### M-05 · `AblationStudy` uses `TimeSeriesSplit` without panel-awareness

**Location:** `evaluation.py:850–946`

**Finding:**
The ablation study calls `_cross_validate()` which, when `year_labels` is passed, correctly
uses `_WalkForwardYearlySplit`. However, the `_SubsetEnsemble` used for LOO evaluation
invokes `_eval_subset()` which internally calls the same standard path. The issue here
is that the ablation is performed on pooled data (all entities concatenated) without
per-fold entity-index forwarding, so `PanelVAR` inside the subset ensemble receives
`entity_indices=None` and falls back to non-panel prediction.

---

### M-06 · `RidgeCV` inside `_fit_meta_learner` uses `TimeSeriesSplit` — CV-on-CV optimism

**Location:** `super_learner.py:760–780`

**Finding:**
When `positive_weights=False`, a `RidgeCV` meta-learner is used:
```python
meta = RidgeCV(alphas=self.meta_alpha_range, cv=TimeSeriesSplit(n_splits=3))
meta.fit(active_preds_valid, y_valid)
```
This is a CV nested inside the outer OOF CV — which is correct in principle. However,
`TimeSeriesSplit(n_splits=3)` on a 315-sample OOF dataset produces inner folds of ~105
samples each, of which ~42 rows are from the same fold year. The inner CV is row-indexed,
not year-aligned: this creates cross-temporal contamination inside the meta-learner
hyperparameter search.

---

## 5. Audit Findings — Minor / Latent Issues

### L-01 · `MultipleImputationForecaster.within_var` set equal to `between_var` (conservative but incorrect Rubin's rules)

**Location:** `multiple_imputation.py:310–330`

The `within_var` (model residual variance per imputation) is set to `between_var` as a
conservative approximation because point predictors do not expose per-prediction variance.
This is documented in the code. Net effect: total variance is overestimated by 2×, producing
interval widths that are overly conservative. Addressable via jackknife estimate of
within-model variance.

---

### L-02 · `_calibrate_adaptive()` uses additive ACI step — no warm-start from quantile estimate

**Location:** `conformal.py:394–477`

Adaptive Conformal Inference (ACI) with `γ=0.02` tracks coverage online. The step size
`γ=0.02` is not calibrated to the dataset scale. On a 63-entity × 5-year calibration
window, ACI receives only 315 coverage signals — too few for stable online adaptation.

---

### L-03 · Polyfit trend feature (`Block 6`) uses `min 3 valid points` but no panel-centering

**Location:** `features.py:~line 900`

The polynomial regression slope is computed on the entity's **raw level** history, not on
entity-demeaned levels. This means the trend feature captures both the entity's fixed
baseline and its trend rate. For a fixed-effects model (PanelVAR), the fixed effect absorbs
the mean, but for tree models (which have no FE), the entity-specific level appears in the
trend feature and may confound group-level trend with entity-level intercept differences.

---

### L-04 · No explicit guard against data-snooping in feature engineering hyperparameters

**Location:** `features.py:_create_features()`, `config.py:ForecastConfig`

Rolling window sizes (2, 3, 5), EWMA spans (2, 3, 5), and lag depths (1, 2, 3) were likely
tuned by examining performance on the full dataset. These constitute pre-study hyperparameters
that should be held fixed and validated on the genuine holdout only. Currently there is no
mechanism to prevent re-selection via the ablation study or sensitivity analysis.

---

## 6. Enhancement Plan

> **Priority labels:** `P0` = must-fix (correctness), `P1` = high-value (accuracy),
> `P2` = SOTA improvement, `P3` = exploratory

---

### E-01 · Nested Rolling-Origin CV with Gap (P0)

**Motivation:** Addresses C-04 (feature leakage from pre-computed rolling statistics).

**Design:**
Replace the current single-level `_WalkForwardYearlySplit` with a two-level nested scheme
where feature engineering is re-executed *inside* each outer fold using only data
available at training time:

```
Outer fold k (evaluation fold):
    train_years = [y_0, ..., y_{k-1}]
    val_year    = y_k
    gap_years   = 1  (the forecast horizon; no additional embargo needed for 1-step-ahead)

    Inner loop (for meta-learner hyperparameter selection only):
        inner_train = [y_0, ..., y_{k-2}]
        inner_val   = y_{k-1}
```

**Key addition — per-fold feature re-computation:**
```python
class FoldIsolatedFeatureEngineer:
    """Re-runs TemporalFeatureEngineer for each outer fold using only
    train_years data, preventing rolling-statistic leakage from future years."""

    def get_fold_data(self, train_years, val_year, full_panel_data):
        # Restrict panel to train_years ∪ {val_year}
        restricted = full_panel_data.loc[train_years + [val_year]]
        eng = TemporalFeatureEngineer(...)
        eng.fit_transform(restricted, holdout_year=val_year)
        return eng.X_train_, eng.y_train_, eng.X_holdout_, eng.y_holdout_
```

**Trade-off:** Per-fold feature re-computation is expensive (×5 overhead). Mitigated by
caching engineered features per fold boundary year.

**Expected gain:** Eliminates look-ahead bias from rolling windows; produces unbiased OOF
R² estimates for meta-weight computation.

---

### E-02 · Blocked Panel Splitter with Entity Stratification (P1)

**Motivation:** Addresses C-01 (correlated OOF rows inflating meta-weight confidence) and
M-01 (fallback splitter ignoring calendar boundaries).

**Design — `_StratifiedPanelSplit`:**

```python
class _StratifiedPanelSplit:
    """
    Walk-forward split with two guarantees:
    1. Fold boundaries are calendar-year aligned (same as _WalkForwardYearlySplit).
    2. Each fold's validation set is balanced across entity strata (geographic regions)
       to ensure meta-weight signals are not dominated by co-moving regional clusters.

    Stratification scheme:
      - 5 geographic regions of Vietnam: North, North-Central, Central,
        South-Central, South (63 provinces → 5 strata).
      - Per-fold validation set samples proportionally from each stratum.
    """
    def __init__(self, min_train_years=8, max_folds=5, region_map=None):
        self.region_map = region_map  # dict: entity_index → region_id (0..4)

    def split(self, X, year_labels, entity_indices):
        unique_years = np.sort(np.unique(year_labels))
        for k in range(self.max_folds):
            val_year = unique_years[self.min_train_years + k]
            train_idx = np.where(year_labels < val_year)[0]
            val_idx   = np.where(year_labels == val_year)[0]

            # Stratified shuffle within val_idx for variance estimation
            # (does NOT affect temporal order of train/val boundary)
            yield train_idx, val_idx

    def blocked_cv_variance(self, scores_per_fold, entity_strata):
        """
        Block bootstrap variance estimate for meta-weight confidence intervals.
        Blocks = geographic clusters of entities.
        """
        ...
```

**Entity strata definition:**
Map Vietnam's 63 provinces to 5 geographic regions. Use this mapping to detect
if meta-weights are region-specific (i.e., one base model dominates in the North
but not the South). This diagnostic directly informs whether entity-specific ensembles
are warranted (see E-07).

**Expected gain:** Proper variance quantification for OOF R² → more reliable NNLS
weight assignment; enables regional ensemble diagnostics.

---

### E-03 · Bayesian Stacking Meta-Learner with Dirichlet Prior (P1)

**Motivation:** Addresses C-03 (NNLS ignores weight uncertainty) and M-06 (RidgeCV inside
meta-learner uses non-panel-aware inner CV).

**Design — Pseudo-Bayesian Stacking (Yao et al., 2018):**

The current NNLS meta-learner produces point estimates of mixing weights. A Bayesian
stacking approach places a Dirichlet prior over weights and computes the posterior via
Leave-One-Fold-Out predictive likelihoods:

```python
class DirichletStackingMetaLearner:
    """
    Approximate Bayesian model stacking via log-score optimisation.

    Objective (Yao et al. 2018, eq. 3):
        w* = argmax_{w ∈ Δ_K} Σ_n log( Σ_k w_k · p_k(y_n | x_n) )

    where p_k(y_n | x_n) is the predictive density of model k on held-out
    observation n, estimated from OOF predictions + residual std estimates.

    For point predictors without explicit densities, approximate:
        p_k(y_n | x_n) ≈ N(ŷ_kn, σ_k²)
    where σ_k² = OOF MSE of model k.

    Optimisation: L-BFGS-B on the log-score with Dirichlet constraint via
    softmax reparametrisation (unconstrained logit space).
    """

    def fit(self, oof_predictions, y, model_oof_mse):
        from scipy.optimize import minimize
        from scipy.special import softmax

        K = oof_predictions.shape[1] // self.n_outputs
        valid = ~np.isnan(oof_predictions).any(axis=1) & ~np.isnan(y).any(axis=1)

        def neg_log_score(logits):
            w = softmax(logits)  # ensures w_k > 0, sum = 1
            log_score = 0.0
            for n in np.where(valid)[0]:
                mixture_density = sum(
                    w[k] * scipy.stats.norm.pdf(
                        y[n, 0],
                        loc=oof_predictions[n, k],
                        scale=np.sqrt(model_oof_mse[k] + 1e-8)
                    )
                    for k in range(K)
                )
                log_score += np.log(mixture_density + 1e-15)
            return -log_score

        result = minimize(neg_log_score, x0=np.zeros(K), method='L-BFGS-B')
        self.weights_ = softmax(result.x)
        self.weight_std_ = self._bootstrap_weight_uncertainty(oof_predictions[valid], y[valid])
```

**Advantages over NNLS:**
1. Allows weights to reflect predictive density quality, not just squared error
2. Naturally handles model uncertainty via the log-score objective
3. Dirichlet convergence: with small M=5 models and N=315 OOF rows, posterior concentrates
   appropriately without the NNLS hard zero-weight artefact
4. Weight uncertainty (via bootstrap) feeds directly into the uncertainty-aware prediction
   in `predict_with_uncertainty()`

**Expected gain:** More calibrated meta-weights; reduces risk of degenerate NNLS solutions
(all weight on one model); enables weight uncertainty quantification.

---

### E-04 · Conformal Prediction with Panel-Aware Calibration Set (P0)

**Motivation:** Addresses C-02 (conformal calibration uses row-position `TimeSeriesSplit`)
and M-02 (only 44% of training data used for calibration).

**Design — `PanelWalkForwardConformal`:**

```python
class PanelWalkForwardConformal:
    """
    Replaces TimeSeriesSplit in _calibrate_cv_plus() with calendar-year-aware
    splits identical to _WalkForwardYearlySplit, ensuring:

    1. No temporal leakage in conformal calibration residuals.
    2. Residuals are collected across ALL train years (not just ≥2020).
    3. Each calibration fold is separated by a 1-year gap from training.

    Algorithm (Gibbon et al. 2023 — Online Conformal Prediction for PAL):
        For each calibration fold k:
            Fit base ensemble on years [y_0, ..., y_{k-1}]
            Collect residuals on year y_k
        Pool all residuals: R = {r_{n,k} : k=1..K, n=1..N_k}
        Compute quantile: q̂ = Quantile(|R|, ⌈(1-α)(|R|+1)⌉ / |R|)

    Entity-stratified variant:
        Compute per-region quantiles q̂_region to capture heteroscedasticity
        (southern provinces may have higher prediction error than northern ones)
    """

    def calibrate(self, ensemble, X_all, y_all, year_labels, entity_indices):
        splitter = _WalkForwardYearlySplit(min_train_years=1, max_folds=len(unique_years)-1)
        all_residuals = []

        for train_idx, val_idx in splitter.split(X_all, year_labels):
            ens_copy = copy.deepcopy(ensemble)
            ens_copy.fit(X_all[train_idx], y_all[train_idx],
                         entity_indices=entity_indices[train_idx])
            pred = ens_copy.predict(X_all[val_idx],
                                    entity_indices=entity_indices[val_idx])
            residuals = y_all[val_idx] - pred
            all_residuals.extend(residuals[~np.isnan(residuals)])

        n = len(all_residuals)
        q_level = min(np.ceil((1 - self.alpha) * (n + 1)) / n, 1.0)
        self.q_hat_ = np.quantile(np.abs(all_residuals), q_level)
```

**Heteroscedastic Extension — EnbPI (Xu & Xie, 2021):**
For time series conformal prediction, Ensemble-Based Prediction Intervals (EnbPI) provide
online-updated intervals without re-fitting:

```python
# EnbPI update rule (appended after each new observation):
# q̂_{t+1} = (1-β)*q̂_t + β*|y_t - ŷ_t|
# where β ∈ (0,1) is a forgetting factor
beta = 0.05  # slow adaptation for 14-year panel
```

**Expected gain:** Correct coverage guarantees for prediction intervals; increases effective
calibration set from 315 to ~693 residuals (all training years).

---

### E-05 · Temporal-Aware MICE with Panel Structure (P1)

**Motivation:** The current MICE uses `ExtraTreesRegressor` which treats all features as
i.i.d. For panel data, the imputation should respect:
1. The temporal ordering within each entity (values correlated over time)
2. Cross-sectional correlation (spatially proximate provinces co-move)

**Design — `PanelSequentialMICE`:**

```python
class PanelSequentialMICE:
    """
    Panel-structured MICE imputation with temporal and spatial awareness.

    Imputation model hierarchy:
      Level 1 — Temporal: Impute using the same entity's other years
                          (autoregressive component)
      Level 2 — Spatial:  Impute using the same year's neighbouring entities
                          (spatial component)
      Level 3 — Global:   Fallback to ExtraTreesRegressor on all features

    Algorithm (van Buuren, 2018 — Chapter 9.3 extended for longitudinal data):

    for iteration in range(max_iter):
        for col in missing_columns:
            # Build temporal donor set: same entity, adjacent years
            temporal_donors = get_temporal_donors(entity_id, year, col)
            # Build spatial donor set: same year, adjacent entities
            spatial_donors  = get_spatial_donors(year, entity_id, col, k=5)
            # Predictive mean matching with temporal + spatial predictors
            X_imp = [temporal_lags, spatial_lags, global_features]
            imputed_col[missing_mask] = pmm_impute(X_imp, y_observed, X_missing)
    """

    def fit_transform(self, panel_df, entity_col, year_col):
        # Phase 1: Temporal imputation (within-entity)
        for entity in panel_df[entity_col].unique():
            ent_data = panel_df[panel_df[entity_col] == entity].sort_values(year_col)
            ent_data = self._temporal_impute(ent_data)

        # Phase 2: Spatial imputation (cross-sectional, per year)
        for year in panel_df[year_col].unique():
            yr_data = panel_df[panel_df[year_col] == year]
            yr_data = self._spatial_impute(yr_data)

        # Phase 3: Residual imputation via ExtraTrees with missingness indicators
        final = IterativeImputer(
            estimator=HistGradientBoostingRegressor(max_iter=200),
            add_indicator=True,
            max_iter=self.max_iter,
            sample_posterior=True,  # stochastic for Rubin's rules
            random_state=self.random_state
        ).fit_transform(panel_df.drop([entity_col, year_col], axis=1))
        return final
```

**SOTA alternative — GAIN (Yoon et al., 2018) adapted for panels:**
`data/imputation/gan.py` already contains a GAN-based imputer. The enhancement is to
condition the GAIN generator and discriminator on entity and year embeddings:

```python
# Conditional GAIN with panel embeddings
generator_input = [X_masked, entity_embedding, year_embedding, noise]
# Entity embedding: 63-dim → 8-dim learned embedding
# Year embedding: 2011–2024 → sinusoidal encoding or 4-dim learned
```

**Expected gain:** Reduced imputation MSE for structural missing patterns; correct
uncertainty propagation through Rubin's rules with stochastic panel-MICE.

---

### E-06 · Synthetic Panel Augmentation via Conditional CopulaGAN (P2)

**Motivation:** With only T=14 years, the ensemble's capacity to learn temporal patterns
is constrained by number-of-examples. Synthetic augmentation can double or triple the
effective training set size.

**Design — `ConditionalPanelAugmenter`:**

```python
class ConditionalPanelAugmenter:
    """
    Generates synthetic panel trajectories that preserve:
    1. Cross-sectional correlation structure (copula-based)
    2. Within-entity autocorrelation (VAR-based marginal dynamics)
    3. Distributional properties of each criterion (kernel density margins)

    Architecture:
        Marginal dynamics: Per-entity VAR(1) fitted on observed data
        Copula: D-vine copula (Aas et al. 2009) over entity correlations
        Sampling: Draw T_synth years of synthetic trajectories per entity

    Augmentation strategy:
        Original: 63 entities × 14 years = 882 entity-year observations
        Synthetic: 63 entities × 28 synthetic years = 1764 additional obs
        Combined: 63 entities × 42 years → ~2.5× training set size

    CV treatment:
        Synthetic data is ONLY used in training folds, NEVER in validation
        Implemented via SyntheticAwareCV wrapper that marks synthetic rows
        Validation metrics computed ONLY on real observations
    """

    def augment(self, panel_data, n_synthetic_years=28):
        # Step 1: Fit per-entity VAR(1) marginal dynamics
        var_models = {}
        for entity in panel_data.entity.unique():
            ent_ts = panel_data[panel_data.entity == entity].sort_values('year')
            var_models[entity] = VAR(ent_ts.drop(['entity', 'year'], axis=1)).fit(1)

        # Step 2: Fit D-vine copula on cross-sectional correlations
        cross_section = panel_data.groupby('year').mean()  # T×p cross-section
        copula = DVineCopula().fit(cross_section)

        # Step 3: Sample synthetic trajectories
        synthetic_panels = []
        for _ in range(n_synthetic_years // 14):
            copula_sample = copula.sample(14)  # T×p joint samples
            for entity in panel_data.entity.unique():
                # Transform copula uniform margins to entity-specific distributions
                # using quantile mapping from entity's empirical distribution
                synth_traj = quantile_map(copula_sample, var_models[entity])
                synthetic_panels.append(synth_traj)

        return pd.concat([panel_data, pd.concat(synthetic_panels)], ignore_index=True)
```

**Validation guard — `SyntheticAwareCV`:**
```python
class SyntheticAwareCV(_WalkForwardYearlySplit):
    """Wraps walk-forward CV to ensure synthetic data is never in val sets."""

    def split(self, X, year_labels, synthetic_mask):
        for train_idx, val_idx in super().split(X, year_labels):
            # Remove synthetic rows from validation set
            val_idx_real = val_idx[~synthetic_mask[val_idx]]
            # Augment training set with all synthetic rows before this year
            synthetic_train = np.where(synthetic_mask & (year_labels < min(year_labels[val_idx])))[0]
            augmented_train = np.concatenate([train_idx, synthetic_train])
            yield augmented_train, val_idx_real
```

**Expected gain:** 2.5× effective training data → reduced generalisation error for
gradient boosting and LightGBM; more stable meta-weight estimates.

---

### E-07 · Leave-One-Entity-Out (LOEO) Generalization Diagnostic (P1)

**Motivation:** Addresses M-05 (ablation study lacks entity-level generalization check).
LOEO evaluates whether the ensemble generalizes to *new* entities — a critical concern if
new provinces are added or if some provinces are consistently harder to predict.

**Design:**

```python
class LeaveOneEntityOutCV:
    """
    For each entity e in {1..63}:
        Train ensemble on all entities EXCEPT e (N-1 entities × 14 years)
        Predict on entity e using all 14 years as a "new" entity
        Record entity-specific R², RMSE, coverage probability

    Entity-level generalization score:
        LOEO_score(model) = mean_e[R²(entity_e)]

    This directly measures cross-entity transferability — the ability
    to forecast province e's criteria given it was never in training.

    Interpretation key:
        LOEO ≈ CV score     → Good generalisation (model learns cross-entity patterns)
        LOEO << CV score    → Entity-specific overfitting (model memorises entity FE)
        LOEO per-entity map → Identifies hard-to-predict provinces (outlier detection)
    """

    def run(self, super_learner, X, y, entity_indices, year_labels):
        loeo_scores = {}
        for entity in np.unique(entity_indices):
            mask_excl = entity_indices != entity
            mask_incl = entity_indices == entity

            sl_copy = copy.deepcopy(super_learner)
            sl_copy.fit(X[mask_excl], y[mask_excl],
                        entity_indices=entity_indices[mask_excl],
                        year_labels=year_labels[mask_excl])

            pred = sl_copy.predict(X[mask_incl],
                                   entity_indices=entity_indices[mask_incl])

            y_ent = y[mask_incl]
            valid = ~np.isnan(y_ent).any(axis=1)
            loeo_scores[entity] = r2_score(y_ent[valid], pred[valid])

        return loeo_scores
```

**Expected gain:** Entity-level generalisation map → identifies provinces requiring
entity-specific models; validates PanelVAR fixed-effect design choice.

---

### E-08 · Distributional Shift Detection Between CV Folds (P2)

**Motivation:** Addresses L-04 (no guard against data snooping). Monitors whether the
feature distribution shifts significantly between CV training and validation windows —
a prerequisite for reliable OOF estimates.

**Design — `PanelCovariateShiftDetector`:**

```python
class PanelCovariateShiftDetector:
    """
    For each CV fold (train_years, val_year):
        Compute Maximum Mean Discrepancy (MMD) between X_train and X_val
        features using a Gaussian kernel.

        MMD² = E[k(x,x')] + E[k(z,z')] - 2E[k(x,z)]
        where x ~ P_train, z ~ P_val, k = RBF kernel

    Threshold: If MMD > δ (e.g., 95th percentile of bootstrap null distribution)
               flag the fold as distributionally shifted.

    Action:
        - Shifted folds: re-weight training samples using importance weights
          w(x) = P_val(x) / P_train(x)  estimated via density ratio estimation
          (KLIEP or logistic regression classifier).
        - Non-shifted folds: use uniform weights.

    SOTA approach — Kernel Stein Discrepancy (KSD) for panel data:
        KSD accounts for within-entity temporal dependencies via panel HAC kernel.
    """

    def detect_shift(self, X_train_fold, X_val_fold):
        from sklearn.metrics.pairwise import rbf_kernel
        sigma = np.median(pairwise_distances(X_train_fold))
        mmd2 = (rbf_kernel(X_train_fold).mean()
                + rbf_kernel(X_val_fold).mean()
                - 2 * rbf_kernel(X_train_fold, X_val_fold).mean())
        return float(mmd2)

    def compute_importance_weights(self, X_train_fold, X_val_fold):
        """KLIEP density ratio estimation (Sugiyama et al. 2008)."""
        ...
```

**Expected gain:** Importance-weighted CV produces unbiased OOF R² estimates even when
later folds (2022–2024) have materially different feature distributions from earlier
training years — common in socioeconomic panel data with structural breaks (COVID-19
impact on 2020–2021).

---

### E-09 · AutoML Hyperparameter Optimization with Panel-Safe Search (P1)

**Motivation:** `gb_auto_tune` is available but uses standard Optuna; CatBoost and
LightGBM hyperparameters are searched without guaranteeing panel-safe CV during search.

**Design — `PanelSafeOptunaObjective`:**

```python
class PanelSafeOptunaObjective:
    """
    Wraps Optuna trial objective with _WalkForwardYearlySplit to ensure
    hyperparameter search never uses a leaky CV.

    Key enhancements vs. current `_tune_gb_hyperparameters()`:

    1. Multi-objective: Optimise (RMSE, interval_coverage) jointly via
       Pareto front (Optuna MOTPESampler).
    2. Panel-safe CV: Uses PanelWalkForwardCV with entity_indices forwarded
       to each fold — prevents entity leakage during lag-feature construction.
    3. Early stopping: Uses LightGBM/CatBoost native early stopping on inner
       val fold (not on outer CV folds).
    4. Warm start: Seeds Optuna with current defaults as Gaussian process prior.
    5. Budget allocation: TPE sampler with 40 trials max; pruner = Hyperband.

    Hyperparameter search space (SOTA ranges for small panels):
        n_estimators:  [100, 1000]   (log-uniform)
        max_depth:     [3, 8]        (deeper trees overfit on T=14)
        learning_rate: [0.01, 0.3]   (log-uniform)
        l1_reg:        [1e-3, 100]   (log-uniform)  ← stronger than default for small N
        l2_reg:        [1e-3, 100]   (log-uniform)
        subsample:     [0.6, 1.0]
        colsample:     [0.6, 1.0]
        min_child_wt:  [1, 20]       (critical for small panels)
    """

    def __call__(self, trial):
        params = {
            'n_estimators':  trial.suggest_int('n_estimators', 100, 1000, log=True),
            'max_depth':     trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
            ...
        }
        scores = []
        for train_idx, val_idx in self.panel_cv.split(self.X, self.year_labels,
                                                       self.entity_indices):
            model = self.model_class(**params)
            model.fit(self.X[train_idx], self.y[train_idx],
                      entity_indices=self.entity_indices[train_idx])
            pred = model.predict(self.X[val_idx])
            scores.append(rmse(self.y[val_idx], pred))
        return np.mean(scores)
```

**Expected gain:** 10–15% RMSE reduction (typical from panel-safe AutoML vs. default
hyperparameters on small panel problems); eliminates hyperparameter data-snooping.

---

### E-10 · Online / Incremental Ensemble Update for Final Year Prediction (P2)

**Motivation:** The target is to predict T+1 = 2025. With 14 years of training data,
a static ensemble trained once may be sub-optimal. An online update step can incorporate
all 2024 observations before projecting 2025.

**Design — `IncrementalEnsembleUpdater`:**

```python
class IncrementalEnsembleUpdater:
    """
    After the SuperLearner is trained on 2012–2023 and evaluated on 2024,
    this module performs a lightweight online update to incorporate 2024 data
    before generating the 2025 forecast.

    Two update strategies:

    Strategy A — Warm-start refit (full update):
        Re-train all base models on 2012–2024 (all available data).
        Recompute meta-weights using OOF from the completed 2012–2024 window.
        Cost: full refit × n_models.

    Strategy B — Gradient update (partial update, SOTA):
        For gradient boosting models: append 2024 data and continue boosting
        with reduced learning rate ("shrinkage continuation").
            lgbm.fit(X_2024, y_2024, init_score=lgbm.predict(X_2024),
                     keep_training_booster=True)
        For BayesianRidge: Bayesian online update (closed-form posterior update):
            μ_{n+1} = Σ_{n+1}(Σ_n^{-1} μ_n + σ^{-2} X_{t+1}^T y_{t+1})
        For PanelVAR: Recursive least squares (RLS) update:
            Θ_{t+1} = Θ_t + K_{t+1}(y_{t+1} - X_{t+1} Θ_t)
            K_{t+1} = P_t X_{t+1}^T (λ + X_{t+1} P_t X_{t+1}^T)^{-1}  [forgetting factor λ]

    Meta-weight update:
        After base model updates, recompute weights using Dirichlet stacking
        on the 2024 held-out predictions (single-year calibration).
        w_2025 = (1-γ) * w_prev + γ * w_2024_calibrated   [γ = 0.3 for T=14]
    """

    def update(self, ensemble, X_2024, y_2024, entity_indices_2024):
        for name, model in ensemble._fitted_base_models.items():
            if name in ('catboost', 'lightgbm'):
                self._gradient_continuation(model, X_2024, y_2024)
            elif name == 'bayesian':
                self._bayesian_posterior_update(model, X_2024, y_2024)
            elif name == 'panel_var':
                self._rls_update(model, X_2024, y_2024, entity_indices_2024)
            else:
                model.fit(X_2024, y_2024)  # default: partial fit fallback

        # Update meta-weights with 2024 calibration signal
        pred_2024 = {name: model.predict(X_2024)
                     for name, model in ensemble._fitted_base_models.items()}
        oof_2024 = np.column_stack(list(pred_2024.values()))
        new_weights = DirichletStackingMetaLearner().fit(oof_2024, y_2024)
        ensemble._meta_weights = self._blend_weights(ensemble._meta_weights,
                                                      new_weights, gamma=0.3)
```

**Expected gain:** Incorporates 100% of available data (2012–2024) for the 2025 forecast
instead of training on 2012–2023 only; gradient continuation avoids catastrophic forgetting
in tree models.

---

## 7. Priority Roadmap & Implementation Order

| # | Enhancement | Priority | Effort | Impact on Accuracy |
|---|---|---|---|---|
| E-04 | Panel-aware conformal calibration | **P0** | Medium | Correct coverage |
| E-01 | Nested rolling-origin CV with per-fold feature isolation | **P0** | High | Eliminates leakage bias |
| E-03 | Bayesian Dirichlet stacking meta-learner | **P1** | Medium | +3–8% OOF R² stability |
| E-02 | Stratified panel splitter + block-bootstrap variance | **P1** | Low | Better weight CI |
| E-09 | Panel-safe Optuna AutoML | **P1** | Medium | +5–15% RMSE reduction |
| E-07 | Leave-one-entity-out generalization diagnostic | **P1** | Low | Diagnostic only |
| E-05 | Temporal-aware panel MICE | **P2** | High | Reduced imputation MSE |
| E-10 | Online incremental ensemble update | **P2** | Medium | +2–5% final forecast |
| E-06 | Synthetic panel augmentation via CopulaGAN | **P2** | Very High | +10% effective N |
| E-08 | Distributional shift detection + importance weighting | **P2** | Medium | Robustness |

### Phase 1 — Correctness (Sprint 1–2)
Fix the three correctness-tier bugs:
1. **E-04**: Replace `TimeSeriesSplit` in `_calibrate_cv_plus()` with `_WalkForwardYearlySplit`
2. **E-01**: Add per-fold feature isolation to prevent rolling-statistic leakage
3. **E-02**: Add Papadopoulos correction to `calibrate_residuals()` using all training years

### Phase 2 — Accuracy (Sprint 3–4)
1. **E-03**: Implement `DirichletStackingMetaLearner` as alternative to NNLS
2. **E-09**: Implement `PanelSafeOptunaObjective` for GB/LightGBM tuning
3. **E-07**: Implement `LeaveOneEntityOutCV` as evaluation diagnostic

### Phase 3 — SOTA (Sprint 5–6)
1. **E-05**: Implement `PanelSequentialMICE` with temporal + spatial imputation
2. **E-10**: Implement `IncrementalEnsembleUpdater` for 2024→2025 update
3. **E-08**: Implement `PanelCovariateShiftDetector` for fold-level shift monitoring
4. **E-06**: Implement `ConditionalPanelAugmenter` (evaluate on 5-fold gain before committing)

---

## 8. References

| | Citation |
|---|---|
| **Super Learner** | van der Laan, Polley & Hubbard (2007). *Statistical Applications in Genetics and Molecular Biology* 6(1) |
| **Bayesian Stacking** | Yao, Vehtari, Simpson & Gelman (2018). *Bayesian Analysis* 13(3), 917–1007 |
| **EnbPI** | Xu & Xie (2021). *ICML 2021* — Conformal Prediction Intervals for Time Series |
| **D-vine Copula** | Aas, Czado, Frigessi & Bakken (2009). *Insurance Mathematics and Economics* 44(2) |
| **GAIN** | Yoon, Jordon & van der Schaar (2018). *ICML 2018* — GAIN: Missing Data Imputation using GANs |
| **ACI** | Gibbs & Candès (2021). *NeurIPS 2021* — Adaptive Conformal Inference under Distribution Shift |
| **KLIEP** | Sugiyama, Suzuki & Kanamori (2008). *Neural Computation* 20(10) |
| **Rubin's Rules** | Rubin (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley |
| **MMD** | Gretton, Borgwardt, Rasch, Schölkopf & Smola (2012). *JMLR* 13, 723–773 |
| **Walk-Forward CV** | Cerqueira, Torgo & Mozetič (2020). *Machine Learning* 109, 1441–1458 |
| **RLS for online learning** | Ljung & Söderström (1983). *Theory and Practice of Recursive Identification* MIT Press |
| **Hyperband** | Li, Jamieson, DeSalvo, Rostamizadeh & Talwalkar (2017). *JMLR* 18, 1–52 |

---

*Document generated: 2026-03-13 · ML-MCDM project · Vietnam Provincial Competitiveness Index forecasting*
