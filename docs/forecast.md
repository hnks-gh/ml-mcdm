# Forecasting Methodology: State-of-the-Art Ensemble Machine Learning

## Overview

This framework implements a **statistically-principled 3-tier ensemble learning system** optimized for small-to-medium panel data (N < 1000). It combines 5 diverse machine learning models with Super Learner meta-learning and distribution-free conformal prediction to forecast future criterion values.

**Key Design Principles:**
- **Model diversity over quantity**: 5 diverse models outperform larger correlated sets
- **Statistical appropriateness**: Optimized for N < 1000 (your dataset: ~756 training rows)
- **Automatic optimal weighting**: Super Learner learns best combination
- **Guaranteed coverage**: Conformal prediction provides 95% valid intervals
- **No redundancy**: Each model captures different patterns (tree, linear, panel, Bayesian)

**Key Features:**
- **5 Model Types**: CatBoost (joint multi-output), Bayesian linear, kernel methods, and quantile forests
- **Super Learner**: Automatic optimal weighting via meta-learning (`PanelWalkForwardCV`)
- **Conformal Prediction**: Distribution-free 95% prediction intervals
- **Distributional Forecasting**: Full predictive distributions via quantile forests
- **Enhanced Feature Engineering**: 12 feature blocks — lag, rolling, stationarity, EWMA, diversity, region dummies (Phase 1)
- **Target Transformation**: Logit/Yeo-Johnson reversible transform for improved Gaussianity (Phase 5)
- **HP Optimisation**: Optional Optuna one-time search for CatBoost (Phase 4)

---

## System Architecture

### Three-Tier Architecture

```
Input: Panel Data (N entities × p components × T years)
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIER 1: BASE MODELS (5 diverse models)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ↓
Temporal Feature Engineering (12 feature blocks — Phase 1)
  ├── Block 1:  Current component values at t
  ├── Block 2:  Lag features (t-1, t-2, t-3) + _was_missing indicators
  ├── Block 3:  Rolling statistics (mean, std, min, max) — windows {2, 3, 5}
  ├── Block 4:  Momentum (Δt) and acceleration (Δ²t)
  ├── Block 5:  Stationarity — entity-demeaned level, demeaned momentum, Δ₂
  ├── Block 6:  Polyfit trend slopes (≥ 3 valid points)
  ├── Block 7:  EWMA levels (spans 2, 3, 5)
  ├── Block 8:  Expanding window mean (long-run baseline)
  ├── Block 9:  Inter-criterion diversity (std and range across components)
  ├── Block 10: Rolling skewness (5-year window)
  ├── Block 11: Panel-relative — percentile, z-score, rank-change Δpercentile
  └── Block 12: Geographic cluster dummies (5 Vietnam regions)
  ↓
Target Transformation (Phase 5) — logit [SAW] or Yeo-Johnson [raw] → ℝ
  ↓
Base Model Training (DIVERSE MODEL TYPES)
  │
   ├── Tree-Based (1 model)
  │   ├── CatBoost Gradient Boosting (MultiRMSE joint multi-output loss)
  │
  ├── Bayesian Linear (1 model)
  │   └── Bayesian Ridge (posterior uncertainty, PLS-compressed features)
  │
  ├── Kernel Methods (2 models)
  │   ├── Kernel Ridge Regression (RBF kernel, L2 regularised)
  │   └── Support Vector Regression (ε-insensitive tube)
  │
  ├── Distributional (1 model)
  │   └── Quantile Random Forest (distributional forecasts)
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIER 2: SUPER LEARNER META-ENSEMBLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ↓
Super Learner (Automatic Optimal Weighting)
  ├── Generate out-of-fold predictions (TimeSeriesSplit)
  ├── Train meta-learner (Ridge regression)
  ├── Positive weight constraint + normalization
  └── Full model retraining on complete data
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIER 3: CONFORMAL PREDICTION CALIBRATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ↓
Conformal Prediction (distribution-free intervals)
  ├── CV+ Conformal (cross-validation calibration)
  ├── Guaranteed coverage: P(y ∈ [L, U]) ≥ 95%
  └── Adaptive to heteroscedasticity
  ↓
Calibrated Prediction Intervals
  ├── Guaranteed coverage: P(y ∈ [L, U]) ≥ 1-α
  └── Adaptive to heteroscedasticity
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output: Predictions + Calibrated Intervals + Diagnostics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ├── Point predictions (Super Learner weighted)
  ├── Calibrated 95% prediction intervals (conformal)
  ├── Distributional forecasts (quantiles from QRF)
  ├── Feature importance (aggregated across models)
  ├── Model contributions (meta-weights)
  ├── CV performance metrics
  └── Residual diagnostics
```

### Configuration

The system uses a **single optimized configuration** designed for small-to-medium panel data (N < 1000):

**Base Models (5, always-on):**
- CatBoost Gradient Boosting (oblivious trees, joint multi-output `MultiRMSE`)
- Bayesian Ridge (linear with posterior uncertainty, PLS-compressed features)
- Quantile RF (distributional forecasts, QRF quantile intervals)
- Kernel Ridge Regression (RBF kernel, L2 regularised)
- Support Vector Regression (ε-insensitive tube, RBF kernel)

**Meta-Ensemble:** Super Learner (`PanelWalkForwardCV` panel-aware CV, NNLS meta-weights)

**Calibration:** Conformal Prediction (95% coverage guarantee, OOF residuals)

**Feature Engineering:** 12 feature blocks — lag, rolling {2,3,5}, stationarity, EWMA, diversity, skewness, region dummies (Phase 1, ≈279 features)

**Target Transformation:** Logit (SAW) / Yeo-Johnson (raw) reversible transform applied before Stage 2 so PLS compression uses the transformed covariance (Phase 5)

**HP Optimisation:** One-time Optuna TPE search for CatBoost when `auto_tune_gb=True` (Phase 4)

**Rationale:** For ~756 training rows (63 provinces × ~12 usable year pairs, 2011–2024 with missingness), five complementary models with Super Learner per-output weighting provide a robust bias-variance tradeoff.

---

## Missing Data Strategy: MICE Imputation (Phase B)

### Imputation Architecture

The forecasting pipeline handles missing features through a **unified MICE (Multivariate Imputation by Chained Equations)** strategy:

```
Input Features (with NaN from insufficient history)
  ├─ Lag features: early years have lag-3 = NaN when history < 3 years
  ├─ Rolling stats: short windows have rolling-5 = NaN when history < 5 years
  ├─ Momentum: unavailable when prior year missing
  └─ Entity-demeaned: isolated entities with few observations → NaN
  
  ↓ [PanelFeatureReducer — IterativeImputer + ExtraTreesRegressor]
  
Output Features (complete, no NaN)
  ├─ All NaN filled via MICE multivariate correlation learning
  ├─ Binary _was_missing indicators appended (mark which were imputed)
  └─ Ready for dimensionality reduction and model training
```

### Why MICE?

**Phase A** used a complex 4-tier fallback hierarchy:
- Tier 1: MICE (multivariate imputation)
- Tier 2: Temporal median (per-entity rolling medians)
- Tier 3: Cross-sectional median (per-year medians)
- Tier 4: Training mean (global fallback) + 0.0 stub

**Phase B** simplified to **MICE-only** because:

✅ **Multivariate awareness**: MICE respects feature correlations. ExtraTreesRegressor learns which features predict others, unlike univariate fallbacks.

✅ **Nonlinear relationships**: Tree ensembles capture nonlinear patterns (e.g., criteria that plateaued post-2020). Medians cannot.

✅ **Panel structure**: MICE inherently uses available data for each entity. Doesn't require explicit per-block tier configuration.

✅ **Uncertainty quantification**: Via multiple imputation (M=5 stochastic imputations), Rubin's Rules pools predictions to estimate among-imputation variance.

✅ **Simplicity**: Single unified approach replaces 280 lines of tier-specific caching logic.

### MICE Configuration

Located in `data/imputation/__init__.py` (`ImputationConfig` dataclass):

```python
use_mice_imputation: bool = True
    # Enable MICE imputation in preprocessing.PanelFeatureReducer

n_imputations: int = 5
    # Number of stochastic imputations (M) for Rubin's Rules
    # M=1: single point estimate (faster, no uncertainty)
    # M=5: standard (recommended) — captures imputation uncertainty
    # M≥10: for very high missingness (>50%)

mice_max_iter: int = 20
    # IterativeImputer convergence iterations

mice_estimator: str = "extra_trees"
    # Regression model: "extra_trees" (fast, adaptive)
    #                  "random_forest" (stable)
    #                  "bayesian_ridge" (probabilistic)

mice_add_indicator: bool = True
    # Append binary _was_missing_{feature} columns
    # Allow models to learn imputation uncertainty representation
```

### Multiple Imputation & Rubin's Rules

When `n_imputations > 1` and `ForecastConfig.use_multiple_imputation=True`:

1. **Generate M imputations**: m=1,...,M, generate independent MICE imputations
2. **Train M models**: For each imputation, fit independent base model
3. **Pool predictions** via **Rubin's Rules (1987)**:

   $$\hat{\mu}_M = \frac{1}{M} \sum_{m=1}^{M} \hat{\mu}_m \quad \text{(pooled point estimate)}$$
   
   $$\text{SE}_M^2 = \overline{V}_m + \left(1 + \frac{1}{M}\right) B_m \quad \text{(total variance)}$$
   
   where:
   - $\overline{V}_m$ = average within-imputation variance
   - $B_m$ = between-imputation variance (sample variance of $\hat{\mu}_m$ across imputations)
   - $(1 + 1/M)$ adjustment reflects additional uncertainty from imputation

4. **Fraction of Missing Information (FMI)**:
   $$\text{FMI} = \frac{(1 + 1/M) B_m}{\text{SE}_M^2}$$
   
   Indicates what fraction of variance comes from imputation uncertainty vs. estimation uncertainty.

**Example:** With 30% missing data and M=5:
- FMI ≈ 0.08–0.12 (8–12% of variance from imputation)
- Prediction intervals widen accordingly

### Migration from Phase A

If you have old configs with **deprecated** phase A parameters:

```python
# OLD (Phase A, still works due to backward compatibility)
ImputationConfig(
    use_advanced_feature_imputation=True,
    block_imputation_tiers={1: "training_mean", 3: "temporal_median", ...},
    temporal_imputation_window=5
)

# NEW (Phase B, recommended)
ImputationConfig(
    use_mice_imputation=True,
    n_imputations=5,  # Enable uncertainty quantification
    mice_estimator="extra_trees"
)
```

Old configs still work (deprecated parameters ignored); new config is clearer.

---

## Part I: Model Types

### 1.1 Tree-Based Ensemble

#### CatBoost Gradient Boosting Forecaster

**Algorithm:** Joint multi-output gradient boosting via CatBoost's `MultiRMSE` loss  
**Library:** `catboost.CatBoostRegressor` (required for this ensemble member)

**Key Parameters:**
```python
n_estimators = 200        # Number of boosting stages
max_depth = 5             # Tree depth: 32 leaves ≈ ~24 samples/leaf at n≈756
                          # (principled mid-point; tunable via ForecastConfig.gb_max_depth)
loss = 'MultiRMSE'        # Joint multi-output loss exploiting cross-criterion correlations
allow_writing_files = False  # No on-disk catboost_info/ directories
```

**Advantages:**
- Joint multi-output training: a single tree structure minimizes total RMSE across all 8 criterion outputs simultaneously, exploiting cross-criterion correlations
- No feature scaling required: CatBoost oblivious trees are invariant to monotone feature transforms
- Feature importance from oblivious-tree splits
- Oblivious-tree splits provide strong regularization and stable multi-output learning on small-to-medium panels

**Why joint multi-output?**

For small-to-medium panel data (N < 1000), training one shared tree model outperforms `MultiOutputRegressor` wrappers:

- **Cross-criterion coupling**: provinces that rank high on one criterion tend to rank high on related criteria; shared split points exploit this automatically
- **Sample efficiency**: joint training uses all 8 output signals to guide each split, rather than fitting each criterion in isolation
- **Low correlation with other ensemble members**: CatBoost + Kernel methods + Bayes covers tree, linear, and kernel modelling families without redundancy

### 1.2 Bayesian Linear Model

#### Bayesian Ridge Forecaster

**Algorithm:** Bayesian linear regression with Gaussian priors  
**Library:** `sklearn.linear_model.BayesianRidge`

**Model:**
$$
p(y|X, w, \alpha, \lambda) = \mathcal{N}(y | Xw, \alpha^{-1})
$$
$$
p(w|\lambda) = \mathcal{N}(w | 0, \lambda^{-1}I)
$$

Where:
- $w$ = regression coefficients
- $\alpha$ = noise precision (learned)
- $\lambda$ = coefficient precision (learned)

**Advantages:**
- Natural uncertainty quantification via posterior variance
- Automatic regularization (no hyperparameter tuning)
- Handles multicollinearity well

**Prediction with Uncertainty:**
$$
p(y_*|X_*, X, y) = \mathcal{N}(y_* | \mu_*, \sigma_*^2)
$$

Where:
$$
\mu_* = X_* \mathbb{E}[w|X, y]
$$
$$
\sigma_*^2 = \frac{1}{\alpha} + X_* \Sigma_w X_*^T
$$

---

### 1.3 Kernel Methods

These models complement the tree-based and linear tracks with non-parametric kernel-based regression.

#### Kernel Ridge Regression Forecaster

**Algorithm:** Kernelized L2 regression via the RBF kernel
**File:** `forecasting/kernel_ridge.py`
**Key Parameters:**
```python
alpha = 1.0     # L2 regularisation strength (ForecastConfig.krr_alpha)
gamma = 'scale' # RBF kernel bandwidth (ForecastConfig.krr_gamma)
```

**Notes:** Wrapped in `MultiOutputRegressor`; features are PLS-compressed (same track as Bayesian Ridge) before fitting to avoid the $O(n^3)$ Gram matrix when feature count is large.

---

#### SVR Forecaster

**Algorithm:** Support Vector Regression with ε-insensitive loss and RBF kernel
**File:** `forecasting/svr.py`
**Key Parameters:**
```python
C       = 1.0   # Regularisation strength (ForecastConfig.svr_C)
epsilon = 0.1   # ε-tube half-width (ForecastConfig.svr_epsilon)
gamma   = 'scale' # RBF bandwidth (ForecastConfig.svr_gamma)
```

**Notes:** Wrapped in `MultiOutputRegressor`; uses the same PLS-compressed feature track as Bayesian Ridge and Kernel Ridge. SVR provides an ε-insensitive margin that ignores small residuals, adding complementary robustness to the ensemble.

---

### 1.3 Distributional Models

#### Quantile Random Forest Forecaster

**Algorithm:** Quantile estimation via leaf-based conditional distribution  
**Library:** Custom implementation on `sklearn.ensemble.RandomForestRegressor`

**Description:**  
Standard Random Forest provides point predictions, but QRF provides **full predictive distributions** by analyzing the distribution of training samples in each leaf node.

**Quantile Prediction Method:**
1. Train standard Random Forest on training data
2. For prediction sample $x_*$, find which leaf node it falls into for each tree
3. Extract all training labels from those leaf nodes
4. Compute weighted quantiles using leaf co-occurrence frequencies

**Quantile Formula:**
$$
\hat{q}_\tau(x_*) = \text{WeightedQuantile}_\tau(\{y_i : x_i \in \text{Leaf}(x_*)\})
$$

**Prediction Intervals:**
$$
[L, U] = [\hat{q}_{\alpha/2}(x_*), \hat{q}_{1-\alpha/2}(x_*)]
$$

**Advantages:**
- Non-parametric distributional forecasts
- Adaptive to heteroscedasticity
- No distributional assumptions needed
- Captures asymmetric uncertainty

**Key Parameters:**
```python
n_estimators = 200        # Number of trees
max_depth = None          # Full tree depth
min_samples_leaf = 5      # Minimum samples per leaf
quantiles = [0.025, 0.5, 0.975]  # Default quantiles
```

**Methods:**
- `predict()`: Point prediction (conditional mean — MSE-compatible, used by Super Learner meta-learner)
- `predict_median()`: Conditional median (MAE-optimal; routes through `predict_quantiles([0.5])`)
- `predict_mean()`: Conditional mean (standard RF average; same as `predict()`)
- `predict_quantiles(quantiles)`: Multiple quantile predictions
- `predict_uncertainty()`: IQR-based uncertainty
- `get_prediction_distribution()`: Full distributional summary (median, mean, IQR, all quantiles)

> **Note (Bug Q-1 fix):** `get_prediction_distribution()["median"]` previously called `predict_mean()`,
> causing the median and mean keys to be identical. It now routes through `predict_median()` (leaf-weight
> weighted quantile at q=0.5), which correctly differs from the RF mean on asymmetric distributions.

---

## Part II: Meta-Ensemble Methods

### 2.1 Super Learner (Stacked Generalization)

**File:** `forecasting/super_learner.py`  
**Algorithm:** Van der Laan et al. (2007), "Super Learner"

**Description:**  
Optimal weighted combination of base models via **nested cross-validation**. Instead of simple weighting, trains a meta-learner on out-of-fold predictions.

**Three-Stage Algorithm:**

**Stage 1: Generate Out-of-Fold Predictions**
```
For each CV fold k:
  Train each base model on folds ≠ k
  Predict on fold k
  Store predictions as meta-features
Result: Z = [ŷ₁_OOF, ŷ₂_OOF, ..., ŷₘ_OOF]
```

**Stage 2: Train Meta-Learner**
```
Train meta-model: y = Z × α
Where α = meta-weights (learned)
Constraints: α ≥ 0, Σα = 1
```

**Stage 3: Retrain Base Models**
```
Train all base models on full training data
Final prediction: ŷ = Σ α_i × ŷ_i
```

**Meta-Learner Types:**

1. **Ridge Regression** (default)
   $$
   \min_\alpha ||y - Z\alpha||^2 + \lambda ||\alpha||^2
   $$
   Subject to: $\alpha \geq 0$, $\sum \alpha_i = 1$

2. **ElasticNet**
   $$
   \min_\alpha ||y - Z\alpha||^2 + \lambda_1 ||\alpha||_1 + \lambda_2 ||\alpha||^2
   $$

3. **Dirichlet Stacking** (`dirichlet_stacking`)
   Optimizes the Dirichlet negative log-likelihood over logit-parameterized weights
   using analytical gradients ($\partial\text{NLL}/\partial\text{logit}_j = n w_j - \sum_n r_{nj}$).
   Per-output weights (`_meta_weights_per_output_`) are stored so each criterion gets
   its own optimal combination. Entity-block bootstrap provides UQ on those weights.

4. **Bayesian Stacking** (`bayesian_stacking`)
   Softmax-of-R² temperature-scaled weighting; simpler than full Dirichlet posterior but
   fast and useful as a baseline.

**Advantages:**
- Optimal weights (oracle inequality guarantees)
- Prevents overfitting via out-of-fold predictions
- Flexible meta-learner choice
- Theoretically principled

**Cross-Validation Strategy:**
Uses `PanelWalkForwardCV` (public alias of `_WalkForwardYearlySplit`) — a panel-aware
walk-forward splitter that creates annual validation folds in sorted year order.
`min_train_years` (default 5) ensures every fold has a minimum number of training
years before validation; `max_folds` (default 5) caps the total number of folds to
prevent excessive runtime on long panels.  Each entity contributes its rows for each
year-fold independently; entities absent from a fold's validation year simply
contribute zero validation rows for that fold.  Falls back to `TimeSeriesSplit` when
year labels are not provided.

**Phase 4: Per-criterion RMSE tracking:**  
In addition to the fold-level mean R² stored in `_cv_scores_`, the SuperLearner
now records per-criterion RMSE per fold in `_cv_scores_per_criterion_` (`Dict[model_name,
List[List[float]]]`).  Stage 6 exposes these as `per_criterion_rmse_mean` and
`per_criterion_rmse_std` arrays in each model's `model_performance` entry.

> **Note (Bug S-2 fix):** The previous implementation used `T = min(entity lengths)`,
> which discarded up to 9 years of data for longer-history provinces when any province
> had a short history. The walk-forward splitter recovers that data.

**Key Parameters:**
```python
meta_learner_type = 'ridge'     # 'ridge', 'elasticnet', 'dirichlet_stacking',
                                # 'bayesian_stacking'
n_cv_folds = 5                  # OOF folds
positive_weights = True         # α ≥ 0 constraint
normalize_weights = True        # Σα = 1 constraint
```

**Methods:**
- `fit(X, y)`: Train Super Learner
- `predict(X)`: Ensemble predictions
- `predict_with_uncertainty(X)`: Predictions + model disagreement
- `get_meta_weights()`: Learned α values
- `get_cv_scores()`: Base model CV performance

---

## Part III: Uncertainty Calibration

### 3.1 Conformal Prediction

**File:** `forecasting/conformal.py`  
**Paper:** Vovk et al. (2005), "Algorithmic Learning in a Random World"

**Description:**  
Provides **distribution-free prediction intervals** with guaranteed finite-sample coverage, regardless of model correctness or data distribution.

**Coverage Guarantee:**
$$
\mathbb{P}(y_{n+1} \in C(X_{n+1})) \geq 1 - \alpha
$$

For any distribution $P$, any model, any finite sample size.

**Three Methods:**

#### Method 1: Split Conformal

**Algorithm:**
1. Split data: (train, calibration)
2. Train model on train set
3. Compute residuals on calibration set: $R_i = |y_i - \hat{y}_i|$
4. Find quantile: $q = \text{Quantile}_{1-\alpha}(R_1, \ldots, R_m)$
5. Prediction interval: $[\hat{y} - q, \hat{y} + q]$

**Coverage:**
$$
\mathbb{P}(y_{n+1} \in [\hat{y} \pm q]) \geq 1 - \alpha
$$

---

#### Method 2: CV+ Conformal

**Algorithm:**
Uses cross-validation residuals (no data splitting):
1. Compute K-fold CV residuals (out-of-fold)
2. $R^{\text{CV}} = \{|y_i - \hat{y}_i^{(-k(i))}|\}_{i=1}^n$
3. Find quantile: $q = \text{Quantile}_{1-\alpha}(R^{\text{CV}})$
4. Prediction interval: $[\hat{y} - q, \hat{y} + q]$

**Advantage:**  
No data loss (uses full training set), while maintaining validity.

> **Coverage Note (B-9):** The implementation uses `TimeSeriesSplit` OOF folds
> instead of leave-one-out (LOO) cross-validation. The finite-sample
> jackknife+ coverage guarantee (Barber et al., 2021) requires LOO folds
> *and* exchangeability of the data — both of which are violated by temporal
> panel data. Coverage is therefore **empirically validated** rather than
> theoretically guaranteed at finite $n$. In practice the method is
> well-calibrated for this dataset (validated by `TestConformalPredictor`),
> but users should treat the 95% label as a target, not a hard guarantee.

---

#### Method 3: Adaptive Conformal Inference (ACI)

**Algorithm:**  
Online adaptive intervals that track coverage in real-time:

**Update Rule:**
$$
\alpha_t = \alpha_{t-1} + \gamma(\text{err}_t - \alpha)
$$

Where:
- $\text{err}_t = \mathbb{1}(y_t \notin C_t)$ (miscoverage indicator)
- $\gamma = 0.02$ (additive step size — from Gibbs & Candès 2021, Eq. 3)
- Target: $\mathbb{E}[\text{err}_t] = \alpha$

> **Note (Bug C-3 fix):** The previous default $\gamma = 0.95$ was borrowed from
> exponential-smoothing *forgetting factors* where the formula is multiplicative.
> With the *additive* ACI update rule a single miss shifts $\alpha_t$ by $\gamma$,
> so large values (e.g. 0.95) cause wild oscillation. The literature recommends
> $\gamma \in [0.005, 0.05]$; the default is now 0.02.

**Adaptive Quantile:**
$$
q_t = \text{Quantile}_{1-\alpha_t}(R_{\text{calibration}})
$$

**Advantage:**  
Adapts to non-stationarity, heteroscedasticity, distribution shift.

---

**Methods:**
- `calibrate(model, X_cal, y_cal)`: Calibrate using raw data (fits model, computes conformity scores). For multi-output, calibrate one predictor per component with Bonferroni-corrected α.
- `calibrate_residuals(residuals)`: Calibrate from **pre-computed OOF residuals** without re-fitting any model. Used by `UnifiedForecaster` to avoid deep-copying the full Super Learner ensemble during conformal calibration (performance fix U-2).
- `predict_intervals(X)`: Return `(lower, upper)` coverage-guaranteed interval arrays
- `evaluate_coverage(y_true, intervals)`: Empirical coverage check
- `get_interval_width()`: Average interval width

---

## Part IV: Evaluation Suite

### 4.1 Forecast Evaluator

**File:** `forecasting/evaluation.py`

**Metrics (7 total):**

1. **R² Score**: $R^2 = 1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$
2. **RMSE**: $\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}$
3. **MAE**: $\frac{1}{n}\sum|y - \hat{y}|$
4. **MedAE**: $\text{median}(|y - \hat{y}|)$
5. **MAPE**: $\frac{100}{n}\sum\frac{|y - \hat{y}|}{|y|}$
6. **Max Error**: $\max|y - \hat{y}|$
7. **Bias**: $\frac{1}{n}\sum(y - \hat{y})$

**Residual Diagnostics:**
- **Durbin-Watson**: Tests for autocorrelation
- **Heteroscedasticity**: Breusch-Pagan test
- **Normality**: Shapiro-Wilk test

**Uncertainty Evaluation:**
- **Winkler Score**: Interval score penalizing width + violations
  $$
  S_\alpha(L, U, y) = (U - L) + \frac{2}{\alpha}(L - y)\mathbb{1}(y < L) + \frac{2}{\alpha}(y - U)\mathbb{1}(y > U)
  $$
- **Calibration Curve**: Empirical vs. nominal coverage
- **Sharpness**: Average interval width

### 4.2 Ablation Study

**Purpose:** Isolate each model's contribution to ensemble performance.

**Algorithm:**
```
For each model i:
  Train ensemble without model i
  Compute performance drop: Δ = Perf(all) - Perf(all \ i)
Rank models by Δ (contribution)
```

**Output:**
- Model importance ranking
- Performance degradation per model
- Identifies redundant models

---

## Part V: Feature Engineering & Usage

### 5.1 Temporal & Panel Feature Engineering (Phase 1)

The feature engineering is handled by the `TemporalFeatureEngineer` class (`forecasting/features.py`), which builds a rich feature set spanning **12 structural blocks**.  Phase 1 added 8 new feature types (G-01–G-08) and corrected 3 bugs (D-01–D-03) relative to the pre-Phase 1 baseline.

#### 12 Feature Blocks

| Block | Name | Features generated | Phase 1 change |
| :---: | :--- | :--- | :--- |
| 1 | Current values | `C01 … C08` (raw levels at $t$) | — |
| 2 | Lag + missingness | `C01_lag1/2/3`, `C01_lag1_was_missing/2/3` | D-01: NaN → cross-sectional median + `_was_missing` flag |
| 3 | Rolling statistics | `C01_roll2/3/5_mean/std/min/max` | G-02: window=5 added to existing {2,3} |
| 4 | Momentum / acceleration | `C01_momentum`, `C01_acceleration` | D-02: `_delta1` removed (duplicated `_momentum`) |
| 5 | Stationarity | `C01_demeaned`, `C01_demeaned_momentum`, `C01_delta2` | — |
| 6 | Polyfit trend | `C01_trend` slope (≥ 3 valid points) | G-08: min-points raised from 2 → 3 |
| 7 | EWMA levels | `C01_ewma2/3/5` | **G-01** (new) |
| 8 | Expanding mean | `C01_expanding_mean` | **G-03** (new) |
| 9 | Inter-criterion diversity | `diversity_std`, `diversity_range` | **G-04** (new) |
| 10 | Rolling skewness | `C01_roll5_skew` | **G-07** (new) |
| 11 | Panel-relative | `C01_percentile`, `C01_zscore`, `C01_pct_change` | D-03: filtered to `active_provinces`; **G-05** rank-change |
| 12 | Regional dummies | `region_0 … region_4` | **G-06** (new, 5 Vietnam geographic regions) |

**Approximate total features (8 criteria):**  
Block 1 (8) + Block 2 (48+24) + Block 3 (96) + Block 4 (16) + Block 5 (24) + Block 6 (8) + Block 7 (24) + Block 8 (8) + Block 9 (2) + Block 10 (8) + Block 11 (24) + Block 12 (5) ≈ **295 features**

After threshold-only variance filtering (Block 2 `reducer_tree_`) the effective count is typically 240–270 for standard runs.

#### Phase 1 Bug Fixes

| ID | Description |
| :--- | :--- |
| **D-01** | Lag NaN values filled with the cross-sectional median for that (year, component) pair rather than 0.0; binary `_was_missing` indicators appended so models can discount imputed inputs |
| **D-02** | `_delta1` ("first difference") removed — it is an exact duplicate of `_momentum`; `_delta2` (lagged first difference) kept as the stationarity signal |
| **D-03** | Cross-entity percentile and z-score features now filtered to `active_provinces` from `panel_data.year_contexts` rather than all provinces; prevents stale provinces from distorting the reference distribution |

#### Phase 1 New Features (G-01–G-08)

| ID | Feature | Rationale |
| :--- | :--- | :--- |
| G-01 | EWMA levels (spans 2, 3, 5) | Exponentially down-weight distant observations; better recency signal than rolling mean for trending criteria |
| G-02 | Rolling window=5 | Captures medium-term 5-year trends missed by windows {2, 3} |
| G-03 | Expanding mean | Long-run unconditional baseline per entity × criterion; useful for mean-reversion detection |
| G-04 | Inter-criterion diversity (std, range) | Province-level diversity across all 8 simultaneous criterion values; signals governance concentration risk |
| G-05 | Rank-change ($\Delta$percentile) | $\text{pct}_t - \text{pct}_{t-1}$: isolates mobility / convergence dynamics |
| G-06 | Regional cluster dummies | 5 Vietnam geographic regions (Northern Mountains, Red River Delta, Central, Central Highlands, Southern); captures systematic regional policy differences |
| G-07 | Rolling skewness (window=5) | Distinguishes breakout provinces (positive skew) from regression-to-mean (negative); asymmetric distribution signal |
| G-08 | Polyfit min-points fix | `≥ 3` valid points required for trend fit; was `≥ 2` which produced unreliable 2-point slopes |

#### Two-Track Preprocessing (Phase 2)

After feature engineering, two separate tracks feed model classes differently:

| Track | Reducer | Models | Notes |
| :--- | :--- | :--- | :--- |
| **PLS** (`reducer_pca_`) | `PLSRegression(n_components=20)` with MI pre-filter | BayesianRidge | Supervised compression maximises covariance with all 8 criterion targets simultaneously; `n_components = min(n//10, 20)` |
| **Threshold-only** (`reducer_tree_`) | Variance threshold filter (no scaling) | CatBoost, QuantileRF | Preserves original feature structure; StandardScaler removed — CatBoost is scale-invariant and QRF applies its own RobustScaler |

#### Example Generated Features (8-criterion dataset)

```
# Block 1: current values
C01, C02, …, C08

# Block 2: lags + missingness
C01_lag1, C01_lag2, C01_lag3
C01_lag1_was_missing, C01_lag2_was_missing, C01_lag3_was_missing

# Block 3: rolling statistics (windows 2, 3, 5)
C01_roll2_mean, C01_roll2_std, C01_roll2_min, C01_roll2_max
C01_roll3_mean, C01_roll3_std, C01_roll3_min, C01_roll3_max
C01_roll5_mean, C01_roll5_std, C01_roll5_min, C01_roll5_max

# Block 4: momentum / acceleration
C01_momentum, C01_acceleration

# Block 5: stationarity
C01_demeaned, C01_demeaned_momentum, C01_delta2

# Block 6: trend
C01_trend

# Block 7: EWMA
C01_ewma2, C01_ewma3, C01_ewma5

# Block 8: expanding mean
C01_expanding_mean

# Block 9: inter-criterion diversity (2 features, not per-criterion)
diversity_std, diversity_range

# Block 10: rolling skewness
C01_roll5_skew

# Block 11: panel-relative
C01_percentile, C01_zscore, C01_pct_change

# Block 12: regional dummies (5, not per-criterion)
region_0, region_1, region_2, region_3, region_4

… (Blocks 2–11 repeated for C02–C08)
```

#### SAW Normalisation

When `use_saw_targets=True` (production default), year-level targets are per-year
column-wise minmax-normalised to `[0, 1]` before training.  This:

1. Removes cross-year level differences (each year rescaled independently)
2. Preserves within-year ordinal structure
3. Avoids CRITIC-weighting bias in raw composites

The Phase 5 **target transformer** further maps these `[0, 1]` values through the
logit function (`log(y/(1-y)) → ℝ`) before Super Learner training, improving the
Gaussianity assumption of BayesianRidge and the Ridge meta-learner.  Conformal bound
validity is preserved because logit is strictly monotone.

### 5.2 Quick Start Guide

**Minimal Example:**
```python
from forecasting import UnifiedForecaster

# State-of-the-art configuration (Super Learner + Conformal)
forecaster = UnifiedForecaster()
result = forecaster.fit_predict(panel_data, target_year=2025)

# Access predictions and intervals
print(result.predictions)  # Point predictions
print(result.prediction_intervals['lower'])  # 95% lower bound
print(result.prediction_intervals['upper'])  # 95% upper bound
print(result.model_contributions)  # Model weights from Super Learner
```

**Custom Configuration:**
```python
# Adjust conformal prediction settings
forecaster = UnifiedForecaster(
    conformal_alpha=0.10,  # 90% coverage (less conservative)
    conformal_method='adaptive',  # Adaptive to non-stationarity
    cv_folds=5,  # More folds for larger datasets
    verbose=True  # Print progress
)

result = forecaster.fit_predict(panel_data, target_year=2025)

# Examine model contributions
for model, weight in result.model_contributions.items():
    print(f"{model}: {weight:.3f}")
```

### 5.3 Installation

**Core Dependencies** (required):
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

**Optional Dependencies** (for alternative implementations):
```bash
pip install mapie   # Alternative conformal prediction implementation
```

Or install all forecasting dependencies:
```bash
pip install -e .[forecasting]
```

---

## Part VI: Advantages & Limitations

### 6.1 State-of-the-Art Advantages

1. **Multi-Tier Architecture**
   - 6 diverse base models capturing different patterns (tree ×2, kernel ×2, linear ×1, distributional ×1)
   - Super Learner meta-learning with automatic optimal weighting
   - Distribution-free calibrated uncertainty (conformal prediction)
   - Full 3-tier pipeline optimized for small-to-medium panel data (N < 1000)

2. **Distributional & Kernel Methods**
   - **Quantile RF**: Full distributional forecasts (not just point estimates)
   - **Kernel Ridge / SVR**: Non-parametric kernel regression complementing linear track

3. **Optimal Ensemble Learning**
   - **Super Learner**: Walk-forward meta-learning (`PanelWalkForwardCV`, panel-aware)
   - **Positive Constraints**: Ensures monotonic relationships and stability
   - **Out-of-Fold Training**: Prevents overfitting in meta-learner
   - **Diversity-First**: 5 diverse models outperform larger correlated sets

4. **Calibrated Uncertainty Quantification**
   - **Conformal Prediction**: Guaranteed finite-sample coverage (≥ 1-α)
   - **Adaptive Conformal**: Tracks non-stationarity online
   - **Distributional Forecasts**: Full quantile predictions from QRF
   - **Multi-Source Uncertainty**: Observation noise + parameter variance

5. **Comprehensive Evaluation**
   - **7 Metrics**: R², RMSE, MAE, MedAE, MAPE, Max Error, Bias
   - **Residual Diagnostics**: Durbin-Watson, heteroscedasticity, normality tests
   - **Uncertainty Scoring**: Winkler score, calibration curves, sharpness
   - **Ablation Studies**: Isolate individual model contributions

6. **Robustness & Flexibility**
   - Outlier-robust gradient boosting with CatBoost `MultiRMSE`
   - Statistically-principled design for small-to-medium data (N < 1000)
   - Extensible architecture (easy to add new models)
   - Panel-aware walk-forward CV (`PanelWalkForwardCV`) prevents temporal leakage
   - Handles panel structure (entity heterogeneity)

7. **Interpretability**
   - Feature importance (aggregated across models)
   - Meta-weights show model contributions

### 6.2 Limitations

1. **Computational Cost**
   - Trains 5 models + Super Learner + Conformal (moderate speed, ~2–5 min)
   - Optional Phase 4 HP tuning adds ~1–2 min for 20 Optuna trials
   - Feature engineering (295 features) increases dimensionality
   - For large-scale production (N > 10,000), consider simplified configurations

2. **Data Requirements**
   - Conformal calibration needs ≥30 calibration samples (guideline)
   - For N < 1000: 5 diverse models (tree, kernel×2, linear, distributional) provide a strong bias-variance tradeoff

3. **Optional Dependency Requirements**
   - **Mapie** optional for alternative conformal implementation
   - Core functionality works without optional dependencies

4. **Hyperparameter Tuning**
   - Base model hyperparameters are preset (reasonable defaults)
   - Super Learner meta-learner type ('ridge', 'elasticnet', 'bayesian_stacking') is fixed
   - Conformal method ('split', 'cv_plus', 'adaptive') is configurable

5. **Extrapolation Risk**
   - All models train on historical patterns
   - May fail during regime changes (e.g., policy shifts, pandemics)
   - Conformal intervals assume exchangeability (may degrade under drift)
   - Adaptive Conformal (ACI) partially mitigates drift but assumes gradual changes

6. **Meta-Learning Requirements**
   - Super Learner needs ≥3 base models for meaningful weights
   - Model diversity more important than quantity (5 diverse > many correlated)
   - Very high correlation between base models reduces ensemble gains
   - For N≈756: 5 diverse models (tree, kernel×2, linear, distributional) without redundancy

### 6.3 Current Capabilities vs. Future Enhancements

#### ✅ Currently Implemented

- ✅ Quantile Random Forest (distributional forecasts)
- ✅ Kernel Ridge Regression (RBF kernel, L2 regularised)
- ✅ Support Vector Regression (ε-insensitive tube, RBF kernel)
- ✅ Super Learner (stacked generalization, `PanelWalkForwardCV`, per-output meta-weights `_meta_weights_per_output_`)
- ✅ Conformal Prediction (split, CV+, adaptive; OOF residuals, per-column masks `_oof_valid_mask_per_col_`)
- ✅ Comprehensive evaluation suite (7 metrics, per-criterion RMSE tracking)
- ✅ **Phase 1**: 12-block temporal & panel feature engineering (D-01/02/03 fixes, G-01–G-08 new features)
- ✅ **Phase 2**: PLS-supervised compression for linear models, threshold-only track for trees
- ✅ **Phase 3**: CatBoost-only gradient boosting track integrated with the ensemble
- ✅ **Phase 4**: Per-criterion RMSE CV tracking; optional Optuna one-time HP search for CatBoost
- ✅ **Phase 5**: Reversible target transformation (logit/Yeo-Johnson); inverse-transform of all pipeline outputs
- ✅ **Phase I (correctness)**: F-01 per-column OOF masks; F-02 per-output meta-weights + `_get_weight()`; F-03 entity-block bootstrap; F-04 analytical Dirichlet gradient; F-05 `cv_min_train_years=5`; F-06 PCA Bonferroni + QRF lower-q floor; F-07 conformal n_cal warning
- ✅ **ML panel imputation**: `build_ml_panel_data()` supplies a 3-stage imputed copy to the forecaster; MCDM phases remain on raw observed data
- ✅ **Province fallback**: provinces absent from last-year active set fall back to most-recent valid year for prediction

#### 🔄 Planned Future Enhancements
   - Temporal Convolutional Networks (TCN)
   - Transformers for time series (Informer, Autoformer)
   - Temporal Fusion Transformers (TFT)
   - Neural ODE for continuous-time dynamics

3. **Causal Methods**
   - Synthetic Control Methods for counterfactuals
   - Difference-in-Differences with panel data
   - Causal forests for heterogeneous treatment effects
   - Granger causality for lead-lag relationships

4. **Distributional Extensions**
   - GAMLSS (Generalized Additive Models for Location, Scale, Shape)
   - Gaussian Processes for full posterior
   - Conditional quantile forests (heteroscedastic quantiles)
   - Copula-based multivariate forecasts

5. **Multi-Step Ahead Forecasting**
   - Direct multi-step: Train separate models for t+1, t+2, ...
   - Recursive: Iterate one-step-ahead predictions
   - Seq2Seq: Neural sequence-to-sequence models

6. **Ensemble Diversity Enhancement**
   - Negative correlation learning
   - Diversity-promoting ensembles (Krogh & Vedelsby)
   - Dynamic ensemble selection (local expertise)

---

## References

### Core Machine Learning

1. **Friedman, J.H.** (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.

2. **Breiman, L.** (2001). Random forests. *Machine Learning*, 45(1), 5-32.

3. **Geurts, P., Ernst, D., & Wehenkel, L.** (2006). Extremely randomized trees. *Machine Learning*, 63(1), 3-42.

4. **MacKay, D.J.C.** (1992). Bayesian interpolation. *Neural Computation*, 4(3), 415-447.

5. **Huber, P.J.** (1964). Robust estimation of a location parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.

### Panel Data Methods

6. **Hsiao, C.** (2014). *Analysis of panel data* (3rd ed.). Cambridge University Press.

7. **Holtz-Eakin, D., Newey, W., & Rosen, H.S.** (1988). Estimating vector autoregressions with panel data. *Econometrica*, 56(6), 1371-1395.

8. **Gelman, A., & Hill, J.** (2006). *Data analysis using regression and multilevel/hierarchical models*. Cambridge University Press.

### Advanced Forecasting

9. **Meinshausen, N.** (2006). Quantile regression forests. *Journal of Machine Learning Research*, 7, 983-999.

10. **Agarwal, R., et al.** (2021). Neural additive models: Interpretable machine learning with neural nets. *NeurIPS 2021*.

11. **Van der Laan, M.J., Polley, E.C., & Hubbard, A.E.** (2007). Super Learner. *Statistical Applications in Genetics and Molecular Biology*, 6(1).

12. **Vovk, V., Gammerman, A., & Shafer, G.** (2005). *Algorithmic learning in a random world*. Springer.

13. **Gibbs, I., & Candes, E.** (2021). Adaptive conformal inference under distribution shift. *NeurIPS 2021*.

### Bayesian Optimization

14. **Akiba, T., et al.** (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD 2019*.

15. **Bergstra, J., et al.** (2011). Algorithms for hyper-parameter optimization. *NeurIPS 2011*.

### Deep Learning

16. **Klambauer, G., et al.** (2017). Self-normalizing neural networks. *NeurIPS 2017*.

17. **Vaswani, A., et al.** (2017). Attention is all you need. *NeurIPS 2017*.

### Time Series Cross-Validation

18. **Bergmeir, C., & Benítez, J.M.** (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.

19. **Cerqueira, V., et al.** (2020). Evaluating time series forecasting models: An empirical study on performance estimation methods. *Machine Learning*, 109, 1997-2028.

### Evaluation & Diagnostics

20. **Gneiting, T., & Raftery, A.E.** (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359-378.

21. **Winkler, R.L.** (1972). A decision-theoretic approach to interval estimation. *Journal of the American Statistical Association*, 67(337), 187-191.