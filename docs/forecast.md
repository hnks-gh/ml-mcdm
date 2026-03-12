# Forecasting Methodology: State-of-the-Art Ensemble Machine Learning

## Overview

This framework implements a **statistically-principled 3-tier ensemble learning system** optimized for small-to-medium panel data (N < 1000). It combines 6 diverse machine learning models with Super Learner meta-learning and distribution-free conformal prediction to forecast future criterion values.

**Key Design Principles:**
- **Model diversity over quantity**: 5 diverse models outperform 11+ correlated models
- **Statistical appropriateness**: Optimized for N < 1000 (your dataset: N=819)
- **Automatic optimal weighting**: Super Learner learns best combination
- **Guaranteed coverage**: Conformal prediction provides 95% valid intervals
- **No redundancy**: Each model captures different patterns (tree, linear, panel, Bayesian)

**Key Features:**
- **6 Model Types**: CatBoost (joint multi-output) + LightGBM (leaf-wise), Bayesian linear, and advanced panel models
- **Super Learner**: Automatic optimal weighting via meta-learning (`PanelWalkForwardCV`)
- **Conformal Prediction**: Distribution-free 95% prediction intervals
- **Distributional Forecasting**: Full predictive distributions via quantile forests
- **Panel Data Methods**: VAR with fixed effects
- **Interpretable Non-Linearity**: Neural Additive Models with shape functions
- **Enhanced Feature Engineering**: 12 feature blocks — lag, rolling, stationarity, EWMA, diversity, region dummies (Phase 1)
- **Target Transformation**: Logit/Yeo-Johnson reversible transform for improved Gaussianity (Phase 5)
- **HP Optimisation**: Optional Optuna one-time search for both GB models (Phase 4)

---

## System Architecture

### Three-Tier Architecture

```
Input: Panel Data (N entities × p components × T years)
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIER 1: BASE MODELS (6 diverse models)
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
  ├── Tree-Based (2 models)
  │   ├── CatBoost Gradient Boosting (MultiRMSE joint multi-output loss)
  │   └── LightGBM (leaf-wise per-output via MultiOutputRegressor)
  │
  ├── Bayesian Linear (1 model)
  │   └── Bayesian Ridge (posterior uncertainty, PLS-compressed features)
  │
  └── Advanced Panel Models (3 models)
      ├── Quantile Random Forest (distributional forecasts)
      ├── Panel VAR (LSDV fixed effects + autoregressive)
      └── Neural Additive Models (interpretable non-linearity)
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

**Base Models (6):**
- CatBoost Gradient Boosting (oblivious trees, joint multi-output `MultiRMSE`)
- LightGBM (leaf-wise growth, `MultiOutputRegressor`, complementary inductive bias)
- Bayesian Ridge (linear with posterior uncertainty, PLS-compressed features)
- Quantile RF + Panel VAR + NAM (panel-specific)

**Meta-Ensemble:** Super Learner (`PanelWalkForwardCV` panel-aware CV, NNLS meta-weights)

**Calibration:** Conformal Prediction (95% coverage guarantee, OOF residuals)

**Feature Engineering:** 12 feature blocks — lag, rolling {2,3,5}, stationarity, EWMA, diversity, skewness, region dummies (Phase 1, ≈279 features)

**Target Transformation:** Logit (SAW) / Yeo-Johnson (raw) reversible transform applied before Stage 2 so PLS compression uses the transformed covariance (Phase 5)

**HP Optimisation:** One-time Optuna TPE search for both GB models when `auto_tune_gb=True` (Phase 4)

**Rationale:** For N=882 (63 provinces × 14 year pairs, 2011–2024), 6 diverse models with Super Learner meta-learning provides optimal bias-variance tradeoff and model diversity.

---

## Part I: Model Types

### 1.1 Tree-Based Ensemble

#### CatBoost Gradient Boosting Forecaster

**Algorithm:** Joint multi-output gradient boosting via CatBoost's `MultiRMSE` loss  
**Library:** `catboost.CatBoostRegressor` (falls back to LightGBM or sklearn GradientBoostingRegressor if not installed)

**Key Parameters:**
```python
n_estimators = 200        # Number of boosting stages
max_depth = 5             # Tree depth: 32 leaves ≈ 24 samples/leaf at n=819
                          # (principled mid-point; tunable via ForecastConfig.gb_max_depth)
loss = 'MultiRMSE'        # Joint multi-output loss exploiting cross-criterion correlations
allow_writing_files = False  # No on-disk catboost_info/ directories
```

**Advantages:**
- Joint multi-output training: a single tree structure minimizes total RMSE across all 8 criterion outputs simultaneously, exploiting cross-criterion correlations
- No feature scaling required: CatBoost oblivious trees are invariant to monotone feature transforms
- Feature importance from oblivious-tree splits
- Complementary to LightGBM: oblivious-tree splits (single split per depth level) provide different inductive bias to LightGBM's leaf-wise splits

**Why joint multi-output?**

For small-to-medium panel data (N < 1000), training one shared tree model outperforms `MultiOutputRegressor` wrappers:

- **Cross-criterion coupling**: provinces that rank high on one criterion tend to rank high on related criteria; shared split points exploit this automatically
- **Sample efficiency**: joint training uses all 8 output signals to guide each split, rather than fitting each criterion in isolation
- **Low correlation with other ensemble members**: CatBoost + Panel models + Bayes covers tree, linear, and panel modelling families without redundancy

---

#### LightGBM Forecaster (Phase 3 addition)

**Algorithm:** Per-output leaf-wise gradient boosting via `MultiOutputRegressor(LGBMRegressor)`  
**Library:** `lightgbm.LGBMRegressor` wrapped in `sklearn.multioutput.MultiOutputRegressor`

**Key Parameters:**
```python
n_estimators   = 200        # Boosting rounds per output
max_depth      = 5          # Tree depth (same scale as CatBoost)
learning_rate  = 0.05       # Step size (identical default)
l2_reg         = 3.0        # L2 leaf regularisation (analogous to CatBoost l2_leaf_reg)
subsample      = 0.8        # Row-sampling fraction for variance reduction
```

**Why LightGBM as an independent member?**

CatBoost and LightGBM share the gradient-boosting family but differ in key inductive biases:

| Property | CatBoost | LightGBM |
| :--- | :--- | :--- |
| Tree growth | Symmetric (oblivious) — same split per depth | Leaf-wise (best-first) — asymmetric |
| Multi-output | Native `MultiRMSE` joint loss | `MultiOutputRegressor` per-criterion |
| Feature routing | Threshold-only track | Threshold-only track |
| Typical strength | Correlated outputs, regularised depth | High-gain leaves, single-output precision |

For N < 1000, these complementary biases produce meaningfully different predictions; Super Learner NNLS weights empirically allocate non-zero weight to both, reducing ensemble variance without adding correlated redundancy.

**Phase 4 HP tuning (when `auto_tune_gb=True`):**  
Optuna TPE study (`gb_tune_n_trials=20`) searches over `{n_estimators, max_depth, learning_rate, l2_reg}` using `PanelWalkForwardCV(min_train_years=7, max_folds=4)`. Tuned parameters override config defaults in `_create_models()`.

**Methods:** Same `BaseForecaster` interface as all other ensemble members:  
`fit(X, y)`, `predict(X)`, `predict_with_uncertainty(X)`, `get_feature_importance()`

---

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

### 1.3 Advanced Panel Models

These state-of-the-art models are specifically designed for panel data.

**Why Advanced Panel Models?**

For panel data (entities × time periods × components), these specialized models outperform generic ML:
- **Panel VAR**: Captures entity heterogeneity + temporal autocorrelation
- **Quantile RF**: Full predictive distributions, not just point estimates
- **Neural Additive Models**: Interpretable non-linearity with visualizable shape functions

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

#### Panel VAR Forecaster

**Algorithm:** Panel Vector Autoregression with fixed effects (LSDV)  
**Library:** Custom implementation on `sklearn.linear_model.Ridge/ElasticNet`

**Description:**  
Combines panel fixed effects (entity-specific intercepts) with vector autoregressive dynamics. Captures both entity heterogeneity and temporal dependencies.

**Model:**
$$
y_{it} = \alpha_i + \sum_{l=1}^p \mathbf{B}_l y_{i,t-l} + \epsilon_{it}
$$

Where:
- $y_{it}$ = vector of components for entity $i$ at time $t$
- $\alpha_i$ = entity fixed effect (LSDV dummy variables)
- $\mathbf{B}_l$ = lag-$l$ coefficient matrix (cross-component dynamics)
- $p$ = number of lags (selected via BIC/AIC)

**Fixed Effects (LSDV):**
Entity dummies are added to feature matrix:
$$
X_{it} = [\mathbf{D}_i, y_{i,t-1}, y_{i,t-2}, \ldots, y_{i,t-p}]
$$

Where $\mathbf{D}_i$ is a one-hot encoded entity indicator.

**Regularization:**  
Ridge or ElasticNet regularization prevents overfitting due to high dimensionality:
$$
\min_{\alpha, \mathbf{B}} \sum_{i,t} ||y_{it} - \alpha_i - \sum_l \mathbf{B}_l y_{i,t-l}||^2 + \lambda ||\mathbf{B}||_2^2
$$

**Lag Selection:**  
Optimal lag order (1–3) is selected using hold-out CV MSE (`pvar_lag_selection_method = "cv"`). Classic penalised-likelihood information criteria (BIC, AIC) are not valid under Ridge regularisation because the effective degrees-of-freedom `tr(X(X'X+λI)⁻¹X') ≪ raw parameter count`.

$$
\text{CV-MSE}(p) = \frac{1}{|\text{val}|} \sum_{(i,t) \in \text{val}} ||y_{it} - \hat{y}_{it}^{(p)}||^2
$$

**Advantages:**
- Captures entity heterogeneity (fixed effects)
- Models cross-component dynamics (VAR)
- Data-driven lag selection via CV
- Regularization handles high dimensionality
- Interpretable coefficients

**Key Parameters:**
```python
n_lags = 2                # Lag order (or 'auto' for CV selection)
alpha = 1.0               # Regularization strength
l1_ratio = 0.0            # ElasticNet mixing (0=Ridge, 1=Lasso)
use_fixed_effects = True  # Include entity dummies
lag_selection_method = 'cv'  # Hold-out CV MSE (only valid method under Ridge)
```

---

#### Neural Additive Model Forecaster

**Algorithm:** Neural Additive Models via Random Kitchen Sinks  
**Paper:** Agarwal et al. (2021), "Neural Additive Models"

**Description:**  
Learns **interpretable non-linear** relationships while maintaining additive structure. Each feature gets its own neural network (shape function), and predictions are additive:

$$
\hat{y} = \beta_0 + \sum_{j=1}^p f_j(x_j)
$$

Where $f_j$ is a neural network for feature $j$ only.

**Architecture:**
1. **Feature Networks**: Separate network per feature
2. **Random Kitchen Sinks (RKS)**: Random Fourier Features for approximation
3. **Backfitting Algorithm**: Cyclic coordinate descent for training

**Random Fourier Features (RKS):**
Approximates kernel methods using random projections:
$$
\phi(x) = \sqrt{\frac{2}{M}} \cos(\omega_1 x + b_1, \ldots, \omega_M x + b_M)
$$

Where $\omega_i \sim \mathcal{N}(0, \sigma^2)$, $b_i \sim \text{Uniform}(0, 2\pi)$.

**Backfitting Algorithm:**
```
Initialize: f_j(x) = 0 for all j
Repeat until convergence:
  For j = 1 to p:
    r = y - Σ_{k≠j} f_k(x_k)     # Partial residuals
    f_j ← Train(x_j, r)           # Fit f_j to residuals
```

**Shape Function Extraction:**
After training, each $f_j(x_j)$ can be visualized as an interpretable curve showing the feature's effect.

**Extended Model (NAM² with pairwise interactions):**

When `include_interactions=True`, optional pairwise shape functions are fitted
on the top-$K$ most important features:

$$
\hat{y} = \beta_0 + \sum_{j=1}^p f_j(x_j) + \sum_{j < k}^K g_{jk}(x_j, x_k)
$$

Each $g_{jk}$ is a separate `_PairwiseFeatureNetwork` mapping $(x_j, x_k)$ through
a 2D Random Fourier basis + Ridge regression onto the additive residual.

**Advantages:**
- Interpretable: Each feature's effect is visualized
- Non-linear: Learns complex shape functions per feature
- Optional interactions: NAM² captures pairwise effects without losing interpretability
- Suitable for small data: RKS reduces parameters
- No optimization difficulties: Backfitting is stable

**Key Parameters:**
```python
n_basis_per_feature = 50    # RKS basis functions
n_iterations = 10           # Backfitting iterations
learning_rate = 0.8         # Step size damping
include_interactions = False # Enable NAM² pairwise g_{jk} terms
max_interaction_features = 5 # Top-K features for pairwise terms
```

**Methods:**
- `predict()`: Standard predictions
- `get_shape_functions(feature_names)`: Extract $f_j$ for visualization
- `get_feature_contributions(X)`: Individual feature effects per sample

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
   
3. **Bayesian Stacking**
   $$
   \alpha \sim \text{Dirichlet}(\mathbf{1})
   $$

**Advantages:**
- Optimal weights (oracle inequality guarantees)
- Prevents overfitting via out-of-fold predictions
- Flexible meta-learner choice
- Theoretically principled

**Cross-Validation Strategy:**
Uses `PanelWalkForwardCV` (public alias of `_WalkForwardYearlySplit`) — a panel-aware
walk-forward splitter that creates annual validation folds in sorted year order.
`min_train_years` (default 7) ensures every fold has a minimum number of training
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
meta_learner_type = 'ridge'     # 'ridge', 'elasticnet', 'bayesian_stacking'
                                # Note: 'bayesian_stacking' is a softmax-of-R²
                                # weighting scheme (temperature-scaled); it is
                                # NOT a full Dirichlet posterior (Yao et al.,
                                # 2018).  The name is kept for backward compat.
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
| **Threshold-only** (`reducer_tree_`) | Variance threshold filter (no scaling) | CatBoost, LightGBM, QuantileRF, PanelVAR, NAM | Preserves original feature structure; StandardScaler removed — CatBoost is scale-invariant and QRF applies its own RobustScaler |

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
   - 6 diverse base models capturing different patterns (tree ×2, linear, panel ×3)
   - Super Learner meta-learning with automatic optimal weighting
   - Distribution-free calibrated uncertainty (conformal prediction)
   - Full 3-tier pipeline optimized for small-to-medium panel data (N < 1000)

2. **Advanced Panel Data Methods**
   - **Panel VAR**: Fixed effects + cross-component dynamics
   - **Quantile RF**: Full distributional forecasts (not just point estimates)
   - **Neural Additive Models**: Interpretable non-linearity with shape functions

3. **Optimal Ensemble Learning**
   - **Super Learner**: Walk-forward meta-learning (`PanelWalkForwardCV`, panel-aware)
   - **Positive Constraints**: Ensures monotonic relationships and stability
   - **Out-of-Fold Training**: Prevents overfitting in meta-learner
   - **Diversity-First**: 6 diverse models outperform 11+ correlated models

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
   - Outlier-robust gradient boosting (CatBoost `MultiRMSE`, LightGBM `regression`)
   - Statistically-principled design for small-to-medium data (N < 1000)
   - Extensible architecture (easy to add new models)
   - Panel-aware walk-forward CV (`PanelWalkForwardCV`) prevents temporal leakage
   - Handles panel structure (entity heterogeneity)

7. **Interpretability**
   - Feature importance (aggregated across models)
   - Neural Additive Models (visualizable shape functions)
   - Meta-weights show model contributions

### 6.2 Limitations

1. **Computational Cost**
   - Trains 6 models + Super Learner + Conformal (moderate speed, ~2–5 min)
   - Optional Phase 4 HP tuning adds ~1–3 min for 20 Optuna trials × 2 models
   - Feature engineering (295 features) increases dimensionality
   - For large-scale production (N > 10,000), consider simplified configurations

2. **Data Requirements**
   - Panel VAR requires ≥3-4 time periods for lag estimation
   - Conformal calibration needs ≥30 calibration samples (guideline)
   - For N < 1000: 5 diverse models optimal (confirmed by statistical theory)

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
   - Model diversity more important than quantity (5-6 diverse > 11+ correlated)
   - Very high correlation between base models reduces ensemble gains
   - For N=819: Removed RF/ET to reduce redundancy with CatBoost GB

### 6.3 Current Capabilities vs. Future Enhancements

#### ✅ Currently Implemented

- ✅ Panel VAR with fixed effects and lag selection
- ✅ Quantile Random Forest (distributional forecasts)
- ✅ Neural Additive Models (interpretable non-linearity)
- ✅ Super Learner (stacked generalization, `PanelWalkForwardCV`)
- ✅ Conformal Prediction (split, CV+, adaptive; OOF residuals, no model re-fit)
- ✅ Comprehensive evaluation suite (7 metrics, per-criterion RMSE tracking)
- ✅ **Phase 1**: 12-block temporal & panel feature engineering (D-01/02/03 fixes, G-01–G-08 new features)
- ✅ **Phase 2**: PLS-supervised compression for linear models, threshold-only track for trees
- ✅ **Phase 3**: LightGBM as independent ensemble member alongside CatBoost
- ✅ **Phase 4**: Per-criterion RMSE CV tracking; optional Optuna one-time HP search for both GB models
- ✅ **Phase 5**: Reversible target transformation (logit/Yeo-Johnson); inverse-transform of all pipeline outputs

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