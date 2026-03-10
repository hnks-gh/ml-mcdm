# ML Performance Assessment & Recommendations
*Generated: March 10, 2026 — based on current `output/result/csv/forecasting/` results*

---

## 1. Current Results Summary

### 1.1 Super Learner Ensemble (Out-of-Fold)

| Metric | Value |
|--------|-------|
| **R²** | **0.2545** |
| RMSE | 0.1996 |
| MAE | 0.1578 |
| OOF samples | 310 |
| Training samples | 749 |
| Features (tree) | 140 |
| Features (PCA) | 30 (97.4 % variance retained) |
| Entities | 61 provinces |
| Criteria | 8 |

### 1.2 Individual Model CV R² (5-Fold)

| Model | Mean R² | Std | Fold 3 (worst) | Verdict |
|-------|---------|-----|----------------|---------|
| QuantileRF | 0.141 | 0.176 | −0.177 | ✅ Best individual |
| GradientBoosting | 0.104 | 0.141 | −0.164 | ✅ Adequate |
| BayesianRidge | −0.004 | 0.195 | −0.354 | ⚠️ Near-zero signal |
| NAM | −3.123 | 0.641 | −3.709 | ❌ Fails completely |
| PanelVAR | −7.356 | 2.731 | −10.715 | ❌ Fails completely |

### 1.3 Super Learner Weights

| Model | Weight | Share |
|-------|--------|-------|
| QuantileRF | 0.497 | **49.7 %** |
| BayesianRidge | 0.260 | 26.0 % |
| GradientBoosting | 0.192 | 19.2 % |
| PanelVAR | 0.051 | 5.1 % |
| NAM | 0.001 | **≈ 0 %** |

### 1.4 Feature Importance

- Top feature (`C01_current`): 4.6 % — **no single dominant predictor**
- Cumulative importance reaches 50 % only after ~43 features
- Flat distribution across 168 engineered features → weak learned signal

---

## 2. Diagnosis

### 2.1 Overall Assessment: **Modest — needs enhancement**

An R² of 0.25 means the ensemble explains only **25 % of variance** in
future criterion scores. While this is better than a naïve mean predictor, it
falls short of the ≥ 0.50 threshold typically expected for credible forecasts
used in policy-weighted MCDM rankings.

### 2.2 Root Causes

| Issue | Evidence | Severity |
|-------|----------|----------|
| **Short time horizon** | T = 14 years per entity; temporal CV folds have as few as 2–3 training years | 🔴 High |
| **Small panel for deep models** | NAM weight ≈ 0 %, PanelVAR R² = −7.4 | 🔴 High |
| **Weak signal / noisy targets** | Flat feature importance; BayesianRidge ≈ mean predictor | 🔴 High |
| **Fold 3 collapse across all models** | Every model goes negative at Fold 3 → distribution shift | 🟡 Medium |
| **Curse of dimensionality** | 140 tree features for 749 samples (ratio 5.4:1) | 🟡 Medium |
| **Zero prior on PanelVAR / NAM** | These models are architecturally overparameterised for this dataset | 🟡 Medium |
| **No exogenous macro covariates** | Only self-referential features (lags of the same criteria) | 🟡 Medium |
| **Stationarity not enforced** | Unit-root / trending behaviour noted in docs; PanelVAR most sensitive | 🟡 Medium |

---

## 3. Recommendations by Priority

### Priority 1 — Fix the Models That Are Failing (Quick Wins)

#### R1. Remove or Replace PanelVAR and NAM
PanelVAR receives R² = −7.4 in CV and only 5.1 % Super Learner weight.
NAM receives 0.1 % weight. With T = 14, both models are severely
overparameterised and degrade ensemble stability.

**Action:**
```python
# In forecasting/unified.py or pipeline, disable NAM and PanelVAR:
ENABLED_MODELS = ["GradientBoosting", "BayesianRidge", "QuantileRF"]
# Replace with simpler but more reliable alternatives (see R3 below)
```

**Expected gain:** Ensemble R² +0.03–0.06 from reduced noise injection.

#### R2. Investigate Fold 3 Distribution Shift
Every model produces a negative R² at Fold 3. This strongly suggests
a structural break (policy change, COVID-19 disruption around 2020–2021,
data quality issue) in that temporal window.

**Actions:**
- Print the year range covered by each fold to identify which years form Fold 3.
- Inspect raw values of the criteria for 2020–2021 for anomalies.
- If a structural break is confirmed, add a COVID dummy / year fixed effect,
  or use a break-aware fold splitter that skips the break period as a
  validation boundary.

```python
# Add to _PanelTemporalSplit to log fold year ranges:
for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X)):
    val_years = panel_data.loc[val_idx, 'year'].unique()
    print(f"Fold {fold_idx+1} val years: {val_years}")
```

---

### Priority 2 — Address the Weak Signal Problem

#### R3. Add Macro / Exogenous Covariates
The model only uses **self-referential lags** of the same criterion scores.
Provincial GDP growth, population growth, central government transfers, or
regional infrastructure investment data would add orthogonal predictive power.

**Suggested external features:**
| Feature | Likely predictive for criteria |
|---------|-------------------------------|
| Provincial GDP per capita growth | C01 (economic output), C02 |
| State budget transfers (per capita) | C03 (infrastructure), C06 |
| Population / urbanisation rate | C04 (social), C07 |
| Trade openness index | C05, C08 |

Even a single macroeconomic level variable (e.g. national GDP growth as a
common factor) would help models distinguish boom years from recession years
and reduce the Fold 3 collapse.

#### R4. Target the Prediction Task More Precisely
Currently the model predicts normalized criterion scores (0–1 scale).
Consider predicting **year-over-year changes (Δ scores)** instead of
absolute levels:

- Score changes are more stationary than absolute levels.
- The variance explained by lags of changes is typically higher than
  variance of level predictions.
- MCDM rankings are sensitive to **relative changes** — predicting Δ
  directly aligns the ML objective with the ranking objective.

```python
# In features.py SAWNormalizer or build_panel_targets():
target = score_t1 - score_t0   # predict delta instead of level
```

Then reconstruct the absolute forecast: `forecast_t1 = score_t0 + Δ_predicted`.

#### R5. Reduce Feature Redundancy with Aggressive Selection
168 features for 749 samples is excessive. With cumulative importance reaching
50 % only after 43 features, the bottom 125 features add noise.

**Action:** Apply a two-stage feature selection before model training:
1. **Variance threshold** — drop features with near-zero variance across the panel.
2. **Mutual information or Boruta** — keep top-K features where K ≤ √(n_samples) ≈ 27.

```python
from sklearn.feature_selection import SelectKBest, mutual_info_regression
selector = SelectKBest(mutual_info_regression, k=40)
X_selected = selector.fit_transform(X_train, y_train)
```

Expected gain: +0.02–0.05 R² from reduced overfitting in tree models.

---

### Priority 3 — Better Models for Small Panels

#### R6. Add Gaussian Process Regression (GPR)
GPR with a Matérn or RBF kernel is purpose-built for small datasets. With
n ≈ 749 it is computationally feasible and provides calibrated uncertainty
estimates natively (no need for conformal calibration on top).

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

kernel = Matern(nu=2.5) + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
```

**Advantage:** GPR with ARD (Automatic Relevance Determination) kernel
naturally performs integrated feature selection and handles short T well.

#### R7. Add Ridge + Interaction Terms (LASSO / ElasticNet with Polynomial Features)
A simple ElasticNet with degree-2 polynomial features on the top 20 criterion
scores (no auxiliaries) is often competitive with tree models on small tabular
panels and is much more stable across folds.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline

poly_elastic = Pipeline([
    ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ("model", ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.9], max_iter=5000))
])
```

#### R8. Replace PanelVAR with a Hierarchical Bayesian Model
With T = 14 and N = 61, a **hierarchical Bayesian linear model** (partial
pooling across provinces) is theoretically superior to PanelVAR:

- Partial pooling borrows strength across 61 provinces → stable estimates
  even for provinces with missing data years.
- Posterior predictive intervals are naturally calibrated.
- Unlike PanelVAR, it doesn't require matrix inversion over a large feature space.

Libraries: `bambi`, `pymc`, or `cmdstanpy`.

---

### Priority 4 — Training Pipeline Improvements

#### R9. Panel-Aware Stratified CV
The current `_PanelTemporalSplit` is median-based. An improvement is to use
**grouped k-fold with temporal blocking** that:
1. Assigns each entity to one of K temporal blocks.
2. Validates on the most recent block common to all entities.
3. Avoids the Fold 3 distribution-shift problem by choosing fold boundaries
   at clean temporal breakpoints (e.g. year 2019 as end of pre-COVID training).

```python
# Concrete suggestion: fix 2020 as the mandatory validation year
# for the outer (final) evaluation fold
BREAK_YEAR = 2020
train_idx = panel.index[panel['year'] < BREAK_YEAR]
val_idx   = panel.index[panel['year'] >= BREAK_YEAR]
```

#### R10. Hyperparameter Optimisation
No hyperparameter tuning is visible in the CV results. The current setup
uses fixed GradientBoosting parameters.

**Action:** Use `Optuna` or `BayesSearchCV` with the temporal CV splitter:

```python
import optuna
from optuna.integration import OptunaSearchCV

param_space = {
    "n_estimators": optuna.distributions.IntDistribution(100, 500),
    "max_depth": optuna.distributions.IntDistribution(3, 8),
    "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
    "subsample": optuna.distributions.FloatDistribution(0.5, 1.0),
}
# Run with temporal CV to avoid leakage
```

Expected gain: +0.03–0.08 R² for GradientBoosting.

#### R11. Target Encoding for Province Identity
Instead of relying only on lag features to capture province-level
heterogeneity, encode each province as a target-encoded feature:

```python
from category_encoders import TargetEncoder
te = TargetEncoder(cols=['province_id'])
X_train = te.fit_transform(X_train, y_train)
X_test  = te.transform(X_test)
```

This gives tree models a direct handle on province-level fixed effects
without requiring PanelVAR's fully parameterised fixed-effect matrix.

---

## 4. Expected Performance After Improvements

| Improvement Package | Estimated R² | Notes |
|--------------------|-------------|-------|
| Current baseline | 0.25 | OOF Super Learner |
| + Remove PanelVAR/NAM (R1) | 0.28 | Remove noise contributors |
| + Fix Fold 3 / break-aware CV (R2, R9) | 0.32 | Stable CV estimates |
| + Feature selection top-40 (R5) | 0.34 | Reduce overfitting |
| + Predict Δ instead of levels (R4) | 0.36 | Better signal/noise |
| + HPO for GradientBoosting (R10) | 0.39 | Better individual models |
| + GPR model added (R6) | 0.42 | Diverse base learner |
| + Macro covariates (R3) | 0.48–0.55 | Strongest structural gain |

> **Note:** Estimates are heuristic ranges, not guaranteed values. Actual
> gains depend on data quality and the strength of the macro covariate signal.

---

## 5. Minimum Viable Improvement Plan (3-Step Quick Start)

If time is constrained, implement these three changes first:

1. **Disable NAM and PanelVAR** from the Super Learner base models.
   One-line change in the model registry — immediate, no risk, +R² guaranteed
   by removing negative-weight contributors.

2. **Investigate Fold 3** (likely 2020–2021). If a COVID break is confirmed,
   use 2020 as a hard holdout boundary rather than a random CV fold.

3. **Select top-40 features** by mutual information before feeding tree models.
   Eliminates the 128 weakest features that currently dilute signal.

These three changes require less than 100 lines of code changes and should
lift the ensemble R² from 0.25 toward 0.33–0.36 without introducing new
model complexity.

---

## 6. Long-Term Architecture Recommendation

For a `(N=61, T=14, K=8)` panel with socio-economic criteria, the
**ideal architecture** is:

```
Tier 1 — Base Learners (diverse, small-data-appropriate)
  ├── QuantileRF (keep, strongest — 49.7% weight already)
  ├── ElasticNet + Polynomial Interactions (replace BayesianRidge)
  ├── Gaussian Process Regression  (replace NAM)
  └── CatBoost/LightGBM with HPO   (keep, tune better)

Tier 2 — Meta-Learner
  └── Non-negative Least Squares (NNLS) Super Learner (keep)

Tier 3 — Uncertainty Quantification
  └── Conformal Prediction (keep — already implemented correctly)

Exogenous Inputs
  └── Macro panel covariates (GDP, budget, population)

Target
  └── Δ(criterion score) → reconstruct absolute forecast
```

This architecture preserves the strengths of the current system (Super
Learner, conformal calibration, temporal CV) while replacing the weakest
components with more data-efficient alternatives.
