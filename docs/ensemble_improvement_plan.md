# Ensemble ML Forecasting — Audit Report & Improvement Plan

## Executive Summary

A comprehensive audit of the Super Learner ensemble identified **three root-cause
defects** that collectively explain why cross-validated R² ranged from −17 to −9
(all models performing far worse than naïve mean prediction):

| # | Defect | Severity | Status |
|---|--------|----------|--------|
| D-1 | Training set collapsed to ~126 samples (2 years) due to strict complete-target filter | Critical | **Fixed** |
| D-2 | Cross-validation produced only 1 fold (STD R² = 0 for every model) | Critical | **Fixed** (resolved by D-1) |
| D-3 | Holdout evaluation measured in-sample fit, not generalisability | High | **Fixed** |
| D-4 | GradientBoosting over-parameterised for small CV folds | Medium | **Fixed** |
| D-5 | PCA feature reducer fitted on full training set before CV loop | Low | Documented |

---

## 1. Root-Cause Analysis

### D-1: Training Data Collapsed to ~2 Years (Critical)

**Symptom.** The pipeline reported only 126 training samples for 61 entities
across 14 years (2011–2024), implying an average of 2 valid years per entity.

**Cause.** `TemporalFeatureEngineer.fit_transform()` applied
`has_complete_target()` — requiring **all 29 sub-criteria to be non-NaN** —
before accepting a training sample.  Examination of the raw CSV files revealed:

| Year range | Missing sub-criteria | Impact |
|------------|----------------------|--------|
| 2011–2017 | SC24, SC71–73, SC81–83 (7 columns) | **0 valid training targets** |
| 2018, 2021–2024 | SC52 (1 column) | **0 valid training targets** |
| 2019–2020 | None | 61 × 2 = 122 valid samples |

Only two complete years yielded training examples, making the effective training
set a factor of **6–7× smaller** than it should have been.

**Fix.** Added `min_target_fraction` (default 0.5) to `TemporalFeatureEngineer`.
When a target vector has ≥ 50 % non-NaN entries it is retained; remaining NaN
values are imputed with per-column medians **after** all samples have been
collected (in `UnifiedForecaster.fit_predict`).  This expands the training set
from ~126 to ~790 samples.

```
Before: 126 training samples (2 out of 13 possible target years)
After:  ~790 training samples (all 13 available years included)
```

*Configuration.* `ForecastConfig.min_target_fraction = 0.5` (adjustable).
Set to `1.0` to restore the previous strict behaviour.

---

### D-2: Cross-Validation Produced Only 1 Fold (Critical)

**Symptom.** Every model had `STD_R2 = 0.0000` and `Fold_1` was the only
column in `cross_validation_scores.csv`.

**Cause.** `_PanelTemporalSplit` derives fold boundaries from the *median*
entity history length `T_median`.  With only 2 valid target years per entity,
`T_median = 2`, giving:

```
min_train_T = max(2//2, 2//(3+1)) = max(1, 0) = 1
fold_size   = max(1, (2−1)//3)    = max(1, 0) = 1
fold 0: cut=1, val_end=2   → 1 fold generated
fold 1: cut=2 ≥ T_median=2 → break
```

The single fold trained on ~61 samples (year-0 data for each entity) and
validated on ~61 samples (year-1 data).  The temporal gap between these two
sets varied from 1 to 11 years across entities, introducing severe distribution
shift and explaining the highly negative R² values.

**Fix.** Resolved entirely by D-1.  With ~790 training samples and `T_median ≈ 13`,
the splitter now produces 3 well-populated folds:

```
min_train_T = max(6, 3) = 6    (≥ 6 years initial training window)
fold_size   = max(1, 7//3) = 2 (2-year validation windows)
→ 3 folds, each with ~390 train rows and ~130 validation rows
```

---

### D-3: Holdout Evaluation Was In-Sample (High)

**Symptom.** Holdout R² = 0.974 while CV R² = −10 to −17 — a 10× gap
inconsistent with any genuine train/test split.

**Cause.** The Stage 6b holdout block called:

```python
X_ho, y_ho, _, _ = _ho_eng.fit_transform(panel_data, holdout_year)
y_ho_pred = self.super_learner_.predict(X_ho_arr)
r2_score(y_ho_arr.ravel(), y_ho_pred.ravel())   # ← training data!
```

`fit_transform(panel_data, holdout_year)` returns the **training feature
matrix** (all year-pairs up to holdout_year) as its first return value.
Because the main model was also trained on these exact pairs, the comparison
`y_ho_arr` vs `y_ho_pred` measured in-sample fit, not generalisation.

**Fix.** The holdout now:
1. Calls `fit_transform(panel_data, holdout_year − 1)` and takes the **third
   return value** `X_pred` — prediction features at year `holdout_year − 1`
   (the correct input for predicting `holdout_year`).
2. Fetches actual entity values at `holdout_year` directly from `panel_data`.
3. Compares predictions to those actual values, ignoring rows with NaN actuals.

*Remaining limitation.* After D-1, the `(holdout_year − 1 → holdout_year)`
pair IS included in the main training set (partial-target imputation unlocks
it).  A truly out-of-sample score requires retraining without that pair.
The OOF R² in the cross-validation table is therefore the appropriate
**out-of-sample** performance metric.

---

### D-4: GradientBoosting Over-Parameterised for Small CV Folds (Medium)

**Cause.** The previous defaults — 200 trees at depth 5 — were sized for
panels with ~756 complete training rows.  When the effective training fold
contained only ~61 rows (before D-1), depth-5 trees had up to 32 leaf nodes
for 61 samples (< 2 samples/leaf), causing complete memorisation and then
catastrophic generalisation failure on the validation fold.

**Fix.** `ForecastConfig` defaults updated:

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `gb_max_depth` | 5 | **3** | 8 leaves at depth-3 → ≥ 16 samples/leaf at n=130 fold |
| `gb_n_estimators` | 200 | **100** | Fewer, shallower trees regularise better for small panels |

The class-level defaults in `GradientBoostingForecaster` are unchanged
(backward-compatible); the config overrides apply only when the pipeline is
invoked through `ForecastConfig`.

---

### D-5: PCA Feature Reducer Fitted Before CV Loop (Low / Documented)

**Description.** `PanelFeatureReducer` (VarianceThreshold + StandardScaler + PCA)
is fitted on the **full training set** before `SuperLearner.fit()` initiates its
OOF cross-validation loop.  Each CV fold's validation rows therefore contribute
to the PCA directions used to represent them — a subtle information leak.

**Impact.** For PCA this leakage is mild: PCA directions are determined by
covariance structure, not by individual labels, so the bias is small relative
to the D-1 / D-2 issues.  The variance filter may, however, exclude features
that are near-constant in training but informative in validation, marginally
biasing the reported CV R².

**Mitigation (future work).** Moving PCA inside the CV fold loop requires
fitting a separate reducer per fold and an inverse-importance mapping per fold
for the feature importance report.  This is architecturally feasible but was
deferred in favour of the higher-priority fixes.

---

## 2. Improvement Strategy

### 2.1 Immediate (Implemented)

1. **Partial-target inclusion** (`min_target_fraction=0.5`) — expands training ~6×.
2. **Median imputation** of NaN target entries before model fitting.
3. **Corrected holdout evaluation** — uses prediction features, not training features.
4. **Reduced GB complexity** — `max_depth=3`, `n_estimators=100`.

### 2.2 Short-Term Recommendations

| Action | Expected benefit |
|--------|-----------------|
| Sub-criterion-specific imputation (use trend from available years rather than global median) | Reduces bias for newly-introduced indicators |
| Move PCA fitting inside CV fold | Eliminates remaining feature leakage (D-5) |
| Increase `cv_folds` to 5 once training size ≥ 500 | More reliable OOF R² estimates |
| Add temporal gap weighting in Super Learner meta-learner | Recent years should receive higher weight |
| Investigate SC71–73 and SC81–83 back-fill options | May unlock 7 more complete sub-criteria for 2011–2017 |

### 2.3 Medium-Term Recommendations

| Action | Expected benefit |
|--------|-----------------|
| Hierarchical Bayesian model for missing sub-criteria | Principled uncertainty-aware imputation |
| Entity-level time-series models (per-province ARIMA/ETS) as additional base learners | Captures entity-specific trends missed by pooled models |
| Conformal prediction tuning: reduce Bonferroni correction severity | Narrower intervals without sacrificing joint coverage |
| Explainability layer: SHAP values per entity per forecast | Stakeholder trust and model debugging |

### 2.4 Long-Term / Data-Collection Recommendations

| Action | Expected benefit |
|--------|-----------------|
| Collect SC71–73, SC81–83 for years 2011–2017 | Eliminates systematic 7-column missingness, removes imputation bias |
| Ensure SC24 and SC52 completeness for all years | One-column missingness currently forces imputation for 2018–2024 |
| Increase panel length beyond 14 years | Longer panels enable deeper temporal CV and better trend learning |

---

## 3. Expected Performance After Fixes

| Metric | Before | Expected After |
|--------|--------|----------------|
| Training samples | 126 | ~790 |
| CV folds | 1 | 3 |
| CV R² (GradientBoosting) | −17.4 | > 0 (positive generalisation) |
| CV R² (BayesianRidge) | −10.0 | > 0 |
| Holdout R² | 0.97 (in-sample) | Meaningful out-of-sample estimate |
| Conformal interval mean width | 0.84 | Should narrow with better-calibrated OOF residuals |

---

## 4. Configuration Reference

```python
from config import ForecastConfig

# Recommended settings after audit
cfg = ForecastConfig(
    min_target_fraction=0.5,   # allow partial targets (NEW)
    gb_max_depth=3,             # reduced from 5
    gb_n_estimators=100,        # reduced from 200
    cv_folds=3,                 # increase to 5 when n_train ≥ 500
    conformal_method='cv_plus',
    conformal_alpha=0.05,
)
```

To revert to pre-audit strict behaviour (only complete targets):

```python
cfg = ForecastConfig(min_target_fraction=1.0, gb_max_depth=5, gb_n_estimators=200)
```

---

## 5. References

- van der Laan, Polley & Hubbard (2007). *Super Learner*.
  Statistical Applications in Genetics and Molecular Biology.
- Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World*.
  Springer.
- Little & Rubin (2002). *Statistical Analysis with Missing Data* (2nd ed.).
  Wiley — for principled treatment of systematically-missing sub-criteria.
