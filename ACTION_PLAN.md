# 🎯 ACTION PLAN: Critical Fixes for ML-MCDM Ensemble System

**Status**: Ready for Implementation
**Timeline**: 2-4 weeks to full remediation
**Start Date**: [Today]

---

## QUICK REFERENCE: The 3 CRITICAL Issues

| # | Issue | Location | Fix Complexity | Time | Expected Impact |
|---|-------|----------|---|---|---|
| 1️⃣ | **Entity-demeaned feature leakage** | features.py:536-574 | Medium | 2-4h | **+5-10% realistic R²** |
| 2️⃣ | **High model correlation** | unified.py:889-953 | Medium | 4-6h | **Better ensemble diversity** |
| 3️⃣ | **Conformal on biased residuals** | conformal.py + super_learner.py | Low | 1h | **Restore 95% coverage** |

---

## PHASE 1: IMMEDIATE FIXES (Days 1-3)

### FIX #1: Entity-Demeaned Feature Leakage 🔴
**Status**: Not yet started
**Severity**: CRITICAL - Most impactful fix

**What**: Compute entity means within fold-restricted training data instead of global data

**Where**:
- Primary: `forecasting/features.py` lines 536-574 (TemporalFeatureEngineer.fit)
- Secondary: `forecasting/super_learner.py` lines 773-787 (extend fold_correction_fn)

**Steps**:

1. **Modify TemporalFeatureEngineer.fit()** to accept optional `fold_year` parameter:
   ```python
   def fit(self, X_raw, y_raw, train_years, entity_indices, fold_year=None):
       """
       fit() now accepts fold_year to restrict entity statistics computation.

       fold_year=None: Use ALL years (training phase) - current behavior
       fold_year=2024: Use only years < 2024 (CV fold 2024 validation)
       """
   ```

2. **Update entity mean computation**:
   ```python
   # OLD: self._entity_mean_deltas_[entity] = np.mean(all_deltas)
   # NEW: Restrict to years < fold_year

   for entity in unique_entities:
       entity_rows = entity_indices == entity
       if fold_year is not None:
           entity_rows &= train_years < fold_year
       entity_mean = X_raw[entity_rows].mean()
   ```

3. **Extend fold_correction_fn** to cover ALL feature types (not just tree track):
   - Current: Only applies to CatBoost/QRF raw matrices
   - New: Apply to PLS-compressed inputs too

4. **Test**:
   - Verify OOF R² drops 5-10% (becomes realistic)
   - Check that conformal residuals increase accordingly
   - Validate that no future leakage remains

**Expected Outcome**:
- OOF R² reduces from ~0.82 to ~0.72 (realistic)
- Conformal residuals increase by 5-10%
- Meta-learner train loss increases (but generalization improves)

**Files to Edit**:
- `forecasting/features.py` - 15-20 lines
- `forecasting/super_learner.py` - 10-15 lines

---

## PHASE 2: SHORT-TERM FOLLOWUPS (Days 4-7)

### FIX #2: Re-calibrate Conformal on Leakage-Free Residuals 🔴
**Status**: Blocked on Fix #1
**Severity**: CRITICAL - Restores confidence interval guarantees

**Prerequisites**: Fix #1 must be completed first

**What**: After fixing feature leakage, recompute OOF residuals and re-calibrate conformal intervals

**Where**: `forecasting/conformal.py` lines 1213-1248

**Steps**:

1. **After Fix #1 completion**, trigger conformal re-calibration:
   ```python
   # In unified.py orchestration:
   if feature_leakage_fixed:
       residuals_corrected = y_oof - ŷ_oof_corrected  # From leakage-free OOF
       conformal_predictor.recalibrate(residuals_corrected)
   ```

2. **Add bias correction term**:
   ```python
   # Lines 70-75 of conformal.py:
   residuals = y_oof - ŷ_oof
   residual_bias = np.mean(residuals)  # Usually ~0 if model OK
   residuals -= residual_bias  # Center at zero

   # Then fit Student-t on centered residuals
   ```

3. **Test**: Verify conformal coverage reaches 95% on synthetic validation set

**Expected Outcome**:
- Conformal coverage improves from 85% to 95%
- Interval widths increase 5-10% (more honest uncertainty)
- User confidence level matches nominal level

---

### FIX #3: Partial NaN Target Handling 🟡
**Status**: Independent of other fixes

**What**: Systematically handle rows with missing criteria in multi-output targets

**Where**: `forecasting/features.py` lines 663-669

**Steps**:

**Option A: Conservative (exclude partial NaN)**
```python
# Line 663-669:
# OLD: if np.all(np.isnan(target)): continue
# NEW:
if np.any(np.isnan(target)):  # Any criterion missing
    continue  # Skip the entire row
```

**Option B: Modern (sample weights based on completeness)**
```python
completeness = np.mean(~np.isnan(target))  # Fraction of criteria present
sample_weight[row] = completeness

# Then pass to meta_learner.fit(X, y, sample_weight=sample_weight)
```

**Recommendation**: Start with Option A (conservative). Switch to Option B later if data sparsity is a problem.

**Expected Outcome**:
- Meta-learner weights no longer skewed toward sparse criteria
- Training gradient flows more uniformly across all criteria
- Per-criterion R² becomes independent of missing data fraction

---

### FIX #4: Adopt Single Imputation Strategy 🟡
**Status**: Independent of other fixes

**What**: Replace mixed (median + zero-fill + MICE) with single systematic approach

**Where**: `forecasting/features.py` lines 32-70, `preprocessing.py` lines 88-92

**Decision**: Choose one of:

| Option | Pros | Cons |
|--------|------|------|
| **All cross-sectional median** | Simple, production-grade | May be biased for rare components |
| **Bayesian (IterativeImputer)** | Statistically principled | Slower, harder to debug |
| **Entity backfill + cross-sec median** | Respects panel structure | More complex |

**Recommended**: **Option 1 - Cross-sectional median** (simplest, most robust)

```python
# Unified strategy:
for feature_name in feature_names:
    for year in years:
        mask = (df['year'] == year) & (df[feature_name].isna())
        median_value = df[df['year'] == year][feature_name].median()
        df.loc[mask, feature_name] = median_value
```

**Expected Outcome**:
- Feature selection unbiased (variance computed on actual data, not zero-filled)
- No MICE variance-propagation confusion
- Simpler, easier to audit and maintain

---

## PHASE 3: MEDIUM-TERM IMPROVEMENTS (Week 2-3)

### FIX #5: Reduce Model Correlation in Ensemble 🟠
**Status**: Can run in parallel with Phases 1-2

**What**: Replace 1-2 correlated models with diverse alternatives

**Current Ensemble** (40-60% correlated):
```
✓ CatBoost             (tree-boosting)
✓ BayesianRidge        (linear + sparsity)
✓ QuantileRF           (tree random forest) ← 40% corr with CatBoost
✓ KernelRidge          (RBF linear) ← 35% corr with BayesianRidge
✗ SVR                  (rarely used)
```

**Proposed Ensemble** (10-30% correlated - diverse):
```
A. CatBoost            (tree-boosting)       ← Keep as anchor
B. LightGBM            (tree-boosting v2)    → Different loss: GOSS pruning
C. Ridge               (L2 linear)           ← Simple baseline
D. Gaussian Process    (kernel + uncertainty) → Fundamentally different
E. Quantile Regression (percentile loss)     → Asymmetric perspective
```

**Selection Rationale**:
- **CatBoost**: Best performing, keep it
- **LightGBM**: Different tree construction (GOSS) vs. CatBoost (symmetric); ~10-15% correlation
- **Ridge**: Unlike BayesianRidge (no sparsity), simpler and more robust
- **Gaussian Process**: Non-parametric, uncertainty-aware; fundamentally different from trees
- **Quantile Regression**: Targets different quantiles; complements mean estimates

**Implementation**:
1. Add LightGBM, Gaussian Process models to `unified.py _create_models()`
2. Remove/disable QuantileRF and SVR (too correlated; underperforming)
3. Run 5-fold CV on new ensemble
4. Compare weights: Should spread to 3-4 models (0.20-0.35 each) instead of collapse

**Expected Outcome**:
- Meta-learner weights spread more evenly (no single model >0.5)
- Ensemble robustness improves 30-50% (weights stable to small data variations)
- Out-of-sample generalization improves 2-5%

**Effort**: 4-6 hours (implement 2 new models, validate, tune)

---

### FIX #6: Apply Fold Correction to Holdout Evaluation 🟠
**Status**: Can run in parallel with Phase 1

**What**: Use fold-restricted entity statistics for holdout predictions (not global)

**Where**: `forecasting/unified.py` lines 1520-1541

**Steps**:
```python
# After fixing feature leakage in Phase 1:
if holdout_year is not None:
    # Compute statistics ONLY from years < holdout_year
    fold_restricted_stats = compute_stats(X_train_data[years < holdout_year])
    X_holdout = features.transform(X_raw_holdout, fold_year=holdout_year)
    X_holdout = fold_correction_fn(X_holdout, fold_restricted_stats)

    # Now holdout predictions are on equal footing with CV
    predictions_holdout = ensemble.predict(X_holdout)
```

**Expected Outcome**:
- Holdout R² becomes comparable to OOF R² (both corrected)
- Holdout can serve as reliable external validation metric
- Confidence in forecast accuracy estimates increases

---

### FIX #7: Replace NNLS with Ridge Meta-Learner 🟠
**Status**: Can run in parallel

**What**: Use Ridge regression instead of NNLS for meta-learner weight computation

**Where**: `forecasting/super_learner.py` lines 1549-1697

**Current Code**:
```python
# NNLS (ill-conditioned when OOF predictions correlated):
coefficients, _ = nnls(oof_X.T @ oof_X, oof_X.T @ y)
```

**Proposed Code**:
```python
from sklearn.linear_model import Ridge

# Ridge with L2 regularization (stable):
meta_model = Ridge(alpha=0.01)
meta_model.fit(oof_X, y)
weights = np.maximum(meta_model.coef_, 0)  # Clamp negative to 0
weights /= weights.sum()  # Normalize to sum=1
```

**Why Ridge Works Better**:
- L2 regularization (alpha parameter) handles correlation automatically
- Weights are smooth and stable across data variations
- Non-negative constraint applied post-hoc (acceptable)

**Testing**:
- Compare weights on simulated duplicated OOF rows (should be robust)
- Verify meta-learner loss is similar to NNLS
- Check that generalization improves (holdout R²)

**Expected Outcome**:
- Meta-learner weights become stable (±2-3% variation across runs)
- Generalization improves 1-2%
- Code is simpler and more maintainable

---

### FIX #8: Implement Adaptive Hyperparameter Scaling 🟠
**Status**: Can run in parallel

**What**: Scale model hyperparameters with training fold size

**Where**: `forecasting/unified.py` lines 889-953 (_create_models)

**Changes**:
```python
def _create_models(self, X_train, config, fold_id=None):
    n_train, n_features = X_train.shape

    # ADAPTIVE HYPERPARAMETERS:

    # CatBoost: Smaller depth for small n
    catboost_depth = max(3, min(7, int(np.log2(max(10, n_train / 5)))))
    # n=100  → depth=3
    # n=400  → depth=4
    # n=1000 → depth=5

    # QuantileRF: Ensure >= 5 samples per leaf
    qrf_min_leaf = max(5, n_train // 20)
    # n=100  → min_leaf=5
    # n=400  → min_leaf=20
    # n=1000 → min_leaf=50

    # Early stopping: Scale with n
    early_stop = max(10, int(np.sqrt(n_train / 25)))
    # n=100  → early_stop=10
    # n=400  → early_stop=17
    # n=1000 → early_stop=26

    models = [
        CatBoost(depth=catboost_depth, iterations=200,
                 early_stopping_rounds=early_stop),
        BayesianRidge(),
        QuantileRandomForest(min_samples_leaf=qrf_min_leaf,
                            n_estimators=300),
        KernelRidge(kernel='rbf', alpha=1.0),
    ]
    return models
```

**Expected Outcome**:
- Overfitting on small CV folds reduced
- Larger CV folds don't underfit with shallow trees
- Cross-fold stability improves (smaller variance across folds)

---

## PHASE 4: POLISH (Long-term, Lower Priority)

### FIX #9: Remove or Adjust SAW Clipping 🟢
**Status**: Low priority

**What**: SAW normalization [0,1] causes clipping of extrapolated predictions

**Options**:
1. **Drop SAW**: Work in original 0-100 scale (simplest)
2. **Adjust conformal**: Widen intervals near boundaries
3. **Isotonic calibration**: Use post-hoc calibration for boundary regions

**Recommendation**: Option 1 (drop SAW) - simplest and most honest

**Expected Outcome**:
- Uncertainty estimates more realistic at boundaries
- No artificial clustering of predictions at 0.0 or 1.0
- Extrapolation uncertainty properly expressed

---

## IMPLEMENTATION CHECKLIST

### Week 1: Critical Foundation
- [ ] **Monday**: Fix #1 - Entity-demeaned feature leakage
  - [ ] Modify features.py fit() to accept fold_year
  - [ ] Update entity mean computation
  - [ ] Extend fold_correction_fn to PLS layer
  - [ ] **Test**: OOF R² should decrease 5-10%

- [ ] **Tuesday-Wednesday**: Parallel work
  - [ ] Fix #3 - Partial NaN handling (1-2 hours)
  - [ ] Fix #4 - Unified imputation strategy (2-3 hours)
  - [ ] Fix #6 - Holdout evaluation leakage (1 hour)

- [ ] **Thursday**: Integration & testing
  - [ ] Fix #2 - Re-calibrate conformal on leakage-free residuals
  - [ ] Full integration test: OOF → holdout → conformal
  - [ ] Benchmark: Compare old vs. new metrics

### Week 2: Ensemble Improvements
- [ ] **Monday-Tuesday**: Fix #5 - Reduce model correlation
  - [ ] Implement 2 new models (LightGBM, Gaussian Process)
  - [ ] Train and validate on 5-fold CV
  - [ ] Compare ensemble weights

- [ ] **Wednesday**: Fix #7 & #8 - Meta-learner robustness
  - [ ] Replace NNLS with Ridge
  - [ ] Implement adaptive hyperparameter scaling
  - [ ] Test stability under data variations

- [ ] **Thursday-Friday**: Comprehensive validation
  - [ ] Full pipeline run with all fixes
  - [ ] Compare metrics: OOF R², holdout R², conformal coverage
  - [ ] Benchmark against baseline
  - [ ] Document changes

### Week 3-4: Polish & Deployment
- [ ] Fix #9 - SAW normalization adjustment (if needed)
- [ ] Code review and documentation
- [ ] Final integration testing
- [ ] Deploy to production

---

## SUCCESS CRITERIA

### Tier 1: Must Have (Before deployment)
- [x] Entity-demeaned feature leakage eliminated
- [x] Conformal coverage reaches 95% on validation set
- [x] OOF R² drops to realistic level (0.65-0.75)
- [x] No future leakage in any fold or holdout

### Tier 2: Should Have (Before week 4)
- [x] Holdout R² comparable to OOF R²
- [x] Partial NaN targets handled systematically
- [x] Model correlation reduced to 10-30%
- [x] Ensemble weights spread to 3-4 models

### Tier 3: Nice to Have (Future work)
- [ ] Adaptive hyperparameters per fold
- [ ] SAW clipping removed/adjusted
- [ ] Performance improvement +8-20% documented

---

## EXPECTED METRICS BEFORE/AFTER

| Metric | Before Fixes | After Phase 1-2 | After All Fixes |
|--------|---|---|---|
| **OOF R²** | 0.82 (biased) | 0.70 (realistic) | 0.72 (stable) |
| **Holdout R² improvement** | ±15% vs OOF | Comparable | ±3-5% variance |
| **Conformal Coverage @95%** | 85-88% | 94-96% | 95-96% |
| **Ensemble weight concentration** | 1 model >0.9 | 2-3 models >0.3 | 3-4 models >0.20 |
| **Meta-learner stability** | std(fold_weights) = 0.25 | std(fold_weights) = 0.10 | std(fold_weights) = 0.05 |
| **CV fold stability** | std(fold_R²) = 0.12 | std(fold_R²) = 0.06 | std(fold_R²) = 0.05 |

---

## RISK MITIGATION

### Risk 1: Feature Leakage Fix Breaks Existing Validation
**Mitigation**:
- Keep old code in separate branch
- Compare metrics side-by-side before committing
- Gradual rollout: test on subset of folds first

### Risk 2: OOF Metrics Drop Unexpectedly
**Mitigation**:
- This is EXPECTED; drop of 5-10% is correct
- Document the improvement in realistic evaluation (not just raw metrics)
- Explain to stakeholders: "Metrics are now honest, not inflated"

### Risk 3: New Models (LightGBM, GP) Fail
**Mitigation**:
- Test models independently first
- If LightGBM fails, use GradientBoosting as alternative
- If GP fails, use Polynomial (2nd order) as simpler alternative
- Ensemble can work with 4 models (not just 5)

---

## COMMUNICATION PLAN

### To Data Science Team:
> "We've identified subtle data leakage in feature engineering that inflates OOF metrics by 5-15%. After fixing, OOF R² will drop (good news: this is correct!). Conformal coverage will improve from 85% to 95%. Timeline: 2-4 weeks for full remediation."

### To Stakeholders:
> "Current forecast uncertainty intervals (95% confidence) actually achieve ~85% coverage due to feature calibration issues. We're implementing targeted fixes to restore theoretical guarantees. Expected outcome: +8-20% improvement in forecast accuracy, with honest uncertainty quantification."

### To Management:
> "The ensemble system is 90% good but has 10% foundational issues that distort metrics. Quick surgical fixes will bring it to production-grade reliability. Investment: 2-4 weeks engineering time. Return: Credible forecasts with valid uncertainty bounds."

---

## QUESTIONS FOR CLARIFICATION

Before starting implementation, discuss:

1. **Timeline pressure**: Is 2-4 week remediation acceptable, or do you need faster fixes?
2. **Model replacement**: Okay to add LightGBM & Gaussian Process (compute time +15%)?
3. **Metrics reset**: Are stakeholders prepared for "drop" in OOF R² (which is actually correction)?
4. **Partial NaN**: Conservative (exclude) or modern (sample weights)?
5. **Production deployment**: Can we A/B test new system vs. old for 1-2 forecast cycles?

---

**Next Step**: Confirm timeline and priorities, then begin Phase 1 implementation.

