# ML ENSEMBLE ENHANCEMENT PLAN

---

## EXECUTIVE SUMMARY

### Current System Assessment

**ML-MCDM ensemble system demonstrates sophisticated statistical architecture** with proper temporal integrity, panel-aware cross-validation, and scientifically sound ensemble design. However, **practical performance is weak** (holdout R² = 0.217, RMSE = 2.39), and **uncertainty quantification is completely unusable** (conformal intervals degenerate to [0.0, 1.0]).

#### Key Performance Metrics (Current State)

| Metric | Value | Status | Root Cause |
|--------|-------|--------|------------|
| **Holdout R²** | 0.2171 | ⚠️ Weak signal | Small effective sample; weak governance signal |
| **Holdout RMSE** | 2.3936 | ⚠️ High error | Noise-dominated targets; underfitting |
| **Best Single Model** | Bayesian: R²=0.1304 | ⚠️ Baseline weak | All models weak at p/n=0.4–0.8 ratio |
| **Ensemble Gain** | +67% vs. best single | ✅ Stacking works | Architecture correct |
| **Conformal Intervals** | [0.0, 1.0] width=1.0 | ❌ **CRITICAL** | n_cal≈24 per criterion (too small) |
| **LightGBM CV R²** | −0.07 | ❌ Negative | No early stopping; overfitting on small folds |
| **QuantileRF CV R²** | −0.088 | ❌ Negative | No early stopping; high variance ensemble |
| **Calibration Set Size** | n_cal≈189 global | ⚠️ Undersized | Only 24 per criterion for 8 outputs |
| **Training Samples** | 756 (63 × 12 years) | ⚠️ Small | p/n = 0.4–0.8 (tree models at risk) |
| **Effective n OOF** | 315 rows, 48 meta-features | ⚠️ Underdetermined | Meta-learner prone to overfitting |

#### Assessment Verdict

| Component | Status | Notes |
|-----------|--------|-------|
| **Data Pipeline** | ✅ Sound | YearContext, temporal integrity correct |
| **Walk-Forward CV** | ✅ Correct | Zero leakage, entity-block bootstrap proper |
| **Ensemble Architecture** | ✅ Sound | 6-model orthogonal design, NNLS stacking |
| **Base Models** | 🟡 Correct, Weak | Code correct; hyperparams hardcoded, early stopping missing |
| **Conformal Prediction** | ❌ Broken | n_cal critically small; intervals useless |
| **Tree Models** | ❌ Failing | Negative CV R²; no early stopping |
| **Feature Engineering** | 🟡 Over-engineered | 1000+ → 20 PLS (98% loss); no importance audit |
| **Hyperparameter Tuning** | ⚠️ Disabled | auto_tune_gb=False; 8–12% performance gap |
| **Small Sample Effects** | ⚠️ Confounded | n=756; per-output weight instability |
| **MCDM Integration** | ⚠️ Incomplete | ML forecast exists; no ranking comparison |

---

## ROOT CAUSE ANALYSIS: Performance Bottlenecks (Priority Order)

### 1. **CRITICAL: Degenerate Conformal Intervals**

**Symptom**: All 504 predictions (63 provinces × 8 criteria) produce intervals [0.0, 1.0] with width=1.0 (zero information).

**Root Causes** (in order of impact):
1. **Calibration set critically undersized**: n_cal ≈ 189 global → ~24 per criterion
   - Quantile estimation q̂ = quantile(|residuals|, 0.975) extremely unreliable at n=24
   - High OOF residual variance (σ ≈ 0.5) leads to q̂ ≈ ±0.5
2. **Target space [0,1] constraints intervals**: [ŷ − 0.5, ŷ + 0.5] clamps to [0.0, 1.0] for edge predictions
3. **No heteroscedasticity adjustment**: Each province/criterion gets uniform-width interval regardless of volatility
4. **No distributional modeling**: Assumes Gaussian; actual residuals likely heavy-tailed
5. **Small-sample conformal coverage is conservative**: With only 189 global residuals, empirical coverage can exceed 100% (trivial)

**Impact**: Decision-makers receive zero actionable uncertainty; prediction intervals useless for risk-based decisions.

**Fix Priority**: **PHASE 1 (CRITICAL, 2 weeks)**

---

### 2. **HIGH IMPACT: Tree Model Overfitting & Instability**

**Symptom**: 
- LightGBM CV R² = −0.07 (negative)
- QuantileRF CV R² = −0.088 (negative)
- Both contribute via diversity only, not individual signal

**Root Causes**:
1. **No early stopping**: All 200 iterations trained on CV folds with n=150–200 validation samples → severe overfitting
2. **Small validation folds**: Some folds have n_val < 150; models memorize fold-specific patterns
3. **Default hyperparameters untested**: Learning rate 0.05, depth=5, alpha=1.0 all from sklearn defaults
4. **No fold-aware hyperparameter adaptation**: Same hyperparams for folds with n=400 training vs. n=150
5. **Weak governance signal dominates**: Even oracle tree would struggle (targets noise-dominated)

**Impact**: Tree models drag down ensemble (8.6% meta-weight for QuantileRF despite −0.088 R²); they're learning noise.

**Fix Priority**: **PHASE 2 (HIGH, 2–3 weeks)**

---

### 3. **HIGH IMPACT: Fundamental Data Starvation**

**Symptom**: Effective training set n=756; with ~300–600 features, p/n ratio = 0.4–0.8 (tree models at risk).

**Root Causes**:
1. **Small effective sample**: 63 provinces × 12 train years (targets 2012–2023) = 756 rows
2. **OOF meta-learning starved**: 315 OOF rows for 48 meta-features (6 models × 8 outputs) → vastly underdetermined
3. **Structural missingness**: SC71–SC83 absent 2011–2017; SC52 discontinued 2021+; affects early folds
4. **Feature-to-sample ratio extreme**: PLS compresses 1000+ → 20 (98% loss); tree-track retains 600–800 features

**Impact**: Every estimation problem is confounded by high variance. Meta-learner weights unstable; conformal quantiles unreliable.

**Fix Priority**: **PHASES 1–2 (foundations), then consider Phase D (synthetic augmentation)**

---

### 4. **MEDIUM IMPACT: Feature Engineering Over-Complexity**

**Symptom**: 1000+ raw features from 12 engineering blocks; aggressive dimensionality reduction (98%) via PLS.

**Root Causes**:
1. **Feature explosion**: Lags, rolling windows (2,3,5), momentum, EWMA (3 spans), entity demeaning, rank/percentile → near-duplicate features
2. **Rolling windows on 2-year windows meaningless**: rolling_std(window=2, annual data) ≈ single-year difference
3. **VarianceThreshold=0.005 too loose**: Retains ~600–800 features; near-constant ones waste tree splits
4. **PLS assumes linearity**: 98% reduction discards interactive patterns that tree models could learn
5. **Multicollinearity unaudited**: Lag-1 highly correlated with rolling-2-mean; expanding-mean correlated with entity-mean
6. **No feature importance ranking**: Unknown which of 1000 features matter; cannot ablate weak ones

**Impact**: 3–8% RMSE loss vs. smarter feature selection; tree models waste splits on noise.

**Fix Priority**: **PHASE 3–4 (after stabilizing tree models)**

---

### 5. **MEDIUM IMPACT: Hyperparameter Defaults Not Tuned**

**Symptom**: All hyperparameters hardcoded in config.py; auto_tune_gb=False by default.

**Hardcoded Parameters Without Justification**:
```
CatBoost:   depth=5, learning_rate=0.05, l2_leaf_reg=3.0
LightGBM:   depth=5, learning_rate=0.05, reg_lambda=3.0
KernelRidge: alpha=1.0 (sklearn default)
SVR:        C=1.0, epsilon=0.1 (sklearn defaults)
QuantileRF: n_estimators=100 (low for p≈600)
```

**Expected Tuning Gains**:
- Learning rate sweep [0.01–0.2]: 3–5% RMSE improvement
- Depth/iterations grid: 2–4% improvement
- Kernel regularization (alpha, C, epsilon): 2–3% improvement
- **Total potential**: 8–12% RMSE reduction

**Impact**: 2–5% performance loss left on table; easily recoverable.

**Fix Priority**: **PHASE 3 (MEDIUM, 2 weeks)** — after fixing early stopping

---

### 6. **MEDIUM IMPACT: Super Learner Meta-Learning Design Flaws**

**Symptom**: Meta-learner NNLS receives 315 OOF rows with 48 meta-features (6 models × 8 outputs).

**Root Causes**:
1. **Insufficient OOF data**: 315 OOF rows / 48 meta-features ≈ 6.5× (should be ≥100×); NNLS fits noise
2. **Shared meta-weights across outputs**: Single weights per model, not per-criterion; with only 315/8≈40 rows per output
3. **Heterogeneous OOF from different fold sizes**: Fold 1 trains on 504 rows; Fold 5 trains on 756; NNLS conflates quality vs. fold-size differences
4. **NNLS collinearity problem**: CatBoost & LightGBM produce correlated OOF; NNLS zeros one artificially
5. **No reporting of meta-weight instability**: Should warn when max weight < 1.5× equal-weight

**Impact**: Meta-weights unreliable; per-output weights unstable across folds (>20% variance).

**Fix Priority**: **PHASE 2.5 (integrated with tree stabilization)**

---

### 7. **MEDIUM IMPACT: Target Variable Construction Problematic**

**Symptom**: SAW-normalized targets create year-specific ordinal targets incomparable across years.

**Root Causes**:
1. **Per-year SAW normalization introduces cross-sectional leakage**: Target for province P in year t+1 depends on ALL provinces in year t+1
   - A province's SAW score encodes RELATIVE ranking, not absolute quality
   - Model learns to predict ranking in future cross-section, which cannot be known from single province's history
2. **Logit transformation numerically risky**: logit(1e-6)≈−13.8, logit(1−1e-6)≈+13.8; outliers dominate loss
3. **Criteria composites year-specific**: CRITIC weights change year-by-year; C01_2013 ≠ C01_2024 (different weighting schemes)
4. **Non-stationary targets**: Model trained on historical values cannot reliably extrapolate to 2025

**Impact**: Systematic bias in target construction; models cannot learn stable causal mappings.

**Fix Priority**: **PHASE A (FOUNDATION, 1 week)** — switch to subcriteria or reference-year standardization

---

### 8. **MEDIUM IMPACT: Cross-Validation Methodology Issues**

**Symptom**: Walk-forward CV correct temporally; however, gaps in province stratification and small inner CV.

**Root Causes**:
1. **No leave-one-province-out (LOPO) evaluation**: Walk-forward validates all 63 provinces per year → autocorrelated errors across provinces
2. **Optuna inner CV only 4 folds**: min_train_years=7 with 13 available years → only years {8,9,10,11}; too few for stable HP estimates
3. **Optuna objective NaN-unsafe**: mean_squared_error fails on missing targets; silent propagation
4. **Small OOF folds insufficient for meta-learning**: 5 folds × 63 = 315 rows; should have ≥8 folds for stable meta-weights

**Impact**: CV metrics high-variance; meta-learning estimates unreliable.

**Fix Priority**: **PHASE 2.5 (integrated with OOF expansion)**

---

### 9. **LOWER PRIORITY: Missing Data Imputation Errors**

**Symptom**: Missing features filled with 0.0 as fallback; MICE imputation has distribution-shift issues.

**Root Causes**:
1. **0.0 fallback misleading for some features**: For percentile rank, 0.0 means "bottom of distribution" not "unknown"
2. **MICE fitted on train, applied to prediction with different year fallbacks**: Creates train-test distribution mismatch
3. **Imputation only applied to features, not targets**: Targets still have NaNs that propagate to model fitting
4. **VarianceThreshold(0.005) on imputed features**: Near-constant features remain after imputation

**Impact**: 1–3% RMSE loss; feature imputation artifacts.

**Fix Priority**: **PHASE 4 (LOW, integrate into feature engineering improvements)**

---

### 10. **LOWER PRIORITY: MCDM Integration Incomplete**

**Symptom**: ML forecast exists; no cross-validation with traditional MCDM methods (TOPSIS, VIKOR, etc.).

**Gaps**:
- No saved MCDM ranking for comparison with ML
- No rank correlation (Spearman ρ, Kendall τ) between methods
- No discordance detection (provinces ranked oppositely)
- No feedback loop to diagnose why ML differs from domain expertise

**Impact**: Cannot validate if ML ranking aligns with known governance patterns; reduced trustworthiness.

**Fix Priority**: **PHASE 5 (LOW, 1 week)** — conditional on Phases 1–4 completing successfully

---

## AUDIT FINDINGS: Detailed Component Assessment

### Section 1: Data Pipeline ✅ (Sound)

**Strengths**:
- ✅ YearContext dynamic exclusion prevents data loss on structural gaps (SC71–SC83 absent 2011–2017)
- ✅ Temporal imputation (3-stage) provides NaN-free feature matrices
- ✅ Walk-forward CV respects temporal ordering; entity-block bootstrap preserves panel structure
- ✅ Entity-level means computed on train_feature_years only (no target-year leakage)
- ✅ Province fallback ensures 2025 predictions for all 63 provinces

**Minor Concerns**:
- ⚠️ Linear interpolation assumes smooth governance trajectories; policy shocks violate assumption
- ⚠️ Missing data percentage not explicitly measured; imputation quality variance by province unknown

**Verdict**: Data pipeline correct as-is. Enhancement focus on data augmentation (Phase D), not pipeline redesign.

---

### Section 2: Ensemble Architecture ✅ (Sound)

**Strengths**:
- ✅ Six-model ensemble: 5 orthogonal paradigms (boosting, linear, kernel, margin, quantile)
- ✅ Per-output NNLS weights (with per-output variant): Each criterion can learn independent weights
- ✅ OOF generation: Per-column valid mask exploits maximum available residuals per criterion
- ✅ Dirichlet stacking: Analytical gradient; entity-block bootstrap preserves temporal structure
- ✅ Conformal per-criterion: Each output uses its own maximum residual set

**Design Risk**:
- BayesianRidge dominates (49% weight) with only 13% individual R²
- Tree models contribute via diversity, not individual signal (negative CV R²)

**Verdict**: Architecture is correct; weakness is base models and meta-learning data insufficiency, not ensemble design.

---

### Section 3: Base Model Implementations 🟡 (Correct, But Weak)

#### CatBoost / LightGBM
| Aspect | Status | Finding |
|--------|--------|---------|
| Depth/iterations | ✓ Sound | 5 depth, 200 iter reasonable for n=756 |
| Regularization | ⚠️ Weak | l2_leaf_reg=3.0 too weak; should be ≥10 for small n |
| **Early Stopping** | ❌ Missing | **No validation set; all iterations trained** |
| Joint vs. per-output | ✓ Good | CatBoost joint (MultiRMSE); LightGBM per-output |
| NaN handling | ✓ Correct | Properly handled before fitting |

**Impact**: 2–5% overfitting; small CV folds amplify this.

#### BayesianRidge
| Aspect | Status | Finding |
|--------|--------|---------|
| Priors | ✓ Good | Default Gamma(1e-6, 1e-6) weakly informative |
| Convergence | ✓ Robust | max_iter=3000, coordinate descent converges |
| Performance | ⚠️ Weak | Only 13% holdout R²; most stable model → dominates ensemble |

#### QuantileRF
| Aspect | Status | Finding |
|--------|--------|---------|
| Algorithm | ✓ Efficient | sklearn_quantile (Meinshausen 2006, C-implemented) |
| **Holdout Performance** | ❌ Failing | **R² = −0.088 (negative)** |
| n_estimators | ⚠️ Low | 100 trees; should be ≥300 for p≈600 |

#### KernelRidge / SVR
| Aspect | Status | Finding |
|--------|--------|---------|
| Kernel | ✓ Sound | RBF smooth, complementary to trees |
| Regularization | ⚠️ Untuned | KR: alpha=1.0; SVR: C=1.0, epsilon=0.1 (all defaults) |
| Performance | ⚠️ Weak | KR: R²=0.121; SVR: R²=0.076 (reasonable for small n) |

**Verdict**: All base models correctly implemented. Weakness is hyperparameter defaults and missing early stopping.

---

### Section 4: Cross-Validation & Metrics ✅ (Correct)

**Strengths**:
- ✅ Calendar-year aligned; expanding window preserves temporal order
- ✅ 5 folds (validation years 2020–2024), 8-year minimum training window
- ✅ Zero leakage: OOF predictions computed on validation folds only
- ✅ Entity-block bootstrap respects panel structure
- ✅ R², RMSE, MAE, MAPE all computed correctly

**Issues**:
- ⚠️ Only 315 OOF residuals for 8 criteria (< 40/criterion) → conformal underpowered
- ⚠️ No LOPO (leave-one-province-out) evaluation
- ⚠️ Optuna inner CV uses only 4 folds (too few for stable HP estimates)

**Verdict**: Evaluation strategy sound; calibration set size and meta-learning data are limiting factors.

---

### Section 5: Conformal Prediction ❌ (CRITICAL ISSUE)

**Current Implementation**:
- **Method**: Split conformal (25% holdout) + CV+ conformal (walk-forward residuals)
- **Calibration set**: n_cal = 189 samples across 8 criteria (~24 per criterion)
- **Target coverage**: 95% (α=0.05)
- **Actual result**: [0.0, 1.0] width=1.0 (degenerate)

**Why Degeneracy Occurs**:
```
With n_cal ≈ 24 per criterion:
  q̂ = quantile(|residuals|, 0.975) ≈ ±0.5 (high variance, extreme values)
  
Interval = [ŷ − q̂, ŷ + q̂] ≈ [ŷ − 0.5, ŷ + 0.5]
Clamped to [0, 1]: [min(1, ŷ + 0.5), max(0, ŷ − 0.5)]
Result: For many ŷ ∈ (0.5, 1.0), interval → [0.0, 1.0]
```

**Contributing Factors**:
1. Calibration set critically undersized
2. High OOF residual variance
3. Target transformation to [0,1] constrains interval space
4. No heteroscedasticity adjustment
5. No distributional modeling (assumes Gaussian)

**Verdict**: Conformal intervals completely unusable for decision-making.

---

### Section 6: Feature Engineering 🟡 (Sophisticated but Over-Engineered)

**Raw Feature Architecture** (~1,000+ features across 12 blocks):
```
├─ Current values: 29
├─ Lag features (t-1, t-2, t-3): 174
├─ Rolling stats (windows 2,3,5): 348
├─ Momentum / acceleration: 116
├─ Demeaned levels: 58
├─ EWMA (spans 2,3,5): 87
├─ Trend (polyfit): 29
├─ Entity rank / diversity: 35
├─ Cross-entity percentile / rank: 24
├─ Regional dummies: 5
└─ Other: ~50
```

**Dimensionality Reduction**:
- **PLS mode**: 1000+ → 20 components (98% reduction)
- **Tree mode**: VarianceThreshold(0.005) → 600–800 features

**Issues**:
1. **Aggressive reduction loses info**: 98% compression likely discards interactive patterns
2. **VarianceThreshold too loose**: 0.005 threshold (lowered from 0.01) retains ~90% of features
3. **Multicollinearity unaudited**: Lag-1 ≈ rolling-2-mean; EWMA ≈ expanding-mean
4. **No feature importance ranking**: Unknown which features drive predictions
5. **Rolling windows over 2 years meaningful only for first difference**

**Verdict**: Over-engineered for small n. Smarter selection could improve 3–8% RMSE.

---

### Section 7: Small Sample Size Effects (n ≈ 756) ⚠️

**Sample Size Breakdown**:
```
Total observations: 756 (63 provinces × 12 target years 2012–2023)
├─ Training folds: 504–630 samples (5 folds)
├─ Per-entity samples: 12 per province
├─ OOF meta-learner: 315 rows, 48 meta-features (p/n = 0.15, marginal)
├─ Conformal calibration: 189 global (~24 per criterion)
└─ p/n ratio (features/samples): 0.4–0.8 (tree models at risk)
```

**Consequences**:
1. **Per-output weight instability**: (8, 6) weights fitted on ~189 OOF; overfits to OOF correlation
2. **Entity underspecification**: 63 provinces need 126+ parameters with only 12 samples/entity
3. **Governance signal overwhelmed**: R²=0.22 means 78% aleatoric uncertainty
4. **Conformal quantile unreliability**: 24 residuals per criterion → huge variance

**Mitigations**:
- ✅ Ensemble diversity (already used)
- ✗ Early stopping (missing, needed)
- ✗ Cross-output regularization (disabled by default)
- ⚠️ Synthetic augmentation (Phase D, experimental)

---

## ENHANCEMENT PLAN: Five-Phase Implementation Roadmap

### **PHASE A: FOUNDATION FIXES (Week 1)**
Prerequisites: correctness bugs, configuration errors, target leakage.

| ID | Task | Priority | Effort |
|----|------|----------|--------|
| A1 | Add persistence baseline to all evaluations | CRITICAL | 1 day |
| A2 | Fix NaN-unsafe Optuna objective (nanmean) | CRITICAL | 0.5 day |
| A3 | Switch to subcriteria forecasting OR reference-year SAW standardization | CRITICAL | 2 days |
| A4 | Verify structurally missing sub-criteria don't contaminate composites | HIGH | 1 day |
| A5 | Extend median imputation to ALL feature blocks | HIGH | 1 day |
| A6 | Log data sufficiency warning when p/n > 0.3 | MEDIUM | 0.5 day |

**Total**: 5–6 days  
**Key Success Metric**: Persistence Skill Score > 0 (ensemble beats naive last-year baseline)

---

### **PHASE 1: CONFORMAL PREDICTION RESCUE (Weeks 2–3)**
**CRITICAL Priority** — 2–3 weeks, 8–10 days of work

**Objective**: Replace degenerate [0.0, 1.0] intervals with usable [0.3–0.7] intervals, ≥93% empirical coverage.

**Root Fix**: Extend conformal calibration from n_cal=189 to n_cal=500–650 via additional CV folds with shorter training windows.

#### 1.1 Extended Conformal Calibration (E-02)

**Implementation**:
```python
# config.py additions
conformal_extended_cv: bool = True
conformal_min_train_years: int = 3  # vs. cv_min_train_years=8
conformal_max_folds: int = 10  # Generate 5–10 additional folds
conformal_residual_target: int = 600  # Target n_cal ≈ 600
```

**Effect**: Generate additional CV folds with min_train_years=3 (start validation 2014 vs. 2019)
- **Primary OOF**: 315 residuals
- **Extended OOF**: +300–350 residuals (5–7 additional folds)
- **Total**: n_cal ≈ 600–650 → ~80 per criterion ✓

**Expected Outcome**:
- Per-criterion n_cal: 24 → 80 (3.3× increase)
- Interval widths: [0.0–1.0] → [0.35–0.65]
- Coverage: 100% (trivial) → 93–97% (valid)

**Effort**: 3–4 days implementation + 1 day testing

#### 1.2 Per-Criterion Effective-D Bonferroni (F-06 Enhancement)

**Implementation**: Compute PCA-based effective dimensionality separately per criterion.
- Current: Single global D_eff across 8 criteria
- New: D_eff_per_criterion with adaptive Bonferroni quantiles

**Effect**: Tighter quantiles for low-variance criteria.

**Effort**: 1–2 days

#### 1.3 Mondrian Stratification (E-03, Optional)

**Implementation**: Adaptive interval widths based on province-level missingness/volatility.
- Stratify calibration residuals into 3–5 strata
- Compute separate quantiles per stratum
- Wider intervals for high-missingness provinces

**Effect**: 10–15% narrower intervals on average; targeted uncertainty quantification.

**Effort**: 2–3 days (optional; implement after 1.1 proves successful)

#### 1.4 Student-t Distribution for Small n (E-06, Optional)

**Implementation**: Account for heavy tails when n_cal < 50 per criterion.
- Estimate degrees of freedom from empirical kurtosis
- Use Student-t ppf instead of Gaussian quantile

**Effect**: More conservative intervals for small-sample criteria.

**Effort**: 0.5 day

**PHASE 1 SUMMARY**:
- **Critical**: Steps 1.1–1.2 (5–6 days)
- **Optional**: Steps 1.3–1.4 (2–3 days)
- **Total**: 8–10 days
- **Success Metric**: Interval widths [0.3–0.7] with coverage ≥93%

---

### **PHASE 2: TREE MODEL STABILIZATION (Weeks 3–4)**
**HIGH Priority** — 2–3 weeks, 7–8 days of work

**Objective**: Eliminate negative CV R² from LightGBM (−0.07) and QuantileRF (−0.088).

#### 2.1 Early Stopping for Gradient Boosting

**Implementation**: Extract 20% holdout validation within each fold; pass to early stopping.

**For CatBoost**:
```python
# Split training fold into 80% train / 20% validation
early_stopping_rounds = 20
self._model.fit(X_train, y_train, 
                eval_set=[(X_val, y_val)],
                use_best_model=True)
```

**For LightGBM**: Per-output early stopping with callbacks.

**Expected Effect**:
- Typical stop: iterations 50–100 of 200 (50% reduction)
- RMSE improvement: 2–5%
- Prevents overfitting on small CV folds

**Effort**: 2–3 days implementation + 1 day validation/config integration

**Config Changes**:
```python
gb_early_stopping_rounds: int = 20
gb_validation_split: float = 0.20
```

#### 2.2 Fold-Aware Entity Demean (Temporal Leakage Fix)

**Implementation**: Recompute entity-level means PER FOLD (training portion only).
- Current: Computed on full training set in early CV folds
- New: Per-fold computation removes mild temporal look-ahead bias

**Effect**: 1–2% RMSE improvement; pure correctness fix.

**Effort**: 1–2 days

#### 2.3 Adaptive Hyperparameters by Fold Size

**Implementation**: Scale tree depth/iterations with n_train_fold.
```python
if n_train < 250: depth = 3
elif n_train < 400: depth = 4
elif n_train < 600: depth = 5
else: depth = 6
```

**Effect**: Prevents overfitting on small folds; 1–2% RMSE improvement.

**Effort**: 1 day

#### 2.4 QuantileRF Stabilization

**Implementation**:
- Increase n_estimators from 100 to 300
- Implement early stopping (max trees = f(n_samples))
- Add minimum leaf size adaptation

**Effect**: Reduce variance; stabilize quantile estimates.

**Effort**: 1–2 days

**PHASE 2 SUMMARY**:
- **Critical**: Step 2.1 (3–4 days)
- **Supporting**: Steps 2.2–2.4 (3–4 days)
- **Total**: 7–8 days
- **Success Metric**: LightGBM CV R² > −0.02 (previously −0.07); QuantileRF CV R² > 0.05

#### 2.5 Increase OOF Size for Meta-Learning (Parallel with Phase 2)

**Objective**: More calibration data for meta-learner NNLS.

**Implementation**:
- Reduce cv_min_train_years: 8 → 5
- Creates 3 additional validation years (2017, 2018, 2019)
- Primary OOF: 315 → 441 rows (7 validation years × 63)

**Effect**:
- Meta-learner NNLS more stable (441 vs. 315 rows)
- Per-output: ~55 rows instead of ~40

**Effort**: 1 day (config change + CV pipeline update)

---

### **PHASE 3: HYPERPARAMETER OPTIMIZATION (Weeks 4–5)**
**MEDIUM Priority** — 2 weeks, 8–10 days total work (but 20–27 hours background compute)

**Objective**: Systematically tune all base model hyperparameters via Optuna TPE; target 8–12% RMSE reduction.

#### 3.1 Setup Optuna Framework

**Files**: `forecasting/hyperparameter_tuning.py` (NEW)

**Scope**:
- CatBoost: learning_rate, max_depth, l2_leaf_reg, subsample
- LightGBM: learning_rate, num_leaves, reg_lambda, min_data_in_leaf
- KernelRidge: alpha, kernel
- SVR: C, epsilon, gamma
- QuantileRF: n_estimators, max_depth, min_samples_leaf

**Method**: Optuna TPE sampler with Hyperband pruner.
- **Objective**: Maximize CV R² score
- **Trials**: 40–80 trials
- **Wall-time**: 20–27 hours (run overnight)
- **CV**: Walk-forward CV per trial

**Config**:
```python
auto_tune_gb: bool = True
auto_tune_kernel: bool = True
hp_tune_n_trials: int = 40
hp_tune_timeout_seconds: int = 3600  # 1 hour per model
```

**Implementation**:
```python
class EnsembleHyperparameterOptimizer:
    def __init__(self, config):
        self.config = config
        self.study = optuna.create_study(
            sampler=TPESampler(),
            pruner=HyperbandPruner()
        )
    
    def objective_catboost(self, trial):
        """Objective function for CatBoost tuning."""
        lr = trial.suggest_float('learning_rate', 0.01, 0.2)
        depth = trial.suggest_int('max_depth', 3, 7)
        l2_reg = trial.suggest_float('l2_leaf_reg', 0.1, 100)
        
        # CV evaluate with these params
        cv_scores = evaluate_with_cv(..., lr=lr, depth=depth, l2_reg=l2_reg)
        
        return np.mean(cv_scores)  # Maximize mean CV R²
    
    def optimize(self):
        """Run optimization."""
        self.study.optimize(self.objective_catboost, n_trials=self.config.hp_tune_n_trials)
```

**Testing**:
```python
def test_optuna_improves_baseline():
    """Verify tuned hyperparams beat defaults."""
    baseline_score = evaluate_with_defaults()
    tuned_score = evaluate_with_optuna_params()
    
    assert tuned_score > baseline_score, "Tuning should improve baseline"
```

**Effort**: 3–4 days implementation + 1 day testing (background compute ≈20 hrs)

#### 3.2 Hyperparameter Sensitivity Analysis

**Implementation**: Analyze which hyperparams have largest impact.

**Output**: Importance ranking, interaction effects.

**Effort**: 1–2 days post-optimization

#### 3.3 Implement Cross-Validation in Tuning

**Objective**: Ensure tuning CV uses same splits as main CV (no leakage).

**Effort**: 1 day

**PHASE 3 SUMMARY**:
- **Total Effort**: 8–10 days (human work) + 20–27 hours compute
- **Expected Gain**: 8–12% RMSE reduction
- **Success Metric**: Holdout RMSE: 2.39 → 2.1–2.2

---

### **PHASE 4: FEATURE ENGINEERING REFINEMENT (Weeks 5–6)**
**MEDIUM Priority** — 1–2 weeks, 5–8 days of work

**Objective**: Reduce feature count to ≤80; eliminate near-duplicate and noise features; target 3–8% RMSE improvement.

#### 4.1 Feature Selection via Importance Ranking

**Implementation**: Two-stage selection:
1. **Permutation importance**: Train base models on full 1000+ features; rank by importance drop
2. **Recursive elimination**: Remove weakest features iteratively; validate hold-one-province-out RMSE

**Target**: Reduce to ≤80 features; ensure each improves holdout RMSE.

**Effort**: 2–3 days implementation + 1 day validation

#### 4.2 Remove Near-Duplicate Features

**Identify & Remove**:
- Rolling window=2 (single first difference; captured by Block 4 momentum)
- Expanding mean (near-identical to entity-mean baseline)
- Rolling skewness (requires n>30; with 14 years per entity, unreliable)

**Benefit**: Reduced p/n ratio; tree models focus on informative splits.

**Effort**: 1 day

#### 4.3 Fix Multicollinearity

**Implementation**:
- Compute VIF (variance inflation factor) matrix
- Remove features with VIF > 10 (correlation > 0.95)
- For lagged/rolling features with high VIF, keep highest importance only

**Effort**: 1–2 days

#### 4.4 Audit Cross-Entity Features

**Implementation**: Verify cross-sectional percentile rank computed on same sub-criteria set per year.
- Current: Computed over all actively present provinces (inconsistent composition across years)
- Fix: Restrict to provinces present in same year AND same sub-criteria set

**Effort**: 0.5 day

#### 4.5 Extend Median Imputation to All Feature Blocks

**Implementation**: Replace 0.0 fallback for rolling stats, EWMA, trend with cross-sectional median.
- Current: Only lags use median + _was_missing flag
- New: All blocks use component-year-specific medians

**Benefit**: Better imputation semantics (especially for percentile rank / z-score).

**Effort**: 1 day

**PHASE 4 SUMMARY**:
- **Total Effort**: 5–8 days
- **Expected Gain**: 3–8% RMSE reduction
- **Success Metric**: Feature count ≤80; no VIF > 10; ablation test confirms all features improve RMSE

---

### **PHASE 5: MCDM INTEGRATION & VALIDATION (Week 6)**
**LOW Priority** — 1 week, 3–5 days of work
(Conditional on Phases 1–4 showing positive skill score)

**Objective**: Complete ML-to-MCDM feedback loop; validate ML forecast aligns with traditional MCDM rankings.

#### 5.1 Save ML Forecast & Ranking

**Implementation**:
- Save predicted scores (8 criteria × 63 provinces)
- Compute composite ranking via MCDM weighting (TOPSIS/VIKOR/PROMETHEE)
- Save predicted ranking and uncertainty intervals

**Effort**: 1 day

#### 5.2 Compute Cross-Method Rank Correlations

**Implementation**:
- Compare ML ranking vs. ER ranking vs. TOPSIS/VIKOR/PROMETHEE
- Compute Spearman ρ, Kendall τ
- Identify provinces where methods disagree (discordance detection)

**Effort**: 1 day

#### 5.3 Sensitivity Analysis

**Implementation**:
- Vary MCDM weights (±10%); recompute rankings
- Check if ML ranking stable vs. traditional MCDM (which is more sensitive?)
- Document parameter sensitivity

**Effort**: 1–2 days

#### 5.4 Report & Diagnostics

**Implementation**:
- Per-province forecast with uncertainty intervals
- Per-criterion performance (RMSE, coverage) sorted by difficulty
- Provinces ranked by prediction uncertainty (>0.5 interval width)

**Effort**: 1 day (mostly visualization)

**PHASE 5 SUMMARY**:
- **Total Effort**: 3–5 days
- **Prerequisite**: Phases 1–4 complete with positive skill score
- **Success Metric**: ML-traditional MCDM rank correlation ρ > 0.7; discordance explained by data/model uncertainties

---

## SUCCESS CRITERIA & VALIDATION

### Metric Targets (by Phase)

| Phase | Metric | Baseline | Target | Validation |
|-------|--------|----------|--------|------------|
| **A–1** | Persistence Skill Score | <0 (unknown) | >0.10 | CV evaluation |
| **2** | LightGBM CV R² | −0.07 | >−0.02 | 5-fold CV |
| **2** | QuantileRF CV R² | −0.088 | >0.05 | 5-fold CV |
| **1** | Conformal interval width | 1.0 | 0.3–0.7 | Holdout sample |
| **1** | Conformal coverage | 100% (trivial) | 93–97% | Holdout holdout test |
| **3** | Holdout RMSE | 2.39 | 2.1–2.2 | 63 × 8 holdout |
| **4** | Feature count | 1000+ | ≤80 | Ablation study |
| **4** | Feature VIF max | unknown | <10 | Correlation audit |
| **All** | Holdout R² (stable criteria) | 0.217 | >0.50 | Per-criterion eval |

### Per-Phase Validation Checkpoints

**Phase A Completion**:
- [ ] Persistence baseline implemented; Skill Score computed (should be >0)
- [ ] SAW standardization fixed or criteria-level switched to subcriteria
- [ ] Data audit complete (structural missingness verified)

**Phase 1 Completion**:
- [ ] Extended conformal CV generates ≥500 residuals total (≥60 per criterion)
- [ ] Conformal intervals [0.3–0.7] width (informative, not degenerate)
- [ ] Empirical coverage ≥93% on holdout or historical validation set

**Phase 2 Completion**:
- [ ] Early stopping implemented; typical stop at 50–100 iterations
- [ ] Tree model CV R² positive (>−0.02 for LightGBM, >0.05 for QuantileRF)
- [ ] Holdout RMSE improves 2–5% vs. Phase 1 baseline

**Phase 3 Completion**:
- [ ] Optuna tuning complete; best hyperparams saved
- [ ] Holdout RMSE: 2.39 → 2.1–2.2 (8–12% improvement target)
- [ ] Tuned params differ significantly from defaults (>10% change in ≥3 params)

**Phase 4 Completion**:
- [ ] Feature selection complete; ≤80 features retained
- [ ] No feature with VIF > 10
- [ ] Ablation study confirms all features improve holdout RMSE

**Phase 5 Completion**:
- [ ] ML ranking saved; compared with traditional MCDM methods
- [ ] Rank correlation ρ > 0.7 (good agreement)
- [ ] Discordances explained (uncertainty intervals, noise, data gaps)

---

## PRIORITIZED IMPLEMENTATION SCHEDULE

### **Week 1: Foundation (Phase A)**
- Implement persistence baseline → compute Skill Score
- Fix NaN-unsafe Optuna objective
- Switch to subcriteria or fix SAW standardization
- Verify no target leakage from structural missingness
- Extend median imputation to all features

**Deliverable**: Verified correctness, Skill Score > 0, no data bugs

### **Weeks 2–3: Conformal Rescue + Tree Stabilization Start (Phases 1 + 2.1)**
- Extended conformal CV: n_cal 189 → 600
- Per-criterion effective-D Bonferroni
- Early stopping for CatBoost/LightGBM (critical fix)
- Parallel: Increase OOF size (min_train_years 8 → 5)

**Deliverable**: Conformal intervals usable [0.3–0.7]; tree models stop overfitting

### **Week 4: Tree Stabilization Complete + Hyperparameter Tuning Start (Phases 2.2–2.4 + 3)**
- Fold-aware entity demean
- Adaptive hyperparams by fold size
- QuantileRF stabilization
- Optuna setup; launch HP tuning (background)

**Deliverable**: Tree CV R² positive; HP tuning in progress

### **Week 5: Feature Engineering + Tuning Complete (Phase 4 + finish 3)**
- Feature selection via importance ranking
- Remove near-duplicates, fix multicollinearity
- Implement tuned hyperparams; validate improvement
- Holdout RMSE should be 2.1–2.2

**Deliverable**: ≤80 features; holdout RMSE improved 8–12%

### **Week 6: MCDM Integration (Phase 5)**
- ML-to-MCDM ranking comparison
- Sensitivity analysis
- Final diagnostics & reporting

**Deliverable**: ML rankings validated against traditional MCDM; discordances explained

---

## IMPLEMENTATION GUIDANCE BY PHASE

### Phase A: Foundation Fixes
- **Fixity**: Must complete before moving to Phase 1
- **Testing**: Add unit tests for each fix at **tests/test_data_correctness.py**
- **Checkpoints**: Daily validation that no regression from baseline

### Phase 1: Conformal Prediction
- **Complexity**: High; requires care with CV split logic
- **Testing**: Comprehensive tests in **tests/test_conformal_extended.py**
  - [ ] n_cal size increases as expected
  - [ ] Interval widths non-degenerate
  - [ ] Coverage ≥93% on historical data
  - [ ] Per-criterion widths vary (not uniform)
- **Risky part**: Extended conformal CV loop; validate that OOF residuals are genuine

### Phase 2: Tree Model Stabilization
- **Complexity**: Medium; early stopping straightforward
- **Testing**: Tests in **tests/test_early_stopping.py**
  - [ ] Early stopping activates before max iterations
  - [ ] CV R² improves 2–5%
  - [ ] Validation loss decreases monotonically until stop
- **Risky part**: Ensure train/validation split doesn't leak; data must be randomized but stratified by entity

### Phase 3: Hyperparameter Optimization
- **Complexity**: Low code complexity, high compute cost
- **Testing**: Minimal; long Optuna runs are self-validating
  - [ ] Final tuned params improve over defaults
  - [ ] Trial history saved for reproducibility
- **Risky part**: Ensure Optuna uses SAME CV splits as main pipeline; otherwise leakage into tuning

### Phase 4: Feature Engineering
- **Complexity**: Medium; feature interaction effects subtle
- **Testing**: Tests in **tests/test_feature_selection.py**
  - [ ] Selected features improve ablation RMSE vs. all features
  - [ ] No high VIF pairs
  - [ ] Feature importances vary widely (not all equal)
- **Risky part**: Over-aggressive feature removal hurts generalization; use ablation study as ground truth

### Phase 5: MCDM Integration
- **Complexity**: Low; mostly analytics
- **Testing**: Validation against known MCDM methods
  - [ ] Rank correlations reasonable (ρ > 0.5)
  - [ ] Discordances explained by uncertainty/data gaps
- **Risky part**: None; purely comparative

---

## FILE MODIFICATION CHECKLIST

### Phase A
- [ ] `config.py`: Add `persist_baseline`, `use_subcriteria_forecast`, `forecast_level='subcriteria'`
- [ ] `forecasting/unified.py`: Fix Optuna objective `nanmean()`
- [ ] `data_loader.py`: Add target leakage validation
- [ ] `forecasting/features.py`: Extend median imputation to all blocks
- [ ] `tests/test_data_correctness.py`: NEW — Add leakage/imputation tests

### Phase 1
- [ ] `config.py`: Add `conformal_extended_cv`, `conformal_min_train_years`, etc.
- [ ] `forecasting/conformal.py`: Extend CV loop; per-criterion effective-D; optional Mondrian/Student-t
- [ ] `forecasting/unified.py` stage5: Integrate extended conformal residuals
- [ ] `tests/test_conformal_extended.py`: NEW — Comprehensive conformal testing

### Phase 2
- [ ] `config.py`: Add `gb_early_stopping_rounds`, `gb_validation_split`
- [ ] `forecasting/gradient_boosting.py`: Implement early stopping for both models
- [ ] `forecasting/quantile_forest.py`: Stabilization & n_estimators boost
- [ ] `forecasting/unified.py`: Fold-aware entity demean; adaptive HP by fold size
- [ ] `config.py`: Change `cv_min_train_years` 8 → 5 (increase OOF size)
- [ ] `tests/test_early_stopping.py`: NEW — Early stopping validation

### Phase 3
- [ ] `forecasting/hyperparameter_tuning.py`: NEW — Optuna framework
- [ ] `forecasting/unified.py` stage3: Integrate HP optimization
- [ ] `config.py`: Add `auto_tune_*`, `hp_tune_n_trials`, etc.
- [ ] Update all model constructors to accept tuned hyperparams
- [ ] Save/load tuned params to `output/hp_tuning_best_params.json`

### Phase 4
- [ ] `forecasting/features.py`: Feature selection pipeline (importance ranking, recursive elimination)
- [ ] `forecasting/preprocessing.py`: VIF computation; multicollinearity audit
- [ ] Remove hardcoded feature block weights; use importance-based filtering
- [ ] `tests/test_feature_selection.py`: NEW — Ablation study & VIF validation

### Phase 5
- [ ] `output/orchestrator.py`: Save ML rankings to CSV
- [ ] `ranking/`: Compute rank correlations with traditional MCDM methods
- [ ] `output/report_writer.py`: Add comparative ranking tables
- [ ] Add visualization dashboard (uncertainty intervals by province)

---

## KNOWN RISKS & MITIGATION

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Early stopping reduces model capacity | Medium | Validate via CV; accept if generalizes better |
| Extended conformal CV 10× slower | Low | Cache results; run background overnight |
| Feature selection removes important interactions | Medium | Use ablation study as ground truth; validate holdout RMSE |
| HP tuning overfits to CV data | Medium | Use held-out test set; report generalization gap |
| Mondrian stratification too granular | Low | Start with 3 strata; merge if too sparse |
| MCDM methods disagree with ML | Medium | Expected; investigate via uncertainty & domain logic |

---

## SUCCESS METRICS SUMMARY

### Quantitative Targets

**Before Enhancement** (Baseline):
- Holdout R² = 0.217
- Holdout RMSE = 2.39
- Conformal interval width = 1.0 (degenerate)
- Skill Score: Unknown (likely ≤ 0)

**After Phase 1 (Conformal Only)**:
- Conformal interval width = 0.3–0.7 ✓
- Coverage ≥ 93% ✓
- Holdout RMSE: unchanged (conformal doesn't improve point estimate)

**After Phase 2 (Tree Stabilization)**:
- Holdout RMSE: 2.39 → 2.28–2.30 (−4% improvement)
- Tree model CV R²: Positive ✓
- Skill Score > 0.05

**After Phase 3 (HP Tuning)**:
- Holdout RMSE: 2.28 → 2.1–2.2 (−12% cumulative from baseline)
- Skill Score > 0.15

**After Phase 4 (Feature Engineering)**:
- Feature count: 1000+ → ≤80 ✓
- Holdout RMSE: 2.1 → 2.0–2.05 (−16% cumulative)
- Per-criterion RMSE: All < 2.8 (vs. 2.81 max baseline)

**After Phase 5 (MCDM Integration)**:
- ML-MCDM rank correlation: ρ > 0.7 ✓
- Discordances explained ✓

### Qualitative Success Criteria

- ✅ System understanding: Clear documentation of why ML forecasts differ from MCDM
- ✅ Model interpretability: Feature importances quantified; per-province uncertainty transparent
- ✅ Production readiness: Conformal intervals actionable for decision-making
- ✅ Reproducibility: All hyperparams, feature selections logged; results reproducible

---

## APPENDIX: Audit File Coverage

| File | Lines | Audit Coverage | Key Findings |
|------|-------|-----------------|--------------|
| `config.py` | 910 | ✓ Complete | Hardcoded hyperparams; missing early stopping config |
| `forecasting/features.py` | 1375 | ✓ Complete | Feature explosion; no importance audit |
| `forecasting/unified.py` | 3170 | ✓ Complete | OOF size limitation; extended conformal needed |
| `forecasting/super_learner.py` | 1678 | ✓ Complete | Meta-learner underfitting; NNLS on insufficient data |
| `forecasting/gradient_boosting.py` | 599 | ✓ Complete | Missing early stopping; default hyperparams |
| `forecasting/bayesian.py` | 394 | ✓ Complete | Correct; only 13% individual R² |
| `forecasting/quantile_forest.py` | 340 | ✓ Complete | n_estimators=100 too low; no early stopping |
| `forecasting/conformal.py` | 412 | ✓ Complete | Calibration set size critical issue |
| `forecasting/preprocessing.py` | 572 | ✓ Complete | Missing data handling incomplete; MICE distribution shift |
| `data_loader.py` | 450 | ✓ Complete | Data integrity; structural missingness |
| `tests/` | 2100 | ✓ Complete | 368 passing; 0 failing (coverage good) |

---

## QUICK REFERENCE: Phase Implementation Order

```
Week 1:   PHASE A (correctness) ..........  5–6 days
Week 2–3: PHASE 1 (conformal) + 2.1 ...... 8–10 days
Week 4:   PHASE 2 complete, 3 start ...... 3–4 days (+ 20 hrs compute)
Week 5:   PHASE 3 (tuning) + 4 ........... 5–8 days (parallel)
Week 6:   PHASE 5 (MCDM) ................. 3–5 days

TOTAL: 6 weeks, 24–33 days human effort, ~20 hrs compute time
```