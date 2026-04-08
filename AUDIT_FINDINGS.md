# COMPREHENSIVE AUDIT: Methods & Results Section
## Vietnam PAPI Hybrid MCDM-Ensemble Learning Framework

**Audit Date**: April 8, 2026  
**Scope**: Methods and Data (Sections 2 & 3)  
**Auditor Role**: Senior Data Scientist specializing in MCDM & Stacking Ensembles

---

## EXECUTIVE SUMMARY

The paper presents an ambitious three-tier framework (hierarchical CRITIC weighting, multi-method MCDM ranking, ensemble forecasting) but contains **13 critical/serious issues** affecting theoretical soundness, algorithmic correctness, and data science integrity. Issues range from fundamental violations in the CRITIC weighting scheme to unjustified design choices and critical reproducibility gaps.

**Overall Assessment**: Paper is scientifically unsound as currently written. Major revisions required before publication.

---

# CRITICAL ISSUES (Prevent Publication)

## **CRITICAL #1: Hierarchical CRITIC Scale Violation**

### **Location**  
Equation (eq:criterion_composite), pages 654-660; Hierarchical CRITIC section, pages 625-670

### **The Problem**

The paper's two-level CRITIC implementation violates the fundamental theoretical foundation of CRITIC weighting.

**Level 1 (Correct):**
```
For each criterion k:
  1. Normalize all sub-criteria: X̃_{k,ij} ← (X_{k,ij} - min) / (max - min)
  2. Compute variance on NORMALIZED scale: σ(X̃_{k,·j})
  3. Compute correlations on NORMALIZED scale
  4. Derive weights: u_{k,j} = σ(X̃_{k,·j}) × Σ(1 - r_{jj'})
```

**Level 1 Output (Critical Error):**
```
But then applies these weights to ORIGINAL (unnormalized) data:
Z_{i,k} = Σ_j u_{k,j} × X_{i,k,j}    <-- Mix of normalized weights + original data
```

### **Why This Violates CRITIC**

1. **CRITIC relies on variance as scale-invariant importance metric**
   - Weights computed from normalized variance: bounded to [0,1] scale
   - But applied to original data: potentially unbounded scale
   - Variance meaning is lost

2. **Mathematical inconsistency**
   - Example: Suppose sub-criterion A has original range [1, 3], sub-criterion B has range [0.1, 0.9]
   - After normalization both [0,1], so σ(X̃_A) ≈ σ(X̃_B)
   - But original scales very different!
   - Applying u_A, u_B to original values distorts the weighting intention

3. **Inversion of CRITIC logic**
   - CRITIC says: "High variance → high weight (discriminatory power)"
   - But if applying normalized weights to unnormalized data:
     - A criterion with naturally large scale gets artificially inflated weight
     - Contradicts the principle

### **Correct Approach**

**Option A (Preserve Scale):**
```
After Level 1 on normalized data, compute:
Z_{i,k} = Σ_j u_{k,j} × X̃_{i,k,j}    <-- Use normalized composite
Then apply Level 2 CRITIC to Z (normalized output)
```

**Option B (Work in Original Scale):**
```
Recompute Level 1 weights on ORIGINAL scale:
  σ_raw(X_{k,·j}) = std of unnormalized values
  C_{k,j} = σ_raw(X_{k,·j}) × Σ(1 - r_{jj'})
Then: Z_{i,k} = Σ_j u_{k,j} × X_{i,k,j}    <-- Consistent scales
```

**The Paper Acknowledges the Problem but Doesn't Solve It:**
> "This design choice is essential because if forced to use normalized values, the composite scores Z_{i,k} would be artificially constrained to a narrow band (approximately [0, 1]), losing information about the true magnitude differences between criteria."

**But the solution is wrong:** You can't escape this by mixing scales! Either work in one space consistently OR accept the compression—but don't mix them.

### **Impact on Results**

- **Weighting bias**: Criteria with naturally higher measurement scales receive inflated weights
- **Ranking distortion**: Provincial rankings potentially biased toward criteria with larger natural scales
- **Sensitivity to measurement units**: Changing from 0-3.33 scale to 0-1 scale would change weights
- **CRITIC theoretical invalidity**: The framework is no longer objective data-driven weighting, but ad-hoc hybrid

---

## **CRITICAL #2: Level 2 CRITIC Double Normalization**

### **Location**  
Level 2 procedure, pages 660-665; Equation (eq:critic_level2)

### **The Problem**

After computing criterion composites Z (already weighted combinations), Level 2 normalizes AGAIN:

```
Step 1: Normalize Z̃_{i,k} = (Z_{i,k} - min_i Z_{i,k}) / (max_i Z_{i,k} - min_i Z_{i,k})
Step 2: Compute variance on normalized composites: σ(Z̃_{·k})
Step 3: Derive Level 2 weights: v_k = σ(Z̃_{·k}) × Σ(1 - r_{kk'})
```

### **Why This Breaks CRITIC**

1. **Discards natural criterion variance information**
   - True variance of criterion k in original units represents discriminatory power
   - Normalizing removes this → all criteria appear to have similar variance band [0,1]
   - Level 2 weights then reflect "relative position in normalized space," not true importance

2. **Confounds data variance with normalization effects**
   - Suppose C06 has true range [1.5, 3.0], C07 has true range [0.2, 2.8]
   - After normalization, both appear [0, 1]
   - But if C06 naturally more tightly clustered (lower variance in original units), it still appears similar to C07 post-normalization
   - Weighting ignored the key insight that C06 is more predictable (tighter clustering)

3. **Violates CRITIC's fundamental assumption**
   - CRITIC requires working on original scale throughout
   - Normalization is step-by-step for pairwise comparisons—not for wisdom loss

### **Correct Approach**

Apply Level 2 CRITIC directly to Z without normalization:
```
C_k = σ(Z_{·k}) × Σ(1 - r(Z_{·k}, Z_{·k'}))
```
Where σ and r are computed on original Z values.

The criterion natural variance dispersion (some criteria more stable, others volatile) is precisely what CRITIC should capture!

---

## **CRITICAL #3: COPRAS-EDAS Perfect Redundancy Unaddressed**

### **Location**  
Method descriptions (pages 745-760); Results section (pages 1405-1410)

### **The Problem**

**Paper claims "five distinct MCDM methods"** but COPRAS and EDAS are mathematically equivalent under PAPI data:

From COPRAS section:
> "Because the PAPI framework strictly comprises benefit-oriented sub-criteria (C = ∅), the penalization term inherently collapses, reducing the analytical formulation to Q_i = S_i^+."

Result: COPRAS reduces to weighted sum aggregation = simple additive.

EDAS also uses weighted sums for benefit criteria.

**Empirical Validation:**
```
Inter-method Spearman correlations:
- COPRAS ↔ EDAS: ρ = 1.00 (perfect agreement)
- TOPSIS ↔ PROMETHEE: ρ = 0.98
- VIKOR ↔ others: ρ = 0.89-0.95
```

### **Why This Matters**

1. **Methodological Diversity Claim False**
   - Claims multi-method robustness across "five distinct paradigms"
   - Actually: three paradigms (distance-based, outranking, proportional)
   - But proportional methods (COPRAS, EDAS) are identical
   - Only 4 truly distinct methods

2. **Inflates Consensus Confidence**
   - Mean inter-method correlation ρ̄ = 0.96 looks robust
   - But "6 methods" actually 4 methods + 2 duplicates
   - If n independent methods, ρ̄ interpretation changes

3. **Wastes Computational Resources**
   - Computing identical rankings (COPRAS and EDAS) twice
   - Could replace with one method or diversify differently

### **Implications**

Table showing inter-method correlations claims:
> "Strong consensus ($\bar{ρ} > 0.70$) signals all methods largely agree...resulting consensus rankings are methodologically robust"

**But**: Is agreement due to genuine robustness or duplicate methods inflating agreement?

### **Required Fix**

**Option 1 (Recommended):** Drop COPRAS, keep EDAS
- Rationale: EDAS is newer, more defensible framework
- Explicitly state: "four core MCDM methods representing three paradigms"
- Revise inter-method agreement statistics excluding COPRAS

**Option 2:** Replace COPRAS with different method
- E.g., TAXIS, SAW-enhanced variant, or other true benefit-type method
- Genuinely increase methodological diversity

**Option 3 (Document Redundancy):** Keep both but explicitly acknowledge
- Add discussion: "Perfect correlation (ρ=1.00) indicates mathematical equivalence"
- Reframe as validation: "EDAS and COPRAS redundancy confirms proportional aggregation consensus"
- Lower methodological diversity claims appropriately

---

## **CRITICAL #4: Feature Imputation Violates Temporal Causality**

### **Location**  
Feature Engineering section, page 1000-1005

### **The Problem**

Paper states:
> "Critically, features that are formally undefined for early panel years (e.g., rolling statistics requiring lookback) are not discarded. Instead, they are imputed through the same MICE framework applied in Section 2.2, preserving the effective sample size and leveraging the multivariate structure of the features themselves."

### **Why This Destroys Temporal Integrity**

1. **MICE assumes exchangeability across observations**
   - Valid for cross-sectional data: one observation = one province
   - INVALID for panel data: observation_t uses information from observation_{t+1, t+2, ...}
   - Temporal causality VIOLATED

2. **Example of the Error**
   ```
   Feature: RollingMean(5yr) for 2011 (Lag-4 = 2007 undefined)
   MICE fills 2007 value using:
     - All other features for 2007
     - Plus 2008, 2009, 2010, 2011, 2012, ... 2024 data (14 years future!)
   Result: 2007 synthetic value uses future information
   ```

3. **Information Leakage**
   - Artificial signal injected from future periods
   - Model learns from information not available at prediction time
   - Overstates forecast accuracy (model has unfair advantage)

### **Correct Approach (Any of These)**

**Option A (Preferred): Exclude Early Years**
```python
# Years 1-4 undefined for features requiring 5-year lookback
# Training data: years 5 onwards
# n_train = 63 provinces × 10 years = 630 (vs 882 current)
# Higher quality than 252 artificial imputations
```

**Option B: Forward-Fill or Local Imputation**
```python
# For undefined rolling statistics at year t:
# Impute from local past data only (t-1, t-2, ..., max lookback)
# NOT from future (t+1, t+2, ...)
# Preserves causality
```

**Option C: Adjust Feature Engineering**
```python
# Instead of 5-year rolling mean for all observations:
# Use 1-year, 2-year, 3-year rolling means adaptively
# Defined for all observations
```

### **Impact on Results**

- **Overstated forecast R² values**
  - Reported: R² = 0.7788 (holdout)
  - Actual: likely lower (depends on leakage magnitude)
  
- **Validity of uncertainty quantification**
  - Conformal intervals may be underestimating true uncertainty
  - Coverage validation may be false

---

## **CRITICAL #5: Feature Dimensionality Unspecified (Reproducibility Critical)**

### **Location**  
Temporal Feature Engineering, page 980-1000

### **The Problem**

Paper claims "$p ≈ 180$" total features but never specifies exact dimensionality.

12 feature blocks × 8 criteria = ? features

**From description:**
- Block 1 (Current levels): K = 8
- Block 2 (Lags + missingness): 3 lags × 2 (missingness) × K = 6K = 48 
- Block 3 (Rolling stats): 4 stats × 3 windows × K = 12K = 96
- Block 4 (Momentum): 2 types × K = 2K = 16
- Block 5 (Entity-demeaned): K = 8
- Block 6 (Linear trend): K = 8
- Block 7 (EWMA): 3 decay × K = 3K = 24
- Block 8 (Expanding mean): K = 8
- Block 9 (Cross-criterion diversity): 2 types = 2 (NOT criterion-specific)
- Block 10 (Rolling skewness): K = 8
- Block 11 (Panel-relative): 3 types = 3 (per criterion = 3K = 24)
- Block 12 (Geographic): 5 dummies = 5 (NOT criterion-specific)

**Rough Total**: 8 + 48 + 96 + 16 + 8 + 8 + 24 + 8 + 2 + 8 + 24 + 5 = **263**

But paper says "≈ 180". Either:
1. Many features not criterion-specific (misunderstood description)
2. Some blocks excluded
3. Calculation wrong

**This uncertainty is a reproducibility nightmare.**

### **Required Fix**

Must provide explicit feature count and mapping:
```
Total features: p = [EXACT NUMBER]
  - Criterion-specific: p_criteria = [NUMBER] (across 8 criteria)
  - Fixed features: p_fixed = [NUMBER] (geographic, etc.)
```

And create feature name mapping:
```
Feature_1 = Level_C01
Feature_2 = Level_C02
...
Feature_47 = Lag1_C03
...
Feature_180 = Geographic_Delta
```

---

## **CRITICAL #6: Meta-Learner Optimization Under-Specified**

### **Location**  
Super Learner Meta-Ensemble, pages 1070-1085

### **The Problem**

Optimization formulation stated as:
```
min_α ||y^(k) - Ẑ^(k) α||²₂ + λ||α||²₂
subject to: α ≥ 0, 1ᵀα = 1
```

Problem: This is NOT standard ridge regression!

1. **Ridge regression** = unconstrained optimization with L2 penalty
2. **Constrained ridge** = partially different beast
3. **Non-negative constrained quadratic program** = requires specialized solver

### **Critical Ambiguities**

1. "The optimization problem is solved via Ridge regression $\ell_2$-regularized least squares"
   - Does NOT match the stated problem
   - Standard ridge: min ||y - Xα||² + λ||α||² (no constraints)
   - Stated problem has constraints

2. "superior numerical stability compared to non-negative least squares (NNLS)"
   - NNLS is standard, well-understood, widely tested
   - Why switch to hybrid formulation without comparing?
   - No numerical stability comparison provided

3. "regularization parameter λ is chosen via internal cross-validation"
   - How? Grid search? Cross-validation on what?
   - How many λ candidates tested?

### **Reproducibility Nightmare**

Cannot reproduce without knowing:
- Optimization algorithm (SLSQP? CVXPY? Custom code?)
- Solver parameters
- Convergence tolerance
- λ selection procedure

### **Required Fix**

Specify ONE of:
1. **Use Standard NNLS** (recommended for simplicity)
2. **Constrained Ridge** (if hybrid desired):
   - Specify solver: cvxpy, scipy.optimize, etc.
   - Show λ selection procedure with examples
   - Compare against NNLS baseline
   - Validate numerical stability claim

---

## **CRITICAL #7: Holdout Evaluation Protocol Ambiguous**

### **Location**  
Ensemble section (pages 1200-1230)

### **The Problem**

Paper claims:
> "Holdout $R² = 0.7788$ exceeds cross-validation mean (0.7078), indicating robust generalization and absence of overfitting."

But critical question: **Was holdout data truly held out from meta-learner training?**

### **The Risk**

**Scenario A (Correct):**
```
1. Training data (2011-2022)
2. Holdout data (2023-2024) - never seen during:
   - Base model training
   - Meta-learner training (OOF generation)
   - Meta-weight optimization
3. Test on holdout → true generalization estimate
```

**Scenario B (Incorrect):**
```
1. All data (2011-2024)
2. 5-fold temporal CV (each fold validates on 1 year)
3. OOF matrix ← includes all years, including what will be "holdout"
4. Meta-weights trained on OOF including holdout year overlap
5. Test "holdout" that was already in OOF data
→ Optimistic bias
```

### **Paper's Description**

"Walk-forward panel cross-validation over calendar years rather than random temporal splits...fold s trains on all years ≤ τ_s - 1 and validates on year τ_s. The resulting OOF matrix is therefore a genuine estimate of generalization performance."

This suggests walk-forward is correct, BUT:

1. How many folds? If 5 folds over 14 years, each fold ~2-3 years
2. Which years in holdout? 2023-2024? Last 2 years?
3. If OOF uses those years, they're not truly "held out"

### **Impact**

- If holdout was in OOF: R² = 0.7788 is OPTIMISTIC BIAS
- Meta-weights overfit to those provinces/years
- True generalization R² likely lower

### **Required Fix**

Explicitly state:
```
Train/Validation Split (for meta-learner):
- Years 2011-2019 (84 month observations): train base models & OOF
- Years 2020-2022 (42 month observations): meta-learner training (OOF)
- Years 2023-2024 (TRULY HOLDOUT): final test

Reported R²=0.7788 from years 2023-2024 only ✓
```

Or if current protocol:
```
We acknowledge that years [X-Y] appear in both OOF and formal holdout;
effective holdout is years [Z-W] only, yielding R²=[ADJUSTED]
```

---

# SERIOUS ISSUES (Major Revisions Needed)

## **SERIOUS #1: CatBoost Weight Allocation Unjustified**

### **Location**  
Meta-Learner Results, Table ~1215-1225

### **The Issue**

```
Individual R² Performance vs. Meta-Weight Allocation:
                R²        Meta-Weight
CatBoost        0.6778    45.2%  ← LOWEST R² gets HIGHEST weight
ElasticNet      0.7323    10.3%  ← HIGHEST R² gets LOWEST weight  
SVR             0.7153    14.5%
Bayesian Ridge  0.6843    29.9%
```

### **Paper's Explanation**

"CatBoost captures joint multi-criterion correlations through its multi-output tree structure...the meta-learner learns that this diversity...significantly enhances the combined prediction."

### **Problems with This Explanation**

1. **No Evidence Provided**
   - Claims CatBoost captures correlations—but how?
   - Base learner performance metrics don't show correlation structure
   - No diagnostic table: cross-criterion prediction accuracy for each model

2. **Alternative Explanations Not Ruled Out**
   - Random variance in fold structure favoring CatBoost?
   - Regularization parameters better tuned?
   - OOF-specific quirk unrelated to true generalization power?

3. **Weighting Stability Not Tested**
   - Are meta-weights robust?
   - Bootstrap resampling of OOF data → stable α estimates?
   - Leave-one-base-learner-out → how much does each contribute?

4. **Ablation Studies Missing**
   - Model performance if only CatBoost? 
   - If CatBoost removed?
   - Each base model in isolation vs. pairs vs. full ensemble?

### **Required Fix**

Provide evidence:
1. **Criterion correlation analysis**
   - Show CatBoost's multi-output structure actually exploits criterion correlations
   - Compare to separate univariate models
   - Quantify this benefit

2. **Ablation study**
   - Leave-one-out analysis: remove each model, retrain meta-learner
   - Report performance drop for each
   - Validates claimed 45.2% contribution

3. **Meta-weight stability**
   - Bootstrap resampling: are α values stable?
   - Perturb OOF matrix: do rankings change?
   - Confidence intervals on meta-weights

4. **Alternative allocation**
   - What if weights matched individual R²? (ElasticNet=37%, etc.)
   - Performance comparison
   - Can explain divergence

---

## **SERIOUS #2: Temporal Non-Stationarity Ignored**

### **Location**  
CV results, Table ~1220

### **The Evidence**

Fold-wise R² for CatBoost:
```
Fold 1: 0.6932
Fold 2: 0.6901  
Fold 3: 0.6725
Fold 4: 0.6960
Fold 5: 0.6370  ← 8.8% drop from Fold 1
```

Bayesian Ridge:
```
Fold 1: 0.7179
Fold 2: 0.5839  ← 19% DROP
Fold 3: 0.6987
Fold 4: 0.6700
Fold 5: 0.7508
```

### **Why This Matters**

1. **Non-stationary dynamics**
   - Performance varies significantly across temporal folds
   - If Fold 5 = 2024 (most recent), recent years harder to forecast
   - Or earlier years easier (historical data more stable)

2. **Invalidates "stable" meta-learner assumption**
   - Meta-weights estimated on folds 1-4 may not work for 2024
   - Suggests different models optimal for different time periods

3. **Suggests Data Quality or Governance Changes**
   - Why do some years forecast worse?
   - Administrative changes? Data collection shifts? Genuine governance volatility?

### **Critical Question**

**Which years map to which folds?** Paper never specifies!

### **Required Fix**

1. Explicitly map folds to years:
```
Fold 1: 2011-2014 (training ≤2010, test 2014)
Fold 2: 2012-2015 (training ≤2011, test 2015)
...
Fold 5: 2015-2018 (training ≤2014, test 2018)
```

Or if 5-fold over 14 years differently structured, explain mapping.

2. Analyze temporal heterogeneity:
```python
Plot: R² vs. year
- Identify years with lowest/highest accuracy
- Statistically test for trend (linear? structural break?)
- Discuss institutional causes
```

3. Consider regime-switching if needed:
```
If recent years (2022-2024) harder (low R²):
- Governance more volatile? 
- Data quality degraded?
- Methodological mismatch?
- Consider separate models for pre/post structural change
```

---

## **SERIOUS #3: Promethee Hyperparameters Unjustified**

### **Location**  
PROMETHEE II method, pages 735-740

### **The Issue**

```
Preference threshold: p = 0.3
Indifference threshold: q = 0.1
```

No justification. No sensitivity analysis.

### **The Risk**

PROMETHEE results depend strongly on these parameters:

**Example:**
```
Province A vs B on criterion j:
  A score: 0.50, B score: 0.48, difference d = 0.02

With q=0.1: d < q → indifference → P_j(A,B) = 0
With q=0.05: d > q → partial preference → P_j(A,B) = (0.02-0.05)/(p-0.05) < 0

Rankings flip!
```

### **Why Cited Values Are Suspicious**

1. (q, p) = (0.1, 0.3) is merely a common "example" in PROMETHEE textbooks
2. Paper gives NO justification why these fit PAPI governance data
3. No sensitivity test: what if (0.05, 0.25) or (0.15, 0.35)?

### **Required Fix**

**Option A (Minimal):** Sensitivity Analysis
```python
for q in [0.05, 0.10, 0.15]:
    for p in [0.20, 0.30, 0.40]:
        Run PROMETHEE
        Compute inter-method rank correlation
        
Report rank correlation across parameter choices
If ρ > 0.90 → parameter-insensitive, OK
If ρ < 0.70 → parameter-sensitive, need justification
```

**Option B (Better):** Governance-Motivated Parameters
- Justify in context: 
  - q = 0.1 because governance differences < 10% not meaningful?
  - p = 0.3 because differences > 30% are definitively conclusive?
- Support with governance expert input

---

## **SERIOUS #4: Conformal Prediction Details Insufficient**

### **Location**  
Conformal Prediction section, pages 1105-1115

### **The Problems**

1. **Quantile Random Forest Implementation Missing**
   - How many trees?
   - Max depth?
   - Split criteria?
   - Quantile loss function: how implemented?
   - Package: scikit-learn? other?

2. **CQR Variant Chosen But Not Cited**
   - "Conformalized Quantile Regression" has multiple variants
   - Romano et al. (2019) cited, but which specific algorithm?
   - Exact conformity score construction required

3. **Over-Coverage Not Discussed**
   - Empirical coverage 96% >> nominal 90%
   - Indicates unnecessarily wide intervals
   - Why not tighten to achieve exactly 90%?
   - Adaptive conformal prediction considered?

### **Impact**

- **Reproducibility impossible** without exact specifications
- **Interval width potentially wasteful** (96% coverage = 6% "unused" confidence)
- **Validity claim unverified** (is 96% coverage by chance or by design?)

### **Required Minimal Fix**

Specify:
```python
# QRF Configuration
n_estimators = [?]
max_depth = [?]
quantiles = [α/2, 1-α/2]

# CQR procedure
conformity_score = function of residuals and quantile bounds
cal_quantile = 0.90-quantile of conformity scores
prediction_interval = [q_α/2 - cal_quantile, q_{1-α/2} + cal_quantile]
```

---

## **SERIOUS #5: Weight Temporal Stability Tests Wrong Metric**

### **Location**  
Temporal Stability Analysis, pages 710-730

### **The Issue**

Paper tests rank correlation of WEIGHT VECTORS across windows:
$$\rho(rank(\bar{\mathbf{w}}_k), rank(\bar{\mathbf{w}}_{k+1}))$$

Interprets high ρ as "weights are stable."

**But**: This tests whether criterion WEIGHT RANKINGS are stable, not whether provincial RANKINGS or OUTCOMES are stable!

### **Example of the Problem**

```
Window 1 weights: C06=16.8%, C05=14.6%, C01=14.2%, ...
Window 2 weights: C06=17.1%, C05=14.5%, C01=14.3%, ...
Weight rank correlation: ρ ≈ 0.99 (extremely high)

BUT provincial governance outcomes could be

 completely reordered!
```

Why? Suppose province A:
- Window 1: Strong in C06 (16.8% weight) → high overall score
- Window 2: Strong in C06 (17.1% weight) → similar score (but maybe small change cascades)

---

**CONTINUED IN NEXT SECTION...**
