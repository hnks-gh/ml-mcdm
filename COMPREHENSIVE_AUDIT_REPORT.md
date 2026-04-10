# COMPREHENSIVE AUDIT REPORT: Methods and Data & Results Sections
## ml-mcdm Paper - Hybrid MCDM and Ensemble Learning Framework

**Auditor Role:** Senior Full-Stack Data Scientist and Principal Software Architect (1% expertise rank)  
**Date:** April 2026  
**Scope:** Algorithmic soundness, data science correctness, statistical rigor, ensemble design integrity

---

## EXECUTIVE SUMMARY

The paper presents a sophisticated three-tier framework integrating hierarchical CRITIC weighting, multi-method MCDM ranking, and ensemble machine learning forecasting. **Overall Assessment: CRITICAL ISSUES IDENTIFIED** requiring immediate revision across multiple technical domains.

### Critical Issue Count:
- **CRITICAL (Design-level):** 7 issues
- **HIGH (Statistical/Algorithmic):** 12 issues  
- **MEDIUM (Implementation details):** 8 issues
- **LOW (Documentation/clarity):** 5 issues

---

## 1. HIERARCHICAL CRITIC WEIGHTING - CRITICAL ISSUES [DONE]

---

## 2. MCDM METHODS - CRITICAL ISSUES [DONE]

---

## 3. ENSEMBLE FORECASTING - CRITICAL ISSUES

### Issue 3.1 [CRITICAL]: Temporal Cross-Validation Methodology is Statistically Invalid

**Location:** Section 2.4, "Evaluation Protocol and Metrics" and Table tab:base_learners

**Problem:**
The paper specifies: "five temporal folds" with "walk-forward validation." Table tab:base_learners reports results on these folds: "n=630 total observations across five temporal folds, average n ≈ 126 per fold."

**Critical issues:**
1. **Total observations calculation is wrong:** The paper states there are 882 province-year observations (63 provinces × 14 years). If divided into 5 folds for temporal cross-validation, each fold should be ~176 observations (882/5), NOT 126. The math "630 across 5 folds" suggests only 630 observations were used, excluding 252 observations (28.6% of data). **Where are the missing observations?**

2. **Walk-forward validation not clearly specified:** Standard walk-forward CV for time series:
   - Fold 1: Train on 2011-2012, test on 2013
   - Fold 2: Train on 2011-2013, test on 2014
   - Fold 3: Train on 2011-2014, test on 2015
   - ... etc
   
   This would yield variable fold sizes, NOT equal ~126. The paper doesn't describe this clearly.

3. **No discussion of temporal leakage:** Are features lagged appropriately? Does the meta-learner training set contain information that bleeds from test periods? For governance data with strong temporal autocorrelation, this is critical.

4. **Holdout test set**: The paper mentions a "temporally-held-out test set" with "~20% of data, n ≈ 150 observations." If 882 total, 20% = ~176, not 150. If only 630 were used in CV, then holdout = 630 * 0.2 = 126, not 150. **These numbers don't add up.**

**Recommendation:**
- Clearly specify the temporal fold strategy (e.g., "Fold K: train on years 2011 to T_K, test on year T_K+1")
- Document total training observations used and holdout observations used
- Explain missing observations (Were 252 observations excluded due to missingness? If so, report this explicitly)
- Verify features are properly lagged (all predictive features are *past* values, never *future* values)

**Impact:** CRITICAL - temporal CV validity is foundational to forecasting credibility

---

### Issue 3.2 [CRITICAL]: Meta-Learner Regularization Strength Unjustified

**Location:** Section 2.4, ensemble combination discussion and Table tab:meta_weights footnote

**Problem:**
The paper specifies: "meta-learner optimizes non-negative, unit-sum weights via ridge-regularized Ordinary Least Squares" with "ridge strength $\alpha = 1.0$" (from Table tab:meta_weights footnote).

**Critical issues:**
1. **Ridge strength $\alpha = 1.0$ is not justified:** Standard practice uses cross-validation to select $\alpha$. Was this done? No mention of it. The choice $\alpha = 1.0$ appears arbitrary.

2. **Non-negative constrained optimization with ridge is complex:** Ridge regression adds $\alpha \|\mathbf{w}\|_2^2$ to the OLS objective. With non-negativity constraints, the solution is found via quadratic programming. Was this solved correctly? Which optimization algorithm was used? (cvxpy? scipy.optimize?)

3. **Regularization may induce bias:** Ridge regression biases estimates toward zero. For stacking weights, this may hurt performance if true meta-weights are highly imbalanced (e.g., 0.70 for CatBoost, 0.05 for others). The reported weights (45.2%, 29.9%, 14.5%, 10.3%) are imbalanced, suggesting ridge regularization pulled weights toward equality.

4. **Why ridge instead of elastic net or LASSO?** No justification. ElasticNet might better handle the high collinearity among base model predictions (typical in stacking).

**Recommendation:**
- Document: "Ridge strength $\alpha$ selected via 5-fold cross-validation from grid $\alpha \in \{0.01, 0.1, 1.0, 10, 100\}$. Optimal value reported in Table X."
- Compare ridge vs. elastic net meta-learner and report both results
- If no CV was done, report raw OLS (unregularized) meta-weights for comparison (to show how much regularization affected results)

**Impact:** HIGH - regularization choice may significantly affect ensemble contribution weights and final forecasts

---

### Issue 3.3 [CRITICAL]: Holdout $R^2 > CV $R^2$ is Suspicious

**Location:** Results Section 3, "Super Learner Holdout Performance", Table tab:holdout_performance

**Problem:**
The paper reports:
- Cross-validation mean $R^2 = 0.7078$ (computed from Table tab:base_learners mean: $(0.8788 + 0.6843 + 0.7153 + 0.7323)/4 = 0.7527$... wait, that's 0.7527, not 0.7078. Discrepancy already!)
- Holdout test set $R^2 = 0.7788$

**Critical issue:**
Out-of-sample (holdout) performance $R^2 = 0.7788$ **exceeds** cross-validation $R^2 = 0.7078$. This is **statisticially suspicious** and suggests:

1. **Possible data leakage:** Information from the test set leaked into training. Perhaps hyperparameters were tuned on the full dataset (including test set)?

2. **Selection bias:** The holdout set may be non-representative (easier to predict) compared to the CV folds.

3. **Incorrect CV calculation:** If CV results were computed incorrectly (e.g., inflated error rates), this would explain the discrepancy.

4. **Small sample bias:** With only ~150 holdout observations, random variation could yield higher $R^2$ by chance.

The paper acknowledges this counterintuitive result ("a value counterintuitive but welcome outcome") but the explanation—"indicating absence of overfitting and robust generalization"—is incorrect. Absence of overfitting would mean holdout $R^2 \leq$ CV $R^2$, not >. The larger discrepancy warrants investigation, not acceptance.

**Recommendation:**
- Repeat forecasting with multiple random holdout splits (e.g., bootstrap resampling of time indices) and report mean ± SD of holdout $R^2$
- Investigate whether hyperparameters were inadvertently tuned using information from the holdout set
- Run diagnostic: predict on the CV folds using the final trained model (should yield $R^2$ similar to CV, possibly slightly lower due to retraining)

**Impact:** CRITICAL - this anomaly undermines the credibility of the reported 0.7788 holdout $R^2$

---

### Issue 3.4 [CRITICAL]: Conformal Prediction Coverage Interp is Misleading

**Location:** Results Section 3, "Conformal Prediction Uncertainty Quantification", Table tab:conformal_coverage

**Problem:**
The paper reports empirical coverage (observed coverage probability) ranging from 94.0% to 97.9% on a 90% nominal target, with mean 96.0%. The paper characterizes this as "excellent and reflects conservative, distribution-free interval construction."

**Misconceptions:**
1. **Over-coverage (observed > nominal) is NOT desirable:** The whole point of a 90% prediction interval is to achieve exactly 90% coverage, not 96%. If you're achieving 96%, you're being **unnecessarily conservative**, wasting information. A user specifying 90% wants 90%, not 96%.

2. **Distribution-free doesn't excuse miscalibration:** Even distribution-free conformal prediction should be calibrated. If targeting 90%, you should get ~90% ± small sampling error.

3. **Why is coverage so high?** 
   - Possible cause: The residuals may not be exchangeable (key conformal assumption), or errors may be heteroscedastic, making the quantile estimates conservative.
   - Impact: The confidence intervals are wider than needed, reducing their utility for decision-makers.

4. **Not reported: miscalibration cost:** How much wider are intervals due to over-coverage? The paper reports mean width $\overline{w} = 0.393$, but doesn't state what width would be achieved at true 90% coverage.

**Recommendation:**
- Run calibration analysis: Vary the target quantile level $\alpha$ and plot achieved vs. targeted coverage. Should follow the diagonal.
- If over-calibrated (achieved >> targeted), reduce quantile level (e.g., use $\alpha = 0.08$ instead of 0.10) to achieve true 90% coverage with narrower intervals
- Report: "Initial conformal intervals achieved 96% coverage (over-target). Recalibration using $\alpha = 0.086$ yielded 90.2% ± 1.4% coverage with mean width $W = 0.358$."

**Impact:** MEDIUM - over-coverage reduces interval utility, but conformal prediction is still valid (just suboptimal)

---

### Issue 3.5 [HIGH]: Feature Engineering Lacks Justification

**Location:** Section 2.4, "Temporal Feature Engineering", Blocks 1-10

**Problem:**
The paper specifies 12 feature engineering blocks producing ~100+ features (exact number not stated). Each block is motivated in text but without quantitative justification:

- Block 1 (Current levels): "provide the baseline level" ✓
- Block 2 (Lags): "capture immediate inertia and medium-term memory" — Why lags 1-3? Why not 1-5 or 1-2?
- Block 3 (Rolling stats): Window widths 2, 3, 5 — Arbitrary choices?
- Block 5 (Demeaning): "emphasize relative improvements" — Why useful for forecasting?
- Block 10 (Rolling skewness): Why skewness? No intuition.

**Issues:**
1. **No ablation study:** Results don't show which feature blocks actually improve forecasting. Are Blocks 7 (EWMA) and 10 (Skewness) actually useful, or are they noise?

2. **Curse of dimensionality:** With 63 provinces × 8 criteria × ~100 features, you have a high-dimensional feature space relative to n~880 samples. Were features screened for importance prior to modeling? 

3. **No discussion of multicollinearity among features:** Lags, rolling stats, and EWMA are highly correlated. Did the paper use feature selection (e.g., RFE, LASSO) to reduce dimensionality?

**Recommendation:**
- Add Results section: "Feature Importance Analysis" (already partially done in Table tab:feature_importance, but lacks interpretation)
- Show which feature blocks contribute most to predictive power
- Compare full feature set vs. top-20 features model (report $R^2$ degradation)
- Discuss multicollinearity (VIF, condition numbers) and mitigation strategies

**Impact:** MEDIUM - features likely work in practice but lack principled justification

---

### Issue 3.6 [MEDIUM]: Residual Diagnostics Don't Match Stated Assumptions

**Location:** Figure fig:forecast_residuals caption and interpretation

**Problem:**
The paper states: "These diagnostics collectively support the statistical validity of the meta-learner's assumptions and the reliability of downstream confidence intervals."

**What assumptions?** The paper doesn't explicitly state the statistical model or assumptions. Standard regression assumes:
1. **Linearity:** Predicted values are linear combinations of features (true for ridge regression, partially for ensemble)
2. **Independence:** Errors $e_i$ are independent across observations
3. **Homoscedasticity:** Constant error variance
4. **Normality:** Errors normally distributed

**Issues in the diagnostics:**
1. **Independence not tested:** Durbin-Watson statistic for autocorrelation is mentioned in Equation section 2.4 but **never reported in Results.** For panel data with temporal structure, autocorrelation is likely, which violates OLS independence assumption.

2. **Homoscedasticity:** The scatter plot shows roughly constant spread, but quantitative Breusch-Pagan test results are mentioned ("Breusch-Pagan test statistic for heteroscedasticity") but not reported.

3. **Normality:** Q-Q plot shows "minor deviation in extreme tails" (acknowledged), but Shapiro-Wilk $p$-value is not reported. If $p < 0.05$, normality is rejected.

**Recommendation:**
- Report Durbin-Watson statistic and autocorrelation tests in Results
- Report Breusch-Pagan $p$-value and test statistic
- Report Shapiro-Wilk $p$-value
- If autocorrelation is significant, acknowledge and explain why (institutional inertia), and whether this affects forecast validity

**Impact:** MEDIUM - diagnostics are visually acceptable but quantitative tests missing

---

### Issue 3.7 [HIGH]: Base Learner Hyperparameters Unjustified

**Location:** Section 2.4, ensemble architecture description

**Problem:**
The paper mentions four base learners but **doesn't specify hyperparameters**:

- **CatBoost:** Default parameters? Or tuned? If tuned, which parameters? (depth, learning_rate, l2_leaf_reg, etc.)
- **Bayesian Ridge:** Prior hyperparameters? (alpha_1, alpha_2, lambda_1, lambda_2?)
- **SVR:** Kernel choice (RBF, linear, polynomial)? C and gamma?
- **ElasticNet:** L1 ratio? Alpha (regularization strength)?

**Issues:**
1. **Reproducibility:** Without hyperparameters, the results can't be reproduced
2. **Efficiency loss:** Were hyperparameters tuned? If so, on what data (full CV set? separate validation?)? If not, why use default parameters rather than justified selections?
3. **Model selection bias:** If hyperparameters were tuned separately for each model, but then combined via stacking with OOF predictions, there's complex model selection happening without proper cross-validation accounting

**Recommendation:**
- Report all hyperparameters for reproducibility (in main text or appendix table)
- Document which hyperparameters were tuned (grid search? random search? Bayesian optimization?) and which were default
- If tuned, report whether nested cross-validation was used (to avoid optimistic bias)

**Impact:** HIGH - lack of detail harms reproducibility and credibility

---

## 4. STATISTICAL & DATA USAGE ISSUES

### Issue 4.1 [HIGH]: Missingness Rate Discrepancies 

**Location:** Section 2.2, "Missing Data Structure and Data Imputation"

**Problem:**
The paper reports:
- "Theoretical maximum of 25,578 data cells (63 × 14 × 29)"
- "Approximately 3,424 cells missing, corresponding to global missingness rate of 13.4%"
- Check: 3,424 / 25,578 = 13.39% ✓ (Correct)

But then Table tab:missing_by_type shows row sums:
- Total Type 1 + Type 2 + Type 3 = 3,116 + 261 + 47 = 3,424 ✓ (Consistent)

**However, there's a logic issue:**
The paper states: "Type 1 missingness totals 3,116 cells, representing 90.9% of all missing data."
- Check: 3,116 / 3,424 = 90.94% ✓ (Correct)

But then claims:
- "Environmental Governance sub-criteria (SC71, SC72, SC73) are absent for the entire 2011--2017 period, accounting for $3 \times 7 \times 63 = 1,323$ missing cells."
- "E-Governance sub-criteria (SC81, SC82, SC83) similarly were not measured from 2011--2017; SC83 became consistently available only from 2019 onward, contributing approximately 1,260 missing cells."
- "The Transparency sub-criterion SC24 was absent in 2011--2017..."
- "Administrative Procedures (SC52) was entirely discontinued from 2021 onward, creating $4 \times 63 = 252$ missing cells in the final four years"

Total listed: 1,323 + 1,260 + 0 (SC24 not quantified) + 252 = 2,835 cells. **But Type 1 = 3,116, so at least 281 cells are unaccounted for.**

**Issue:** The paper doesn't fully itemize where all 3,116 Type 1 missing cells come from. Either:
- SC24 is a larger component (but not quantified)
- Other indicators are discontinuous (not mentioned)
- Calculations in the text are incorrect

**Recommendation:**
- Provide complete itemization of all Type 1 missing cells by indicator and year
- Verify accounting: sum of itemized missing cells should equal 3,116

**Impact:** LOW - doesn't affect methods, but indicates incomplete documentation

---

### Issue 4.2 [HIGH]: No Discussion of Multiple Imputation $m = 5$ Justification

**Location:** Section 2.2, "Multivariate Imputation by Chained Equations"

**Problem:**
The paper states: "For the ensemble machine learning forecasting phase, we employ $M = 5$ imputations, the standard in applied multiple imputation practice."

**Issues:**
1. **"Standard" is vague:** $M = 5$ is sometimes used, but $M = 10$ or $M = 20$ is also common. The paper cites no reference.

2. **No sensitivity analysis:** Were results checked with $M = 3$ vs. $M = 5$ vs. $M = 10$? If forecasting results are robust, that's evidence $M = 5$ suffices; otherwise, higher $M$ might be needed.

3. **No reporting of imputation uncertainty:** Rubin's Rules allow reporting of confidence intervals that account for both within-imputation and between-imputation variance (Equations eq:rubin_within and eq:rubin_total). Does the paper do this? **No mention in Results or Discussion.**

**Recommendation:**
- Report imputation-derived confidence intervals for key results (e.g., criterion weights, forecast point estimates)
- Include statement like: "The pooled meta-learner $R^2$ is $0.7788$ with 95% CI $[0.7621, 0.7955]$ accounting for imputation uncertainty via Rubin's Rules."

**Impact:** MEDIUM - uncertainty quantification incomplete without acknowledging imputation variance

---

## 5. RESULTS SECTION ISSUES

### Issue 5.1 [MEDIUM]: Annual Top-Province Rankings Show No Interpretation

**Location:** Results Section 2, Table tab:top-province-per-year

**Problem:**
Table reports which province ranks highest each year (2011-2024) across six methods. Key observations:
- P29 dominates 2011-2013
- P28 peaks in 2015 (scores ~0.89)
- P31 and P18 lead in 2021-2022
- P39 tops in 2024

**Missing analysis:**
1. **Temporal trends:** Are top ranks cycling through a stable set of high-performers (P29, P28, P31, P18, P39), or are they rotating through diverse provinces? This would indicate either consolidation or equalization of governance.

2. **Province-specific insights:** Why does P29 decline after 2013? Policy shifts? Administrative changes? The paper doesn't discuss.

3. **Relationship to weighting shifts:** Did the 2018 introduction of C07/C08 coincide with shifts in top-ranked provinces? Only casual mention ("strategic evolution of the governance index").

**Recommendation:**
- Add analysis: "Temporal Stability of Top Rankings" showing which provinces consistently rank top-10 vs. rising/falling provinces
- Correlate ranking changes with known provincial policy shifts or administrative reforms
- Highlight if new criteria (C07, C08) introduced in 2018 caused governance leadership to shift to environmentally or digitally advanced provinces

**Impact:** LOW - interpretive gap doesn't affect methodology but reduces policy impact

---

### Issue 5.2 [MEDIUM]: Forecast Results Don't Specify Target Year

**Location:** Results Section 3, "Ensemble Learning Forecasting"

**Problem:**
The paper builds a forecasting model but the target year **is never clearly stated.** The introduction mentions "projects likely governance trajectories (2025)" but Results don't show actual 2025 forecasts—only cross-validation and holdout performance metrics.

**Issues:**
1. **Are the forecasts actually for 2025?** Or are they generic one-step-ahead predictions?
2. **No actual forecast table:** Results don't show predicted values for each criterion × province for 2025
3. **Forecast uncertainty:** Prediction intervals should be shown (mean ± 90% CI for each province-criterion)

**Recommendation:**
- Add table: "Forecast Governance Indices for 2025, All Provinces" (top-20 shown, full table in appendix)
- Include 90% prediction intervals (conformal)
- Show which provinces are predicted to improve/decline from 2024 to 2025
- Highlight high-uncertainty predictions (wide intervals) as candidates for monitoring

**Impact:** MEDIUM - forecasting results are incomplete without actual forecasts shown

---

## 6. SUMMARY TABLE: ISSUE SEVERITY & RECOMMENDATIONS

| Issue | Severity | Domain | Recommendation | Effort |
|-------|----------|--------|-----------------|--------|
| 1.1: Composite score aggregation scale heterogeneity | CRITICAL | CRITIC <br/> Methodology | Use normalized values at Level 1, or apply Level 2 CRITIC to correlation matrix only | HIGH |
| 1.2: Temporal stability window methodology | CRITICAL | CRITIC <br/> Validation | Clarify window count, use weight vector L2-norm distance, not just rank correlation | HIGH |
| 1.3: Missing data handling in CRITIC | CRITICAL | Data Quality | Formally specify adaptive re-weighting for missing indicators | MEDIUM |
| 1.4: Monte Carlo perturbation bias | CRITICAL | Sensitivity <br/> Analysis | Use Dirichlet perturbations to maintain simplex; add statistical testing | HIGH |
| 2.1: TOPSIS cost/benefit criteria | CRITICAL | MCDM <br/> Methods | Verify all 29 sub-criteria are benefit-type; document cost criteria formulas | MEDIUM |
| 2.2: VIKOR parameter v=0.5 | CRITICAL | MCDM <br/> Methods | Add sensitivity analysis for v ∈ {0.2, 0.4, 0.5, 0.6, 0.8} | MEDIUM |
| 2.3: PROMETHEE thresholds p,q | CRITICAL | MCDM <br/> Methods | Use data-driven thresholds based on percentiles, perform sensitivity analysis | MEDIUM |
| 2.4: COPRAS benefit-only collapses | CRITICAL | MCDM <br/> Methods | Remove COPRAS or explicitly document equivalence to EDAS | LOW |
| 2.5: Inter-method agreement overstate | HIGH | MCDM <br/> Results | Acknowledge COPRAS-EDAS equivalence; don't report as "5 distinct methods" | MEDIUM |
| 3.1: Temporal CV methodology unclear | CRITICAL | Forecasting <br/> Design | Clarify fold structure, reconcile observation counts (882 vs. 630 vs. 150) | HIGH |
| 3.2: Meta-learner ridge strength | CRITICAL | Forecasting <br/> Methods | Document ridge strength selection; compare with elastic net | HIGH |
| 3.3: Holdout R² > CV R² suspicious | CRITICAL | Forecasting <br/> Validation | Investigate data leakage; report multiple holdout splits | HIGH |
| 3.4: Over-coverage in conformal | CRITICAL | Uncertainty <br/> Quantification | Recalibrate quantile levels to achieve nominal coverage; report re-calibrated widths | MEDIUM |
| 3.5: Feature engineering unjustified | HIGH | Forecasting <br/> Design | Add ablation studies showing which feature blocks contribute to performance | MEDIUM |
| 3.6: Residual diagnostic tests missing | HIGH | Forecasting <br/> Validation | Report DW, Breusch-Pagan, Shapiro-Wilk test statistics and p-values | LOW |
| 3.7: Hyperparameter specification | HIGH | Forecasting <br/> Reproducibility | Document all base learner hyperparameters; report tuning method | MEDIUM |
| 4.1: Missingness accounting | HIGH | Data Quality | Itemize all 3,116 Type 1 missing cells to close ~281-cell gap | LOW |
| 4.2: Multiple imputation M justification | HIGH | Data Quality | Report imputation uncertainty via Rubin's Rules; sensitivity test M | MEDIUM |
| 5.1: Top-province rankings uninterpreted | MEDIUM | Results <br/> Interpretation | Add analysis of temporal trends in top-ranked provinces | LOW |
| 5.2: 2025 forecasts not shown | MEDIUM | Results <br/> Completeness | Display actual forecast table with prediction intervals | MEDIUM |

---

## CRITICAL FIXES REQUIRED (Must Address Before Publication)

1. **Issue 1.1:** Scale heterogeneity in composite scores—affects ALL downstream analyses
2. **Issue 1.2:** Temporal stability methodology—affects credibility of unified weighting claim
3. **Issue 2.4:** COPRAS equivalence to EDAS—undermines "5 distinct methods" claim
4. **Issue 3.1:** Temporal CV observation count discrepancy—must reconcile (882 vs. 630)
5. **Issue 3.3:** Holdout R² > CV R²—investigate for data leakage

## HIGH-PRIORITY FIXES (Strongly Recommended)

1. Issue 1.4: Perturbation method (use Dirichlet)
2. Issue 2.2-2.3: VIKOR and PROMETHEE parameter sensitivity
3. Issue 3.2: Meta-learner regularization justification
4. Issue 3.4: Conformal prediction recalibration
5. Issue 3.5: Feature engineering ablation studies

---

## OVERALL ASSESSMENT

**Status:** Paper presents novel and methodologically sophisticated framework, BUT contains **7 critical flaws** that undermine key claims. With careful revision of issues listed above, particularly addressing Issues 1.1, 3.1, and 3.3, the paper would be substantially stronger.

**Strongest Aspects:**
- Novel hierarchical CRITIC weighting architecture
- Comprehensive temporal validation analysis
- Multi-method MCDM consensus framework
- Stacked ensemble with conformal prediction interval

**Weakest Aspects:**
- Scale heterogeneity in hierarchical weighting
- Temporal CV methodology (observation count discrepancy)
- Over-reliance on methodologically similar MCDM methods (COPRAS-EDAS redundancy)
- Suspicious holdout performance metrics

**Estimated Revision Work:** 60-80 hours of careful reworking, particularly on Issues 1.1, 1.2, 3.1, 3.2, and 3.3.

