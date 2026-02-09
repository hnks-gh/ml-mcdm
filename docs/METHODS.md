# ML-MCDM Methods Summary

This document provides a high-level summary of the methods implemented in the ML-MCDM framework. For detailed mathematical formulations, parameters, and usage examples, refer to the documentation in each module's `docs/` folder.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Weight Calculation Methods](#2-weight-calculation-methods)
3. [MCDM Methods](#3-mcdm-methods)
4. [ML Forecasting](#4-ml-forecasting)
5. [Ensemble & Aggregation](#5-ensemble--aggregation)
6. [Analysis & Validation](#6-analysis--validation)

---

## 1. Architecture Overview

The framework processes panel data through multiple analytical stages:

```
Panel Data (Entities × Years × Criteria)
    │
    ├──► Weight Calculation ──► Entropy, CRITIC, PCA, Integrated Hybrid Ensemble
    │
    ├──► Current Year Analysis
    │       ├── Traditional MCDM (5 methods)
    │       └── Fuzzy MCDM (5 methods with uncertainty)
    │
    ├──► ML Forecasting (Random Forest Time-Series)
    │       └── Next Year Predictions (2025)
    │
    ├──► Ensemble Integration
    │       ├── Stacking (Meta-learner)
    │       └── Rank Aggregation (Borda, Copeland, Kemeny-Young)
    │
    └──► Advanced Analysis
            ├── Convergence (σ/β analysis)
            └── Sensitivity (Monte Carlo)
```

---

## 2. Weight Calculation Methods

**Module:** `src/weighting/` | **Details:** [weighting/docs/README.md](../src/weighting/docs/README.md)

### 2.1 Individual Methods

| Method | Purpose | Information Level |
|--------|---------|------------------|
| **Entropy** | Assigns weights based on information content (higher variation = higher weight) | Univariate (order 1) |
| **CRITIC** | Considers both contrast intensity (std dev) and inter-criteria correlation | Bivariate (order 2) |
| **PCA** | Derives weights from full multivariate variance-covariance eigenstructure | Multivariate (order n) |

The three methods form a **complementary triad** — each captures a different level of information from the decision matrix.

### 2.2 Ensemble Strategies

| Strategy | Approach |
|----------|----------|
| **Integrated Hybrid** (default) | Three-stage: PCA→Modified CRITIC→Entropy-weighted integration. PCA residual correlations inform CRITIC, entropy-of-weights determines integration coefficients |
| **Game Theory** | Min-deviation optimization with entropy-based confidence. Methods with more differentiated weights get higher influence |
| **Bayesian Bootstrap** | Bootstrap resampling to estimate method stability. Inverse-variance weighting auto-downweights unstable methods |
| **Geometric Mean** | Product of weights (equivalent to minimum KL-divergence) |
| **Arithmetic Mean** | Weighted sum with configurable method importance |
| **Harmonic Mean** | Reciprocal average, conservative combination |

---

## 3. MCDM Methods

### 3.1 Traditional MCDM

**Module:** `src/mcdm/traditional/` | **Details:** [traditional/docs/README.md](../src/mcdm/traditional/docs/README.md)

| Method | Approach |
|--------|----------|
| **TOPSIS** | Distance to ideal/anti-ideal solutions |
| **VIKOR** | Compromise ranking (group utility + individual regret) |
| **PROMETHEE** | Pairwise preference flows (positive/negative) |
| **COPRAS** | Proportional assessment (benefit/cost separation) |
| **EDAS** | Distance from average solution |

### 3.2 Fuzzy MCDM

**Module:** `src/mcdm/fuzzy/` | **Details:** [fuzzy/docs/README.md](../src/mcdm/fuzzy/docs/README.md)

All traditional methods have fuzzy variants using **Triangular Fuzzy Numbers (TFN)** to incorporate uncertainty from temporal variance in panel data.

- Fuzzy numbers: `(lower, modal, upper)` derived from historical variance
- Supports fuzzy arithmetic operations
- Defuzzification via centroid method

---

## 4. ML Forecasting

**Module:** `src/ml/forecasting/` | **Details:** [forecasting/docs/README.md](../src/ml/forecasting/docs/README.md)

### Unified Forecasting with 7 Models

The project uses a **UnifiedForecaster** that combines 7 different ML models with weighted averaging based on cross-validation performance:

| Model | Type | Purpose |
|-------|------|----------|
| **Gradient Boosting** | Tree Ensemble | Primary predictor with high accuracy |
| **Random Forest** | Tree Ensemble | Robust predictions with feature importance |
| **Extra Trees** | Tree Ensemble | Reduced overfitting via random splits |
| **Bayesian Ridge** | Linear | Uncertainty quantification |
| **Huber Regressor** | Linear | Robust to outliers |
| **Neural MLP** | Neural Network | Non-linear pattern capture |
| **Attention Network** | Neural Network | Temporal dependencies via self-attention |

| Component | Description |
|-----------|-------------|
| **Validation** | Time-series aware cross-validation (no future data leakage) |
| **Features** | Component values, temporal features, lag variables |
| **Outputs** | Feature importance, CV scores, predictions, model weights |

### Model Selection

The `UnifiedForecaster` automatically selects and weights models based on cross-validation R² scores. Models with poor performance are excluded, and remaining models contribute proportionally to their validation accuracy.

---

## 5. Ensemble & Aggregation

**Module:** `src/ensemble/aggregation/` | **Details:** [aggregation/docs/README.md](../src/ensemble/aggregation/docs/README.md)

### Stacking Ensemble

- Combines predictions from multiple MCDM methods
- Meta-learner (Ridge Regression) learns optimal weights
- Outputs final scores and model contribution weights

### Rank Aggregation

| Method | Approach |
|--------|----------|
| **Borda Count** | Point-based aggregation of rankings |
| **Copeland** | Pairwise comparison (wins - losses) |
| **Kemeny** | Optimal ranking minimizing disagreement |

**Quality Metric:** Kendall's W measures agreement between methods.

---

## 6. Analysis & Validation

**Module:** `src/analysis/` | **Details:** [analysis/docs/README.md](../src/analysis/docs/README.md)

### Convergence Analysis

| Type | Description |
|------|-------------|
| **Sigma (σ)** | Coefficient of variation over time (dispersion trend) |
| **Beta (β)** | Regression-based catch-up analysis (speed of convergence) |

### Sensitivity Analysis

- Monte Carlo simulation with weight perturbation
- Identifies criteria with highest impact on rankings
- Overall robustness score

### Validation

- Bootstrap validation for confidence intervals
- Cross-validation metrics (R², MAE, RMSE)
- Rank correlation (Spearman) for prediction accuracy

---

## Module Documentation Links

| Module | Location |
|--------|----------|
| Weight Calculation | `src/weighting/docs/README.md` |
| Traditional MCDM | `src/mcdm/traditional/docs/README.md` |
| Fuzzy MCDM | `src/mcdm/fuzzy/docs/README.md` |
| ML Forecasting | `src/ml/forecasting/docs/README.md` |
| Ensemble Aggregation | `src/ensemble/aggregation/docs/README.md` |
| Analysis | `src/analysis/docs/README.md` |

---

## References

1. Hwang, C.L., Yoon, K. (1981). Multiple Attribute Decision Making
2. Opricovic, S., Tzeng, G.H. (2004). Compromise solution by MCDM methods: VIKOR
3. Brans, J.P., Vincke, P. (1985). PROMETHEE methods
4. Zavadskas, E.K., et al. (2008). COPRAS method
5. Keshavarz Ghorabaee, M., et al. (2015). EDAS method
