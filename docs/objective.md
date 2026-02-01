# ML-MCDM Project Objectives

## Dataset
Panel data of 20 component scores for the sustainable development index of 64 provinces in Vietnam from 2020-2024.

## Project Objectives

### Objective 1: Current Year Index Calculation (2024)
Calculate composite sustainability index scores and rankings for 64 provinces using 20 component scores (2024) with MCDM enhanced by Machine Learning.

### Objective 2: Future Year Forecasting (2025)
Forecast sustainability index scores for 64 provinces using historical data (2020-2024) with ML-enhanced MCDM.

---

## Appropriate Analytical Techniques

### For Objective 1: Current Year MCDM Ranking

| Technique | Purpose | Justification |
|-----------|---------|---------------|
| **Entropy Weights** | Objective weight calculation | Assigns weights based on information content (variance) in data |
| **CRITIC Weights** | Objective weight calculation | Considers both contrast intensity and inter-criteria correlation |
| **TOPSIS** | Primary ranking method | Distance-based method with clear interpretation |
| **VIKOR** | Compromise ranking | Balances group utility and individual regret |
| **PROMETHEE** | Outranking method | Pairwise preference-based with configurable thresholds |
| **COPRAS** | Proportional assessment | Handles both maximizing and minimizing criteria |
| **EDAS** | Distance from average | Intuitive average-based comparison |
| **Fuzzy TOPSIS** | Uncertainty handling | Uses triangular fuzzy numbers from temporal variance |
| **Borda/Copeland** | Rank aggregation | Combines multiple MCDM methods for robust final ranking |
| **Sensitivity Analysis** | Validation | Tests robustness of rankings to weight perturbations |
| **Random Forest** | Feature importance | Identifies which components most influence sustainability rankings |

### For Objective 2: Future Year Forecasting (2025)

| Technique | Purpose | Justification |
|-----------|---------|---------------|
| **Gradient Boosting** | Component forecasting | Strong performance on tabular time-series data |
| **Random Forest Forecaster** | Component forecasting | Handles non-linear patterns with built-in feature selection |
| **Extra Trees** | Ensemble diversity | Adds variance reduction through extreme randomization |
| **Bayesian Ridge** | Uncertainty quantification | Provides prediction intervals with probabilistic framework |
| **Ensemble Averaging** | Forecast combination | Reduces individual model bias and variance |
| **TOPSIS on Predictions** | Future ranking | Apply MCDM to forecasted component values |
| **VIKOR on Predictions** | Compromise future ranking | Alternative ranking perspective for forecasted data |

---


---

## Recommended Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    OBJECTIVE 1: RANKING                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  2024 Data ──► Entropy/CRITIC Weights                       │
│                       │                                      │
│                       ▼                                      │
│              Multiple MCDM Methods                           │
│         (TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS)            │
│                       │                                      │
│                       ▼                                      │
│              Rank Aggregation (Borda)                        │
│                       │                                      │
│                       ▼                                      │
│              Sensitivity Analysis (Validation)               │
│                       │                                      │
│                       ▼                                      │
│              Final 2024 Rankings                             │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                   OBJECTIVE 2: FORECASTING                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  2020-2024 Data ──► ML Ensemble Forecasting                 │
│                     (GB, RF, ET, Bayesian Ridge)            │
│                           │                                  │
│                           ▼                                  │
│                   Predicted 2025 Components                  │
│                           │                                  │
│                           ▼                                  │
│              TOPSIS/VIKOR on Predictions                    │
│                           │                                  │
│                           ▼                                  │
│              Predicted 2025 Rankings                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Metrics for Validation

### For MCDM (Objective 1)
- **Kendall's W**: Inter-method agreement (target > 0.8)
- **Sensitivity Robustness**: Ranking stability under weight perturbation (target > 0.9)
- **Spearman Correlation**: Pairwise method correlation

### For Forecasting (Objective 2)
- **R²**: Variance explained by forecasting models
- **MAE/RMSE**: Absolute prediction errors
- **Rank Correlation**: Predicted vs actual ranking alignment
- **Cross-Validation Scores**: Out-of-sample performance