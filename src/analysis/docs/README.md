# Analysis Module

This module provides validation and sensitivity analysis tools for MCDM rankings and ML models.

## Overview

| Component | Purpose | Key Methods |
|-----------|---------|-------------|
| Sensitivity Analysis | Test ranking robustness | Monte Carlo, Weight Perturbation |
| Cross-Validation | Model validation | K-Fold, Time Series Split |
| Bootstrap Validation | Confidence intervals | Bootstrap sampling |
| Ranking Validation | MCDM-specific validation | Perturbation stability |

## Quick Start

```python
from src.analysis.sensitivity import SensitivityAnalysis, run_sensitivity_analysis
from src.analysis.validation import CrossValidator, BootstrapValidator

# Sensitivity analysis
result = run_sensitivity_analysis(
    matrix=decision_matrix,
    weights=criterion_weights,
    ranking_func=topsis_ranking,
    n_simulations=1000
)
print(result.summary())

# Cross-validation
validator = CrossValidator(n_folds=5)
cv_result = validator.validate(X, y, model_func, metrics)
```

## Sensitivity Analysis

### Monte Carlo Sensitivity

Tests ranking stability under random weight perturbations.

```python
from src.analysis.sensitivity import SensitivityAnalysis

analyzer = SensitivityAnalysis(
    n_simulations=1000,      # Number of Monte Carlo runs
    perturbation_range=0.2,  # ±20% weight variation
    seed=42
)

result = analyzer.analyze(
    decision_matrix=matrix,
    weights=weights,
    ranking_function=topsis_ranking,
    criteria_names=['Cost', 'Quality', 'Time'],
    alternative_names=['A', 'B', 'C', 'D', 'E']
)

# Results include:
# - weight_sensitivity: Which criteria affect rankings most
# - rank_stability: Which alternatives have stable ranks
# - critical_weights: Weight ranges maintaining top ranking
# - top_n_stability: Stability of top 3, 5, 10 alternatives
# - overall_robustness: Overall robustness score (0-1)
```

### Weight Sensitivity Index

Measures how sensitive rankings are to each criterion weight:

$$S_j = \frac{1}{N} \sum_i \frac{|\Delta \text{rank}|}{n \times |\Delta w_j|}$$

```python
# High sensitivity (closer to 1) = rankings change significantly
# Low sensitivity (closer to 0) = rankings are stable

for criterion, sensitivity in sorted(
    result.weight_sensitivity.items(), 
    key=lambda x: x[1], 
    reverse=True
):
    print(f"{criterion}: {sensitivity:.3f}")
```

### Systematic Weight Perturbation

```python
from src.analysis.sensitivity import WeightPerturbation

# One-at-a-time: Vary single weight
oat_results = WeightPerturbation.one_at_a_time(
    weights=weights,
    matrix=matrix,
    ranking_func=topsis_ranking,
    steps=11  # From 0.5× to 2× weight
)

# Pairwise exchange: Transfer weight between criteria
exchange_results = WeightPerturbation.pairwise_exchange(
    weights=weights,
    matrix=matrix,
    ranking_func=topsis_ranking,
    exchange_amount=0.1  # Transfer 10% of weight
)
```

### Critical Weight Ranges

Find weight ranges that maintain current top ranking:

```python
# result.critical_weights gives (lower, upper) bounds
for criterion, (low, high) in result.critical_weights.items():
    print(f"{criterion}: [{low:.3f}, {high:.3f}]")
    
# Interpretation: Current top alternative remains #1 as long as
# criterion weight stays within these bounds
```

## Validation Methods

### Cross-Validation

```python
from src.analysis.validation import CrossValidator

# Standard K-Fold
validator = CrossValidator(
    n_folds=5,
    time_series_split=False,
    shuffle=True,
    seed=42
)

# Define model function
def model_func(X_train, y_train, X_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model.predict(X_test)

# Define metrics
metrics = {
    'R2': lambda y, p: 1 - np.sum((y-p)**2) / np.sum((y-y.mean())**2),
    'MAE': lambda y, p: np.mean(np.abs(y - p))
}

result = validator.validate(X, y, model_func, metrics)
print(f"R² = {result.cv_scores['R2']:.4f} ± {result.cv_std['R2']:.4f}")
```

### Time Series Cross-Validation

For panel data with temporal structure:

```python
validator = CrossValidator(
    n_folds=5,
    time_series_split=True  # Expanding window CV
)

result = validator.validate(
    X, y, model_func, metrics,
    time_indices=year_column  # Time period for each observation
)
```

### Bootstrap Validation

```python
from src.analysis.validation import BootstrapValidator

validator = BootstrapValidator(
    n_bootstrap=1000,
    confidence_level=0.95,
    seed=42
)

result = validator.validate(X, y, model_func, metrics)

# 95% confidence intervals
for metric, (low, high) in result.bootstrap_ci.items():
    print(f"{metric}: [{low:.4f}, {high:.4f}]")
```

### MCDM Ranking Validation

Specialized validation for ranking stability:

```python
from src.analysis.validation import RankingValidator

validator = RankingValidator(
    n_bootstrap=500,
    perturbation_std=0.05,  # 5% data perturbation
    seed=42
)

result = validator.validate_ranking(
    decision_matrix=matrix,
    weights=weights,
    ranking_func=topsis_ranking
)

print(f"Rank correlation: {result['mean_correlation']:.3f}")
print(f"Correlation std: {result['std_correlation']:.3f}")

# Confidence intervals for each alternative's rank
for alt_idx, (low, high) in result['rank_ci'].items():
    print(f"Alternative {alt_idx}: rank in [{low:.1f}, {high:.1f}]")
```

## Metrics Reference

### Regression Metrics

```python
from src.analysis.validation import r2_score, mse_score, mae_score

# Built-in convenience functions
r2 = r2_score(y_true, y_pred)
mse = mse_score(y_true, y_pred)
mae = mae_score(y_true, y_pred)
```

### Agreement Metrics

For comparing rankings:

```python
# Kendall's W (coefficient of concordance)
# 0 = no agreement, 1 = perfect agreement

# Spearman correlation
# -1 = perfect inverse, 0 = no correlation, 1 = perfect correlation
```

## Integration Example

Complete analysis workflow:

```python
import numpy as np
from src.mcdm.traditional import topsis
from src.analysis.sensitivity import run_sensitivity_analysis
from src.analysis.validation import RankingValidator

# 1. Get MCDM ranking
def ranking_func(matrix, weights):
    result = topsis(matrix, weights, criteria_types=['+', '+', '-'])
    return result.ranking

# 2. Sensitivity analysis
sensitivity = run_sensitivity_analysis(
    matrix=matrix,
    weights=weights,
    ranking_func=ranking_func,
    n_simulations=1000
)

print("=== Sensitivity Analysis ===")
print(f"Overall Robustness: {sensitivity.overall_robustness:.3f}")
print(f"\nMost sensitive criteria:")
for crit, sens in sorted(sensitivity.weight_sensitivity.items(), 
                         key=lambda x: x[1], reverse=True)[:3]:
    print(f"  {crit}: {sens:.3f}")

# 3. Ranking validation
validator = RankingValidator(n_bootstrap=500)
validation = validator.validate_ranking(matrix, weights, ranking_func)

print(f"\n=== Ranking Validation ===")
print(f"Mean correlation: {validation['mean_correlation']:.3f}")

# 4. Identify robust top alternatives
base_ranking = validation['base_ranking']
top_3_indices = np.argsort(base_ranking)[:3]

print(f"\nTop 3 alternatives stability:")
for idx in top_3_indices:
    ci_low, ci_high = validation['rank_ci'][idx]
    stability = sensitivity.rank_stability.get(f'A{idx+1}', 0)
    print(f"  Alternative {idx}: rank=[{ci_low:.1f}, {ci_high:.1f}], stability={stability:.3f}")
```

## API Reference

### Classes
- `SensitivityAnalysis` - Monte Carlo sensitivity analysis
- `WeightPerturbation` - Systematic weight variation
- `CrossValidator` - K-fold and time-series cross-validation
- `BootstrapValidator` - Bootstrap confidence intervals
- `RankingValidator` - MCDM ranking validation

### Data Classes
- `SensitivityResult` - Sensitivity analysis results
- `ValidationResult` - Cross-validation/bootstrap results

### Functions
- `run_sensitivity_analysis()` - Quick sensitivity analysis
- `bootstrap_validation()` - Quick bootstrap validation
- `r2_score()`, `mse_score()`, `mae_score()` - Metric functions

## References

1. Triantaphyllou, E. (2000). "Multi-Criteria Decision Making Methods"
2. Wolters & Mareschal (1995). "Novel types of sensitivity analysis for MCDM"
3. Efron, B. (1979). "Bootstrap methods: another look at the jackknife"
4. Bergmeir & Benítez (2012). "On the use of cross-validation for time series predictor evaluation"
