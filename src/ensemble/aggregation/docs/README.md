# Ensemble Aggregation Methods

This module provides rank aggregation and ensemble techniques for combining results from multiple decision-making methods.

## Overview

| Method | Type | Use Case | Complexity |
|--------|------|----------|------------|
| Borda Count | Rank Aggregation | General consensus | O(m × n) |
| Copeland | Rank Aggregation | Condorcet consistency | O(m × n²) |
| Kemeny-Young | Rank Aggregation | Optimal consensus | O(n!) / O(m × n²) |
| Median Rank | Rank Aggregation | Outlier robustness | O(m × n log m) |
| Stacking | Meta-Learning | Prediction combination | O(n × k²) |

Where: m = methods, n = alternatives, k = base models

## Quick Start

```python
from src.ensemble.aggregation import (
    aggregate_rankings,
    BordaCount,
    CopelandMethod,
    StackingEnsemble
)
import numpy as np

# Example: Aggregate MCDM rankings
rankings = {
    'TOPSIS': np.array([1, 3, 2, 4, 5]),
    'VIKOR': np.array([2, 1, 3, 4, 5]),
    'PROMETHEE': np.array([1, 2, 3, 5, 4])
}

# Quick aggregation
result = aggregate_rankings(rankings, method='borda')
print(f"Final ranking: {result.final_ranking}")
print(f"Kendall's W: {result.kendall_w:.3f}")
```

## Rank Aggregation Methods

### 1. Borda Count

Positional voting where alternatives receive points based on rank position.

**Formula:**
$$B(i) = \sum_j w_j \times (n - r_j(i))$$

**Properties:**
- ✅ Monotonic (improving position helps ranking)
- ✅ Simple and interpretable
- ❌ Not Condorcet consistent

```python
from src.ensemble.aggregation import BordaCount

borda = BordaCount()
result = borda.aggregate(rankings)

# With custom weights
weights = {'TOPSIS': 0.4, 'VIKOR': 0.35, 'PROMETHEE': 0.25}
result = borda.aggregate(rankings, weights=weights)

# Different scoring functions
scores = borda.calculate_positional_scores(
    rankings, 
    score_function='exponential'  # 'linear', 'exponential', 'logarithmic'
)
```

### 2. Copeland Method

Pairwise comparison counting wins minus losses.

**Formula:**
$$C(i) = \sum_{j \neq i} [\text{sign}(P(i,j) - P(j,i))]$$

where $P(i,j)$ = weighted proportion preferring i over j

**Properties:**
- ✅ Condorcet consistent (selects Condorcet winner if exists)
- ✅ More robust than Borda to strategic voting
- ❌ Higher computational cost

```python
from src.ensemble.aggregation import CopelandMethod

copeland = CopelandMethod()
result = copeland.aggregate(rankings)

# Check for Condorcet winner
winner_idx = copeland.find_condorcet_winner(rankings)
if winner_idx is not None:
    print(f"Condorcet winner: Alternative {winner_idx}")

# Get pairwise preference matrix
pairwise = copeland.get_pairwise_matrix(rankings)
```

### 3. Kemeny-Young Method

Finds ranking minimizing total Kendall tau distance to all inputs.

**Formula:**
$$\pi^* = \arg\min_\pi \sum_j w_j \times \tau(\pi, \pi_j)$$

where $\tau$ is Kendall tau distance (number of pairwise disagreements)

**Properties:**
- ✅ Optimal consensus ranking
- ✅ Condorcet consistent
- ❌ NP-hard (exact solution exponential)

```python
from src.ensemble.aggregation import KemenyYoung

# Exact for small n, approximate for large n
kemeny = KemenyYoung(max_exact=8)
result = kemeny.aggregate(rankings)
```

### 4. Median Rank

Uses median rank across methods (robust to outliers).

**Formula:**
$$\text{MedianRank}(i) = \text{median}\{r_j(i) : j = 1, \ldots, m\}$$

```python
from src.ensemble.aggregation import MedianRank

median = MedianRank()
result = median.aggregate(rankings)
```

## Stacking Ensemble

Meta-learning approach for combining predictions.

### Basic Usage

```python
from src.ensemble.aggregation import StackingEnsemble

# Base model predictions
predictions = {
    'RandomForest': rf_preds,
    'GradientBoosting': gb_preds,
    'NeuralNet': nn_preds
}

# Train stacking meta-learner
stacker = StackingEnsemble(
    meta_learner='ridge',  # 'ridge', 'bayesian', 'elastic', 'linear'
    alpha=1.0
)
result = stacker.fit_predict(predictions, y_true)

print(f"Meta-model R²: {result.meta_model_r2:.4f}")
print(f"Weights: {dict(zip(predictions.keys(), result.meta_model_weights))}")

# Make new predictions
new_preds = stacker.predict(new_base_predictions)
```

### Temporal Stacking (Panel Data)

```python
from src.ensemble.aggregation import TemporalStackingEnsemble

# For panel data with time dimension
stacker = TemporalStackingEnsemble(
    meta_learner='ridge',
    temporal_decay=0.9  # Recent observations weighted more
)

result = stacker.fit_predict_temporal(
    predictions,
    y_true,
    time_indices  # Time period for each observation
)
```

### Meta-Learner Options

| Meta-Learner | Regularization | Best For |
|--------------|----------------|----------|
| `ridge` | L2 | Multicollinearity, stable weights |
| `bayesian` | Adaptive | Uncertainty estimation |
| `elastic` | L1 + L2 | Sparse selection |
| `linear` | None | No regularization needed |

## Agreement Metrics

### Kendall's W (Coefficient of Concordance)

Measures agreement among rankers:

$$W = \frac{12S}{m^2(n^3 - n)}$$

where S = sum of squared deviations from mean rank sum

- W = 0: No agreement (random)
- W = 1: Perfect agreement

```python
result = aggregate_rankings(rankings, method='borda')
print(f"Kendall's W: {result.kendall_w:.3f}")

# Interpret
if result.kendall_w > 0.7:
    print("Strong agreement")
elif result.kendall_w > 0.5:
    print("Moderate agreement")
else:
    print("Weak agreement")
```

### Spearman Correlation Matrix

Pairwise correlations between methods:

```python
import pandas as pd

# Agreement matrix from result
agreement_df = pd.DataFrame(
    result.agreement_matrix,
    index=list(rankings.keys()),
    columns=list(rankings.keys())
)
print(agreement_df)
```

## Choosing an Aggregation Method

| Criterion | Recommended Method |
|-----------|-------------------|
| Speed | Borda Count |
| Condorcet consistency | Copeland |
| Optimal consensus | Kemeny-Young (small n) |
| Outlier robustness | Median Rank |
| Weighted combination | Borda with weights |

## Integration with MCDM Pipeline

```python
from src.mcdm.traditional import topsis, vikor, promethee
from src.ensemble.aggregation import aggregate_rankings

# Run multiple MCDM methods
topsis_result = topsis(matrix, weights, criteria_types)
vikor_result = vikor(matrix, weights, criteria_types)
promethee_result = promethee(matrix, weights, criteria_types)

# Aggregate rankings
rankings = {
    'TOPSIS': topsis_result.ranking,
    'VIKOR': vikor_result.ranking,
    'PROMETHEE': promethee_result.ranking
}

# Create consensus ranking
final = aggregate_rankings(
    rankings, 
    method='borda',
    weights={'TOPSIS': 0.4, 'VIKOR': 0.3, 'PROMETHEE': 0.3}
)

print(final.summary())
```

## API Reference

### Classes
- `BordaCount` - Borda count aggregator
- `CopelandMethod` - Copeland pairwise aggregator
- `KemenyYoung` - Kemeny-Young optimal aggregator
- `MedianRank` - Median rank aggregator
- `StackingEnsemble` - Stacking meta-learner
- `TemporalStackingEnsemble` - Time-aware stacking

### Data Classes
- `AggregatedRanking` - Rank aggregation result
- `StackingResult` - Stacking ensemble result

### Functions
- `aggregate_rankings()` - Quick aggregation function
- `borda_count()` - Borda convenience function
- `copeland_method()` - Copeland convenience function
- `kemeny_young()` - Kemeny-Young convenience function
- `median_rank()` - Median rank convenience function
- `stacking_ensemble()` - Stacking convenience function

## References

1. de Borda, J.C. (1781). "Mémoire sur les élections au scrutin"
2. Copeland, A.H. (1951). "A 'reasonable' social welfare function"
3. Kemeny, J.G. (1959). "Mathematics without numbers"
4. Wolpert, D.H. (1992). "Stacked Generalization"
5. Dwork et al. (2001). "Rank aggregation revisited"
