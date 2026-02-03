# ML Forecasting Methods

This module provides machine learning forecasting methods for multi-criteria decision making with temporal/panel data.

## Overview

The forecasting module combines multiple ML approaches into a unified ensemble system that:
- Engineers temporal features from panel data
- Applies multiple model types (tree, linear, neural)
- Automatically weights models based on cross-validation
- Quantifies prediction uncertainty

---

## Module Structure

```
src/ml/forecasting/
├── __init__.py       # Module exports
├── base.py           # Base classes and result containers
├── features.py       # Temporal feature engineering
├── tree_ensemble.py  # Gradient Boosting, Random Forest, Extra Trees
├── linear.py         # Bayesian Ridge, Huber, Ridge regression
├── neural.py         # MLP, Attention networks
├── unified.py        # Unified ensemble orchestrator
└── docs/
    └── README.md     # This documentation
```

---

## Quick Start

```python
from src.ml.forecasting import UnifiedForecaster, ForecastMode

# Create forecaster
forecaster = UnifiedForecaster(mode=ForecastMode.BALANCED)

# Fit and predict
result = forecaster.fit_predict(panel_data, target_year=2025)

# View results
print(result.get_summary())
predictions = result.predictions
uncertainty = result.uncertainty
```

---

## Forecasting Modes

| Mode | Description | Speed | Accuracy |
|------|-------------|-------|----------|
| `FAST` | Minimal models, quick results | ⭐⭐⭐ | ⭐ |
| `BALANCED` | Good trade-off | ⭐⭐ | ⭐⭐ |
| `ACCURATE` | Maximum accuracy | ⭐ | ⭐⭐⭐ |
| `NEURAL` | Neural network focused | ⭐ | ⭐⭐ |
| `ENSEMBLE` | All available models | ⭐ | ⭐⭐⭐ |

---

## Model Types

### 1. Tree-Based Ensemble

**File:** `tree_ensemble.py`

#### GradientBoostingForecaster
- Uses Huber loss for robustness to outliers
- Early stopping with validation
- Feature importance from tree splits

```python
from src.ml.forecasting import GradientBoostingForecaster

forecaster = GradientBoostingForecaster(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1
)
forecaster.fit(X_train, y_train)
predictions = forecaster.predict(X_test)
```

#### RandomForestForecaster
- Natural uncertainty estimation from tree variance
- Out-of-bag (OOB) score for validation
- Parallelized training

```python
from src.ml.forecasting import RandomForestForecaster

forecaster = RandomForestForecaster(n_estimators=100)
forecaster.fit(X_train, y_train)
predictions = forecaster.predict(X_test)
uncertainty = forecaster.predict_uncertainty(X_test)
```

#### ExtraTreesForecaster
- Extra randomization for lower variance
- Fast training due to random splits

---

### 2. Linear Methods

**File:** `linear.py`

#### BayesianForecaster
- Bayesian Ridge Regression
- Natural uncertainty quantification
- Automatic regularization tuning

```python
from src.ml.forecasting import BayesianForecaster

forecaster = BayesianForecaster()
forecaster.fit(X_train, y_train)
mean, std = forecaster.predict_with_uncertainty(X_test)
```

#### HuberForecaster
- Robust to outliers (Huber loss)
- Identifies outlier samples

#### RidgeForecaster
- Fast linear regression with L2 regularization

---

### 3. Neural Networks

**File:** `neural.py`

#### NeuralForecaster
- Multi-layer Perceptron
- SELU activation for self-normalization
- Dropout regularization
- Early stopping

```python
from src.ml.forecasting import NeuralForecaster

forecaster = NeuralForecaster(
    hidden_dims=[256, 128, 64],
    activation='selu',
    dropout_rate=0.1,
    n_epochs=100
)
forecaster.fit(X_train, y_train)
predictions = forecaster.predict(X_test)
```

#### AttentionForecaster
- Self-attention mechanism
- Learns feature importance
- Residual connections

```python
from src.ml.forecasting import AttentionForecaster

forecaster = AttentionForecaster(
    hidden_dim=128,
    n_attention_heads=4,
    n_layers=2
)
forecaster.fit(X_train, y_train)
predictions = forecaster.predict(X_test)
```

---

## Feature Engineering

**File:** `features.py`

The `TemporalFeatureEngineer` creates rich features from panel data:

### Feature Types

| Feature | Description |
|---------|-------------|
| `{comp}_current` | Current period value |
| `{comp}_lag{n}` | Value n periods ago |
| `{comp}_roll{w}_mean` | Rolling mean over w periods |
| `{comp}_roll{w}_std` | Rolling std over w periods |
| `{comp}_roll{w}_min/max` | Rolling min/max |
| `{comp}_momentum` | Change from previous period |
| `{comp}_acceleration` | Change in momentum |
| `{comp}_trend` | Linear trend slope |
| `{comp}_percentile` | Rank percentile vs other entities |
| `{comp}_zscore` | Z-score vs other entities |

### Usage

```python
from src.ml.forecasting import TemporalFeatureEngineer

engineer = TemporalFeatureEngineer(
    lag_periods=[1, 2],
    rolling_windows=[2, 3],
    include_momentum=True,
    include_cross_entity=True
)

X_train, y_train, X_pred, _ = engineer.fit_transform(panel_data, target_year=2025)
```

---

## Unified Forecaster

**File:** `unified.py`

The main orchestrator that combines all methods:

### Features
- Automatic model selection based on mode
- Performance-based model weighting
- Time-series cross-validation
- Ensemble prediction with uncertainty

### Result Attributes

| Attribute | Description |
|-----------|-------------|
| `predictions` | DataFrame of predictions (entities × components) |
| `uncertainty` | DataFrame of uncertainty estimates |
| `prediction_intervals` | Dict with 'lower' and 'upper' bounds |
| `model_contributions` | Weight of each model in ensemble |
| `model_performance` | CV metrics per model |
| `feature_importance` | Aggregated feature importance |
| `cross_validation_scores` | CV R² scores per model |

### Example

```python
from src.ml.forecasting import UnifiedForecaster, ForecastMode

# Initialize
# Note: Neural networks are disabled by default due to insufficient panel data
forecaster = UnifiedForecaster(
    mode=ForecastMode.BALANCED,
    include_neural=False,  # Disabled by default - insufficient data for reliable neural training
    include_tree_ensemble=True,
    include_linear=True,
    cv_folds=3,
    verbose=True
)

# Fit and predict
result = forecaster.fit_predict(panel_data, target_year=2025)

# Access results
print(result.get_summary())

# Get top predictions
best_entities = result.predictions.mean(axis=1).nlargest(10)

# Get uncertainty
high_uncertainty = result.uncertainty.mean(axis=1).nlargest(5)

# Get model weights
print(result.model_contributions)

# Export to dict
results_dict = result.to_dict()
```

---

## Model Weighting

Models are weighted using softmax over cross-validation R² scores:

$$w_i = \frac{\exp(5 \cdot R^2_i)}{\sum_j \exp(5 \cdot R^2_j)}$$

The temperature factor (5) controls how much weight concentrates on best models.

---

## Uncertainty Quantification

Uncertainty is estimated from:
1. **Model disagreement**: Standard deviation across model predictions
2. **Bayesian models**: Posterior predictive variance
3. **Tree variance**: Variance across trees in Random Forest

Prediction intervals are calculated as:
$$\text{CI}_{95\%} = \hat{y} \pm 1.96 \cdot \sigma$$

---

## Cross-Validation

Uses `TimeSeriesSplit` for proper temporal validation:
- Respects temporal ordering
- No future data leakage
- Configurable number of folds

---

## Best Practices

1. **Data Preparation**
   - Ensure panel data has sufficient history (≥4 years)
   - Handle missing values before forecasting
   - Normalize if components have different scales

2. **Mode Selection**
   - Use `FAST` for quick iterations
   - Use `BALANCED` for production
   - Use `ACCURATE` for final predictions

3. **Interpretation**
   - Check model weights to understand ensemble
   - Review feature importance for insights
   - Monitor uncertainty for unreliable predictions

4. **Validation**
   - Always check cross-validation scores
   - Use holdout year for final validation
   - Compare against baseline (e.g., last known value)
