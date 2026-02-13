# Forecasting Methodology: Ensemble Machine Learning

## Overview

This framework implements a **unified ensemble learning system** that combines multiple machine learning models to forecast future criterion values for multi-criteria panel data.

**Key Features:**
- **7 Model Types**: Tree-based, linear, and neural network models
- **Automated Weighting**: Performance-based model combination
- **Uncertainty Quantification**: Prediction intervals and confidence estimates
- **Temporal Feature Engineering**: Rich lag/rolling/momentum features
- **Time-Series Cross-Validation**: Proper temporal validation

---

## System Architecture

```
Input: Panel Data (N entities × p components × T years)
  ↓
Stage 1: Temporal Feature Engineering
  ├── Lag features (t-1, t-2, ...)
  ├── Rolling statistics (mean, std, min, max)
  ├── Momentum & acceleration
  ├── Trend indicators
  └── Cross-entity percentiles
  ↓
Stage 2: Model Training (7 Models × CV Folds)
  │
  ├── Tree-Based Ensemble (3 models)
  │   ├── Gradient Boosting (Huber loss, early stopping)
  │   ├── Random Forest (OOB uncertainty)
  │   └── Extra Trees (extra randomization)
  │
  ├── Linear Models (3 models)
  │   ├── Bayesian Ridge (uncertainty quantification)
  │   ├── Huber Regression (outlier robust)
  │   └── Ridge Regression (L2 regularization)
  │
  └── Neural Networks (2 models) [Optional]
      ├── Multi-Layer Perceptron (SELU activation)
      └── Self-Attention Network (learned feature importance)
  ↓
Stage 3: Time-Series Cross-Validation
  ├── TimeSeriesSplit (preserve temporal order)
  ├── Compute R² scores per model
  └── Calculate performance metrics
  ↓
Stage 4: Performance-Based Weighting
  ├── Softmax over CV R² scores
  └── w_i = exp(5 × R²_i) / Σ exp(5 × R²_j)
  ↓
Stage 5: Ensemble Prediction
  ├── Weighted average: ŷ = Σ w_i × ŷ_i
  ├── Uncertainty: σ = √(Σ w_i × σ_i² + model_disagreement)
  └── Prediction intervals: ŷ ± 1.96σ
  ↓
Output: Predictions + Uncertainty + Model Diagnostics
```

---

## Part I: Model Types

### 1.1 Tree-Based Ensemble

#### Gradient Boosting Forecaster

**Algorithm:** Gradient Boosting Trees with Huber loss  
**Library:** `sklearn.ensemble.GradientBoostingRegressor`

**Key Parameters:**
```python
n_estimators = 200        # Number of boosting iterations
max_depth = 6             # Tree depth (prevents overfitting)
learning_rate = 0.1       # Shrinkage parameter
subsample = 0.8           # Stochastic gradient boosting
loss = 'huber'            # Robust to outliers
alpha = 0.9               # Huber loss parameter
```

**Advantages:**
- High predictive accuracy
- Robust to outliers via Huber loss
- Feature importance from tree splits
- Early stopping prevents overfitting

**Huber Loss Function:**
$$
L_\delta(y, f) = \begin{cases}
\frac{1}{2}(y - f)^2 & \text{if } |y - f| \leq \delta \\
\delta(|y - f| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$

Where $\delta = 0.9$ (threshold for outlier detection).

---

#### Random Forest Forecaster

**Algorithm:** Ensemble of decision trees with bootstrap aggregation  
**Library:** `sklearn.ensemble.RandomForestRegressor`

**Key Parameters:**
```python
n_estimators = 100        # Number of trees
max_depth = None          # Grow to full depth
min_samples_leaf = 2      # Minimum leaf size
max_features = 'sqrt'     # Features per split
bootstrap = True          # Bootstrap sampling
oob_score = True          # Out-of-bag validation
```

**Advantages:**
- Natural uncertainty from tree variance
- OOB validation without separate holdout set
- Parallel training (fast)
- Robust to overfitting

**Uncertainty Quantification:**
$$
\sigma^2_{RF}(x) = \frac{1}{T}\sum_{t=1}^T \left(\hat{y}_t(x) - \bar{\hat{y}}(x)\right)^2
$$

Where $T$ = number of trees, $\hat{y}_t$ = prediction from tree $t$.

---

#### Extra Trees Forecaster

**Algorithm:** Extremely Randomized Trees  
**Library:** `sklearn.ensemble.ExtraTreesRegressor`

**Difference from Random Forest:**
- Random splits instead of best splits
- Uses full dataset (no bootstrap)
- Faster training, lower variance

**Key Parameters:**
```python
n_estimators = 100
max_depth = None
min_samples_leaf = 2
max_features = 'sqrt'
```

**Advantages:**
- Very fast training
- Lower variance than Random Forest
- Good for noisy data

---

### 1.2 Linear Models

#### Bayesian Ridge Forecaster

**Algorithm:** Bayesian linear regression with Gaussian priors  
**Library:** `sklearn.linear_model.BayesianRidge`

**Model:**
$$
p(y|X, w, \alpha, \lambda) = \mathcal{N}(y | Xw, \alpha^{-1})
$$
$$
p(w|\lambda) = \mathcal{N}(w | 0, \lambda^{-1}I)
$$

Where:
- $w$ = regression coefficients
- $\alpha$ = noise precision (learned)
- $\lambda$ = coefficient precision (learned)

**Advantages:**
- Natural uncertainty quantification via posterior variance
- Automatic regularization (no hyperparameter tuning)
- Handles multicollinearity well

**Prediction with Uncertainty:**
$$
p(y_*|X_*, X, y) = \mathcal{N}(y_* | \mu_*, \sigma_*^2)
$$

Where:
$$
\mu_* = X_* \mathbb{E}[w|X, y]
$$
$$
\sigma_*^2 = \frac{1}{\alpha} + X_* \Sigma_w X_*^T
$$

---

#### Huber Regressor

**Algorithm:** Linear regression with Huber loss  
**Library:** `sklearn.linear_model.HuberRegressor`

**Objective:**
$$
\min_w \sum_i L_\delta(y_i - X_i w) + \alpha ||w||_2^2
$$

Where $L_\delta$ is the Huber loss (quadratic for small errors, linear for large).

**Advantages:**
- Robust to outliers
- Identifies outlier samples (via `outliers_` attribute)
- L2 regularization prevents overfitting

**Key Parameters:**
```python
epsilon = 1.35        # Huber loss threshold
alpha = 0.0001        # L2 regularization strength
max_iter = 100
```

---

#### Ridge Regressor

**Algorithm:** Linear regression with L2 regularization  
**Library:** `sklearn.linear_model.Ridge`

**Objective:**
$$
\min_w ||y - Xw||_2^2 + \alpha ||w||_2^2
$$

**Advantages:**
- Fast (closed-form solution)
- Handles multicollinearity
- Simple and interpretable

**Key Parameters:**
```python
alpha = 1.0           # Regularization strength
```

---

### 1.3 Neural Networks

**Note:** Neural networks are **disabled by default** due to insufficient panel data (14 years may be too limited for reliable deep learning).

#### Multi-Layer Perceptron (MLP)

**Architecture:**
- Input layer: N features
- Hidden layers: [256, 128, 64] neurons (configurable)
- Output layer: 1 neuron (regression)

**Activation:** SELU (Self-Normalizing)
$$
\text{SELU}(x) = \lambda \begin{cases}
x & \text{if } x > 0 \\
\alpha(e^x - 1) & \text{if } x \leq 0
\end{cases}
$$

Where $\lambda = 1.0507$, $\alpha = 1.6733$ (ensure self-normalization).

**Regularization:**
- Dropout: 10% (default)
- Early stopping: Patience 10 epochs
- Learning rate decay

**Optimizer:** Adam with learning rate = 0.001

**Advantages:**
- Non-linear feature interactions
- Self-normalizing activations (SELU) reduce vanishing gradients
- Flexible architecture

**Disadvantages:**
- Requires large datasets (>1000 samples)
- Prone to overfitting with limited data
- Slow training

---

#### Self-Attention Network

**Architecture:**
```
Input (N features)
  ↓
Linear Projection
  ↓
Self-Attention Layer (multi-head)
  │  Q = W_Q × X
  │  K = W_K × X
  │  V = W_V × X
  │  Attention(Q,K,V) = softmax(QK^T/√d_k) V
  ↓
Residual Connection + LayerNorm
  ↓
Feed-Forward Network
  ↓
Output Linear Layer
```

**Self-Attention Formula:**
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**Multi-Head Attention:**
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

Where:
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**Advantages:**
- Learns feature importance automatically
- Captures long-range dependencies
- Residual connections prevent gradient vanishing

**Key Parameters:**
```python
hidden_dim = 128          # Attention dimension
n_attention_heads = 4     # Number of heads
n_layers = 2              # Stacked attention layers
dropout_rate = 0.1
```

---

## Part II: Feature Engineering

### 2.1 Temporal Feature Types

**File:** `src/ml/forecasting/features.py`  
**Class:** `TemporalFeatureEngineer`

#### Lag Features
Captures historical values:

$$
\text{lag}_k(t) = x(t - k)
$$

**Example:** `lag_periods=[1, 2]` creates:
- `{component}_lag1`: Previous year value
- `{component}_lag2`: Two years ago value

**Purpose:** Autoregressive dependencies (current value depends on past).

---

#### Rolling Statistics

Captures recent trends:

$$
\text{roll\_mean}_w(t) = \frac{1}{w}\sum_{i=0}^{w-1} x(t - i)
$$

$$
\text{roll\_std}_w(t) = \sqrt{\frac{1}{w}\sum_{i=0}^{w-1} (x(t-i) - \text{roll\_mean}_w(t))^2}
$$

**Example:** `rolling_windows=[2, 3]` creates:
- `{component}_roll2_mean`, `{component}_roll2_std`
- `{component}_roll3_mean`, `{component}_roll3_std`, etc.

**Purpose:** Smooths noise, captures recent volatility.

---

#### Momentum Features

First and second derivatives:

$$
\text{momentum}(t) = x(t) - x(t-1)
$$

$$
\text{acceleration}(t) = \text{momentum}(t) - \text{momentum}(t-1)
$$

**Purpose:** Captures growth/decline patterns.

---

#### Trend Indicator

Linear trend slope over recent window:

$$
\text{trend}(t) = \frac{\text{cov}(x_{[t-w:t]}, [1, 2, \ldots, w])}{\text{var}([1, 2, \ldots, w])}
$$

**Purpose:** Long-term directional movement.

---

#### Cross-Entity Features

Relative position among all entities:

$$
\text{percentile}(t) = \frac{\text{rank}(x_i(t))}{\text{N\_entities}}
$$

$$
\text{zscore}(t) = \frac{x_i(t) - \mu_{\text{all}}(t)}{\sigma_{\text{all}}(t)}
$$

**Purpose:** Captures competitive position.

---

### 2.2 Feature Engineering Workflow

```python
class TemporalFeatureEngineer:
    def __init__(self, 
                 lag_periods=[1, 2],
                 rolling_windows=[2, 3],
                 include_momentum=True,
                 include_trend=True,
                 include_cross_entity=True):
        ...
    
    def fit_transform(self, panel_data, target_year):
        """
        Creates features for training and prediction.
        
        Returns:
        --------
        X_train : Training features (entity-year observations before target_year)
        y_train : Training targets (next year values)
        X_pred : Features for target_year prediction
        feature_names : List of feature column names
        """
        # Extract panel structure
        entities = panel_data.provinces
        years = panel_data.years
        components = panel_data.hierarchy.all_subcriteria
        
        # For each entity-year combination BEFORE target_year
        for entity in entities:
            for year in years[:-1]:  # Exclude last year (used for y)
                features = {}
                
                # Lag features
                for k in lag_periods:
                    if year - k in years:
                        features[f'{comp}_lag{k}'] = data[entity, year-k, comp]
                
                # Rolling features
                for w in rolling_windows:
                    window_data = [data[entity, year-i, comp] 
                                   for i in range(w) if year-i in years]
                    features[f'{comp}_roll{w}_mean'] = np.mean(window_data)
                    features[f'{comp}_roll{w}_std'] = np.std(window_data)
                
                # Momentum
                if year-1 in years:
                    momentum = data[entity, year, comp] - data[entity, year-1, comp]
                    features[f'{comp}_momentum'] = momentum
                
                # Target (next year)
                target = data[entity, year+1, comp]
                
                X_train.append(features)
                y_train.append(target)
        
        # For prediction at target_year
        for entity in entities:
            features_pred = compute_features(entity, target_year)
            X_pred.append(features_pred)
        
        return X_train, y_train, X_pred, feature_names
```

---

## Part III: Model Weighting & Ensemble

### 3.1 Time-Series Cross-Validation

**Method:** `TimeSeriesSplit` (preserves temporal order)

**Example with 3 folds:**
```
Fold 1:  Train [Y1, Y2, Y3] → Validate [Y4]
Fold 2:  Train [Y1, Y2, Y3, Y4] → Validate [Y5]
Fold 3:  Train [Y1, Y2, Y3, Y4, Y5] → Validate [Y6]
```

**Key Property:** No future data leakage (always train on past, validate on future).

**Metrics Computed:**
- R² (coefficient of determination)
- MAE (mean absolute error)
- RMSE (root mean squared error)

### 3.2 Performance-Based Weighting

**Objective:** Weight models by cross-validation performance.

**Softmax Weighting:**
$$
w_i = \frac{\exp(\beta \cdot R^2_i)}{\sum_{j=1}^M \exp(\beta \cdot R^2_j)}
$$

Where:
- $R^2_i$ = Cross-validation R² score for model $i$
- $\beta = 5$ = Temperature parameter (controls concentration)
- $M$ = Number of models

**Properties:**
- Models with higher R² get exponentially more weight
- Temperature $\beta$ controls sharpness:
  - $\beta \to 0$: Uniform weights (equal ensemble)
  - $\beta \to \infty$: All weight on best model
- Always sums to 1: $\sum_i w_i = 1$

**Example:**
```
Model           R²       exp(5×R²)    Weight
--------------------------------------------
GradientBoost   0.85     75.19        0.42
RandomForest    0.82     55.12        0.31
Bayesian        0.78     40.45        0.23
Huber           0.70     24.53        0.04
Total                                  1.00
```

### 3.3 Ensemble Prediction

**Weighted Average:**
$$
\hat{y}_{\text{ensemble}}(x) = \sum_{i=1}^M w_i \cdot \hat{y}_i(x)
$$

**Uncertainty Estimation:**

Combines two sources of uncertainty:

1. **Within-Model Uncertainty** (for models that provide it):
   - Bayesian: Posterior predictive variance
   - Random Forest: Tree variance

2. **Between-Model Disagreement**:
   $$
   \sigma^2_{\text{disagreement}} = \sum_{i=1}^M w_i \left(\hat{y}_i(x) - \hat{y}_{\text{ensemble}}(x)\right)^2
   $$

**Total Uncertainty:**
$$
\sigma^2_{\text{total}} = \sum_{i=1}^M w_i \sigma^2_i(x) + \sigma^2_{\text{disagreement}}
$$

**95% Prediction Interval:**
$$
\text{CI}_{95\%} = \hat{y}_{\text{ensemble}} \pm 1.96 \sigma_{\text{total}}
$$

---

## Part IV: Unified Forecaster

### 4.1 Forecasting Modes

**File:** `src/ml/forecasting/unified.py`  
**Enum:** `ForecastMode`

| Mode | Models Included | Speed | Accuracy | Use Case |
|------|----------------|-------|----------|----------|
| **FAST** | GradientBoost, Ridge | ⭐⭐⭐ | ⭐ | Quick iterations, prototyping |
| **BALANCED** | GB, RF, Bayesian, Ridge | ⭐⭐ | ⭐⭐ | **Production default** |
| **ACCURATE** | All tree + all linear | ⭐ | ⭐⭐⭐ | Final predictions |
| **NEURAL** | MLP, Attention, Ridge | ⭐ | ⭐⭐ | Experimental (needs large data) |
| **ENSEMBLE** | All 7 models | ⭐ | ⭐⭐⭐ | Maximum accuracy (slow) |

**Default:** `BALANCED` (excludes neural networks due to data insufficiency).

### 4.2 API Usage

```python
from src.ml.forecasting import UnifiedForecaster, ForecastMode

# Initialize
forecaster = UnifiedForecaster(
    mode=ForecastMode.BALANCED,
    include_neural=False,       # Disabled by default
    include_tree_ensemble=True,
    include_linear=True,
    cv_folds=3,                 # Time-series CV folds
    random_state=42,
    verbose=True
)

# Fit and predict
result = forecaster.fit_predict(
    panel_data=panel_data,
    target_year=2025
)

# Access results
print(result.get_summary())

# Predictions (DataFrame: entities × components)
predictions = result.predictions

# Uncertainty estimates
uncertainty = result.uncertainty

# Prediction intervals (95%)
lower_bound = result.prediction_intervals['lower']
upper_bound = result.prediction_intervals['upper']

# Model diagnostics
model_weights = result.model_contributions
cv_scores = result.cross_validation_scores
feature_importance = result.feature_importance

# Export all
results_dict = result.to_dict()
```

### 4.3 Result Structure

**Class:** `UnifiedForecastResult`

```python
@dataclass
class UnifiedForecastResult:
    predictions: pd.DataFrame           # Shape: (n_entities, n_components)
    uncertainty: pd.DataFrame           # Prediction uncertainty
    prediction_intervals: Dict[str, pd.DataFrame]  # 'lower', 'upper'
    model_contributions: Dict[str, float]  # Model weights
    model_performance: Dict[str, Dict]  # CV metrics per model
    feature_importance: pd.DataFrame    # Aggregated feature importance
    cross_validation_scores: Dict[str, List[float]]  # CV R² per fold
    feature_names: List[str]
    target_year: int
    
    def get_summary(self) -> str:
        """Human-readable summary of forecast results."""
        ...
    
    def to_dict(self) -> Dict:
        """Export to dictionary for JSON serialization."""
        ...
    
    def get_top_predictions(self, n: int = 10) -> pd.DataFrame:
        """Return top n entities by average predicted performance."""
        ...
    
    def get_high_uncertainty(self, n: int = 5) -> pd.DataFrame:
        """Return entities with highest prediction uncertainty."""
        ...
```

---

## Part V: Implementation Details

### 5.1 Training Pipeline

**Pseudocode:**

```python
class UnifiedForecaster:
    def fit_predict(self, panel_data, target_year):
        # Stage 1: Feature Engineering
        engineer = TemporalFeatureEngineer(...)
        X_train, y_train, X_pred, feature_names = engineer.fit_transform(
            panel_data, target_year
        )
        
        # Stage 2: Time-Series CV
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = {}
        
        for model_name in selected_models:
            model = self._get_model(model_name)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                scores.append(r2_score(y_val, y_pred))
            
            cv_scores[model_name] = np.mean(scores)
        
        # Stage 3: Performance-Based Weighting
        r2_values = np.array([cv_scores[m] for m in selected_models])
        weights = softmax(5 * r2_values)  # Temperature = 5
        model_contributions = dict(zip(selected_models, weights))
        
        # Stage 4: Final Training on All Data
        trained_models = {}
        for model_name in selected_models:
            model = self._get_model(model_name)
            model.fit(X_train, y_train)
            trained_models[model_name] = model
        
        # Stage 5: Ensemble Prediction
        predictions = []
        uncertainties = []
        
        for model_name, weight in model_contributions.items():
            model = trained_models[model_name]
            pred = model.predict(X_pred)
            predictions.append(weight * pred)
            
            # Uncertainty (if available)
            if hasattr(model, 'predict_uncertainty'):
                unc = model.predict_uncertainty(X_pred)
                uncertainties.append(weight * unc**2)
        
        # Weighted ensemble
        ensemble_pred = np.sum(predictions, axis=0)
        
        # Model disagreement
        disagreement = np.sum([
            w * (pred/w - ensemble_pred)**2 
            for w, pred in zip(weights, predictions)
        ], axis=0)
        
        # Total uncertainty
        ensemble_unc = np.sqrt(
            np.sum(uncertainties, axis=0) + disagreement
        )
        
        # Prediction intervals
        lower = ensemble_pred - 1.96 * ensemble_unc
        upper = ensemble_pred + 1.96 * ensemble_unc
        
        # Aggregate feature importance
        feature_importance = self._aggregate_feature_importance(
            trained_models, model_contributions
        )
        
        return UnifiedForecastResult(
            predictions=ensemble_pred,
            uncertainty=ensemble_unc,
            prediction_intervals={'lower': lower, 'upper': upper},
            model_contributions=model_contributions,
            model_performance=cv_scores,
            feature_importance=feature_importance,
            ...
        )
```

### 5.2 Model Selection

```python
def _get_models_for_mode(mode: ForecastMode) -> List[str]:
    if mode == ForecastMode.FAST:
        return ['GradientBoosting', 'Ridge']
    elif mode == ForecastMode.BALANCED:
        return ['GradientBoosting', 'RandomForest', 'Bayesian', 'Ridge']
    elif mode == ForecastMode.ACCURATE:
        return ['GradientBoosting', 'RandomForest', 'ExtraTrees',
                'Bayesian', 'Huber', 'Ridge']
    elif mode == ForecastMode.NEURAL:
        return ['MLP', 'Attention', 'Ridge']
    elif mode == ForecastMode.ENSEMBLE:
        return ['GradientBoosting', 'RandomForest', 'ExtraTrees',
                'Bayesian', 'Huber', 'Ridge', 'MLP', 'Attention']
```

---

## Part VI: Advantages & Limitations

### 6.1 Advantages

1. **Robustness**
   - Multiple model types capture different patterns
   - Performance-based weighting adapts to data
   - Outlier-robust models (Huber loss)

2. **Uncertainty Quantification**
   - Bayesian models provide natural uncertainty
   - Model disagreement quantifies epistemic uncertainty
   - Prediction intervals guide decision-making

3. **Feature Engineering**
   - Rich temporal features capture complex dynamics
   - Cross-entity features preserve competitive structure
   - Automatic feature generation

4. **Temporal Validity**
   - Time-series CV prevents data leakage
   - Respects temporal ordering
   - Realistic performance estimates

5. **Flexibility**
   - Multiple forecasting modes (fast/balanced/accurate)
   - Configurable feature engineering
   - Extensible architecture (easy to add new models)

### 6.2 Limitations

1. **Data Requirements**
   - Neural networks need >1000 samples (14 years × 64 provinces = 896 insufficient)
   - Feature engineering needs ≥4 years of history
   - Missing data reduces sample size

2. **Computational Cost**
   - ENSEMBLE mode trains 7 models × CV folds (slow)
   - Feature engineering increases dimensionality
   - Neural networks training is expensive

3. **Hyperparameter Sensitivity**
   - Model hyperparameters not automatically tuned
   - Temperature parameter (β=5) affects weight concentration
   - Feature engineering parameters (lag, windows) require domain knowledge

4. **Extrapolation Risk**
   - Models trained on historical patterns
   - May fail during regime changes (e.g., COVID-19)
   - Uncertainty estimates assume stationary process

### 6.3 Future Enhancements

1. **Automatic Hyperparameter Optimization**
   - Bayesian optimization for model hyperparameters
   - Grid search with cross-validation
   - Adaptive feature selection

2. **Advanced Uncertainty**
   - Conformal prediction for distribution-free intervals
   - MCMC for full posterior distributions
   - Quantile regression forests

3. **Improved Neural Architectures**
   - Temporal Convolutional Networks (TCN)
   - Transformers for time series
   - Autoregressive models (ARIMA+NN hybrids)

4. **Causal Inference**
   - Incorporate intervention effects
   - Counterfactual forecasting
   - Policy impact modeling

---

## References

1. **Friedman, J.H.** (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.

2. **Breiman, L.** (2001). Random forests. *Machine Learning*, 45(1), 5-32.

3. **Geurts, P., Ernst, D., & Wehenkel, L.** (2006). Extremely randomized trees. *Machine Learning*, 63(1), 3-42.

4. **MacKay, D.J.C.** (1992). Bayesian interpolation. *Neural Computation*, 4(3), 415-447.

5. **Huber, P.J.** (1964). Robust estimation of a location parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.

6. **Klambauer, G., et al.** (2017). Self-normalizing neural networks. *NeurIPS 2017*.

7. **Vaswani, A., et al.** (2017). Attention is all you need. *NeurIPS 2017*.

8. **Bergmeir, C., & Benítez, J.M.** (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.

---

**Document Version:** 1.0  
**Last Updated:** February 14, 2026  
**Status:** Production
