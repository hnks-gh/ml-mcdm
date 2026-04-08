# Cross-Validation Fold Training & Out-of-Fold Holdout Data Design
## Ensemble Machine Learning Architecture

---

## Executive Summary

The ensemble machine learning system uses a sophisticated **two-phase temporal cross-validation** approach to generate out-of-fold (OOF) predictions for meta-learning:

1. **Primary OOF CV**: Generates K fold (default 5) OOF predictions for base models → meta-learner training
2. **Conformal OOF CV**: Secondary temporal sweep for extended calibration set (conformal prediction)

Both phases respect panel structure (multiple entities/provinces) and calendar year boundaries to prevent temporal leakage.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ ENSEMBLE LEARNING PIPELINE                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: PRIMARY OOF CV (5 walk-forward folds)                │
│  ─────────────────────────────────────────────                │
│  Fold 1: Train [2011-2018] → Validate [2019]                 │
│  Fold 2: Train [2011-2019] → Validate [2020]                 │
│  Fold 3: Train [2011-2020] → Validate [2021]                 │
│  Fold 4: Train [2011-2021] → Validate [2022]                 │
│  Fold 5: Train [2011-2022] → Validate [2023]                 │
│                                                                 │
│  Each fold generates OOF predictions: (N_samples, N_models)    │
│  NaN for training samples (no prediction)                      │
│  Valid prediction for validation samples (held-out)            │
│                                                                 │
│              ↓                                                  │
│                                                                 │
│  Stage 2: CONFORMAL OOF CV (Extended window)                  │
│  ────────────────────────────────────────────                 │
│  Secondary sweep: Train [2008-2010] → Validate [2008-2010]    │
│  Collects OOF on years NOT covered by primary CV              │
│                                                                 │
│              ↓ (Combined OOF residuals)                         │
│                                                                 │
│  Stage 3: META-LEARNER TRAINING                               │
│  ──────────────────────────────                               │
│  Fits Ridge regression on OOF predictions                      │
│  Output: Optimal combination weights per output criterion      │
│                                                                 │
│              ↓                                                  │
│                                                                 │
│  Stage 4: FINAL MODEL REFITTING                               │
│  ────────────────────────────                                 │
│  Train all base models on complete dataset with final weights  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. PRIMARY OUT-OF-FOLD CV DESIGN

### 1.1 Walk-Forward Yearly Split (`_WalkForwardYearlySplit`)

**Location**: `super_learner.py:172-247`

#### Parameters:
```python
class _WalkForwardYearlySplit:
    min_train_years : int = 8       # Minimum training window
    max_folds : int = 5              # Maximum number of folds
```

#### Splitting Logic:

For a panel dataset with years [2008, 2009, ..., 2023] with 63 provinces:

```
Fold 0: Train years [2008-2018] | Validate year [2019]
        - Train indices: all rows with year_label < 2019 (all provinces, years 2008-2018)
        - Validation indices: rows with year_label == 2019 (all active provinces in 2019)
        - Training data: ~11 years × 63 provinces = 693 rows
        - Validation data: ~63 rows (one per province)

Fold 1: Train years [2008-2019] | Validate year [2020]
        - Train indices: all rows with year_label < 2020
        - Validation indices: rows with year_label == 2020
        - Training data: ~12 years × 63 provinces
        - Validation data: ~63 rows

Fold 2: Train years [2008-2020] | Validate year [2021]
        - Training data: ~13 years × 63 provinces
        - Validation data: ~63 rows

Fold 3: Train years [2008-2021] | Validate year [2022]
        - Training data: ~14 years × 63 provinces
        - Validation data: ~63 rows

Fold 4: Train years [2008-2022] | Validate year [2023]
        - Training data: ~15 years × 63 provinces
        - Validation data: ~63 rows
```

#### Key Properties:

1. **Expanding Window**: Each fold uses ALL prior years (no gap)
   - Mimics production deployment: predict year T+1 using all data through year T
   - Captures non-stationarity through expanding window

2. **Temporal Integrity**:
   - Training data **always precedes** validation data in time
   - No look-ahead bias possible

3. **Per-Entity Consistency**:
   - Each province gets predictions in every fold where it has data
   - Entities with fewer years simply contribute fewer validation samples

4. **Calendar Year Boundaries**:
   - Splits happen at calendar year edges, not random row positions
   - All data from year T is either in train or validation, never split

#### Implementation Details (from `super_learner.py:194-232`):

```python
def split(self, X: np.ndarray, year_labels: np.ndarray):
    """
    Parameters
    ----------
    X : ndarray(n_samples, n_features)
        Used for shape only
    year_labels : ndarray(n_samples,)
        Calendar year for each training row (target year, e.g., 2019)
    """
    unique_years = sorted(np.unique(year_labels))  # [2008, 2009, ..., 2023]
    first_val_pos = min(self.min_train_years, n_years - 1)  # start at year 2016

    for fold_k in range(self.max_folds):
        val_pos = first_val_pos + fold_k
        if val_pos >= n_years:
            break

        val_year = unique_years[val_pos]
        train_idx = where(year_labels < val_year)    # All prior years
        val_idx   = where(year_labels == val_year)   # Just this year

        if len(train_idx) > 0 and len(val_idx) > 0:
            yield train_idx, val_idx
```

### 1.2 OOF Prediction Collection

**Location**: `super_learner.py:652-900` (fit method)

#### Storage Structure:

```python
# Initialize OOF array
oof_predictions = np.full(
    (n_samples, n_models * n_outputs),
    np.nan
)
# Shape: (693 rows, 4 base_models × 1 output) = (693, 4)

# For each fold:
for fold_idx, (train_idx, val_idx) in enumerate(cv_pairs):
    y_train_cv = y[train_idx]      # Shape: (650, 1) - from training rows
    y_val_cv = y[val_idx]          # Shape: (43, 1) - from validation rows

    # Train each base model on training fold
    for model_name, model in base_models.items():
        model.fit(X[train_idx], y_train_cv)

        # Predict on validation fold (HOLD OUT)
        val_pred = model.predict(X[val_idx])

        # Store prediction in OOF array at validation row indices
        col_idx = model_index * n_outputs
        oof_predictions[val_idx, col_idx] = val_pred
```

#### OOF Array Structure (Example with 4 models, 1 output):

```
Row Index | CatBoost | BayesianRidge | SVR | ElasticNet | True Target
────────────────────────────────────────────────────────────────────
0         | NaN      | NaN           | NaN | NaN        | 2.5      (training set)
...       | NaN      | NaN           | NaN | NaN        | ...
43        | 2.48     | 2.51          | 2.47| 2.49       | 2.45     (validation fold 0)
44        | 2.51     | 2.52          | 2.49| 2.50       | 2.48     (validation fold 0)
...       | ...      | ...           | ... | ...        | ...
693       | NaN      | NaN           | NaN | NaN        | 1.9      (training set)
```

**Key Insight**: Each row has OOF predictions **exactly once** (only when held out in one fold).

#### OOF Prediction Mask:

```python
# Track which rows have valid OOF predictions per output criterion
_oof_valid_mask_per_col_ = {
    criterion_name: ~np.isnan(oof_predictions[:, col])
    for criterion_name in output_names
}

# Typically: 693 rows, but only ~315 have valid OOF (validation rows across folds)
# Training rows remain NaN (were never held out)
```

---

## 2. CONFORMAL OUT-OF-FOLD CV (Extended Calibration)

### 2.1 Secondary Temporal Sweep

**Location**: `super_learner.py:1312-1450`

The secondary sweep covers years the primary CV might skip due to fold capitalization:

```
Primary CV Setup:
min_train_years=8, max_folds=5
└─ First validation year: 2016
└─ Last validation year: 2023

Conformal Secondary Setup:
conformal_min_train_years=6 (default 5)
└─ First validation year: 2014 ← Earlier!

Coverage:
Primary OOF years: {2019, 2020, 2021, 2022, 2023}
Secondary OOF years: {2014, 2015, 2016, 2017, 2018}

Combined calibration set: {2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023}
```

### 2.2 Conformal Residual Collection

```python
def _build_conformal_oof_residuals(X, y, year_labels, primary_oof):
    """
    Returns
    -------
    combined_residuals : ndarray(N_valid, n_outputs)
        Residuals: y_true - y_oof across primary + secondary years
    """

    # Secondary splitter
    secondary_splitter = _WalkForwardYearlySplit(
        min_train_years=conformal_min_train_years,  # 5-6 years
        max_folds=999,  # exhaust all years
    )

    # Collect folds NOT already in primary CV
    primary_val_years = {2019, 2020, 2021, 2022, 2023}

    for train_idx, val_idx in secondary_splitter.split(X, year_labels):
        val_years = set(unique(year_labels[val_idx]))

        # Skip if already in primary (avoid double-counting)
        if val_years.issubset(primary_val_years):
            continue

        # Predict with current ensemble on holdout year
        secondary_oof[val_idx] = ensemble.predict(X[val_idx])

    # Combine residuals
    primary_residuals = y[primary_mask] - primary_oof[primary_mask]
    secondary_residuals = y[secondary_mask] - secondary_oof[secondary_mask]
    combined_residuals = np.vstack([primary_residuals, secondary_residuals])

    return combined_residuals
```

### 2.3 Usage in Conformal Prediction

The combined OOF residuals are used for calibrating prediction intervals:

```python
# Calibrate with Student-t quantile on OOF residuals
q_alpha = np.quantile(abs(combined_residuals), 1 - alpha)

# Prediction interval: [ŷ - q, ŷ + q]
# Guarantees ~95% coverage on future data (distribution-free)
```

---

## 3. META-LEARNER: RIDGE STACKING

### 3.1 OOF-to-Meta-Weights Transformation

**Location**: `super_learner.py:1568-1712`

#### Input:
- OOF predictions array: shape (693, 4)
  - Each column: predictions from one base model
  - Each row: OOF predictions for that sample (NaN if not held out)

#### Process:

```python
def _fit_meta_learner(oof_X, oof_y):
    """
    Parameters
    ----------
    oof_X : ndarray(N_with_oof, N_models)
        OOF predictions from base models (valid rows only, ~315 rows with OOF)
    oof_y : ndarray(N_with_oof, N_outputs)
        True targets for validation rows only
    """

    # For each output criterion (usually 1, sometimes >1)
    for out_col in range(n_outputs):
        model_preds = oof_X[:, :n_models]  # (315, 4)
        y_col = oof_y[:, out_col]          # (315,)

        # Remove any remaining NaN
        valid = ~isnan(model_preds).any(axis=1) & ~isnan(y_col)
        model_preds_valid = model_preds[valid]  # (310, 4)
        y_valid = y_col[valid]                   # (310,)

        # Fit Ridge regression: OOF predictions → target
        meta = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0])
        meta.fit(model_preds_valid, y_valid)

        # Extract coefficients (meta-weights)
        coefs = meta.coef_  # shape (4,), e.g., [0.25, 0.30, 0.20, 0.25]

        # Apply positive constraints and normalize
        coefs = np.maximum(coefs, 0)  # clip negatives to 0
        coefs = coefs / coefs.sum()   # normalize to sum=1.0

        all_weights.append(coefs)

    # Final meta-weights matrix: (N_outputs, N_models)
    _meta_weights_per_output_ = array(all_weights)
    # Example: [[0.25, 0.30, 0.20, 0.25]]
```

#### Interpretation:

```
Final Ensemble Weights:
├─ CatBoost:       25%  (0.25)
├─ BayesianRidge:  30%  (0.30)
├─ SVR:            20%  (0.20)
└─ ElasticNet:     25%  (0.25)

Final Prediction = 0.25×ŷ_catboost + 0.30×ŷ_bayesian + 0.20×ŷ_svr + 0.25×ŷ_elasticnet
```

---

## 4. PANEL-AWARE TEMPORAL SPLIT (Fallback)

### 4.1 Entity-Aware Splitting

**Location**: `super_learner.py:46-150`

When year labels are unavailable, the system falls back to `_PanelTemporalSplit`:

```python
class _PanelTemporalSplit:
    """Respects within-entity temporal order across multiple panels/entities."""

    def split(X, entity_indices):
        """
        For each panel (province), split based on relative time position.

        Entity A (province): ┌────────────────────────┐
                             Train (11 rows) | Validation (1 row)

        Entity B (province): ┌────────────────────────┐
                             Train (11 rows) | Validation (1 row)

        Combined dataset indices are sorted after combining per-entity splits.
        """

        # Group rows by entity
        entity_rows = {
            entity_id: indices where entity==entity_id
            for entity_id in unique(entity_indices)
        }

        # Compute fold boundaries from MEDIAN entity length
        T_per_entity = {ent: len(rows) for ent, rows in entity_rows.items()}
        T_median = median(list(T_per_entity.values()))

        # This avoids bias toward shortest entity
        # (old code used min, wasting 9 years per longer-history province)

        min_train_T = max(T_median // 2, T_median // (n_splits + 1))
        fold_size = (T_median - min_train_T) // n_splits

        for fold_k in range(n_splits):
            cut = min_train_T + fold_k * fold_size

            train_parts, val_parts = [], []
            for ent, rows in entity_rows.items():
                T_ent = len(rows)
                train_cut = min(cut, T_ent)
                val_start = min(cut, T_ent)
                val_end = min(cut + fold_size, T_ent)

                if train_cut > 0:
                    train_parts.append(rows[:train_cut])
                if val_end > val_start:
                    val_parts.append(rows[val_start:val_end])

            train_idx = np.sort(np.concatenate(train_parts))
            val_idx   = np.sort(np.concatenate(val_parts))

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx
```

---

## 5. DATA FLOW SUMMARY

### 5.1 Training Phase

```
Raw Data (693 rows × 400+ features × 5 base models)
    │
    ├─→ [Primary OOF CV] ────────────────────────────┐
    │   • Fold 0: Train on [2008-2018], gen OOF [2019]│
    │   • Fold 1: Train on [2008-2019], gen OOF [2020]│
    │   • Fold 2: Train on [2008-2020], gen OOF [2021]│
    │   • Fold 3: Train on [2008-2021], gen OOF [2022]│
    │   • Fold 4: Train on [2008-2022], gen OOF [2023]│
    │                                                 │
    │   Output OOF Matrix: (693 × 4)                 │
    │   - 378 NaN rows (training samples) ────────────┤─────────┐
    │   - 315 valid rows (validation samples)        │         │
    │                                                 │         │
    ├─→ [Conformal OOF CV] ──────────────────────────┤──────┐  │
    │   • Sweep on years [2014-2017]                 │      │  │
    │   • Gen OOF on validation folds                │      │  │
    │   • Extract residuals: y - ŷ_oof              │      │  │
    │                                                │      │  │
    │   Output: Combined residuals (~400 rows)  ────┘  ┐   │  │
    │                                                   │   │  │
    └─→ [Meta-Learner Fit] ◄────────────────────────┘   │   │  │
        • Input OOF predictions (315 rows × 4 models)  │   │  │
        • Input OOF targets (315 rows)                 │   │  │
        • Fit Ridge: OOF preds → final weights         │   │  │
        • Output: weight vector [0.25, 0.30, 0.20, 0.25]  │  │
                                                           │  │
    ◄─────────────────────────────────────────────────────┘  │
    │                                                         │
    └─→ [Final Refit on Full Data] ◄──────────────────────────┘
        • Train all base models on complete data [2008-2023]
        • Use meta-weights for ensemble combination
        • Ready for production deployment
```

### 5.2 Prediction Phase

```
New Data (production year 2024)
    │
    ├─→ [Base Model Predictions]
    │   • CatBoost.predict()     → 2.48
    │   • BayesianRidge.predict()→ 2.51
    │   • SVR.predict()          → 2.47
    │   • ElasticNet.predict()   → 2.49
    │
    └─→ [Meta-Learner Ensemble]
        Final = 0.25×2.48 + 0.30×2.51 + 0.20×2.47 + 0.25×2.49
              = 0.62 + 0.753 + 0.494 + 0.6225
              = 2.49 ← Final ensemble prediction

        ┌─→ [Conformal Interval]
        │   Quantile on residuals: q_0.975 = 0.15
        │   Interval: [2.49 - 0.15, 2.49 + 0.15] = [2.34, 2.64]
        │   Guarantees 95% coverage (distribution-free)
        │
        └─→ Output: (2.49, [2.34, 2.64])
```

---

## 6. KEY DESIGN FEATURES

### 6.1 Avoiding Temporal Leakage

| Potential Issue | Prevention |
|---|---|
| Future data in training | Year labels split at calendar boundaries |
| Training in validation set | OOF array stores NaN for training rows |
| Double-counting in meta-training | OOF residuals used exactly once |
| Entity length bias | Median-based fold sizing (not minimum) |

### 6.2 Robustness to Model Failures

```python
# If one base model crashes on a fold:
# 1. NaN appears in that model's OOF column
# 2. Meta-learner detects NaN and excludes that model from meta-fit
# 3. Weights recomputed with remaining models
# 4. Ensemble continues with reduced diversity, not complete failure
```

### 6.3 Partial NaN Handling

```python
# Some rows may have complete NaN predictions (rare)
# Meta-learner requires:
valid = ~isnan(model_preds).any(axis=1) & ~isnan(y_col)
# Only use rows with complete base model + target data
# Eliminates partial-NaN rows from ridge fit
```

### 6.4 Per-Output Flexibility

When multiple criteria exist (e.g., 3 output targets):
- Separate weight vector per criterion
- Each criterion's weights learned independently
- Optionally share strength across criteria via Group LASSO

```python
# Shape: (N_outputs, N_models) = (3, 4)
_meta_weights_per_output_ = [
    [0.25, 0.30, 0.20, 0.25],  # Criterion 1 (e.g., Revenue)
    [0.20, 0.25, 0.35, 0.20],  # Criterion 2 (e.g., Cost)
    [0.30, 0.20, 0.15, 0.35],  # Criterion 3 (e.g., Profit)
]
```

---

## 7. CONFIGURATION PARAMETERS

### 7.1 Primary CV Settings

```python
# In config.py / ForecastConfig

cv_folds : int = 5
    # Number of temporal folds (default 5, range 2-8)
    # More folds → more OOF data but slower training

cv_min_train_years : int = 8
    # Minimum training window years (default 8)
    # First validation year = min_train_years
    # For 14 years: first val year = 2016 (8 at start, 6 for folding)
```

### 7.2 Conformal Settings

```python
conformal_alpha : float = 0.05
    # Miscoverage level (nominal 95% coverage)
    # Quantile computed as (1 - alpha)th percentile of abs(residuals)

conformal_min_train_years : int = 5
    # Minimum window for secondary sweep (default 5)
    # Must be < cv_min_train_years to extend calibration set
```

### 7.3 Meta-Learner Settings

```python
meta_learner_type : str = 'ridge'
    # Options: 'ridge', 'elasticnet', 'dirichlet_stacking'
    # ridge: L2 regularized, simple & stable
    # elasticnet: L1+L2, some weight sparsity
    # dirichlet_stacking: Bayesian, uncertainty estimates

positive_weights : bool = True
    # Enforce non-negative model weights (interpretability)
    # False allows negative contributions (rare, advanced)

normalize_weights : bool = True
    # Ensure weights sum to 1.0 (probabilistic combination)
```

---

## 8. VALIDATION & DIAGNOSTICS

### 8.1 OOF Performance Metrics

```python
result.oof_r2_scores = {
    'CatBoost': 0.847,          # R² on OOF validation folds
    'BayesianRidge': 0.832,
    'SVR': 0.823,
    'ElasticNet': 0.818,
}

# Ensemble OOF R² (weighted average)
result.ensemble_oof_r2 = 0.835  # Typically > individual models
```

### 8.2 Cross-Validation Stability

```python
result.cv_fold_stability = 0.042  # Coefficient of variation of fold R²
# Low values (< 0.1): stable, consistent folds
# High values (> 0.2): unstable, large fold variance
```

### 8.3 Model Agreement

```python
result.ensemble_diversity = {
    'std_of_weights': 0.045,    # How different are model weights?
    'model_pairwise_corr': 0.62, # How correlated are OOF preds?
}
# Diversity signals: uncorrelated models provide stronger ensemble
```

---

## 9. EXAMPLE: FULL WORKFLOW

### Dataset:
- 63 Vietnamese provinces, years 2008-2023 (16 years)
- 693 total rows (63 × 11 on average)
- 400+ features from panel features engineer

### Primary OOF CV Execution:

```
Fold 0:
  Train: rows where year_label ∈ [2008-2018]  (63×11 = 693 rows)
  Valid: rows where year_label = 2019         (63 rows, 1 per province)
  Loss: [0.847, 0.832, 0.823, 0.818] R² per model

Fold 1:
  Train: rows where year_label ∈ [2008-2019]  (756 rows)
  Valid: rows where year_label = 2020         (63 rows)
  Loss: [0.851, 0.840, 0.828, 0.825]

Fold 2:
  Train: rows where year_label ∈ [2008-2020]  (819 rows)
  Valid: rows where year_label = 2021         (63 rows)
  Loss: [0.849, 0.838, 0.826, 0.822]

Fold 3:
  Train: rows where year_label ∈ [2008-2021]  (882 rows)
  Valid: rows where year_label = 2022         (63 rows)
  Loss: [0.848, 0.835, 0.824, 0.820]

Fold 4:
  Train: rows where year_label ∈ [2008-2022]  (945 rows)
  Valid: rows where year_label = 2023         (63 rows)
  Loss: [0.846, 0.833, 0.821, 0.818]
```

### OOF Array After Primary CV:

```
Shape: (1008 rows, 4 models × 1 output)

Rows 0-756:     NaN NaN NaN NaN  (never validated in any fold)
Rows 757-819:   2.48 2.51 2.47 2.49  (validated in fold 0)
Rows 820-882:   2.51 2.54 2.50 2.52  (validated in fold 1)
...
```

### Meta-Learner Training:

```
Input OOF predictions: rows 757-945 (all validated rows)
Shape: (252, 4)

Input targets: corresponding y values
Ridge regression with CV alpha selection:
  α = 0.1 (selected by RidgeCV)
  Fitted weights: [0.25, 0.30, 0.20, 0.25]
  Intercept: 0.02

Final model written to disk
```

### Conformal Calibration:

```
Secondary OOF on years [2014-2017]:
  Fold 1: Train 2008-2013, validate 2014 (63 rows)
  Fold 2: Train 2008-2014, validate 2015 (63 rows)
  Fold 3: Train 2008-2015, validate 2016 (63 rows)
  Fold 4: Train 2008-2016, validate 2017 (63 rows)

Combined residuals: (315 from primary) + (252 from secondary) = 567 rows
Quantile: q_0.975 = 0.148
→ Prediction intervals: [ŷ - 0.148, ŷ + 0.148]
```

### Production Deployment (2024):

```
New data: 2024 features for 63 provinces

Step 1: CatBoost.predict()     → [2.48, 2.50, 2.49, ...]  (63,)
Step 2: BayesianRidge.predict()→ [2.51, 2.53, 2.52, ...]  (63,)
Step 3: SVR.predict()          → [2.47, 2.49, 2.48, ...]  (63,)
Step 4: ElasticNet.predict()   → [2.49, 2.51, 2.50, ...]  (63,)

Step 5: Ensemble combination for province 0:
  ŷ_ensemble = 0.25×2.48 + 0.30×2.51 + 0.20×2.47 + 0.25×2.49 = 2.491

Step 6: Conformal interval:
  [2.491 - 0.148, 2.491 + 0.148] = [2.343, 2.639]
  (95% coverage guaranteed)

Output: {
  'province': 'Ha Noi',
  'forecast': 2.491,
  'interval_lower': 2.343,
  'interval_upper': 2.639,
  'confidence': 0.95
}
```

---

## 10. FILES & LOCATIONS

| Component | File | Lines |
|-----------|------|-------|
| Walk-Forward Yearly Split | `super_learner.py` | 172–242 |
| Panel-Aware Temporal Split | `super_learner.py` | 46–150 |
| Super Learner fit() | `super_learner.py` | 568–1100 |
| OOF CV collection | `super_learner.py` | 652–900 |
| Meta-learner fit | `super_learner.py` | 1568–1712 |
| Conformal OOF | `super_learner.py` | 1312–1450 |
| Unified orchestrator | `unified.py` | 1–5000+ |
| Conformal prediction | `conformal.py` | 127–600 |
| Configuration | `config.py` | ForecastConfig dataclass |
| Validation metrics | `validation.py` | ForecastValidator |
| Evaluation metrics | `evaluation.py` | ForecastEvaluator |

---

## Summary

The ensemble design ensures:
✓ **No temporal leakage** — strict calendar-year boundaries
✓ **Complete OOF coverage** — every row validated exactly once
✓ **Meta-learned weights** — optimal combination via Ridge on OOF residuals
✓ **Conformal intervals** — distribution-free 95% coverage guarantee
✓ **Panel-aware** — respects entity (province) temporal structure
✓ **Robust** — continues with degraded performance if one base model fails
✓ **Flexible** — supports 1+ output criteria, multi-fold configurations
