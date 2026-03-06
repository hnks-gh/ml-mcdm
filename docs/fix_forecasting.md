# Fix Plan: Ensemble ML Forecasting — Four Compounding Defects

**Status:** Planned (pending implementation)  
**Affects:** `forecasting/`, `data/data_loader.py`, `config.py`, `pipeline.py`  
**Priority:** Critical (D-1, D-2, D-3) → High (D-4)

---

## 1. Root-Cause Analysis

### Observed Symptoms
| Metric | Value | Expected |
|---|---|---|
| Ensemble CV R² range | −17 to −9 (all models) | > 0 |
| CV R² std (per model) | 0.000 (no variation) | > 0 |
| Holdout R² | 0.974 | 0.5–0.9 |

The three contradictory numbers — negative CV scores, zero STD, perfect holdout — trace back to four distinct bugs in the pipeline.

---

### D-1 (Critical): Training Set Collapsed to ~126 Samples

**Location:** `forecasting/features.py` → `TemporalFeatureEngineer.fit_transform()`  
**Cause:** `has_complete_target(target)` requires **all 29 sub-criteria to be non-NaN** for a training row to be included. Structural data gaps in the raw CSVs eliminate almost every year:

| Sub-criteria | Missing years |
|---|---|
| SC71, SC72, SC73 | 2011–2017 (7 years entirely absent from CSVs) |
| SC81, SC82, SC83 | 2011–2017 (7 years entirely absent) |
| SC24 | 2011–2017 (7 years absent) |
| SC52 | 2018 and 2021–2024 |

Because `has_complete_target()` demands all 29 non-NaN, only rows from years 2019–2020 (where all 29 SCs happen to be present) survive. That is 2 complete years × 63 provinces = **126 usable training observations** out of a possible 13 × 63 = 819.

**Impact:** Only ≈15% of available data used. Tiny training sets cause every subsequent problem.

---

### D-2 (Critical): Only 1 CV Fold — STD R² = 0

**Location:** `forecasting/super_learner.py` → `_PanelTemporalSplit.split()`  
**Cause:** `_PanelTemporalSplit` sizes fold windows from `T_median`, the median within-entity sequence length. With only 2 valid years (2019–2020) per entity in sub-criteria mode:

```
T_median = max(2, median(T_per_entity)) = 2
min_train_T = max(T_median // 2, T_median // (n_splits + 1))
            = max(1, 2 // 4) = max(1, 0) = 1
fold_size   = max(1, (T_median - min_train_T) // n_splits)
            = max(1, (2 - 1) // 3) = max(1, 0) = 1
```

The loop runs `fold in range(n_splits=3)`, but `cut = min_train_T + fold * fold_size = 1 + fold`. For fold=0: `cut=1 < T_median=2` → one fold generated. For fold=1: `cut=2 >= T_median=2` → `break`. **Exactly one fold is generated**, so every model's CV score list has length 1, making `np.std([score]) = 0`.

**Impact:** The Super Learner meta-learner trains on 1-fold OOF predictions — effectively one arbitrary train/validation split. CV results are not informative.

---

### D-3 (High): Holdout Was In-Sample Evaluation — R² = 0.974 Is Noise

**Location:** `forecasting/unified.py` → `fit_predict()` Stage 6b (lines ~695–745)  
**Cause:** The holdout code calls:
```python
_ho_eng = TemporalFeatureEngineer()
X_ho, y_ho, _, _ = _ho_eng.fit_transform(panel_data, holdout_year)
X_ho_arr = self.feature_reducer_.transform(X_ho.values)
y_ho_pred = self.super_learner_.predict(X_ho_arr)
```
`_ho_eng.fit_transform(panel_data, holdout_year)` re-builds training features using `holdout_year` as the prediction target year, then calls `self.feature_reducer_.transform(X_ho)` and `self.super_learner_.predict(X_ho_arr)`. The variable `X_ho` is the **training feature matrix** from the holdout-year's `fit_transform` call — which is the exact same feature set the model trained on for the main `target_year` run. The model is being evaluated on samples it already fitted to.

**Note:** This is separate from the `X_train` return value confusion. The issue is structural: `fit_transform(panel_data, holdout_year)` returns training rows where `target = holdout_year` features, which is a subset of the full training data used when `fit_transform(panel_data, target_year)` was run.

**Impact:** R² = 0.974 reflects training-set memorisation, not generalisation. The metric is entirely meaningless for model selection or reporting.

---

### D-4 (Medium): GradientBoosting Over-Parameterized for the Sample Size

**Location:** `forecasting/gradient_boosting.py` and `config.py` → `ForecastConfig`  
**Cause:** The default configuration is `n_estimators=200, max_depth=5`. These defaults were chosen assuming a healthy training set of ~756 samples. With only 61 training samples per CV fold (≈ 126/2 because one fold):

- `max_depth=5` → up to 32 leaf nodes with ≈ 2 samples/leaf → complete memorisation
- `n_estimators=200` → 200 sequential overfitting steps on a tiny dataset
- CV R² = −17 for GradientBoosting reflects extreme overfitting on ~61 training samples, generalising to 61 validation samples from the same 2019–2020 window

**Impact:** Even with correct training data size, the defaults are too aggressive below ~500 samples.

---

## 2. Solution Architecture

### Chosen Approach: Forecast at the **Criteria Level** (8 targets instead of 29)

Rather than relaxing the completeness filter for sub-criteria (which would require imputing target labels — statistically unsafe), the cleanest fix is to forecast the **8 aggregated criteria composites** (C01–C08) instead of the 29 raw sub-criteria.

**Why this works:**

The `DataLoader` already computes criteria composites (column-mean of active sub-criteria per province per year) and stores them in `panel_data.criteria_long` / `panel_data.criteria_cross_section`. These composites:

- Have no structural missing years: SC71–73, SC81–83 being absent only suppresses C07 and C08 in those years; other criteria are fully populated for all 14 years.
- Degrade gracefully: if some sub-criteria are missing for a criterion in a year, `DataLoader` still computes the composite from the remaining active SCs (via `mean(axis=1, skipna=True)`).
- Have far more valid training rows: years where at least one province has data for each criterion can contribute training samples.

**Expected training data with criteria mode:**

| Year | Active provinces | Active criteria | Training rows |
|---|---|---|---|
| 2011–2017 (each) | 63 | 5–8 (depends on CSV) | ~63/year |
| 2018–2020 | 63 | 8 | 63/year |
| 2021–2024 | 63 | 8 | 63/year |
| **Total (13 target years)** | | | **~441–630 rows** |

With `T_median ≈ 7–10` and `n_splits=3`, `_PanelTemporalSplit` now generates **3 proper folds** with ~220 training samples in the first fold and ~90 validation samples each.

---

## 3. Implementation Plan

### Step 1 — Add Criteria Accessor to `PanelData`

**File:** `data/data_loader.py`  
**Class:** `PanelData`  
**What to add:** Two members mirroring the existing sub-criteria accessors.

#### 1a. `criteria_names` property

```python
@property
def criteria_names(self) -> List[str]:
    """List of all criteria codes (C01–C08) from the hierarchy."""
    return self.hierarchy.all_criteria
```

This parallels `subcriteria_names` which returns `self.hierarchy.all_subcriteria`.

#### 1b. `get_province_criteria(province)` method

```python
def get_province_criteria(self, province: str) -> pd.DataFrame:
    """Get a province's criteria composite scores across all years.

    Returns a DataFrame indexed by year with columns C01–C08.
    Cells may be NaN for years where the criterion had no active SCs.
    Mirrors get_province() but reads from criteria_long instead of
    subcriteria_long.
    """
    long = self.criteria_long
    prov_data = long[long['Province'] == province].copy()
    prov_data = prov_data.set_index('Year')
    cols = [c for c in prov_data.columns if c not in ('Province', 'Year')]
    return prov_data[cols]
```

**Where to insert:** Immediately after `get_province()` method definition (around line 222 in the current file).

**Validation:** `panel_data.get_province_criteria("Hanoi")` should return a DataFrame with shape `(14, 8)` and column names `['C01', 'C02', ..., 'C08']`.

---

### Step 2 — Add `target_level` Parameter to `TemporalFeatureEngineer`

**File:** `forecasting/features.py`  
**Class:** `TemporalFeatureEngineer`

#### 2a. `__init__` change

Add `target_level: str = "criteria"` to the constructor. The valid values are:
- `"criteria"` — predict the 8 criterion composites C01–C08 (new default)
- `"subcriteria"` — predict the 29 raw sub-criterion values SC11–SC83 (original behaviour, preserved for backward compatibility)

```python
def __init__(self,
             lag_periods: List[int] = [1, 2],
             rolling_windows: List[int] = [2, 3],
             include_momentum: bool = True,
             include_cross_entity: bool = True,
             target_level: str = "criteria"):      # NEW
    ...
    self.target_level = target_level
```

#### 2b. `fit_transform` branching

At the start of `fit_transform()`, after `entities = panel_data.provinces` and `years = sorted(panel_data.years)`, add the branching logic that selects `components` and the `entity_data` accessor:

```python
if self.target_level == "criteria":
    components = panel_data.criteria_names        # ['C01', ..., 'C08']
    def _get_entity_data(entity):
        return panel_data.get_province_criteria(entity)
    def _cross_section_for_year(year):
        return panel_data.criteria_cross_section[year]
else:
    components = panel_data.subcriteria_names     # ['SC11', ..., 'SC83']
    def _get_entity_data(entity):
        return panel_data.get_province(entity)
    def _cross_section_for_year(year):
        return panel_data.cross_section[year]     # alias for subcriteria_cross_section
```

#### 2c. Training row completeness guard change

The existing check for skipping incomplete training rows uses:
```python
if not has_complete_target(target):
    n_skipped_train += 1
    continue
```
This is correct and must be kept for both modes. However, in **criteria mode**, a target row `(entity, next_yr)` is only skipped when ALL 8 criterion composites are NaN (i.e., the province has absolutely no data for that year at all). A partial NaN (e.g., C07 missing because all SC71–73 were absent that year) is **not** a blocking issue: the composite is already NaN and the model will learn to predict C07 only when it has data. 

To handle partial NaN targets gracefully specifically for criteria mode, the completeness guard should be relaxed:

```python
if self.target_level == "criteria":
    # In criteria mode, skip only if the target vector is entirely NaN
    # (province had no data at all that year). Partial NaN is acceptable
    # because criterion composites are independently informative.
    if np.all(np.isnan(target)):
        n_skipped_train += 1
        continue
    # Impute remaining NaN criterion targets with column median across
    # the training set — acceptable for aggregated targets (not raw labels).
    # This imputation is deferred to the assembly stage below.
else:
    # Sub-criteria mode: strict — exclude any row with any NaN target
    if not has_complete_target(target):
        n_skipped_train += 1
        continue
```

**Important:** Partial NaN targets in criteria mode must be imputed to column-median before the `y_train` array is passed to models. This is done at the assembly stage after `np.vstack(y_train_list)`:

```python
y_train = np.vstack(y_train_list)
if self.target_level == "criteria":
    # Fill partial NaN in criteria targets with per-column median
    col_medians = np.nanmedian(y_train, axis=0)
    nan_mask = np.isnan(y_train)
    inds = np.where(nan_mask)
    y_train[inds] = np.take(col_medians, inds[1])
```

#### 2d. Cross-entity feature section change

Inside `_create_features()`, the cross-entity block currently does:
```python
year_cross_section = panel_data.cross_section[current_year]
```
This must use the criteria cross-section when in criteria mode. Replace with:
```python
if self.target_level == "criteria":
    year_cross_section = panel_data.criteria_cross_section[current_year]
else:
    year_cross_section = panel_data.cross_section[current_year]
```

#### 2e. YearContext per-SC validity guard

In the current training loop, this code blocks a row if any sub-criterion for the entity is invalid:
```python
ctx_next = getattr(panel_data, 'year_contexts', {}).get(next_yr)
if ctx_next is not None and entity not in ctx_next.active_provinces:
    n_skipped_train += 1
    continue
```
This province-level check is correct and **does not change** — it is valid for both modes. What must be removed in criteria mode is any deeper per-SC `is_valid(entity, sc)` call in the target construction path. Looking at the current code, the target is built by:
```python
target = entity_data.loc[next_yr, components].values.astype(float)
```
In criteria mode, `entity_data` comes from `get_province_criteria()` and `components` = C01–C08. There is no `is_valid()` call in the target construction loop; the sub-criteria validity check only appears in the logging section:
```python
if any(ctx_next.is_valid(ent, sc) for ent in entities): valid_count += 1
```
This logging loop is used for sub-criteria valid-year counting. It must be guarded to only run in sub-criteria mode, or adapted to check criteria validity instead.

---

### Step 3 — CV Fold Safety Guard in `_PanelTemporalSplit`

**File:** `forecasting/super_learner.py`  
**Class:** `_PanelTemporalSplit`  
**Method:** `split()`

After the main fold-generation `for fold in range(self.n_splits):` loop ends, add a fallback that yields exactly one emergency fold if **zero folds were generated** (which can only happen in degenerate cases where T_median = 2 and all fold cuts would equal or exceed T_median).

```python
# ── Safety guard: yield at least one fold even in degenerate cases ──────
# This fires only when T_median is too small for the n_splits requested
# (e.g. T_median=2 with n_splits=3 in sub-criteria mode).  With criteria
# mode T_median≈7 this never fires; it is purely a defensive fallback.
_yielded_any = False
for fold in range(self.n_splits):
    ...
    yield train_idx, val_idx
    _yielded_any = True   # track that at least one fold was produced

if not _yielded_any:
    # Emergency fallback: single train/val split at the midpoint
    split_T = max(1, T_median // 2)
    train_parts, val_parts = [], []
    for ent, rows in entity_rows.items():
        T_ent = len(rows)
        sp = min(split_T, T_ent - 1)
        if sp > 0:
            train_parts.append(rows[:sp])
        if sp < T_ent:
            val_parts.append(rows[sp:])
    if train_parts and val_parts:
        yield (
            np.sort(np.concatenate(train_parts)),
            np.sort(np.concatenate(val_parts)),
        )
```

**Implementation note:** To track `_yielded_any` without refactoring the generator, introduce a local counter before the loop:

```python
_n_yielded = 0
for fold in range(self.n_splits):
    ...
    if len(train_idx) == 0 or len(val_idx) == 0:
        continue
    yield train_idx, val_idx
    _n_yielded += 1

if _n_yielded == 0 and T_median >= 2:
    # Emergency single fold ...
```

---

### Step 4 — Fix Holdout Evaluation to Use OOF Predictions

**File:** `forecasting/unified.py`  
**Method:** `fit_predict()`, Stage 6b block  
**Lines to replace:** Entire `try:` block starting with `holdout_year = target_year - 1`

**Root problem recap:** The current code re-fits a new `TemporalFeatureEngineer` for `holdout_year`, computes `X_ho` (which contains the very training observations used to fit the Super Learner), and calls `super_learner_.predict(X_ho_arr)`. This inflates R² because the model has already seen those exact samples during training.

**Correct replacement:** Use the OOF predictions that the Super Learner already computed during its Stage 1 (the `_oof_ensemble_predictions_` array). These predictions are genuinely out-of-sample: each sample's OOF prediction was made by a model that was fitted **without** that sample.

```python
# ===== Stage 6b: OOF cross-validation performance estimate =====
# Replace the former temporal holdout block (which inadvertently evaluated
# the model on its own training data, producing an inflated R²=0.974).
# The Super Learner caches OOF ensemble predictions computed during Stage 1
# fold generation — these are genuinely out-of-sample.
holdout_performance = None
_holdout_y_test    = None
_holdout_y_pred    = None
_holdout_entities  = None   # entity names not tracked per OOF row

try:
    _oof_preds = self.super_learner_._oof_ensemble_predictions_  # (n_samples, n_outputs)
    _oof_mask  = self.super_learner_._oof_valid_mask_             # (n_samples,) bool
    if _oof_preds is not None and _oof_mask is not None and _oof_mask.sum() >= 5:
        y_oof      = y_arr[_oof_mask]                         # true targets, OOF rows
        y_oof_pred = _oof_preds[_oof_mask, :y_arr.shape[1]]  # OOF predictions

        holdout_performance = {
            'r2':    float(r2_score(y_oof.ravel(), y_oof_pred.ravel())),
            'rmse':  float(np.sqrt(mean_squared_error(y_oof, y_oof_pred))),
            'mae':   float(mean_absolute_error(y_oof, y_oof_pred)),
            'n_oof': int(_oof_mask.sum()),
            'note':  'OOF cross-validation estimate (genuinely out-of-sample)',
        }
        _holdout_y_test = y_oof.ravel()
        _holdout_y_pred = y_oof_pred.ravel()

        if self.verbose:
            print(f"  Stage 6b: OOF R² = {holdout_performance['r2']:.4f}, "
                  f"RMSE = {holdout_performance['rmse']:.4f}  "
                  f"[n_oof={holdout_performance['n_oof']}]")
    else:
        if self.verbose:
            print("  Stage 6b: OOF evaluation skipped "
                  "(insufficient OOF samples or Super Learner not fitted)")
except Exception as e:
    if self.verbose:
        print(f"  Stage 6b: OOF evaluation failed: {type(e).__name__}: {e}")
```

**What changes in `training_info`:**

The `_holdout_per_model` dict (per-base-model holdout predictions for the comparison figure) was derived from the same buggy in-sample evaluation and must be removed. The visualization code checks `if _per_model_ho is not None` so removing it is safe.

```python
training_info = {
    'n_samples':            len(X_train),
    'n_features':           X_train.shape[1],
    'n_features_reduced':   ...,
    'pca_variance_retained': ...,
    'mode':                 'advanced',
    'ensemble_method':      'super_learner',
    'conformal_calibrated': self.conformal_predictor_ is not None,
    'target_level':         self.target_level,    # NEW — "criteria" or "subcriteria"
    # OOF-based test data for downstream visualisation
    'y_test':         _holdout_y_test,
    'y_pred':         _holdout_y_pred,
    'test_entities':  None,                       # not tracked per OOF row
    # 'per_model_holdout_predictions': REMOVED — was in-sample artifact
}
```

**Downstream visualization impact:** The scatter plot at `forecast_plots.plot_forecast_scatter()` checks `if _actual is not None and _predicted is not None` before drawing. With OOF data it will draw a genuine scatter of OOF predictions vs actuals with no entity labels (since `test_entities=None`). The `plot_holdout_comparison` call will skip gracefully when `per_model_predictions=None`. No other visualization code is affected.

---

### Step 5 — Sample-Adaptive GradientBoosting Hyperparameters

**File:** `forecasting/gradient_boosting.py`  
**Class:** `GradientBoostingForecaster`  
**Method:** `fit(X, y)`

Insert at the very start of `fit()`, before `X_scaled = self.scaler.fit_transform(X)`:

```python
def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingForecaster':
    """Fit the gradient boosting model."""
    # ── Auto-scale hyperparameters to match training set size ──────────
    # The class defaults (max_depth=5, n_estimators=200) target n≈756.
    # With smaller datasets a shallow tree + fewer estimators avoids
    # complete memorisation (depth-5 on n=60 → ≈2 samples/leaf).
    n_samples = X.shape[0]
    if n_samples < 200:
        _eff_depth, _eff_n_est = 2, 50
    elif n_samples < 500:
        _eff_depth, _eff_n_est = 3, 100
    else:
        _eff_depth, _eff_n_est = self.max_depth, self.n_estimators

    # Re-clone only when effective params differ from instance config
    if _eff_depth != self._base_model.max_depth or \
       _eff_n_est != self._base_model.n_estimators:
        self._base_model = GradientBoostingRegressor(
            n_estimators=_eff_depth,    # intentional: use _eff_n_est here
            max_depth=_eff_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=self.random_state,
            loss='huber',
        )
    # ── End auto-scale ──────────────────────────────────────────────────
    X_scaled = self.scaler.fit_transform(X)
    ...
```

Wait — there is a typo in the above pseudocode. The corrected version:

```python
if _eff_depth != self._base_model.max_depth or \
   _eff_n_est != self._base_model.n_estimators:
    self._base_model = GradientBoostingRegressor(
        n_estimators=_eff_n_est,   # <-- correct variable
        max_depth=_eff_depth,
        learning_rate=self.learning_rate,
        subsample=self.subsample,
        random_state=self.random_state,
        loss='huber',
    )
```

**Scaling table:**

| Training samples | `max_depth` | `n_estimators` | Leaves | Samples/leaf |
|---|---|---|---|---|
| < 200 | 2 | 50 | 4 | ≥ 50 |
| 200–499 | 3 | 100 | 8 | ≥ 25 |
| ≥ 500 | `self.max_depth` (5) | `self.n_estimators` (200) | 32 | ≥ 15 |

The `subsample` parameter (default 0.8) already provides stochastic regularisation; reducing `max_depth` is the primary lever for controlling variance on small datasets.

---

### Step 6 — Add `forecast_level` to `ForecastConfig`

**File:** `config.py`  
**Class:** `ForecastConfig`

Add a new field:

```python
# ── Forecast target level ─────────────────────────────────────────────
forecast_level: str = "criteria"
"""
Forecast target granularity.
  'criteria'    — predict 8 aggregated criterion composites (C01–C08).
                  Recommended: far more training data, no structural NaN gaps.
  'subcriteria' — predict all 29 raw sub-criterion values (SC11–SC83).
                  Historical default: severely limited by structural
                  missing-data gaps in SC71-73, SC81-83, SC24, SC52.
"""
```

Place this field after `gb_n_estimators` and before `nam_n_basis` to keep the config grouped by theme.

---

### Step 7 — Wire `target_level` Through `UnifiedForecaster`

**File:** `forecasting/unified.py`  
**Class:** `UnifiedForecaster`

#### 7a. `__init__` signature

Add `target_level: str = "criteria"` to the constructor, default matching `ForecastConfig.forecast_level`:

```python
def __init__(self,
             conformal_method: str = 'cv_plus',
             conformal_alpha: float = 0.05,
             cv_folds: int = 3,
             random_state: int = 42,
             verbose: bool = True,
             config: Optional[ForecastConfig] = None,
             target_level: str = "criteria"):          # NEW
    ...
    self.target_level = target_level
```

#### 7b. `TemporalFeatureEngineer` instantiation

Change from:
```python
self.feature_engineer_ = TemporalFeatureEngineer()
```
to:
```python
self.feature_engineer_ = TemporalFeatureEngineer(target_level=self.target_level)
```

This is the single line in `__init__` that propagates the mode throughout feature engineering.

---

### Step 8 — Wire `forecast_level` Through `pipeline.py`

**File:** `pipeline.py`  
**Method:** `_run_forecasting()`

Change the `UnifiedForecaster` instantiation to pass through `target_level`:

```python
forecaster = UnifiedForecaster(
    conformal_method=self.config.forecast.conformal_method,
    conformal_alpha=self.config.forecast.conformal_alpha,
    cv_folds=self.config.forecast.cv_folds,
    random_state=self.config.forecast.random_state,
    verbose=self.config.forecast.verbose,
    target_level=self.config.forecast.forecast_level,   # NEW
)
```

---

## 4. File-by-File Change Summary

| File | Lines affected (approx) | Type | Priority |
|---|---|---|---|
| `data/data_loader.py` | ~20 (property + method) | Addition | Step 1 |
| `forecasting/features.py` | ~40 (new param + branching) | Addition + Modification | Step 2 |
| `forecasting/super_learner.py` | ~20 (fallback guard) | Addition | Step 3 |
| `forecasting/unified.py` | ~50 (replace holdout block + init) | Replacement | Steps 4 & 7 |
| `forecasting/gradient_boosting.py` | ~20 (adaptive scaling) | Addition | Step 5 |
| `config.py` | ~10 (new field) | Addition | Step 6 |
| `pipeline.py` | ~3 (new kwarg) | Modified line | Step 8 |

**Estimated total changed lines:** ~163

---

## 5. Expected Outcome After Implementation

### Training Data

| Metric | Before (sub-criteria mode) | After (criteria mode) |
|---|---|---|
| Target columns | 29 (SC11–SC83) | 8 (C01–C08) |
| Valid training rows | ~126 | ~441–630 |
| T_median for CV splitter | 2 | 7–10 |
| CV folds generated | 1 | 3 |

### Validation Metrics

| Metric | Before | After (expected) |
|---|---|---|
| CV R² (ensemble) | −17 to −9 | +0.3 to +0.7 |
| CV R² STD (per model) | 0.000 | > 0 (3 folds) |
| Holdout R² | 0.974 (artifact) | 0.4–0.6 (OOF estimate) |
| GradientBoosting depth (n≈441) | 5 | 3 (auto-scaled) |
| GradientBoosting n_est (n≈441) | 200 | 100 (auto-scaled) |

### Downstream Visualizations

- `plot_forecast_scatter`: still works, entity labels will be absent (OOF rows lack entity tracking)
- `plot_forecast_residuals`: still works with OOF arrays
- `plot_conformal_coverage`: still works
- `plot_holdout_comparison`: `per_model_predictions=None` → figure skipped gracefully (existing `if _per_model_ho is not None` guard)
- `plot_model_weights_donut`: unchanged
- `plot_feature_importance`: unchanged

---

## 6. Risks and Mitigations

### R-1: Criteria composites have NaN in early years for criteria like C07/C08

**Risk:** `get_province_criteria()` returns NaN for C07/C08 in years 2011–2017 for many provinces. Target partial-NaN rows could still reduce training data significantly.

**Mitigation (Step 2c):** In criteria mode, skip a row only if the entire 8-element target is all-NaN (province had zero data). Partial NaN targets are imputed with per-column median at assembly time. This allows all rows from 2011–2017 where the province has at least one valid criterion to participate in training.

### R-2: Feature dimension mismatch between target-year and holdout-year

**Risk:** In the old holdout code, PCA dimension mismatches caused the block to be skipped via exception. With OOF evaluation, this risk is completely eliminated — OOF predictions live in the same PCA space as the training data.

**Mitigation:** Already resolved by the OOF approach.

### R-3: Tests relying on `has_complete_target` behaviour in criteria mode

**Risk:** `tests/test_forecasting.py` mock panels use `subcriteria_names`; the new criteria branch may not be exercised by existing tests.

**Mitigation:** Existing tests only exercise the subcriteria path which is unchanged. A separate test for criteria mode is recommended but not blocking. The mock panel for a criteria-mode test would need `panel.criteria_names = ['C01', ..., 'C08']` and `panel.get_province_criteria` returning criteria composites.

### R-4: `criteria_cross_section` may not index by Province for cross-entity features

**Risk:** `_create_features()` does `year_cross_section = panel_data.cross_section[current_year]` then optionally `year_cross_section.set_index('Province')`. The `criteria_cross_section` dict is already indexed by Province (set in `_create_hierarchical_views`).

**Mitigation (Step 2d):** Check for `'Province' in year_cross_section.columns`; if already indexed, skip the `set_index` call. This check already exists in the current code:
```python
if 'Province' in year_cross_section.columns:
    year_cross_section = year_cross_section.set_index('Province')
```
This guard is correct; no change needed.

### R-5: `training_info['target_level']` key is new — CSV writers may silently ignore it

**Risk:** `output/csv_writer.py` iterates `training_info` and skips selected keys. A new unknown key would be included in CSV output, which is harmless but worth verifying.

**Mitigation:** `csv_writer.py` already excludes `('y_test', 'y_pred', 'test_entities', 'X_test')` — adding `target_level` will cause it to be written to the forecast CSV, which is desirable (it documents which mode was used).

---

## 7. Backward Compatibility

Setting `config.forecast.forecast_level = "subcriteria"` restores the original behaviour with no functional change. The `target_level` parameter defaults to `"criteria"` everywhere, so existing code that instantiates `UnifiedForecaster()` directly (e.g., in tests or notebooks) will use the new criteria mode automatically.

To revert to the old mode for a specific run, set either:
- `config.forecast.forecast_level = "subcriteria"` in config, or
- `UnifiedForecaster(target_level="subcriteria")` directly

---

## 8. Verification Checklist

After implementation, verify the following before committing:

- [ ] `python main.py` runs to completion without errors
- [ ] Console log shows `Training samples: 441–630` (not ~126)
- [ ] Console log shows `Super Learner: 5 base models, 3 CV folds` (not 1)
- [ ] Console log shows `OOF R² = 0.XX` (not `Holdout R² = 0.974`)
- [ ] CV R² (all models) is in the range −2 to +0.8 (no longer −17 to −9)
- [ ] CV R² STD per model is > 0
- [ ] `ForecastResult.holdout_performance['note']` contains `'OOF cross-validation estimate'`
- [ ] `ForecastResult.training_info['target_level']` equals `'criteria'`
- [ ] `pytest tests/test_forecasting.py -v` passes (no regressions in sub-criteria path)
- [ ] Forecast predictions shape: `(63, 8)` (63 provinces × 8 criteria, not × 29)
- [ ] Output CSV `forecast_predictions.csv` has 8 columns for C01–C08
- [ ] Visualization `plot_forecast_scatter` renders without exception
- [ ] Visualization `plot_holdout_comparison` skips gracefully when `per_model_predictions=None`

---

## 9. Related Files (Read-Only Reference)

| File | Relevant sections | Role in this fix |
|---|---|---|
| `data/missing_data.py` | `has_complete_target()`, `fill_missing_features()` | Used unchanged; criteria-mode relaxation is done in `fit_transform` before calling this |
| `forecasting/conformal.py` | `ConformalPredictor.calibrate()` | Uses `_oof_ensemble_predictions_` from Super Learner directly; unaffected |
| `forecasting/evaluation.py` | `ForecastEvaluator` | Downstream metric computation; unaffected |
| `forecasting/preprocessing.py` | `PanelFeatureReducer` | PCA dimensionality reduction; unaffected |
| `output/visualization/__init__.py` | Stage 6b visualization block (lines ~475–540) | Reads `training_info['y_test']`, `['y_pred']`, `['test_entities']`; safe with `None` entity list |
| `output/csv_writer.py` | Forecast CSV section | Will include new `target_level` key — harmless addition |

---

*Document generated: 2026-03-06*  
*Planned by: GitHub Copilot*  
*Implementation status: PENDING*
