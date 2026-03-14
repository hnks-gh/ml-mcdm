# Phase C Enhancement & Adjustment Plan

**Date**: 2026-03-15
**Prerequisites**: Phase A (T-01, T-04a, T-04b complete), Phase B (T-02, T-03 complete)
**Status**: In Implementation
**Scope**: Ensemble architecture flowchart fix + comprehensive forecasting figure data flow hardening

---

## Table of Contents

1. [T-04c: Fix Ensemble Architecture Flowchart](#t-04c-fix-ensemble-architecture-flowchart)
2. [F-05a: Fix `_inc` → `_safe` in `generate_all()`](#f-05a-fix-_inc--_safe-in-generate_all)
3. [F-05b: Harden Stage 6 OOF Evaluation Data Flow](#f-05b-harden-stage-6-oof-evaluation-data-flow)
4. [F-05c: Populate `per_model_*` in `training_info`](#f-05c-populate-per_model_-in-training_info)
5. [Cross-Module Dependency Graph](#cross-module-dependency-graph)
6. [Test Plan](#test-plan)

---

## T-04c: Fix Ensemble Architecture Flowchart

### Problem

`ForecastPlotter.plot_ensemble_architecture()` (`forecast_plots.py` ~L732) renders
a 3-tier flowchart with **hardcoded** stale base model names:

```python
base_models = [
    ('LightGBM', 'Gradient\nBoosting'),
    ('XGBoost', 'Gradient\nBoosting'),      # ← XGBoost is NOT in the pipeline
    ('Random\nForest', 'Bagging'),           # ← wrong; QuantileRF ≠ Random Forest
    ('Neural\nAdditive', 'NAM'),
    ('Hierarchical\nBayes', 'Bayesian'),
    ('Panel\nVAR', 'Time-Series'),
]
```

After Phase B the default 6-model ensemble is:
`CatBoost | LightGBM | BayesianRidge | KernelRidge | SVR | QuantileRF`
with `PanelVAR` and `NAM` as optional opt-in additions.

The figure also:
- Is **never called** in `generate_all()` — it exists as a method but is never invoked.
- Shows a fixed 6-box layout that breaks if fewer or more models are active.

### Fix Design

**1. Add `model_names` parameter** — accept the live model name list:

```python
def plot_ensemble_architecture(
    self,
    model_names: Optional[List[str]] = None,
    model_contributions: Optional[Dict[str, float]] = None,
    save_name: str = 'fig16b_ensemble_architecture.png',
) -> Optional[str]:
```

Resolution order:
- If `model_names` provided → use directly.
- Else if `model_contributions` provided → `list(model_contributions.keys())`.
- Else → internal default 6-model fallback list.

**2. Build a canonical label mapping** for each known model key:

| Key | Display Name | Sub-label |
|-----|-------------|-----------|
| `CatBoost` | `CatBoost` | `Symmetric GB` |
| `LightGBM` | `LightGBM` | `Leaf-wise GB` |
| `BayesianRidge` | `Bayesian\nRidge` | `Group Lasso` |
| `KernelRidge` | `Kernel\nRidge` | `RBF L2` |
| `SVR` | `SVR` | `ε-insensitive` |
| `QuantileRF` | `Quantile\nRF` | `Distributional` |
| `PanelVAR` | `Panel\nVAR` | `Fixed Effects` |
| `NAM` | `Neural\nAdditive` | `Interpretable` |
| *(unknown)* | *(key as-is)* | *(empty)* |

**3. Dynamic layout** — compute `np.linspace(x_left, x_right, n_models)` at render time.
- Box width = `min(2.1, 13.6 / n_models - 0.3)` to prevent overlap.
- Arrow base comes from the data-source box top-center, not a hardcoded x=8.

**4. Wire into `generate_all()`** — add a `_safe()` call just before the existing
`plot_model_weights_donut` (fig19) call, passing `_contribs` for model order:

```python
# fig16b — ensemble architecture flowchart
_safe(self.forecast.plot_ensemble_architecture,
      model_names=list(_contribs.keys()) if _contribs else None,
      model_contributions=_contribs or None)
```

### Files Modified

| File | Change |
|------|--------|
| `output/visualization/forecast_plots.py` | Rewrite `plot_ensemble_architecture()` to accept `model_names` param; data-driven layout |
| `output/visualization/__init__.py` | Add `_safe(self.forecast.plot_ensemble_architecture, ...)` call in forecast block |

---

## F-05a: Fix `_inc` → `_safe` in `generate_all()`

### Problem

The entire forecast block in `generate_all()` (~lines 478–579) is wrapped in ONE
outer `try/except`. All individual figure calls use `_inc()`, which increments a counter
but does **not catch exceptions**. Therefore:

```
if forecast_result is not None:
    try:
        ...
        _inc(fn1(...))   ← crash here → jumps to outer except
        _inc(fn2(...))   ← never reached
        ...
    except Exception as _exc:
        _logger.warning(...)  ← all subsequent figures skipped
```

A single exception in any of the 13+ figure calls silently suppresses all later figures.
This is why only `fig18`, `fig19`, `fig19b` (which appear to succeed) generate, while
`fig16`, `fig17`, `fig20`, `fig21`, `fig22`, `fig23*` are silently skipped.

### Fix Design

**Remove the outer `try/except`** and **replace every `_inc(fn(...))` with `_safe(fn, ...)`**:

```python
# BEFORE:
try:
    _inc(self.forecast.plot_forecast_scatter(_a, _p, entity_names=_ent))
except Exception as _exc:
    _logger.warning(...)

# AFTER:
_safe(self.forecast.plot_forecast_scatter, _a, _p, entity_names=_ent)
```

`_safe()` already exists in `generate_all()` and has the correct semantics:
- Wraps call in `try/except`
- Logs `WARNING: Figure skipped [<method>]: <exc>` on failure
- Increments `count` only on success
- Does NOT abort subsequent figures

**Signature change for `_safe`**: Each plotmethod is passed as the callable,
positional args as `*args`, keyword args as `**kwargs`. This means converting:

```python
_inc(self.forecast.plot_holdout_comparison(
    _a, _p,
    per_model_predictions=_per_model_ho,
    entity_names=_ent,
    model_contributions=_contribs or None,
))
```

to:

```python
_safe(self.forecast.plot_holdout_comparison,
      _a, _p,
      per_model_predictions=_per_model_ho,
      entity_names=_ent,
      model_contributions=_contribs or None)
```

### Guard Conditions Preserved

The guard `if _actual is not None and _predicted is not None:` remains — `_safe()`
cannot check pre-conditions, so explicit guards stay for figures that would be
meaningless without data. The key change is that failure of one figure within a
guard does not cascade to other figures in the same guard block.

### Files Modified

| File | Change |
|------|--------|
| `output/visualization/__init__.py` | Remove outer `try/except`; replace all `_inc(fn(...))` with `_safe(fn, ...)` in forecast block |

---

## F-05b: Harden Stage 6 OOF Evaluation Data Flow

### Problem

`stage6_evaluate_models()` in `forecasting/unified.py` Stage 6b:

```python
y_oof = y_arr[_oof_mask]           # ← may contain NaN (governance targets)
y_oof_pred = _oof_preds[_oof_mask, :y_arr.shape[1]]
self.holdout_performance_ = {
    'r2': float(r2_score(y_oof.ravel(), y_oof_pred.ravel())),  # ← NaN → crash
    ...
}
self._holdout_y_test_ = y_oof.ravel()   # ← NaN contaminated
self._holdout_y_pred_ = y_oof_pred.ravel()
```

Issues:
1. **NaN in `y_oof`**: Governance targets (marked NaN by M-04 complete-case strategy)
   cause `r2_score` to crash or return NaN, then the `except Exception` at line 2311
   swallows the error and leaves `_holdout_y_test_` = `None`.
2. **No fallback to stage 6a data**: When stage 6b fails or OOF < 5 samples,
   `_holdout_y_test_` stays `None` even though stage 6a may have produced valid
   genuine holdout predictions in `_ens_ho_arr` / `y_holdout_`.
3. **Entity names not collected**: `training_info['test_entities']` is always `None`
   because entity collection never happens in either stage 6a or 6b.

### Fix Design

#### Sub-fix B1: NaN-safe OOF filtering

After extracting `y_oof` and `y_oof_pred`, filter out NaN rows *before* computing
metrics and assigning to `_holdout_y_test_`:

```python
y_oof      = y_arr[_oof_mask]
y_oof_pred = _oof_preds[_oof_mask, :y_arr.shape[1]]

# Remove rows where the actual target contains NaN (governance targets)
_nan_row_mask = np.isnan(y_oof).any(axis=1)
_clean        = ~_nan_row_mask
if _clean.sum() < 5:
    # still insufficient after NaN removal
    ...
else:
    y_oof_clean      = y_oof[_clean]
    y_oof_pred_clean = y_oof_pred[_clean]
    self.holdout_performance_ = {
        'r2':   float(r2_score(y_oof_clean.ravel(), y_oof_pred_clean.ravel())),
        'rmse': float(np.sqrt(mean_squared_error(y_oof_clean, y_oof_pred_clean))),
        'mae':  float(mean_absolute_error(y_oof_clean, y_oof_pred_clean)),
        'n_oof': int(_clean.sum()),
        'note': 'OOF cross-validation estimate (genuinely out-of-sample; NaN rows excluded)',
    }
    self._holdout_y_test_ = y_oof_clean.ravel()
    self._holdout_y_pred_ = y_oof_pred_clean.ravel()
    # Phase 5 inverse-transform (unchanged)
```

#### Sub-fix B2: OOF entity names

Capture entity names for the OOF clean rows:

```python
# After the _clean mask is computed:
_oof_indices     = np.where(_oof_mask)[0]       # absolute row positions in X_train_
_oof_clean_idx   = _oof_indices[_clean]         # rows surviving NaN filter
_oof_entity_names = list(self.X_train_.index[_oof_clean_idx])
# stored into training_info below
```

#### Sub-fix B3: Stage 6a fallback for y_test/y_pred

If stage 6b leaves `_holdout_y_test_` = `None` (OOF insufficient or failed), and
stage 6a ran successfully (genuine holdout exists), populate from the holdout year:

```python
# After stage 6b block, before _training_info_ construction:
if self._holdout_y_test_ is None and hasattr(self, '_ho_y_test_fallback_'):
    self._holdout_y_test_ = self._ho_y_test_fallback_
    self._holdout_y_pred_ = self._ho_y_pred_fallback_
```

Where `_ho_y_test_fallback_` is set in stage 6a:
```python
# In stage 6a, after _ens_ho_arr is computed:
_y_ho_flat  = self.y_holdout_.values.ravel()
_y_ho_pred_flat = _ens_ho_arr.ravel()
_nan_ho = np.isnan(_y_ho_flat) | np.isnan(_y_ho_pred_flat)
if (~_nan_ho).sum() >= 5:
    self._ho_y_test_fallback_  = _y_ho_flat[~_nan_ho]
    self._ho_y_pred_fallback_  = _y_ho_pred_flat[~_nan_ho]
    self._ho_entity_names_fallback_ = list(
        self.X_holdout_.index[~_nan_ho[:len(self.X_holdout_)]]
    )
```

### Summary of stage6 data flow after fixes:

```
Stage 6a (genuine holdout year)
  ├── Compute _ens_ho_arr (ensemble predictions on holdout)
  ├── Compute per-model holdout predictions → _per_model_ho_preds_
  ├── Set _ho_y_test_fallback_, _ho_y_pred_fallback_
  └── Set _ho_entity_names_fallback_ = list(X_holdout_.index)

Stage 6b (OOF CV)
  ├── Extract y_oof, y_oof_pred from OOF mask
  ├── Remove NaN rows → y_oof_clean, y_oof_pred_clean
  ├── Set _holdout_y_test_ = y_oof_clean.ravel()
  ├── Set _holdout_y_pred_ = y_oof_pred_clean.ravel()
  └── Set _oof_entity_names_ = X_train_.index[oof_clean_indices]

Fallback (when stage 6b fails):
  ├── _holdout_y_test_ ← _ho_y_test_fallback_
  ├── _holdout_y_pred_ ← _ho_y_pred_fallback_
  └── _oof_entity_names_ ← _ho_entity_names_fallback_

_training_info_ construction:
  ├── y_test        = _holdout_y_test_
  ├── y_pred        = _holdout_y_pred_
  └── test_entities = _oof_entity_names_ (or holdout fallback)
```

### Files Modified

| File | Change |
|------|--------|
| `forecasting/unified.py` | Stage 6a: compute `_ho_y_test_fallback_`; Stage 6b: NaN-safe filtering + entity names; fallback after 6b; update `_training_info_` |

---

## F-05c: Populate `per_model_*` in `training_info`

### Problem

`training_info['per_model_holdout_predictions']` and `training_info['test_entities']`
are hardcoded `None`. Four data keys needed for Phase D figures are never populated:

| Key | Needed By | Currently |
|-----|-----------|-----------|
| `per_model_feature_importance` | fig18b | Never set |
| `per_model_oof_predictions` | fig24a, fig24b | Never set |
| `per_model_holdout_predictions` | fig16c, fig17b, fig20c | Always `None` |
| `test_entities` | fig23d | Always `None` |

### Fix Design

#### Part C1: `per_model_feature_importance`

After the ensemble feature importance is aggregated (line 2319), iterate fitted models:

```python
_per_model_imp = {}
for _mname, _mobj in self.super_learner_._fitted_base_models.items():
    try:
        _imp = _mobj.get_feature_importance()
        if _imp is not None and len(_imp) > 0:
            _per_model_imp[_mname] = np.asarray(_imp, dtype=float)
    except Exception:
        pass   # model does not support importance
```

The importance array from each base model has length equal to its input feature count
(PCA track: `n_components_pca`; tree track: `n_components_tree`). For uniform
fig18b heatmap rendering, the dict is stored as-is and the visualization layer handles
alignment.

#### Part C2: `per_model_oof_predictions` — SuperLearner change

**`forecasting/super_learner.py`**: After the `oof_ensemble` assembly
(currently ends at line 717), the local `oof_predictions` array (shape
`(n_samples, n_models × n_outputs)`) is discarded. We save per-model slices:

```python
# Immediately after line 717:
# self._oof_valid_mask_ = ~np.isnan(oof_ensemble).any(axis=1)
_oof_valid = self._oof_valid_mask_
self._oof_predictions_per_model_ = {}
for m_idx, name in enumerate(self.base_models):
    _cols = slice(m_idx * self._n_outputs, (m_idx + 1) * self._n_outputs)
    _per = oof_predictions[:, _cols]            # (n_samples, n_outputs)
    # Restrict to the same rows as the ensemble OOF (rows valid for all models)
    self._oof_predictions_per_model_[name] = _per[_oof_valid]  # (n_oof, n_outputs)
```

This costs ~290 KB for 6 models × 630 OOF rows × 8 outputs × 8 bytes = negligible.

**`forecasting/unified.py`**: In `_training_info_` construction, retrieve:

```python
_per_model_oof = getattr(self.super_learner_, '_oof_predictions_per_model_', {})
```

#### Part C3: `per_model_holdout_predictions`

In stage 6a, after `_per_model_X_holdout` is built and before `compare_all_models()`:

```python
_per_model_ho_preds = {}
for _mname, _mobj in self.super_learner_._fitted_base_models.items():
    if _mname in _per_model_X_holdout:
        try:
            _pho = _mobj.predict(_per_model_X_holdout[_mname])
            if np.ndim(_pho) == 0:
                _pho = np.atleast_2d(_pho)
            elif _pho.ndim == 1:
                _pho = _pho.reshape(-1, 1)
            _per_model_ho_preds[_mname] = _pho  # (n_holdout, n_outputs)
        except Exception:
            pass
```

These predictions are in **transformed space** if `target_transformer_` was applied.
For visualization (scatter, residual plots), both actual and predicted in the same
space is sufficient — cross-model comparison is self-consistent.

#### Part C4: `test_entities` (already covered in F-05b)

From Sub-fix B2, `_oof_entity_names_` is set in stage 6b (or fallback from stage 6a).
Used in `training_info['test_entities']`.

#### Updated `_training_info_` dict

```python
self._training_info_ = {
    # ... existing keys unchanged ...
    'y_test':         self._holdout_y_test_,
    'y_pred':         self._holdout_y_pred_,
    'test_entities':  getattr(self, '_oof_entity_names_', None),
    'per_model_holdout_predictions': _per_model_ho_preds,  # from C3 (stage 6a)
    'per_model_feature_importance':  _per_model_imp,        # from C1
    'per_model_oof_predictions':     _per_model_oof,        # from C2 (SuperLearner)
}
```

### Files Modified

| File | Change |
|------|--------|
| `forecasting/super_learner.py` | After line 717: save `_oof_predictions_per_model_` dict from local `oof_predictions` |
| `forecasting/unified.py` | Stage 6a: compute `_per_model_ho_preds`; stage 6b: entity name tracking; `_training_info_`: add 3 new keys + populate `test_entities` |

---

## Cross-Module Dependency Graph

```
Phase C Data Flow (after all fixes)
─────────────────────────────────────────────────────────────────────────
forecasting/super_learner.py
  └── fit():
        ├── oof_predictions (local, n_samples × n_models*n_outputs)
        ├── _oof_ensemble_predictions_  (existing)
        ├── _oof_valid_mask_            (existing)
        └── _oof_predictions_per_model_ (NEW: Dict[model_name → (n_oof, n_outputs)])

forecasting/unified.py  stage6_evaluate_models()
  ├── Stage 6a:
  │     ├── _per_model_X_holdout        (existing, dynamic per B)
  │     ├── _per_model_ho_preds         (NEW: per-model holdout predictions)
  │     ├── _ho_y_test_fallback_        (NEW: flat holdout actuals, NaN-filtered)
  │     ├── _ho_y_pred_fallback_        (NEW: flat ensemble holdout predictions)
  │     └── _ho_entity_names_fallback_  (NEW: holdout entity name list)
  ├── Stage 6b:
  │     ├── _holdout_y_test_           (hardened: NaN-filtered OOF actuals)
  │     ├── _holdout_y_pred_           (hardened: NaN-filtered OOF predictions)
  │     └── _oof_entity_names_         (NEW: entity names for OOF valid rows)
  ├── Fallback:
  │     └── _holdout_y_test_/_pred_ ← stage 6a data when 6b unavailable
  └── _training_info_:
        ├── y_test                          (hardened: populated reliably)
        ├── y_pred                          (hardened: populated reliably)
        ├── test_entities                   (hardened: populated reliably)
        ├── per_model_holdout_predictions   (NEW)
        ├── per_model_feature_importance    (NEW)
        └── per_model_oof_predictions       (NEW)

output/visualization/__init__.py  generate_all()
  ├── fig16b: _safe(plot_ensemble_architecture, model_names, ...)   (NEW call)
  ├── fig16:  _safe(plot_forecast_scatter, ...)                      (was _inc)
  ├── fig17:  _safe(plot_forecast_residuals, ...)                    (was _inc)
  ├── fig22b: _safe(plot_conformal_coverage, ...)                    (was _inc)
  ├── fig16c: _safe(plot_holdout_comparison, ...)                    (was _inc)
  ├── fig18:  _safe(plot_feature_importance, ...)                    (was _inc)
  ├── fig19:  _safe(plot_model_weights_donut, ...)                   (was _inc)
  ├── fig19b: _safe(plot_model_contribution_dots, ...)               (was _inc)
  ├── fig20:  _safe(plot_model_performance, ...)                     (was _inc)
  ├── fig21:  _safe(plot_cv_boxplots, ...)                           (was _inc)
  ├── fig22:  _safe(plot_prediction_intervals, ...)                  (was _inc)
  └── fig23/23b/23c: _safe(plot_rank_change_bubble/..., ...)         (was _inc)
```

---

## Test Plan

### T-04c Tests

1. **Unit**: `plot_ensemble_architecture(model_names=['CatBoost', 'LightGBM', 'BayesianRidge', 'KernelRidge', 'SVR', 'QuantileRF'])` → figure saved, no crash, titled correctly.
2. **Unit**: `plot_ensemble_architecture()` (no args) → uses default fallback, renders 6 boxes.
3. **Unit**: `plot_ensemble_architecture(model_contributions={'CatBoost': 0.4, 'LightGBM': 0.3, 'SVR': 0.3})` → derives model_names from contributions.
4. **Unit**: 8-model set with PanelVAR + NAM → dynamically spaces 8 boxes, no overlap.
5. **Regression**: call from `generate_all()` → fig16b generated in figures/forecasting/.

### F-05a Tests

1. **Unit**: Mock `generate_all()` where first figure raises `RuntimeError` → remaining figures still generated.
2. **Regression**: Full pipeline run → all figures that have data generate independently.

### F-05b Tests

1. **Unit**: `stage6_evaluate_models()` with `y_arr` containing 20% NaN rows → `_holdout_y_test_` non-None, `holdout_performance_['n_oof']` reflects cleaned count.
2. **Unit**: OOF count < 5 after NaN removal + stage 6a supplied → `_holdout_y_test_` populated from holdout fallback.
3. **Unit**: Both stage 6a and 6b succeed → stage 6b preferred (OOF), `test_entities` = OOF entity names.
4. **Regression**: Full pipeline → `training_info['y_test']` non-None, `training_info['y_pred']` non-None.

### F-05c Tests

1. **Unit**: `super_learner_._oof_predictions_per_model_` → dict with n_models entries, each shape `(n_oof, n_outputs)`.
2. **Unit**: `training_info['per_model_feature_importance']` → dict with one array per model; tree-track models have `n_features_tree` elements, PCA-track models have `n_features_pca` elements.
3. **Unit**: `training_info['per_model_holdout_predictions']` → dict when holdout year exists; `None` or empty dict when no holdout.
4. **Regression**: Full pipeline → `fig16c_holdout_comparison.png` renders with per-model lines (not Meta-Learner-only).
5. **Regression**: All 358+ existing tests still pass (no regression).

---

## Implementation Checklist

- [ ] `docs/phase_c_enhancement_plan.md` — this document ✓
- [ ] `output/visualization/forecast_plots.py` — T-04c: data-driven `plot_ensemble_architecture()`
- [ ] `output/visualization/__init__.py` — F-05a: `_inc` → `_safe`; add fig16b call
- [ ] `forecasting/super_learner.py` — F-05c C2: save `_oof_predictions_per_model_`
- [ ] `forecasting/unified.py` — F-05b + F-05c: stage 6 hardening + training_info population

*End of Phase C Plan.*
