# FIX #1: Extend Fold Correction to PLS Track - Implementation Notes

## Problem Summary
- Entity demeaning happens GLOBALLY (using all years) before PLS compression
- PLS compression on globally demeaned features → non-linear absorption of future-year information
- fold_correction_fn only corrects tree models, not PLS models (BayesianRidge, KernelRidge)
- Result: 50% of ensemble (linear models) contains 5-15% optimistic bias from future-year leakage

## Current Architecture
1. UnifiedForecaster calls `feature_engineer.fit_transform(panel_data, target_year, holdout_year=_holdout_year)` ONCE
2. Entity means computed globally across all train_feature_years
3. Features created with global entity demeaning
4. Features reduced to tree track (via PanelFeatureReducer_tree) → X_train_tree_
5. Features reduced to PLS track (via PLS reducer) → PLS-compressed features
6. SuperLearner.fit() applies `fold_correction_fn` to tree models only
   - Computes fold-aware entity means using `compute_fold_entity_corrections(fold_max_feature_year)`
   - Adds correction offsets to _demeaned and _demeaned_momentum columns
   - **But**: PLS features remain uncorrected (already absorbed global demeaning non-linearly)

## Solution: Make Entity Demeaning Fold-Aware
Goal: Ensure entity means reflect ONLY fold's training window (no future-year leakage)

### Approach
Rather than calling fit_transform per-fold (expensive), we:
1. Modify fit_transform to accept optional `fold_year` parameter
2. When `fold_year` is provided, compute entity means from years < fold_year only
3. Call fit_transform per-fold from the CV loop (or batch-compute per-fold features)
4. This ensures BOTH tree and PLS tracks are fold-aware

### Implementation Steps

#### STEP 1: Modify fit_transform signature and entity mean computation
**Files**: `features.py`
- Add `fold_year: Optional[int] = None` parameter to `fit_transform()`
- Modify entity mean computation (lines 556-573) to:
  - If fold_year is None → use all years (current behavior)
  - If fold_year is provided → filter to years < fold_year before computing means
- Store fold_year as instance attribute for later reference

#### STEP 2: Trace fold_year propagation through  feature engineering
**Files**: `features.py` (_create_features and related methods)
- No changes needed; entity means are pre-computed and stored
- All entity-demeaned features derive from pre-computed means

#### STEP 3: Call fold-aware fit_transform from CV loop
**Files**: `unified.py` and/or `super_learner.py`
- Need to determine fold_max_feature_year for each CV fold
- Call fit_transform with appropriate fold_year for each fold
- Alternative: Compute fold-aware entity means post-hoc

#### STEP 4: Remove tree-only bypass in fold_correction_fn
**Files**: `unified.py` lines 543-544
- Remove the check that skips correction for non-tree models
- Apply correction uniformly to all models (correci no-op for truly fold-aware features)

## Technical Details

### Entity Mean Computation Logic
```python
# Current (lines 556-573):
for entity in unique_entities:
    entity_rows = entity_indices == entity
    _vals = X_feature[entity_rows, :]  ← Uses ALL rows
    mean = np.mean(_vals)

# New:
for entity in unique_entities:
    entity_rows = entity_indices == entity
    if fold_year is not None:
        entity_rows &= (train_years < fold_year)  ← Restrict
    _vals = X_feature[entity_rows, :]
    mean = np.mean(_vals)
```

### Per-Fold Feature Engineering Triggering
Current flow:
```
UnifiedForecaster.fit()
  → feature_engineer.fit_transform(panel_data, target_year, holdout_year=...)  [GLOBAL]
    → X_train_tree_, X_train_pls (global)
  → SuperLearner.fit(
       fold_correction_fn=_build_fold_correction_fn(...)  [POST-HOC]
    )
```

Proposed flow:
```
UnifiedForecaster.fit()
  → For each fold OR batch:
      → feature_engineer.fit_transform(
           panel_data,
           target_year,
           fold_year=fold_max_feature_year,  [FOLD-SPECIFIC]
           holdout_year=...
         )
    → X_train_tree_fold, X_train_pls_fold (fold-aware)
  → SuperLearner.fit(
       fold_correction_fn=None  [No correction needed; features already fold-aware]
    )
```

## Expected Outcomes
- OOF R² drops 5-10% (leakage correction; expected)
- All 4 base models (tree + linear) now use fold-aware features
- No future-year information in model inputs
- PLS track no longer optimistically biased
- Meta-learner sees unbiased OOF predictions

## Files to Modify
1. `features.py`
   - fit_transform() signature
   - Entity mean computation logic

2. `unified.py`
   - Strategy for per-fold fit_transform calls OR batch fold-aware feature prep
   - Remove tree-only bypass in fold_correction_fn

3. `super_learner.py`
   - If per-fold fit_transform is feasible: pass fold_year info through CV loop

## Risk Mitigation
- **Risk**: OOF metrics drop significantly
  - **Mitigation**: Document as EXPECTED and CORRECT (leakage removal)
- **Risk**: Per-fold fit_transform becomes performance bottleneck
  - **Mitigation**: Profile; consider batch processing or caching fold-aware entity means
- **Risk**: Changes to feature values break model expectations
  - **Mitigation**: Comprehensive validation; holdout comparison before/after

