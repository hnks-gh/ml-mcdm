# Audit Remediation Plan

## Objective

Repair the forecast stack in the order most likely to recover real signal:

1. Preserve missingness structure until the panel imputer runs.
2. Make fold-aware demeaning consistent across all evaluation paths.
3. Stop discarding usable partially observed targets.
4. Rebuild conformal calibration only after the upstream stack is fixed.

## Priority 1: Fix the feature path

### Problem

`_create_features_safe()` currently row-fills residual NaNs before the panel-level imputation stack can act.

### Action

- Change the feature builder so residual NaNs survive until `PanelSequentialMICE` or the configured imputer runs.
- If a fallback is still needed, use training-column fallback values plus an explicit missingness mask.
- Preserve `_was_missing` indicators end-to-end.

### Exit Criteria

- Feature matrices entering preprocessing still contain NaNs when expected.
- Panel imputation reduces NaNs and appends indicators as designed.
- No row-local mean imputation remains in the feature path.

## Priority 2: Make fold correction complete

### Problem

Entity-demean correction is applied to tree-track CV, but not to the PLS/PCA track or the current holdout metric path.

### Action

- Extend fold-aware correction to the linear track, or recompute those features before reduction.
- Apply the same correction logic to holdout evaluation.
- Rename any metric that is not truly untouched validation.

### Exit Criteria

- Tree and linear tracks are corrected with the same fold boundary.
- Holdout metrics are no longer optimistic because of uncorrected demeaning.

## Priority 3: Fix partial-target training semantics

### Problem

The upstream feature builder keeps partially observed rows, but `SuperLearner.fit()` drops any row with any NaN target.

### Action

- Train per criterion using rows where that criterion is observed.
- If multi-output models remain, pass criterion-specific masks or sample weights.
- Remove dead code paths that assume partial-target weighting is active when it is not.

### Exit Criteria

- Usable partial rows contribute to training.
- Data retention is measured per criterion and documented.

## Priority 4: Recalibrate uncertainty after the fixes

### Problem

Conformal intervals are calibrated on residuals produced by the current validation stack.

### Action

- Recompute OOF residuals after the feature and fold-correction fixes.
- Refit conformal calibration on the leakage-reduced residuals.
- Re-check empirical coverage on an untouched validation split.

### Exit Criteria

- Nominal and empirical coverage are aligned within acceptable tolerance.
- Interval width is stable across folds.

## Priority 5: Reassess model diversity only after the data path is fixed

### Problem

The current model set is smaller and less redundant than the original draft claimed, but the stack may still be overly coupled.

### Action

- Measure OOF prediction correlation after the upstream fixes.
- Run ablations on CatBoost, QuantileRF, BayesianForecaster, and KernelRidge.
- Replace a model only if the ablation shows no marginal value.

### Exit Criteria

- Each base model has a measurable contribution.
- Meta-weights are stable across folds.

## Verification Checklist

- [ ] Feature vectors retain NaNs until the intended imputer runs.
- [ ] Fold correction covers tree and linear tracks.
- [ ] Holdout metrics are clearly labeled and genuinely out-of-sample.
- [ ] Partial targets are handled with per-output masks or equivalent.
- [ ] Conformal coverage is re-validated after remediation.

## Recommended Order

1. Patch the feature path.
2. Patch fold correction.
3. Patch target handling.
4. Re-run conformal calibration and holdout evaluation.
5. Decide whether the model set needs to change.
