# Ensemble ML Execution Audit and Remediation Plan

Date: 2026-03-21
Scope: Forecasting ensemble execution path (Stage 3 focus), CV behavior, and runtime risk on CPU-only Google Colab.

## 1) Executive Conclusion

The pipeline does not show a true infinite-loop defect in the ensemble training path. The primary issue is compute amplification from nested multi-output model fits and extra post-fold passes that are easy to mistake for a hang.

If logs show progress only up to the 5th CV fold, the job is often still in expensive downstream work that has weak progress visibility:

- Persistence baseline CV loop
- Extended conformal OOF sweep
- Full-data re-fit of all base models

On CPU-only Colab, this can add multiple hours after the visible fold logs stop.

## 2) What Was Audited

Key files reviewed:

- forecasting/unified.py
- forecasting/super_learner.py
- forecasting/quantile_forest.py
- forecasting/catboost_forecaster.py
- forecasting/hyperparameter_tuning.py
- config.py

## 3) Findings

### F1. No unbounded fold loop found (no clear forever execution bug)

- Main CV splitter (_WalkForwardYearlySplit) is bounded by max_folds and available years.
- Primary Stage 3 training loop is finite.
- No unguarded while-loop detected in the audited ensemble path.

Impact: Low (correctness), but does not solve runtime pain.

### F2. Hidden heavy work after "Fold 5" is likely causing "stuck" perception

In SuperLearner.fit(), after primary fold logs are done, the following still runs:

- Persistence CV loop
- Extended conformal residual sweep (_build_conformal_oof_residuals)
- Full re-fit of all base models

These sections have much less logging than the primary fold loop.

Impact: High (operability/user experience).

### F3. Runtime is very large by design under current defaults (especially QRF + subcriteria mode)

Current defaults imply:

- forecast_level = subcriteria (28 outputs)
- qrf_n_estimators = 300
- cv_folds = 5
- conformal secondary sweep enabled when cv_conformal_min_train_years < cv_min_train_years

QuantileRandomForestForecaster fits one forest per output column. This multiplies fit cost dramatically:

- Primary CV: 5 folds x 28 output forests
- Secondary sweep: additional folds x 28 output forests
- Final refit: 28 output forests

Impact: Critical (performance).

### F4. Secondary conformal sweep is compute-heavy and can duplicate expensive late-year folds

_build_conformal_oof_residuals uses max_folds=999 (bounded by years), then skips only years already present in primary OOF.

Because primary CV itself is capped by n_cv_folds, any uncovered years can still trigger secondary folds, including late years. This is finite but expensive.

Impact: High (performance).

### F5. Hyperparameter tuning splitter call is likely incorrect when tuning is enabled

In forecasting/hyperparameter_tuning.py, objective loops call:

- self.cv_splitter.split(X, y)

But the yearly splitter API expects split(X, year_labels), not y.

Today auto_tune_* defaults are false, so this is dormant in default runs. If enabled, this can lead to invalid split behavior and wasted compute.

Impact: Medium (latent bug, correctness/performance when tuning enabled).

## 4) Runtime Estimate for Current Situation

Assuming your log is at "Fold 5" in the primary CV loop, a practical CPU-only estimate is:

- Remaining time: about 2.5 to 4.5 more hours
- Total wall-clock time: about 5.5 to 7.5 hours

Reasoning:

- Primary fold loop is only part of Stage 3.
- Secondary conformal sweep can approach another fold block of similar complexity.
- Full re-fit and later stages still add substantial time.

This is an estimate band; exact time depends on Colab CPU quota, thread scheduling, and whether CatBoost / sklearn-quantile use all cores efficiently.

## 5) Consolidated Fix Plan (Priority Ordered)

### Phase A: Immediate safeguards (same day)

1. Add explicit stage-level progress logging and timers in SuperLearner.fit()
- Log start/end and elapsed for:
  - Primary CV
  - Persistence CV
  - Secondary conformal sweep
  - Final full-data refit
- Log per-fold elapsed time for all loops (including secondary sweep).

2. Add hard runtime guardrails in ForecastConfig
- max_total_stage3_minutes
- max_secondary_conformal_folds
- allow_skip_secondary_conformal_when_slow

3. Make "fast profile" config for Colab CPU
- forecast_level: criteria (temporary speed mode)
- qrf_n_estimators: 80-120
- cv_folds: 3
- cv_conformal_min_train_years = cv_min_train_years (disable secondary sweep)
- disable optional heavy modules (panel MICE, augmentation, shift detection)

Success criterion:
- User can see continuous progress after Fold 5.
- End-to-end runtime reduced by >= 50% in fast mode.

### Phase B: Correctness and latent bug fixes (1-2 days)

1. Fix hyperparameter tuning split API
- Pass year_labels to splitter when yearly CV is used.
- Keep backward-compatible fallback for non-year splitters.

2. Restrict secondary conformal sweep to true "early-gap years only"
- Explicitly compute uncovered years before the primary CV start.
- Do not include late years only missing due to fold cap (unless explicitly requested).

3. Add deterministic termination checks and telemetry
- Emit n_folds planned vs completed for each loop.
- Save per-loop diagnostics JSON to output/logs for post-mortem.

Success criterion:
- Tuning works with valid folds when enabled.
- Secondary sweep no longer performs unnecessary late-year expensive folds.

### Phase C: Structural performance optimization (2-5 days)

1. Introduce model-specific CV schedules
- Keep CatBoost on all folds.
- Reduce QRF/SVR/KRR fold count independently (e.g., 3 folds) and blend with robust meta-learner regularization.

2. Cache and reuse fold artifacts
- Avoid repeated transformations and repeated per-fold deep copies where safe.
- Cache per-fold transformed matrices for each track (tree/pca).

3. Add optional output-dimension batching
- Train only high-impact outputs first (criteria-level or subset of subcriteria).
- Run full 28-output mode as an offline "slow quality" profile.

Success criterion:
- Default CPU runtime drops to acceptable range (< 2.5 hours target for Colab CPU profile).

## 6) Recommended Default Profiles

### Colab CPU Fast (interactive)

- cv_folds = 3
- cv_min_train_years = 6
- cv_conformal_min_train_years = 6
- qrf_n_estimators = 100
- forecast_level = criteria
- auto_tune_* = false

Expected total runtime: roughly 1 to 2.5 hours.

### Research Full (offline)

- Keep current rich ensemble + subcriteria mode
- Run outside Colab free CPU, or schedule overnight
- Keep full diagnostics enabled

Expected total runtime: roughly 5.5 to 8+ hours depending on hardware.

## 7) Verification Checklist

- Confirm fold counters and stage timers are printed continuously.
- Confirm secondary sweep planned fold count is finite and expected.
- Compare runtime before/after with same random seed.
- Validate forecast metrics (R2/RMSE) do not regress beyond tolerance.
- Validate conformal coverage remains acceptable after sweep changes.

## 8) Final Recommendation

For immediate productivity on CPU-only Colab, use a fast profile and disable secondary conformal sweep duplication first. Then apply the Stage 3 logging and splitter fixes. This will resolve the "stuck at fold 5" observability issue and substantially reduce wall-clock time while preserving a path to full-quality offline training.
