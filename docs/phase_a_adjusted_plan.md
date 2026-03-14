# Phase A — Adjusted Implementation Plan

**Date**: 2026-03-15
**Status**: Implementation in Progress
**Scope**: T-01 (ER config toggle), T-04a (CatBoost rename), T-04b (Meta-Learner rename)
**Source plan**: `docs/enhancement_plan_config_models_figures.md`

---

## Overview

Phase A groups three independent, parallel workstreams that share zero runtime
dependencies on each other and on Phases B/C/D. All three can be merged atomically.

| Task | Files | Nature |
|------|-------|--------|
| T-01 | `config.py`, `ranking/hierarchical_pipeline.py`, `pipeline.py` | Behavioural toggle |
| T-04a | `forecasting/unified.py`, `forecasting/evaluation.py`, `pipeline.py`, `output/visualization/forecast_plots.py`, `tests/test_forecasting.py` | Rename dict key |
| T-04b | `output/visualization/forecast_plots.py`, `output/report_writer.py`, `pipeline.py` | Display string rename |

---

## T-01: ER Config Toggle — `use_evidential_reasoning`

### Design Rationale (adjusted from original plan)

The original plan proposed a weighted-average-of-method-scores fallback when ER is
disabled. **This fallback is removed** per user decision:

> "after turn off ER, there is method aggregation, i dont need any MCDM method
> aggregation (e.g., weighted average of method scores) as an alternative, just keep
> the 5 mcdm and the Base separate normally. any new MCDM method aggregation
> implementation is redundant."

**Correct design when `use_evidential_reasoning=False`:**

- Stage 1 (per-criterion MCDM) runs normally → 6 method scores per criterion stored.
- Stage 2 ER aggregation is **entirely skipped** — no composite ranking produced.
- `er_result = None`; `final_scores`, `final_ranking`, `kendall_w` all return `None`.
- `rank_fast()` returns `None` when ER is disabled.
- No new aggregation logic is introduced.

This is the statistically correct design: the 5 MCDM methods and Base remain as
independent, mutually comparable outputs. Consumers (sensitivity analysis, visualization)
already guard against `er_result is None` via existing try/except blocks.

### Config Change

```python
# config.py — RankingConfig
use_evidential_reasoning: bool = True   # True = preserve existing behaviour
"""When True (default), execute two-stage ER aggregation (Yang & Xu, 2002).
When False, Stage 1 MCDM scores are computed and stored but Stage 2 ER
aggregation is skipped entirely — no composite score is produced.
The 5 MCDM methods (TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS) and the
Base baseline remain as separate independent outputs.
"""
```

**Default `True`** — preserves all existing behaviour and all 350 tests pass unchanged.

### `HierarchicalRankingResult` Changes

```python
# Field becomes Optional
er_result: Optional[HierarchicalERResult]   # None when ER disabled

# All ER-delegating properties become None-safe
@property
def final_ranking(self) -> Optional[pd.Series]:
    return self.er_result.final_ranking if self.er_result is not None else None

@property
def final_scores(self) -> Optional[pd.Series]:
    return self.er_result.final_scores if self.er_result is not None else None

@property
def kendall_w(self) -> Optional[float]:
    return self.er_result.kendall_w if self.er_result is not None else None

def top_n(self, n: int = 10) -> Optional[pd.DataFrame]:
    return self.er_result.top_n(n) if self.er_result is not None else None

def summary(self) -> str:
    return self.er_result.summary() if self.er_result is not None else (
        "ER disabled — no composite ranking."
    )
```

### `HierarchicalRankingPipeline` Changes

```python
def __init__(self, ..., use_evidential_reasoning: bool = True):
    self._use_er = use_evidential_reasoning
    # er_aggregator is still instantiated for rank_fast() when ER is on

def rank(self, ...) -> HierarchicalRankingResult:
    # Stage 1 runs unconditionally
    # Stage 2 gated:
    if self._use_er:
        er_result = self.er_aggregator.aggregate(...)
        logger.info(f"  Kendall's W = {er_result.kendall_w:.4f}")
    else:
        er_result = None
        logger.info("  ER disabled — Stage 2 skipped. Individual method scores preserved.")
    return HierarchicalRankingResult(er_result=er_result, ...)

def rank_fast(self, ...) -> Optional[HierarchicalERResult]:
    if not self._use_er:
        return None
    return self.er_aggregator.aggregate(...)
```

### `pipeline.py` Changes

```python
pipeline = HierarchicalRankingPipeline(
    n_grades=self.config.er.n_grades,
    method_weight_scheme=self.config.er.method_weight_scheme,
    use_evidential_reasoning=self.config.ranking.use_evidential_reasoning,  # ← new
)
```

Both instantiation sites (`_run_hierarchical_ranking()` and `_rank_year()` inner
function in `_run_all_years_ranking()`) must pass the toggle.

### Downstream Impact Matrix

| Consumer | `er_result is None` handling |
|----------|-------------------------------|
| Sensitivity analysis (`ERSensitivityAnalysis`) | `rank_fast()` returns `None` → skip ER SA; ML SA unaffected |
| Visualization (fig01d belief heatmap, fig01e, fig15) | Already skip when `er_result is None` via try/except |
| Report writer | Gates ER sections on `er_result is not None` (existing guard) |
| `HierarchicalRankingResult.final_ranking` callers | Now returns `Optional[pd.Series]`; any caller must check `is not None` |

### Test Plan

1. `use_evidential_reasoning=True` (default) → all 350 existing tests pass unchanged.
2. New test `test_er_disabled_skips_stage2`: `use_evidential_reasoning=False` → `er_result is None`, `final_scores is None`, Stage 1 scores present.
3. New test `test_rank_fast_returns_none_when_er_disabled`.

---

## T-04a: Rename `GradientBoosting` → `CatBoost`

### Problem

`_create_models()` registers CatBoostForecaster under the key `'GradientBoosting'`.
This stale name propagates to all figures, CSV outputs, logs, and reports.

### Single Source of Truth

The fix is to change the dict key from `'GradientBoosting'` to `'CatBoost'` at the
single point where it is set: `models['GradientBoosting'] = CatBoostForecaster(...)`.
All downstream code iterates model dicts by key, so the rename propagates automatically.

### Exhaustive Change List

| File | Location | Old | New |
|------|----------|-----|-----|
| `forecasting/unified.py` | `_create_models()` model key | `models['GradientBoosting']` | `models['CatBoost']` |
| `forecasting/unified.py` | `_tune_gb_hyperparameters()` trial | `('GradientBoosting', ...)` | `('CatBoost', ...)` |
| `forecasting/unified.py` | HP lookup | `.get('GradientBoosting', {})` | `.get('CatBoost', {})` |
| `forecasting/unified.py` | `_per_model_X` dicts (all) | `'GradientBoosting'` | `'CatBoost'` |
| `forecasting/unified.py` | Model key branch (`if model_key == 'GradientBoosting'`) | `'GradientBoosting'` | `'CatBoost'` |
| `forecasting/unified.py` | Tree model name list (line ~1577) | `'GradientBoosting'` | `'CatBoost'` |
| `forecasting/evaluation.py` | Docstring identifier list | `'GradientBoosting'` | `'CatBoost'` |
| `output/visualization/forecast_plots.py` | `_family_color` dict | `'gradient': '#E74C3C'` (missing) | Add `'catboost': '#E74C3C'` |
| `pipeline.py` | `_base_model_names` list | `"GradientBoosting"` | `"CatBoost"` |
| `tests/test_forecasting.py` | Model key assertions | `"GradientBoosting"` | `"CatBoost"` |

**Exempt** (comments/docstrings explaining rationale — not key values):
- `forecasting/unified.py` docstring mentioning "Gradient Boosting (CatBoost)"
- `forecasting/gradient_boosting.py` module docstring
- `config.py` comment block header `# GradientBoosting hyperparameters`

### Color mapping

Add `'catboost': '#E74C3C'` to `_family_color` so CatBoost renders in red (same family
as xgboost/tree models). The existing `'gradient': '#E74C3C'` entry in the color map
never matched because the key `'GradientBoosting'` contains 'gradient' but the color
lookup is `.lower()` substring — however, `'catboost'` does not contain 'gradient', so
an explicit mapping is required.

---

## T-04b: Rename "Super Learner" → "Meta-Learner" (Display Strings Only)

### Scope

Only user-facing display strings (figure titles, axis labels, log messages, report
section headers). The `SuperLearner` Python class name is **not** changed.

### Exhaustive Change List

| File | Line(s) | Old | New |
|------|---------|-----|-----|
| `output/visualization/forecast_plots.py` | ~251 | `'Super Learner — Base Model Weights'` | `'Meta-Learner — Base Model Weights'` |
| `output/visualization/forecast_plots.py` | ~254 | `'Super\nLearner'` | `'Meta-\nLearner'` |
| `output/visualization/forecast_plots.py` | ~502 | `model_preds['Super Learner']` | `model_preds['Meta-Learner']` |
| `output/visualization/forecast_plots.py` | ~543 | `name == 'Super Learner'` | `name == 'Meta-Learner'` |
| `output/visualization/forecast_plots.py` | ~819 | flowchart Super Learner box | `'Meta-Learner'` |
| `output/visualization/forecast_plots.py` | ~875,955,957 | bubble chart titles | `'Meta-Learner'` |
| `output/report_writer.py` | ~520 | `'Super Learner meta-ensemble'` | `'Meta-Learner ensemble'` |
| `output/report_writer.py` | ~530 | `'## 8.1 Super Learner Model Contributions'` | `'## 8.1 Meta-Learner Model Contributions'` |
| `output/report_writer.py` | ~730 | `'Super Learner (van der Laan...)'` | `'Meta-Learner (van der Laan...)'` |
| `pipeline.py` | ~11 | `Super Learner` phase comment | `Meta-Learner` |
| `pipeline.py` | ~128 | `Super Learner meta-ensemble` docstring | `Meta-Learner ensemble` |
| `pipeline.py` | ~719 | `Super Learner meta-ensemble` in docstring | `Meta-Learner ensemble` |
| `pipeline.py` | ~744 | `"Meta-learner: Super Learner (Ridge)"` | `"Meta-learner: Meta-Learner (Ridge)"` |
| `pipeline.py` | ~761 | `"Super Learner weights:"` | `"Meta-Learner weights:"` |

---

## Implementation Order

```
T-04a rename (no logic change, mechanical substitution — lowest risk)
T-04b rename (same, pure display strings)
T-01  toggle  (logic change — add after renames so tests are cleaner to debug)
```

---

## Verification Checklist

- [ ] `grep -r "'GradientBoosting'" forecasting/ pipeline.py output/` → zero hits in runtime code
- [ ] `grep -r '"GradientBoosting"' forecasting/ pipeline.py output/ tests/` → zero hits
- [ ] `grep -r 'Super Learner' output/ pipeline.py` → zero hits (class `SuperLearner` exempt)
- [ ] `use_evidential_reasoning=True` → 350 existing tests pass
- [ ] `use_evidential_reasoning=False` → new tests pass, `er_result is None`
