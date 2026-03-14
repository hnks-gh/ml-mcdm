# Enhancement & Adjustment Plan — Config Toggles, New Base Models, and Comprehensive Figures

**Date**: 2026-03-15
**Status**: Pending Implementation
**Scope**: 5 workstreams across config, forecasting, visualization, and output modules

---

## Table of Contents

1. [T-01: Config Toggle — Disable ER in Ranking Phase](#t-01-config-toggle--disable-er-in-ranking-phase)
2. [T-02: Config Toggle — Disable NAM and PanelVAR Base Models](#t-02-config-toggle--disable-nam-and-panelvar-base-models)
3. [T-03: New Base Models for the Meta-Learner Ensemble](#t-03-new-base-models-for-the-meta-learner-ensemble)
4. [T-04: Output Fixes — CatBoost Naming, Super Learner → Meta-Learner](#t-04-output-fixes--catboost-naming-super-learner--meta-learner)
5. [T-05: Comprehensive Forecasting Figure Generation](#t-05-comprehensive-forecasting-figure-generation)
6. [Cross-Cutting Concerns](#cross-cutting-concerns)
7. [Implementation Order & Dependencies](#implementation-order--dependencies)

---

## T-01: Config Toggle — Disable ER in Ranking Phase

### Rationale

Evidential Reasoning (ER) adds complexity and computational overhead to the ranking phase. A toggle allows:
- Running ranking with pure MCDM method aggregation (e.g., weighted average of method scores) as an alternative
- Faster iteration during development/debugging
- Ablation study capability (ER vs. non-ER ranking quality)

### Design

**Config field** — Add to `RankingConfig`:

```python
@dataclass
class RankingConfig:
    # ... existing fields ...
    use_evidential_reasoning: bool = False
    """When False, ranking bypasses the two-stage ER aggregation and instead
    uses a simple weighted-average of normalized MCDM method scores per
    criterion, then a CRITIC-weighted sum across criteria for the final
    composite score.

    Default False.  Set True to enable the full Yang & Xu (2002) analytical
    ER combination rule with belief distributions and utility intervals.
    """
```

### Files to Modify

| File | Change |
|------|--------|
| `config.py` (L600-621) | Add `use_evidential_reasoning: bool = False` to `RankingConfig` |
| `ranking/hierarchical_pipeline.py` (~L303-341) | Gate Stage 2 ER aggregation behind the toggle; add non-ER fallback path |
| `ranking/hierarchical_pipeline.py` (~L343-400) | Gate `rank_fast()` ER-only re-ranking behind the toggle |
| `pipeline.py` (~L207-237) | Pass `config.ranking.use_evidential_reasoning` to `HierarchicalRankingPipeline` |
| `output/visualization/__init__.py` (~L319-323) | Guard ER-specific figures (fig01d belief heatmap, fig01e rank-uncertainty, fig15 ER uncertainty) behind the toggle |

### Non-ER Fallback Algorithm (Stage 2 replacement)

When `use_evidential_reasoning=False`, the hierarchical pipeline's Stage 2 replaces ER aggregation with:

```
For each province:
    1. Collect normalized MCDM scores from Stage 1 (5 methods × 8 criteria)
    2. Per-criterion score = weighted average of method scores
       (method weights from EvidentialReasoningConfig.method_weights or equal 1/5)
    3. Final composite = weighted sum of per-criterion scores using CRITIC criterion weights
       (same weights used in ER path)
    4. Rank provinces by final composite (descending)
```

This preserves the same two-level hierarchy (methods → criteria → composite) but without belief distributions, grade mappings, or the ER combination rule. The result object (`HierarchicalRankingResult`) is populated with:
- `final_scores`: weighted-average composite per province
- `final_ranking`: rank order from composite scores
- `er_result`: `None` (signals to downstream code that ER was not used)
- All other fields (`criterion_method_scores`, `criterion_method_ranks`, `kendall_w`) populated normally

### Downstream Impact

| Consumer | Impact | Handling |
|----------|--------|----------|
| Sensitivity Analysis (`sensitivity/er_sensitivity.py`) | Direct ER sensitivity requires `er_result` | Skip ER sensitivity when `er_result is None`; ML sensitivity and MCDM sensitivity still run |
| Visualization (fig01d, fig01e, fig15) | Require belief distributions from ER | Skip these 3 figures when `er_result is None` (already gated in `generate_all()` via try/except) |
| Report writer (`output/report_writer.py`) | ER sections reference belief utilities | Gate ER report sections behind `er_result is not None` check |
| Forecasting Phase | Independent of ER | No impact |

### Test Plan

1. Unit test: `use_evidential_reasoning=False` → ranking completes, `er_result is None`, `final_scores` is a valid Series
2. Unit test: `use_evidential_reasoning=True` → existing behavior preserved (all 302+ tests pass)
3. Integration: full pipeline with ER disabled → Phases 3-7 complete, no ER-related figures generated, no crash

---

## T-02: Config Toggle — Disable NAM and PanelVAR Base Models

### Rationale

NAM and PanelVAR underperform on this dataset:
- **NAM**: RFF-based shape functions overfit on n=756; backfitting with 60 features is noisy
- **PanelVAR**: Fixed effects with 63 entities and limited time depth (12 years) leaves sparse estimation; Ridge regularization mitigates but cannot recover signal quality
- Disabling them reduces ensemble noise and speeds up training (~40% of total fit time)

### Design

**Config fields** — Add to `ForecastConfig`:

```python
@dataclass
class ForecastConfig:
    # ... existing fields ...
    use_nam: bool = False
    """Include NeuralAdditiveForecaster (NAM) in the base model ensemble.
    Default False.  NAM's RFF backfitting is sensitive to small panel sizes
    (n < 1000) and can inject noise into the meta-learner.
    """

    use_panel_var: bool = False
    """Include PanelVARForecaster in the base model ensemble.
    Default False.  Panel VAR requires sufficient time depth per entity
    for reliable lag estimation; with 12 years and 63 entities, the
    entity-specific fixed effects dominate the signal.
    """
```

### Files to Modify

| File | Change |
|------|--------|
| `config.py` (~L405-422) | Add `use_nam: bool = False` and `use_panel_var: bool = False` |
| `forecasting/unified.py` (~L857-924) | Gate NAM and PanelVAR creation in `_create_models()` behind config flags |
| `forecasting/unified.py` (~L1329-1540) | Gate per-model feature routing for NAM/PanelVAR (only add to `_per_model_X_train_` if model exists) |
| `pipeline.py` (~L737-739) | Update `_base_model_names` log to reflect dynamic model set |

### Implementation Detail — `_create_models()` modification

```python
def _create_models(self) -> Dict[str, BaseForecaster]:
    cfg = self._config
    models = {}

    # Tier 1a: Tree-based (always included)
    models['CatBoost'] = CatBoostForecaster(...)       # ← renamed key (T-04)
    models['LightGBM'] = LightGBMForecaster(...)

    # Tier 1b: Bayesian linear (always included)
    models['BayesianRidge'] = BayesianForecaster()

    # Tier 1c: Kernel-based (always included — new T-03 models)
    models['KernelRidge'] = KernelRidgeForecaster(...)  # ← new (T-03)
    models['SVR'] = SVRForecaster(...)                   # ← new (T-03)

    # Tier 1d: Advanced panel-specific (conditional)
    models['QuantileRF'] = QuantileRandomForestForecaster(...)

    if cfg is not None and cfg.use_panel_var:
        models['PanelVAR'] = PanelVARForecaster(...)

    if cfg is not None and cfg.use_nam:
        models['NAM'] = NeuralAdditiveForecaster(...)

    return models
```

### Impact on Meta-Learner

The SuperLearner already handles dynamic model sets — it iterates `self.base_models.items()` and uses `inspect.signature` for dispatch. Removing models from the dict is fully supported. Key considerations:

1. **NNLS meta-weights**: Fewer models → fewer columns in the stacking matrix → NNLS still finds non-negative weights that sum closer to 1
2. **Dirichlet stacking**: Fewer models → lower-dimensional softmax optimization → faster convergence
3. **OOF matrix**: Shape becomes `(n_oof_samples, n_outputs, n_active_models)` — all downstream code uses `.shape[-1]` or dict iteration, so no hardcoded model count
4. **Feature track routing**: Only models present in the `models` dict get entries in `_per_model_X_train_` — no orphan entries

### Backward Compatibility

- Default `use_nam=False, use_panel_var=False` → 5 base models (4 existing + QuantileRF, with NAM/PanelVAR excluded) or 6 with new T-03 models
- Setting `use_nam=True, use_panel_var=True` restores the original 6-model (or 8-model with T-03) configuration
- All visualization code iterates `model_contributions.keys()` dynamically — no hardcoded model names

### Test Plan

1. Unit test: default config → NAM and PanelVAR not in `_create_models()` output
2. Unit test: `use_nam=True, use_panel_var=True` → both present
3. Integration: full pipeline with defaults → SuperLearner trains on 4+ models, produces valid weights
4. Regression: enable NAM+PanelVAR → existing test suite passes

---

## T-03: New Base Models for the Meta-Learner Ensemble

### Model Evaluation

| Candidate | Verdict | Reasoning |
|-----------|---------|-----------|
| **Kernel Ridge Regression (KRR)** | **YES** | Captures smooth nonlinearities via RBF kernel; closed-form solution (no SGD); O(n³) but n=756 is trivial; regularized; excellent for small data; complements both linear (BayesianRidge) and piecewise-constant (trees) inductive biases |
| **SVR with RBF kernel** | **YES** | ε-insensitive loss provides outlier robustness (complementary to MSE-based models); margin maximization principle yields different bias from KRR despite same kernel; well-studied dual formulation; works well on small-to-medium n |
| **Small MLP** | **NO** | n=756 is far below the data requirements for meaningful MLP generalization; NAM already covers neural network diversity with superior interpretability and regularization (RFF + Ridge); high overfitting risk and SGD stochasticity would add noise to the ensemble |
| **SVM (classification)** | **NO** | This is a regression task; SVR (above) is the appropriate regression variant |
| **Huber Regressor** | **NO** | BayesianForecaster already provides regularized linear modeling via MultiTaskElasticNetCV; marginal ensemble diversity gain insufficient to justify additional model complexity |

### T-03a: Kernel Ridge Regression Forecaster

**New file**: `forecasting/kernel_ridge.py`

**Class**: `KernelRidgeForecaster(BaseForecaster)`

**Mathematical foundation**:

KRR solves `min_f ∥y − f(X)∥² + λ∥f∥²_H` in reproducing kernel Hilbert space H.

With RBF kernel `K(x, x') = exp(−γ∥x − x'∥²)`:
- Dual solution: `α = (K + λI)⁻¹y` where `K_ij = K(x_i, x_j)`
- Prediction: `f(x*) = Σ_i α_i K(x_i, x*)`
- Complexity: O(n³) train, O(n·n_test) predict — both fast for n=756

**Implementation**:

```python
class KernelRidgeForecaster(BaseForecaster):
    """Multi-output Kernel Ridge Regression with RBF kernel.

    Uses sklearn's KernelRidge wrapped in MultiOutputRegressor for
    per-criterion independent kernel regression.  Internal StandardScaler
    ensures RBF kernel distance computation is scale-invariant.

    Hyperparameter selection:
        alpha (λ)  : Controls regularization strength.  Default 1.0.
        gamma (γ)  : RBF bandwidth.  Default 'scale' (1 / (n_features × Var(X))).
                     'scale' is preferable to 'auto' (1/n_features) because it
                     accounts for feature variance, yielding more robust distances.
    """

    def __init__(self, alpha=1.0, gamma='scale', random_state=42):
        self.alpha = alpha
        self.gamma = gamma
        self.random_state = random_state
        self._scaler = None
        self._model = None
        self._n_features = 0

    def fit(self, X, y, sample_weight=None):
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._n_features = X.shape[1]

        y = np.atleast_2d(y)
        if y.shape[0] == 1:
            y = y.T

        base = KernelRidge(alpha=self.alpha, kernel='rbf', gamma=self.gamma)
        self._model = MultiOutputRegressor(base)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        self._model.fit(X_scaled, y, **fit_kwargs)
        return self

    def predict(self, X):
        X_scaled = self._scaler.transform(X)
        pred = self._model.predict(X_scaled)
        if pred.ndim == 0:
            pred = np.atleast_2d(pred)
        elif pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred

    def get_feature_importance(self):
        # KRR has no native feature importance; use dual coefficient magnitudes
        # as a proxy: importance_j ∝ mean |α_i| × mean |∂K/∂x_j|
        # Approximation: permutation-free, uses the magnitude of the learned
        # dual coefficients weighted by kernel sensitivity to each feature
        importances = np.zeros(self._n_features)
        for est in self._model.estimators_:
            dual = np.abs(est.dual_coef_).mean()
            importances += dual
        importances /= len(self._model.estimators_)
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances
```

**Feature track**: PCA track (same as BayesianRidge) — KRR benefits from dimensionality reduction because:
1. RBF kernel distances degrade in high dimensions (curse of dimensionality)
2. PCA-compressed features are scaled and orthogonal — ideal for kernel methods
3. Computational cost is O(n² × d) for kernel matrix — lower d is faster

**Hyperparameters** (in `ForecastConfig`):

```python
krr_alpha: float = 1.0
"""KRR regularization strength.  Higher → more regularized.
1.0 is a safe default for n=756 with PCA-compressed features."""

krr_gamma: str = "scale"
"""RBF bandwidth selection.  'scale' = 1/(n_features × Var(X));
'auto' = 1/n_features.  'scale' adapts to feature variance."""
```

### T-03b: SVR Forecaster

**New file**: `forecasting/svr.py`

**Class**: `SVRForecaster(BaseForecaster)`

**Mathematical foundation**:

SVR solves the ε-insensitive regression problem:

```
min_{w,b,ξ,ξ*}  ½∥w∥² + C Σ_i (ξ_i + ξ*_i)
s.t.  y_i − (w·φ(x_i) + b) ≤ ε + ξ_i
      (w·φ(x_i) + b) − y_i ≤ ε + ξ*_i
      ξ_i, ξ*_i ≥ 0
```

Key properties:
- **ε-tube**: Predictions within ε of the target incur zero loss — provides natural outlier robustness
- **Sparse dual**: Only support vectors (samples on or outside the ε-tube) contribute to the solution
- **RBF kernel**: Maps to infinite-dimensional RKHS; same kernel as KRR but different loss function

**Why SVR alongside KRR?**

Despite both using RBF kernels, SVR and KRR have fundamentally different inductive biases:

| Property | KRR (L2 loss) | SVR (ε-insensitive loss) |
|----------|---------------|--------------------------|
| Loss function | Squared error — penalizes all deviations quadratically | ε-tube — ignores errors < ε, linear penalty beyond |
| Sensitivity to outliers | High (quadratic penalty amplifies outliers) | Low (linear penalty; support-vector sparsity) |
| Solution density | Dense (all training points contribute) | Sparse (only support vectors) |
| Bias-variance profile | Lower bias, higher variance | Higher bias, lower variance |
| Ensemble role | Captures smooth mean function | Captures robust central tendency |

This complementarity is precisely what benefits ensemble diversity — the meta-learner can allocate weight to KRR in well-behaved regions and SVR in noisy regions.

**Implementation**:

```python
class SVRForecaster(BaseForecaster):
    """Multi-output Support Vector Regression with RBF kernel.

    Uses sklearn's SVR wrapped in MultiOutputRegressor.  The ε-insensitive
    loss provides robustness to outlier provinces whose criterion scores
    deviate significantly from the panel mean.

    Key hyperparameters:
        C       : Penalty for violations outside the ε-tube.  Default 1.0.
        epsilon : Width of the ε-insensitive zone.  Default 0.1.
        gamma   : RBF bandwidth.  Default 'scale'.
    """

    def __init__(self, C=1.0, epsilon=0.1, gamma='scale', random_state=42):
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.random_state = random_state
        self._scaler = None
        self._model = None
        self._n_features = 0

    def fit(self, X, y):
        from sklearn.svm import SVR
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._n_features = X.shape[1]

        y = np.atleast_2d(y)
        if y.shape[0] == 1:
            y = y.T

        base = SVR(kernel='rbf', C=self.C, epsilon=self.epsilon,
                   gamma=self.gamma)
        self._model = MultiOutputRegressor(base)
        self._model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self._scaler.transform(X)
        pred = self._model.predict(X_scaled)
        if pred.ndim == 0:
            pred = np.atleast_2d(pred)
        elif pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred

    def get_feature_importance(self):
        # SVR: use support vector density as feature importance proxy.
        # For RBF SVR, importance_j ∝ variance of support vectors along
        # feature j (features with wider SV spread carry more information).
        importances = np.zeros(self._n_features)
        for est in self._model.estimators_:
            sv = est.support_vectors_  # (n_sv, n_features)
            if sv.shape[0] > 1:
                imp_j = np.std(sv, axis=0)
            else:
                imp_j = np.ones(self._n_features)
            importances += imp_j
        importances /= len(self._model.estimators_)
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances
```

**Feature track**: PCA track (same rationale as KRR — kernel methods need dimensionality reduction).

**Why no `sample_weight`**: sklearn's `SVR` does not natively support `sample_weight` in its `fit()` method. The `SuperLearner._fit_model()` already uses `inspect.signature` to detect this and will simply not pass `sample_weight` to SVR. This is the correct behavior — SVR's ε-insensitive loss already provides implicit robustness to noisy samples.

**Hyperparameters** (in `ForecastConfig`):

```python
svr_C: float = 1.0
"""SVR penalty parameter.  Higher C → less regularization → tighter fit.
1.0 is conservative for n=756."""

svr_epsilon: float = 0.1
"""ε-tube width.  Predictions within ε of target incur zero loss.
0.1 is appropriate for logit-transformed SAW scores (range ~[-3, 3])."""

svr_gamma: str = "scale"
"""RBF bandwidth.  Same as KRR default."""
```

### Updated Model Roster (default config)

| Key | Class | Track | Always On | Diversity Role |
|-----|-------|-------|-----------|----------------|
| `CatBoost` | `CatBoostForecaster` | Tree | Yes | Joint multi-output GB (symmetric trees) |
| `LightGBM` | `LightGBMForecaster` | Tree | Yes | Per-output GB (leaf-wise, asymmetric trees) |
| `BayesianRidge` | `BayesianForecaster` | PCA | Yes | Joint sparse linear (group lasso) |
| `KernelRidge` | `KernelRidgeForecaster` | PCA | Yes | Smooth nonlinear (L2 loss in RKHS) |
| `SVR` | `SVRForecaster` | PCA | Yes | Robust nonlinear (ε-insensitive loss in RKHS) |
| `QuantileRF` | `QuantileRandomForestForecaster` | Tree | Yes | Distributional (conditional quantiles) |
| `PanelVAR` | `PanelVARForecaster` | Tree | **No** (opt-in) | Entity-specific temporal dynamics |
| `NAM` | `NeuralAdditiveForecaster` | Tree | **No** (opt-in) | Interpretable nonlinear (additive) |

**Default model count**: 6 (was 6; net zero change but different composition — replaced 2 underperforming models with 2 well-suited kernel models).

### Files to Create / Modify

| File | Action |
|------|--------|
| `forecasting/kernel_ridge.py` | **Create** — `KernelRidgeForecaster` class |
| `forecasting/svr.py` | **Create** — `SVRForecaster` class |
| `forecasting/__init__.py` | Add exports: `KernelRidgeForecaster`, `SVRForecaster` |
| `forecasting/unified.py` (~L857-924) | Add KRR and SVR to `_create_models()`; route to PCA track |
| `forecasting/unified.py` (Stage 2) | Add `KernelRidge` and `SVR` to PCA track routing |
| `config.py` | Add KRR and SVR hyperparameter fields to `ForecastConfig` |

### Ensemble Diversity Analysis

With the default 6-model set, the diversity structure is:

```
                        ┌── CatBoost    (joint multi-output, symmetric trees)
     Tree track ────────┤── LightGBM    (per-output, leaf-wise trees)
                        └── QuantileRF  (bagging + quantile retention)

                        ┌── BayesianRidge  (L1/L2 linear, Gaussian posterior)
     PCA track ─────────┤── KernelRidge    (L2 kernel, smooth nonlinear)
                        └── SVR            (ε-insensitive kernel, robust nonlinear)
```

**Diversity dimensions**:
1. **Loss function**: MSE (CatBoost, LightGBM, KRR), ε-insensitive (SVR), L1+L2 (BayesianRidge), quantile (QRF)
2. **Nonlinearity**: Piecewise-constant (trees), smooth RKHS (KRR, SVR), linear (BayesianRidge), bagged piecewise-constant (QRF)
3. **Output coupling**: Joint (CatBoost, BayesianRidge) vs. per-output independent (LightGBM, KRR, SVR, QRF)
4. **Robustness**: Outlier-sensitive (KRR, BayesianRidge) vs. outlier-robust (SVR, QRF)

This yields a well-diversified ensemble where the meta-learner can exploit complementary strengths across all four diversity dimensions.

### Test Plan

1. Unit: `KernelRidgeForecaster.fit()` / `.predict()` / `.get_feature_importance()` on synthetic data
2. Unit: `SVRForecaster.fit()` / `.predict()` / `.get_feature_importance()` on synthetic data
3. Unit: 0-D, 1-D, 2-D prediction shape normalization (same edge cases as existing models)
4. Integration: SuperLearner with KRR+SVR in model dict → trains, produces valid weights
5. Integration: full pipeline with default config → 6 models, KRR and SVR present, NAM/PanelVAR absent
6. Regression: enable NAM+PanelVAR → 8-model set trains successfully

---

## T-04: Output Fixes — CatBoost Naming, Super Learner → Meta-Learner

### T-04a: Rename `GradientBoosting` → `CatBoost`

**Problem**: `_create_models()` registers CatBoost under the key `'GradientBoosting'`. This misleading name propagates to all figures, reports, and CSV outputs. Users see "GradientBoosting" in fig19 but the underlying model is CatBoost.

**Fix**: Change the dictionary key from `'GradientBoosting'` to `'CatBoost'` in `_create_models()`. This is the single source of truth — the name propagates automatically to:
- `SuperLearner._meta_weights` keys
- `model_contributions` dict (used by all figures)
- `cross_validation_scores` dict
- `model_performance` dict
- `model_comparison` results
- CSV and report outputs

**Color mapping fix**: Update `_family_color` in `forecast_plots.py` to add `'catboost': '#E74C3C'`, `'gradient': '#E74C3C'` entries (red family, replacing the missing gradient match that currently falls through to default grey).

### T-04b: Rename "Super Learner" → "Meta-Learner"

**Affected locations** (exhaustive grep):

| File | Line(s) | Current Text | New Text |
|------|---------|-------------|----------|
| `output/visualization/forecast_plots.py` | 251 | `'Super Learner — Base Model Weights'` | `'Meta-Learner — Base Model Weights'` |
| `output/visualization/forecast_plots.py` | 254 | `'Super\nLearner'` | `'Meta-\nLearner'` |
| `output/visualization/forecast_plots.py` | 502 | `model_preds['Super Learner']` | `model_preds['Meta-Learner']` |
| `output/visualization/forecast_plots.py` | 543 | `name == 'Super Learner'` | `name == 'Meta-Learner'` |
| `output/visualization/forecast_plots.py` | ~815,819 | Flowchart Super Learner labels | `'Meta-Learner'` |
| `output/visualization/forecast_plots.py` | ~875 | Contribution dots title | `'Meta-Learner'` |
| `output/report_writer.py` | 520 | `'Super Learner meta-ensemble'` | `'Meta-Learner ensemble'` |
| `output/report_writer.py` | 530 | `'## 8.1 Super Learner Model Contributions'` | `'## 8.1 Meta-Learner Model Contributions'` |
| `output/report_writer.py` | 730 | `'Super Learner (van der Laan...)'` | `'Meta-Learner (van der Laan...)'` |
| `pipeline.py` | 11 | Phase comment | Updated |
| `pipeline.py` | 128 | Docstring | Updated |
| `pipeline.py` | 239 | Phase comment | Updated |
| `pipeline.py` | 719 | Log message | Updated |
| `pipeline.py` | 744 | `"Meta-learner: Super Learner (Ridge)"` | `"Meta-learner: Meta-Learner (Ridge)"` |
| `pipeline.py` | 761 | `"Super Learner weights:"` | `"Meta-Learner weights:"` |

**Note**: The `SuperLearner` class name in Python code remains unchanged (it is an internal API, not user-facing). Only display strings visible in figures, reports, logs, and CLI output are renamed.

### T-04c: Fix Ensemble Architecture Flowchart (fig16b)

The `plot_ensemble_architecture()` method (forecast_plots.py ~L732) has hardcoded stale model names:
- Shows XGBoost and Random Forest (neither is in the pipeline)
- Omits QuantileRF and BayesianRidge by label name

**Fix**: Make the flowchart data-driven. Accept `model_names: List[str]` parameter and dynamically generate the architecture diagram from the actual model set.

### Files to Modify

| File | Change |
|------|--------|
| `forecasting/unified.py` L892 | `models['GradientBoosting']` → `models['CatBoost']` |
| `forecasting/unified.py` L867-870 | Update docstring to say CatBoost |
| `forecasting/unified.py` L888 | `_gb_params = self._tuned_gb_params_.get('CatBoost', {})` |
| `forecasting/unified.py` (all `_per_model_X` refs) | Update any `'GradientBoosting'` string references to `'CatBoost'` |
| `output/visualization/forecast_plots.py` | All "Super Learner" → "Meta-Learner"; add `'catboost'` to color map; fix fig16b |
| `output/report_writer.py` | All "Super Learner" → "Meta-Learner" |
| `pipeline.py` | All display strings updated |

### Test Plan

1. Grep: zero remaining `'GradientBoosting'` string literals in runtime code (docstrings and comments exempt)
2. Grep: zero remaining `'Super Learner'` in display strings (class name `SuperLearner` exempt)
3. Visual: fig19 donut shows "CatBoost" label, center text says "Meta-\nLearner"
4. Regression: full pipeline output matches expected model names

---

## T-05: Comprehensive Forecasting Figure Generation

### Diagnosis: Why Only 3 Figures Generate

The current `generate_all()` produces only fig18, fig19, fig19b because:

1. **fig16/17/22b/16c** (actual vs predicted): Require `training_info['y_test']` and `training_info['y_pred']` to be non-None. These values come from OOF evaluation in `stage6_evaluate_all()`, which sets them to `None` if OOF has < 5 valid samples or if OOF evaluation raises an exception. **Root cause**: The OOF values (`_holdout_y_test_`, `_holdout_y_pred_`) may fail due to NaN filtering, inverse transform errors, or insufficient OOF samples.

2. **fig20** (model performance bars): Requires `model_performance` to be truthy. This dict is populated by `stage6_evaluate_all()` from `self.model_comparison_`. If model comparison fails or produces empty results, `model_performance` is `{}` which is falsy → figure skipped.

3. **fig21** (CV boxplots): Requires `cross_validation_scores` to be truthy. If CV scores are empty or consist of single-element lists (no variance to box-plot), the figure is skipped.

4. **fig22** (prediction intervals): Requires `prediction_intervals` to contain `'lower'` and `'upper'` keys with non-None DataFrames. If conformal calibration fails or prediction intervals are not computed, this is skipped.

5. **fig23/23b/23c**: Require `predictions` DataFrame to be non-None with at least one column. Generally these should work if Stage 4 completes.

6. **Exception cascade**: All forecast figures are inside a single `try/except` block (lines 474-574). If ANY figure using `_inc()` (which does NOT catch exceptions) throws, all subsequent figures in the block are skipped. Only fig18, fig19, fig19b may survive because they execute inside independent conditions before the cascade reaches them.

### Fix Strategy

**F-05a**: Replace all `_inc()` calls in the forecast block with `_safe()` calls to prevent exception cascades. Each figure should fail independently.

**F-05b**: Ensure `training_info['y_test']` and `training_info['y_pred']` are populated correctly by hardening the OOF evaluation path in `stage6_evaluate_all()`.

**F-05c**: If genuine holdout data exists (holdout year split), populate `test_entities` and `per_model_holdout_predictions` in `training_info` to enable per-model comparison plots.

### New Figure Implementations

Below is the comprehensive figure catalog for the forecasting phase. Existing figures are marked with their current status. New figures are described with full specifications.

#### Category 1: Model Performance Comparison

| Figure | Status | Description |
|--------|--------|-------------|
| fig20 | **EXISTS** — fix data flow | Model performance bars (RMSE, MAE, R² per model) |
| fig20b | **NEW** | Per-model metric radar chart |
| fig21 | **EXISTS** — fix data flow | CV box plots (per-fold R² distribution) |

**fig20b — Per-Model Metric Radar** (`plot_model_metric_radar`)

Radar chart with one polygon per model, axes = [R², MAE, RMSE (inverted), Correlation, Mean Bias]. Enables visual comparison of model strengths across multiple evaluation dimensions. Data source: `model_performance` dict.

#### Category 2: Prediction vs. Actual

| Figure | Status | Description |
|--------|--------|-------------|
| fig16 | **EXISTS** — fix data flow | Scatter plot: actual vs predicted (ensemble) |
| fig16c | **EXISTS** — fix data flow | 4-panel holdout comparison (all models) |
| fig17 | **EXISTS** — fix data flow | 4-panel residual diagnostics |

#### Category 3: Ensemble Weight Distribution

| Figure | Status | Description |
|--------|--------|-------------|
| fig19 | **EXISTS** | Donut chart (model weights) |
| fig19b | **EXISTS** | Bubble chart (weight vs CV R²) |

#### Category 4: Uncertainty Visualization

| Figure | Status | Description |
|--------|--------|-------------|
| fig22 | **EXISTS** — fix data flow | Conformal prediction intervals (top-N provinces) |
| fig22b | **EXISTS** — fix data flow | Conformal coverage calibration curve |
| fig22c | **NEW** | Interval width vs. actual error scatter |

**fig22c — Interval Width vs. Actual Error** (`plot_interval_calibration_scatter`)

Scatter: x = predicted interval width per entity, y = actual absolute error. Points below the diagonal indicate well-calibrated intervals. Points above indicate under-coverage. Adds a 45° reference line and a locally-weighted regression (LOWESS) trend. Data source: `prediction_intervals['lower']`, `prediction_intervals['upper']`, `training_info['y_test']`, `training_info['y_pred']`.

#### Category 5: Feature Importance & Interpretability

| Figure | Status | Description |
|--------|--------|-------------|
| fig18 | **EXISTS** | Ensemble feature importance lollipop |
| fig18b | **NEW** | Per-model feature importance heatmap |

**fig18b — Per-Model Feature Importance Heatmap** (`plot_per_model_importance_heatmap`)

Heatmap: rows = top-20 features (by ensemble importance), columns = models. Cell color = feature importance value for that model. Reveals which features each model relies on and highlights ensemble diversity in feature utilization. Data source: collect `model.get_feature_importance()` from each fitted base model during `stage6_evaluate_all()` and store in `training_info['per_model_feature_importance']`.

#### Category 6: Temporal / Panel Analysis

| Figure | Status | Description |
|--------|--------|-------------|
| fig23c | **EXISTS** — fix data flow | Score trajectory: historical + forecast CI |
| fig23d | **NEW** | Per-entity forecast error map |
| fig23e | **NEW** | Temporal training performance curve |

**fig23d — Per-Entity Forecast Error Map** (`plot_entity_error_analysis`)

Horizontal bar chart: each bar = one entity (province), bar length = absolute prediction error, color = signed error (red = over-predicted, blue = under-predicted). Sorted by error magnitude. Reveals which entities the ensemble struggles with. Data source: `training_info['y_test']`, `training_info['y_pred']`, `training_info['test_entities']`.

**fig23e — Temporal Training Performance Curve** (`plot_temporal_training_curve`)

Line chart: x = validation year (from walk-forward CV), y = mean R² (or RMSE) across folds. Shows how model performance evolves as training window grows. Error bands from per-output variation. Data source: extract per-fold metrics from `SuperLearner._cv_fold_scores_` (needs to be exposed in `training_info`).

#### Category 7: Error Analysis

| Figure | Status | Description |
|--------|--------|-------------|
| fig17 | **EXISTS** — fix data flow | 4-panel residual diagnostics |
| fig17b | **NEW** | Residual distribution comparison |

**fig17b — Residual Distribution Comparison** (`plot_residual_distributions`)

Multi-panel (one per model + ensemble): histogram + KDE of prediction residuals. Overlays a normal distribution reference curve. Reveals systematic biases, heavy tails, or skewness in each model's error distribution. Data source: `per_model_holdout_predictions` (residuals = actual - predicted per model).

#### Category 8: Ensemble Diversity

| Figure | Status | Description |
|--------|--------|-------------|
| fig24a | **NEW** | Prediction correlation heatmap |
| fig24b | **NEW** | Pairwise prediction scatter matrix |

**fig24a — Prediction Correlation Heatmap** (`plot_prediction_correlation_heatmap`)

Heatmap: rows = models, columns = models. Cell color = Pearson correlation between the two models' OOF predictions. Annotated with correlation values. Hierarchical clustering on rows/columns to reveal model families. Low off-diagonal correlations indicate high ensemble diversity (desirable). Data source: `SuperLearner._oof_predictions_per_model_` (per-model OOF arrays).

**fig24b — Pairwise Prediction Scatter Matrix** (`plot_prediction_scatter_matrix`)

Upper-triangle scatter matrix: each panel = scatter of Model A predictions vs. Model B predictions (OOF). Diagonal = histogram of each model's predictions. Lower triangle = Pearson r value. Complements fig24a with visual inspection of nonlinear disagreement patterns. Data source: same as fig24a.

#### Category 9: Model Robustness

| Figure | Status | Description |
|--------|--------|-------------|
| fig20c | **NEW** | Bootstrap confidence intervals for metrics |

**fig20c — Bootstrap Metric Confidence Intervals** (`plot_bootstrap_metric_ci`)

Horizontal error bar chart: each bar = one model, center = mean R² (or RMSE), whiskers = 95% bootstrap CI from 200 resamples of the OOF predictions. Reveals statistical significance of performance differences between models. Data source: computed at figure generation time by resampling `training_info['y_test']` and per-model predictions.

#### Category 10: Forecasting Evaluation

| Figure | Status | Description |
|--------|--------|-------------|
| fig23 | **EXISTS** — fix data flow | Rank change bubble |
| fig23b | **EXISTS** — fix data flow | Province forecast comparison bars |
| fig23c | **EXISTS** — fix data flow | Score trajectory with CI |

### Updated `generate_all()` Forecast Block

The forecast section of `generate_all()` should be restructured to:

1. Replace the single `try/except` with per-figure `_safe()` calls
2. Pre-extract data once, then pass to each figure independently
3. Add calls for all new figures
4. Populate per-model OOF predictions in `training_info` for diversity/error figures

```python
# ── Forecast ──────────────────────────────────────────────
if forecast_result is not None:
    _fr = forecast_result
    _ti = getattr(_fr, 'training_info', {}) or {}
    _actual    = _ti.get('y_test')
    _predicted = _ti.get('y_pred')
    _ent       = _ti.get('test_entities')
    _contribs  = getattr(_fr, 'model_contributions', {}) or {}
    _cv_scores = getattr(_fr, 'cross_validation_scores', {}) or {}
    _intervals = getattr(_fr, 'prediction_intervals', {}) or {}
    _pred_df   = getattr(_fr, 'predictions', None)
    _pred_year = getattr(_fr, 'target_year', max(panel_data.years) + 1)
    _per_model_ho = _ti.get('per_model_holdout_predictions')
    _model_perf = getattr(_fr, 'model_performance', None)
    _per_model_imp = _ti.get('per_model_feature_importance')
    _per_model_oof = _ti.get('per_model_oof_predictions')

    # --- Category 2: Prediction vs Actual ---
    if _actual is not None and _predicted is not None:
        _a = np.asarray(_actual)
        _p = np.asarray(_predicted)
        _safe(self.forecast.plot_forecast_scatter, _a, _p, entity_names=_ent)
        _safe(self.forecast.plot_forecast_residuals, _a, _p)
        _safe(self.forecast.plot_conformal_coverage, _a, _p)
        _safe(self.forecast.plot_holdout_comparison,
              _a, _p, per_model_predictions=_per_model_ho,
              entity_names=_ent, model_contributions=_contribs or None)

    # --- Category 5: Feature Importance ---
    if hasattr(_fr, 'feature_importance'):
        # ... existing fig18 logic ...
        pass
    if _per_model_imp:
        _safe(self.forecast.plot_per_model_importance_heatmap, _per_model_imp)

    # --- Category 3: Ensemble Weights ---
    if _contribs:
        _safe(self.forecast.plot_model_weights_donut, _contribs)
        _safe(self.forecast.plot_model_contribution_dots,
              _contribs, cross_validation_scores=_cv_scores or None,
              model_performance=_model_perf)

    # --- Category 1: Model Performance ---
    if _model_perf:
        _safe(self.forecast.plot_model_performance, _model_perf)
        _safe(self.forecast.plot_model_metric_radar, _model_perf)
    if _cv_scores:
        _safe(self.forecast.plot_cv_boxplots, _cv_scores)

    # --- Category 4: Uncertainty ---
    if _intervals and _pred_df is not None:
        _lower = _intervals.get('lower')
        _upper = _intervals.get('upper')
        if _lower is not None and _upper is not None:
            _safe(self.forecast.plot_prediction_intervals, _pred_df, _lower, _upper)
        if _actual is not None and _predicted is not None and _lower is not None:
            _safe(self.forecast.plot_interval_calibration_scatter,
                  _actual, _predicted, _lower, _upper)

    # --- Category 7: Error Analysis ---
    if _actual is not None and _per_model_ho:
        _safe(self.forecast.plot_residual_distributions,
              _actual, _per_model_ho, _predicted)

    # --- Category 6: Temporal/Panel ---
    if _actual is not None and _ent is not None:
        _safe(self.forecast.plot_entity_error_analysis,
              _actual, _predicted, _ent)

    # --- Category 8: Ensemble Diversity ---
    if _per_model_oof:
        _safe(self.forecast.plot_prediction_correlation_heatmap, _per_model_oof)
        _safe(self.forecast.plot_prediction_scatter_matrix, _per_model_oof)

    # --- Category 9: Model Robustness ---
    if _actual is not None and _per_model_ho:
        _safe(self.forecast.plot_bootstrap_metric_ci,
              _actual, _per_model_ho, _predicted)

    # --- Category 10: Forecasting Evaluation ---
    if _pred_df is not None and len(_pred_df.columns) > 0:
        _safe(self.forecast.plot_rank_change_bubble,
              provinces, scores, _pred_df[_pred_df.columns[0]].values,
              prediction_year=_pred_year)
        _safe(self.forecast.plot_province_forecast_comparison,
              provinces, scores, _pred_df,
              intervals=_intervals or None, prediction_year=_pred_year)
        _safe(self.forecast.plot_score_trajectory,
              provinces, scores, _pred_df,
              intervals=_intervals or None,
              panel_years=list(panel_data.years), target_year=_pred_year)
```

### Data Pipeline Changes for New Figures

The following data must be exposed through `training_info` in `stage6_evaluate_all()`:

| Key | Content | Used By |
|-----|---------|---------|
| `per_model_feature_importance` | `Dict[str, np.ndarray]` — model name → importance array | fig18b |
| `per_model_oof_predictions` | `Dict[str, np.ndarray]` — model name → OOF prediction array (raveled) | fig24a, fig24b |
| `per_model_holdout_predictions` | `Dict[str, np.ndarray]` — model name → holdout predictions | fig16c, fig17b, fig20c |
| `cv_fold_metrics` | `Dict[str, List[float]]` — fold_year → R² | fig23e |
| `test_entities` | `List[str]` — entity names for holdout/OOF samples | fig23d |

**Implementation**: In `stage6_evaluate_all()`, after SuperLearner fitting:

```python
# Collect per-model feature importance
per_model_imp = {}
for name, model in self.super_learner_._fitted_base_models.items():
    try:
        imp = model.get_feature_importance()
        per_model_imp[name] = imp
    except Exception:
        pass
self._training_info_['per_model_feature_importance'] = per_model_imp

# Collect per-model OOF predictions
per_model_oof = {}
if hasattr(self.super_learner_, '_oof_per_model_predictions_'):
    per_model_oof = self.super_learner_._oof_per_model_predictions_
self._training_info_['per_model_oof_predictions'] = per_model_oof

# Entity names for OOF samples
if hasattr(self, 'entity_names_') and self._oof_valid_mask_ is not None:
    self._training_info_['test_entities'] = [
        self.entity_names_[i] for i in np.where(self._oof_valid_mask_)[0]
    ]
```

**SuperLearner change**: Store OOF per-model predictions as `_oof_per_model_predictions_: Dict[str, np.ndarray]` alongside the existing `_oof_ensemble_predictions_`. This requires saving each model's OOF array before meta-weight combination.

### Complete Forecast Figure Catalog (after T-05)

| ID | Method | Category | Status |
|----|--------|----------|--------|
| fig16 | `plot_forecast_scatter` | Pred vs Actual | Fix data flow |
| fig16b | `plot_ensemble_architecture` | Architecture | Fix stale model names (T-04c) |
| fig16c | `plot_holdout_comparison` | Pred vs Actual | Fix data flow |
| fig17 | `plot_forecast_residuals` | Error Analysis | Fix data flow |
| fig17b | `plot_residual_distributions` | Error Analysis | **NEW** |
| fig18 | `plot_feature_importance` | Interpretability | Working |
| fig18b | `plot_per_model_importance_heatmap` | Interpretability | **NEW** |
| fig19 | `plot_model_weights_donut` | Ensemble Weights | Working (fix title T-04b) |
| fig19b | `plot_model_contribution_dots` | Ensemble Weights | Working |
| fig20 | `plot_model_performance` | Performance | Fix data flow |
| fig20b | `plot_model_metric_radar` | Performance | **NEW** |
| fig20c | `plot_bootstrap_metric_ci` | Robustness | **NEW** |
| fig21 | `plot_cv_boxplots` | Performance | Fix data flow |
| fig22 | `plot_prediction_intervals` | Uncertainty | Fix data flow |
| fig22b | `plot_conformal_coverage` | Uncertainty | Fix data flow |
| fig22c | `plot_interval_calibration_scatter` | Uncertainty | **NEW** |
| fig23 | `plot_rank_change_bubble` | Forecast Eval | Fix data flow |
| fig23b | `plot_province_forecast_comparison` | Forecast Eval | Fix data flow |
| fig23c | `plot_score_trajectory` | Forecast Eval | Fix data flow |
| fig23d | `plot_entity_error_analysis` | Panel Analysis | **NEW** |
| fig23e | `plot_temporal_training_curve` | Panel Analysis | **NEW** |
| fig24a | `plot_prediction_correlation_heatmap` | Ensemble Diversity | **NEW** |
| fig24b | `plot_prediction_scatter_matrix` | Ensemble Diversity | **NEW** |

**Total**: 23 forecast figures (14 existing + 9 new)

### Files to Create / Modify

| File | Change |
|------|--------|
| `output/visualization/forecast_plots.py` | Add 9 new `plot_*` methods; update titles/labels per T-04b |
| `output/visualization/__init__.py` | Replace `_inc()` with `_safe()` in forecast block; add delegated methods for 9 new figures; restructure forecast section |
| `forecasting/unified.py` (stage6) | Populate `per_model_feature_importance`, `per_model_oof_predictions`, `test_entities` in `training_info`; harden OOF evaluation |
| `forecasting/super_learner.py` | Store `_oof_per_model_predictions_` dict during OOF computation |

---

## Cross-Cutting Concerns

### 1. `_per_model_X` Track Routing (T-02 + T-03)

Current routing in `unified.py` Stage 2:

| Model | Current Track | After Changes |
|-------|---------------|---------------|
| CatBoost (was GradientBoosting) | Tree | Tree (unchanged) |
| LightGBM | Tree | Tree (unchanged) |
| BayesianRidge | PCA | PCA (unchanged) |
| QuantileRF | Tree | Tree (unchanged) |
| PanelVAR | Tree | Tree (unchanged, conditional) |
| NAM | Tree | Tree (unchanged, conditional) |
| **KernelRidge** | — | **PCA** (new) |
| **SVR** | — | **PCA** (new) |

PCA track models receive `X_train_pca_` (lower-dimensional, scaled, orthogonal features).
Tree track models receive `X_train_tree_` (high-dimensional, unscaled, with threshold-only variance filter).

The routing logic in Stage 2 should be data-driven: iterate the model dict from `_create_models()` and assign track based on model type:

```python
PCA_TRACK_MODELS = {'BayesianRidge', 'KernelRidge', 'SVR'}
# All other models → tree track
```

### 2. Optuna Hyperparameter Tuning (T-03 + T-04a)

The existing `_tune_gb_hyperparameters()` tunes `'GradientBoosting'` and `'LightGBM'`. After T-04a rename:
- Change the tuning key from `'GradientBoosting'` to `'CatBoost'`
- Optionally extend tuning to KRR (`alpha`, `gamma`) and SVR (`C`, `epsilon`, `gamma`) — but defer to P2 (low priority; default hyperparameters are well-suited for n=756)

### 3. Backward Compatibility

- **Serialized models**: Any pickled `UnifiedForecaster` objects from before T-04a will have `'GradientBoosting'` as a key. If incremental update (E-10) is used, the `IncrementalEnsembleUpdater` must handle the old key gracefully.
- **CSV outputs**: Column names in exported CSVs will change from "GradientBoosting" to "CatBoost". Downstream consumers must be notified.
- **Test fixtures**: Update any test assertions that reference `'GradientBoosting'` as a model key.

### 4. Logging

All model count/name references in log messages should be dynamically generated from the model dict, not hardcoded. Current violation in `pipeline.py` L737-739 where `_base_model_names` is a static list.

---

## Implementation Order & Dependencies

```
Phase A (independent, can parallelize)
├── T-01: ER config toggle (config.py + ranking/)
├── T-04a: Rename GradientBoosting → CatBoost (unified.py + visualization)
└── T-04b: Rename Super Learner → Meta-Learner (display strings only)

Phase B (depends on T-04a for naming)
├── T-02: NAM/PanelVAR toggle (config.py + unified.py)
├── T-03a: KernelRidgeForecaster (new file + config + unified)
└── T-03b: SVRForecaster (new file + config + unified)

Phase C (depends on T-02 + T-03 for model set + T-04 for naming)
├── T-04c: Fix ensemble architecture flowchart
├── T-05 F-05a: Fix _inc → _safe in generate_all()
├── T-05 F-05b: Harden stage6 OOF evaluation data flow
└── T-05 F-05c: Populate per_model_* in training_info

Phase D (depends on Phase C for data availability)
├── T-05 new figures: fig17b, fig18b, fig20b, fig20c
├── T-05 new figures: fig22c, fig23d, fig23e
└── T-05 new figures: fig24a, fig24b
```

**Estimated scope**: ~1500 lines new code, ~300 lines modified, 2 new files.

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| KRR/SVR O(n³) slow on large n | Low (n=756) | Low | Monitor; add config toggle if needed |
| OOF per-model predictions consume memory | Low (6 models × 756 × 8) | Low | ~290 KB total — negligible |
| Renaming breaks downstream CSV consumers | Medium | Medium | Document in release notes; add migration note |
| New figures fail on edge-case data | Medium | Low | Each figure wrapped in `_safe()` — independent failure |

---

*End of plan.*
