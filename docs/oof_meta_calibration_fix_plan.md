# OOF Generation, Meta-Learner Stacking & Uncertainty Calibration — Audit, Fix & Enhancement Plan

**Date**: 2026-03-16
**Updated**: 2026-03-16 (Phase II complete)
**Scope**: `forecasting/super_learner.py`, `forecasting/conformal.py`, `forecasting/unified.py`, `config.py`
**Data context**: 14 target years, 63 provinces × 29 criteria (~756 rows after feature engineering), heavy missingness, predict t+1

**Status**: ✅ Phase I complete (F-01..F-07) | ✅ Phase II complete (E-01..E-06) | 358 tests pass

---

## 0. Executive Summary

Three subsystems contain **7 correctness bugs** and **6 accuracy-limiting weaknesses** that compound in the small-panel regime. The most damaging issues are:

1. The meta-learner is trained on ≤315 of its 756 available training rows (42% coverage — the rest are never in a validation fold).
2. Per-output optimal weights are collapsed to a single scalar vector by averaging, violating the Super Learner oracle inequality for multi-output targets.
3. Conformal calibration uses a joint "all-8-outputs valid" row mask, shrinking per-component calibration sets by 15–40% unnecessarily.
4. Dirichlet stacking bootstrap ignores panel structure, producing underestimated weight uncertainty.
5. QRF Bonferroni correction at α/8 = 0.00625 requests the 0.3th percentile from leaves with ~15 samples — effectively the observed minimum, making lower bounds outlier-anchored.

The plan is structured as **Phase I (correctness fixes — 7 items)** then **Phase II (SOTA accuracy enhancements — 6 items)**. All enhancements are designed for the small-panel regime: ≤14 years, ≤756 rows, heavy missingness.

---

## 1. Codebase Architecture Recap

```
SuperLearner.fit()
│
├── Stage A: Walk-forward OOF generation
│   ├── _WalkForwardYearlySplit(min_train_years=8, max_folds=5)
│   │   └── val years 2020–2024 only → 315/756 rows have OOF predictions
│   ├── Per-model copy.deepcopy + _fit_model + _predict_model
│   └── oof_predictions: (756, n_models * 8)  # NaN for years 2012–2019
│
├── Stage B: _fit_meta_learner
│   ├── For each of 8 output columns → NNLS → full_coefs[out_col]
│   ├── >>> ISSUE: normed_coefs averaged → scalar weights (not per-output) <<<
│   └── _meta_weights: Dict[model_name, float]  # ONE weight for all outputs
│
├── Stage C: Ensemble OOF cache
│   ├── _oof_ensemble_predictions_: (756, 8)  # meta-weighted, NaN for pre-2020
│   └── >>> ISSUE: _oof_valid_mask_ requires ALL 8 cols non-NaN <<<
│
└── Stage D: E-02 secondary conformal OOF residuals
    └── years 2015–2019 residuals, but using weights from post-2019 only

UnifiedForecaster.stage5_compute_intervals()
├── QRF path: alpha_bonferroni = 0.05/8 = 0.00625
│   └── >>> ISSUE: lower_q = 0.003125 → extreme tail, outlier-anchored <<<
└── Conformal path:
    └── >>> ISSUE: uses _oof_valid_mask_ (joint) per component <<<
```

---

## 2. Phase I — Correctness Fixes

### F-01 — Per-Component Valid Mask for Conformal Calibration
**File**: `unified.py` lines 2060–2091; `super_learner.py` line 717
**Severity**: HIGH — reduces calibration set size by 15–40% for components with sparse OOF
**Root cause**: `_oof_valid_mask_` is `~np.isnan(oof_ensemble).any(axis=1)`, requiring all 8 outputs to be non-NaN. A single sparse criterion (e.g., one model failing for criterion 7 in 2 folds) silently removes those rows from ALL 8 calibration sets.

**Fix**:

In `super_learner.py`, after computing `_oof_ensemble_predictions_`, add per-column masks:
```python
# Current (line 717):
self._oof_valid_mask_ = ~np.isnan(oof_ensemble).any(axis=1)

# Replace with per-column masks:
self._oof_valid_mask_ = ~np.isnan(oof_ensemble).any(axis=1)            # keep for backward compat
self._oof_valid_mask_per_col_ = ~np.isnan(oof_ensemble)                # (n_samples, n_outputs)
```

In `unified.py` stage5 conformal loop (line 2061):
```python
# Current:
valid      = sl._oof_valid_mask_
oof_pred_d = sl._oof_ensemble_predictions_[valid, d]

# Replace with:
_pmask = getattr(sl, '_oof_valid_mask_per_col_', None)
if _pmask is not None:
    valid = _pmask[:, d]                                               # per-component mask
else:
    valid = sl._oof_valid_mask_                                        # backward compat
oof_pred_d = sl._oof_ensemble_predictions_[valid, d]
```

**Expected gain**: 10–40% more calibration residuals per component with sparse base models. Coverage of conformal predictor improves from potentially 200 → 280 calibration samples.

---

### F-02 — Per-Output Meta-Weight Vectors (Multi-Output Stacking)
**File**: `super_learner.py` lines 1114–1132
**Severity**: HIGH — violates Super Learner oracle inequality for multi-output regression
**Root cause**: Each of the 8 output columns gets its own NNLS solution `full_coefs[out_col]`, but all 8 solutions are then averaged `avg_coefs = np.mean(normed_coefs, axis=0)` into a single weight vector applied uniformly to all outputs. This is suboptimal when base models have different relative strengths across criteria.

**Fix**: Store and apply per-output weight vectors.

```python
# In _fit_meta_learner, replace the averaging block:

# Current (lines 1114–1132):
normed_coefs = [c/s ... for c in all_coefs ...]
avg_coefs = np.mean(normed_coefs, axis=0)
...
self._meta_weights = dict(zip(self.base_models.keys(), avg_coefs))

# Replace with:
normed_coefs = []
for c in all_coefs:
    s = float(np.sum(c))
    normed_coefs.append(c / s if s > 1e-15 else np.ones(len(c)) / len(c))
normed_coefs = np.array(normed_coefs)            # (n_outputs, n_models)

# Store per-output weights
model_names = list(self.base_models.keys())
self._meta_weights_per_output_: np.ndarray = normed_coefs    # (n_outputs, n_models)
self._meta_weights_col_names_: list = model_names

# Backward-compat scalar weights = mean across outputs (used for conformal weighting)
avg_coefs = normed_coefs.mean(axis=0)
coef_sum  = float(avg_coefs.sum())
if coef_sum > 1e-15: avg_coefs /= coef_sum
else: avg_coefs = np.ones(len(model_names)) / len(model_names)
self._meta_weights = dict(zip(model_names, avg_coefs))
```

Update the ensemble OOF cache and prediction to apply per-output weights:

```python
# In _cache_ensemble_oof() and in predict():
# For output column i_out, use self._meta_weights_per_output_[i_out, m_idx]
# instead of self._meta_weights[name]
```

In `predict()`, add:
```python
def _get_weight(self, model_name: str, out_col: int) -> float:
    """Returns per-output weight if available, else scalar fallback."""
    if hasattr(self, '_meta_weights_per_output_'):
        idx = self._meta_weights_col_names_.index(model_name)
        return float(self._meta_weights_per_output_[out_col, idx])
    return self._meta_weights.get(model_name, 0.0)
```

**Expected gain**: 3–8% RMSE reduction on criteria where one model dominates. Most impact on criteria that are mechanistically different (e.g., infrastructure vs. governance).

---

### F-03 — Entity-Block Bootstrap for Dirichlet Weight Uncertainty
**File**: `super_learner.py` lines 1246–1284
**Severity**: MEDIUM — weight standard deviations are underestimated; affects model comparison and pruning decisions
**Root cause**: Bootstrap resamples rows as i.i.d. Panel rows `(entity_t, year_t)` are correlated along both entity and time axes. Naive bootstrap underestimates variance by ignoring dependence.

**Fix**: Cluster bootstrap by entity (resample whole entity time-series):
```python
# In _compute_dirichlet_weight_std, replace lines 1249–1251:

# Current:
idx = rng.randint(0, n_samples, size=n_samples)
b_X, b_y = oof_X[idx], oof_y[idx]

# Replace with entity-block bootstrap:
if entity_indices is not None:
    unique_entities = np.unique(entity_indices)
    sampled_entities = rng.choice(unique_entities,
                                  size=len(unique_entities),
                                  replace=True)
    idx = np.concatenate([
        np.where(entity_indices == e)[0]
        for e in sampled_entities
    ])
else:
    # Fallback: i.i.d. if no entity info
    idx = rng.randint(0, n_samples, size=n_samples)
b_X, b_y = oof_X[idx], oof_y[idx]
```

Propagate `entity_indices` to `_compute_dirichlet_weight_std`:
```python
# Add entity_indices parameter:
def _compute_dirichlet_weight_std(self, oof_X, oof_y, n_boot=200,
                                   entity_indices=None) -> Dict[str, float]:
```

**Expected gain**: Correct 95% confidence intervals on meta-weights. Enables statistically valid model pruning (remove models whose weight CI includes zero).

---

### F-04 — Analytical Gradient for Dirichlet Stacking Optimizer
**File**: `super_learner.py` lines 1192–1215
**Severity**: MEDIUM — numerical gradient requires 2K function evals per iteration; bootstrap with K=6, n_boot=200, max_iter=200 ≈ 9.6M operations
**Root cause**: `minimize(..., method='L-BFGS-B')` without `jac=` triggers numerical differentiation.

**Fix**: Provide analytical gradient:
```python
def _neg_log_score_and_grad(logits):
    w = _sfx(logits)                                  # (K,)
    log_w = np.log(w + 1e-30)

    diff2   = (y[:, np.newaxis] - preds) ** 2        # (n, K)
    log_pk  = -inv_2sigma2 * diff2 - log_sigma        # (n, K)
    lse_in  = log_w[np.newaxis, :] + log_pk           # (n, K)

    lse_max = lse_in.max(axis=1, keepdims=True)
    exp_in  = np.exp(lse_in - lse_max)               # (n, K)
    mix_sum = exp_in.sum(axis=1, keepdims=True)       # (n, 1)
    log_mix = lse_max[:, 0] + np.log(mix_sum[:, 0] + 1e-30)  # (n,)

    # Gradient: d(-sum log_mix) / d(logits)
    # p_k_given_n = exp_in[:, k] / mix_sum[:, 0]      (n, K)
    # d(-log L)/d(logit_k) = Σ_n [w_k - p(k|n)]
    #   where p(k|n) = exp(log_w_k + log_pk_n) / mix_n
    responsibility = exp_in / mix_sum                 # (n, K)  = p(k|n)

    # dL/dw_k = Σ_n [1/w_k * p(k|n)]  →  chain rule through softmax
    # dL/dlogit_k = Σ_j (δ_kj - w_j) * w_j * [Σ_n 1/w_j * p(j|n)]
    # Simplified:
    # dNLL/d(logit_k) = Σ_n [w_k - p(k|n)]   (score function)
    grad_w = w - responsibility.mean(axis=0)          # (K,)  ∂NLL/∂w_k

    # Jacobian: ∂w_k/∂logit_j = w_k * (δ_kj - w_j)
    # ∂NLL/∂logit_j = Σ_k (∂NLL/∂w_k) * w_k * (δ_kj - w_j)
    #               = w_j * grad_w[j] - w_j * dot(w, grad_w)
    grad_logit = n_samples * (w * grad_w - w * np.dot(w, grad_w))

    return float(-np.sum(log_mix)), grad_logit.astype(np.float64)

res = minimize(_neg_log_score_and_grad, x0,
               method='L-BFGS-B',
               jac=True,
               options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
```

**Expected gain**: 5–8× speedup in Dirichlet stacking and 200-bootstrap. Enables higher `n_boot` (500) for tighter CI.

---

### F-05 — Fix `cv_min_train_years` Discrepancy
**File**: `config.py` line 308; `unified.py` line 620
**Severity**: MEDIUM — silent behavioral difference based on instantiation path

```python
# config.py — keep at 8 (correct for 14-year dataset)
cv_min_train_years: int = 8

# unified.py UnifiedForecaster.__init__ — align default:
# Current: cv_min_train_years: int = 7
# Fix:
cv_min_train_years: int = 8
```

Add a guard in `UnifiedForecaster.__init__` to log when config overrides:
```python
if config is not None and hasattr(config, 'cv_min_train_years'):
    if self.cv_min_train_years != config.cv_min_train_years:
        logger.info(f"cv_min_train_years: class default {self.cv_min_train_years} "
                    f"overridden by config {config.cv_min_train_years}")
    self.cv_min_train_years = config.cv_min_train_years
```

---

### F-06 — Adaptive Bonferroni for QRF: Replace Hard Division by n_components
**File**: `unified.py` lines 1944–1951
**Severity**: MEDIUM — `lower_q = 0.003125` is statistically unresolvable with ~15-sample leaves
**Root cause**: Bonferroni assumes 8 stochastically independent components. MCDM criteria are correlated (governance, infrastructure, etc.), so Bonferroni overcorrects, inflating the quantile level to near-extreme values.

**Fix**: Replace Bonferroni with Benjamini-Yekutieli (BY) correction (valid under arbitrary dependence, less conservative):

```python
# Current (lines 1948–1950):
alpha_bonferroni = self.conformal_alpha / n_components
lower_q = alpha_bonferroni / 2.0
upper_q = 1.0 - alpha_bonferroni / 2.0

# Replace with Šidák correction (less conservative for correlated tests):
# P(at least one failure) ≤ α  →  per-component α* = 1 - (1-α)^(1/D)
# For α=0.05, D=8: α* = 0.00641 (vs Bonferroni 0.00625) — minor improvement

# Better: use estimated effective number of independent components
# via PCA explained variance of OOF residuals
def _effective_n_components(oof_residuals_matrix: np.ndarray, threshold=0.95) -> int:
    """Estimate effective independent dimensions via PCA variance explained."""
    from sklearn.decomposition import PCA
    valid_rows = ~np.isnan(oof_residuals_matrix).any(axis=1)
    if valid_rows.sum() < 3:
        return oof_residuals_matrix.shape[1]
    pca = PCA()
    pca.fit(oof_residuals_matrix[valid_rows])
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_eff = int(np.searchsorted(cumvar, threshold)) + 1
    return min(n_eff, oof_residuals_matrix.shape[1])

# Apply in stage5:
_oof_res_matrix = self._get_oof_residual_matrix()  # (n_cal, n_outputs)
n_eff = _effective_n_components(_oof_res_matrix)
alpha_adj = self.conformal_alpha / n_eff            # effective Bonferroni
lower_q = alpha_adj / 2.0
upper_q = 1.0 - alpha_adj / 2.0

# Safety floor: never go below the 1st percentile
lower_q = max(lower_q, 0.01)
upper_q = min(upper_q, 0.99)
```

**Expected gain**: `lower_q` rises from 0.003125 to ~0.01–0.025 depending on criterion correlations, producing QRF intervals that are well-supported by the training data rather than anchored to extreme outliers.

---

### F-07 — Minimum Calibration Samples: Raise from 3 to 30
**File**: `conformal.py` line 303; `ConformalPredictor.calibrate_residuals`
**Severity**: LOW-MEDIUM — fewer than 3 samples produces max-residual upper bound (infinitely wide intervals in practice)

```python
# Current:
if len(residuals) < 3:
    raise ValueError(...)

# Replace with:
_MIN_CAL_SAMPLES = 30
if len(residuals) < _MIN_CAL_SAMPLES:
    logger.warning(
        f"ConformalPredictor: only {len(residuals)} calibration residuals "
        f"(< {_MIN_CAL_SAMPLES} recommended). Intervals may be unreliable."
    )
if len(residuals) < 3:
    raise ValueError(...)
```

Also: when `n_cal < 30` and `_calibrate_cv_plus` is the source, fall through to a wider expanded calibration sweep (E-01 in Phase II).

---

## 3. Phase II — SOTA Accuracy Enhancements

### E-01 — Repeated Walk-Forward CV for OOF Coverage Expansion
**Target**: Increase OOF coverage from 315 (42%) to >650 (86%) rows
**Background**: With `min_train_years=8`, only years 2020–2024 receive OOF predictions. Years 2012–2019 (504 rows) serve only as training data and never contribute to meta-learner calibration or conformal residuals. This means the meta-learner sees a biased calibration set: only the most-recent 5 years of model performance.

**Approach**: Repeated Walk-Forward with multiple starting offsets (RWFCV).

```python
class _RepeatedWalkForwardSplit:
    """
    Generates multiple walk-forward schedules by varying the training start year.

    Strategy:
    - Pass 1: min_train=8 (production schedule, years 2020-2024 are val)
    - Pass 2: min_train=6 (years 2018-2024 are val, includes 2018-2019)
    - Pass 3: min_train=4 (years 2016-2024 are val, includes 2016-2017)

    Pass 1 OOF predictions are used for BOTH meta-learner training AND conformal.
    Pass 2-3 OOF predictions augment conformal calibration ONLY (not meta-weights),
    to avoid leakage from early low-data folds biasing meta-learner toward overfitted
    early-year base models.

    The key insight: base models trained on 4-6 years may overfit differently than
    those trained on 8+ years. Their OOF residuals reflect the calibration-period
    model performance, which is appropriate for conformal interval widths.
    """
    def __init__(self, primary_min_train: int = 8,
                 extra_passes: List[int] = (6, 4),
                 max_folds: int = 5):
        self.primary_min_train = primary_min_train
        self.extra_passes = extra_passes
        self.max_folds = max_folds

    def split_primary(self, X, year_labels):
        """Primary schedule — used for meta-weights."""
        return _WalkForwardYearlySplit(
            min_train_years=self.primary_min_train,
            max_folds=self.max_folds
        ).split(X, year_labels)

    def split_augment(self, X, year_labels):
        """Augmentation schedules — for conformal residuals only."""
        for min_tr in self.extra_passes:
            yield from _WalkForwardYearlySplit(
                min_train_years=min_tr,
                max_folds=self.max_folds
            ).split(X, year_labels)
```

**Integration in `SuperLearner.fit()`**:
- Stage A uses `split_primary()` as today (no change to meta-weight quality)
- After Stage C (ensemble OOF cache), run a new Stage C2 using `split_augment()`:
  - For each fold in each augmentation pass, fit copies of all base models
  - Compute ensemble OOF predictions using **frozen** `_meta_weights` from Stage B
  - Store augmented residuals in `_oof_conformal_residuals_extended_`

**Config fields to add**:
```python
cv_oof_augment_passes: List[int] = field(default_factory=lambda: [6, 4])
# Min training years for each augmentation pass. Empty list = no augmentation.
```

**Expected gain**: Conformal calibration set grows from ~315 to ~630-750 rows (depending on missingness). More reliable quantile estimation, especially for extreme coverage levels.

---

### E-02 — Conformalized Quantile Regression (CQR) with QRF
**Target**: Replace fixed ±q̂ conformal intervals with CQR adaptive intervals
**Background**: Current conformal path uses `q̂ = quantile(|y - ŷ|)` — a single scalar applied uniformly across all test points. This ignores heteroscedasticity: provinces with many missing features have wider predictive uncertainty than provinces with complete data. CQR (Angelopoulos et al. 2021) conditions interval width on the QRF prediction interval, preserving marginal coverage while being locally adaptive.

**Method**:
```
CQR interval: [Q̂_α/2(x) - q̂_CQR, Q̂_{1-α/2}(x) + q̂_CQR]
where:
  Q̂_α/2, Q̂_{1-α/2} = QRF lower/upper quantile predictions (from calibration set)
  q̂_CQR = (1-α)(1+1/n_cal) quantile of conformity scores
  conformity_i = max(q̂_{α/2}(x_i) - y_i,  y_i - q̂_{1-α/2}(x_i))
```

The conformity score `conformity_i` is positive only when `y_i` falls outside `[Q̂_{α/2}(x_i), Q̂_{1-α/2}(x_i)]`, giving a continuous measure of miscoverage. Unlike split conformal, CQR adapts width to local QRF uncertainty.

**Implementation**:
```python
class CQRConformalPredictor:
    """
    Conformalized Quantile Regression (Romano et al. 2019).
    Uses QRF lower/upper predictions as base, calibrates residual adjustment q̂.
    """
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self._q_cqr: Optional[float] = None

    def calibrate(self,
                  qrf_lower_cal: np.ndarray,   # (n_cal,) QRF lower on cal set
                  qrf_upper_cal: np.ndarray,   # (n_cal,) QRF upper on cal set
                  y_cal: np.ndarray):
        scores = np.maximum(qrf_lower_cal - y_cal,
                            y_cal - qrf_upper_cal)    # (n_cal,)
        n_cal = len(scores)
        q_level = np.ceil((1 - self.alpha) * (n_cal + 1)) / n_cal
        q_level = min(q_level, 1.0)
        self._q_cqr = float(np.quantile(scores, q_level))

    def predict_intervals(self,
                          qrf_lower_test: np.ndarray,
                          qrf_upper_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._q_cqr is None:
            raise RuntimeError("calibrate() must be called first")
        return qrf_lower_test - self._q_cqr, qrf_upper_test + self._q_cqr
```

**Integration in `stage5_compute_intervals()`**:
- After QRF predicts `lower_arr, upper_arr` on prediction set, also predict on OOF calibration set
- Compute conformity scores from OOF (y_cal_d, lower_oof_d, upper_oof_d)
- Calibrate `CQRConformalPredictor` per component
- Final intervals: `predict_intervals(lower_test_d, upper_test_d)`

**Config fields to add**:
```python
use_cqr_calibration: bool = True
# When True and QRF is available, apply CQR post-calibration to QRF intervals.
# Falls back to current split-conformal if QRF OOF unavailable.
```

**Expected gain**: 15–25% tighter prediction intervals for provinces with complete data, 10–15% wider for provinces with heavy missingness — improving interval sharpness while preserving coverage guarantees.

---

### E-03 — Mondrian Conformal Stratified by Missingness Rate
**Target**: Provide guaranteed coverage separately for high-missingness and low-missingness provinces
**Background**: With heavy missing data, provinces can be split into strata based on their missingness rate. Current conformal gives marginal coverage over all provinces jointly, but this subsumes structural differences: a province with 60% missing features has systematically larger prediction errors than one with 5% missing. Mondrian CP (Venn-ABERS style) gives per-stratum coverage guarantees.

Note: A basic `_stratified` flag already exists in `ConformalPredictor`. This enhancement properly stratifies by missingness rate with automatic stratum boundaries.

**Method**:
```python
class MissignessStratifiedConformal:
    """
    Mondrian conformal predictor stratified by missingness rate.

    Strata are defined by quantile boundaries of missingness rate over
    the calibration set. Each stratum maintains an independent calibration
    set and produces stratum-specific quantiles.

    Guarantees: P(Y ∈ Γ(X) | stratum(X) = s) ≥ 1 - α for each stratum s.

    With n_strata=3 and n_cal≈300:
    - Stratum 0 (0–33% missingness):  ~100 samples  → robust quantile
    - Stratum 1 (33–67% missingness): ~100 samples  → robust quantile
    - Stratum 2 (67–100% missingness):~100 samples  → robust quantile
    """
    def __init__(self, alpha: float = 0.05, n_strata: int = 3):
        self.alpha = alpha
        self.n_strata = n_strata
        self._stratum_quantiles: Dict[int, float] = {}
        self._stratum_boundaries: np.ndarray = np.array([])

    def calibrate(self, residuals: np.ndarray,
                  missingness_rates: np.ndarray):
        # Define strata boundaries via quantiles of missingness distribution
        self._stratum_boundaries = np.quantile(
            missingness_rates,
            np.linspace(0, 1, self.n_strata + 1)
        )
        for s in range(self.n_strata):
            lo = self._stratum_boundaries[s]
            hi = self._stratum_boundaries[s + 1]
            mask = (missingness_rates >= lo) & (
                (missingness_rates < hi) | (s == self.n_strata - 1)
            )
            res_s = np.abs(residuals[mask])
            if len(res_s) < 5:
                # Too few — use global quantile as fallback
                self._stratum_quantiles[s] = float(np.quantile(np.abs(residuals), 1 - self.alpha))
                continue
            n_s = len(res_s)
            q_level = min(np.ceil((1 - self.alpha) * (n_s + 1)) / n_s, 1.0)
            self._stratum_quantiles[s] = float(np.quantile(res_s, q_level))

    def predict_intervals(self, point_preds: np.ndarray,
                          missingness_rates: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.empty_like(point_preds)
        upper = np.empty_like(point_preds)
        for i, mr in enumerate(missingness_rates):
            s = int(np.searchsorted(self._stratum_boundaries[1:], mr, side='right'))
            s = min(s, self.n_strata - 1)
            q_s = self._stratum_quantiles[s]
            lower[i] = point_preds[i] - q_s
            upper[i] = point_preds[i] + q_s
        return lower, upper
```

**Config fields**:
```python
use_mondrian_conformal: bool = False
conformal_n_strata: int = 3
```

**Expected gain**: Provinces with heavy missingness receive wider, calibrated intervals (reducing false-narrow coverage); low-missingness provinces receive tighter intervals. Stratum-conditional coverage guarantees are more meaningful for policy users.

---

### E-04 — True Per-Output NNLS with Joint Multi-Output Regularization
**Target**: Replace averaged scalar weights with per-output vectors + joint regularization
**Background**: F-02 stores per-output weight vectors but the NNLS fit for each output column is independent — it ignores that the optimal weights for criterion 1 and criterion 2 are correlated (base models that are good at one criterion tend to be good at related criteria). Joint multi-output stacking can borrow strength across outputs.

**Method**: Multi-Task NNLS with grouped LASSO regularization.

```python
def _fit_multi_output_nnls(
        self,
        oof_X: np.ndarray,      # (n_samples, n_models * n_outputs)
        oof_y: np.ndarray,      # (n_samples, n_outputs)
        lambda_group: float = 0.01,
) -> np.ndarray:
    """
    Joint NNLS with group-lasso penalty encouraging consistent weights across outputs.

    Objective (for weight matrix W: n_models × n_outputs):
        min_{W≥0} ||Φ * W - Y||²_F + λ * Σ_k ||W_k.||_2
    where Φ[i, k] = oof prediction of model k at sample i (averaged across outputs).

    This is a Group LASSO problem solved by FISTA with proximal operator.

    Simple approximate solution (faster for small K, D):
    1. Fit per-output NNLS (current approach)
    2. Apply soft-sharing: W_shrunk = (1-γ) * W_per_output + γ * W_mean
       where γ = λ / (λ + median(||W_col||)) is adaptive shrinkage
    """
    n_models = len(self.base_models)

    # Step 1: Fit per-output
    W = np.zeros((n_models, oof_y.shape[1]))
    for out_col in range(oof_y.shape[1]):
        model_preds = np.column_stack([
            oof_X[:, m * oof_y.shape[1] + out_col]
            for m in range(n_models)
        ])
        y_col = oof_y[:, out_col]
        valid = ~np.isnan(model_preds).any(axis=1) & ~np.isnan(y_col)
        if valid.sum() < 3:
            W[:, out_col] = 1.0 / n_models
            continue
        coefs, _ = nnls(model_preds[valid], y_col[valid])
        s = coefs.sum()
        W[:, out_col] = coefs / s if s > 1e-15 else np.ones(n_models) / n_models

    # Step 2: Soft-share toward mean (group LASSO proximal step)
    W_mean = W.mean(axis=1, keepdims=True)             # (n_models, 1)
    spread  = np.linalg.norm(W - W_mean, axis=1).mean()
    gamma   = lambda_group / (lambda_group + spread + 1e-12)
    W_shared = (1 - gamma) * W + gamma * W_mean

    # Renormalize each column
    for d in range(W_shared.shape[1]):
        s = W_shared[:, d].sum()
        if s > 1e-15: W_shared[:, d] /= s

    return W_shared   # (n_models, n_outputs)
```

**Config fields**:
```python
meta_group_lasso_lambda: float = 0.01
# Soft-sharing strength across output criteria. 0 = fully independent NNLS.
# 1 = fully shared weights. Default: 0.01 (light regularization).
```

**Expected gain**: Better meta-weight estimation when n_cal is small (<100 per output). Group regularization acts as a prior that similar provinces should have similar base model preferences.

---

### E-05 — Locally Weighted Conformal Prediction (Weighted Split CP)
**Target**: Replace homoscedastic `±q̂` with locally adaptive confidence widths
**Background**: Tibshirani et al. (2019) "Conformal Prediction Under Covariate Shift" introduces weighted conformal prediction where calibration residuals are re-weighted by the density ratio `w(x) = p_test(x) / p_cal(x)`. For the t+1 prediction scenario, the test distribution (future year features) may systematically differ from the calibration years (covariate shift). E-08 already uses MMD² shift detection globally; this enhancement applies local density-ratio weighting to conformal calibration.

**Method**:
```python
class LocallyWeightedConformalPredictor:
    """
    Weighted Split Conformal Predictor (Tibshirani et al. 2019).

    Assigns importance weights to calibration residuals based on similarity
    to the test point. For panel data, similarity is defined by:
    1. Feature-space distance (RBF kernel)
    2. Entity membership weight (same entity = higher weight)

    For MCDM: province X in test year t+1 is most similar to the same province
    in calibration years, then to similar provinces in all years.
    """
    def __init__(self, alpha: float = 0.05,
                 kernel: str = 'rbf',
                 bandwidth: str = 'median_heuristic',
                 entity_weight: float = 2.0):
        self.alpha        = alpha
        self.kernel       = kernel
        self.bandwidth_   = bandwidth
        self.entity_weight = entity_weight
        self._X_cal: Optional[np.ndarray] = None
        self._residuals_cal: Optional[np.ndarray] = None
        self._entity_cal: Optional[np.ndarray] = None

    def calibrate(self, X_cal: np.ndarray,
                  residuals: np.ndarray,
                  entity_indices: Optional[np.ndarray] = None):
        self._X_cal       = X_cal
        self._residuals_cal = np.abs(residuals)
        self._entity_cal  = entity_indices

        # Bandwidth via median heuristic on calibration set
        if self.bandwidth_ == 'median_heuristic':
            from sklearn.metrics.pairwise import euclidean_distances
            D = euclidean_distances(X_cal)
            self.bandwidth_ = float(np.median(D[D > 0]))

    def _compute_weights(self, x_test: np.ndarray,
                         entity_test: Optional[int] = None) -> np.ndarray:
        """Returns unnormalized weights for each calibration point."""
        from sklearn.metrics.pairwise import rbf_kernel
        gamma = 1.0 / (2 * self.bandwidth_ ** 2)
        K = rbf_kernel(x_test.reshape(1, -1), self._X_cal, gamma=gamma)
        w = K.ravel()   # (n_cal,)

        # Boost same-entity calibration points
        if entity_test is not None and self._entity_cal is not None:
            same_entity = (self._entity_cal == entity_test)
            w[same_entity] *= self.entity_weight

        w = w / (w.sum() + 1e-30)   # normalize to probability simplex
        return w

    def predict_interval(self, x_test: np.ndarray,
                         point_pred: float,
                         entity_test: Optional[int] = None
                         ) -> Tuple[float, float]:
        """Predict interval for a single test point."""
        w = self._compute_weights(x_test, entity_test)
        n_cal = len(self._residuals_cal)

        # Weighted quantile with +∞ guard
        sorted_idx = np.argsort(self._residuals_cal)
        sorted_r   = self._residuals_cal[sorted_idx]
        sorted_w   = w[sorted_idx]

        # Augment with w(∞) = 1/(n_cal+1) for +∞ residual
        w_inf      = 1.0 / (n_cal + 1)
        cum_w      = np.concatenate([sorted_w, [w_inf]])
        aug_r      = np.concatenate([sorted_r, [np.inf]])

        # Find q̂: smallest r such that cumulative weight ≥ α adjusted
        q_target   = (1 - self.alpha) * (1 + w_inf)  # approximately 1-α
        cum        = np.cumsum(cum_w)
        idx_q      = np.searchsorted(cum, q_target)
        idx_q      = min(idx_q, len(aug_r) - 1)
        q_hat      = aug_r[idx_q]

        return float(point_pred - q_hat), float(point_pred + q_hat)

    def predict_intervals(self, X_test: np.ndarray,
                          point_preds: np.ndarray,
                          entity_test: Optional[np.ndarray] = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
        n_test = len(X_test)
        lower  = np.empty(n_test)
        upper  = np.empty(n_test)
        for i in range(n_test):
            ent = int(entity_test[i]) if entity_test is not None else None
            l, u = self.predict_interval(X_test[i], float(point_preds[i]), ent)
            lower[i] = l
            upper[i] = u
        return lower, upper
```

**Config fields**:
```python
use_locally_weighted_conformal: bool = False
conformal_entity_weight: float = 2.0
# Weight multiplier for same-entity calibration residuals in locally weighted CP.
```

**Performance note**: With n_cal≈300, n_test≈63, n_features≈50: 63 × (300 kernel evals + sort) ≈ 0.05s — acceptable.

**Expected gain**: Narrower intervals for provinces whose feature signature is well-represented in calibration data. For provinces present in all 14 years, same-entity weighting boosts up to 13 calibration points (the entity's own history), producing province-specific interval widths.

---

### E-06 — Student-t Predictive Distribution for Small-n Calibration
**Target**: Replace Gaussian-tail quantiles with Student-t for calibration sets that are small (n < 50)
**Background**: When calibration set falls below 50 samples (e.g., per-stratum conformal, or early adopter scenarios), Gaussian-assumption quantile estimates have high variance. Student-t with `df = n_cal - 1` provides heavier tails that better reflect estimation uncertainty, giving valid finite-sample coverage guarantees.

**Method**: In `calibrate_residuals`, fit Student-t MLE and use `t.ppf`:
```python
from scipy.stats import t as t_dist

def _calibrate_with_studentt(residuals: np.ndarray, alpha: float) -> float:
    """
    Fit Student-t to absolute residuals and return (1-α) predictive quantile.
    Uses MLE for (df, loc, scale) via scipy.stats.t.fit with loc=0 forced.
    Falls back to empirical quantile if MLE fails or n > 100.
    """
    n = len(residuals)
    if n > 100:
        # Large sample: empirical quantile is reliable, no need for parametric
        q_level = min(np.ceil((1-alpha)*(n+1))/n, 1.0)
        return float(np.quantile(np.abs(residuals), q_level))

    abs_res = np.abs(residuals)
    try:
        # Fit t-distribution with location=0 (mean-zero residuals)
        df, loc, scale = t_dist.fit(abs_res, floc=0)
        # Predictive interval must account for fitting uncertainty:
        # use df-1 to penalize for estimated scale
        df_pred = max(df - 1, 1.0)
        q_hat = float(t_dist.ppf(1.0 - alpha, df=df_pred, loc=0, scale=scale))
    except Exception:
        # MLE failed — use empirical
        q_level = min(np.ceil((1-alpha)*(n+1))/n, 1.0)
        q_hat = float(np.quantile(abs_res, q_level))

    return q_hat
```

Integrate in `calibrate_residuals` when `len(residuals) < 50`:
```python
if len(residuals) < 50 and getattr(self, 'use_studentt_small_n', True):
    self._q_hat = _calibrate_with_studentt(residuals, self.alpha)
else:
    # existing empirical quantile code
```

**Config fields**:
```python
conformal_studentt_small_n: bool = True
conformal_studentt_threshold: int = 50
# Use Student-t predictive distribution when n_cal < threshold.
```

**Expected gain**: Reduces coverage violations on small calibration sets (Mondrian strata, cross-entity CV) without requiring more data. Especially critical for the scenario where mis-specified stratum boundaries produce 15-sample strata.

---

## 4. Implementation Order

| Priority | Fix/Enhancement | File(s) | Estimated Impact |
|----------|----------------|---------|----------------|
| P0 (blocker) | F-02: Per-output meta-weights | super_learner.py | +3–8% RMSE |
| P0 (blocker) | F-01: Per-component OOF mask | super_learner.py, unified.py | +10–40% cal samples |
| P0 (blocker) | F-06: Adaptive Bonferroni / effective D | unified.py | ±15% interval sharpness |
| P1 (correctness) | F-03: Entity-block bootstrap | super_learner.py | correct weight UQ |
| P1 (correctness) | F-04: Analytical gradient | super_learner.py | 5–8× speedup |
| P1 (correctness) | F-05: cv_min_train_years alignment | config.py, unified.py | reproducibility |
| P1 (correctness) | F-07: Min cal samples warning | conformal.py | robustness |
| P2 (accuracy) | E-04: Multi-output group NNLS | super_learner.py | +2–5% RMSE (small n) |
| P2 (accuracy) | E-01: Repeated walk-forward OOF | super_learner.py | +50% cal residuals |
| P2 (accuracy) | E-02: CQR with QRF | conformal.py, unified.py | 15–25% sharper PI |
| P3 (robustness) | E-03: Mondrian × missingness | conformal.py | per-stratum coverage |
| P3 (robustness) | E-05: Locally weighted CP | conformal.py | province-adaptive PI |
| P3 (robustness) | E-06: Student-t small-n | conformal.py | safety for sparse strata |

---

## 5. Testing Strategy

Each fix/enhancement must pass:

1. **Unit test** (new): Verify per-output weights differ across output columns on a synthetic 4-model × 3-output OOF dataset
2. **Coverage test** (extend existing): At α=0.05, conformal predictor achieves empirical coverage in [0.93, 0.99] on held-out synthetic panel
3. **OOF coverage count test**: After E-01, `_oof_conformal_residuals_extended_` has `n_rows > 1.5 * primary_n_rows`
4. **Weight reproducibility test**: Dirichlet bootstrap with entity-block resampling has wider CI than i.i.d. on synthetic auto-correlated panel
5. **CQR adaptive width test**: CQR intervals are narrower for low-missingness test points and wider for high-missingness test points (compare average width by quartile of missingness)
6. **Student-t coverage test**: With n_cal=20, Student-t path achieves ≥95% coverage over 500 simulation runs; empirical quantile path does not

---

## 6. Key References

- **Barber et al. (2021)**: Predictive Inference with the Jackknife+ — theoretical foundation for OOF conformal
- **Romano et al. (2019)**: Conformalized Quantile Regression — CQR adaptive intervals (E-02)
- **Tibshirani et al. (2019)**: Conformal Prediction Under Covariate Shift — weighted split CP (E-05)
- **Angelopoulos & Bates (2023)**: Conformal Risk Control — multi-output coverage via RAPS
- **Van der Laan et al. (2007)**: Super Learner — oracle inequality requires per-output meta-fitting (F-02)
- **Yao et al. (2018)**: Using Stacking to Average Bayesian Predictive Distributions — Dirichlet stacking (F-03, F-04)
- **Vovk et al. (2005)**: Mondrian conformal prediction — stratified coverage guarantees (E-03)
- **Meeker & Escobar (1998)**: Statistical Intervals — Student-t predictive intervals for small-n (E-06)
