# -*- coding: utf-8 -*-
"""
Unified Forecasting Orchestrator
================================

State-of-the-art ensemble forecasting system optimised for small-to-medium
panel data (N < 1000).  Orchestrates all forecasting sub-components through
a clean single-entry-point API.

Model ensemble (6 diverse types)
---------------------------------
- Gradient Boosting (CatBoost)  — joint multi-output oblivious trees (MultiRMSE)
- LightGBM                      — leaf-wise per-output trees (MultiOutputRegressor)
- Bayesian Ridge                — linear model with posterior uncertainty
- Quantile RF                   — full predictive distributions via leaf quantiles
- Panel VAR                     — LSDV fixed effects + autoregressive dynamics
- Neural Additive Model — interpretable shape functions (optional NAM²)

Meta-ensemble
-------------
- Super Learner (``_PanelTemporalSplit`` panel-aware CV, NNLS meta-weights)
- OOF predictions cached once; never deep-copied during conformal calibration

Uncertainty calibration
-----------------------
- ConformalPredictor calibrated from pre-computed OOF residuals (no model
  re-fitting), supporting Split, CV+ and Adaptive Conformal Inference modes

Feature engineering
-------------------
- Dynamic exclusion via ``panel_data.year_contexts`` (provinces / sub-criteria
  absent from a target year are silently excluded from training)
- NaN feature values filled with 0.0 ("no prior information")

Reliability
-----------
- Decision-matrix NaN imputation: back-fill from prior years + column median
- Ranking phase wrapped in try/except; pipeline continues on partial failure
- Per-component feature importance via NAM ``_per_output_importance_``
- Pairwise ablation study available through ``AblationStudy``
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import copy

logger = logging.getLogger('ml_mcdm')

from .base import BaseForecaster
from .gradient_boosting import CatBoostForecaster, LightGBMForecaster
from .bayesian import BayesianForecaster
from .features import TemporalFeatureEngineer

# State-of-the-art advanced models
from .panel_var import PanelVARForecaster
from .quantile_forest import QuantileRandomForestForecaster
from .neural_additive import NeuralAdditiveForecaster
from .super_learner import SuperLearner, _WalkForwardYearlySplit as PanelWalkForwardCV
from .conformal import ConformalPredictor
from .preprocessing import PanelFeatureReducer
from .evaluation import ForecastEvaluator, AblationStudy, ModelComparisonResult, compare_all_models
# Phase 3 — SOTA modules (E-05, E-06, E-08, E-10)
from .panel_mice import PanelSequentialMICE
from .augmentation import ConditionalPanelAugmenter
from .shift_detection import PanelCovariateShiftDetector
from .incremental_update import IncrementalEnsembleUpdater
import functools

from config import ForecastConfig


def _silence_warnings(func):
    """Scope all warning filters to the duration of *func* only."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return func(*args, **kwargs)
    return wrapper


@dataclass
class _TargetTransformer:
    """
    Reversible per-column target transformation for the forecasting pipeline.

    Modes
    -----
    'logit'
        ``f(y) = log(y / (1-y))``  (SAW-normalized [0,1] targets).
        Clips ``y`` to ``(clip_eps, 1-clip_eps)`` before applying logit.
        Inverse: sigmoid ``f⁻¹(z) = 1 / (1 + exp(-z))``.
    'yeo_johnson'
        Fits ``sklearn.preprocessing.PowerTransformer(method='yeo-johnson',
        standardize=True)`` per column on the training set.  Maps raw
        criterion composites toward N(0,1), improving Gaussian-assumption
        estimators (BayesianRidge, RidgeCV meta-learner).
    'identity'
        Pass-through; no transformation applied.

    Key guarantee: both ``logit`` and ``yeo_johnson`` are **strictly monotone**
    transformations → applying ``f⁻¹`` to conformal bounds ``[lower, upper]``
    (computed in transformed space) yields valid conformal bounds in original
    space (distribution-free coverage is preserved by reparameterisation).
    """
    mode: str = 'yeo_johnson'
    clip_eps: float = 1e-6
    _pt: Any = field(default=None, repr=False)  # PowerTransformer (yeo_johnson only)

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit on ``y_train`` and return transformed array."""
        if self.mode == 'yeo_johnson':
            from sklearn.preprocessing import PowerTransformer
            self._pt = PowerTransformer(method='yeo-johnson', standardize=True)
            return self._pt.fit_transform(y)
        if self.mode == 'logit':
            y_c = np.clip(y, self.clip_eps, 1.0 - self.clip_eps)
            return np.log(y_c / (1.0 - y_c))
        return y.copy()  # identity

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform ``y`` using the already-fitted transformer."""
        if self.mode == 'yeo_johnson':
            if self._pt is None:
                raise RuntimeError(
                    "_TargetTransformer not fitted; call fit_transform first."
                )
            return self._pt.transform(y)
        if self.mode == 'logit':
            y_c = np.clip(y, self.clip_eps, 1.0 - self.clip_eps)
            return np.log(y_c / (1.0 - y_c))
        return y.copy()  # identity

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Invert the transformation (back to original target space)."""
        if self.mode == 'yeo_johnson':
            if self._pt is None:
                raise RuntimeError(
                    "_TargetTransformer not fitted; call fit_transform first."
                )
            return self._pt.inverse_transform(y)
        if self.mode == 'logit':
            # sigmoid: maps transformed ℝ back to (0, 1)
            return 1.0 / (1.0 + np.exp(-np.clip(y, -500, 500)))
        return y.copy()  # identity

    @property
    def is_identity(self) -> bool:
        """True when the transformer is a no-op (mode='identity')."""
        return self.mode == 'identity'


@dataclass
class UnifiedForecastResult:
    """
    Comprehensive result container for unified forecasting.
    
    Attributes:
        predictions: Entity × Component predictions
        uncertainty: Prediction uncertainty estimates
        prediction_intervals: 95% confidence intervals
        model_contributions: Weight of each model
        model_performance: Model-wise metrics
        feature_importance: Aggregated feature importance
        cross_validation_scores: CV scores per model
        holdout_performance: Performance on holdout set
        training_info: Training details
        data_summary: Data summary statistics
    """
    
    # Primary outputs
    predictions: pd.DataFrame
    uncertainty: pd.DataFrame
    prediction_intervals: Dict[str, pd.DataFrame]
    
    # Model analysis
    model_contributions: Dict[str, float]
    model_performance: Dict[str, Dict[str, float]]
    feature_importance: pd.DataFrame
    
    # Validation
    cross_validation_scores: Dict[str, List[float]]
    holdout_performance: Optional[Dict[str, float]]

    # SAW target outputs (populated when use_saw_targets=True)
    composite_predictions: Optional[pd.Series]
    """CRITIC-weighted composite of per-criterion SAW predictions.

    When ``UnifiedForecaster`` is configured with ``use_saw_targets=True``,
    the ensemble predicts per-criterion SAW-normalized scores in [0, 1].
    ``composite_predictions`` is the single aggregate score derived by
    applying ``CRITICWeightCalculator`` to the predicted cross-section and
    computing the weighted sum:

        composite_i = Σ_j  w_j(predicted) × saw_predicted_ij

    This mirrors the MCDM pipeline's composite step but uses the *forecast*
    weights rather than any historical year's weights. ``None`` when
    ``use_saw_targets=False`` or when CRITIC derivation fails.
    """

    # Metadata
    training_info: Dict[str, Any]
    data_summary: Dict[str, Any]

    # Model comparison (populated when holdout data is available)
    best_model_name: Optional[str] = None
    best_model_predictions: Optional[pd.DataFrame] = None
    model_comparison: Optional[List] = None
    forecast_criterion_weights: Optional[Dict[str, float]] = None
    """CRITIC criterion weights (C01–C08) derived from the ML-predicted
    forecasted cross-section.  ``None`` when ``use_saw_targets=False`` or
    when the CRITIC calculation fails.  Populated by Stage 7."""

    def get_summary(self) -> str:
        """Generate comprehensive summary report."""
        lines = [
            "\n" + "=" * 80,
            "UNIFIED ML FORECASTING REPORT",
            "=" * 80,
            "",
            "## Data Summary",
            f"- Entities: {self.data_summary.get('n_entities', 'N/A')}",
            f"- Components: {self.data_summary.get('n_components', 'N/A')}",
            f"- Training samples: {self.training_info.get('n_samples', 'N/A')}",
            f"- Features: {self.training_info.get('n_features', 'N/A')}",
            "",
            "## Model Contributions",
        ]
        
        for model, weight in sorted(self.model_contributions.items(),
                                    key=lambda x: x[1], reverse=True):
            bar = "█" * int(weight * 40)
            lines.append(f"  {model:25s}: {weight:6.3f} {bar}")
        
        lines.extend([
            "",
            "## Cross-Validation Performance",
        ])
        
        for model, scores in self.cross_validation_scores.items():
            mean_r2 = np.nanmean(scores)
            std_r2 = np.nanstd(scores)
            lines.append(f"  {model:25s}: R² = {mean_r2:.4f} ± {std_r2:.4f}")
        
        if self.holdout_performance:
            lines.extend([
                "",
                "## Holdout Validation",
            ])
            for metric, value in self.holdout_performance.items():
                lines.append(f"  {metric}: {value:.4f}")
        
        lines.extend([
            "",
            "## Top 15 Most Important Features",
        ])
        
        if not self.feature_importance.empty:
            mean_importance = self.feature_importance.mean(axis=1).nlargest(15)
            for feat, imp in mean_importance.items():
                lines.append(f"  {feat}: {imp:.4f}")
        
        lines.extend([
            "",
            "## Prediction Summary",
            f"- Mean prediction: {self.predictions.values.mean():.4f}",
            f"- Std prediction: {self.predictions.values.std():.4f}",
            f"- Mean uncertainty: {self.uncertainty.values.mean():.4f}",
        ])

        if self.composite_predictions is not None:
            lines.extend([
                "",
                "## Composite Score (CRITIC-weighted SAW)",
                f"- Mean composite: {self.composite_predictions.mean():.4f}",
                f"- Std  composite: {self.composite_predictions.std():.4f}",
                f"- Min  composite: {self.composite_predictions.min():.4f}",
                f"- Max  composite: {self.composite_predictions.max():.4f}",
            ])

        if self.model_comparison:
            lines.extend(["", "## Model Comparison (Genuine Holdout)"])
            _best_base = next(
                (r for r in self.model_comparison
                 if r.model_name != 'Ensemble' and not np.isnan(r.holdout_r2)),
                None,
            )
            _ens = next(
                (r for r in self.model_comparison if r.model_name == 'Ensemble'),
                None,
            )
            for r in self.model_comparison:
                tag = " [BEST]" if r.is_best else ""
                r2_str  = f"{r.holdout_r2:.4f}"  if not np.isnan(r.holdout_r2)  else "N/A"
                mse_str = f"{r.holdout_rmse:.4f}" if not np.isnan(r.holdout_rmse) else "N/A"
                lines.append(
                    f"  {r.model_name:25s}: R²={r2_str}, RMSE={mse_str}{tag}"
                )
            if self.best_model_name and _ens and _best_base:
                if self.best_model_name == 'Ensemble':
                    lines.append(
                        f"  Best model: Ensemble (R²={_ens.holdout_r2:.4f}) "
                        f"outperforms best base model "
                        f"{_best_base.model_name} (R²={_best_base.holdout_r2:.4f})"
                    )
                else:
                    _best_r2 = next(
                        (r.holdout_r2 for r in self.model_comparison
                         if r.model_name == self.best_model_name), float('nan')
                    )
                    lines.append(
                        f"  Best model: {self.best_model_name} (R²={_best_r2:.4f}) "
                        f"outperforms Ensemble (R²={_ens.holdout_r2:.4f})"
                    )

        lines.extend([
            "",
            "=" * 80,
        ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export results to dictionary."""
        return {
            'predictions': self.predictions.to_dict(),
            'uncertainty': self.uncertainty.to_dict(),
            'model_weights': self.model_contributions,
            'cv_scores': self.cross_validation_scores,
            'feature_importance': self.feature_importance.to_dict()
        }


def _get_per_output_importance(
    model: Any, n_outputs: int, n_features: int
) -> np.ndarray:
    """Extract per-output feature importance from a fitted base model.

    Tries a cascade of model-specific attribute lookups before falling back
    to broadcasting the model's global importance across all outputs.

    Returns:
        Array of shape ``(n_features, n_outputs)`` where each column is the
        normalised importance vector for one output component.  All columns
        sum to 1.0 (or are all-zero for degenerate cases).
    """
    imp = np.zeros((n_features, n_outputs))

    def _normalise(arr: np.ndarray) -> np.ndarray:
        col_sums = arr.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        return arr / col_sums

    def _safe_vec(raw: np.ndarray, col: int) -> None:
        """Write a 1-D importance vector into column ``col`` of imp."""
        raw = np.asarray(raw, dtype=float).ravel()
        length = min(len(raw), n_features)
        imp[:length, col] = np.abs(raw[:length])

    # ------------------------------------------------------------------
    # 1. NeuralAdditiveForecaster — stores (n_outputs, n_features) matrix
    # ------------------------------------------------------------------
    if (
        hasattr(model, "_per_output_importance_")
        and model._per_output_importance_ is not None
    ):
        poi = np.asarray(model._per_output_importance_)
        if poi.ndim == 2 and poi.shape == (n_outputs, n_features):
            return _normalise(poi.T)  # (n_features, n_outputs)

    # ------------------------------------------------------------------
    # 2. PanelVARForecaster — models_[col] is a Ridge on lagged features
    # ------------------------------------------------------------------
    if hasattr(model, "models_") and hasattr(model, "_n_base_features"):
        n_base = int(model._n_base_features)
        for col in range(n_outputs):
            try:
                coef = model.models_[col].coef_
                _safe_vec(coef[:n_base], col)
            except Exception:
                pass
        return _normalise(imp)

    # ------------------------------------------------------------------
    # 3. MultiOutputRegressor wrapper (BayesianForecaster)
    #    model.model.estimators_[col]  (CatBoostForecaster uses path 4)
    # ------------------------------------------------------------------
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "estimators_"):
        for col in range(n_outputs):
            try:
                est = inner.estimators_[col]
                if hasattr(est, "feature_importances_"):
                    _safe_vec(est.feature_importances_, col)
                elif hasattr(est, "coef_"):
                    _safe_vec(est.coef_, col)
            except Exception:
                pass
        return _normalise(imp)

    # ------------------------------------------------------------------
    # 4. Fallback: broadcast global importance across all outputs
    # ------------------------------------------------------------------
    try:
        global_imp = np.abs(np.asarray(model.get_feature_importance(), dtype=float))
        length = min(len(global_imp), n_features)
        imp[:length, :] = global_imp[:length, np.newaxis]
    except Exception:
        imp[:, :] = 1.0 / n_features  # uniform fallback

    return _normalise(imp)


# ---------------------------------------------------------------------------
# E-01 helper: fold-aware entity demean correction factory
# ---------------------------------------------------------------------------

def _build_fold_correction_fn(
    feature_engineer,
    reducer_tree,
    year_labels_arr,
    entity_indices_arr,
    per_model_tree_names=None,
):
    """
    Build a fold-correction callable for ``SuperLearner.fit(fold_correction_fn=...)``.

    The returned function ``correct(model_name, X_fold, train_idx, fold_entity_indices)``
    adjusts the ``_demeaned`` and ``_demeaned_momentum`` feature columns in
    ``X_fold`` so that they reflect entity means computed from the fold's
    training years only — eliminating the look-ahead bias from the globally
    pre-computed entity statistics.

    **Why only _demeaned / _demeaned_momentum?**
    These are the only two feature blocks whose values depend on *all*
    training years at once (via ``_entity_means_`` and ``_entity_mean_deltas_``).
    All other features (lags, rolling windows, EWMA, etc.) are already
    time-local: they only read from data up to ``current_year`` (= feature year
    = target year − 1), so they contain no future leakage.

    **Correction formula**
    Let ``μ_global(e,c)`` be the entity mean over all training feature years,
    and ``μ_fold(e,c)`` be the restricted mean over years ≤ max_fold_feature_year.
    The pre-computed feature contains ``raw − μ_global``.  The corrected value
    should be ``raw − μ_fold = (raw − μ_global) + (μ_global − μ_fold)``.
    Adding the offset ``Δμ = μ_global − μ_fold`` to the stored column achieves
    the correction without recomputing raw values.

    **PLS-track:** BayesianRidge receives PLS-compressed features.  The
    compression matrix absorbs the entity demean non-linearly; no closed-form
    column correction is possible.  Those folds are left uncorrected (mild
    transductive leakage — second-order effect on meta-weights).

    Parameters
    ----------
    feature_engineer : TemporalFeatureEngineer
        Fitted feature engineer; must have ``_entity_means_``,
        ``_entity_mean_deltas_``, ``_entity_yearly_values_``,
        ``_entities_``, ``_components_``, ``_all_train_feature_years_``.
    reducer_tree : PanelFeatureReducer
        Fitted threshold-only reducer; must expose
        ``get_demeaned_column_indices()``.
    year_labels_arr : ndarray (n_cv_samples,)
        Calendar target-year labels for every row in the CV dataset.
    entity_indices_arr : ndarray (n_cv_samples,) or None
        Integer entity indices for every row in the CV dataset.
    per_model_tree_names : set of str, optional
        Model names that use the tree track.  Correction applied only to
        these models' feature matrices.  Others are returned unchanged.

    Returns
    -------
    callable or None
        ``None`` when prerequisite attributes are absent (safe fallback).
    """
    if feature_engineer is None or reducer_tree is None:
        return None

    # Retrieve column indices of demeaned features in the tree-track matrix
    try:
        demeaned_cols, demeaned_mom_cols = reducer_tree.get_demeaned_column_indices()
    except Exception:
        return None

    if not demeaned_cols and not demeaned_mom_cols:
        # No demeaned columns survived variance threshold — nothing to correct
        return None

    if not hasattr(feature_engineer, '_entity_yearly_values_'):
        # Feature engineer was not updated to store per-year raw values
        return None

    _entities    = feature_engineer._entities_     if hasattr(feature_engineer, '_entities_') else []
    _components  = feature_engineer._components_   if hasattr(feature_engineer, '_components_') else []
    _tree_names  = per_model_tree_names or set()
    _global_means        = feature_engineer._entity_means_
    _global_mean_deltas  = feature_engineer._entity_mean_deltas_

    def _correct(model_name, X_fold, train_idx, fold_entity_indices):
        """Apply fold-restricted entity-demean correction to X_fold."""
        # Only correct tree-track models; PLS-compressed matrices are skipped
        if _tree_names and model_name not in _tree_names:
            return X_fold

        if year_labels_arr is None or len(train_idx) == 0:
            return X_fold

        # Determine max feature year for this fold's training window
        fold_max_target_year = int(np.max(year_labels_arr[train_idx]))
        fold_max_feature_year = fold_max_target_year - 1

        try:
            fold_means, fold_mean_deltas = (
                feature_engineer.compute_fold_entity_corrections(fold_max_feature_year)
            )
        except Exception:
            return X_fold

        # No correction needed for the last fold (uses all training data)
        if fold_means is fold_mean_deltas and fold_means is feature_engineer._entity_means_:
            return X_fold

        X_corrected = X_fold.copy()
        n_rows = len(X_fold)

        for row_i in range(n_rows):
            ent_idx = (
                int(fold_entity_indices[row_i])
                if (fold_entity_indices is not None and row_i < len(fold_entity_indices))
                else -1
            )
            entity_name = _entities[ent_idx] if 0 <= ent_idx < len(_entities) else None
            if entity_name is None:
                continue

            for ci, comp in enumerate(_components):
                g_mean  = _global_means.get((entity_name, comp), 0.0)
                f_mean  = fold_means.get((entity_name, comp), g_mean)
                Δ_mean  = g_mean - f_mean  # correction offset

                g_delta = _global_mean_deltas.get((entity_name, comp), 0.0)
                f_delta = fold_mean_deltas.get((entity_name, comp), g_delta)
                Δ_delta = g_delta - f_delta

                if ci < len(demeaned_cols):
                    col_dm = demeaned_cols[ci]
                    if col_dm < X_corrected.shape[1]:
                        X_corrected[row_i, col_dm] += Δ_mean

                if ci < len(demeaned_mom_cols):
                    col_dm_mom = demeaned_mom_cols[ci]
                    if col_dm_mom < X_corrected.shape[1]:
                        X_corrected[row_i, col_dm_mom] += Δ_delta

        return X_corrected

    return _correct


class UnifiedForecaster:
    """
    State-of-the-art unified forecasting system.

    Optimized for small-to-medium panel data (N < 1000) with statistically-principled
    ensemble design emphasizing model diversity over quantity.

    Tier 1 - Base Models (6 diverse models):
        1. Gradient Boosting / CatBoost (joint multi-output oblivious trees)
        2. LightGBM (leaf-wise per-output trees, complementary inductive bias)
        3. Bayesian Ridge (linear with uncertainty quantification)
        4. Quantile Random Forest (distributional forecasting)
        5. Panel VAR (panel fixed effects + autoregressive)
        6. Neural Additive Models (interpretable non-linearity)

    Tier 2 - Meta-Ensemble:
        - Super Learner: Trains meta-learner on out-of-fold predictions
        - Optimal weighting learned from validation performance

    Tier 3 - Uncertainty Calibration:
        - Conformal Prediction: Distribution-free guaranteed intervals
        - Quantile RF: Full predictive distributions

    Features:
    - Automatic model weighting via Super Learner
    - Comprehensive temporal feature engineering
    - Multi-level ensemble stacking with meta-learning
    - Calibrated uncertainty quantification (95% coverage guarantee)
    - Time-series aware cross-validation

    Parameters:
        conformal_method: Conformal method ('split', 'cv_plus', 'adaptive')
        conformal_alpha: Miscoverage rate for conformal intervals (default: 0.05 = 95% coverage)
        cv_folds: Number of cross-validation folds (default: 3)
        random_state: Random seed for reproducibility
        verbose: Print progress messages

    Example:
        >>> forecaster = UnifiedForecaster()
        >>> result = forecaster.fit_predict(panel_data, target_year=2025)
        >>>
        >>> # With custom conformal settings
        >>> forecaster = UnifiedForecaster(conformal_alpha=0.10, cv_folds=5)
        >>> result = forecaster.fit_predict(panel_data, target_year=2025)
    """

    def __init__(self,
                 conformal_method: str = 'cv_plus',
                 conformal_alpha: float = 0.05,
                 cv_folds: int = 3,
                 cv_min_train_years: int = 7,
                 random_state: int = 42,
                 verbose: bool = True,
                 config: Optional[ForecastConfig] = None,
                 target_level: str = "criteria",
                 uncertainty_method: str = 'conformal'):
        self.conformal_method = conformal_method
        self.conformal_alpha = conformal_alpha
        # When a ForecastConfig is supplied, its uncertainty_method takes precedence.
        # The default 'conformal' preserves backward-compatibility for all tests
        # that construct UnifiedForecaster() without passing a config.
        self.uncertainty_method = (
            config.uncertainty_method if config is not None else uncertainty_method
        )
        self.cv_folds = cv_folds
        self.cv_min_train_years = cv_min_train_years
        self.random_state = random_state
        self.verbose = verbose
        self.target_level = target_level
        # ForecastConfig instance for model-level hyperparameters (gb_max_depth,
        # gb_n_estimators, nam_n_basis, nam_n_iterations, pvar_lag_selection_method).
        # When None, _create_models() falls back to hardened production defaults.
        self._config: Optional[ForecastConfig] = config

        # ── SAW target normalization & true holdout ───────────────────────
        # Resolved from ForecastConfig when provided; otherwise production defaults.
        # use_saw_targets=True: train on per-year minmax-normalised [0,1] scores
        # and compute a CRITIC-weighted composite after prediction (Phase 1).
        self.use_saw_targets: bool = (
            config.use_saw_targets if config is not None else True
        )
        # holdout_year=None: auto-set at runtime to max(training years) so the
        # most recent complete year serves as a true out-of-sample evaluation set.
        self.holdout_year: Optional[int] = (
            config.holdout_year if config is not None else None
        )

        # Phase 5: target transformation (logit for SAW, yeo-johnson otherwise)
        self.use_target_transform: bool = (
            config.use_target_transform if config is not None else True
        )
        self.target_transformer_: Optional[_TargetTransformer] = None
        self.y_train_raw_: Optional[pd.DataFrame] = None
        self.y_holdout_raw_: Optional[pd.DataFrame] = None
        # Phase 4: tuned GB hyperparameters (populated by _tune_gb_hyperparameters)
        self._tuned_gb_params_: Dict[str, Dict] = {}

        self.models_: Dict[str, BaseForecaster] = {}
        self.model_weights_: Dict[str, float] = {}
        self.feature_engineer_ = TemporalFeatureEngineer(target_level=self.target_level)
        self.reducer_pca_: Optional[PanelFeatureReducer] = None
        self.reducer_tree_: Optional[PanelFeatureReducer] = None
        self.super_learner_: Optional[SuperLearner] = None
        self.conformal_predictor_: Optional[ConformalPredictor] = None
        self.evaluator_: Optional[ForecastEvaluator] = None

        # ── Shared phase-transition outputs always available via X_holdout_ ──
        self.X_holdout_: pd.DataFrame = pd.DataFrame()
        self.y_holdout_: pd.DataFrame = pd.DataFrame()
        self.holdout_year_: Optional[int] = None

        # ── Pipeline mode (Phase 8) ───────────────────────────────────────────
        # Config takes precedence; bare constructor defaults to 'full'.
        self.pipeline_mode: str = (
            config.pipeline_mode if config is not None else 'full'
        )

        # ── Stage 1 public outputs ────────────────────────────────────────────
        self.X_train_: pd.DataFrame = pd.DataFrame()
        self.y_train_: pd.DataFrame = pd.DataFrame()
        self.X_pred_: pd.DataFrame = pd.DataFrame()
        self.entity_info_: pd.DataFrame = pd.DataFrame()
        # Internal helpers (prefixed _) for downstream stage methods
        self._entity_indices_: Optional[np.ndarray] = None
        self._year_labels_arr_: Optional[np.ndarray] = None
        self._pred_entity_indices_: Optional[np.ndarray] = None
        self._panel_data_: Any = None
        self._target_year_: Optional[int] = None

        # ── Stage 2 public outputs ────────────────────────────────────────────
        self.X_train_pca_: Optional[np.ndarray] = None
        self.X_train_tree_: Optional[np.ndarray] = None
        self.X_pred_pca_: Optional[np.ndarray] = None
        self.X_pred_tree_: Optional[np.ndarray] = None
        self._per_model_X_train_: Dict[str, np.ndarray] = {}
        self._per_model_X_pred_: Dict[str, np.ndarray] = {}

        # ── Stage 3 public outputs ────────────────────────────────────────────
        self.oof_predictions_: Optional[np.ndarray] = None
        self._cv_scores_: Dict[str, List[float]] = {}
        self._oof_conformal_residuals_: Optional[np.ndarray] = None  # E-02

        # ── Stage 4 public outputs ────────────────────────────────────────────
        self._predictions_arr_: Optional[np.ndarray] = None
        self._uncertainty_arr_: Optional[np.ndarray] = None
        self._pred_df_: pd.DataFrame = pd.DataFrame()
        self._unc_df_: pd.DataFrame = pd.DataFrame()
        self._intervals_: Dict[str, pd.DataFrame] = {}

        # ── Stage 5 public outputs ────────────────────────────────────────────
        self.prediction_intervals_: Dict[str, pd.DataFrame] = {}
        self.conformal_predictors_: Dict[str, ConformalPredictor] = {}

        # ── Stage 6 public outputs ────────────────────────────────────────────
        self.model_comparison_: Optional[List] = None
        self.holdout_performance_: Optional[Dict[str, Any]] = None
        self._feature_importance_: pd.DataFrame = pd.DataFrame()
        self._model_performance_: Dict[str, Any] = {}
        self._training_info_: Dict[str, Any] = {}
        self._holdout_y_test_: Optional[np.ndarray] = None
        self._holdout_y_pred_: Optional[np.ndarray] = None

        # ── Stage 7 public outputs ────────────────────────────────────────────
        self.composite_predictions_: Optional[pd.Series] = None

        # ── Phase 3 — SOTA module instances (E-05, E-06, E-08, E-10) ─────────
        self._panel_mice_:       Optional[PanelSequentialMICE]        = None
        self._augmenter_:        Optional[ConditionalPanelAugmenter]   = None
        self._shift_detector_:   Optional[PanelCovariateShiftDetector] = None
        self._incremental_updater_: Optional[IncrementalEnsembleUpdater] = None

    def _tune_gb_hyperparameters(self) -> None:
        """Phase 4: Panel-safe Optuna HP search for CatBoost and LightGBM (E-09).

        Runs a single TPE study per GB model over the full training set using
        ``PanelWalkForwardCV(min_train_years=7, max_folds=4)``.  Skipped
        silently when optuna is not installed.  Results stored in
        ``self._tuned_gb_params_`` and consumed by ``_create_models()``.
        Only called when ``config.auto_tune_gb=True``.

        Improvements over the original (E-09):
        - Expanded search space covering subsample, colsample, min_child,
          l1 regularisation, and deeper/longer tree budgets.
        - MedianPruner: intermediate per-fold RMSE values are reported so
          that Optuna can prune unpromising trials early (saves ~40% wall time).
        - Default n_trials increased to 40 for better Pareto coverage; still
          controllable via ``config.gb_tune_n_trials``.
        - Panel-safe CV: ``PanelWalkForwardCV`` ensures fold boundaries align
          with calendar years so no holdout-year leakage occurs during search.

        Note on entity_indices: CatBoost and LightGBM are tree models that do
        not use entity-level random effects; the panel structure is handled
        purely by the walk-forward CV splitter.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("HP tuning skipped (optuna not installed).")
            return

        n_trials = (
            getattr(self._config, 'gb_tune_n_trials', 40)
            if self._config is not None else 40
        )
        X_tree      = self.X_train_tree_
        y           = self.y_train_.values
        year_labels = self._year_labels_arr_
        logger.info(
            f"Phase 4 (E-09): Panel-safe Optuna HP search "
            f"({n_trials} trials × 2 models, MedianPruner)..."
        )

        def _cv_objective(trial, forecaster_class, model_key):
            """Build params from trial, run walk-forward CV, report per-fold RMSE."""
            params: Dict[str, Any] = {}

            if model_key == 'CatBoost':
                params['iterations']         = trial.suggest_int('iterations', 50, 600, step=50)
                params['depth']              = trial.suggest_int('depth', 3, 8)
                params['learning_rate']      = trial.suggest_float('learning_rate', 5e-3, 0.3, log=True)
                params['l2_leaf_reg']        = trial.suggest_float('l2_leaf_reg', 0.05, 50.0, log=True)
                params['subsample']          = trial.suggest_float('subsample', 0.6, 1.0)
                params['colsample_bylevel']  = trial.suggest_float('colsample_bylevel', 0.6, 1.0)
                params['min_data_in_leaf']   = trial.suggest_int('min_data_in_leaf', 2, 20)
            else:  # LightGBM
                params['n_estimators']       = trial.suggest_int('n_estimators', 50, 600, step=50)
                params['max_depth']          = trial.suggest_int('max_depth', 3, 8)
                params['learning_rate']      = trial.suggest_float('learning_rate', 5e-3, 0.3, log=True)
                params['num_leaves']         = trial.suggest_int('num_leaves', 15, 80)
                params['l2_reg']             = trial.suggest_float('l2_reg', 0.05, 50.0, log=True)
                params['l1_reg']             = trial.suggest_float('l1_reg', 1e-4, 10.0, log=True)
                params['min_child_samples']  = trial.suggest_int('min_child_samples', 5, 50)
                params['subsample']          = trial.suggest_float('subsample', 0.6, 1.0)
                params['colsample_bytree']   = trial.suggest_float('colsample_bytree', 0.6, 1.0)

            inner_cv = PanelWalkForwardCV(min_train_years=7, max_folds=4)
            rmse_list: List[float] = []
            try:
                model = forecaster_class(
                    **params, random_state=self.random_state
                )
                for step, (tr, va) in enumerate(inner_cv.split(X_tree, year_labels)):
                    m = copy.deepcopy(model)
                    m.fit(X_tree[tr], y[tr])
                    pred = m.predict(X_tree[va])
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, 1)
                    fold_rmse = float(np.sqrt(mean_squared_error(y[va], pred)))
                    rmse_list.append(fold_rmse)
                    # Report intermediate value for MedianPruner
                    trial.report(float(np.mean(rmse_list)), step)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
            except optuna.exceptions.TrialPruned:
                raise
            except Exception:
                return float('inf')
            return float(np.mean(rmse_list)) if rmse_list else float('inf')

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,    # don't prune until 5 completed trials
            n_warmup_steps=1,      # don't prune before fold 1 result
            interval_steps=1,
        )
        for name, cls, model_key in [
            ('CatBoost', CatBoostForecaster,  'CatBoost'),
            ('LightGBM',         LightGBMForecaster,  'LightGBM'),
        ]:
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(
                    seed=self.random_state, multivariate=True
                ),
                pruner=pruner,
            )
            study.optimize(
                lambda t, c=cls, k=model_key: _cv_objective(t, c, k),
                n_trials=n_trials,
                show_progress_bar=False,
            )
            self._tuned_gb_params_[name] = study.best_params
            pruned = len([t for t in study.trials
                          if t.state == optuna.trial.TrialState.PRUNED])
            logger.info(
                f"  {name} best HPs: {study.best_params} "
                f"(best RMSE={study.best_value:.4f}, "
                f"{pruned}/{n_trials} trials pruned)"
            )

    def _create_models(self) -> Dict[str, BaseForecaster]:
        """
        Create all base model instances (6 diverse models).

        Hyperparameters are resolved from the ForecastConfig passed to
        ``__init__`` (if provided), otherwise production defaults are used.
        All tunable parameters are exposed in ``ForecastConfig`` so they can
        be adjusted without modifying source code.

        Default decisions:
            CatBoost         : max_depth=5 (32 leaves ≈ 24 samples/leaf at
                               n=756), n_estimators=200 (class default)
            LightGBM         : max_depth=5, n_estimators=200 — same scale as
                               CatBoost; leaf-wise growth provides complementary
                               inductive bias as independent ensemble member
            NAM              : n_basis=30 (60 effective params ≈ PLS dims),
                               n_iterations=10 (sufficient backfitting)
            PanelVAR         : lag_selection_method='cv' (hold-out MSE;
                               only correct method for Ridge-regularised VAR)
        """
        # Resolve hyperparameters: config takes priority, else use defaults
        cfg = self._config
        gb_n_est    = cfg.gb_n_estimators           if cfg is not None else 200
        gb_depth    = cfg.gb_max_depth              if cfg is not None else 5
        nam_n_basis = cfg.nam_n_basis               if cfg is not None else 30
        nam_n_iter  = cfg.nam_n_iterations          if cfg is not None else 10
        pvar_method = cfg.pvar_lag_selection_method if cfg is not None else "cv"

        models = {}

        # ── Phase 4: merge tuned HPs with config defaults (tuned take priority)
        _gb_params   = self._tuned_gb_params_.get('CatBoost', {})
        _lgbm_params = self._tuned_gb_params_.get('LightGBM', {})

        # --- Tier 1a: Tree-based (two independent GB members) --------------
        models['CatBoost'] = CatBoostForecaster(
            iterations=_gb_params.get('iterations', gb_n_est),
            depth=_gb_params.get('depth', gb_depth),
            learning_rate=_gb_params.get('learning_rate', 0.05),
            l2_leaf_reg=_gb_params.get('l2_leaf_reg', 3.0),
            random_state=self.random_state,
        )
        models['LightGBM'] = LightGBMForecaster(
            n_estimators=_lgbm_params.get('n_estimators', gb_n_est),
            max_depth=_lgbm_params.get('max_depth', gb_depth),
            learning_rate=_lgbm_params.get('learning_rate', 0.05),
            l2_reg=_lgbm_params.get('l2_reg', 3.0),
            random_state=self.random_state,
        )

        # --- Tier 1b: Bayesian linear -------------------------------------
        models['BayesianRidge'] = BayesianForecaster()

        # --- Tier 1c: Advanced panel-specific models ----------------------
        models['QuantileRF'] = QuantileRandomForestForecaster(
            n_estimators=100, random_state=self.random_state
        )
        models['PanelVAR'] = PanelVARForecaster(
            n_lags=1, alpha=0.5, use_fixed_effects=True,
            lag_selection_method=pvar_method,
            random_state=self.random_state
        )
        models['NAM'] = NeuralAdditiveForecaster(
            n_basis_per_feature=nam_n_basis, n_iterations=nam_n_iter,
            random_state=self.random_state
        )

        return models

    def _compute_critic_composite(
        self,
        predicted_saw_scores: pd.DataFrame,
    ) -> pd.Series:
        """
        Derive CRITIC-weighted composite scores from predicted per-criterion
        SAW-normalized values.

        When the model is trained on SAW-normalized targets (bounded [0, 1]),
        its predictions represent an estimated per-criterion position within
        the forecast-year decision matrix.  To produce a single composite score
        per province — mirroring the MCDM pipeline — we apply CRITIC weighting
        to the *predicted* cross-section, then compute the weighted row sum.

        Using the predicted cross-section (rather than any historical year's
        weights) ensures the composite reflects the information content of
        the forecast decision matrix: criteria with higher predicted spread and
        lower inter-criteria correlation receive larger weights.

        Algorithm
        ---------
        1. Clip predictions to [0, 1] (SAW normalization domain).  Model
           outputs may slightly overshoot due to extrapolation; clamping
           ensures consistency with the SAW training domain.
        2. Drop zero-variance predicted criteria (constant column → CRITIC
           weight = 0 regardless; excluding them avoids numerical instability
           in the correlation matrix and in the standard deviation term).
        3. Impute any residual NaN cells in the active columns with the
           column mean — a rare edge case arising if a province is predicted
           but its raw criterion level is entirely unobserved.
        4. Run ``CRITICWeightCalculator.calculate(predicted_matrix)`` to
           compute information content weights for each criterion.
        5. Re-normalise weights to sum to 1 over *all* criteria (including
           any dropped constant columns, which receive weight 0).
        6. Return ``(clipped_scores × weights).sum(axis=1)``.

        Parameters
        ----------
        predicted_saw_scores : pd.DataFrame
            Per-criterion SAW predictions, shape (n_entities, n_criteria).
            Index = province names; columns = criterion identifiers (C01..C08).

        Returns
        -------
        pd.Series
            Composite score per entity, bounded approximately in [0, 1].
            Index = entity names (same as ``predicted_saw_scores.index``).
            Named ``'composite_score'``.

        Notes
        -----
        Fallback to equal-weight mean is used when:
        * Fewer than 2 entities (CRITIC requires ≥ 2 observations).
        * All criteria are constant (zero variance → no information content).
        * ``CRITICWeightCalculator.calculate()`` raises any exception.
        """
        from weighting.critic import CRITICWeightCalculator

        # ── Step 1: clip to SAW domain ────────────────────────────────────
        scores = predicted_saw_scores.clip(lower=0.0, upper=1.0)

        # ── Guard: CRITIC needs ≥ 2 observations ─────────────────────────
        if len(scores) < 2:
            if self.verbose:
                print(
                    "    _compute_critic_composite: fewer than 2 entities — "
                    "using equal-weight composite."
                )
            return scores.mean(axis=1).rename('composite_score')

        # ── Step 2: drop zero-variance columns ───────────────────────────
        col_range = scores.max() - scores.min()
        non_const_cols = col_range[col_range > 1e-8].index
        scores_active = scores[non_const_cols]

        if scores_active.empty:
            if self.verbose:
                print(
                    "    _compute_critic_composite: all predicted criteria are "
                    "constant — using equal-weight composite."
                )
            return scores.mean(axis=1).rename('composite_score')

        # ── Step 3: exclude rows with residual NaN (no imputation) ───────
        # Consistent with the complete-case strategy applied throughout the
        # pipeline: rows (entities) whose predicted criterion scores contain
        # NaN are excluded from the weight-estimation step rather than filled
        # with synthetic means.  Criterion weights are column properties and
        # can be reliably estimated from the remaining complete-case entities;
        # the full ``scores`` matrix is then used for the final composite.
        if scores_active.isnull().any().any():
            scores_active = scores_active.dropna(how='any')
            if len(scores_active) < 2:
                if self.verbose:
                    print(
                        "    _compute_critic_composite: fewer than 2 "
                        "complete-case rows after NaN exclusion — "
                        "using equal-weight composite."
                    )
                return scores.mean(axis=1).rename('composite_score')

        # ── Steps 4–6: CRITIC weights → weighted composite ───────────────
        try:
            weight_result = CRITICWeightCalculator().calculate(scores_active)

            # Build weight Series aligned to ALL criteria; dropped constant
            # columns receive weight 0 (their CRITIC information content is 0)
            weights = pd.Series(0.0, index=predicted_saw_scores.columns)
            for c, w in weight_result.weights.items():
                if c in weights.index:
                    weights[c] = w

            # Re-normalise so weights sum to 1 over the active criteria
            w_sum = weights.sum()
            if w_sum > 0:
                weights = weights / w_sum

            # Store on self so the result object can expose them
            self.forecast_criterion_weights_ = weights.to_dict()

            composite = (scores * weights).sum(axis=1)

        except Exception as exc:
            if self.verbose:
                print(
                    f"    _compute_critic_composite: CRITICWeightCalculator "
                    f"failed ({exc}) — falling back to equal-weight mean."
                )
            composite = scores.mean(axis=1)

        return composite.rename('composite_score')

    # =========================================================================
    # Phase 8 — Pipeline Decoupling: 7 public stage methods
    # =========================================================================

    def stage1_engineer_features(
        self,
        panel_data: Any,
        target_year: int,
    ) -> None:
        """Stage 1: Temporal feature engineering.

        Calls ``TemporalFeatureEngineer.fit_transform()`` to build training and
        prediction feature matrices, SAW-normalised targets (when
        ``use_saw_targets=True``), and row-level entity metadata.

        All outputs are stored on ``self`` so that subsequent stage methods and
        :meth:`get_stage_outputs` can access them without recomputation.

        Pre-requisite: none (entry point of the pipeline).

        Outputs stored on ``self``
        -------------------------
        X_train_         Training feature matrix ``(n_train, n_features)``.
        y_train_         Training targets ``(n_train, n_components)``.
        X_pred_          Prediction feature matrix ``(n_entities, n_features)``.
        X_holdout_       Holdout features  ``(n_holdout, n_features)`` or empty.
        y_holdout_       Holdout targets   ``(n_holdout, n_components)`` or empty.
        entity_info_     Row-level metadata (``entity_index``, ``year_label``).
        holdout_year_    Calendar year held out; ``None`` if insufficient history.
        """
        logger.info("Stage 1: Engineering temporal features...")

        self._panel_data_ = panel_data
        self._target_year_ = target_year

        # ── Resolve holdout year ──────────────────────────────────────────
        # When holdout_year is None (default), auto-set to the most recent
        # training year (max(years) < target_year).  This ensures the latest
        # complete historical year is always reserved for OOS evaluation.
        _holdout_year = self.holdout_year
        if _holdout_year is None:
            _all_years = sorted(panel_data.years)
            _train_years = [y for y in _all_years if y < target_year]
            _holdout_year = max(_train_years) if _train_years else None
        self.holdout_year_ = _holdout_year

        logger.debug(f"  Target year: {target_year}, Holdout year: {_holdout_year}")

        X_train, y_train, X_pred, entity_info, X_holdout, y_holdout = (
            self.feature_engineer_.fit_transform(
                panel_data,
                target_year,
                use_saw_normalization=self.use_saw_targets,
                holdout_year=_holdout_year,
            )
        )

        logger.info(
            f"  Features engineered: "
            f"X_train={X_train.shape}, y_train={y_train.shape}, "
            f"X_pred={X_pred.shape}"
        )

        # ── Public stage outputs ─────────────────────────────────────────
        self.X_train_ = X_train
        self.y_train_ = y_train
        self.X_pred_ = X_pred
        self.entity_info_ = entity_info
        self.X_holdout_ = X_holdout
        self.y_holdout_ = y_holdout

        # ── Phase 5: Optional target transformation ──────────────────────
        # Logit for bounded SAW [0,1] targets; Yeo-Johnson for raw values.
        # Transform is applied BEFORE stage2 so the PLS track compresses
        # features against the transformed target (correct covariance).
        # `y_train_raw_` / `y_holdout_raw_` preserve original-space values
        # for later inverse-transformation of pipeline outputs.
        if self.use_target_transform:
            _tt_mode = (
                'logit' if self.use_saw_targets else 'yeo_johnson'
            )
        else:
            _tt_mode = 'identity'
        self.target_transformer_ = _TargetTransformer(mode=_tt_mode)
        if not self.target_transformer_.is_identity:
            self.y_train_raw_   = y_train.copy()
            self.y_holdout_raw_ = (
                y_holdout.copy() if not y_holdout.empty else pd.DataFrame()
            )
            _y_train_t = self.target_transformer_.fit_transform(
                y_train.values
            )
            self.y_train_ = pd.DataFrame(
                _y_train_t, index=y_train.index, columns=y_train.columns
            )
            if not y_holdout.empty:
                _y_ho_t = self.target_transformer_.transform(y_holdout.values)
                self.y_holdout_ = pd.DataFrame(
                    _y_ho_t, index=y_holdout.index, columns=y_holdout.columns
                )
            _t_min = float(self.y_train_.values.min())
            _t_max = float(self.y_train_.values.max())
            logger.info(
                f"  Target transform '{_tt_mode}': "
                f"y_train ∈ [{_t_min:.4f}, {_t_max:.4f}]"
            )
        else:
            # identity: still call fit_transform for consistent state
            self.target_transformer_.fit_transform(y_train.values)

        if self.verbose and _holdout_year is not None:
            if not X_holdout.empty:
                print(
                    f"    Holdout: {len(X_holdout)} samples reserved "
                    f"(target year = {_holdout_year})"
                )
                if self.use_saw_targets:
                    y_min = y_train.values.min()
                    y_max = y_train.values.max()
                    print(
                        f"    SAW targets: y_train ∈ [{y_min:.4f}, {y_max:.4f}] "
                        f"(should be ≈ [0, 1])"
                    )

        # ── Internal helpers for downstream stage methods ─────────────────
        self._entity_indices_ = (
            entity_info['entity_index'].values
            if 'entity_index' in entity_info.columns else None
        )
        self._year_labels_arr_ = (
            entity_info['year_label'].values
            if 'year_label' in entity_info.columns else None
        )

        # Compute prediction entity indices (for PanelVAR fixed effects).
        _ent_to_idx = {e: i for i, e in enumerate(panel_data.provinces)}
        if _ent_to_idx:
            _missing = [e for e in X_pred.index if e not in _ent_to_idx]
            if _missing:
                warnings.warn(
                    f"UnifiedForecaster: {len(_missing)} prediction province(s)"
                    f" are absent from the training entity map and will use the"
                    f" reference entity's fixed effects in PanelVARForecaster."
                    f"  Affected: {_missing[:5]}"
                    f"{'...' if len(_missing) > 5 else ''}."
                    " Verify TemporalFeatureEngineer exclusion logic.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self._pred_entity_indices_ = (
            np.array([_ent_to_idx.get(e, 0) for e in X_pred.index], dtype=int)
            if _ent_to_idx else None
        )

    def stage2_reduce_features(self) -> None:
        """Stage 2: Two-track dimensionality reduction.

        Fits two :class:`PanelFeatureReducer` objects on the training matrix
        and applies both to the training and prediction sets:

        * **PLS track** (``reducer_pca_``) — supervised dimensionality
          reduction via ``PLSRegression`` with MI pre-filter.  Finds the 20
          linear combinations of X with maximum covariance with all 8 criterion
          targets simultaneously.  Used exclusively by ``BayesianRidge`` /
          ``MultiTaskElasticNetCV``; target-aware compression is strictly
          superior to PCA for forecasting accuracy.
          ``n_components = min(n // 10, 20)`` → p/n ≤ 0.024.

        * **Threshold-only track** (``reducer_tree_``) — removes near-zero-
          variance features only, no scaling, no compression.  Used by all
          non-linear models (CatBoost, QRF, PanelVAR, NAM); preserves the
          original feature structure so tree splits and NAM shape functions
          capture the real interactions.  StandardScaler removed to prevent
          double-scaling with QRF's internal RobustScaler and CatBoost's
          scale-invariant trees.

        Pre-requisite: :meth:`stage1_engineer_features` must have been called.

        Outputs stored on ``self``
        -------------------------
        X_train_pca_     PLS-compressed training features.
        X_train_tree_    Threshold-only training features.
        X_pred_pca_      PLS-compressed prediction features.
        X_pred_tree_     Threshold-only prediction features.
        reducer_pca_     Fitted PLS reducer (attribute name kept for compat).
        reducer_tree_    Fitted threshold-only reducer.
        """
        logger.info("Stage 2: Two-track dimensionality reduction...")

        X_arr = self.X_train_.values
        y_arr = self.y_train_.values
        feature_names = self.feature_engineer_.get_feature_names()

        # ── E-05: Three-phase PanelSequentialMICE imputation ──────────────
        # Applied to the raw feature matrices BEFORE dimensionality reduction
        # so imputed values flow into PLS supervision correctly.  Only runs
        # when enabled in ForecastConfig and when NaN is actually present.
        if (
            getattr(self._config, 'use_panel_mice', False)
            and self._entity_indices_ is not None
            and self._year_labels_arr_ is not None
            and np.isnan(X_arr).any()
        ):
            self._panel_mice_ = PanelSequentialMICE(verbose=self.verbose)
            X_arr = self._panel_mice_.fit_transform(
                X_arr,
                self._entity_indices_,
                self._year_labels_arr_,
            )
            logger.info(
                f"  PanelMICE: {self._panel_mice_.nan_before_} NaN → "
                f"{self._panel_mice_.nan_after_} "
                f"({self._panel_mice_.nan_reduction_pct:.1f}% eliminated)"
            )
            # Apply transform() to the prediction feature matrix (target year,
            # no hold-out data available so only Phase 1+2+3 transform is used).
            if self._pred_entity_indices_ is not None:
                _pred_yr = np.full(
                    len(self.X_pred_), int(self._target_year_), dtype=int
                )
                _pred_imp = self._panel_mice_.transform(
                    self.X_pred_.values,
                    self._pred_entity_indices_,
                    _pred_yr,
                )
                self.X_pred_ = pd.DataFrame(
                    _pred_imp,
                    index=self.X_pred_.index,
                    columns=self.X_pred_.columns,
                )
            # Apply transform() to the holdout feature matrix when present.
            if (
                not self.X_holdout_.empty
                and self.holdout_year_ is not None
                and self._panel_data_ is not None
            ):
                _ent_to_idx = {
                    e: i for i, e in enumerate(self._panel_data_.provinces)
                }
                _ho_ent_idx = np.array(
                    [_ent_to_idx.get(e, 0) for e in self.X_holdout_.index],
                    dtype=int,
                )
                _ho_yr = np.full(
                    len(self.X_holdout_), int(self.holdout_year_), dtype=int
                )
                _ho_imp = self._panel_mice_.transform(
                    self.X_holdout_.values, _ho_ent_idx, _ho_yr
                )
                self.X_holdout_ = pd.DataFrame(
                    _ho_imp,
                    index=self.X_holdout_.index,
                    columns=self.X_holdout_.columns,
                )

        # PLS track for linear models: supervised compression with MI pre-filter
        self.reducer_pca_ = PanelFeatureReducer(mode='pls', mi_prefilter=True)
        # Threshold-only track for tree models: variance filter, no scaling
        self.reducer_tree_ = PanelFeatureReducer(mode='threshold_only')

        self.X_train_pca_ = self.reducer_pca_.fit_transform(
            X_arr, y=y_arr, feature_names=feature_names
        )
        self.X_pred_pca_ = self.reducer_pca_.transform(self.X_pred_.values)

        self.X_train_tree_ = self.reducer_tree_.fit_transform(
            X_arr, feature_names=feature_names
        )
        self.X_pred_tree_ = self.reducer_tree_.transform(self.X_pred_.values)

        # Per-model routing: BayesianRidge → PLS track; trees/NAM → threshold.
        self._per_model_X_train_ = {
            'BayesianRidge':    self.X_train_pca_,
            'CatBoost': self.X_train_tree_,
            'LightGBM':         self.X_train_tree_,
            'QuantileRF':       self.X_train_tree_,
            'PanelVAR':         self.X_train_tree_,
            'NAM':              self.X_train_tree_,
        }
        self._per_model_X_pred_ = {
            'BayesianRidge':    self.X_pred_pca_,
            'CatBoost': self.X_pred_tree_,
            'LightGBM':         self.X_pred_tree_,
            'QuantileRF':       self.X_pred_tree_,
            'PanelVAR':         self.X_pred_tree_,
            'NAM':              self.X_pred_tree_,
        }

        logger.info(
            f"  PLS track:      {self.reducer_pca_.get_summary()}"
        )
        logger.info(
            f"  Threshold-only: {self.reducer_tree_.get_summary()}"
        )

    def stage2b_augment_data(self) -> None:
        """Stage 2b (E-06): Conditional synthetic data augmentation.

        Optionally augments the training feature and target matrices with
        synthetic entity-year rows generated by a Gaussian-copula + VAR(1)
        model fitted per entity.  Augmentation is committed only when a 5-fold
        walk-forward CV shows ΔR² > ``config.augment_gain_threshold`` (default
        0.005), preventing spurious data inflation.

        Augmented rows are appended to the TREE track (``X_train_tree_``,
        ``_per_model_X_train_`` for tree models) so that the linear / PLS
        track (``BayesianRidge``) always trains on the original data only.
        Synthetic rows receive entity+year labels so ``SyntheticAwareCV``
        ensures they never appear in CV validation folds.

        Pre-requisite: :meth:`stage2_reduce_features` must have been called.

        Outputs modified on ``self``
        ----------------------------
        X_train_tree_         May be row-extended with synthetic rows.
        y_train_ (values)     Corresponding synthetic targets appended.
        _entity_indices_      Extended with synthetic entity IDs.
        _year_labels_arr_     Extended with synthetic year labels.
        _per_model_X_train_   Tree-model entries updated to extended matrix.
        augmenter_            Fitted :class:`ConditionalPanelAugmenter`.
        """
        if not getattr(self._config, 'use_data_augmentation', False):
            return

        if (
            self.X_train_tree_ is None
            or self._entity_indices_ is None
            or self._year_labels_arr_ is None
        ):
            logger.info("Stage 2b: Augmentation skipped — stage2 not complete.")
            return

        logger.info(
            "Stage 2b: Conditional panel augmentation (Gaussian copula + "
            "VAR(1))..."
        )

        y_arr = self.y_train_.values
        threshold = float(
            getattr(self._config, 'augment_gain_threshold', 0.005)
        )

        augmenter = ConditionalPanelAugmenter(
            gain_threshold=threshold,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # fit_augment_if_beneficial runs 5-fold walk-forward CV on the
        # tree-track matrix (consistent with how base models see features).
        X_aug, y_aug, ent_aug, yr_aug, committed = (
            augmenter.fit_augment_if_beneficial(
                X=self.X_train_tree_,
                y=y_arr,
                entity_indices=self._entity_indices_,
                year_labels=self._year_labels_arr_,
            )
        )

        self._augmenter_ = augmenter

        if not committed:
            if self.verbose:
                print(
                    f"    Augmentation rejected: ΔR² below threshold "
                    f"({threshold:.4f})"
                )
            return

        if self.verbose:
            n_synth = len(X_aug) - len(self.X_train_tree_)
            print(
                f"    Augmentation committed: +{n_synth} synthetic rows "
                f"({len(X_aug)} total)"
            )

        # Extend tree-track arrays only (PLS/Bayesian track unchanged)
        self.X_train_tree_ = X_aug
        self._entity_indices_ = ent_aug
        self._year_labels_arr_ = yr_aug
        # y_aug is already in transformed target space because fit_augment
        # receives self.y_train_.values (transformed); augmenter synthesises
        # entity means of transformed values → no further transformation needed.
        _aug_index = pd.RangeIndex(len(y_aug))
        # Rebuild y_train_ DataFrame with augmented rows
        self.y_train_ = pd.DataFrame(
            y_aug,
            index=_aug_index,
            columns=self.y_train_.columns,
        )
        # Also update X_train_ to stay consistent (using NaN-extended index)
        # NOTE: Only tree models use the extended data; PLS models continue
        # to use their own unreduced per_model_X_train_ entry.
        self._per_model_X_train_['CatBoost'] = self.X_train_tree_
        self._per_model_X_train_['LightGBM']         = self.X_train_tree_
        self._per_model_X_train_['QuantileRF']        = self.X_train_tree_
        self._per_model_X_train_['PanelVAR']          = self.X_train_tree_
        self._per_model_X_train_['NAM']               = self.X_train_tree_

    def stage3_fit_base_models(self) -> None:
        """Stage 3: Create base models and train the Super Learner ensemble.

        Creates the six base forecasters (CatBoost, LightGBM, BayesianRidge,
        QuantileRF, PanelVAR, NAM) and delegates training to
        ``SuperLearner.fit()``, which
        executes three sub-steps atomically:

        1. Panel-aware walk-forward CV → per-fold out-of-fold (OOF) ensemble
           predictions for the meta-learner.
        2. NNLS-constrained meta-learner fitted on the stacked OOF predictions
           → optimal non-negative weights that sum to 1.
        3. Full re-fit of every base model on the **complete** training set
           (no holdout rows included — zero leakage).

        The OOF ensemble predictions are cached in ``oof_predictions_`` for
        use by Stage 5 conformal calibration (residuals without refitting)
        and Stage 6 evaluation (genuinely held-out performance estimate).

        Pre-requisite: :meth:`stage2_reduce_features` must have been called.

        Outputs stored on ``self``
        -------------------------
        models_           Dict ``name → unfitted model template``.
                          Fitted versions live in
                          ``super_learner_._fitted_base_models``.
        super_learner_    Fitted :class:`SuperLearner` ensemble.
        oof_predictions_  OOF ensemble predictions
                          ``(n_train, n_components)`` array or ``None``.
        _cv_scores_       Per-model cross-validation R² score lists.
        """
        logger.info("Stage 3: Training Meta-Learner ensemble...")

        # Phase 4: conditionally run one-time Optuna HP search for GB models
        if self._config is not None and getattr(self._config, 'auto_tune_gb', False):
            self._tune_gb_hyperparameters()

        self.models_ = self._create_models()
        logger.info(f"  {len(self.models_)} base models created:")
        for name in self.models_:
            logger.debug(f"    - {name}")

        y_arr = self.y_train_.values
        _n_train = len(self.X_train_tree_)

        # ── Build CV dataset: append holdout year so the last fold can
        # validate it (Fold 5: train 2011–2023, validate 2024).
        # Base models are still retrained on training-only data (refit_X)
        # so Stage 6a holdout evaluation remains leakage-free.
        if (
            not self.X_holdout_.empty
            and self.holdout_year_ is not None
            and self._year_labels_arr_ is not None
        ):
            _X_ho_arr  = self.X_holdout_.values
            _X_ho_tree = self.reducer_tree_.transform(_X_ho_arr)
            _X_ho_pca  = self.reducer_pca_.transform(_X_ho_arr)
            _ho_year_labels = np.full(
                len(_X_ho_arr), self.holdout_year_, dtype=int
            )
            _ent_to_idx = {
                e: i for i, e in enumerate(self._panel_data_.provinces)
            }
            _ho_entity_idx = np.array(
                [_ent_to_idx.get(e, 0) for e in self.X_holdout_.index],
                dtype=int,
            )
            _X_cv_tree  = np.vstack([self.X_train_tree_, _X_ho_tree])
            _X_cv_pca   = np.vstack([self.X_train_pca_,  _X_ho_pca])
            _y_cv       = np.vstack([y_arr, self.y_holdout_.values])
            _year_cv    = np.concatenate(
                [self._year_labels_arr_, _ho_year_labels]
            )
            _ent_cv     = (
                np.concatenate([self._entity_indices_, _ho_entity_idx])
                if self._entity_indices_ is not None else None
            )
            _per_model_cv = {
                'BayesianRidge':    _X_cv_pca,
                'CatBoost': _X_cv_tree,
                'LightGBM':         _X_cv_tree,
                'QuantileRF':       _X_cv_tree,
                'PanelVAR':         _X_cv_tree,
                'NAM':              _X_cv_tree,
            }
        else:
            _X_cv_tree    = self.X_train_tree_
            _y_cv         = y_arr
            _year_cv      = self._year_labels_arr_
            _ent_cv       = self._entity_indices_
            _per_model_cv = self._per_model_X_train_

        self.super_learner_ = SuperLearner(
            base_models=self.models_,
            meta_learner_type=getattr(
                self._config, 'meta_learner_type', 'ridge'
            ),
            n_cv_folds=self.cv_folds,
            cv_min_train_years=self.cv_min_train_years,
            conformal_min_train_years=getattr(
                self._config, 'cv_conformal_min_train_years', 3
            ),
            positive_weights=True,
            normalize_weights=True,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # ── E-01: Build fold-correction callable ──────────────────────────
        # Returns a function (model_name, X_fold, train_idx, fold_entity_indices)
        # → X_corrected that patches entity-demeaned feature columns in the
        # tree-track matrices to use fold-restricted entity means instead of
        # global (all-years) means. Eliminates look-ahead bias from the
        # _demeaned and _demeaned_momentum feature blocks in early CV folds.
        _fold_correction_fn = _build_fold_correction_fn(
            feature_engineer=self.feature_engineer_,
            reducer_tree=self.reducer_tree_,
            year_labels_arr=_year_cv,   # CV dataset year labels (includes holdout when appended)
            entity_indices_arr=_ent_cv,
            per_model_tree_names={
                'CatBoost', 'LightGBM', 'QuantileRF', 'PanelVAR', 'NAM'
            },
        )
        # ── E-08: Create shift detector when enabled ──────────────────────────
        # PanelCovariateShiftDetector is passed to SuperLearner.fit() which
        # runs per-fold MMD² and computes importance weights for shifted folds.
        _shift_det = None
        if getattr(self._config, 'use_shift_detection', False):
            _shift_det = PanelCovariateShiftDetector(
                alpha=0.05,
                n_bootstrap=200,
                max_weight_ratio=10.0,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            self._shift_detector_ = _shift_det
            logger.info("  Covariate shift detection enabled (E-08, MMD² α=0.05)")

        self.super_learner_.fit(
            _X_cv_tree,
            _y_cv,
            entity_indices=_ent_cv,
            per_model_X=_per_model_cv,
            year_labels=_year_cv,
            refit_X=self.X_train_tree_,
            refit_y=y_arr,
            refit_per_model_X=self._per_model_X_train_,
            refit_entity_indices=self._entity_indices_,
            fold_correction_fn=_fold_correction_fn,
            shift_detector=_shift_det,
        )

        # Trim OOF arrays to n_train rows so Stage 5 (conformal calibration)
        # and Stage 6b (OOF R²) index correctly against y_train_.
        _oof_full  = self.super_learner_._oof_ensemble_predictions_
        _mask_full = self.super_learner_._oof_valid_mask_
        if _oof_full is not None and len(_oof_full) > _n_train:
            self.super_learner_._oof_ensemble_predictions_ = _oof_full[:_n_train]
            self.super_learner_._oof_valid_mask_ = (
                _mask_full[:_n_train] if _mask_full is not None else None
            )

        # Cache OOF predictions (Stage 5 conformal uses them for residuals)
        self.oof_predictions_ = self.super_learner_._oof_ensemble_predictions_
        self._cv_scores_ = self.super_learner_.get_cv_scores()

        # E-02: cache extended conformal OOF residuals when available.
        # These cover ALL training years (not just primary CV window) and
        # are passed to stage5 for conformal calibration.
        self._oof_conformal_residuals_ = (
            self.super_learner_._oof_conformal_residuals_
        )

        logger.info("  Meta-Learner fitted with CV scores computed")

    def stage3b_incremental_update(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        entity_indices_new: Optional[np.ndarray] = None,
        X_all: Optional[np.ndarray] = None,
        y_all: Optional[np.ndarray] = None,
        entity_indices_all: Optional[np.ndarray] = None,
    ) -> None:
        """Stage 3b (E-10): Incremental ensemble update for newly available data.

        When new observations become available (e.g., 2024 data arrives after
        the 2024→2025 pipeline was already fitted), this method updates base
        models efficiently without repeating the full Stage 3 pipeline:

        * **CatBoost** — gradient continuation via ``init_model=`` (50 extra
          rounds at ``lr × 0.5``).
        * **LightGBM** — warm-start via ``warm_start=True`` and
          ``n_estimators += increment``.
        * **PanelVAR (Ridge)** — Recursive Least Squares update with
          forgetting factor λ (default 0.95) — tracks structural shifts.
        * **All other models** — full retrain on ``X_all / y_all`` (or
          ``X_new / y_new`` if historical data is not supplied).

        Meta-weights are re-calibrated on new predictions via NNLS and
        γ-blended with the previous weights (``w = (1-γ)·w_prev + γ·w_new``).

        Pre-requisite: :meth:`stage3_fit_base_models` must have been called.

        Parameters
        ----------
        X_new : ndarray, shape (n_new, n_features)
            New observation feature matrix (tree-track, threshold-only).
        y_new : ndarray, shape (n_new, n_outputs)
            New observation target values.
        entity_indices_new : ndarray, shape (n_new,), optional
            Entity IDs for the new rows.
        X_all : ndarray, shape (n_total, n_features), optional
            Full historical + new feature matrix, used for full-retrain fallback.
            When ``None``, full-retrain models use ``X_new`` only.
        y_all : ndarray, shape (n_total, n_outputs), optional
            Full historical + new targets.  When ``None``, full-retrain uses
            ``y_new`` only.
        entity_indices_all : ndarray, optional
            Entity IDs for the full dataset.

        Outputs modified on ``self``
        ----------------------------
        super_learner_    Deep-copied, updated ensemble (base models +
                          re-calibrated meta-weights).
        _incremental_updater_  Fitted :class:`IncrementalEnsembleUpdater`.
        """
        if not getattr(self._config, 'use_incremental_update', False):
            return
        if self.super_learner_ is None:
            raise RuntimeError(
                "stage3b_incremental_update() requires a fitted SuperLearner. "
                "Call stage3_fit_base_models() first."
            )
        logger.info("Stage 3b: Incremental ensemble update (E-10)...")

        strategy = str(
            getattr(self._config, 'incremental_update_strategy', 'auto')
        )
        gamma = float(
            getattr(self._config, 'incremental_update_gamma', 0.3)
        )

        updater = IncrementalEnsembleUpdater(
            strategy=strategy,
            gamma=gamma,
            verbose=self.verbose,
        )
        self._incremental_updater_ = updater

        # per_model_X_new: new rows per model (both tracks use tree-track here)
        _per_model_X_new = {
            'BayesianRidge':    X_new,
            'CatBoost': X_new,
            'LightGBM':         X_new,
            'QuantileRF':       X_new,
            'PanelVAR':         X_new,
            'NAM':              X_new,
        }
        _per_model_X_all = None
        if X_all is not None:
            _per_model_X_all = {
                'BayesianRidge':    X_all,
                'CatBoost': X_all,
                'LightGBM':         X_all,
                'QuantileRF':       X_all,
                'PanelVAR':         X_all,
                'NAM':              X_all,
            }

        self.super_learner_ = updater.update(
            ensemble=self.super_learner_,
            X_new=X_new,
            y_new=y_new,
            entity_indices_new=entity_indices_new,
            X_all=X_all,
            y_all=y_all,
            entity_indices_all=entity_indices_all,
            per_model_X_new=_per_model_X_new,
            per_model_X_all=_per_model_X_all,
        )

        if self.verbose:
            logger.info("  Incremental update complete — meta-weights re-calibrated.")

    def stage4_fit_meta_learner(self) -> None:
        """Stage 4: Extract meta-weights and generate ensemble predictions.

        Retrieves the NNLS meta-weights from the fitted SuperLearner, then
        runs the ensemble forward pass over the prediction year to produce
        point predictions and epistemic uncertainty estimates.  Also
        constructs conservative Gaussian fallback prediction intervals that
        Stage 5 will replace with statistically tighter estimates.

        The **fallback intervals** are calibrated to each criterion's empirical
        training-set standard deviation::

            half_width_j = z_{1-α/2} × σ̂_j,   j = 1 … D

        where σ̂_j is computed from ``y_train_`` (ddof=1) and z_{1-α/2} is
        the Gaussian critical value at ``conformal_alpha``.  This is the
        worst-case homoscedastic baseline; Stage 5 always replaces it.

        Pre-requisite: :meth:`stage3_fit_base_models` must have been called.

        Outputs stored on ``self``
        -------------------------
        model_weights_      Meta-ensemble weights ``name → float``.
        _predictions_arr_   Raw point predictions ``(n_entities, n_components)``.
        _uncertainty_arr_   Epistemic uncertainty (same shape).
        _pred_df_           Predictions as a labelled DataFrame.
        _unc_df_            Uncertainty as a labelled DataFrame.
        _intervals_         Dict ``'lower'/'upper'`` with fallback Gaussian
                            intervals (DataFrames matching ``_pred_df_``).
        """
        logger.info("Stage 4: Extracting meta-weights and generating predictions...")

        self.model_weights_ = self.super_learner_.get_meta_weights()
        logger.debug(f"  Meta-weights: {self.model_weights_}")

        predictions_arr, uncertainty_arr = (
            self.super_learner_.predict_with_uncertainty(
                self.X_pred_tree_,
                entity_indices=self._pred_entity_indices_,
                per_model_X_pred=self._per_model_X_pred_,
            )
        )

        # Normalise to 2-D (n_entities, n_components)
        if predictions_arr.ndim == 1:
            predictions_arr = predictions_arr.reshape(-1, 1)
        if uncertainty_arr.ndim == 1:
            uncertainty_arr = uncertainty_arr.reshape(-1, 1)

        self._predictions_arr_ = predictions_arr
        self._uncertainty_arr_ = uncertainty_arr

        n_components = self.y_train_.shape[1]
        self._pred_df_ = pd.DataFrame(
            predictions_arr[:, :n_components]
            if predictions_arr.shape[1] >= n_components
            else np.column_stack([predictions_arr] * n_components)[:, :n_components],
            index=self.X_pred_.index,
            columns=self.y_train_.columns,
        )
        self._unc_df_ = pd.DataFrame(
            uncertainty_arr[:, :n_components]
            if uncertainty_arr.shape[1] >= n_components
            else np.column_stack([uncertainty_arr] * n_components)[:, :n_components],
            index=self.X_pred_.index,
            columns=self.y_train_.columns,
        )

        # ── Conservative Gaussian fallback intervals ────────────────────
        # Replaced by proper QRF / conformal intervals in stage5.
        # Why training-target SD (not model uncertainty): epistemic uncertainty
        # (weighted SD across base model predictions) conflates model spread
        # with predictive spread; calibrating to the target SD is the
        # distribution-free baseline for Gaussian interval coverage.
        try:
            from scipy.stats import norm as _norm
            _z = float(_norm.ppf(1.0 - self.conformal_alpha / 2))
        except ImportError:
            _z = {0.01: 2.576, 0.05: 1.960, 0.10: 1.645}.get(
                round(self.conformal_alpha, 4), 1.960
            )
        _y_col_stds = np.std(self.y_train_.values, axis=0, ddof=1)
        _fallback_hw = pd.DataFrame(
            np.tile(_y_col_stds * _z, (len(self.X_pred_), 1)),
            index=self.X_pred_.index,
            columns=self.y_train_.columns,
        )
        self._intervals_ = {
            'lower': self._pred_df_ - _fallback_hw,
            'upper': self._pred_df_ + _fallback_hw,
        }

    def _inverse_transform_pipeline_outputs(self) -> None:
        """Phase 5: Inverse-transform all pipeline outputs to original space.

        Called at the end of :meth:`stage5_compute_intervals` once conformal
        intervals are finalised (all quantities are in transformed space at
        that point).  Applies ``target_transformer_.inverse_transform`` to:

        * ``_predictions_arr_`` — raw prediction array (used by stage6a)
        * ``_pred_df_``         — prediction DataFrame (labels, reports)
        * ``prediction_intervals_['lower'/'upper']``  — conformal bounds
        * ``_intervals_['lower'/'upper']``            — fallback Gaussian bounds

        ``_unc_df_`` (epistemic uncertainty) is intentionally left in
        transformed space — it is a relative, comparative quantity and its
        absolute magnitude has no natural interpretation in original space.
        """
        if (
            self.target_transformer_ is None
            or self.target_transformer_.is_identity
        ):
            return

        tt = self.target_transformer_

        def _inv_df(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            return pd.DataFrame(
                tt.inverse_transform(df.values),
                index=df.index,
                columns=df.columns,
            )

        if self._predictions_arr_ is not None:
            self._predictions_arr_ = tt.inverse_transform(
                self._predictions_arr_
            )
        self._pred_df_ = _inv_df(self._pred_df_)
        for _store in (self.prediction_intervals_, self._intervals_):
            for k in ('lower', 'upper'):
                if k in _store:
                    _store[k] = _inv_df(_store[k])

    def stage5_compute_intervals(self) -> None:
        """Stage 5: Compute prediction intervals.

        Supports two modes governed by ``self.uncertainty_method``:

        **'qrf_quantile'** (production default)
            Heteroscedastic leaf-quantile intervals from the fitted
            QuantileRF base model.  Leaf-set quantiles map each entity's
            feature vector to the empirical distribution of training labels
            in its leaf nodes — volatile entities receive wider bands, stable
            ones narrower.

            Bonferroni-corrected per-component quantiles::

                lower_q = α / (2D),   upper_q = 1 − α / (2D)

            guarantee *joint* coverage ≥ 1 − α across all D criteria
            simultaneously (union bound).

        **'conformal'** (fallback)
            Distribution-free conformal intervals calibrated from the OOF
            residuals cached by Stage 3.  Homoscedastic (constant width per
            criterion), marginal coverage guarantee.  Also used automatically
            when the QRF model is absent or raises an exception.

        Pre-requisite: :meth:`stage4_fit_meta_learner` must have been called.

        Outputs stored on ``self``
        -------------------------
        prediction_intervals_  Dict ``'lower'/'upper'`` with refined intervals.
        conformal_predictors_  Dict ``col → ConformalPredictor`` (conformal path).
        conformal_predictor_   Single predictor reference (backward-compat).
        """
        n_components = self.y_train_.shape[1]
        component_cols = self.y_train_.columns.tolist()
        alpha_bonferroni = self.conformal_alpha / max(n_components, 1)

        # Start from a copy of the fallback intervals built in stage4;
        # overwrite with QRF or conformal estimates below.
        intervals: Dict[str, pd.DataFrame] = {
            'lower': self._intervals_['lower'].copy(),
            'upper': self._intervals_['upper'].copy(),
        }

        self.conformal_predictors_ = {}
        self.conformal_predictor_ = None
        _use_conformal = (self.uncertainty_method != 'qrf_quantile')

        if self.uncertainty_method == 'qrf_quantile':
            # ── QRF heteroscedastic path ──────────────────────────────────
            # Per-component Bonferroni quantiles:
            #   lower_q = α_bonf / 2,  upper_q = 1 − α_bonf / 2
            # With α=0.05, D=8: lower_q≈0.003125, upper_q≈0.996875.
            lower_q = alpha_bonferroni / 2.0
            upper_q = 1.0 - alpha_bonferroni / 2.0

            if self.verbose:
                print(
                    f"  Stage 5: QRF heteroscedastic intervals "
                    f"(α_per_crit={alpha_bonferroni:.5f}, "
                    f"q=[{lower_q:.6f}, {upper_q:.6f}])"
                )

            _qrf_model = (
                self.super_learner_._fitted_base_models.get('QuantileRF')
                if self.super_learner_ is not None else None
            )
            _qrf_ok = False
            if _qrf_model is not None and hasattr(_qrf_model, 'predict_intervals'):
                try:
                    lower_arr, upper_arr = _qrf_model.predict_intervals(
                        self.X_pred_tree_, lower_q=lower_q, upper_q=upper_q
                    )
                    if lower_arr.shape[1] > n_components:
                        lower_arr = lower_arr[:, :n_components]
                        upper_arr = upper_arr[:, :n_components]
                    intervals['lower'] = pd.DataFrame(
                        lower_arr,
                        index=self.X_pred_.index,
                        columns=self.y_train_.columns,
                    )
                    intervals['upper'] = pd.DataFrame(
                        upper_arr,
                        index=self.X_pred_.index,
                        columns=self.y_train_.columns,
                    )
                    _qrf_ok = True
                    if self.verbose:
                        widths = upper_arr - lower_arr
                        print(
                            f"    Interval widths: mean={widths.mean():.4f}, "
                            f"std={widths.std():.4f}, "
                            f"min={widths.min():.4f}, max={widths.max():.4f} "
                            f"(heteroscedastic)"
                        )
                        print(
                            f"    Joint coverage guarantee: "
                            f"{(1 - self.conformal_alpha) * 100:.0f}% "
                            f"(Bonferroni across {n_components} criteria)"
                        )
                except Exception as _qrf_exc:
                    if self.verbose:
                        print(
                            f"    QRF interval estimation failed "
                            f"({type(_qrf_exc).__name__}: {_qrf_exc}); "
                            f"falling back to conformal prediction."
                        )
            else:
                if self.verbose:
                    print(
                        "    QuantileRF model not available; "
                        "falling back to conformal prediction."
                    )

            if not _qrf_ok:
                _use_conformal = True

        if _use_conformal:
            # ── Conformal prediction path ─────────────────────────────────
            logger.info("Stage 5: Per-component conformal prediction calibration...")

            class _SingleOutputWrapper:
                """Wrap a multi-output model to expose a single column."""
                def __init__(self, model, col_index: int):
                    self._model = model
                    self._col = col_index

                def predict(self, X: np.ndarray) -> np.ndarray:
                    pred = self._model.predict(X)
                    if pred.ndim == 1:
                        return pred
                    return pred[:, self._col]

                def __getattr__(self, name: str):
                    # Guard against infinite recursion during copy.deepcopy:
                    # deepcopy creates a blank instance before populating
                    # __dict__; accessing self._model re-enters __getattr__.
                    if name in ("_model", "_col"):
                        raise AttributeError(name)
                    return getattr(self._model, name)

            y_arr = self.y_train_.values
            try:
                # U-2: calibrate from pre-computed OOF residuals (no deep-copy
                # of the full SuperLearner ensemble per component).
                sl = self.super_learner_
                _has_oof = (
                    hasattr(sl, "_oof_ensemble_predictions_")
                    and sl._oof_ensemble_predictions_ is not None
                    and hasattr(sl, "_oof_valid_mask_")
                    and sl._oof_valid_mask_ is not None
                    and int(sl._oof_valid_mask_.sum()) >= 3
                )

                for d, col in enumerate(component_cols):
                    wrapper = _SingleOutputWrapper(self.super_learner_, d)
                    y_col = y_arr[:, d] if y_arr.ndim > 1 else y_arr

                    cp = ConformalPredictor(
                        method=self.conformal_method,
                        alpha=alpha_bonferroni,
                        random_state=self.random_state,
                    )

                    if _has_oof:
                        valid = sl._oof_valid_mask_
                        oof_pred_d = sl._oof_ensemble_predictions_[valid, d]

                        # E-02: prefer extended conformal residuals when they
                        # cover more training years than the primary OOF.
                        # ``_oof_conformal_residuals_`` already encodes
                        # (y − ŷ) across the combined primary + secondary sweep.
                        # We extract the d-th output column for this component
                        # and pass the (n_valid,) residual vector directly to
                        # calibrate_residuals — no y_col indexing needed.
                        _ext_residuals = getattr(
                            self, '_oof_conformal_residuals_', None
                        )
                        if (
                            _ext_residuals is not None
                            and _ext_residuals.ndim > 1
                            and d < _ext_residuals.shape[1]
                        ):
                            # Use extended residuals (more cal points, E-02)
                            _d_residuals = _ext_residuals[:, d]
                            _d_residuals = _d_residuals[~np.isnan(_d_residuals)]
                            if len(_d_residuals) >= 5:
                                cp.calibrate_residuals(_d_residuals, base_model=wrapper)
                            else:
                                # Fallback to primary OOF residuals
                                oof_residuals = y_col[valid] - oof_pred_d
                                cp.calibrate_residuals(oof_residuals, base_model=wrapper)
                        else:
                            # Primary OOF residuals only (original U-2 path)
                            oof_residuals = y_col[valid] - oof_pred_d
                            cp.calibrate_residuals(oof_residuals, base_model=wrapper)
                    else:
                        # Fallback: re-calibrate via cv_plus (panel-aware, E-04)
                        cp.calibrate(
                            wrapper, self.X_train_tree_, y_col,
                            cv_folds=self.cv_folds,
                            year_labels=self._year_labels_arr_,
                            entity_indices=self._entity_indices_,
                        )

                    point_d = self._pred_df_[col].values
                    lower_d, upper_d = cp.predict_intervals(
                        self.X_pred_tree_, point_predictions=point_d
                    )
                    intervals['lower'][col] = lower_d
                    intervals['upper'][col] = upper_d
                    self.conformal_predictors_[col] = cp

                self.conformal_predictor_ = next(
                    iter(self.conformal_predictors_.values()), None
                )

                if self.verbose:
                    widths = [
                        cp.get_interval_width()
                        for cp in self.conformal_predictors_.values()
                    ]
                    print(
                        f"    Per-component widths: "
                        f"min={min(widths):.4f}, max={max(widths):.4f}"
                    )
                    print(
                        f"    Bonferroni α/D = {alpha_bonferroni:.5f} "
                        f"(D={n_components})"
                    )
                    print(
                        f"    Joint coverage guarantee: "
                        f"{(1 - self.conformal_alpha) * 100:.0f}%"
                    )

            except Exception as e:
                logger.warning(f"Conformal calibration failed: {e}")
                logger.warning("Using standard Gaussian intervals as fallback.")
                self.conformal_predictor_ = None

        self.prediction_intervals_ = intervals
        # Phase 5: inverse-transform all outputs from transformed → original space
        self._inverse_transform_pipeline_outputs()

    def stage6_evaluate_all(self) -> None:
        """Stage 6: Holdout model comparison and cross-validation evaluation.

        Executes two complementary evaluation passes:

        **6a — Genuine holdout comparison**: Applies every fitted base model
        and the SuperLearner ensemble to the withheld calendar year
        (``holdout_year_``).  Zero refitting — all models were trained on
        data strictly before the holdout year in Stage 3, so this is a true
        OOS test.  Results are stored in ``model_comparison_``.

        **6b — OOF cross-validation estimate**: Reads the SuperLearner's
        cached OOF ensemble predictions (generated during the CV loop in
        Stage 3) to compute aggregate R²/RMSE/MAE across all held-out
        folds.  Primary diagnostic for ensemble calibration quality.
        Stored in ``holdout_performance_``.

        Also aggregates per-component feature importance
        (``_feature_importance_``) and packages the training-info dict
        (``_training_info_``) consumed by :meth:`_assemble_result`.

        Pre-requisite: :meth:`stage4_fit_meta_learner` must have been called.

        Outputs stored on ``self``
        -------------------------
        model_comparison_     List of :class:`ModelComparisonResult` or None.
        holdout_performance_  OOF performance dict or None.
        _feature_importance_  Feature-importance DataFrame.
        _model_performance_   Per-model CV R² dict.
        _training_info_       Training-info dict for UnifiedForecastResult.
        """
        logger.info("Stage 6: Evaluation and feature importance...")

        n_comps = self.y_train_.shape[1]
        y_arr = self.y_train_.values

        # ── Stage 6a: Genuine holdout model comparison ────────────────────
        self.model_comparison_ = None
        if not self.X_holdout_.empty and len(self.y_holdout_) > 0:
            logger.info("  Stage 6a: Evaluating all models on genuine holdout set...")
            try:
                _X_ho_arr  = self.X_holdout_.values
                _X_ho_pca  = self.reducer_pca_.transform(_X_ho_arr)
                _X_ho_tree = self.reducer_tree_.transform(_X_ho_arr)
                _per_model_X_holdout = {
                    'BayesianRidge':    _X_ho_pca,
                    'CatBoost': _X_ho_tree,
                    'LightGBM':         _X_ho_tree,
                    'QuantileRF':       _X_ho_tree,
                    'PanelVAR':         _X_ho_tree,
                    'NAM':              _X_ho_tree,
                }
                try:
                    _ens_ho_arr, _ = self.super_learner_.predict_with_uncertainty(
                        _X_ho_tree,
                        per_model_X_pred=_per_model_X_holdout,
                    )
                except Exception as _ens_ho_exc:
                    logger.warning(
                        f"Ensemble holdout inference failed: {_ens_ho_exc}"
                    )
                    _ens_ho_arr = np.full(
                        (len(_X_ho_arr), n_comps), np.nan
                    )
                self.model_comparison_ = compare_all_models(
                    fitted_base_models=self.super_learner_._fitted_base_models,
                    super_learner=self.super_learner_,
                    X_holdout_per_model=_per_model_X_holdout,
                    y_holdout=self.y_holdout_.values,
                    ensemble_preds_holdout=_ens_ho_arr,
                    X_target_per_model=self._per_model_X_pred_,
                    ensemble_preds_target=self._predictions_arr_,
                    component_names=self.y_train_.columns.tolist(),
                    target_entities=list(self.X_pred_.index),
                )
                if self.model_comparison_:
                    _best_mc = next(
                        (r for r in self.model_comparison_ if r.is_best), None
                    )
                    _ens_mc = next(
                        (r for r in self.model_comparison_
                         if r.model_name == 'Ensemble'), None
                    )
                    _base_mc = next(
                        (r for r in self.model_comparison_
                         if r.model_name != 'Ensemble'
                         and not np.isnan(r.holdout_r2)), None
                    )
                    if _best_mc and _ens_mc and _base_mc:
                        if _best_mc.model_name == 'Ensemble':
                            logger.info(
                                f"  Best model: Ensemble "
                                f"(R²={_best_mc.holdout_r2:.4f}) outperforms "
                                f"best base model {_base_mc.model_name} "
                                f"(R²={_base_mc.holdout_r2:.4f})"
                            )
                        else:
                            logger.info(
                                f"  Best model: {_best_mc.model_name} "
                                f"(R²={_best_mc.holdout_r2:.4f}) outperforms "
                                f"Ensemble (R²={_ens_mc.holdout_r2:.4f})"
                            )
            except Exception as _cmp_exc:
                logger.warning(
                    f"Stage 6a failed: {type(_cmp_exc).__name__}: {_cmp_exc}"
                )
        else:
            logger.info(
                "Stage 6a: Skipped (no holdout data — "
                "insufficient training history)."
            )

        # ── Stage 6b: OOF cross-validation performance estimate ───────────
        # Uses SuperLearner's cached OOF ensemble predictions (genuinely OOS)
        # rather than refitting or evaluating on training data.
        self.holdout_performance_ = None
        self._holdout_y_test_ = None
        self._holdout_y_pred_ = None
        try:
            _oof_preds = self.super_learner_._oof_ensemble_predictions_
            _oof_mask  = self.super_learner_._oof_valid_mask_
            if (
                _oof_preds is not None
                and _oof_mask is not None
                and _oof_mask.sum() >= 5
            ):
                y_oof      = y_arr[_oof_mask]
                y_oof_pred = _oof_preds[_oof_mask, :y_arr.shape[1]]
                self.holdout_performance_ = {
                    'r2':    float(r2_score(y_oof.ravel(), y_oof_pred.ravel())),
                    'rmse':  float(
                        np.sqrt(mean_squared_error(y_oof, y_oof_pred))
                    ),
                    'mae':   float(mean_absolute_error(y_oof, y_oof_pred)),
                    'n_oof': int(_oof_mask.sum()),
                    'note':  (
                        'OOF cross-validation estimate '
                        '(genuinely out-of-sample)'
                    ),
                }
                self._holdout_y_test_ = y_oof.ravel()
                self._holdout_y_pred_ = y_oof_pred.ravel()
                # Phase 5: inverse-transform OOF estimates to original space
                # for meaningful external reporting (R²/RMSE in target units)
                if (
                    self.target_transformer_ is not None
                    and not self.target_transformer_.is_identity
                ):
                    try:
                        self._holdout_y_test_ = (
                            self.target_transformer_
                            .inverse_transform(y_oof)
                            .ravel()
                        )
                        self._holdout_y_pred_ = (
                            self.target_transformer_
                            .inverse_transform(y_oof_pred)
                            .ravel()
                        )
                    except Exception:
                        pass  # keep transformed-space fallback
                if self.verbose:
                    print(
                        f"    Stage 6b: OOF R² = "
                        f"{self.holdout_performance_['r2']:.4f}, "
                        f"RMSE = {self.holdout_performance_['rmse']:.4f}  "
                        f"[n_oof={self.holdout_performance_['n_oof']}]"
                    )
            else:
                if self.verbose:
                    print(
                        "    Stage 6b: OOF evaluation skipped "
                        "(insufficient OOF samples)"
                    )
        except Exception as e:
            if self.verbose:
                print(
                    f"    Stage 6b: OOF evaluation failed: "
                    f"{type(e).__name__}: {e}"
                )

        # ── Feature importance ────────────────────────────────────────────
        self._feature_importance_ = self._aggregate_feature_importance(
            self.feature_engineer_.get_feature_names(),
            self.y_train_.columns.tolist(),
        )

        # ── Per-model CV performance summary ─────────────────────────────
        self._model_performance_ = {}
        _crit_scores = getattr(
            self.super_learner_, '_cv_scores_per_criterion_', {}
        ) or {}
        for name, scores in self._cv_scores_.items():
            if scores:
                self._model_performance_[name] = {
                    'mean_r2': float(np.nanmean(scores)),
                    'std_r2':  float(np.nanstd(scores)),
                }
                # Phase 4: per-criterion RMSE breakdown from SuperLearner CV
                if name in _crit_scores and _crit_scores[name]:
                    _crit_arr = np.array(_crit_scores[name])
                    if _crit_arr.size > 0:
                        self._model_performance_[name][
                            'per_criterion_rmse_mean'
                        ] = _crit_arr.mean(axis=0).tolist()
                        self._model_performance_[name][
                            'per_criterion_rmse_std'
                        ] = _crit_arr.std(axis=0).tolist()

        # ── Training-info dict (consumed by _assemble_result) ─────────────
        self._training_info_ = {
            'n_samples':  len(self.X_train_),
            'n_features': self.X_train_.shape[1],
            'n_features_pca': (
                self.reducer_pca_.n_components
                if self.reducer_pca_ else self.X_train_.shape[1]
            ),
            'n_features_tree': (
                self.reducer_tree_.n_components
                if self.reducer_tree_ else self.X_train_.shape[1]
            ),
            'pca_variance_retained': (
                self.reducer_pca_.explained_variance_ratio
                if self.reducer_pca_ else 1.0
            ),
            'mode': 'advanced',
            'ensemble_method': 'super_learner',
            'conformal_calibrated': self.conformal_predictor_ is not None,
            'target_level': self.target_level,
            # Phase 5: flag for downstream reporting
            'target_transformed': (
                self.target_transformer_ is not None
                and not self.target_transformer_.is_identity
            ),
            # OOF residuals for downstream visualisation
            'y_test':    self._holdout_y_test_,
            'y_pred':    self._holdout_y_pred_,
            'test_entities': None,
            'per_model_holdout_predictions': None,
        }

        if self.verbose:
            print(
                f"    {len(self.model_weights_)} models combined; "
                f"feature importance aggregated."
            )

    def stage7_postprocess(
        self,
        y_saw_predicted: Optional[pd.DataFrame] = None,
        panel_data: Any = None,
    ) -> None:
        """Stage 7: CRITIC-weighted composite score derivation.

        Applies :class:`CRITICWeightCalculator` to the predicted per-criterion
        SAW-normalised scores to produce a single composite performance score
        per entity.  This mirrors the MCDM pipeline's composite step but uses
        the **forecast-year** CRITIC weights rather than any historical year's
        weights.

        Only executed when ``use_saw_targets=True`` (the production default).
        When ``use_saw_targets=False``, targets are raw criterion values
        without a natural [0, 1] bound and CRITIC weighting is undefined;
        ``composite_predictions_`` is set to ``None`` in that case.

        Parameters
        ----------
        y_saw_predicted : pd.DataFrame, optional
            Predicted SAW scores ``(n_entities, n_criteria)``.  When ``None``
            (default), uses ``self._pred_df_`` set by Stage 4.
        panel_data : Any, optional
            Original panel data.  Not used in the current implementation;
            reserved for future enrichment (e.g., applying province-specific
            benefit/cost criterion metadata).

        Outputs stored on ``self``
        -------------------------
        composite_predictions_  Per-entity CRITIC-weighted composite score.
                                Named ``'composite_score'``.
                                ``None`` when ``use_saw_targets=False`` or
                                derivation fails.
        """
        _pred = y_saw_predicted if y_saw_predicted is not None else self._pred_df_
        self.composite_predictions_ = None

        if not self.use_saw_targets or _pred is None or _pred.empty:
            return

        try:
            self.composite_predictions_ = self._compute_critic_composite(_pred)
            if self.verbose:
                _c = self.composite_predictions_
                print(
                    f"  Stage 7: CRITIC composite from predicted SAW scores — "
                    f"mean={_c.mean():.4f}, std={_c.std():.4f}, "
                    f"range=[{_c.min():.4f}, {_c.max():.4f}]"
                )
        except Exception as exc:
            if self.verbose:
                print(
                    f"  Stage 7: Composite derivation failed "
                    f"({type(exc).__name__}: {exc}). "
                    f"composite_predictions_ will be None."
                )

    def get_stage_outputs(self) -> Dict[str, Any]:
        """Return every intermediate artifact stored by the 7 stage methods.

        All values are the **live** objects held on ``self``; mutating them
        modifies the forecaster's internal state.  Values are ``None`` (or
        empty DataFrame/dict) for attributes whose corresponding stage has
        not yet been executed.

        Returns
        -------
        Dict[str, Any]
            === Stage 1 ===
            ``X_train``         Training feature matrix (DataFrame).
            ``y_train``         Training targets (DataFrame).
            ``X_pred``          Prediction feature matrix (DataFrame).
            ``X_holdout``       Holdout feature matrix (DataFrame or empty).
            ``y_holdout``       Holdout targets (DataFrame or empty).
            ``entity_info``     Row-level metadata (DataFrame).

            === Stage 2 ===
            ``X_train_pca``   PCA-reduced training features (array or None).
            ``X_train_tree``  Threshold-only training features (array or None).
            ``X_pred_pca``    PCA-reduced prediction features (array or None).
            ``X_pred_tree``   Threshold-only prediction features (array or None).
            ``reducer_pca``   Fitted PCA reducer or None.
            ``reducer_tree``  Fitted threshold-only reducer or None.

            === Stage 3 ===
            ``models``         Dict of unfitted base model templates (or {}).
            ``oof_predictions`` OOF ensemble predictions array or None.

            === Stage 4 ===
            ``super_learner``  Fitted SuperLearner or None.
            ``model_weights``  Meta-weight dict (or {}).

            === Stage 5 ===
            ``prediction_intervals``  Dict 'lower'/'upper' DataFrames or {}.

            === Stage 6 ===
            ``model_comparison``    List of ModelComparisonResult or None.
            ``holdout_performance`` OOF performance dict or None.

            === Stage 7 ===
            ``composite_predictions``  Per-entity composite Series or None.
        """
        _g = lambda attr: getattr(self, attr, None)  # noqa: E731
        return {
            # Stage 1
            'X_train':               _g('X_train_'),
            'y_train':               _g('y_train_'),
            'X_pred':                _g('X_pred_'),
            'X_holdout':             _g('X_holdout_'),
            'y_holdout':             _g('y_holdout_'),
            'entity_info':           _g('entity_info_'),
            # Stage 2
            'X_train_pca':           _g('X_train_pca_'),
            'X_train_tree':          _g('X_train_tree_'),
            'X_pred_pca':            _g('X_pred_pca_'),
            'X_pred_tree':           _g('X_pred_tree_'),
            'reducer_pca':           _g('reducer_pca_'),
            'reducer_tree':          _g('reducer_tree_'),
            # Stage 3
            'models':                _g('models_'),
            'oof_predictions':       _g('oof_predictions_'),
            # Stage 4
            'super_learner':         _g('super_learner_'),
            'model_weights':         _g('model_weights_'),
            # Stage 5
            'prediction_intervals':  _g('prediction_intervals_'),
            # Stage 6
            'model_comparison':      _g('model_comparison_'),
            'holdout_performance':   _g('holdout_performance_'),
            # Stage 7
            'composite_predictions': _g('composite_predictions_'),
        }

    def _assemble_result(self) -> UnifiedForecastResult:
        """Build UnifiedForecastResult from stage outputs stored on self.

        Called at the end of :meth:`fit_predict` in ``'full'`` and
        ``'evaluate_only'`` modes.  All component values are read directly
        from stage-output attributes, guaranteeing consistency with the last
        completed pipeline run.
        """
        return UnifiedForecastResult(
            predictions=self._pred_df_,
            uncertainty=self._unc_df_,
            prediction_intervals=self.prediction_intervals_,
            model_contributions=self.model_weights_,
            model_performance=self._model_performance_,
            feature_importance=self._feature_importance_,
            cross_validation_scores=self._cv_scores_,
            holdout_performance=self.holdout_performance_,
            composite_predictions=self.composite_predictions_,
            forecast_criterion_weights=self.forecast_criterion_weights_,
            training_info=self._training_info_,
            data_summary={
                'n_entities':   len(self.X_pred_),
                'n_components': self.y_train_.shape[1],
            },
            best_model_name=(
                next(
                    (r.model_name for r in self.model_comparison_ if r.is_best),
                    None,
                )
                if self.model_comparison_ else None
            ),
            best_model_predictions=(
                next(
                    (r.predictions for r in self.model_comparison_ if r.is_best),
                    None,
                )
                if self.model_comparison_ else None
            ),
            model_comparison=self.model_comparison_,
        )

    @_silence_warnings
    def fit_predict(self,
                   panel_data,
                   target_year: int,
                   weights: Optional[Dict[str, float]] = None
                   ) -> Optional[UnifiedForecastResult]:
        """Fit the 6-model ensemble and forecast ``target_year``.

        Orchestrates the 7 stage methods in sequence.  ``pipeline_mode``
        controls early exit:

        ``'full'`` (default)
            All 7 stages; returns :class:`UnifiedForecastResult`.

        ``'features_only'``
            Stages 1–2 (feature engineering + dimensionality reduction),
            then returns ``None``.  Use :meth:`get_stage_outputs` or
            ``forecaster.X_train_`` / ``forecaster.X_pred_`` to inspect
            the engineered features before committing to model training.

        ``'fit_only'``
            Stages 1–4 (feature engineering → dimensionality reduction →
            base-model training → ensemble predictions), then returns
            ``None``.  Skips interval estimation (Stage 5) and evaluation
            (Stages 6–7).  Combine with a subsequent
            ``fit_predict(..., mode='evaluate_only')`` call (or invoke
            :meth:`stage5_compute_intervals`, :meth:`stage6_evaluate_all`,
            and :meth:`stage7_postprocess` directly).

        ``'evaluate_only'``
            Stages 5–7 only.  Requires that Stages 1–4 have been completed
            by a previous ``fit_predict()`` call with mode ``'full'`` or
            ``'fit_only'``.  Useful for re-running interval estimation or
            evaluation with different settings (e.g., switching
            ``uncertainty_method``) without re-fitting the ensemble.

        Parameters
        ----------
        panel_data : PanelData
            Panel data object.  Ignored in ``'evaluate_only'`` mode.
        target_year : int
            Forecast-horizon year.
        weights : dict, optional
            Reserved for a future API; currently unused.

        Returns
        -------
        UnifiedForecastResult or None
            Full result in ``'full'`` / ``'evaluate_only'`` mode.
            ``None`` in ``'features_only'`` / ``'fit_only'`` mode.

        Notes
        -----
        All intermediate artifacts remain on ``self`` after every mode and
        are accessible via :meth:`get_stage_outputs`.
        """
        _mode = self.pipeline_mode

        logger.info(f"Starting ML Forecasting for {target_year}...")
        if self.verbose:
            logger.debug(f"Pipeline mode: {_mode}")

        # ── evaluate_only: stages 5–7 on an already-fitted forecaster ──────
        if _mode == 'evaluate_only':
            if self.super_learner_ is None or self._pred_df_.empty:
                raise ValueError(
                    "pipeline_mode='evaluate_only' requires Stages 1–4 to be "
                    "completed first.  Call fit_predict() with mode 'full' or "
                    "'fit_only' on this forecaster before switching to "
                    "'evaluate_only'."
                )
            self.stage5_compute_intervals()
            self.stage6_evaluate_all()
            self.stage7_postprocess()
            return self._assemble_result()

        # ── Stages 1–2: Feature engineering + dimensionality reduction ────
        self.stage1_engineer_features(panel_data, target_year)

        if _mode == 'features_only':
            logger.info(
                "Pipeline mode='features_only': stopping after Stage 2 "
                "(feature inspection only — no model fitting)."
            )
            self.stage2_reduce_features()
            return None

        self.stage2_reduce_features()

        # E-06: Conditional synthetic augmentation — applied after feature
        # reduction so the augmenter works on the compressed tree-track
        # features and commits only when it improves 5-fold CV R².
        self.stage2b_augment_data()

        # ── Stages 3–4: Base model training + ensemble predictions ─────────
        self.stage3_fit_base_models()
        self.stage4_fit_meta_learner()

        if _mode == 'fit_only':
            logger.info(
                "Pipeline mode='fit_only': stopping after Stage 4 "
                "(no interval estimation or evaluation)."
            )
            return None

        # ── Stages 5–7: Intervals + evaluation + composite ────────────────
        self.stage5_compute_intervals()
        self.stage6_evaluate_all()
        self.stage7_postprocess()

        logger.info(
            f"Forecasting complete. {len(self.model_weights_)} models combined."
        )

        return self._assemble_result()

    def get_best_model_predictions(self) -> Optional[pd.DataFrame]:
        """
        Return the target-year forecast predictions for the best holdout model.

        The "best" model is the one with the highest R² on the withheld
        calendar year (``holdout_year_``), evaluated during Stage 3b of
        ``fit_predict()``.  Target-year predictions are stored in
        ``ModelComparisonResult.predictions`` (province × criterion).

        Returns
        -------
        pd.DataFrame or None
            Province × criterion predictions for the forecast year from the
            top-performing model.  Index = province names; columns = criterion
            identifiers.  Returns ``None`` when no holdout comparison was
            performed (e.g., insufficient training history or first call
            before ``fit_predict()``).
        """
        if not hasattr(self, 'model_comparison_') or not self.model_comparison_:
            return None
        best = next((r for r in self.model_comparison_ if r.is_best), None)
        return best.predictions if best is not None else None

    def _aggregate_feature_importance(self,
                                     feature_names: List[str],
                                     component_names: List[str]
                                     ) -> pd.DataFrame:
        """Aggregate per-component feature importance across fitted models.

        When a ``PanelFeatureReducer`` is active, base models operate in
        PCA-space.  Importances are first computed in PC-space, then
        mapped back to the original feature names via
        ``inverse_importance()``.
        """
        n_components = len(component_names)

        # Use the FITTED base models from SuperLearner, not the unfitted originals
        fitted_models = (
            self.super_learner_._fitted_base_models
            if self.super_learner_ is not None
            else {}
        )

        if not fitted_models:
            return pd.DataFrame()

        # Model-to-reducer mapping for two-track preprocessing
        _pca_model_names = {'BayesianRidge'}

        # Accumulate importances in the ORIGINAL feature space so that models
        # from different tracks (PCA vs threshold-only) can be averaged
        # meaningfully after per-model inverse mapping.
        matrices_orig: List[np.ndarray] = []
        for name, model in fitted_models.items():
            try:
                # Determine which reducer applies to this model
                if (name in _pca_model_names
                        and getattr(self, 'reducer_pca_', None) is not None
                        and self.reducer_pca_._fitted):
                    reducer = self.reducer_pca_
                elif (getattr(self, 'reducer_tree_', None) is not None
                        and self.reducer_tree_._fitted):
                    reducer = self.reducer_tree_
                elif (getattr(self, 'feature_reducer_', None) is not None
                        and self.feature_reducer_._fitted):
                    reducer = self.feature_reducer_  # backward compat
                else:
                    reducer = None

                n_model_feat = reducer.n_components if reducer is not None else len(feature_names)
                per_out = _get_per_output_importance(model, n_components, n_model_feat)
                if per_out.shape == (n_model_feat, n_components):
                    if reducer is not None:
                        per_out_orig = reducer.inverse_importance(per_out)
                    else:
                        per_out_orig = per_out
                    matrices_orig.append(per_out_orig)
            except Exception:
                pass

        if not matrices_orig:
            return pd.DataFrame()

        # Average in original feature space, then re-normalise each column
        avg = np.mean(matrices_orig, axis=0)  # (n_original_features, n_components)
        col_sums = avg.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        avg /= col_sums

        return pd.DataFrame(avg, index=feature_names, columns=component_names)
