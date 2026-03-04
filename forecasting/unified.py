# -*- coding: utf-8 -*-
"""
Unified Forecasting Orchestrator
================================

State-of-the-art ensemble forecasting system optimised for small-to-medium
panel data (N < 1000).  Orchestrates all forecasting sub-components through
a clean single-entry-point API.

Model ensemble (5 diverse types)
---------------------------------
- Gradient Boosting     — Huber-loss sequential trees
- Bayesian Ridge        — linear model with posterior uncertainty
- Quantile RF           — full predictive distributions via leaf quantiles
- Panel VAR             — LSDV fixed effects + autoregressive dynamics
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
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import copy

from .base import BaseForecaster
from .gradient_boosting import GradientBoostingForecaster
from .bayesian import BayesianForecaster
from .features import TemporalFeatureEngineer

# State-of-the-art advanced models
from .panel_var import PanelVARForecaster
from .quantile_forest import QuantileRandomForestForecaster
from .neural_additive import NeuralAdditiveForecaster
from .super_learner import SuperLearner
from .conformal import ConformalPredictor
from .preprocessing import PanelFeatureReducer
from .evaluation import ForecastEvaluator, AblationStudy
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
    
    # Metadata
    training_info: Dict[str, Any]
    data_summary: Dict[str, Any]
    
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
    # 3. MultiOutputRegressor wrapper (GradientBoostingForecaster,
    #    BayesianForecaster) — model.model.estimators_[col]
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


class UnifiedForecaster:
    """
    State-of-the-art unified forecasting system.

    Optimized for small-to-medium panel data (N < 1000) with statistically-principled
    ensemble design emphasizing model diversity over quantity.

    Tier 1 - Base Models (5 diverse models):
        1. Gradient Boosting (robust tree ensemble)
        2. Bayesian Ridge (linear with uncertainty quantification)
        3. Quantile Random Forest (distributional forecasting)
        4. Panel VAR (panel fixed effects + autoregressive)
        5. Neural Additive Models (interpretable non-linearity)

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
                 random_state: int = 42,
                 verbose: bool = True,
                 config: Optional[ForecastConfig] = None):
        self.conformal_method = conformal_method
        self.conformal_alpha = conformal_alpha
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose
        # ForecastConfig instance for model-level hyperparameters (gb_max_depth,
        # gb_n_estimators, nam_n_basis, nam_n_iterations, pvar_lag_selection_method).
        # When None, _create_models() falls back to hardened production defaults.
        self._config: Optional[ForecastConfig] = config

        self.models_: Dict[str, BaseForecaster] = {}
        self.model_weights_: Dict[str, float] = {}
        self.feature_engineer_ = TemporalFeatureEngineer()
        self.feature_reducer_: Optional[PanelFeatureReducer] = None
        self.super_learner_: Optional[SuperLearner] = None
        self.conformal_predictor_: Optional[ConformalPredictor] = None
        self.evaluator_: Optional[ForecastEvaluator] = None
    
    def _create_models(self) -> Dict[str, BaseForecaster]:
        """
        Create all base model instances (5 diverse models).

        Hyperparameters are resolved from the ForecastConfig passed to
        ``__init__`` (if provided), otherwise production defaults are used.
        All tunable parameters are exposed in ``ForecastConfig`` so they can
        be adjusted without modifying source code.

        Default decisions:
            GradientBoosting : max_depth=5 (32 leaves ≈ 24 samples/leaf at
                               n=756), n_estimators=200 (class default)
            NAM              : n_basis=30 (60 effective params ≈ PCA dims),
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

        # --- Tier 1a: Tree-based -------------------------------------------
        models['GradientBoosting'] = GradientBoostingForecaster(
            n_estimators=gb_n_est, max_depth=gb_depth,
            random_state=self.random_state
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
    
    @_silence_warnings
    def fit_predict(self,
                   panel_data,
                   target_year: int,
                   weights: Optional[Dict[str, float]] = None
                   ) -> UnifiedForecastResult:
        """
        Fit models and make predictions for target year.

        Pipeline:
            1. Temporal feature engineering (lags, rolling stats, momentum, trend)
            2. Train 5 diverse base models (tree, linear, panel-specific)
            3. Super Learner meta-ensemble (automatic optimal weighting)
            4. Conformal prediction calibration (distribution-free intervals)
            5. Return predictions with uncertainty quantification

        Args:
            panel_data: Panel data object with temporal data
            target_year: Year to predict
            weights: Optional pre-specified model weights (overrides Super Learner)

        Returns:
            UnifiedForecastResult with predictions, intervals, and diagnostics
        """
        if self.verbose:
            print(f"Starting state-of-the-art forecasting for {target_year}...")

        # ===== Stage 1: Feature engineering =====
        if self.verbose:
            print("  Stage 1: Engineering temporal features...")

        X_train, y_train, X_pred, entity_info = self.feature_engineer_.fit_transform(
            panel_data, target_year
        )

        # Extract entity indices for models that use panel structure
        if 'entity_index' in entity_info.columns:
            entity_indices = entity_info['entity_index'].values
        else:
            entity_indices = None

        # Compute entity indices for the *prediction* frame.
        # X_pred.index contains province names (pred_entities from fit_transform).
        # Map each to the same integer index used during training so panel-aware
        # base models (PanelVAR) apply the correct fixed-effect and group
        # intercept at prediction time.
        _ent_to_idx = {e: i for i, e in enumerate(panel_data.provinces)}

        # ── B-7 guard: provinces absent from training entity map ──────────────
        # Provinces not in _ent_to_idx fall back to index 0 (reference entity).
        # In practice TemporalFeatureEngineer already excludes such provinces
        # from X_pred, so this should never fire.  If it does, the log entry
        # makes the hidden assumption auditable rather than silent.
        if _ent_to_idx:
            _missing_entities = [
                e for e in X_pred.index if e not in _ent_to_idx
            ]
            if _missing_entities:
                warnings.warn(
                    f"UnifiedForecaster: {len(_missing_entities)} prediction "
                    f"province(s) are absent from the training entity map and "
                    f"will use the reference entity's fixed effects in "
                    f"PanelVARForecaster.  Affected: {_missing_entities[:5]}"
                    f"{'...' if len(_missing_entities) > 5 else ''}. "
                    "Verify TemporalFeatureEngineer exclusion logic.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        pred_entity_indices = np.array(
            [_ent_to_idx.get(e, 0) for e in X_pred.index], dtype=int
        ) if _ent_to_idx else None

        # ===== Stage 2: Create base models =====
        self.models_ = self._create_models()

        if self.verbose:
            print(f"  Stage 2: {len(self.models_)} diverse base models created:")
            for name in self.models_:
                print(f"    - {name}")

        # ===== Stage 2b: Dimensionality reduction =====
        if self.verbose:
            print("  Stage 2b: Reducing feature dimensionality (PCA)...")

        X_arr = X_train.values
        y_arr = y_train.values

        self.feature_reducer_ = PanelFeatureReducer()
        X_arr = self.feature_reducer_.fit_transform(
            X_arr, feature_names=self.feature_engineer_.get_feature_names()
        )
        X_pred_arr = self.feature_reducer_.transform(X_pred.values)

        if self.verbose:
            print(f"    {self.feature_reducer_.get_summary()}")

        # ===== Stage 3: Super Learner meta-ensemble =====
        
        if self.verbose:
            print("  Stage 3: Training Super Learner meta-ensemble...")

        self.super_learner_ = SuperLearner(
            base_models=self.models_,
            meta_learner_type='ridge',
            n_cv_folds=self.cv_folds,
            positive_weights=True,
            normalize_weights=True,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.super_learner_.fit(X_arr, y_arr, entity_indices=entity_indices)

        self.model_weights_ = self.super_learner_.get_meta_weights()
        cv_scores = self.super_learner_.get_cv_scores()

        predictions_arr, uncertainty_arr = (
            self.super_learner_.predict_with_uncertainty(
                X_pred_arr, entity_indices=pred_entity_indices
            )
        )

        # Reshape if needed
        if predictions_arr.ndim == 1:
            predictions_arr = predictions_arr.reshape(-1, 1)
        if uncertainty_arr.ndim == 1:
            uncertainty_arr = uncertainty_arr.reshape(-1, 1)

        # ===== Stage 4: Create result DataFrames =====
        n_components = y_train.shape[1]
        pred_cols = min(predictions_arr.shape[1], n_components)

        pred_df = pd.DataFrame(
            predictions_arr[:, :n_components] if predictions_arr.shape[1] >= n_components
            else np.column_stack([predictions_arr] * n_components)[:, :n_components],
            index=X_pred.index,
            columns=y_train.columns
        )

        unc_df = pd.DataFrame(
            uncertainty_arr[:, :n_components] if uncertainty_arr.shape[1] >= n_components
            else np.column_stack([uncertainty_arr] * n_components)[:, :n_components],
            index=X_pred.index,
            columns=y_train.columns
        )

        # Conservative fallback intervals calibrated to the per-component
        # empirical SD of the training targets (= the naive "predict mean"
        # baseline error), scaled by the z_{1-α/2} Gaussian quantile.
        #
        # Why NOT use 1.96 × unc_df (model disagreement) as was done before:
        #   unc_df is the EPISTEMIC uncertainty (weighted SD across base model
        #   point predictions).  It is NOT the TOTAL predictive SD of Y.
        #   Using it inside a Gaussian interval formula conflates two distinct
        #   quantities and produces intervals with unknown, typically poor,
        #   empirical coverage.
        #
        # This fallback is replaced by proper conformal intervals in Stage 5;
        # it only activates if conformal calibration raises an exception.
        try:
            from scipy.stats import norm as _norm
            _z = float(_norm.ppf(1.0 - self.conformal_alpha / 2))
        except ImportError:
            _z = {0.01: 2.576, 0.05: 1.960, 0.10: 1.645}.get(
                round(self.conformal_alpha, 4), 1.960
            )
        # Per-component training-target SD (shape: n_components)
        _y_col_stds = np.std(y_arr, axis=0, ddof=1)
        _fallback_hw = pd.DataFrame(
            np.tile(_y_col_stds * _z, (len(X_pred), 1)),
            index=X_pred.index,
            columns=y_train.columns,
        )
        intervals = {
            'lower': pred_df - _fallback_hw,
            'upper': pred_df + _fallback_hw,
        }

        # ===== Stage 5: Per-component conformal prediction (Bonferroni) =====
        if self.verbose:
            print("  Stage 4: Per-component conformal prediction calibration...")

        n_components = y_train.shape[1]
        component_cols = y_train.columns.tolist()

        # Bonferroni correction: α_per_component = α / D
        # guarantees joint coverage ≥ 1 − α across all D components.
        alpha_bonferroni = self.conformal_alpha / max(n_components, 1)

        self.conformal_predictors_: Dict[str, ConformalPredictor] = {}

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
            # Forward other attributes (e.g. fit) untouched
            def __getattr__(self, name: str):
                # Guard against infinite recursion during copy.deepcopy:
                # deepcopy creates a blank instance before populating __dict__;
                # accessing self._model before it exists re-enters __getattr__.
                if name in ("_model", "_col"):
                    raise AttributeError(name)
                return getattr(self._model, name)

        try:
            # U-2: avoid deep-copying the full SuperLearner ensemble for every
            # fold × component during cv_plus calibration.
            #
            # SuperLearner.fit() now caches OOF ensemble predictions
            # (_oof_ensemble_predictions_, _oof_valid_mask_).  We calibrate
            # each component's conformal predictor directly from those
            # pre-computed residuals, cutting ~87 SuperLearner deep-copies
            # down to zero.  A single _SingleOutputWrapper is still created
            # per component so that predict_intervals can fall back to
            # model.predict() when point_predictions are not passed (this
            # is a lightweight reference wrapper, not a deep-copy).
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
                    # Fast path: calibrate from pre-computed OOF residuals
                    valid = sl._oof_valid_mask_
                    oof_pred_d = sl._oof_ensemble_predictions_[valid, d]
                    oof_residuals = y_col[valid] - oof_pred_d
                    cp.calibrate_residuals(oof_residuals, base_model=wrapper)
                else:
                    # Fallback: standard calibration (may deep-copy ensemble)
                    cp.calibrate(wrapper, X_arr, y_col, cv_folds=self.cv_folds)

                # Predict intervals for this component
                point_d = pred_df[col].values
                lower_d, upper_d = cp.predict_intervals(
                    X_pred.values, point_predictions=point_d)

                intervals['lower'][col] = lower_d
                intervals['upper'][col] = upper_d
                self.conformal_predictors_[col] = cp

            # Backward-compat: keep a reference accessible as single predictor
            self.conformal_predictor_ = next(
                iter(self.conformal_predictors_.values()), None)

            if self.verbose:
                widths = [cp.get_interval_width()
                          for cp in self.conformal_predictors_.values()]
                print(f"    Per-component widths: "
                      f"min={min(widths):.4f}, max={max(widths):.4f}")
                print(f"    Bonferroni α/D = {alpha_bonferroni:.5f} "
                      f"(D={n_components})")
                print(f"    Joint coverage guarantee: "
                      f"{(1 - self.conformal_alpha) * 100:.0f}%")

        except Exception as e:
            if self.verbose:
                print(f"    Warning: Conformal calibration failed: {e}")
                print("    Using standard Gaussian intervals as fallback.")
            self.conformal_predictor_ = None

        # ===== Stage 6: Aggregate results =====
        # Map PCA-space feature importance back to original feature names
        feature_importance = self._aggregate_feature_importance(
            self.feature_engineer_.get_feature_names(),
            y_train.columns.tolist()
        )

        model_performance = {}
        for name, scores in cv_scores.items():
            if scores:
                model_performance[name] = {
                    'mean_r2': float(np.nanmean(scores)),
                    'std_r2': float(np.nanstd(scores))
                }

        # ===== Stage 6b: Temporal holdout evaluation =====
        holdout_performance = None
        try:
            holdout_year = target_year - 1
            available_years = sorted(panel_data.years)
            if holdout_year in available_years and holdout_year > available_years[2]:
                if self.verbose:
                    print(f"  Stage 6b: Holdout evaluation (year {holdout_year})...")

                # B-5 fix: use a *fresh* TemporalFeatureEngineer instance so that
                # calling fit_transform for holdout_year does not overwrite
                # self.feature_engineer_.feature_names_ (which carries the
                # target-year column list used in _aggregate_feature_importance
                # and by callers inspecting get_feature_names() after fit_predict).
                _ho_eng = TemporalFeatureEngineer()
                X_ho, y_ho, _, _ = _ho_eng.fit_transform(panel_data, holdout_year)

                X_ho_arr = self.feature_reducer_.transform(X_ho.values)
                y_ho_arr = y_ho.values
                y_ho_pred = self.super_learner_.predict(X_ho_arr)
                if y_ho_pred.ndim == 1:
                    y_ho_pred = y_ho_pred.reshape(-1, 1)
                holdout_performance = {
                    'r2': float(r2_score(
                        y_ho_arr.ravel(), y_ho_pred[:, :y_ho_arr.shape[1]].ravel()
                    )),
                    'rmse': float(np.sqrt(mean_squared_error(
                        y_ho_arr, y_ho_pred[:, :y_ho_arr.shape[1]]
                    ))),
                    'mae': float(mean_absolute_error(
                        y_ho_arr, y_ho_pred[:, :y_ho_arr.shape[1]]
                    )),
                }
                if self.verbose:
                    print(f"    Holdout R\u00b2 = {holdout_performance['r2']:.4f}, "
                          f"RMSE = {holdout_performance['rmse']:.4f}")
        except (ValueError, RuntimeError, AttributeError) as e:
            # Catch shape mismatches (feature dim mismatch between holdout and
            # target year), sklearn state errors, and attribute errors from an
            # incompatible holdout year — but let unexpected errors propagate.
            if self.verbose:
                print(f"    Holdout evaluation skipped: {type(e).__name__}: {e}")

        # Build training info
        training_info = {
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'n_features_reduced': self.feature_reducer_.n_components if self.feature_reducer_ else X_train.shape[1],
            'pca_variance_retained': self.feature_reducer_.explained_variance_ratio if self.feature_reducer_ else 1.0,
            'mode': 'advanced',
            'ensemble_method': 'super_learner',
            'conformal_calibrated': self.conformal_predictor_ is not None,
        }

        if self.verbose:
            print(f"  Forecasting complete. {len(self.model_weights_)} models combined.")

        return UnifiedForecastResult(
            predictions=pred_df,
            uncertainty=unc_df,
            prediction_intervals=intervals,
            model_contributions=self.model_weights_,
            model_performance=model_performance,
            feature_importance=feature_importance,
            cross_validation_scores=cv_scores,
            holdout_performance=holdout_performance,
            training_info=training_info,
            data_summary={
                'n_entities': len(X_pred),
                'n_components': y_train.shape[1]
            }
        )

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

        # When PCA is active, models see n_pca features, not n_original
        if self.feature_reducer_ is not None and self.feature_reducer_._fitted:
            n_model_features = self.feature_reducer_.n_components
        else:
            n_model_features = len(feature_names)

        # Accumulate (n_model_features, n_components) matrices from each model
        matrices: List[np.ndarray] = []
        for name, model in fitted_models.items():
            try:
                per_out = _get_per_output_importance(model, n_components, n_model_features)
                if per_out.shape == (n_model_features, n_components):
                    matrices.append(per_out)
            except Exception:
                pass

        if not matrices:
            return pd.DataFrame()

        # Average across models, then re-normalise each component column
        avg = np.mean(matrices, axis=0)  # (n_model_features, n_components)
        col_sums = avg.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        avg /= col_sums

        # Map PCA-space importance back to original features if needed
        if self.feature_reducer_ is not None and self.feature_reducer_._fitted:
            avg = self.feature_reducer_.inverse_importance(avg)
            # Re-normalise after inverse mapping
            col_sums = avg.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1.0
            avg /= col_sums

        return pd.DataFrame(avg, index=feature_names, columns=component_names)
