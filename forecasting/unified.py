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
from .gradient_boosting import CatBoostForecaster
from .bayesian import BayesianForecaster
from .features import TemporalFeatureEngineer

# State-of-the-art advanced models
from .panel_var import PanelVARForecaster
from .quantile_forest import QuantileRandomForestForecaster
from .neural_additive import NeuralAdditiveForecaster
from .super_learner import SuperLearner
from .conformal import ConformalPredictor
from .preprocessing import PanelFeatureReducer
from .evaluation import ForecastEvaluator, AblationStudy, ModelComparisonResult, compare_all_models
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
        gb_backend  = cfg.gb_backend                if cfg is not None else 'catboost'
        nam_n_basis = cfg.nam_n_basis               if cfg is not None else 30
        nam_n_iter  = cfg.nam_n_iterations          if cfg is not None else 10
        pvar_method = cfg.pvar_lag_selection_method if cfg is not None else "cv"

        models = {}

        # --- Tier 1a: Tree-based -------------------------------------------
        models['GradientBoosting'] = CatBoostForecaster(
            iterations=gb_n_est, depth=gb_depth,
            preferred_backend=gb_backend,
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

        # ── Step 3: impute residual NaN with column mean ──────────────────
        if scores_active.isnull().any().any():
            scores_active = scores_active.fillna(scores_active.mean())

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
        if self.verbose:
            print("  Stage 1: Engineering temporal features...")

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

        X_train, y_train, X_pred, entity_info, X_holdout, y_holdout = (
            self.feature_engineer_.fit_transform(
                panel_data,
                target_year,
                use_saw_normalization=self.use_saw_targets,
                holdout_year=_holdout_year,
            )
        )

        # ── Public stage outputs ─────────────────────────────────────────
        self.X_train_ = X_train
        self.y_train_ = y_train
        self.X_pred_ = X_pred
        self.entity_info_ = entity_info
        self.X_holdout_ = X_holdout
        self.y_holdout_ = y_holdout

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

        * **PCA track** (``reducer_pca_``) — retains ≥ 99 % explained
          variance, capped at 30 components.  Used exclusively by
          ``BayesianRidge``; PCA linearity matches the Bayesian linear model
          and provides implicit L2 regularisation via truncation.

        * **Threshold-only track** (``reducer_tree_``) — removes near-zero-
          variance features only.  Used by all non-linear models (CatBoost,
          QRF, PanelVAR, NAM); preserves the original feature structure so
          tree splits and NAM shape functions capture the real interactions.

        Pre-requisite: :meth:`stage1_engineer_features` must have been called.

        Outputs stored on ``self``
        -------------------------
        X_train_pca_     PCA-reduced training features.
        X_train_tree_    Threshold-only training features.
        X_pred_pca_      PCA-reduced prediction features.
        X_pred_tree_     Threshold-only prediction features.
        reducer_pca_     Fitted PCA reducer.
        reducer_tree_    Fitted threshold-only reducer.
        """
        if self.verbose:
            print("  Stage 2: Two-track dimensionality reduction...")

        X_arr = self.X_train_.values
        feature_names = self.feature_engineer_.get_feature_names()

        self.reducer_pca_ = PanelFeatureReducer(
            mode='pca', pca_variance_ratio=0.99, max_components=30
        )
        self.reducer_tree_ = PanelFeatureReducer(mode='threshold_only')

        self.X_train_pca_ = self.reducer_pca_.fit_transform(
            X_arr, feature_names=feature_names
        )
        self.X_pred_pca_ = self.reducer_pca_.transform(self.X_pred_.values)

        self.X_train_tree_ = self.reducer_tree_.fit_transform(
            X_arr, feature_names=feature_names
        )
        self.X_pred_tree_ = self.reducer_tree_.transform(self.X_pred_.values)

        # Per-model routing: BayesianRidge → PCA track; trees/NAM → threshold.
        self._per_model_X_train_ = {
            'BayesianRidge':    self.X_train_pca_,
            'GradientBoosting': self.X_train_tree_,
            'QuantileRF':       self.X_train_tree_,
            'PanelVAR':         self.X_train_tree_,
            'NAM':              self.X_train_tree_,
        }
        self._per_model_X_pred_ = {
            'BayesianRidge':    self.X_pred_pca_,
            'GradientBoosting': self.X_pred_tree_,
            'QuantileRF':       self.X_pred_tree_,
            'PanelVAR':         self.X_pred_tree_,
            'NAM':              self.X_pred_tree_,
        }

        if self.verbose:
            print(f"    PCA track:      {self.reducer_pca_.get_summary()}")
            print(f"    Threshold-only: {self.reducer_tree_.get_summary()}")

    def stage3_fit_base_models(self) -> None:
        """Stage 3: Create base models and train the Super Learner ensemble.

        Creates the five base forecasters (CatBoost, BayesianRidge, QuantileRF,
        PanelVAR, NAM) and delegates training to ``SuperLearner.fit()``, which
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
        if self.verbose:
            print("  Stage 3: Training Super Learner meta-ensemble...")

        self.models_ = self._create_models()
        if self.verbose:
            print(f"    {len(self.models_)} diverse base models:")
            for name in self.models_:
                print(f"    - {name}")

        y_arr = self.y_train_.values
        self.super_learner_ = SuperLearner(
            base_models=self.models_,
            meta_learner_type='ridge',
            n_cv_folds=self.cv_folds,
            positive_weights=True,
            normalize_weights=True,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.super_learner_.fit(
            self.X_train_tree_,
            y_arr,
            entity_indices=self._entity_indices_,
            per_model_X=self._per_model_X_train_,
            year_labels=self._year_labels_arr_,
        )

        # Cache OOF predictions (Stage 5 conformal uses them for residuals)
        self.oof_predictions_ = self.super_learner_._oof_ensemble_predictions_
        self._cv_scores_ = self.super_learner_.get_cv_scores()

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
        if self.verbose:
            print("  Stage 4: Extracting meta-weights and generating predictions...")

        self.model_weights_ = self.super_learner_.get_meta_weights()

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
            if self.verbose:
                print("  Stage 5: Per-component conformal prediction calibration...")

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
                        oof_residuals = y_col[valid] - oof_pred_d
                        cp.calibrate_residuals(oof_residuals, base_model=wrapper)
                    else:
                        cp.calibrate(
                            wrapper, self.X_train_tree_, y_col,
                            cv_folds=self.cv_folds,
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
                if self.verbose:
                    print(f"    Warning: Conformal calibration failed: {e}")
                    print("    Using standard Gaussian intervals as fallback.")
                self.conformal_predictor_ = None

        self.prediction_intervals_ = intervals

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
        if self.verbose:
            print("  Stage 6: Evaluation and feature importance...")

        n_comps = self.y_train_.shape[1]
        y_arr = self.y_train_.values

        # ── Stage 6a: Genuine holdout model comparison ────────────────────
        self.model_comparison_ = None
        if not self.X_holdout_.empty and len(self.y_holdout_) > 0:
            if self.verbose:
                print("    Stage 6a: Evaluating all models on genuine holdout set...")
            try:
                _X_ho_arr  = self.X_holdout_.values
                _X_ho_pca  = self.reducer_pca_.transform(_X_ho_arr)
                _X_ho_tree = self.reducer_tree_.transform(_X_ho_arr)
                _per_model_X_holdout = {
                    'BayesianRidge':    _X_ho_pca,
                    'GradientBoosting': _X_ho_tree,
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
                    if self.verbose:
                        print(
                            f"      Ensemble holdout inference failed: "
                            f"{_ens_ho_exc}"
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
                if self.verbose and self.model_comparison_:
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
                            print(
                                f"      Best model: Ensemble "
                                f"(R²={_best_mc.holdout_r2:.4f}) outperforms "
                                f"best base model {_base_mc.model_name} "
                                f"(R²={_base_mc.holdout_r2:.4f})"
                            )
                        else:
                            print(
                                f"      Best model: {_best_mc.model_name} "
                                f"(R²={_best_mc.holdout_r2:.4f}) outperforms "
                                f"Ensemble (R²={_ens_mc.holdout_r2:.4f})"
                            )
            except Exception as _cmp_exc:
                if self.verbose:
                    print(
                        f"      Stage 6a failed: "
                        f"{type(_cmp_exc).__name__}: {_cmp_exc}"
                    )
        elif self.verbose:
            print(
                "    Stage 6a: Skipped (no holdout data — "
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
        for name, scores in self._cv_scores_.items():
            if scores:
                self._model_performance_[name] = {
                    'mean_r2': float(np.nanmean(scores)),
                    'std_r2':  float(np.nanstd(scores)),
                }

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
        """Fit the 5-model ensemble and forecast ``target_year``.

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

        if self.verbose:
            print(f"Starting ML Forecasting for {target_year}...")

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
            if self.verbose:
                print(
                    "  pipeline_mode='features_only': stopping after Stage 2 "
                    "(feature inspection only — no model fitting)."
                )
            self.stage2_reduce_features()
            return None

        self.stage2_reduce_features()

        # ── Stages 3–4: Base model training + ensemble predictions ─────────
        self.stage3_fit_base_models()
        self.stage4_fit_meta_learner()

        if _mode == 'fit_only':
            if self.verbose:
                print(
                    "  pipeline_mode='fit_only': stopping after Stage 4 "
                    "(no interval estimation or evaluation)."
                )
            return None

        # ── Stages 5–7: Intervals + evaluation + composite ────────────────
        self.stage5_compute_intervals()
        self.stage6_evaluate_all()
        self.stage7_postprocess()

        if self.verbose:
            print(
                f"  Forecasting complete. "
                f"{len(self.model_weights_)} models combined."
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
