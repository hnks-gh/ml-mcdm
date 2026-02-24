# -*- coding: utf-8 -*-
"""
Unified Forecasting Orchestrator
================================

State-of-the-art ensemble forecasting system optimized for small-to-medium
panel data (N < 1000).

Features:
- 5-6 diverse model types (tree, linear, panel-specific)
- Super Learner meta-ensemble with automatic weighting
- Conformal prediction for distribution-free intervals
- Comprehensive feature engineering
- Time-series aware cross-validation
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
from .hierarchical_bayes import HierarchicalBayesForecaster
from .neural_additive import NeuralAdditiveForecaster
from .super_learner import SuperLearner
from .conformal import ConformalPredictor
from .evaluation import ForecastEvaluator, AblationStudy
import functools


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
            mean_r2 = np.mean(scores)
            std_r2 = np.std(scores)
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


class UnifiedForecaster:
    """
    State-of-the-art unified forecasting system.

    Optimized for small-to-medium panel data (N < 1000) with statistically-principled
    ensemble design emphasizing model diversity over quantity.

    Tier 1 - Base Models (6 diverse models):
        1. Gradient Boosting (robust tree ensemble)
        2. Bayesian Ridge (linear with uncertainty quantification)
        3. Quantile Random Forest (distributional forecasting)
        4. Panel VAR (panel fixed effects + autoregressive)
        5. Hierarchical Bayesian (partial pooling)
        6. Neural Additive Models (interpretable non-linearity)

    Tier 2 - Meta-Ensemble:
        - Super Learner: Trains meta-learner on out-of-fold predictions
        - Optimal weighting learned from validation performance

    Tier 3 - Uncertainty Calibration:
        - Conformal Prediction: Distribution-free guaranteed intervals
        - Quantile RF: Full predictive distributions
        - Hierarchical Bayes: Posterior predictive uncertainty

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
                 verbose: bool = True):
        self.conformal_method = conformal_method
        self.conformal_alpha = conformal_alpha
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose

        self.models_: Dict[str, BaseForecaster] = {}
        self.model_weights_: Dict[str, float] = {}
        self.feature_engineer_ = TemporalFeatureEngineer()
        self.super_learner_: Optional[SuperLearner] = None
        self.conformal_predictor_: Optional[ConformalPredictor] = None
        self.evaluator_: Optional[ForecastEvaluator] = None
    
    def _create_models(self) -> Dict[str, BaseForecaster]:
        """Create all base model instances (6 diverse models)."""
        models = {}

        # Tree-based model (1 model)
        models['GradientBoosting'] = GradientBoostingForecaster(
            n_estimators=100, random_state=self.random_state
        )

        # Bayesian linear model (1 model)
        models['BayesianRidge'] = BayesianForecaster()

        # Advanced panel-specific models (4 models)
        models['QuantileRF'] = QuantileRandomForestForecaster(
            n_estimators=100, random_state=self.random_state
        )
        models['PanelVAR'] = PanelVARForecaster(
            n_lags=2, alpha=1.0, use_fixed_effects=True,
            random_state=self.random_state
        )
        models['HierarchicalBayes'] = HierarchicalBayesForecaster(
            n_em_iterations=20, random_state=self.random_state
        )
        models['NAM'] = NeuralAdditiveForecaster(
            n_basis_per_feature=10, n_iterations=3,
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
            2. Train 6 diverse base models (tree, linear, panel-specific)
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

        # ===== Stage 2: Create base models =====
        self.models_ = self._create_models()

        if self.verbose:
            print(f"  Stage 2: {len(self.models_)} diverse base models created:")
            for name in self.models_:
                print(f"    - {name}")

        # ===== Stage 3: Super Learner meta-ensemble =====
        X_arr = X_train.values
        y_arr = y_train.values
        
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
            self.super_learner_.predict_with_uncertainty(X_pred.values)
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

        # Default intervals from uncertainty
        intervals = {
            'lower': pred_df - 1.96 * unc_df,
            'upper': pred_df + 1.96 * unc_df
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
            for d, col in enumerate(component_cols):
                wrapper = _SingleOutputWrapper(self.super_learner_, d)
                y_col = y_arr[:, d] if y_arr.ndim > 1 else y_arr

                cp = ConformalPredictor(
                    method=self.conformal_method,
                    alpha=alpha_bonferroni,
                    random_state=self.random_state,
                )
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
        feature_importance = self._aggregate_feature_importance(
            self.feature_engineer_.get_feature_names(),
            y_train.columns.tolist()
        )

        model_performance = {}
        for name, scores in cv_scores.items():
            if scores:
                model_performance[name] = {
                    'mean_r2': np.mean(scores),
                    'std_r2': np.std(scores)
                }

        # Build training info
        training_info = {
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'mode': 'advanced',  # Always use ADVANCED mode
            'ensemble_method': 'super_learner',  # Always use Super Learner
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
            holdout_performance=None,
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
        """Aggregate feature importance across fitted models."""
        importance_dict = {}
        
        # Use the FITTED base models from SuperLearner, not the unfitted originals
        fitted_models = (
            self.super_learner_._fitted_base_models
            if self.super_learner_ is not None
            else {}
        )
        
        for name, model in fitted_models.items():
            try:
                imp = model.get_feature_importance()
                importance_dict[name] = imp
            except Exception:
                pass
        
        if not importance_dict:
            return pd.DataFrame()
        
        # Average importance across models
        avg_importance = np.mean(list(importance_dict.values()), axis=0)
        
        return pd.DataFrame(
            {comp: avg_importance for comp in component_names},
            index=feature_names
        )
