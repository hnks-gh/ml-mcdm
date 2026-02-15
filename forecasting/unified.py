# -*- coding: utf-8 -*-
"""
Unified Forecasting Orchestrator
================================

Combines multiple ML forecasting methods into a unified ensemble,
with automatic model selection and weighting.

Features:
- Multiple model types (tree ensemble, linear, neural)
- Automatic performance-based weighting
- Cross-validation with time series split
- Uncertainty quantification
- Comprehensive result reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import copy

from .base import BaseForecaster, ForecastResult
from .tree_ensemble import GradientBoostingForecaster, RandomForestForecaster, ExtraTreesForecaster
from .linear import BayesianForecaster, HuberForecaster, RidgeForecaster
from .neural import NeuralForecaster, AttentionForecaster
from .features import TemporalFeatureEngineer

# State-of-the-art advanced models
from .panel_var import PanelVARForecaster
from .quantile_forest import QuantileRandomForestForecaster
from .hierarchical_bayes import HierarchicalBayesForecaster
from .neural_additive import NeuralAdditiveForecaster
from .super_learner import SuperLearner
from .conformal import ConformalPredictor
from .auto_ensemble import AutoEnsembleSelector
from .evaluation import ForecastEvaluator, AblationStudy

warnings.filterwarnings('ignore')


class ForecastMode(Enum):
    """Forecasting mode selection."""
    FAST = "fast"             # Quick prediction with fewer models
    BALANCED = "balanced"     # Good trade-off between speed and accuracy
    ACCURATE = "accurate"     # Maximum accuracy with all models
    NEURAL = "neural"         # Neural network focused
    ENSEMBLE = "ensemble"     # Full ensemble
    ADVANCED = "advanced"     # State-of-the-art: Super Learner + all advanced models
    AUTO = "auto"             # Bayesian optimization-selected ensemble


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

    Combines multiple forecasting approaches in a tiered architecture:

    Tier 1 - Base Models:
        1. Gradient Boosting Ensemble (GBM, RF, ET)
        2. Quantile Random Forest (distributional forecasting)
        3. Panel VAR (panel fixed effects + autoregressive)
        4. Hierarchical Bayesian (partial pooling)
        5. Neural Additive Models (interpretable non-linearity)
        6. Linear Methods (Bayesian Ridge, Huber, Ridge)
        7. Neural Networks (MLP, Attention) [optional]

    Tier 2 - Meta-Ensemble:
        - Super Learner: Trains meta-learner on out-of-fold predictions
        - Auto-Ensemble: Bayesian optimization for model selection
        - Replaces simple weighted averaging with learned combination

    Tier 3 - Uncertainty Calibration:
        - Conformal Prediction: Distribution-free guaranteed intervals
        - Quantile RF: Full predictive distributions
        - Hierarchical Bayes: Posterior predictive uncertainty

    Features:
    - Automatic model selection and weighting via Super Learner
    - Comprehensive feature engineering
    - Multi-level ensemble stacking with meta-learning
    - Calibrated uncertainty quantification
    - Time-series aware validation
    - Bayesian optimization for model selection (AUTO mode)

    Parameters:
        mode: Forecasting mode (FAST, BALANCED, ACCURATE, NEURAL,
              ENSEMBLE, ADVANCED, AUTO)
        include_neural: Whether to include neural models
        include_tree_ensemble: Whether to include tree-based models
        include_linear: Whether to include linear models
        include_advanced: Whether to include advanced models
                         (Panel VAR, QRF, Hier. Bayes, NAM)
        use_super_learner: Whether to use Super Learner meta-ensemble
                          instead of simple weighted averaging
        use_conformal: Whether to add conformal prediction intervals
        conformal_method: Conformal method ('split', 'cv_plus', 'adaptive')
        conformal_alpha: Miscoverage rate for conformal intervals
        cv_folds: Number of cross-validation folds
        random_state: Random seed
        verbose: Print progress messages

    Example:
        >>> # Standard mode (backward compatible)
        >>> forecaster = UnifiedForecaster(mode=ForecastMode.BALANCED)
        >>> result = forecaster.fit_predict(panel_data, target_year=2025)
        >>>
        >>> # Advanced mode with Super Learner + all models
        >>> forecaster = UnifiedForecaster(mode=ForecastMode.ADVANCED)
        >>> result = forecaster.fit_predict(panel_data, target_year=2025)
        >>>
        >>> # Auto mode with Bayesian optimization
        >>> forecaster = UnifiedForecaster(mode=ForecastMode.AUTO)
        >>> result = forecaster.fit_predict(panel_data, target_year=2025)
    """

    def __init__(self,
                 mode: ForecastMode = ForecastMode.BALANCED,
                 include_neural: bool = False,
                 include_tree_ensemble: bool = True,
                 include_linear: bool = True,
                 include_advanced: bool = True,
                 use_super_learner: bool = True,
                 use_conformal: bool = True,
                 conformal_method: str = 'cv_plus',
                 conformal_alpha: float = 0.05,
                 cv_folds: int = 3,
                 random_state: int = 42,
                 verbose: bool = True):
        self.mode = mode
        self.include_neural = include_neural
        self.include_tree_ensemble = include_tree_ensemble
        self.include_linear = include_linear
        self.include_advanced = include_advanced
        self.use_super_learner = use_super_learner
        self.use_conformal = use_conformal
        self.conformal_method = conformal_method
        self.conformal_alpha = conformal_alpha
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose

        # Auto-configure based on mode
        if mode == ForecastMode.ADVANCED:
            self.include_advanced = True
            self.use_super_learner = True
            self.use_conformal = True
        elif mode == ForecastMode.AUTO:
            self.include_advanced = True
            self.use_super_learner = False  # Auto uses Optuna instead
            self.use_conformal = True
        elif mode == ForecastMode.FAST:
            self.include_advanced = False
            self.use_super_learner = False
            self.use_conformal = False

        self.models_: Dict[str, BaseForecaster] = {}
        self.model_weights_: Dict[str, float] = {}
        self.feature_engineer_ = TemporalFeatureEngineer()
        self.super_learner_: Optional[SuperLearner] = None
        self.auto_ensemble_: Optional[AutoEnsembleSelector] = None
        self.conformal_predictor_: Optional[ConformalPredictor] = None
        self.evaluator_: Optional[ForecastEvaluator] = None
    
    def _create_models(self) -> Dict[str, BaseForecaster]:
        """Create model instances based on mode."""
        models = {}

        # --- Tree-based ensemble models ---
        if self.include_tree_ensemble:
            if self.mode in [ForecastMode.BALANCED, ForecastMode.ACCURATE,
                             ForecastMode.ENSEMBLE, ForecastMode.ADVANCED,
                             ForecastMode.AUTO]:
                models['GradientBoosting'] = GradientBoostingForecaster(
                    n_estimators=200, random_state=self.random_state
                )
                models['RandomForest'] = RandomForestForecaster(
                    n_estimators=100, random_state=self.random_state
                )
            if self.mode in [ForecastMode.ACCURATE, ForecastMode.ENSEMBLE,
                             ForecastMode.ADVANCED, ForecastMode.AUTO]:
                models['ExtraTrees'] = ExtraTreesForecaster(
                    n_estimators=100, random_state=self.random_state
                )
            if self.mode == ForecastMode.FAST:
                models['GradientBoosting'] = GradientBoostingForecaster(
                    n_estimators=50, random_state=self.random_state
                )

        # --- Linear models ---
        if self.include_linear:
            if self.mode in [ForecastMode.BALANCED, ForecastMode.ACCURATE,
                             ForecastMode.ENSEMBLE, ForecastMode.ADVANCED,
                             ForecastMode.AUTO]:
                models['BayesianRidge'] = BayesianForecaster()
                models['Huber'] = HuberForecaster()
            if self.mode == ForecastMode.FAST:
                models['Ridge'] = RidgeForecaster()

        # --- Advanced models (new state-of-the-art) ---
        if self.include_advanced:
            if self.mode in [ForecastMode.BALANCED, ForecastMode.ACCURATE,
                             ForecastMode.ENSEMBLE, ForecastMode.ADVANCED,
                             ForecastMode.AUTO]:
                # Quantile Random Forest (distributional forecasting)
                models['QuantileRF'] = QuantileRandomForestForecaster(
                    n_estimators=200, random_state=self.random_state
                )
                # Panel VAR (fixed effects + autoregressive)
                models['PanelVAR'] = PanelVARForecaster(
                    n_lags=2, alpha=1.0, use_fixed_effects=True,
                    random_state=self.random_state
                )
                # Hierarchical Bayesian (partial pooling)
                models['HierarchicalBayes'] = HierarchicalBayesForecaster(
                    n_em_iterations=50, random_state=self.random_state
                )
                # Neural Additive Model (interpretable non-linearity)
                models['NAM'] = NeuralAdditiveForecaster(
                    n_basis_per_feature=50, n_iterations=10,
                    random_state=self.random_state
                )

        # --- Neural network models (optional) ---
        if self.include_neural:
            if self.mode in [ForecastMode.NEURAL, ForecastMode.ACCURATE,
                             ForecastMode.ENSEMBLE, ForecastMode.ADVANCED]:
                models['NeuralMLP'] = NeuralForecaster(
                    hidden_dims=[128, 64], n_epochs=50, seed=self.random_state
                )
            if self.mode in [ForecastMode.NEURAL, ForecastMode.ENSEMBLE]:
                models['Attention'] = AttentionForecaster(
                    hidden_dim=64, n_epochs=50, seed=self.random_state
                )

        return models
    
    def fit_predict(self,
                   panel_data,
                   target_year: int,
                   weights: Optional[Dict[str, float]] = None
                   ) -> UnifiedForecastResult:
        """
        Fit models and make predictions for target year.

        Architecture (ADVANCED / AUTO modes):
            1. Feature engineering (temporal lags, rolling stats, etc.)
            2. Create base models (tree, linear, advanced)
            3. Super Learner meta-ensemble OR Auto-Ensemble selection
            4. Conformal prediction interval calibration
            5. Return predictions + calibrated uncertainty

        Args:
            panel_data: Panel data object with temporal data
            target_year: Year to predict
            weights: Optional pre-specified model weights

        Returns:
            UnifiedForecastResult with predictions and analysis
        """
        if self.verbose:
            print(f"Starting unified forecasting for {target_year}...")
            print(f"  Mode: {self.mode.value}")

        # ===== Stage 1: Feature engineering =====
        if self.verbose:
            print("  Stage 1: Engineering features...")

        X_train, y_train, X_pred, _ = self.feature_engineer_.fit_transform(
            panel_data, target_year
        )

        # ===== Stage 2: Create base models =====
        self.models_ = self._create_models()

        if self.verbose:
            print(f"  Stage 2: {len(self.models_)} base models created:")
            for name in self.models_:
                print(f"    - {name}")

        # ===== Stage 3: Model combination strategy =====
        X_arr = X_train.values
        y_arr = y_train.values

        if self.mode == ForecastMode.AUTO:
            # AUTO mode: Bayesian optimization ensemble selection
            if self.verbose:
                print("  Stage 3: Auto-Ensemble (Bayesian Optimization)...")

            self.auto_ensemble_ = AutoEnsembleSelector(
                base_models=self.models_,
                n_trials=50,
                cv_folds=self.cv_folds,
                min_models=2,
                max_models=min(6, len(self.models_)),
                random_state=self.random_state,
                verbose=self.verbose,
            )
            self.auto_ensemble_.fit(X_arr, y_arr)

            self.model_weights_ = self.auto_ensemble_.get_best_weights()
            cv_scores = {name: [] for name in self.models_}
            predictions_arr = self.auto_ensemble_.predict(X_pred.values)
            unc_arr = self.auto_ensemble_.predict_with_uncertainty(X_pred.values)
            if isinstance(unc_arr, tuple):
                predictions_arr, uncertainty_arr = unc_arr
            else:
                uncertainty_arr = np.zeros_like(predictions_arr)

        elif self.use_super_learner and self.mode in [
            ForecastMode.ADVANCED, ForecastMode.ACCURATE,
            ForecastMode.ENSEMBLE
        ]:
            # ADVANCED mode: Super Learner meta-ensemble
            if self.verbose:
                print("  Stage 3: Super Learner meta-ensemble...")

            self.super_learner_ = SuperLearner(
                base_models=self.models_,
                meta_learner_type='ridge',
                n_cv_folds=min(self.cv_folds + 2, 5),
                positive_weights=True,
                normalize_weights=True,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            self.super_learner_.fit(X_arr, y_arr)

            self.model_weights_ = self.super_learner_.get_meta_weights()
            cv_scores = self.super_learner_.get_cv_scores()

            predictions_arr, uncertainty_arr = (
                self.super_learner_.predict_with_uncertainty(X_pred.values)
            )

        else:
            # Legacy mode: simple weighted averaging
            if self.verbose:
                print("  Stage 3: Cross-validation + weighted averaging...")

            cv_scores = self._cross_validate(X_arr, y_arr)

            if weights is None:
                self.model_weights_ = self._calculate_weights(cv_scores)
            else:
                self.model_weights_ = weights

            # Fit all models on full training data
            for name, model in self.models_.items():
                model.fit(X_arr, y_arr)

            predictions_arr, uncertainty_arr = self._ensemble_predict(X_pred.values)

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

        # ===== Stage 5: Conformal prediction calibration =====
        if self.use_conformal:
            if self.verbose:
                print("  Stage 5: Conformal prediction calibration...")

            try:
                self.conformal_predictor_ = ConformalPredictor(
                    method=self.conformal_method,
                    alpha=self.conformal_alpha,
                    random_state=self.random_state,
                )

                # Determine which model to calibrate against
                if self.super_learner_ is not None:
                    cal_model = self.super_learner_
                elif self.auto_ensemble_ is not None:
                    cal_model = self.auto_ensemble_
                else:
                    # Use a simple wrapper for the weighted ensemble
                    cal_model = self._create_ensemble_wrapper()

                self.conformal_predictor_.calibrate(
                    cal_model, X_arr, y_arr, cv_folds=self.cv_folds
                )

                # Overwrite intervals with conformal-calibrated intervals
                cp_lower, cp_upper = self.conformal_predictor_.predict_intervals(
                    X_pred.values, point_predictions=pred_df.values.mean(axis=1)
                )

                # Apply conformal intervals to all components
                for col in y_train.columns:
                    intervals['lower'][col] = cp_lower
                    intervals['upper'][col] = cp_upper

                if self.verbose:
                    width = self.conformal_predictor_.get_interval_width()
                    print(f"    Conformal interval width: {width:.4f}")
                    print(f"    Coverage guarantee: {(1 - self.conformal_alpha) * 100:.0f}%")

            except Exception as e:
                if self.verbose:
                    print(f"    Warning: Conformal calibration failed: {e}")
                    print("    Using standard Gaussian intervals as fallback.")

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
            'mode': self.mode.value,
            'ensemble_method': (
                'super_learner' if self.super_learner_ is not None
                else 'auto_ensemble' if self.auto_ensemble_ is not None
                else 'weighted_averaging'
            ),
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

    def _create_ensemble_wrapper(self):
        """Create a simple wrapper object with predict() for conformal calibration."""
        models = self.models_
        weights = self.model_weights_

        class _EnsembleWrapper:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights

            def fit(self, X, y):
                for name, model in self.models.items():
                    model.fit(X, y)
                return self

            def predict(self, X):
                all_preds = []
                for name, model in self.models.items():
                    pred = model.predict(X)
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, 1)
                    all_preds.append(pred * self.weights.get(name, 0.0))
                result = np.sum(all_preds, axis=0)
                return result.mean(axis=1) if result.ndim > 1 else result

        return _EnsembleWrapper(models, weights)
    
    def _cross_validate(self,
                       X: np.ndarray,
                       y: np.ndarray
                       ) -> Dict[str, List[float]]:
        """Run time-series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        cv_scores = {name: [] for name in self.models_.keys()}
        
        for train_idx, val_idx in tscv.split(X):
            X_cv_train, X_cv_val = X[train_idx], X[val_idx]
            y_cv_train, y_cv_val = y[train_idx], y[val_idx]
            
            for name, model in self.models_.items():
                # Clone model for CV
                model_copy = copy.deepcopy(model)
                model_copy.fit(X_cv_train, y_cv_train)
                
                pred = model_copy.predict(X_cv_val)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                if y_cv_val.ndim == 1:
                    y_cv_val = y_cv_val.reshape(-1, 1)
                
                # Calculate R² for each output and average
                r2_scores = []
                for col in range(y_cv_val.shape[1]):
                    r2 = r2_score(y_cv_val[:, col], pred[:, min(col, pred.shape[1]-1)])
                    r2_scores.append(r2)
                cv_scores[name].append(np.mean(r2_scores))
        
        return cv_scores
    
    def _calculate_weights(self,
                          cv_scores: Dict[str, List[float]]
                          ) -> Dict[str, float]:
        """Calculate model weights based on CV performance."""
        mean_scores = {name: np.mean(scores) for name, scores in cv_scores.items()}
        
        # Use softmax over scores (shifted for numerical stability)
        scores_arr = np.array(list(mean_scores.values()))
        scores_shifted = scores_arr - scores_arr.max()
        exp_scores = np.exp(scores_shifted * 5)  # Temperature scaling
        weights = exp_scores / exp_scores.sum()
        
        return dict(zip(mean_scores.keys(), weights))
    
    def _ensemble_predict(self,
                         X: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions with uncertainty."""
        all_predictions = []
        
        for name, model in self.models_.items():
            pred = model.predict(X)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            all_predictions.append(pred * self.model_weights_[name])
        
        # Weighted ensemble prediction
        ensemble_pred = np.sum(all_predictions, axis=0)
        
        # Uncertainty from prediction disagreement
        pred_array = np.stack([p / self.model_weights_[n] 
                              for p, n in zip(all_predictions, self.models_.keys())], axis=0)
        uncertainty = np.std(pred_array, axis=0)
        
        return ensemble_pred, uncertainty
    
    def _aggregate_feature_importance(self,
                                     feature_names: List[str],
                                     component_names: List[str]
                                     ) -> pd.DataFrame:
        """Aggregate feature importance across models."""
        importance_dict = {}
        
        for name, model in self.models_.items():
            try:
                imp = model.get_feature_importance()
                importance_dict[name] = imp
            except:
                pass
        
        if not importance_dict:
            return pd.DataFrame()
        
        # Average importance across models
        avg_importance = np.mean(list(importance_dict.values()), axis=0)
        
        return pd.DataFrame(
            {comp: avg_importance for comp in component_names},
            index=feature_names
        )
