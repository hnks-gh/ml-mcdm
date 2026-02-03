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

warnings.filterwarnings('ignore')


class ForecastMode(Enum):
    """Forecasting mode selection."""
    FAST = "fast"           # Quick prediction with fewer models
    BALANCED = "balanced"   # Good trade-off between speed and accuracy
    ACCURATE = "accurate"   # Maximum accuracy with all models
    NEURAL = "neural"       # Neural network focused
    ENSEMBLE = "ensemble"   # Full ensemble


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
    
    Combines multiple forecasting approaches:
    1. Gradient Boosting Ensemble (GBM, RF, ET)
    2. Neural Network Ensemble (MLP, Attention)
    3. Bayesian Methods (for uncertainty)
    4. Robust Linear Methods (Huber, Ridge)
    
    Features:
    - Automatic model selection and weighting
    - Comprehensive feature engineering
    - Multi-level ensemble stacking
    - Uncertainty quantification
    - Time-series aware validation
    
    Parameters:
        mode: Forecasting mode (FAST, BALANCED, ACCURATE, NEURAL, ENSEMBLE)
        include_neural: Whether to include neural models (default False - disabled due to data limitations)
        include_tree_ensemble: Whether to include tree-based models
        include_linear: Whether to include linear models
        cv_folds: Number of cross-validation folds
        random_state: Random seed
        verbose: Print progress messages
    
    Example:
        >>> forecaster = UnifiedForecaster(mode=ForecastMode.BALANCED)
        >>> result = forecaster.fit_predict(panel_data, target_year=2025)
        >>> print(result.get_summary())
    """
    
    def __init__(self,
                 mode: ForecastMode = ForecastMode.BALANCED,
                 include_neural: bool = False,  # Disabled by default - insufficient data for neural networks
                 include_tree_ensemble: bool = True,
                 include_linear: bool = True,
                 cv_folds: int = 3,
                 random_state: int = 42,
                 verbose: bool = True):
        self.mode = mode
        self.include_neural = include_neural
        self.include_tree_ensemble = include_tree_ensemble
        self.include_linear = include_linear
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose
        
        self.models_: Dict[str, BaseForecaster] = {}
        self.model_weights_: Dict[str, float] = {}
        self.feature_engineer_ = TemporalFeatureEngineer()
    
    def _create_models(self) -> Dict[str, BaseForecaster]:
        """Create model instances based on mode."""
        models = {}
        
        if self.include_tree_ensemble:
            if self.mode in [ForecastMode.BALANCED, ForecastMode.ACCURATE, ForecastMode.ENSEMBLE]:
                models['GradientBoosting'] = GradientBoostingForecaster(
                    n_estimators=200, random_state=self.random_state
                )
                models['RandomForest'] = RandomForestForecaster(
                    n_estimators=100, random_state=self.random_state
                )
            if self.mode in [ForecastMode.ACCURATE, ForecastMode.ENSEMBLE]:
                models['ExtraTrees'] = ExtraTreesForecaster(
                    n_estimators=100, random_state=self.random_state
                )
            if self.mode == ForecastMode.FAST:
                models['GradientBoosting'] = GradientBoostingForecaster(
                    n_estimators=50, random_state=self.random_state
                )
        
        if self.include_linear:
            if self.mode in [ForecastMode.BALANCED, ForecastMode.ACCURATE, ForecastMode.ENSEMBLE]:
                models['BayesianRidge'] = BayesianForecaster()
                models['Huber'] = HuberForecaster()
            if self.mode == ForecastMode.FAST:
                models['Ridge'] = RidgeForecaster()
        
        if self.include_neural:
            if self.mode in [ForecastMode.NEURAL, ForecastMode.ACCURATE, ForecastMode.ENSEMBLE]:
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
        
        Args:
            panel_data: Panel data object with temporal data
            target_year: Year to predict
            weights: Optional pre-specified model weights
        
        Returns:
            UnifiedForecastResult with predictions and analysis
        """
        if self.verbose:
            print(f"Starting unified forecasting for {target_year}...")
        
        # Feature engineering
        if self.verbose:
            print("  Engineering features...")
        
        X_train, y_train, X_pred, _ = self.feature_engineer_.fit_transform(
            panel_data, target_year
        )
        
        # Create models
        self.models_ = self._create_models()
        
        # Cross-validation for model selection
        if self.verbose:
            print("  Running cross-validation...")
        
        cv_scores = self._cross_validate(X_train.values, y_train.values)
        
        # Calculate model weights from CV performance
        if weights is None:
            self.model_weights_ = self._calculate_weights(cv_scores)
        else:
            self.model_weights_ = weights
        
        # Fit all models on full training data
        if self.verbose:
            print("  Fitting models on full data...")
        
        for name, model in self.models_.items():
            model.fit(X_train.values, y_train.values)
        
        # Make predictions
        if self.verbose:
            print("  Generating predictions...")
        
        predictions, uncertainty = self._ensemble_predict(X_pred.values)
        
        # Create result DataFrames
        pred_df = pd.DataFrame(
            predictions, 
            index=X_pred.index,
            columns=y_train.columns
        )
        
        unc_df = pd.DataFrame(
            uncertainty,
            index=X_pred.index,
            columns=y_train.columns
        )
        
        # Prediction intervals
        intervals = {
            'lower': pred_df - 1.96 * unc_df,
            'upper': pred_df + 1.96 * unc_df
        }
        
        # Feature importance
        feature_importance = self._aggregate_feature_importance(
            self.feature_engineer_.get_feature_names(),
            y_train.columns.tolist()
        )
        
        # Model performance summary
        model_performance = {}
        for name, scores in cv_scores.items():
            model_performance[name] = {
                'mean_r2': np.mean(scores),
                'std_r2': np.std(scores)
            }
        
        return UnifiedForecastResult(
            predictions=pred_df,
            uncertainty=unc_df,
            prediction_intervals=intervals,
            model_contributions=self.model_weights_,
            model_performance=model_performance,
            feature_importance=feature_importance,
            cross_validation_scores=cv_scores,
            holdout_performance=None,
            training_info={
                'n_samples': len(X_train),
                'n_features': X_train.shape[1],
                'mode': self.mode.value
            },
            data_summary={
                'n_entities': len(X_pred),
                'n_components': y_train.shape[1]
            }
        )
    
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
