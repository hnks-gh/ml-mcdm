# -*- coding: utf-8 -*-
"""
Base Classes for ML Forecasting
===============================

This module provides abstract base classes and result containers
for all forecasting methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class ForecastResult:
    """
    Result container for ML forecasting.
    
    Attributes:
        predictions: Entity × Component predictions DataFrame
        prediction_intervals: Dictionary with 'lower' and 'upper' DataFrames
        feature_importance: Feature importance scores per component
        model_performance: Model-wise metrics dictionary
        ensemble_weights: Optimal model weights
        cv_scores: Cross-validation scores per model
        training_history: Training details
        metadata: Additional metadata
    """
    predictions: pd.DataFrame
    prediction_intervals: Dict[str, pd.DataFrame]
    feature_importance: pd.DataFrame
    model_performance: Dict[str, Dict[str, float]]
    ensemble_weights: Dict[str, float]
    cv_scores: Dict[str, List[float]]
    training_history: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"\n{'='*70}",
            "ML FORECASTING RESULTS",
            f"{'='*70}",
            f"\nPredicted entities: {len(self.predictions)}",
            f"Predicted components: {len(self.predictions.columns)}",
            f"\nEnsemble Model Weights:",
        ]
        for model, weight in sorted(self.ensemble_weights.items(), 
                                   key=lambda x: x[1], reverse=True):
            bar = '█' * int(weight * 30)
            lines.append(f"  {model:20s}: {weight:.3f} {bar}")
        
        lines.append(f"\nCross-Validation Performance:")
        for model, scores in self.cv_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            lines.append(f"  {model:20s}: R² = {mean_score:.4f} ± {std_score:.4f}")
        
        lines.append(f"\nTop 10 Important Features:")
        if not self.feature_importance.empty:
            top_features = self.feature_importance.mean(axis=1).nlargest(10)
            for feat, imp in top_features.items():
                lines.append(f"  {feat}: {imp:.4f}")
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary format."""
        return {
            'predictions': self.predictions.to_dict(),
            'prediction_intervals': {k: v.to_dict() for k, v in self.prediction_intervals.items()},
            'feature_importance': self.feature_importance.to_dict(),
            'model_performance': self.model_performance,
            'ensemble_weights': self.ensemble_weights,
            'cv_scores': self.cv_scores,
        }


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.
    
    All forecasters must implement:
    - fit(): Train the model
    - predict(): Make predictions
    - get_feature_importance(): Return feature importance scores
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseForecaster':
        """
        Fit the model to training data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) or (n_samples, n_outputs)
        
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
        
        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_outputs)
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns:
            Array of shape (n_features,) with importance scores
        """
        pass
    
    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray) -> np.ndarray:
        """
        Fit model and make predictions in one call.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
        
        Returns:
            Predictions on test data
        """
        self.fit(X_train, y_train)
        return self.predict(X_test)
