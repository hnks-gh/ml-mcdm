# -*- coding: utf-8 -*-
"""
Tree-Based Forecaster
====================

Gradient Boosting forecasting model optimized for small panel data.

This method is well-suited for:
- Handling non-linear relationships
- Providing feature importance
- Robust to outliers (Huber loss)
- Sample-efficient learning (sequential boosting)
"""

import numpy as np
from typing import Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler

from .base import BaseForecaster


class GradientBoostingForecaster(BaseForecaster):
    """
    Gradient Boosting forecaster with Huber loss for robustness.
    
    Uses gradient boosting with Huber loss function to be robust
    against outliers while maintaining good predictive performance.
    
    Optimal for small-to-medium panel data (N < 1000) due to:
    - Sequential learning that efficiently uses all training data
    - Built-in regularization (learning rate, early stopping, subsampling)
    - Superior bias-variance tradeoff compared to Random Forest/Extra Trees
    
    Parameters:
        n_estimators: Number of boosting stages
        max_depth: Maximum depth of individual trees
        learning_rate: Shrinkage factor
        subsample: Fraction of samples for each tree
        random_state: Random seed
    
    Example:
        >>> forecaster = GradientBoostingForecaster(n_estimators=200)
        >>> forecaster.fit(X_train, y_train)
        >>> predictions = forecaster.predict(X_test)
    """
    
    def __init__(self,
                 n_estimators: int = 200,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        
        self._base_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state,
            loss='huber',  # Robust to outliers
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4
        )
        self.model = None  # Will be set during fit
        self.scaler = RobustScaler()
        self.feature_importance_: Optional[np.ndarray] = None
        self._is_multi_output = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingForecaster':
        """Fit the gradient boosting model."""
        X_scaled = self.scaler.fit_transform(X)
        
        # Handle multi-output case
        if y.ndim > 1 and y.shape[1] > 1:
            self._is_multi_output = True
            self.model = MultiOutputRegressor(self._base_model)
            self.model.fit(X_scaled, y)
            # Average feature importance across outputs
            self.feature_importance_ = np.mean(
                [est.feature_importances_ for est in self.model.estimators_], axis=0
            )
        else:
            self._is_multi_output = False
            self.model = self._base_model
            self.model.fit(X_scaled, y.ravel() if y.ndim > 1 else y)
            self.feature_importance_ = self.model.feature_importances_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet")
        return self.feature_importance_
