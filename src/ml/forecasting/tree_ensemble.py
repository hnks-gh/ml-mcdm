# -*- coding: utf-8 -*-
"""
Tree-Based Ensemble Forecasters
===============================

Gradient Boosting, Random Forest, and Extra Trees based forecasting models.

These methods are well-suited for:
- Handling non-linear relationships
- Providing feature importance
- Robust to outliers (with appropriate loss functions)
"""

import numpy as np
from typing import Optional
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.preprocessing import RobustScaler

from .base import BaseForecaster


class GradientBoostingForecaster(BaseForecaster):
    """
    Gradient Boosting forecaster with Huber loss for robustness.
    
    Uses gradient boosting with Huber loss function to be robust
    against outliers while maintaining good predictive performance.
    
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
        
        self.model = GradientBoostingRegressor(
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
        self.scaler = RobustScaler()
        self.feature_importance_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingForecaster':
        """Fit the gradient boosting model."""
        X_scaled = self.scaler.fit_transform(X)
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


class RandomForestForecaster(BaseForecaster):
    """
    Random Forest forecaster with uncertainty estimation.
    
    Random Forest provides natural uncertainty estimation through
    the variance of predictions across trees.
    
    Parameters:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees (None = unlimited)
        min_samples_split: Minimum samples to split a node
        min_samples_leaf: Minimum samples in a leaf
        random_state: Random seed
    
    Example:
        >>> forecaster = RandomForestForecaster(n_estimators=100)
        >>> forecaster.fit(X_train, y_train)
        >>> predictions = forecaster.predict(X_test)
        >>> uncertainty = forecaster.predict_uncertainty(X_test)
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            oob_score=True
        )
        self.scaler = RobustScaler()
        self.feature_importance_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestForecaster':
        """Fit the random forest model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y.ravel() if y.ndim > 1 else y)
        self.feature_importance_ = self.model.feature_importances_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using mean of tree predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate prediction uncertainty from tree variance.
        
        Returns:
            Standard deviation of predictions across trees
        """
        X_scaled = self.scaler.transform(X)
        predictions = np.array([tree.predict(X_scaled) 
                               for tree in self.model.estimators_])
        return np.std(predictions, axis=0)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet")
        return self.feature_importance_
    
    @property
    def oob_score(self) -> float:
        """Get out-of-bag score (R²)."""
        return self.model.oob_score_


class ExtraTreesForecaster(BaseForecaster):
    """
    Extremely Randomized Trees forecaster.
    
    Extra Trees adds additional randomization compared to Random Forest
    by also randomizing the split thresholds. This often leads to
    lower variance at the cost of slightly higher bias.
    
    Parameters:
        n_estimators: Number of trees
        max_depth: Maximum tree depth (None = unlimited)
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in leaf
        random_state: Random seed
    
    Example:
        >>> forecaster = ExtraTreesForecaster(n_estimators=100)
        >>> forecaster.fit(X_train, y_train)
        >>> predictions = forecaster.predict(X_test)
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True
        )
        self.scaler = RobustScaler()
        self.feature_importance_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ExtraTreesForecaster':
        """Fit the extra trees model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y.ravel() if y.ndim > 1 else y)
        self.feature_importance_ = self.model.feature_importances_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Estimate prediction uncertainty from tree variance."""
        X_scaled = self.scaler.transform(X)
        predictions = np.array([tree.predict(X_scaled) 
                               for tree in self.model.estimators_])
        return np.std(predictions, axis=0)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet")
        return self.feature_importance_
