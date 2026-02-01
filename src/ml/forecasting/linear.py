# -*- coding: utf-8 -*-
"""
Linear Forecasting Methods
==========================

Bayesian Ridge, Huber, and Ridge regression for forecasting.

These methods are useful for:
- Interpretable models
- Uncertainty quantification (Bayesian Ridge)
- Robustness to outliers (Huber)
- Fast training and prediction
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler

from .base import BaseForecaster


class BayesianForecaster(BaseForecaster):
    """
    Bayesian Ridge Regression forecaster with uncertainty.
    
    Provides natural uncertainty quantification through the
    posterior distribution over weights.
    
    Parameters:
        alpha_1: Shape parameter for Gamma prior over alpha
        alpha_2: Inverse scale for Gamma prior over alpha
        lambda_1: Shape parameter for Gamma prior over lambda
        lambda_2: Inverse scale for Gamma prior over lambda
        max_iter: Maximum iterations
    
    Example:
        >>> forecaster = BayesianForecaster()
        >>> forecaster.fit(X_train, y_train)
        >>> mean, std = forecaster.predict_with_uncertainty(X_test)
    """
    
    def __init__(self,
                 alpha_1: float = 1e-6,
                 alpha_2: float = 1e-6,
                 lambda_1: float = 1e-6,
                 lambda_2: float = 1e-6,
                 max_iter: int = 300):
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.max_iter = max_iter
        
        self.model = BayesianRidge(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            max_iter=max_iter,
            compute_score=True
        )
        self.scaler = StandardScaler()
        self.feature_importance_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianForecaster':
        """Fit the Bayesian Ridge model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y.ravel() if y.ndim > 1 else y)
        # Use absolute coefficient values as importance
        self.feature_importance_ = np.abs(self.model.coef_)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make point predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty quantification.
        
        Returns:
            Tuple of (mean prediction, standard deviation)
        """
        X_scaled = self.scaler.transform(X)
        mean, std = self.model.predict(X_scaled, return_std=True)
        return mean, std
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from absolute coefficients."""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet")
        return self.feature_importance_
    
    @property
    def alpha(self) -> float:
        """Get estimated alpha (precision of weights)."""
        return self.model.alpha_
    
    @property
    def lambda_(self) -> float:
        """Get estimated lambda (precision of noise)."""
        return self.model.lambda_


class HuberForecaster(BaseForecaster):
    """
    Huber Regression forecaster robust to outliers.
    
    Uses Huber loss which is quadratic for small errors and
    linear for large errors, providing robustness against outliers.
    
    Parameters:
        epsilon: Threshold at which to switch from squared to linear loss
        max_iter: Maximum number of iterations
        alpha: Regularization strength
    
    Example:
        >>> forecaster = HuberForecaster(epsilon=1.35)
        >>> forecaster.fit(X_train, y_train)
        >>> predictions = forecaster.predict(X_test)
    """
    
    def __init__(self,
                 epsilon: float = 1.35,
                 max_iter: int = 100,
                 alpha: float = 0.0001):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.alpha = alpha
        
        self.model = HuberRegressor(
            epsilon=epsilon,
            max_iter=max_iter,
            alpha=alpha
        )
        self.scaler = StandardScaler()
        self.feature_importance_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HuberForecaster':
        """Fit the Huber regression model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y.ravel() if y.ndim > 1 else y)
        self.feature_importance_ = np.abs(self.model.coef_)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from absolute coefficients."""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet")
        return self.feature_importance_
    
    @property
    def outliers_(self) -> np.ndarray:
        """Get mask of detected outliers."""
        return self.model.outliers_


class RidgeForecaster(BaseForecaster):
    """
    Ridge Regression forecaster with L2 regularization.
    
    Standard linear regression with L2 penalty to prevent overfitting.
    
    Parameters:
        alpha: Regularization strength
        fit_intercept: Whether to calculate intercept
        solver: Solver to use
    
    Example:
        >>> forecaster = RidgeForecaster(alpha=1.0)
        >>> forecaster.fit(X_train, y_train)
        >>> predictions = forecaster.predict(X_test)
    """
    
    def __init__(self,
                 alpha: float = 1.0,
                 fit_intercept: bool = True,
                 solver: str = 'auto'):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        
        self.model = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            solver=solver
        )
        self.scaler = StandardScaler()
        self.feature_importance_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeForecaster':
        """Fit the Ridge regression model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y.ravel() if y.ndim > 1 else y)
        self.feature_importance_ = np.abs(self.model.coef_)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from absolute coefficients."""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet")
        return self.feature_importance_
