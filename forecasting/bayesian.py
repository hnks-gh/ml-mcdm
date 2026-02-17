# -*- coding: utf-8 -*-
"""
Bayesian Ridge Regression Forecaster
====================================

Bayesian Ridge regression for forecasting with uncertainty quantification.

This method is useful for:
- Interpretable linear models
- Natural uncertainty quantification via posterior distribution
- Automatic regularization (no manual hyperparameter tuning)
- Fast training and prediction
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.linear_model import BayesianRidge
from sklearn.multioutput import MultiOutputRegressor
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
        
        self._base_model = BayesianRidge(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            max_iter=max_iter,
            compute_score=True
        )
        self.model = None  # Will be set during fit
        self.scaler = StandardScaler()
        self.feature_importance_: Optional[np.ndarray] = None
        self._is_multi_output = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianForecaster':
        """Fit the Bayesian Ridge model."""
        X_scaled = self.scaler.fit_transform(X)
        
        # Handle multi-output case
        if y.ndim > 1 and y.shape[1] > 1:
            self._is_multi_output = True
            self.model = MultiOutputRegressor(self._base_model)
            self.model.fit(X_scaled, y)
            # Average importance across outputs
            self.feature_importance_ = np.mean(
                [np.abs(est.coef_) for est in self.model.estimators_], axis=0
            )
        else:
            self._is_multi_output = False
            self.model = self._base_model
            self.model.fit(X_scaled, y.ravel() if y.ndim > 1 else y)
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
