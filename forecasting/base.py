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
