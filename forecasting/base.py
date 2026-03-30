# -*- coding: utf-8 -*-
"""
Forecasting Base Architecture
=============================

Provides the foundational abstract interfaces and standard result containers 
for the ML ensemble forecasting system. All specific forecasting models 
(e.g., CatBoost, Ridge, SuperLearner) inherit from these base classes to 
ensure a consistent API for training, prediction, and feature importance 
analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


class BaseForecaster(ABC):
    """
    Abstract base interface for all forecasting models in the pipeline.

    Defines the standard protocol for model training, inference, and 
    interpretability (feature importance). This ensures that base models and 
    the meta-ensemble can be used interchangeably in evaluation loops.
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseForecaster':
        """
        Train the model on supervised historical data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target values or matrix of shape (n_samples,) or (n_samples, n_outputs).

        Returns
        -------
        BaseForecaster
            Self for method chaining.
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Produce target estimates for new observations.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples,) or (n_samples, n_outputs).
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        """
        Retrieve relative importance scores for input features.

        Returns
        -------
        np.ndarray
            Array of shape (n_features,) containing positive importance weights.
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
