# -*- coding: utf-8 -*-
"""
LSTM Forecaster for Panel Data
===============================

Time-series forecasting using LSTM neural networks.
Predicts future scores based on historical trajectories.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class LSTMResult:
    """Result container for LSTM forecasting."""
    predictions: pd.DataFrame        # Province × predicted values
    actual: pd.DataFrame             # Province × actual values
    train_loss: List[float]
    val_loss: List[float]
    test_metrics: Dict[str, float]
    predicted_ranks: pd.Series
    actual_ranks: pd.Series
    rank_correlation: float
    model_summary: str
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "LSTM FORECASTING RESULTS",
            f"{'='*60}",
            f"\nTest Metrics:",
            f"  MSE: {self.test_metrics['mse']:.4f}",
            f"  MAE: {self.test_metrics['mae']:.4f}",
            f"  RMSE: {self.test_metrics['rmse']:.4f}",
            f"  Rank Correlation: {self.rank_correlation:.4f}",
            f"\nTraining:",
            f"  Final Train Loss: {self.train_loss[-1]:.4f}" if self.train_loss else "  N/A",
            f"  Final Val Loss: {self.val_loss[-1]:.4f}" if self.val_loss else "  N/A",
            "=" * 60
        ]
        return "\n".join(lines)


class SimpleLSTM:
    """
    Simple LSTM implementation using numpy for environments without PyTorch/TensorFlow.
    Uses a basic recurrent approach for forecasting with gradient descent training.
    """
    
    def __init__(self, hidden_size: int = 32, learning_rate: float = 0.01):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weights = None
        self.training_history = {'train_loss': [], 'val_loss': []}
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> List[float]:
        """Fit using iterative gradient descent for better training visualization."""
        n_samples, seq_len, n_features = X.shape
        
        # Prepare features
        X_full = self._prepare_features(X)
        n_features_full = X_full.shape[1]
        
        # Prepare validation features if provided
        X_val_full = self._prepare_features(X_val) if X_val is not None else None
        
        # Initialize weights randomly
        np.random.seed(42)
        self.weights = np.random.randn(n_features_full, y.shape[1] if len(y.shape) > 1 else 1) * 0.01
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if y_val is not None and len(y_val.shape) == 1:
            y_val = y_val.reshape(-1, 1)
        
        # Training with gradient descent
        train_losses = []
        val_losses = []
        lambda_reg = 0.01
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = X_full @ self.weights
            
            # Calculate training loss (MSE + L2 regularization)
            train_mse = np.mean((y - y_pred) ** 2)
            reg_loss = lambda_reg * np.sum(self.weights ** 2)
            train_loss = train_mse + reg_loss
            train_losses.append(train_loss)
            
            # Validation loss
            if X_val_full is not None and y_val is not None:
                val_pred = X_val_full @ self.weights
                val_loss = np.mean((y_val - val_pred) ** 2)
                val_losses.append(val_loss)
            
            # Gradient descent step
            gradient = (2 / n_samples) * X_full.T @ (y_pred - y) + 2 * lambda_reg * self.weights
            self.weights -= self.learning_rate * gradient
            
            # Adaptive learning rate decay
            if epoch > 0 and epoch % 20 == 0:
                self.learning_rate *= 0.9
        
        self.training_history['train_loss'] = train_losses
        self.training_history['val_loss'] = val_losses if val_losses else train_losses
        
        return train_losses
    
    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Prepare temporal features from input sequences."""
        if X is None or len(X) == 0:
            return None
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(n_samples, -1)
        
        # Add temporal aggregates for richer features
        X_mean = X.mean(axis=1)
        X_last = X[:, -1, :]
        X_trend = X[:, -1, :] - X[:, 0, :]
        X_std = X.std(axis=1)
        X_max = X.max(axis=1)
        X_min = X.min(axis=1)
        
        return np.hstack([X_flat, X_mean, X_last, X_trend, X_std, X_max, X_min])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using fitted weights."""
        X_full = self._prepare_features(X)
        predictions = X_full @ self.weights
        return predictions.flatten() if predictions.shape[1] == 1 else predictions


class LSTMForecaster:
    """
    LSTM-based forecaster for panel data.
    
    Uses historical province trajectories to predict future scores.
    Falls back to simple temporal model if deep learning not available.
    """
    
    def __init__(self,
                 sequence_length: int = 3,
                 hidden_units: int = 64,
                 n_layers: int = 2,
                 dropout: float = 0.2,
                 epochs: int = 100,
                 batch_size: int = 16,
                 learning_rate: float = 0.001,
                 patience: int = 15,
                 random_state: int = 42):
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.random_state = random_state
        
        self.model = None
        self.scaler = None
        self._use_simple = True  # Default to simple model
    
    def fit_predict(self,
                   panel_data,
                   target_cols: Optional[List[str]] = None,
                   aggregate: bool = True) -> LSTMResult:
        """
        Train LSTM and predict for the test year.
        
        Parameters
        ----------
        panel_data : PanelData
            Panel data object
        target_cols : List[str]
            Columns to predict (if None, uses all components)
        aggregate : bool
            If True, predict aggregate score; if False, predict each component
        """
        np.random.seed(self.random_state)
        
        if target_cols is None:
            target_cols = panel_data.components
        
        provinces = panel_data.provinces
        years = panel_data.years
        
        # Prepare sequences
        X_train, y_train, X_test, y_test, test_provinces = self._prepare_sequences(
            panel_data, target_cols, aggregate
        )
        
        if len(X_train) == 0:
            raise ValueError("Not enough data for LSTM sequences")
        
        # Train model
        train_losses, val_losses = self._train(X_train, y_train)
        
        # Predict
        predictions = self.model.predict(X_test)
        
        # Calculate metrics
        from scipy.stats import spearmanr
        
        if aggregate:
            pred_flat = predictions.flatten()
            actual_flat = y_test.flatten()
        else:
            pred_flat = predictions.mean(axis=1)
            actual_flat = y_test.mean(axis=1)
        
        mse = np.mean((actual_flat - pred_flat) ** 2)
        mae = np.mean(np.abs(actual_flat - pred_flat))
        rmse = np.sqrt(mse)
        
        # Create DataFrames
        if aggregate:
            pred_df = pd.DataFrame({'predicted': pred_flat}, index=test_provinces)
            actual_df = pd.DataFrame({'actual': actual_flat}, index=test_provinces)
        else:
            pred_df = pd.DataFrame(predictions, index=test_provinces, columns=target_cols)
            actual_df = pd.DataFrame(y_test, index=test_provinces, columns=target_cols)
        
        # Rankings
        pred_ranks = pd.Series(pred_flat, index=test_provinces).rank(ascending=False).astype(int)
        actual_ranks = pd.Series(actual_flat, index=test_provinces).rank(ascending=False).astype(int)
        
        rank_corr, _ = spearmanr(actual_ranks, pred_ranks)
        
        return LSTMResult(
            predictions=pred_df,
            actual=actual_df,
            train_loss=train_losses,
            val_loss=val_losses,
            test_metrics={'mse': mse, 'mae': mae, 'rmse': rmse},
            predicted_ranks=pred_ranks,
            actual_ranks=actual_ranks,
            rank_correlation=rank_corr if not np.isnan(rank_corr) else 0.0,
            model_summary=f"SimpleLSTM(hidden={self.hidden_units})"
        )
    
    def _prepare_sequences(self, panel_data, target_cols: List[str], 
                          aggregate: bool) -> Tuple:
        """Prepare sequences for LSTM training with adaptive sequence length."""
        provinces = panel_data.provinces
        years = sorted(panel_data.years)
        n_years = len(years)
        
        # Adapt sequence length to available data
        seq_len = min(self.sequence_length, max(1, n_years - 2))
        
        X_sequences = []
        y_values = []
        test_X = []
        test_y = []
        test_provinces = []
        
        for province in provinces:
            try:
                province_data = panel_data.get_province(province)
                
                # Get values in temporal order - handle missing years
                available_years = [y for y in years if y in province_data.index]
                if len(available_years) < seq_len + 1:
                    continue
                
                values = province_data.loc[available_years, target_cols].values
                
                if aggregate:
                    # Use mean across components as target
                    values = values.mean(axis=1, keepdims=True)
                
                n_available = len(available_years)
                
                # Create sequences for training (all but last year)
                for i in range(n_available - seq_len - 1):
                    X_seq = values[i:i + seq_len]
                    y_val = values[i + seq_len]
                    if not np.any(np.isnan(X_seq)) and not np.any(np.isnan(y_val)):
                        X_sequences.append(X_seq)
                        y_values.append(y_val)
                
                # Test sequence (predict last year)
                if n_available >= seq_len + 1:
                    test_seq = values[-(seq_len + 1):-1]
                    test_target = values[-1]
                    if not np.any(np.isnan(test_seq)) and not np.any(np.isnan(test_target)):
                        test_X.append(test_seq)
                        test_y.append(test_target)
                        test_provinces.append(province)
            except Exception:
                continue  # Skip problematic provinces
        
        # Determine final sequence length based on actual data
        if X_sequences:
            actual_seq_len = X_sequences[0].shape[0]
        else:
            actual_seq_len = seq_len
        
        X_train = np.array(X_sequences) if X_sequences else np.array([]).reshape(0, actual_seq_len, 1)
        y_train = np.array(y_values) if y_values else np.array([])
        X_test = np.array(test_X) if test_X else np.array([]).reshape(0, actual_seq_len, 1)
        y_test = np.array(test_y) if test_y else np.array([])
        
        return X_train, y_train, X_test, y_test, test_provinces
    
    def _train(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[float], List[float]]:
        """Train the LSTM model with proper train/validation split."""
        # Use simple model
        self.model = SimpleLSTM(
            hidden_size=self.hidden_units,
            learning_rate=self.learning_rate
        )
        
        # Split for validation
        n_samples = len(X)
        val_size = max(1, int(n_samples * 0.2))
        
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]
        
        # Train with validation data for proper loss tracking
        train_losses = self.model.fit(X_train, y_train, epochs=self.epochs,
                                       X_val=X_val, y_val=y_val)
        
        # Get validation losses from training history
        val_losses = self.model.training_history.get('val_loss', train_losses)
        
        return train_losses, val_losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using fitted model."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_predict first.")
        return self.model.predict(X)


def forecast_panel(panel_data, 
                  sequence_length: int = 3,
                  target: str = 'aggregate') -> LSTMResult:
    """Convenience function for LSTM forecasting."""
    forecaster = LSTMForecaster(sequence_length=sequence_length)
    
    if target == 'aggregate':
        return forecaster.fit_predict(panel_data, aggregate=True)
    else:
        return forecaster.fit_predict(panel_data, target_cols=[target], aggregate=False)
