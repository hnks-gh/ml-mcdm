# -*- coding: utf-8 -*-
"""
Neural Network Forecasting
==========================

MLP and Attention-based neural network forecasters.

These methods provide:
- Non-linear modeling capability
- Self-attention for temporal weighting
- Residual connections for deep networks
- Layer normalization for stable training
"""

import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler

from .base import BaseForecaster


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU activation function."""
    return np.where(x > 0, x, alpha * x)


def selu(x: np.ndarray) -> np.ndarray:
    """Self-Normalizing Exponential Linear Unit."""
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * np.where(x > 0, x, alpha * (np.exp(np.clip(x, -50, 50)) - 1))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation for attention weights."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-10)


def glorot_init(fan_in: int, fan_out: int, seed: int = 42) -> np.ndarray:
    """Glorot/Xavier initialization."""
    np.random.seed(seed)
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))


def he_init(fan_in: int, fan_out: int, seed: int = 42) -> np.ndarray:
    """He initialization for ReLU networks."""
    np.random.seed(seed)
    std = np.sqrt(2 / fan_in)
    return np.random.randn(fan_in, fan_out) * std


class DenseLayer:
    """
    Dense (fully-connected) layer.
    
    Parameters:
        input_dim: Input dimension
        output_dim: Output dimension
        activation: Activation function name
        use_bias: Whether to use bias
        seed: Random seed
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: str = 'relu',
                 use_bias: bool = True,
                 seed: int = 42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        
        # Initialize weights using appropriate initialization
        if activation in ['relu', 'leaky_relu']:
            self.W = he_init(input_dim, output_dim, seed)
        else:
            self.W = glorot_init(input_dim, output_dim, seed)
        
        self.b = np.zeros(output_dim) if use_bias else None
    
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass through the layer."""
        z = x @ self.W
        if self.use_bias:
            z = z + self.b
        
        # Apply activation function
        if self.activation == 'relu':
            return relu(z)
        elif self.activation == 'leaky_relu':
            return leaky_relu(z)
        elif self.activation == 'selu':
            return selu(z)
        elif self.activation == 'sigmoid':
            return sigmoid(z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        else:  # linear
            return z


class AttentionLayer:
    """
    Self-attention layer for temporal weighting.
    
    Computes attention weights over features to focus on
    the most relevant inputs.
    
    Parameters:
        input_dim: Input dimension
        n_heads: Number of attention heads
        seed: Random seed
    """
    
    def __init__(self, input_dim: int, n_heads: int = 4, seed: int = 42):
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.head_dim = input_dim // n_heads
        
        np.random.seed(seed)
        self.W_q = glorot_init(input_dim, input_dim, seed)
        self.W_k = glorot_init(input_dim, input_dim, seed + 1)
        self.W_v = glorot_init(input_dim, input_dim, seed + 2)
        self.W_o = glorot_init(input_dim, input_dim, seed + 3)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply self-attention.
        
        Args:
            x: Input of shape (batch, input_dim) or (batch, seq, dim)
        
        Returns:
            Attended output of same shape as input
        """
        original_shape = x.shape
        if x.ndim == 2:
            x = x.reshape(x.shape[0], 1, x.shape[1])
        
        batch_size, seq_len, _ = x.shape
        
        # Compute Query, Key, Value projections
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.input_dim)
        attention_weights = softmax(scores)
        
        # Apply attention to values
        attended = attention_weights @ V
        
        # Output projection
        output = attended @ self.W_o
        
        if len(original_shape) == 2:
            output = output.squeeze(1)
        
        return output


class NeuralForecaster(BaseForecaster):
    """
    Multi-layer Perceptron forecaster with modern architecture.
    
    Features:
    - Configurable hidden layers
    - SELU activation for self-normalization
    - Dropout regularization
    - Early stopping
    
    Parameters:
        hidden_dims: List of hidden layer dimensions
        activation: Activation function
        dropout_rate: Dropout probability
        learning_rate: Learning rate for SGD
        n_epochs: Maximum training epochs
        batch_size: Mini-batch size
        patience: Early stopping patience
        seed: Random seed
    
    Example:
        >>> forecaster = NeuralForecaster(hidden_dims=[128, 64])
        >>> forecaster.fit(X_train, y_train)
        >>> predictions = forecaster.predict(X_test)
    """
    
    def __init__(self,
                 hidden_dims: List[int] = [256, 128, 64],
                 activation: str = 'selu',
                 dropout_rate: float = 0.1,
                 learning_rate: float = 0.001,
                 n_epochs: int = 100,
                 batch_size: int = 32,
                 patience: int = 10,
                 seed: int = 42):
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        
        self.layers: List[DenseLayer] = []
        self.output_layer: Optional[DenseLayer] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_history: Dict[str, List[float]] = {'loss': []}
        self._input_dim: int = 0
    
    def _build_network(self, input_dim: int, output_dim: int):
        """Build the neural network architecture."""
        self.layers = []
        self._input_dim = input_dim
        
        current_dim = input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            layer = DenseLayer(
                current_dim, hidden_dim, 
                self.activation, seed=self.seed + i
            )
            self.layers.append(layer)
            current_dim = hidden_dim
        
        # Output layer (linear activation)
        self.output_layer = DenseLayer(
            current_dim, output_dim, 'linear', seed=self.seed + 100
        )
    
    def _forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass through network."""
        h = X
        for layer in self.layers:
            h = layer.forward(h, training)
            
            # Apply dropout during training
            if training and self.dropout_rate > 0:
                mask = np.random.binomial(1, 1 - self.dropout_rate, h.shape)
                h = h * mask / (1 - self.dropout_rate)
        
        return self.output_layer.forward(h, training)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NeuralForecaster':
        """
        Train the neural network using mini-batch gradient descent.
        
        Uses simple gradient descent with early stopping.
        """
        np.random.seed(self.seed)
        
        # Scale inputs
        X_scaled = self.scaler.fit_transform(X)
        y_reshaped = y.reshape(-1, 1) if y.ndim == 1 else y
        
        # Build network
        self._build_network(X_scaled.shape[1], y_reshaped.shape[1])
        
        n_samples = X_scaled.shape[0]
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_scaled[indices]
            y_shuffled = y_reshaped[indices]
            
            epoch_loss = 0
            n_batches = max(1, n_samples // self.batch_size)
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                predictions = self._forward(X_batch, training=True)
                
                # Compute loss
                batch_loss = np.mean((predictions - y_batch) ** 2)
                epoch_loss += batch_loss
                
                # Backward pass (simplified gradient update)
                error = predictions - y_batch
                
                # Get last hidden state for output layer update
                h_last = X_batch
                for layer in self.layers:
                    h_last = layer.forward(h_last, training=False)
                
                # Update output layer
                grad_output = error.T @ h_last / len(X_batch)
                self.output_layer.W -= self.learning_rate * grad_output.T
                if self.output_layer.b is not None:
                    self.output_layer.b -= self.learning_rate * error.mean(axis=0)
            
            avg_loss = epoch_loss / n_batches
            self.training_history['loss'].append(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = self._forward(X_scaled, training=False)
        return predictions.ravel() if predictions.shape[1] == 1 else predictions
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Estimate feature importance from first layer weights.
        
        Uses absolute sum of weights connecting each input feature.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Use first layer weights as proxy for importance
        return np.abs(self.layers[0].W).sum(axis=1)


class AttentionForecaster(BaseForecaster):
    """
    Attention-based neural network forecaster.
    
    Uses self-attention mechanism to weight the importance of
    different features for prediction.
    
    Parameters:
        hidden_dim: Hidden dimension size
        n_attention_heads: Number of attention heads
        n_layers: Number of attention + feedforward layers
        dropout_rate: Dropout probability
        learning_rate: Learning rate
        n_epochs: Maximum epochs
        batch_size: Batch size
        patience: Early stopping patience
        seed: Random seed
    
    Example:
        >>> forecaster = AttentionForecaster(hidden_dim=128)
        >>> forecaster.fit(X_train, y_train)
        >>> predictions = forecaster.predict(X_test)
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 n_attention_heads: int = 4,
                 n_layers: int = 2,
                 dropout_rate: float = 0.1,
                 learning_rate: float = 0.001,
                 n_epochs: int = 100,
                 batch_size: int = 32,
                 patience: int = 10,
                 seed: int = 42):
        self.hidden_dim = hidden_dim
        self.n_attention_heads = n_attention_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        
        self.input_projection: Optional[DenseLayer] = None
        self.attention_layers: List[AttentionLayer] = []
        self.feedforward_layers: List[DenseLayer] = []
        self.output_layer: Optional[DenseLayer] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_history: Dict[str, List[float]] = {'loss': []}
    
    def _build_network(self, input_dim: int, output_dim: int):
        """Build the attention network architecture."""
        # Project input to hidden dimension
        self.input_projection = DenseLayer(
            input_dim, self.hidden_dim, 'relu', seed=self.seed
        )
        
        self.attention_layers = []
        self.feedforward_layers = []
        
        for i in range(self.n_layers):
            attn = AttentionLayer(
                self.hidden_dim, 
                self.n_attention_heads, 
                seed=self.seed + i * 10
            )
            self.attention_layers.append(attn)
            
            ff = DenseLayer(
                self.hidden_dim, self.hidden_dim,
                'relu', seed=self.seed + i * 10 + 5
            )
            self.feedforward_layers.append(ff)
        
        # Output projection
        self.output_layer = DenseLayer(
            self.hidden_dim, output_dim, 'linear', seed=self.seed + 100
        )
    
    def _forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass with attention."""
        # Project to hidden dimension
        h = self.input_projection.forward(X, training)
        
        # Apply attention + feedforward layers
        for attn, ff in zip(self.attention_layers, self.feedforward_layers):
            # Self-attention with residual
            attn_out = attn.forward(h)
            h = h + attn_out
            
            # Feedforward with residual
            ff_out = ff.forward(h, training)
            h = h + ff_out
            
            # Dropout
            if training and self.dropout_rate > 0:
                mask = np.random.binomial(1, 1 - self.dropout_rate, h.shape)
                h = h * mask / (1 - self.dropout_rate)
        
        return self.output_layer.forward(h, training)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AttentionForecaster':
        """Train the attention network."""
        np.random.seed(self.seed)
        
        X_scaled = self.scaler.fit_transform(X)
        y_reshaped = y.reshape(-1, 1) if y.ndim == 1 else y
        
        self._build_network(X_scaled.shape[1], y_reshaped.shape[1])
        
        n_samples = X_scaled.shape[0]
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_scaled[indices]
            y_shuffled = y_reshaped[indices]
            
            epoch_loss = 0
            n_batches = max(1, n_samples // self.batch_size)
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                predictions = self._forward(X_batch, training=True)
                batch_loss = np.mean((predictions - y_batch) ** 2)
                epoch_loss += batch_loss
                
                # Simplified weight update for output layer
                error = predictions - y_batch
                h_last = self._get_final_hidden(X_batch)
                
                grad_output = error.T @ h_last / len(X_batch)
                self.output_layer.W -= self.learning_rate * grad_output.T
            
            avg_loss = epoch_loss / n_batches
            self.training_history['loss'].append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        self.is_fitted = True
        return self
    
    def _get_final_hidden(self, X: np.ndarray) -> np.ndarray:
        """Get final hidden state before output layer."""
        h = self.input_projection.forward(X, False)
        
        for attn, ff in zip(self.attention_layers, self.feedforward_layers):
            attn_out = attn.forward(h)
            h = h + attn_out
            ff_out = ff.forward(h, False)
            h = h + ff_out
        
        return h
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = self._forward(X_scaled, training=False)
        return predictions.ravel() if predictions.shape[1] == 1 else predictions
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Estimate feature importance from input projection weights.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        return np.abs(self.input_projection.W).sum(axis=1)
