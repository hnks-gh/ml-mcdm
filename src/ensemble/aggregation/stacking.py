# -*- coding: utf-8 -*-
"""Stacking ensemble meta-learner for combining predictions."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StackingResult:
    """Result container for Stacking Ensemble.
    
    Attributes
    ----------
    final_predictions : np.ndarray
        Combined predictions from the meta-model
    meta_model_weights : np.ndarray
        Learned weights for each base model
    base_model_predictions : Dict[str, np.ndarray]
        Original predictions from each base model
    cv_scores : Dict[str, float]
        Cross-validation R² scores for each base model
    meta_model_r2 : float
        R² score of the meta-model
    meta_model_mse : float
        Mean squared error of the meta-model
    """
    final_predictions: np.ndarray
    meta_model_weights: np.ndarray
    base_model_predictions: Dict[str, np.ndarray]
    cv_scores: Dict[str, float]
    meta_model_r2: float
    meta_model_mse: float
    
    def summary(self) -> str:
        """Generate summary string of stacking results."""
        lines = [
            f"\n{'='*60}",
            "STACKING ENSEMBLE RESULTS",
            f"{'='*60}",
            f"\nMeta-Model Performance:",
            f"  R²: {self.meta_model_r2:.4f}",
            f"  MSE: {self.meta_model_mse:.6f}",
            f"\nBase Model Cross-Validation Scores:"
        ]
        for name, score in self.cv_scores.items():
            lines.append(f"  {name}: {score:.4f}")
        lines.append(f"\nMeta-Model Weights:")
        for i, (name, weight) in enumerate(zip(self.base_model_predictions.keys(), 
                                               self.meta_model_weights)):
            lines.append(f"  {name}: {weight:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)


class StackingEnsemble:
    """
    Stacking Ensemble Meta-Learner.
    
    Stacking (stacked generalization) combines predictions from multiple
    base models using a second-level meta-learner. This creates a two-tier
    model that can learn optimal combinations of base model predictions.
    
    Mathematical Formulation
    ------------------------
    Given base model predictions f₁(x), f₂(x), ..., fₖ(x):
    
    Final prediction:
        ŷ = g(f₁(x), f₂(x), ..., fₖ(x))
        
    where g is the meta-learner (e.g., ridge regression):
        g(z) = Σᵢ wᵢzᵢ + b
    
    The meta-learner is typically trained on out-of-fold predictions
    to avoid overfitting.
    
    Supported Meta-Learners
    -----------------------
    - 'ridge': Ridge regression with L2 regularization
    - 'bayesian': Bayesian ridge regression
    - 'elastic': Elastic net (L1 + L2)
    - 'linear': Ordinary least squares
    
    Example
    -------
    >>> from src.ensemble.aggregation import StackingEnsemble
    >>> 
    >>> base_predictions = {
    ...     'RF': rf_model.predict(X),
    ...     'GBM': gbm_model.predict(X),
    ...     'LSTM': lstm_model.predict(X)
    ... }
    >>> 
    >>> stacker = StackingEnsemble(meta_learner='ridge', alpha=1.0)
    >>> result = stacker.fit_predict(base_predictions, y_true)
    >>> print(result.summary())
    
    References
    ----------
    [1] Wolpert, D.H. (1992). "Stacked Generalization"
    [2] Breiman, L. (1996). "Stacked Regressions"
    [3] Van der Laan et al. (2007). "Super Learner"
    """
    
    def __init__(self,
                 meta_learner: str = 'ridge',
                 cv_folds: int = 5,
                 alpha: float = 1.0,
                 use_features: bool = True):
        """
        Initialize Stacking Ensemble.
        
        Parameters
        ----------
        meta_learner : str, default='ridge'
            Type of meta-learner:
            - 'ridge': Ridge regression
            - 'bayesian': Bayesian ridge
            - 'elastic': Elastic net
            - 'linear': OLS
        cv_folds : int, default=5
            Number of cross-validation folds
        alpha : float, default=1.0
            Regularization strength for ridge/elastic
        use_features : bool, default=True
            Whether to include original features in meta-model
        """
        self.meta_learner = meta_learner
        self.cv_folds = cv_folds
        self.alpha = alpha
        self.use_features = use_features
        self.meta_weights_ = None
        self.meta_bias_ = None
    
    def fit_predict(self,
                   base_predictions: Dict[str, np.ndarray],
                   target: np.ndarray,
                   features: Optional[np.ndarray] = None) -> StackingResult:
        """
        Fit stacking ensemble and generate predictions.
        
        Parameters
        ----------
        base_predictions : Dict[str, np.ndarray]
            Dictionary of base model predictions {name: predictions}
        target : np.ndarray
            Target values for training the meta-learner
        features : np.ndarray, optional
            Original features to include in meta-model
            
        Returns
        -------
        StackingResult
            Stacking ensemble results
        """
        # Stack base predictions into matrix
        pred_names = list(base_predictions.keys())
        pred_matrix = np.column_stack([base_predictions[name] for name in pred_names])
        
        # Add features if requested
        if self.use_features and features is not None:
            meta_features = np.hstack([pred_matrix, features])
        else:
            meta_features = pred_matrix
        
        # Cross-validation scores for base models
        cv_scores = self._calculate_cv_scores(base_predictions, target)
        
        # Train meta-learner
        meta_weights, meta_bias = self._train_meta_learner(pred_matrix, target)
        
        # Generate final predictions
        final_predictions = pred_matrix @ meta_weights + meta_bias
        
        # Store weights
        self.meta_weights_ = meta_weights
        self.meta_bias_ = meta_bias
        
        # Calculate meta-model performance
        r2 = self._calculate_r2(target, final_predictions)
        mse = np.mean((target - final_predictions) ** 2)
        
        return StackingResult(
            final_predictions=final_predictions,
            meta_model_weights=meta_weights,
            base_model_predictions=base_predictions,
            cv_scores=cv_scores,
            meta_model_r2=r2,
            meta_model_mse=mse
        )
    
    def predict(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate predictions using trained meta-learner.
        
        Parameters
        ----------
        base_predictions : Dict[str, np.ndarray]
            New base model predictions
            
        Returns
        -------
        np.ndarray
            Meta-learner predictions
        """
        if self.meta_weights_ is None:
            raise ValueError("Model not fitted. Call fit_predict first.")
        
        pred_matrix = np.column_stack([base_predictions[name] 
                                        for name in base_predictions.keys()])
        
        return pred_matrix @ self.meta_weights_ + self.meta_bias_
    
    def _train_meta_learner(self, 
                           X: np.ndarray, 
                           y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Train meta-learner model."""
        n_features = X.shape[1]
        
        if self.meta_learner == 'ridge':
            weights, bias = self._ridge_regression(X, y, self.alpha)
        elif self.meta_learner == 'bayesian':
            weights, bias = self._bayesian_ridge(X, y)
        elif self.meta_learner == 'elastic':
            weights, bias = self._elastic_net(X, y, self.alpha)
        else:
            weights, bias = self._ols_regression(X, y)
        
        # Ensure non-negative weights for interpretability
        weights = np.maximum(weights, 0)
        
        # Normalize weights to sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_features) / n_features
        
        return weights, bias
    
    def _ridge_regression(self, 
                         X: np.ndarray, 
                         y: np.ndarray, 
                         alpha: float) -> Tuple[np.ndarray, float]:
        """Ridge regression with L2 regularization."""
        n_samples, n_features = X.shape
        
        # Center data
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean
        
        # Ridge solution: (X'X + αI)^{-1} X'y
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered
        
        identity = np.eye(n_features)
        weights = np.linalg.solve(XtX + alpha * identity, Xty)
        
        # Calculate bias
        bias = y_mean - X_mean @ weights
        
        return weights, bias
    
    def _bayesian_ridge(self, 
                       X: np.ndarray, 
                       y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Bayesian ridge regression with evidence maximization."""
        n_samples, n_features = X.shape
        
        # Initialize hyperparameters
        alpha = 1.0
        lambda_ = 1.0
        
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean
        
        # Iterative updates (simplified EM)
        for _ in range(50):
            # Posterior precision
            A = alpha * np.eye(n_features) + lambda_ * X_centered.T @ X_centered
            
            # Posterior mean (weights)
            weights = lambda_ * np.linalg.solve(A, X_centered.T @ y_centered)
            
            # Update hyperparameters
            y_pred = X_centered @ weights
            residual_var = np.mean((y_centered - y_pred) ** 2)
            
            gamma = n_features - alpha * np.trace(np.linalg.inv(A))
            alpha = gamma / (np.sum(weights ** 2) + 1e-10)
            lambda_ = (n_samples - gamma) / (n_samples * residual_var + 1e-10)
        
        bias = y_mean - X_mean @ weights
        
        return weights, bias
    
    def _elastic_net(self, 
                    X: np.ndarray, 
                    y: np.ndarray,
                    alpha: float) -> Tuple[np.ndarray, float]:
        """Elastic Net with L1 + L2 regularization using coordinate descent."""
        n_samples, n_features = X.shape
        l1_ratio = 0.5
        
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean
        
        # Coordinate descent
        weights = np.zeros(n_features)
        
        for _ in range(100):
            for j in range(n_features):
                # Compute partial residual
                residual = y_centered - X_centered @ weights + X_centered[:, j] * weights[j]
                
                # Compute raw update
                rho = X_centered[:, j] @ residual
                
                # Soft thresholding
                threshold = alpha * l1_ratio * n_samples
                denom = np.sum(X_centered[:, j]**2) + alpha * (1 - l1_ratio) * n_samples
                
                if rho < -threshold:
                    weights[j] = (rho + threshold) / denom
                elif rho > threshold:
                    weights[j] = (rho - threshold) / denom
                else:
                    weights[j] = 0
        
        bias = y_mean - X_mean @ weights
        
        return weights, bias
    
    def _ols_regression(self, 
                       X: np.ndarray, 
                       y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Ordinary least squares regression."""
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean
        
        # OLS solution with regularization for numerical stability
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered
        
        weights = np.linalg.solve(XtX + 1e-8 * np.eye(XtX.shape[0]), Xty)
        
        bias = y_mean - X_mean @ weights
        
        return weights, bias
    
    def _calculate_cv_scores(self, 
                            base_predictions: Dict[str, np.ndarray],
                            target: np.ndarray) -> Dict[str, float]:
        """Calculate R² scores for base models."""
        scores = {}
        
        for name, preds in base_predictions.items():
            r2 = self._calculate_r2(target, preds)
            scores[name] = r2
        
        return scores
    
    def _calculate_r2(self, 
                     y_true: np.ndarray, 
                     y_pred: np.ndarray) -> float:
        """Calculate R² score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - ss_res / ss_tot


class TemporalStackingEnsemble(StackingEnsemble):
    """
    Stacking Ensemble with Temporal Awareness.
    
    Extends StackingEnsemble to handle panel data by applying
    temporal weighting that gives more importance to recent observations.
    
    Mathematical Formulation
    ------------------------
    Sample weight for observation at time t:
        w(t) = γ^(T - t)
        
    where:
    - γ = temporal decay factor (0 < γ ≤ 1)
    - T = maximum time period
    
    This downweights older observations, making the model
    more responsive to recent patterns.
    
    Example
    -------
    >>> from src.ensemble.aggregation import TemporalStackingEnsemble
    >>> 
    >>> stacker = TemporalStackingEnsemble(
    ...     meta_learner='ridge',
    ...     temporal_decay=0.9
    ... )
    >>> result = stacker.fit_predict_temporal(
    ...     base_predictions,
    ...     target,
    ...     time_indices
    ... )
    """
    
    def __init__(self,
                 meta_learner: str = 'ridge',
                 cv_folds: int = 5,
                 alpha: float = 1.0,
                 temporal_decay: float = 0.9):
        """
        Initialize Temporal Stacking Ensemble.
        
        Parameters
        ----------
        meta_learner : str, default='ridge'
            Type of meta-learner
        cv_folds : int, default=5
            Number of CV folds
        alpha : float, default=1.0
            Regularization strength
        temporal_decay : float, default=0.9
            Decay factor for older observations (0 < γ ≤ 1)
            Values closer to 1 mean slower decay
        """
        super().__init__(meta_learner, cv_folds, alpha)
        self.temporal_decay = temporal_decay
    
    def fit_predict_temporal(self,
                            base_predictions: Dict[str, np.ndarray],
                            target: np.ndarray,
                            time_indices: np.ndarray) -> StackingResult:
        """
        Fit with temporal weighting.
        
        Parameters
        ----------
        base_predictions : Dict[str, np.ndarray]
            Base model predictions
        target : np.ndarray
            Target values
        time_indices : np.ndarray
            Time period index for each observation
            
        Returns
        -------
        StackingResult
            Stacking results with temporal weighting applied
        """
        # Calculate temporal weights
        unique_times = np.unique(time_indices)
        max_time = unique_times.max()
        
        weights = np.array([
            self.temporal_decay ** (max_time - t) 
            for t in time_indices
        ])
        weights = weights / weights.sum() * len(weights)
        
        # Apply weights via sample transformation
        sqrt_weights = np.sqrt(weights)
        
        weighted_preds = {
            name: preds * sqrt_weights
            for name, preds in base_predictions.items()
        }
        weighted_target = target * sqrt_weights
        
        # Fit on weighted data
        return self.fit_predict(weighted_preds, weighted_target)


def stacking_ensemble(base_predictions: Dict[str, np.ndarray],
                     target: np.ndarray,
                     meta_learner: str = 'ridge',
                     alpha: float = 1.0) -> StackingResult:
    """
    Convenience function for stacking ensemble.
    
    Parameters
    ----------
    base_predictions : Dict[str, np.ndarray]
        Base model predictions
    target : np.ndarray
        Target values
    meta_learner : str, default='ridge'
        Meta-learner type
    alpha : float, default=1.0
        Regularization strength
        
    Returns
    -------
    StackingResult
        Stacking results
    """
    stacker = StackingEnsemble(meta_learner=meta_learner, alpha=alpha)
    return stacker.fit_predict(base_predictions, target)
