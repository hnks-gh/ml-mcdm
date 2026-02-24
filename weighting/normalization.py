# -*- coding: utf-8 -*-
"""
Normalization Methods for MCDM Weight Calculation

Provides normalization functions for preparing criteria data before weight calculation.
"""

import numpy as np
from typing import Optional


def global_min_max_normalize(
    X: np.ndarray,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Global min-max normalization with epsilon shift.
    
    Normalizes each column (criterion) to [0,1] scale using global min/max
    across all observations. Adds epsilon shift to ensure strictly positive
    values for logarithmic operations.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_observations, n_criteria)
        Input data matrix to normalize.
    epsilon : float, default=1e-10
        Small constant added after normalization to avoid exact zeros.
        Critical for entropy calculation where log(0) is undefined.
    
    Returns
    -------
    X_norm : np.ndarray, shape (n_observations, n_criteria)
        Normalized matrix with values in [epsilon, 1+epsilon].
    
    Notes
    -----
    **Why global normalization?**
    - Preserves temporal trends across time series panels
    - Year-by-year normalization would erase growth/decline patterns
    - Essential for MCDM forecasting applications
    
    **Formula:**
    ```
    x_norm = (x - min_global) / (max_global - min_global) + ε
    ```
    
    **Epsilon shift:**
    - Ensures no exact zeros for log operations
    - Does not affect relative magnitudes
    - Standard practice in information-theoretic methods
    
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 10], [2, 20], [3, 30]])
    >>> X_norm = global_min_max_normalize(X, epsilon=1e-10)
    >>> X_norm.min(axis=0)  # Should be epsilon
    array([1.e-10, 1.e-10])
    >>> X_norm.max(axis=0)  # Should be 1 + epsilon
    array([1., 1.])
    
    References
    ----------
    Standard practice in MCDM normalization. See:
    - Hwang & Yoon (1981). Multiple Attribute Decision Making.
    - Zavadskas & Turskis (2010). A new additive ratio assessment (ARAS)
      method in multicriteria decision-making. Technological and Economic
      Development of Economy, 16(2), 159-172.
    """
    # Compute global min/max for each criterion (NaN-aware: missing cells are ignored)
    col_min = np.nanmin(X, axis=0)
    col_max = np.nanmax(X, axis=0)

    # Handle all-NaN columns: nanmin/nanmax return NaN for columns where every
    # observation is missing.  Replace NaN col_min with 0 so that the
    # subtraction (X - col_min) does not propagate a spurious NaN into
    # the output for rows that happen to have a real value.
    all_nan_cols = np.isnan(col_min)          # True for fully-missing criteria
    col_min = np.where(all_nan_cols, 0.0, col_min)
    col_max = np.where(all_nan_cols, epsilon, col_max)

    # Handle constant columns (where min == max) and residual NaN denom
    denom = col_max - col_min
    denom = np.where(np.isnan(denom), epsilon, denom)  # Safety: remaining NaN
    denom[denom < epsilon] = epsilon

    # Normalize to [0, 1] and add epsilon shift
    X_norm = (X - col_min) / denom + epsilon

    return X_norm


class GlobalNormalizer:
    """
    Stateful normalizer that preserves min/max statistics for consistent
    transformation of new data.
    
    Useful when you need to normalize training data, then apply the same
    transformation to test/forecast data.
    
    Parameters
    ----------
    epsilon : float, default=1e-10
        Numerical stability constant.
    
    Attributes
    ----------
    col_min_ : np.ndarray or None
        Minimum values for each criterion (fitted on training data).
    col_max_ : np.ndarray or None
        Maximum values for each criterion (fitted on training data).
    is_fitted_ : bool
        Whether the normalizer has been fitted.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Training data
    >>> X_train = np.array([[1, 10], [2, 20], [3, 30]])
    >>> normalizer = GlobalNormalizer()
    >>> X_train_norm = normalizer.fit_transform(X_train)
    >>> 
    >>> # New data (e.g., forecast year)
    >>> X_new = np.array([[4, 40]])
    >>> X_new_norm = normalizer.transform(X_new)
    >>> # Uses same min/max from training data
    """
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
        self.col_min_: Optional[np.ndarray] = None
        self.col_max_: Optional[np.ndarray] = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray) -> 'GlobalNormalizer':
        """
        Fit normalizer by computing global min/max statistics.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_observations, n_criteria)
            Training data.
        
        Returns
        -------
        self : GlobalNormalizer
            Fitted normalizer instance.
        """
        # NaN-aware: missing cells are ignored when computing statistics
        self.col_min_ = np.nanmin(X, axis=0)
        self.col_max_ = np.nanmax(X, axis=0)
        # Guard all-NaN columns
        all_nan = np.isnan(self.col_min_)
        self.col_min_ = np.where(all_nan, 0.0, self.col_min_)
        self.col_max_ = np.where(all_nan, self.epsilon, self.col_max_)
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted min/max statistics.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_observations, n_criteria)
            Data to normalize.
        
        Returns
        -------
        X_norm : np.ndarray
            Normalized data.
        
        Raises
        ------
        RuntimeError
            If normalizer has not been fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("Normalizer must be fitted before transform(). "
                             "Call fit() or fit_transform() first.")
        
        # Handle constant columns
        denom = self.col_max_ - self.col_min_
        denom = np.where(np.isnan(denom), self.epsilon, denom)  # residual NaN guard
        denom[denom < self.epsilon] = self.epsilon

        # Apply fitted transformation
        X_norm = (X - self.col_min_) / denom + self.epsilon

        return X_norm
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit normalizer and transform data in one step.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_observations, n_criteria)
            Training data.
        
        Returns
        -------
        X_norm : np.ndarray
            Normalized training data.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        """
        Reverse normalization to recover original scale.
        
        Parameters
        ----------
        X_norm : np.ndarray, shape (n_observations, n_criteria)
            Normalized data (with epsilon shift).
        
        Returns
        -------
        X_original : np.ndarray
            Data in original scale.
        
        Raises
        ------
        RuntimeError
            If normalizer has not been fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("Normalizer must be fitted before inverse_transform().")
        
        # Remove epsilon shift
        X_no_eps = X_norm - self.epsilon
        
        # Reverse normalization
        denom = self.col_max_ - self.col_min_
        denom[denom < self.epsilon] = self.epsilon
        
        X_original = X_no_eps * denom + self.col_min_
        
        return X_original
