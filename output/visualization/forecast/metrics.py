# -*- coding: utf-8 -*-
"""
Forecast Visualization Metrics

Centralized, deterministic metric implementations used across all chart modules.
Single source of truth for R², RMSE, MAE, MAPE, conformal quantile, bootstrap CI.

All functions:
- Accept numpy arrays or array-like
- Return Python floats or numpy arrays (callable from plotting code)
- Are deterministic and reproducible
- Include guards against edge cases (constant targets, NaN propagation)

This module is imported by:
- All chart modules (accuracy.py, ensemble.py, uncertainty.py, etc.)
- Validators (cross-check consistency)
- Tests (metric reference verification)
"""

import numpy as np
from typing import Tuple, Optional, List


# ============================================================================
# Regression Metrics
# ============================================================================

def r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    R² (Coefficient of Determination) with epsilon guard.
    
    R² = 1 - (SS_res / SS_tot)
    
    where:
    - SS_res = Σ(y_true - y_pred)²
    - SS_tot = Σ(y_true - mean(y_true))²
    
    Properties:
    - R² = 1.0 means perfect prediction
    - R² = 0.0 means predicting the mean (baseline)
    - R² < 0.0 means worse than baseline
    - Handles constant targets gracefully via eps guard
    
    Args:
        y_true: Ground truth values. Shape (n,).
        y_pred: Predictions. Shape (n,).
        eps: Small value to guard against division by zero (SS_tot = 0).
    
    Returns:
        R² score as float.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    
    return float(1.0 - ss_res / (ss_tot + eps))


def rmse_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Root Mean Squared Error (RMSE).
    
    RMSE = sqrt(mean((y_true - y_pred)²))
    
    Properties:
    - RMSE >= 0
    - Sensitive to outliers
    - In same units as y_true
    
    Args:
        y_true: Ground truth values. Shape (n,).
        y_pred: Predictions. Shape (n,).
    
    Returns:
        RMSE as float.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Mean Absolute Error (MAE).
    
    MAE = mean(|y_true - y_pred|)
    
    Properties:
    - MAE >= 0
    - Less sensitive to outliers than RMSE (MAE <= RMSE)
    - In same units as y_true
    
    Args:
        y_true: Ground truth values. Shape (n,).
        y_pred: Predictions. Shape (n,).
    
    Returns:
        MAE as float.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    return float(np.mean(np.abs(y_true - y_pred)))


def mape_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-9,
) -> float:
    """
    Mean Absolute Percentage Error (MAPE).
    
    MAPE = 100 * mean(|y_true - y_pred| / |y_true|)
    
    Handles division by zero by clipping denominator to eps.
    
    Properties:
    - MAPE is in percentage (0-100% or unbounded)
    - Zero denominator values are treated as eps (neutral zone)
    - Interpretation: average percent error magnitude
    
    Args:
        y_true: Ground truth values. Shape (n,).
        y_pred: Predictions. Shape (n,).
        eps: Small value for zero-denominator guard.
    
    Returns:
        MAPE as percentage float.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    denom = np.where(np.abs(y_true) < eps, 1.0, np.abs(y_true))
    return float(100.0 * np.mean(np.abs(y_true - y_pred) / denom))


def bias_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Bias (Mean Signed Error).
    
    Bias = mean(y_true - y_pred)
    
    Properties:
    - Positive bias: predictions are on average too low
    - Negative bias: predictions are on average too high
    - Zero bias: unbiased predictor (on average)
    
    Args:
        y_true: Ground truth values. Shape (n,).
        y_pred: Predictions. Shape (n,).
    
    Returns:
        Bias as float.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    return float(np.mean(y_true - y_pred))


def correlation_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Pearson correlation coefficient between y_true and y_pred.
    
    Correlation = cov(y_true, y_pred) / (std(y_true) * std(y_pred))
    
    Properties:
    - Ranges from -1 to +1
    - Measures linear association strength
    - Invariant to scale/shift
    
    Args:
        y_true: Ground truth values. Shape (n,).
        y_pred: Predictions. Shape (n,).
    
    Returns:
        Correlation coefficient as float.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    # Use std with ddof=0 (population std)
    std_true = np.std(y_true, ddof=0)
    std_pred = np.std(y_pred, ddof=0)
    
    if std_true == 0 or std_pred == 0:
        return 0.0
    
    cov = np.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    return float(cov / (std_true * std_pred))


# ============================================================================
# Conformal Prediction Metrics
# ============================================================================

def conformal_quantile(
    residuals: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    Finite-sample conformal prediction quantile.
    
    Computes the (ceiling((n+1)*(1-α)) / n)-th order statistic of |residuals|.
    This ensures a coverage guarantee that empirical coverage >= nominal coverage
    (up to a small tolerance).
    
    Formula:
        q_idx = ceil((n+1) * (1-α)) - 1  (converted to 0-indexed)
        quantile = sorted_residuals[q_idx]
    
    Properties:
    - Monotonic in α (larger α → smaller quantile)
    - Handles finite samples correctly (no asymptotic assumptions)
    
    Args:
        residuals: Absolute residuals. Shape (n,). dtype float.
        alpha: Miscoverage rate (e.g., 0.05 for 95% coverage). Range (0, 1).
    
    Returns:
        Conformal quantile as float (in units of residuals).
    """
    residuals = np.asarray(residuals, dtype=float).ravel()
    n = len(residuals)
    
    if n == 0:
        return 0.0
    
    # Finite-sample index (1-indexed nominally, convert to 0-indexed)
    q_idx = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    q_idx = min(max(q_idx, 0), n - 1)  # Clamp to [0, n-1]
    
    sorted_residuals = np.sort(residuals)
    return float(sorted_residuals[q_idx])


def conformal_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> float:
    """
    Empirical coverage rate given predictions and a quantile.
    
    Coverage = fraction of samples where |y_true - y_pred| <= quantile.
    
    Args:
        y_true: Ground truth. Shape (n,).
        y_pred: Predictions. Shape (n,).
        quantile: Width parameter (threshold for coverage).
    
    Returns:
        Coverage as float in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    residuals = np.abs(y_true - y_pred)
    
    return float(np.mean(residuals <= quantile))


def conformal_interval_size(
    quantile: float,
) -> float:
    """
    Size of conformal prediction interval.
    
    Interval = [y_pred - quantile, y_pred + quantile]
    Size = 2 * quantile
    
    Args:
        quantile: Half-width from conformal quantile computation.
    
    Returns:
        Full interval width as float.
    """
    return float(2.0 * quantile)


# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func,
    n_bootstrap: int = 1000,
    ci_lower: float = 2.5,
    ci_upper: float = 97.5,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a metric.
    
    Resamples (y_true, y_pred) pairs with replacement n_bootstrap times,
    computes metric on each resample, then returns percentiles.
    
    Args:
        y_true: Ground truth. Shape (n,).
        y_pred: Predictions. Shape (n,).
        metric_func: Function with signature metric_func(y_true, y_pred) -> float.
        n_bootstrap: Number of bootstrap resamples.
        ci_lower: Lower percentile for CI (e.g., 2.5).
        ci_upper: Upper percentile for CI (e.g., 97.5).
        random_state: Seed for reproducibility.
    
    Returns:
        (median, ci_lower_val, ci_upper_val) as tuple of floats.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    rng = np.random.RandomState(random_state)
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        metric_val = metric_func(y_true[idx], y_pred[idx])
        bootstrap_metrics.append(metric_val)
    
    bootstrap_metrics = np.array(bootstrap_metrics)
    
    median = float(np.percentile(bootstrap_metrics, 50.0))
    lower = float(np.percentile(bootstrap_metrics, ci_lower))
    upper = float(np.percentile(bootstrap_metrics, ci_upper))
    
    return median, lower, upper


# ============================================================================
# Metric Aggregation and Summary
# ============================================================================

def compute_metric_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute standard metric suite (R², RMSE, MAE, MAPE, Bias, Correlation).
    
    Args:
        y_true: Ground truth. Shape (n,).
        y_pred: Predictions. Shape (n,).
    
    Returns:
        Dictionary with keys: 'r2', 'rmse', 'mae', 'mape', 'bias', 'correlation'.
    """
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': rmse_score(y_true, y_pred),
        'mae': mae_score(y_true, y_pred),
        'mape': mape_score(y_true, y_pred),
        'bias': bias_score(y_true, y_pred),
        'correlation': correlation_score(y_true, y_pred),
    }


__all__ = [
    'r2_score',
    'rmse_score',
    'mae_score',
    'mape_score',
    'bias_score',
    'correlation_score',
    'conformal_quantile',
    'conformal_coverage',
    'conformal_interval_size',
    'bootstrap_ci',
    'compute_metric_summary',
]
