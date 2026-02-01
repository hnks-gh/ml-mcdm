"""
Random Forest Time-Series Module

Specialized Random Forest implementation for time-series panel data
with proper temporal cross-validation.
"""

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr


class TimeSeriesSplit:
    """
    Simple time series cross-validator.
    
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    """
    
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
    
    def split(self, years: np.ndarray):
        """
        Generate indices to split data into training and test sets.
        
        Parameters
        ----------
        years : np.ndarray
            Array of year values for each sample.
            
        Yields
        ------
        train : np.ndarray
            Training set indices.
        test : np.ndarray
            Test set indices.
        """
        unique_years = np.sort(np.unique(years))
        n_years = len(unique_years)
        
        for i in range(1, min(self.n_splits + 1, n_years)):
            train_years = unique_years[:n_years - self.n_splits + i - 1]
            test_year = unique_years[n_years - self.n_splits + i - 1]
            
            train_idx = np.where(np.isin(years, train_years))[0]
            test_idx = np.where(years == test_year)[0]
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


@dataclass
class RandomForestTSResult:
    """
    Result container for Random Forest time-series analysis.
    
    Attributes
    ----------
    model : RandomForestRegressor
        Fitted Random Forest model.
    feature_importance : pd.Series
        Feature importance scores sorted by importance.
    cv_scores : Dict[str, List[float]]
        Cross-validation scores for each fold.
    test_predictions : pd.Series
        Predictions on test set.
    test_actual : pd.Series
        Actual values for test set.
    test_metrics : Dict[str, float]
        Test set metrics (R², MSE, MAE).
    predicted_ranks : pd.Series
        Predicted rankings.
    actual_ranks : pd.Series
        Actual rankings.
    rank_correlation : float
        Spearman rank correlation between predicted and actual.
    feature_names : List[str]
        Names of features used.
    """
    model: RandomForestRegressor
    feature_importance: pd.Series
    cv_scores: Dict[str, List[float]]
    test_predictions: pd.Series
    test_actual: pd.Series
    test_metrics: Dict[str, float]
    predicted_ranks: pd.Series
    actual_ranks: pd.Series
    rank_correlation: float
    feature_names: List[str]


class RandomForestTS:
    """
    Random Forest for time-series panel data.
    
    Provides time-series aware cross-validation and feature importance
    analysis for panel data with province-year structure.
    
    Parameters
    ----------
    n_estimators : int, default=200
        Number of trees in the forest.
    max_depth : int, optional
        Maximum depth of trees.
    min_samples_split : int, default=5
        Minimum samples required to split an internal node.
    min_samples_leaf : int, default=2
        Minimum samples required to be at a leaf node.
    max_features : str, default='sqrt'
        Number of features to consider for the best split.
    n_splits : int, default=5
        Number of time-series cross-validation splits.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Examples
    --------
    >>> rf_ts = RandomForestTS(n_estimators=100, n_splits=3)
    >>> result = rf_ts.fit_predict(
    ...     panel_data=df,
    ...     target_col='topsis_score',
    ...     feature_cols=['component1', 'component2'],
    ...     province_col='Province',
    ...     year_col='Year'
    ... )
    >>> print(result.feature_importance)
    >>> print(result.test_metrics)
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = 'sqrt',
        n_splits: int = 5,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_splits = n_splits
        self.random_state = random_state
        
        self.model: Optional[RandomForestRegressor] = None
        self.feature_names: List[str] = []
    
    def fit_predict(
        self,
        panel_data: Any,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        province_col: str = 'Province',
        year_col: str = 'Year'
    ) -> RandomForestTSResult:
        """
        Fit model with time-series CV and return comprehensive results.
        
        Parameters
        ----------
        panel_data : PanelData or DataFrame
            Panel data with province/year structure.
        target_col : str
            Target variable column name (e.g., TOPSIS score).
        feature_cols : List[str], optional
            Feature column names. If None, auto-detects.
        province_col : str, default='Province'
            Name of province identifier column.
        year_col : str, default='Year'
            Name of year column.
            
        Returns
        -------
        RandomForestTSResult
            Comprehensive results including model, importance, and metrics.
        """
        # Get DataFrame
        if hasattr(panel_data, 'long'):
            df = panel_data.long.copy()
            if feature_cols is None:
                feature_cols = panel_data.components
        else:
            df = panel_data.copy()
            if feature_cols is None:
                exclude = [province_col, year_col, target_col]
                feature_cols = [c for c in df.columns if c not in exclude]
        
        self.feature_names = feature_cols
        
        # Prepare data
        X = df[feature_cols].values
        y = df[target_col].values
        years = df[year_col].values
        provinces = df[province_col].values
        
        # Time-series CV
        ts_split = TimeSeriesSplit(n_splits=self.n_splits)
        cv_scores = {'r2': [], 'mse': [], 'mae': [], 'rank_corr': []}
        
        for train_idx, test_idx in ts_split.split(years):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            rf = self._create_model()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rf.fit(X_train, y_train)
            
            y_pred = rf.predict(X_test)
            
            cv_scores['r2'].append(r2_score(y_test, y_pred))
            cv_scores['mse'].append(mean_squared_error(y_test, y_pred))
            cv_scores['mae'].append(mean_absolute_error(y_test, y_pred))
            
            # Rank correlation
            if len(y_test) > 2:
                corr, _ = spearmanr(y_test, y_pred)
                cv_scores['rank_corr'].append(corr if not np.isnan(corr) else 0)
        
        # Final model: Train on all but last year, test on last year
        unique_years = np.sort(np.unique(years))
        train_years = unique_years[:-1]
        test_year = unique_years[-1]
        
        train_mask = np.isin(years, train_years)
        test_mask = years == test_year
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        provinces_test = provinces[test_mask]
        
        self.model = self._create_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        # Feature importance
        importance = pd.Series(
            self.model.feature_importances_,
            index=feature_cols,
            name='importance'
        ).sort_values(ascending=False)
        
        # Test metrics
        test_metrics = {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        # Rankings
        pred_series = pd.Series(y_pred, index=provinces_test, name='Predicted')
        actual_series = pd.Series(y_test, index=provinces_test, name='Actual')
        
        pred_ranks = pred_series.rank(ascending=False).astype(int)
        actual_ranks = actual_series.rank(ascending=False).astype(int)
        
        rank_corr, _ = spearmanr(actual_ranks, pred_ranks)
        
        return RandomForestTSResult(
            model=self.model,
            feature_importance=importance,
            cv_scores=cv_scores,
            test_predictions=pred_series,
            test_actual=actual_series,
            test_metrics=test_metrics,
            predicted_ranks=pred_ranks,
            actual_ranks=actual_ranks,
            rank_correlation=rank_corr,
            feature_names=feature_cols
        )
    
    def _create_model(self) -> RandomForestRegressor:
        """Create a new Random Forest model."""
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict using fitted model.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Features to predict on.
            
        Returns
        -------
        np.ndarray
            Predicted values.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_predict first.")
        
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        
        return self.model.predict(X)


def calculate_shap_importance(
    model: RandomForestRegressor,
    X: np.ndarray,
    feature_names: List[str],
    max_samples: int = 100
) -> pd.DataFrame:
    """
    Calculate SHAP values for feature importance.
    
    Falls back to model's built-in feature importance if SHAP not available.
    
    Parameters
    ----------
    model : RandomForestRegressor
        Fitted Random Forest model.
    X : np.ndarray
        Feature matrix.
    feature_names : List[str]
        Names of features.
    max_samples : int, default=100
        Maximum samples for SHAP calculation.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature names and importance scores.
    """
    try:
        import shap
        
        # Sample if too large
        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Mean absolute SHAP values
        importance = np.abs(shap_values).mean(axis=0)
        
        return pd.DataFrame({
            'feature': feature_names,
            'shap_importance': importance
        }).sort_values('shap_importance', ascending=False)
    
    except ImportError:
        # Fallback to model's feature importance
        return pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
