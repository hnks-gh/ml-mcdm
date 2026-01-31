# -*- coding: utf-8 -*-
"""
Random Forest with Time-Series Cross-Validation
=================================================

Proper ML validation for panel data using temporal splits.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr
import warnings


@dataclass
class RandomForestTSResult:
    """Result container for Random Forest with time-series CV."""
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
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "RANDOM FOREST TIME-SERIES CV RESULTS",
            f"{'='*60}",
            f"\nTest Set Metrics:",
            f"  R²: {self.test_metrics['r2']:.4f}",
            f"  MSE: {self.test_metrics['mse']:.4f}",
            f"  MAE: {self.test_metrics['mae']:.4f}",
            f"  Rank Correlation: {self.rank_correlation:.4f}",
            f"\nCV Scores (mean ± std):",
            f"  R²: {np.mean(self.cv_scores['r2']):.4f} ± {np.std(self.cv_scores['r2']):.4f}",
            f"\nTop 10 Important Features:",
        ]
        
        top_features = self.feature_importance.nlargest(10)
        for feat, imp in top_features.items():
            lines.append(f"  {feat}: {imp:.4f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class TimeSeriesSplit:
    """Time-series cross-validation split for panel data."""
    
    def __init__(self, n_splits: int = 3, gap: int = 0):
        self.n_splits = n_splits
        self.gap = gap
    
    def split(self, years: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for time-series CV."""
        unique_years = np.sort(np.unique(years))
        n_years = len(unique_years)
        
        # Automatically adjust n_splits if not enough years
        actual_splits = min(self.n_splits, max(1, n_years - 1))
        
        if n_years < 2:
            # Return a simple split using all data for both train and test
            all_idx = np.arange(len(years))
            return [(all_idx, all_idx)]
        
        splits = []
        for i in range(actual_splits):
            # Train on years up to split point
            train_end = n_years - actual_splits + i
            test_start = train_end + self.gap
            
            train_years = unique_years[:train_end + 1]
            test_years = unique_years[test_start:test_start + 1] if test_start < n_years else []
            
            if len(test_years) == 0:
                continue
            
            train_idx = np.where(np.isin(years, train_years))[0]
            test_idx = np.where(np.isin(years, test_years))[0]
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        
        # Fallback: if no valid splits, create one using all-but-last year for train
        if not splits and n_years >= 2:
            train_years = unique_years[:-1]
            test_years = unique_years[-1:]
            train_idx = np.where(np.isin(years, train_years))[0]
            test_idx = np.where(np.isin(years, test_years))[0]
            splits.append((train_idx, test_idx))
        
        return splits


class RandomForestTS:
    """
    Random Forest with Time-Series Cross-Validation.
    
    Properly handles temporal structure of panel data.
    """
    
    def __init__(self,
                 n_estimators: int = 200,
                 max_depth: Optional[int] = 10,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 max_features: str = "sqrt",
                 n_splits: int = 3,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_splits = n_splits
        self.random_state = random_state
        
        self.model = None
        self.feature_names = None
    
    def fit_predict(self,
                   panel_data,
                   target_col: str,
                   feature_cols: Optional[List[str]] = None,
                   province_col: str = 'Province',
                   year_col: str = 'Year') -> RandomForestTSResult:
        """
        Fit Random Forest with time-series CV and predict on test year.
        
        Parameters
        ----------
        panel_data : PanelData or DataFrame
            Panel data
        target_col : str
            Target variable (e.g., TOPSIS score or composite)
        feature_cols : List[str]
            Feature columns (if None, uses all component columns)
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
        """Predict using fitted model."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_predict first.")
        
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        
        return self.model.predict(X)


def calculate_shap_importance(model: RandomForestRegressor,
                             X: np.ndarray,
                             feature_names: List[str],
                             max_samples: int = 100) -> pd.DataFrame:
    """
    Calculate SHAP values for feature importance.
    
    Falls back to permutation importance if SHAP not available.
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
