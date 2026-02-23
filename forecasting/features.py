# -*- coding: utf-8 -*-
"""
Temporal Feature Engineering
============================

Advanced feature engineering for time series panel data, creating
rich feature sets for ML forecasting models.

Features include:
- Lag features (historical values)
- Rolling statistics (mean, std, min, max)
- Momentum and acceleration
- Trend features (linear slope)
- Cross-entity features (relative position)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class TemporalFeatureEngineer:
    """
    Advanced feature engineering for time series panel data.
    
    Creates rich feature set including:
    - Lag features (t-1, t-2, ...)
    - Rolling statistics (mean, std, min, max, trend)
    - Seasonal features
    - Cross-entity features (relative position)
    - Momentum and acceleration features
    
    Parameters:
        lag_periods: List of lag periods to include [1, 2]
        rolling_windows: Window sizes for rolling statistics [2, 3]
        include_momentum: Whether to include rate of change features
        include_cross_entity: Whether to include relative position features
    
    Example:
        >>> engineer = TemporalFeatureEngineer(lag_periods=[1, 2])
        >>> X_train, y_train, X_pred, _ = engineer.fit_transform(panel_data, 2025)
    """
    
    def __init__(self,
                 lag_periods: List[int] = [1, 2],
                 rolling_windows: List[int] = [2, 3],
                 include_momentum: bool = True,
                 include_cross_entity: bool = True):
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows
        self.include_momentum = include_momentum
        self.include_cross_entity = include_cross_entity
        self.feature_names_: List[str] = []
    
    def fit_transform(self,
                      panel_data,
                      target_year: int
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create feature matrix for training and prediction.
        
        Args:
            panel_data: Panel data object with provinces, years, components
            target_year: Year to predict
        
        Returns:
            X_train: Features for training
            y_train: Targets for training
            X_pred: Features for prediction (next year)
            entity_index: Entity identifiers
        """
        entities = panel_data.provinces
        components = panel_data.subcriteria_names
        years = sorted(panel_data.years)
        
        # Use target_year to determine the prediction boundary.
        # All years before target_year are available for training pairs;
        # the last available year's features are used to predict target_year.
        if target_year in years:
            # target_year already in data → use it as the last training target
            last_year = target_year
        else:
            # target_year is the next unseen year → predict beyond data
            last_year = years[-1]
        
        # Training pairs: features(year_t) → target(year_{t+1})
        # for every consecutive pair up to and including last_year
        train_feature_years = [y for y in years if y < last_year]
        
        X_train_list = []
        y_train_list = []
        X_pred_list = []
        entity_indices = []
        
        for ent_idx, entity in enumerate(entities):
            entity_data = panel_data.get_province(entity)
            
            # Create features for each training year (predicting next year)
            for i, year in enumerate(train_feature_years):
                next_year = years[years.index(year) + 1]
                features = self._create_features(
                    entity_data, entity, years, year, entities, panel_data
                )
                target = entity_data.loc[next_year, components].values
                
                X_train_list.append(features)
                y_train_list.append(target)
                entity_indices.append(ent_idx)
            
            # Create features for prediction (using last_year to predict target_year)
            pred_features = self._create_features(
                entity_data, entity, years, last_year, entities, panel_data
            )
            X_pred_list.append(pred_features)
        
        # Stack into arrays
        X_train = np.vstack(X_train_list)
        y_train = np.vstack(y_train_list)
        X_pred = np.vstack(X_pred_list)
        entity_indices_arr = np.array(entity_indices, dtype=int)
        
        # Create DataFrames
        X_train_df = pd.DataFrame(X_train, columns=self.feature_names_)
        y_train_df = pd.DataFrame(y_train, columns=components)
        X_pred_df = pd.DataFrame(X_pred, columns=self.feature_names_, index=entities)
        entity_index_df = pd.DataFrame(
            {'entity_index': entity_indices_arr},
        )
        
        return X_train_df, y_train_df, X_pred_df, entity_index_df
    
    def _create_features(self,
                         entity_data: pd.DataFrame,
                         entity: str,
                         years: List[int],
                         current_year: int,
                         all_entities: List[str],
                         panel_data) -> np.ndarray:
        """Create feature vector for a single entity-year combination."""
        components = list(entity_data.columns)
        features = []
        feature_names = []
        
        # Current values
        current_values = entity_data.loc[current_year, components].values
        features.extend(current_values)
        feature_names.extend([f"{c}_current" for c in components])
        
        # Lag features
        year_idx = years.index(current_year)
        for lag in self.lag_periods:
            if year_idx - lag >= 0:
                lag_year = years[year_idx - lag]
                lag_values = entity_data.loc[lag_year, components].values
            else:
                lag_values = current_values  # Use current if not enough history
            features.extend(lag_values)
            feature_names.extend([f"{c}_lag{lag}" for c in components])
        
        # Rolling statistics
        available_years = [y for y in years if y <= current_year]
        for window in self.rolling_windows:
            if len(available_years) >= window:
                window_years = available_years[-window:]
                window_data = entity_data.loc[window_years, components]
                
                # Mean
                features.extend(window_data.mean().values)
                feature_names.extend([f"{c}_roll{window}_mean" for c in components])
                
                # Std
                std_vals = window_data.std().fillna(0).values
                features.extend(std_vals)
                feature_names.extend([f"{c}_roll{window}_std" for c in components])
                
                # Min/Max
                features.extend(window_data.min().values)
                feature_names.extend([f"{c}_roll{window}_min" for c in components])
                features.extend(window_data.max().values)
                feature_names.extend([f"{c}_roll{window}_max" for c in components])
            else:
                # Pad with current values
                features.extend(current_values)
                feature_names.extend([f"{c}_roll{window}_mean" for c in components])
                features.extend(np.zeros(len(components)))
                feature_names.extend([f"{c}_roll{window}_std" for c in components])
                features.extend(current_values)
                feature_names.extend([f"{c}_roll{window}_min" for c in components])
                features.extend(current_values)
                feature_names.extend([f"{c}_roll{window}_max" for c in components])
        
        # Momentum features (rate of change)
        if self.include_momentum and year_idx > 0:
            prev_year = years[year_idx - 1]
            prev_values = entity_data.loc[prev_year, components].values
            momentum = current_values - prev_values
            features.extend(momentum)
            feature_names.extend([f"{c}_momentum" for c in components])
            
            # Acceleration (change in momentum)
            if year_idx > 1:
                prev_prev_year = years[year_idx - 2]
                prev_prev_values = entity_data.loc[prev_prev_year, components].values
                prev_momentum = prev_values - prev_prev_values
                acceleration = momentum - prev_momentum
                features.extend(acceleration)
                feature_names.extend([f"{c}_acceleration" for c in components])
            else:
                features.extend(np.zeros(len(components)))
                feature_names.extend([f"{c}_acceleration" for c in components])
        else:
            features.extend(np.zeros(len(components)))
            feature_names.extend([f"{c}_momentum" for c in components])
            features.extend(np.zeros(len(components)))
            feature_names.extend([f"{c}_acceleration" for c in components])
        
        # Trend feature (slope of linear fit)
        if len(available_years) >= 2:
            for c in components:
                y_vals = entity_data.loc[available_years, c].values
                x_vals = np.arange(len(y_vals))
                if len(y_vals) > 1:
                    slope = np.polyfit(x_vals, y_vals, 1)[0]
                else:
                    slope = 0
                features.append(slope)
                feature_names.append(f"{c}_trend")
        else:
            features.extend(np.zeros(len(components)))
            feature_names.extend([f"{c}_trend" for c in components])
        
        # Cross-entity features (relative position)
        if self.include_cross_entity:
            year_cross_section = panel_data.cross_section[current_year]
            if 'Province' in year_cross_section.columns:
                year_cross_section = year_cross_section.set_index('Province')
            
            for c in components:
                if c in year_cross_section.columns:
                    col_values = year_cross_section[c]
                    entity_value = current_values[components.index(c)]
                    
                    # Percentile rank
                    percentile = (col_values < entity_value).mean()
                    features.append(percentile)
                    feature_names.append(f"{c}_percentile")
                    
                    # Z-score
                    mean_val = col_values.mean()
                    std_val = col_values.std()
                    z_score = (entity_value - mean_val) / (std_val + 1e-10)
                    features.append(z_score)
                    feature_names.append(f"{c}_zscore")
                else:
                    features.extend([0.5, 0])
                    feature_names.extend([f"{c}_percentile", f"{c}_zscore"])
        
        # Validate that every entity-year produces the same feature vector
        if self.feature_names_ and len(self.feature_names_) != len(feature_names):
            raise ValueError(
                f"Feature dimension mismatch: previous call produced "
                f"{len(self.feature_names_)} features, this call produced "
                f"{len(feature_names)}. Check cross_section data for "
                f"entity={entity}, year={current_year}."
            )
        self.feature_names_ = feature_names
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self.feature_names_
