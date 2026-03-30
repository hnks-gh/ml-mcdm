"""
Offline Recursive Feature Elimination (RFE) for ML Forecasting.

This utility script performs an automated, permutation-importance based 
feature selection process to reduce the high-dimensional panel feature 
space to a production-hardened core subset. It evaluates subsets using 
Hold-One-Province-Out (HOPO) cross-validation to prevent spatial leakage.
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import List
from sklearn.inspection import permutation_importance
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut

from data.data_loader import load_and_preprocess_data
from config import get_config
from forecasting.unified import UnifiedForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("offline_rfe")

def run_rfe() -> None:
    """
    Execute the recursive feature elimination pipeline.

    Steps:
    1. Load and preprocess the provincial panel data.
    2. Engineer the primary feature matrix (EWMA, lags, cross-entity).
    3. Iteratively drop features with the lowest permutation importance.
    4. Save the optimized feature subset to 'selected_features.json'.
    """
    config = get_config()
    target_count = config.forecast.target_max_features
    
    # 1. Load data
    panel_data = load_and_preprocess_data(config)
    target_year = config.forecast.target_year or max(panel_data.years) + 1
    
    # 2. Setup Forecaster
    forecaster = UnifiedForecaster(
        config=config,
        target_level="subcriteria",
        use_saw_targets=False
    )
    
    logger.info("Running Stage 1-2 to prepare feature matrices...")
    forecaster.stage1_engineer_features(panel_data, target_year)
    forecaster.stage2_reduce_features()
    
    X_train = forecaster.X_train_tree_
    y_train = forecaster.y_train_.values
    feature_names = forecaster.reducer_tree_._selected_feature_names_
    groups = forecaster._entity_indices_

    if len(feature_names) <= target_count:
        logger.info(f"Already at {len(feature_names)} features, <= {target_count}. Quitting.")
        return

    logger.info(f"Starting RFE. Current features: {X_train.shape[1]}, Target: {target_count}")
    
    current_features = list(range(X_train.shape[1]))
    
    def evaluate_hopo(X_sub, y_sub, groups_sub):
        logo = LeaveOneGroupOut()
        rmses = []
        for train_idx, test_idx in logo.split(X_sub, y_sub, groups_sub):
            X_tr, X_te = X_sub[train_idx], X_sub[test_idx]
            y_tr, y_te = y_sub[train_idx], y_sub[test_idx]
            
            model = CatBoostRegressor(
                iterations=120,
                depth=5,
                learning_rate=0.05,
                verbose=0,
                allow_writing_files=False,
                random_seed=42,
            )
            y_mean_tr = y_tr.mean(axis=1)
            y_mean_te = y_te.mean(axis=1)
            model.fit(X_tr, y_mean_tr)
            preds = model.predict(X_te)
            rmses.append(np.sqrt(mean_squared_error(y_mean_te, preds)))
        return np.mean(rmses)

    initial_rmse = evaluate_hopo(X_train, y_train, groups)
    logger.info(f"Initial Hold-One-Province-Out RMSE (all features): {initial_rmse:.4f}")

    step = 0
    while len(current_features) > target_count:
        step += 1
        X_sub = X_train[:, current_features]
        
        y_mean = y_train.mean(axis=1)
        model = CatBoostRegressor(
            iterations=120,
            depth=5,
            learning_rate=0.05,
            verbose=0,
            allow_writing_files=False,
            random_seed=42,
        )
        model.fit(X_sub, y_mean)
        
        m = permutation_importance(model, X_sub, y_mean, n_repeats=3, random_state=42, n_jobs=-1)
        importance = m.importances_mean
        
        # Drop lowest 5% or minimum 1
        n_drop = max(1, int(len(current_features) * 0.05))
        if len(current_features) - n_drop < target_count:
            n_drop = len(current_features) - target_count
            
        drop_indices = np.argsort(importance)[:n_drop]
        drop_indices_sorted = sorted(drop_indices, reverse=True)
        for idx in drop_indices_sorted:
            current_features.pop(idx)
            
        logger.info(f"Step {step}: Dropped {n_drop} features. Remaining: {len(current_features)}")

    # Final eval
    final_rmse = evaluate_hopo(X_train[:, current_features], y_train, groups)
    logger.info(f"Final Hold-One-Province-Out RMSE (subset features): {final_rmse:.4f}")
    
    selected_feature_names = [feature_names[i] for i in current_features]
    
    logger.info(f"RFE complete. Saving {len(selected_feature_names)} features to 'selected_features.json'")
    with open('selected_features.json', 'w') as f:
        json.dump(selected_feature_names, f, indent=4)

if __name__ == "__main__":
    run_rfe()
