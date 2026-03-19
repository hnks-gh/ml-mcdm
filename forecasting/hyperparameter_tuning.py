# -*- coding: utf-8 -*-
"""
Hyperparameter Tuning Module
=============================

Provides the `EnsembleHyperparameterOptimizer` class to run Optuna TPE search
for base models: CatBoost, LightGBM, KernelRidge, SVR, and QuantileRF.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import HyperbandPruner, MedianPruner
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


class EnsembleHyperparameterOptimizer:
    """
    Optuna-based Hyperparameter Optimizer for ML-MCDM base models.
    
    Optimises hyperparameters using Tree-structured Parzen Estimator (TPE)
    and Hyperband / Median pruners. Designed to integrate with `UnifiedForecaster`.
    """

    def __init__(self, config, cv_splitter, random_state: int = 42):
        """
        Parameters
        ----------
        config : ForecastConfig
            Configuration containing hp_tune_n_trials, hp_tune_timeout_seconds.
        cv_splitter : object
            Cross-validation splitter (e.g., PanelWalkForwardCV) for evaluation.
        random_state : int
            Random seed for reproducibility.
        """
        self.config = config
        self.cv_splitter = cv_splitter
        self.random_state = random_state
        self.best_params: Dict[str, Dict[str, Any]] = {}

        if _OPTUNA_AVAILABLE:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

    def optimize_catboost(self, X: np.ndarray, y: np.ndarray, year_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Tuning for CatBoostForecaster."""
        if not _OPTUNA_AVAILABLE:
            logger.warning("Optuna not installed. Skipping tuning for CatBoost.")
            return {}

        def objective(trial: optuna.Trial) -> float:
            from forecasting.gradient_boosting import CatBoostForecaster
            # SOTA expanded search space
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'depth': trial.suggest_int('depth', 3, 7),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'iterations': 300,  # Let early stopping handle the exact number
                'early_stopping_rounds': 20
            }

            scores = []
            # Manual CV loop to support pruner
            for fold_idx, (train_idx, val_idx) in enumerate(self.cv_splitter.split(X, y)):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                
                model = CatBoostForecaster(**params)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                score = self._compute_r2(y_val, preds)
                scores.append(score)
                
                trial.report(score, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_warmup_steps=1)
        )
        study.optimize(
            objective, 
            n_trials=self.config.hp_tune_n_trials,
            timeout=self.config.hp_tune_timeout_seconds
        )
        logger.info(f"CatBoost tuned: best R2={study.best_value:.4f}, params={study.best_params}")
        return study.best_params

    def optimize_lightgbm(self, X: np.ndarray, y: np.ndarray, year_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Tuning for LightGBMForecaster."""
        if not _OPTUNA_AVAILABLE:
            return {}

        def objective(trial: optuna.Trial) -> float:
            from forecasting.gradient_boosting import LightGBMForecaster
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'l2_reg': trial.suggest_float('l2_reg', 0.1, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'n_estimators': 300,
                'early_stopping_rounds': 20
            }

            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(self.cv_splitter.split(X, y)):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                
                model = LightGBMForecaster(**params)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                score = self._compute_r2(y_val, preds)
                scores.append(score)
                
                trial.report(score, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_warmup_steps=1)
        )
        study.optimize(
            objective, 
            n_trials=self.config.hp_tune_n_trials,
            timeout=self.config.hp_tune_timeout_seconds
        )
        logger.info(f"LightGBM tuned: best R2={study.best_value:.4f}, params={study.best_params}")
        return study.best_params

    def optimize_kernel_ridge(self, X: np.ndarray, y: np.ndarray, year_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Tuning for KernelRidgeForecaster."""
        if not _OPTUNA_AVAILABLE:
            return {}

        def objective(trial: optuna.Trial) -> float:
            from forecasting.kernel_ridge import KernelRidgeForecaster
            params = {
                'alpha': trial.suggest_float('alpha', 1e-3, 1e2, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'laplacian', 'polynomial'])
            }

            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(self.cv_splitter.split(X, y)):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                
                model = KernelRidgeForecaster(**params)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                score = self._compute_r2(y_val, preds)
                scores.append(score)
                
                trial.report(score, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_warmup_steps=1)
        )
        study.optimize(
            objective, 
            n_trials=self.config.hp_tune_n_trials,
            timeout=self.config.hp_tune_timeout_seconds
        )
        logger.info(f"KernelRidge tuned: best R2={study.best_value:.4f}, params={study.best_params}")
        return study.best_params

    def optimize_svr(self, X: np.ndarray, y: np.ndarray, year_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Tuning for SVRForecaster."""
        if not _OPTUNA_AVAILABLE:
            return {}

        def objective(trial: optuna.Trial) -> float:
            from forecasting.svr import SVRForecaster
            params = {
                'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
                'epsilon': trial.suggest_float('epsilon', 1e-3, 1.0, log=True),
            }

            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(self.cv_splitter.split(X, y)):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                
                model = SVRForecaster(**params)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                score = self._compute_r2(y_val, preds)
                scores.append(score)
                
                trial.report(score, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_warmup_steps=1)
        )
        study.optimize(
            objective, 
            n_trials=self.config.hp_tune_n_trials,
            timeout=self.config.hp_tune_timeout_seconds
        )
        logger.info(f"SVR tuned: best R2={study.best_value:.4f}, params={study.best_params}")
        return study.best_params

    def optimize_quantilerf(self, X: np.ndarray, y: np.ndarray, year_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Tuning for QuantileRandomForestForecaster."""
        if not _OPTUNA_AVAILABLE:
            return {}

        def objective(trial: optuna.Trial) -> float:
            from forecasting.quantile_forest import QuantileRandomForestForecaster
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }

            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(self.cv_splitter.split(X, y)):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                
                model = QuantileRandomForestForecaster(**params)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                score = self._compute_r2(y_val, preds)
                scores.append(score)
                
                trial.report(score, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_warmup_steps=1)
        )
        study.optimize(
            objective, 
            n_trials=self.config.hp_tune_n_trials,
            timeout=self.config.hp_tune_timeout_seconds // 2  # QRF is slower, maybe allocate diff timeout
        )
        logger.info(f"QuantileRF tuned: best R2={study.best_value:.4f}, params={study.best_params}")
        return study.best_params

    def _compute_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import r2_score
        # Flatten and remove NaNs (if any)
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if not np.any(mask):
            return -1.0
        # Multi-output macro average
        return r2_score(y_true[mask], y_pred[mask])

    def save_best_params(self, param_dict: Dict[str, Dict[str, Any]], output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(param_dict, f, indent=4)
        logger.info(f"Saved tuned hyperparameters to {path}")

    def load_best_params(self, input_path: str) -> Dict[str, Dict[str, Any]]:
        path = Path(input_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                params = json.load(f)
            logger.info(f"Loaded tuned hyperparameters from {path}")
            return params
        return {}
