from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

if TYPE_CHECKING:
    from .__init__ import ImputationConfig

class MICEImputer:
    """
    Tier 3 Enhancement M-02, M-08: Activated MissForest / MICE.
    
    Uses IterativeImputer with tuned ExtraTreesRegressor estimator.
    """
    def __init__(self, config: "ImputationConfig"):
        self.config = config
        self.imputer = None
        self._is_fitted = False
        self._train_means = None
        
    def _create_estimator(self):
        if self.config.mice_estimator == "extra_trees":
            return ExtraTreesRegressor(
                n_estimators=100,
                max_features='sqrt',
                min_samples_leaf=3,
                bootstrap=True,
                random_state=self.config.random_state
            )
        elif self.config.mice_estimator == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=3,
                random_state=self.config.random_state
            )
        else:
            # Default to bayesian ridge (standard MICE)
            from sklearn.linear_model import BayesianRidge
            return BayesianRidge()
            
    def fit(self, X: np.ndarray) -> "MICEImputer":
        """Fit the imputer on training data."""
        self.imputer = IterativeImputer(
            estimator=self._create_estimator(),
            max_iter=self.config.mice_max_iter,
            initial_strategy='median',
            n_nearest_features=min(self.config.mice_n_nearest_features, X.shape[1]),
            add_indicator=self.config.add_missingness_indicators,
            random_state=self.config.random_state,
            tol=5e-3,                     # relaxed from 1e-3 for achievability
            imputation_order='roman'
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"\[IterativeImputer\] Early stopping criterion not reached\.",
                category=ConvergenceWarning,
            )
            self.imputer.fit(X)
        self._is_fitted = True
        
        # Cache training means as fallback for fill_missing_features extension (M-01)
        self._train_means = np.nanmean(X, axis=0)
        # Avoid NaN in fallback if whole columns are NaN
        self._train_means = np.where(np.isnan(self._train_means), 0.0, self._train_means)
        
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply imputation to new data."""
        if not self._is_fitted:
            raise ValueError("MICEImputer must be fitted before transform")
        return self.imputer.transform(X)
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(X)
        return self.transform(X)
        
    def get_fallback_values(self) -> np.ndarray:
        """Return the fitted training means for Tier 1 safety-net fill (M-01)."""
        return self._train_means
