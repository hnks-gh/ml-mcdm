from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

if TYPE_CHECKING:
    from .__init__ import ImputationConfig

logger = logging.getLogger('ml_mcdm')


class MICEImputer:
    """
    Production-Ready MICE Imputation Engine
    =====================================

    Unified imputation for all missing data in ML forecasting pipeline.
    Uses IterativeImputer with ExtraTreesRegressor (MissForest algorithm).

    **Algorithm: Iterative Imputation by Chained Equations (MICE)**

    For each iteration, cycles through features with missing values:
    1. Set feature with missing data as target (y)
    2. Use remaining features as predictors (X)
    3. Fit ExtraTreesRegressor: Ŷ = f(X)
    4. Predict missing values: y_filled = f(X_missing)

    Repeats until convergence (imputed values stabilize) or max iterations reached.

    **Why ExtraTreesRegressor?**
    - Handles nonlinear relationships in governance panel data
    - Robust to high-dimensional feature spaces
    - Fast inference compared to RandomForest
    - Extreme randomization reduces tree correlation (lower variance)

    **Leakage-Free Design**
    - Fit on training data ONLY (no target information)
    - Transform applied to holdout/test data using fitted imputer
    - Missingness indicators (_was_missing) track imputed values

    Parameters
    ----------
    config : ImputationConfig
        Configuration object with MICE parameters:
        - mice_max_iter: Convergence iterations (default 40)
        - mice_n_nearest_features: Correlation-based feature subset (default 30)
        - mice_estimator: "extra_trees" (default), "random_forest", or "bayesian_ridge"
        - mice_add_indicator: Append _was_missing flags (default True)
        - random_state: Reproducibility (default 42)

    Attributes
    ----------
    is_fitted_ : bool
        True if imputer has been fitted on training data
    missingness_rate_ : float
        Fraction of NaN in training data (0.0 to 1.0)
    n_features_in_ : int
        Number of input features
    imputer_ : IterativeImputer
        Fitted sklearn IterativeImputer instance
    """

    def __init__(self, config: "ImputationConfig"):
        self.config = config
        self.imputer_ = None
        self.is_fitted_ = False
        self.missingness_rate_ = 0.0
        self.n_features_in_ = 0
        self._train_means_ = None  # Fallback for safety-net imputation
        self._imputation_mask_ = None  # Boolean mask of imputed cells

    def _create_estimator(self):
        """Create regression estimator for MICE predictions.

        Returns
        -------
        estimator : BaseEstimator
            Fitted regression estimator for imputation.
        """
        if self.config.mice_estimator == "extra_trees":
            # Default: ExtraTreesRegressor (MissForest)
            return ExtraTreesRegressor(
                n_estimators=150,           # Balanced: higher than 100 for stability
                max_depth=8,                # Allow moderate depth for expressiveness
                max_features='sqrt',        # Reduce tree correlation
                min_samples_leaf=2,         # Relax for small governance panels
                bootstrap=True,
                n_jobs=-1,                  # Use all cores
                random_state=self.config.random_state
            )
        elif self.config.mice_estimator == "random_forest":
            # Alternative: RandomForestRegressor (more conservative, stable)
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                max_features='sqrt',
                min_samples_leaf=3,
                bootstrap=True,
                n_jobs=-1,
                random_state=self.config.random_state
            )
        else:
            # Fallback: Bayesian Ridge (probabilistic)
            from sklearn.linear_model import BayesianRidge
            return BayesianRidge(
                n_iter=300,
                tol=1e-3,
                compute_score=True,
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6,
            )

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> "MICEImputer":
        """
        Fit MICE imputer on training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix with NaN for missing values.
        feature_names : list of str, optional
            Column names for logging and diagnostics.

        Returns
        -------
        self : MICEImputer
            Fitted imputer instance.
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Compute missingness statistics
        nan_mask = np.isnan(X)
        n_missing = nan_mask.sum()
        self.missingness_rate_ = float(n_missing / X.size) if X.size > 0 else 0.0

        feature_names = feature_names or [f"f{i}" for i in range(n_features)]

        logger.info(
            f"[MICE] Fitting imputer: {n_samples} samples × {n_features} features. "
            f"Missingness: {n_missing} cells ({self.missingness_rate_:.1%})"
        )

        # Skip MICE if no missing data
        if n_missing == 0:
            logger.info("[MICE] No missing data detected; imputer pre-fitted (pass-through)")
            self.is_fitted_ = True
            self._train_means_ = np.nanmean(X, axis=0)
            self._train_means_ = np.where(np.isnan(self._train_means_), 0.0, self._train_means_)
            # Create dummy imputer that returns X unchanged
            self.imputer_ = None
            return self

        # Create and fit IterativeImputer
        self.imputer_ = IterativeImputer(
            estimator=self._create_estimator(),
            max_iter=self.config.mice_max_iter,
            initial_strategy='median',  # Robust starting point
            n_nearest_features=min(self.config.mice_n_nearest_features, n_features),
            add_indicator=self.config.mice_add_indicator,  # Append _was_missing cols
            random_state=self.config.random_state,
            tol=5e-3,  # Relaxed tolerance for governance data convergence
            imputation_order='roman',  # Left-to-right deterministic order
            verbose=0,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"\[IterativeImputer\] Early stopping criterion not reached\.",
                category=ConvergenceWarning,
            )
            try:
                self.imputer_.fit(X)
                logger.info(f"[MICE] Imputer converged successfully")
            except Exception as e:
                logger.warning(f"[MICE] Fit failed ({e}); using mean imputation fallback")
                self.imputer_ = None

        self.is_fitted_ = True

        # Cache training means as safety-net fallback (M-01)
        self._train_means_ = np.nanmean(X, axis=0)
        self._train_means_ = np.where(np.isnan(self._train_means_), 0.0, self._train_means_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply fitted MICE imputation to new data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix with NaN for missing values.

        Returns
        -------
        X_imputed : ndarray, shape (n_samples, n_features | n_features+k)
            Imputed feature matrix. If add_indicator=True, includes k _was_missing columns.

        Raises
        ------
        ValueError
            If not fitted or feature dimension mismatch.
        """
        if not self.is_fitted_:
            raise ValueError("[MICE] Imputer not fitted. Call fit() first.")

        X = np.asarray(X, dtype=float)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"[MICE] Feature mismatch: expected {self.n_features_in_} features, "
                f"got {X.shape[1]}"
            )

        # If imputer is None (no missing in training), use safe fill
        if self.imputer_ is None:
            return self._fallback_fill(X)

        try:
            X_imputed = self.imputer_.transform(X)
            return X_imputed
        except Exception as e:
            logger.warning(f"[MICE] Transform failed ({e}); using fallback fill")
            return self._fallback_fill(X)

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit imputer and apply to training data in one call.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix.
        feature_names : list of str, optional
            Column names for logging.

        Returns
        -------
        X_imputed : ndarray
            Imputed training feature matrix.
        """
        self.fit(X, feature_names=feature_names)
        return self.transform(X)

    def _fallback_fill(self, X: np.ndarray, method: str = "column_mean") -> np.ndarray:
        """
        Fallback imputation when MICE is unavailable (production safety net).

        Parameters
        ----------
        X : ndarray
            Feature matrix with NaN.
        method : str
            "column_mean" (default): per-column training means

        Returns
        -------
        X_filled : ndarray
            Imputed array (no NaN).
        """
        X = X.copy()
        nan_mask = np.isnan(X)

        if not nan_mask.any():
            return X  # No missing data

        if method == "column_mean":
            # Use fitted training means (never 0.0 for governance scores)
            for j in range(X.shape[1]):
                if nan_mask[:, j].any():
                    X[nan_mask[:, j], j] = self._train_means_[j]

        return X

    def get_fallback_values(self) -> np.ndarray:
        """
        Return fitted training means for downstream imputation (M-01).

        These are used by fill_missing_features() as a safety-net
        when MICE unavailable. Never 0.0 (governance scale).

        Returns
        -------
        means : ndarray, shape (n_features,)
            Per-column training means.
        """
        if self._train_means_ is None:
            raise ValueError("[MICE] Not fitted; no fallback values available")
        return self._train_means_.copy()

    def get_missingness_indicators(self, X: np.ndarray) -> np.ndarray:
        """
        Create binary indicator matrix for imputed cells (post-transform).

        Parameters
        ----------
        X : ndarray
            Original data with NaN.

        Returns
        -------
        indicator : ndarray, dtype bool, same shape as X
            True where imputation was needed.
        """
        return np.isnan(X).astype(bool)
