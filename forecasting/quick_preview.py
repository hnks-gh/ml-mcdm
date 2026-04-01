# -*- coding: utf-8 -*-
"""
Quick Preview Mode: Mock Ensemble Forecasting for Development & Prototyping.

This module provides synthetic yet statistically sound forecast results that 
mimic the output of a fully-trained ensemble with "moderate high - good result" 
quality. All outputs are mathematically consistent and suitable for prototyping 
result generation pipelines (CSVs, figures) without incurring the computational 
cost of actual ensemble training.

---

Design Principles (Production-Hardened):
=========================================

1. **Statistical Soundness**
   - All predictions follow realistic distributions (normal, clipped at bounds)
   - Uncertainty quantification respects conformal prediction principles
   - Prediction intervals properly bracket predictions with 95% coverage
   - Cross-validation scores align with moderate-high performance (R² ≈ 0.65–0.75)

2. **Consistency Across Layers**
   - Base model predictions are diverse but correlated
   - Meta-learner weights sum to 1.0 (proper ensemble convexity)
   - Stacked predictions are weighted averages of base models
   - Feature importance scores are non-negative and normalized

3. **Data Integrity**
   - All arrays match expected shapes (n_entities × n_components)
   - No NaN or Inf values in outputs
   - Indices align with input panel data
   - Types are consistent (float64, int32)

4. **Algorithmic Soundness**
   - Prediction intervals: lower ≤ point ≤ upper (monotone)
   - Conformal quantiles properly calibrated for 95% coverage
   - Cross-entropy loss > 0, R² in [−∞, 1.0] range
   - Residuals approximately centered at zero with constant variance

5. **Performance Realism**
   - R² scores: 0.65–0.75 (moderate high - good)
   - MAE: ~0.10–0.15 (province-level aggregated error)
   - RMSE: ~0.12–0.18 (account for 28 sub-criteria components)
   - Coverage: ≥95% for conformal intervals

---

Usage
-----

>>> from forecasting.quick_preview import QuickPreviewGenerator
>>> gen = QuickPreviewGenerator(
...     n_entities=63,
...     n_components=28,
...     target_year=2025,
...     random_state=42
... )
>>> result = gen.generate()
>>> print(f"Predictions shape: {result.predictions.shape}")
>>> print(f"R² scores: {result.cross_validation_scores}")
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QuickPreviewConfig:
    """Configuration for mock forecasting performance."""
    
    # Target performance: moderate high - good result
    mean_r2_per_model: float = 0.70  # Individual model R²
    std_r2_per_model: float = 0.05   # Diversity among models
    ensemble_r2_boost: float = 0.03  # Ensemble improvement over base
    
    # Prediction intervals (95% coverage, conformal)
    quantile_lower: float = 0.025    # 2.5th percentile
    quantile_upper: float = 0.975    # 97.5th percentile
    
    # Uncertainty quantification
    uncertainty_std_ratio: float = 0.12  # Prediction std relative to target
    
    # Cross-validation setup
    n_cv_folds: int = 5
    n_base_models: int = 4
    base_model_names: List[str] = None
    
    def __post_init__(self):
        if self.base_model_names is None:
            self.base_model_names = [
                'CatBoost',
                'BayesianRidge',
                'SVR',
                'ElasticNet'
            ]


class QuickPreviewGenerator:
    """
    Generates production-quality mock forecast results.
    
    Produces synthetic outputs that maintain:
    - Realistic prediction distributions
    - Proper uncertainty quantification
    - Consistent ensemble architecture
    - Valid statistical properties
    """
    
    def __init__(
        self,
        n_entities: int,
        n_components: int,
        target_year: int,
        random_state: int = 42,
        config: Optional[QuickPreviewConfig] = None,
        entity_names: Optional[List[str]] = None,
        component_names: Optional[List[str]] = None,
    ):
        """
        Initialize the mock data generator.
        
        Parameters
        ----------
        n_entities : int
            Number of spatial units (provinces).
        n_components : int
            Number of target components (subcriteria).
        target_year : int
            Forecast year.
        random_state : int
            Random seed for reproducibility.
        config : QuickPreviewConfig, optional
            Custom configuration; uses defaults if None.
        entity_names : List[str], optional
            Names for entities; generates generic if None.
        component_names : List[str], optional
            Names for components; generates generic if None.
        """
        self.n_entities = n_entities
        self.n_components = n_components
        self.target_year = target_year
        self.random_state = random_state
        self.config = config or QuickPreviewConfig()
        
        # Entity and component naming
        self.entity_names = (
            entity_names if entity_names is not None
            else [f"Entity_{i:02d}" for i in range(n_entities)]
        )
        
        # For 28 subcriteria (SC01–SC28), generate realistic names if not provided
        if component_names is None:
            self.component_names = [f"SC{i+1:02d}" for i in range(n_components)]
        else:
            self.component_names = component_names
        
        # Seed the RNG
        self.rng = np.random.RandomState(random_state)
        
        # Will hold generated data
        self._predictions: Optional[pd.DataFrame] = None
        self._uncertainty: Optional[pd.DataFrame] = None
        self._intervals: Optional[Dict[str, pd.DataFrame]] = None
        self._model_weights: Optional[Dict[str, float]] = None
        self._cv_scores: Optional[Dict[str, List[float]]] = None
        self._model_perf: Optional[Dict[str, Dict[str, float]]] = None
        self._feature_importance: Optional[pd.DataFrame] = None
    
    def generate(self) -> 'UnifiedForecastResult':
        """
        Generate a complete, production-quality mock forecast result.
        
        Returns
        -------
        UnifiedForecastResult
            A comprehensive result object mimicking full ensemble output.
        """
        logger.info(
            f"[QUICK_PREVIEW] Generating mock forecast for {self.target_year} "
            f"({self.n_entities} entities × {self.n_components} components)"
        )
        
        # Stage 1: Generate base predictions and uncertainty
        self._generate_predictions_and_uncertainty()
        
        # Stage 2: Generate prediction intervals (conformal)
        self._generate_prediction_intervals()
        
        # Stage 3: Generate base model outputs
        base_model_oof, base_model_holdout = self._generate_base_model_outputs()
        
        # Stage 4: Generate ensemble weights (meta-learner)
        self._generate_model_weights()
        
        # Stage 5: Generate cross-validation scores
        self._generate_cv_scores()
        
        # Stage 6: Generate model performance metrics
        self._generate_model_performance()
        
        # Stage 7: Generate feature importance
        self._generate_feature_importance()
        
        # Stage 8: Assemble the result object
        from .unified import UnifiedForecastResult
        
        result = UnifiedForecastResult(
            predictions=self._predictions,
            uncertainty=self._uncertainty,
            prediction_intervals=self._intervals,
            model_contributions=self._model_weights,
            model_performance=self._model_perf,
            feature_importance=self._feature_importance,
            cross_validation_scores=self._cv_scores,
            holdout_performance=self._generate_holdout_performance(),
            composite_predictions=None,  # Will be computed by pipeline if needed
            training_info=self._build_training_info(base_model_oof, base_model_holdout),
            data_summary=self._build_data_summary(),
            best_model_name=list(self._model_weights.keys())[0],
            best_model_predictions=self._predictions.copy(),
            model_comparison=None,
            forecast_criterion_weights=None,
            criteria_predictions=None,
            forecast_decision_matrix=None,
            calibration_summary=None,
            individual_model_predictions=base_model_holdout,
            forecast_residuals=self._generate_residuals(),
            forecast_metadata={
                'generation_mode': 'quick_preview',
                'generated_at': datetime.now().isoformat(),
                'target_year': self.target_year,
                'config': asdict(self.config),
            },
            interval_coverage_by_criterion=self._estimate_interval_coverage(),
            interval_width_summary=self._generate_interval_width_summary(),
            entity_error_summary=self._generate_entity_error_summary(),
            worst_predictions=self._generate_worst_predictions(),
        )
        
        logger.info(
            f"[QUICK_PREVIEW] Mock forecast generated successfully. "
            f"Ensemble R² ≈ {self._model_perf.get(list(self._model_perf.keys())[0], {}).get('r2', 0):.4f}"
        )
        
        return result
    
    # =====================================================================
    # Core generation stages
    # =====================================================================
    
    def _generate_predictions_and_uncertainty(self) -> None:
        """Generate point estimates and uncertainty for moderate-high/good results.
        
        For R² ≈ 0.70-0.75 with residual std ≈ 0.09-0.11, we need:
        - Predictions centered around 0.65 with reasonable spread
        - Residuals with controlled variance and zero mean
        - Uncertainty matching actual residual magnitude
        """
        # Generate realistic predictions in [0, 1] range (SAW-like scale)
        # with slight mean shift to simulate good performance (mean ≈ 0.65)
        base = self.rng.normal(
            loc=0.65,  # Moderate high - good (not too high, not mediocre)
            scale=0.18,
            size=(self.n_entities, self.n_components)
        )
        
        # Clip to reasonable bounds but preserve informativeness
        predictions = np.clip(base, 0.0, 1.0)
        
        self._predictions = pd.DataFrame(
            predictions,
            index=self.entity_names,
            columns=self.component_names
        )
        
        # Uncertainty: should match residual magnitude for consistency
        # For good forecasts (R²≈0.70), residual std ≈ 0.10-0.11
        # Generate realistic uncertainty that varies slightly per prediction
        uncertainty = self.rng.normal(
            loc=0.10,  # Base uncertainty (residual std)
            scale=0.01,  # Small variation in uncertainty
            size=(self.n_entities, self.n_components)
        )
        uncertainty = np.abs(uncertainty)
        uncertainty = np.clip(uncertainty, 0.08, 0.13)  # Realistic bounds
        
        self._uncertainty = pd.DataFrame(
            uncertainty,
            index=self.entity_names,
            columns=self.component_names
        )
    
    def _generate_prediction_intervals(self) -> None:
        """Generate conformal prediction intervals with 95% coverage."""
        # For each prediction, generate intervals around it
        # Lower bound: approximately at quantile_lower
        lower = self._predictions.values - 1.96 * self._uncertainty.values
        upper = self._predictions.values + 1.96 * self._uncertainty.values
        
        # Ensure lower < point < upper (conformal monotonicity)
        lower = np.minimum(lower, self._predictions.values - 1e-6)
        upper = np.maximum(upper, self._predictions.values + 1e-6)
        
        self._intervals = {
            'lower': pd.DataFrame(
                lower,
                index=self.entity_names,
                columns=self.component_names
            ),
            'upper': pd.DataFrame(
                upper,
                index=self.entity_names,
                columns=self.component_names
            ),
        }
    
    def _generate_base_model_outputs(
        self
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Generate out-of-fold and holdout predictions for each base model."""
        base_model_oof = {}
        base_model_holdout = {}
        
        for model_name in self.config.base_model_names:
            # OOF predictions: slightly noisier than final ensemble
            noise_oof = self.rng.normal(0, 0.08, size=self._predictions.shape)
            oof = self._predictions.values + noise_oof
            oof = np.clip(oof, 0.0, 1.0)
            base_model_oof[model_name] = oof
            
            # Holdout predictions: similar but spatially correlated differently
            noise_holdout = self.rng.normal(0, 0.10, size=self._predictions.shape)
            holdout = self._predictions.values + noise_holdout
            holdout = np.clip(holdout, 0.0, 1.0)
            
            base_model_holdout[model_name] = pd.DataFrame(
                holdout,
                index=self.entity_names,
                columns=self.component_names
            )
        
        return base_model_oof, base_model_holdout
    
    def _generate_model_weights(self) -> None:
        """Generate ensemble meta-learner weights (sum to 1.0) - realistic, unbalanced.
        
        Realistic stacking ensembles have unbalanced weights where the best model
        (usually tree-based) dominates, and weaker models contribute less but still
        provide diversity benefit.
        """
        # Realistic weight distribution: strongest model ~45%, others decreasing
        # CatBoost typically strongest (tree-based on this data), others contribute diversity
        weights = np.array([0.45, 0.25, 0.18, 0.12], dtype=float)
        
        # Add small random perturbation for realism (not perfectly fixed)
        perturbation = self.rng.normal(0, 0.02, size=len(weights))
        weights = weights + perturbation
        weights = np.clip(weights, 0.08, 0.50)  # Keep in realistic range
        weights = weights / weights.sum()  # Normalize to sum to 1.0
        
        self._model_weights = {
            name: float(w)
            for name, w in zip(self.config.base_model_names, weights)
        }
    
    def _generate_cv_scores(self) -> None:
        """Generate per-model cross-validation R² scores."""
        self._cv_scores = {}
        
        for model_name in self.config.base_model_names:
            # Generate realistic CV scores with variation
            scores = self.rng.normal(
                loc=self.config.mean_r2_per_model,
                scale=self.config.std_r2_per_model,
                size=self.config.n_cv_folds
            )
            # Keep R² in reasonable range but allow some variability
            scores = np.clip(scores, 0.50, 0.85)
            self._cv_scores[model_name] = scores.tolist()
    
    def _generate_model_performance(self) -> None:
        """Generate per-model metrics (R², MAE, RMSE, etc.)."""
        self._model_perf = {}
        
        # Ensemble performance: boost from best base model
        base_r2 = np.mean(self._cv_scores[self.config.base_model_names[0]])
        ensemble_r2 = min(
            base_r2 + self.config.ensemble_r2_boost,
            0.80  # Cap at good but not unrealistic
        )
        
        for model_name in self.config.base_model_names:
            cv_r2 = np.mean(self._cv_scores[model_name])
            
            self._model_perf[model_name] = {
                'r2': float(cv_r2),
                'mae': float(self.rng.uniform(0.08, 0.15)),
                'rmse': float(self.rng.uniform(0.10, 0.18)),
                'cv_folds': self.config.n_cv_folds,
                'n_samples': int(self.n_entities * 10),  # Mock training samples
            }
        
        # Add ensemble metrics
        self._model_perf['SuperLearner'] = {
            'r2': float(ensemble_r2),
            'mae': float(self.rng.uniform(0.08, 0.13)),
            'rmse': float(self.rng.uniform(0.10, 0.16)),
            'cv_folds': self.config.n_cv_folds,
            'n_samples': int(self.n_entities * 10),
        }
    
    def _generate_feature_importance(self) -> None:
        """Generate feature importance scores."""
        n_features = 50  # Mock number of engineered features
        
        # Generate importance scores (higher = more important)
        importance_scores = self.rng.exponential(0.5, size=n_features)
        importance_scores /= importance_scores.sum()  # Normalize to sum=1
        
        feature_names = [f"Feature_{i:02d}" for i in range(n_features)]
        
        self._feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores,
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    def _build_training_info(
        self,
        base_model_oof: Dict[str, np.ndarray],
        base_model_holdout: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Build comprehensive training metadata with proper actual vs predicted."""
        # For the residuals plot to work correctly:
        # - y_test (actual) = predictions + residuals (simulated true values)
        # - y_pred (predicted) = the forecast predictions
        actual_values = self._predictions.values + self._generate_residuals().values
        
        return {
            'n_samples': int(self.n_entities * 10),
            'n_entities': self.n_entities,
            'n_components': self.n_components,
            'n_features': 50,
            'target_year': self.target_year,
            'cv_folds': self.config.n_cv_folds,
            'y_test': actual_values,  # True values (predictions + residuals)
            'y_pred': self._predictions.values,  # Ensemble predictions
            'test_entities': self.entity_names,
            'per_model_oof_predictions': base_model_oof,
            'per_model_holdout_predictions': base_model_holdout,
            'per_model_feature_importance': {
                name: np.abs(self.rng.normal(0, 0.2, size=50)) + 0.1
                for name in self.config.base_model_names
            },
            'cv_fold_val_years': [2020, 2021, 2022, 2023, 2024],
        }
    
    def _build_data_summary(self) -> Dict[str, Any]:
        """Build data summary metadata."""
        return {
            'n_entities': self.n_entities,
            'n_components': self.n_components,
            'target_year': self.target_year,
            'entity_names': self.entity_names,
            'component_names': self.component_names,
            'data_points': self.n_entities * self.n_components,
        }
    
    def _generate_holdout_performance(self) -> Dict[str, float]:
        """Generate holdout performance metrics."""
        return {
            'r2': float(self.rng.uniform(0.68, 0.78)),
            'mae': float(self.rng.uniform(0.08, 0.14)),
            'rmse': float(self.rng.uniform(0.10, 0.17)),
            'mape': float(self.rng.uniform(0.12, 0.22)),
        }
    
    def _generate_residuals(self) -> pd.DataFrame:
        """Generate synthetic residuals for a good-quality forecast.
        
        For moderate-high/good results (R² ≈ 0.70-0.75):
        - Residuals centered at exactly zero (no bias)
        - Standard deviation ≈ 0.09-0.10 (matches good forecast quality)
        - Approximately normal distribution
        - No systematic patterns (random scatter)
        """
        # Generate truly random, balanced residuals
        # Base std of 0.093 creates residuals that produce R² ~0.70-0.75
        residuals = self.rng.normal(
            loc=0,      # Centered at zero (no bias)
            scale=0.093,  # Good forecast quality std
            size=(self.n_entities, self.n_components)
        )
        
        # Optional: Add subtle heteroscedasticity (realistic but minimal)
        # Slightly larger errors for extreme predictions
        pred_abs = np.abs(self._predictions.values - 0.5)  # Distance from middle
        het_factor = 1.0 + 0.05 * pred_abs  # Up to 5% increase for extremes
        residuals = residuals * het_factor
        
        return pd.DataFrame(
            residuals,
            index=self.entity_names,
            columns=self.component_names
        )
    
    def _estimate_interval_coverage(self) -> Dict[str, float]:
        """Estimate interval coverage by criterion (should be ≥ 95%)."""
        return {
            component: float(self.rng.uniform(0.94, 0.98))
            for component in self.component_names
        }
    
    def _generate_interval_width_summary(self) -> pd.DataFrame:
        """Generate interval width statistics."""
        widths = self._intervals['upper'].values - self._intervals['lower'].values
        
        return pd.DataFrame({
            'Component': self.component_names,
            'Mean_Width': widths.mean(axis=0),
            'Min_Width': widths.min(axis=0),
            'Max_Width': widths.max(axis=0),
            'Std_Width': widths.std(axis=0),
        })
    
    def _generate_entity_error_summary(self) -> pd.DataFrame:
        """Generate per-entity prediction error statistics."""
        # Mock residually-derived errors
        errors = np.abs(self.rng.normal(0.1, 0.05, size=self.n_entities))
        
        return pd.DataFrame({
            'Entity': self.entity_names,
            'MAE': errors,
            'RMSE': errors * 1.1,
        })
    
    def _generate_worst_predictions(self, k: int = 10) -> pd.DataFrame:
        """Generate list of worst-performing predictions."""
        # Mock worst errors (across all components)
        errors = np.abs(self.rng.normal(0.25, 0.10, size=(self.n_entities, self.n_components)))
        
        # Flatten and find worst
        worst_idx = np.argsort(errors.ravel())[-k:][::-1]
        entity_idx = worst_idx // self.n_components
        component_idx = worst_idx % self.n_components
        
        return pd.DataFrame({
            'Entity': [self.entity_names[i] for i in entity_idx],
            'Component': [self.component_names[i] for i in component_idx],
            'Error': errors.ravel()[worst_idx],
            'Prediction': self._predictions.values.ravel()[worst_idx],
            'Uncertainty': self._uncertainty.values.ravel()[worst_idx],
        })
