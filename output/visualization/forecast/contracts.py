# -*- coding: utf-8 -*-
"""
Forecast Visualization Payload Contract

Defines ForecastVizPayload dataclass with strictly typed fields.
This is the single source of truth for data flow from UnifiedForecastResult
to all chart generation modules.

Separation of concerns:
- contracts.py: typing and structure (no validation, no computation)
- validators.py: precondition and invariant checks
- metrics.py: statistical calculations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class ForecastVizPayload:
    """
    Strictly typed payload for forecast visualization.
    
    Essential fields (required for basic essential figure suite):
    These fields must be present and non-None for the essential figure set to execute.
    
    Advanced optional fields (enable advanced figure generation):
    These fields are optional. Charts depending on them are skipped with a reason code.
    """
    
    # =====================================================================
    # ESSENTIAL FIELDS (Required for essential figures)
    # =====================================================================
    
    # Actual vs predicted (accuracy figures)
    y_test: np.ndarray
    """Test set ground truth. Shape (n_samples,). dtype float64."""
    
    y_pred_ensemble: np.ndarray
    """Ensemble predictions. Shape (n_samples,). dtype float64."""
    
    # Entity and temporal context
    entity_names: Optional[List[str]] = None
    """Entity/alternative names. Length must equal len(y_test).
    Required for entity-level aggregation (holdout comparison, entity errors).
    Optional for aggregate figures (overall scatter, residual distribution).
    """
    
    provinces: Optional[List[str]] = None
    """Province/region names. May differ from entity_names if entities aggregated.
    Used for rank change bubble, province comparison."""
    
    prediction_year: Optional[int] = None
    """Year of forecast prediction (e.g., 2025).
    Used for temporal context and score trajectory."""
    
    current_scores: Optional[np.ndarray] = None
    """Current/baseline scores for each entity (pre-forecast).
    Shape (n_samples,). Used for rank change analysis."""
    
    # Ensemble composition and performance
    model_contributions: Dict[str, float] = field(default_factory=dict)
    """Model weights. {model_name: weight}. Weights should sum to 1.0 (or normalized on plot)."""
    
    model_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """Per-model metrics. {model_name: {metric_name: value}}.
    Expected keys: 'r2', 'rmse', 'mae' at minimum."""
    
    cv_scores: Dict[str, List[float]] = field(default_factory=dict)
    """Cross-validation scores per model. {model_name: [fold_scores]}.
    Used for CV distribution plots."""
    
    # Prediction intervals and uncertainty
    predictions_df: Optional[pd.DataFrame] = None
    """DataFrame of per-model predictions. Index = entity_names.
    Used for diversity analysis (correlation heatmap, scatter matrix)."""
    
    interval_lower_df: Optional[pd.DataFrame] = None
    """Lower bound of prediction intervals. Index = entity_names.
    Used for interval calibration, conformal coverage."""
    
    interval_upper_df: Optional[pd.DataFrame] = None
    """Upper bound of prediction intervals. Index = entity_names.
    Used for interval calibration, conformal coverage."""
    
    # =====================================================================
    # ADVANCED OPTIONAL FIELDS (for advanced figure suite)
    # =====================================================================
    
    per_model_oof_predictions: Optional[Dict[str, np.ndarray]] = None
    """Out-of-fold predictions per base model.
    {model_name: array shape (n_samples,)}.
    Used for diversity analysis (prediction correlation)."""
    
    per_model_holdout_predictions: Optional[Dict[str, np.ndarray]] = None
    """Holdout predictions per base model.
    {model_name: array shape (n_holdout_samples,)}.
    Used for per-model holdout comparison."""
    
    per_model_feature_importance: Optional[Dict[str, np.ndarray]] = None
    """Feature importance per model (normalized or raw).
    {model_name: array shape (n_features,)}.
    Used for per-model importance heatmap."""
    
    cv_fold_val_years: Optional[List[int]] = None
    """Validation years for each CV fold.
    Used for temporal training curve (walk-forward by fold year)."""
    
    # =====================================================================
    # METADATA (optional but recommended)
    # =====================================================================
    
    target_name: Optional[str] = None
    """Name of target variable (e.g., 'Criterion C01: Safety')."""
    
    model_names: Optional[List[str]] = None
    """Names of base models in ensemble (ordering for deterministic coloring)."""
    
    feature_names: Optional[List[str]] = None
    """Names of features (used in importance plots)."""
    
    @property
    def is_minimal(self) -> bool:
        """Check if payload has minimum essential fields."""
        return (
            self.y_test is not None
            and self.y_pred_ensemble is not None
            and len(self.y_test) == len(self.y_pred_ensemble)
        )
    
    @property
    def has_entity_details(self) -> bool:
        """Check if payload has entity-level information."""
        return self.entity_names is not None and len(self.entity_names) == len(self.y_test)
    
    @property
    def has_intervals(self) -> bool:
        """Check if payload has prediction intervals."""
        return (
            self.interval_lower_df is not None
            and self.interval_upper_df is not None
        )
    
    @property
    def has_advanced_options(self) -> bool:
        """Check if payload has any advanced optional fields."""
        return (
            self.per_model_oof_predictions is not None
            or self.per_model_holdout_predictions is not None
            or self.per_model_feature_importance is not None
            or self.cv_fold_val_years is not None
        )


__all__ = ['ForecastVizPayload']
