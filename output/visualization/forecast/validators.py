# -*- coding: utf-8 -*-
"""
Forecast Visualization Payload Validation

Provides strict precondition and invariant checks for ForecastVizPayload.
All validation is explicit (no silent corrections) with meaningful error messages.

Responsibility:
- Hard type checks
- Shape/length alignment checks
- NaN/Inf finiteness checks
- Index consistency checks for DataFrames
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any

from output.visualization.forecast.contracts import ForecastVizPayload


class PayloadValidationError(ValueError):
    """Standardized exception for payload validation failures."""
    pass


class ValidatorWarning:
    """Container for non-fatal validation warnings (optional fields missing)."""
    
    def __init__(self, field_name: str, reason: str):
        self.field_name = field_name
        self.reason = reason
    
    def __repr__(self):
        return f"ValidatorWarning(field={self.field_name}, reason='{self.reason}')"


# ============================================================================
# Core Array Validators
# ============================================================================

def validate_array_finite(
    arr: np.ndarray,
    array_name: str,
    allow_empty: bool = False,
) -> None:
    """Validate that array is finite (no NaN, no Inf)."""
    if not isinstance(arr, np.ndarray):
        raise PayloadValidationError(
            f"{array_name} must be np.ndarray, got {type(arr)}"
        )
    
    if len(arr) == 0:
        if allow_empty:
            return
        raise PayloadValidationError(f"{array_name} is empty (length=0)")
    
    if not np.isfinite(arr).all():
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        raise PayloadValidationError(
            f"{array_name} contains {n_nan} NaN and {n_inf} Inf values. "
            "Validators expect clean data upstream."
        )


def validate_arrays_aligned(
    arr1: np.ndarray,
    arr2: np.ndarray,
    arr1_name: str,
    arr2_name: str,
) -> None:
    """Validate that two arrays have the same length."""
    if len(arr1) != len(arr2):
        raise PayloadValidationError(
            f"Alignment mismatch: len({arr1_name})={len(arr1)} "
            f"!= len({arr2_name})={len(arr2)}"
        )


# ============================================================================
# DataFrame Validators
# ============================================================================

def validate_dataframe_finite(
    df: pd.DataFrame,
    df_name: str,
) -> None:
    """Validate that DataFrame contains only finite values."""
    if not isinstance(df, pd.DataFrame):
        raise PayloadValidationError(
            f"{df_name} must be pd.DataFrame, got {type(df)}"
        )
    
    if df.empty:
        raise PayloadValidationError(f"{df_name} is empty (shape={df.shape})")
    
    # Check for NaN/Inf in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Use map() for pandas 2.1+ (previously applymap)
        non_finite = ~df[numeric_cols].map(np.isfinite).all()
        if non_finite.any():
            bad_cols = non_finite[non_finite].index.tolist()
            raise PayloadValidationError(
                f"{df_name} columns {bad_cols} contain NaN or Inf values"
            )


def validate_dataframe_index_aligned(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df1_name: str,
    df2_name: str,
) -> None:
    """Validate that two DataFrames have identical index."""
    if not df1.index.equals(df2.index):
        raise PayloadValidationError(
            f"Index mismatch: {df1_name}.index != {df2_name}.index. "
            f"{df1_name} has {len(df1)} rows, {df2_name} has {len(df2)} rows. "
            f"First 5 {df1_name} index: {df1.index[:5].tolist()}. "
            f"First 5 {df2_name} index: {df2.index[:5].tolist()}."
        )


# ============================================================================
# Payload Validators (Main Interface)
# ============================================================================

def validate_payload_essential(
    payload: ForecastVizPayload,
    warn=True,
) -> List[ValidatorWarning]:
    """
    Validate ForecastVizPayload for essential figure generation.
    
    Raises PayloadValidationError on hard precondition violations.
    Returns list of warnings for optional missing fields.
    
    Args:
        payload: ForecastVizPayload to validate.
        warn: If True, log warnings for missing optional fields. (Not implemented here.)
    
    Returns:
        List of ValidatorWarning for missing optional fields.
    
    Raises:
        PayloadValidationError: On alignment, type, or finiteness violations.
    """
    warnings = []
    
    # ====== Essential Fields: y_test, y_pred_ensemble ======
    validate_array_finite(payload.y_test, "y_test", allow_empty=False)
    validate_array_finite(payload.y_pred_ensemble, "y_pred_ensemble", allow_empty=False)
    validate_arrays_aligned(
        payload.y_test, payload.y_pred_ensemble,
        "y_test", "y_pred_ensemble"
    )
    
    # ====== Optional: entity_names alignment ======
    if payload.entity_names is not None:
        if len(payload.entity_names) != len(payload.y_test):
            raise PayloadValidationError(
                f"entity_names length mismatch: len(entity_names)={len(payload.entity_names)} "
                f"!= len(y_test)={len(payload.y_test)}"
            )
        if len(set(payload.entity_names)) != len(payload.entity_names):
            # Entity names have duplicates
            warnings.append(
                ValidatorWarning(
                    "entity_names",
                    "Contains duplicate names (will affect entity-level charts)"
                )
            )
    
    # ====== Optional: current_scores alignment ======
    if payload.current_scores is not None:
        validate_array_finite(payload.current_scores, "current_scores", allow_empty=False)
        validate_arrays_aligned(
            payload.current_scores, payload.y_test,
            "current_scores", "y_test"
        )
    
    # ====== Optional: Prediction intervals alignment ======
    if payload.interval_lower_df is not None or payload.interval_upper_df is not None:
        if payload.interval_lower_df is None or payload.interval_upper_df is None:
            raise PayloadValidationError(
                "interval_lower_df and interval_upper_df must both be provided or both be None "
                "(cannot have only one)"
            )
        
        validate_dataframe_finite(payload.interval_lower_df, "interval_lower_df")
        validate_dataframe_finite(payload.interval_upper_df, "interval_upper_df")
        validate_dataframe_index_aligned(
            payload.interval_lower_df, payload.interval_upper_df,
            "interval_lower_df", "interval_upper_df"
        )
        
        # Check that lower <= upper for all values
        lower = payload.interval_lower_df
        upper = payload.interval_upper_df
        if (lower > upper).any().any():
            n_bad = (lower > upper).sum().sum()
            raise PayloadValidationError(
                f"Interval consistency violated: {n_bad} cells have "
                "interval_lower_df > interval_upper_df (should be <=)"
            )
    
    # ====== Optional: predictions_df alignment ======
    if payload.predictions_df is not None:
        validate_dataframe_finite(payload.predictions_df, "predictions_df")
        if payload.entity_names is not None:
            pred_entities = set(payload.predictions_df.index)
            actual_entities = set(payload.entity_names)
            if pred_entities != actual_entities:
                warnings.append(
                    ValidatorWarning(
                        "predictions_df",
                        f"Index entities ({len(pred_entities)}) do not match entity_names ({len(actual_entities)})"
                    )
                )
    
    # ====== Optional: model_performance keys match model_contributions ======
    if payload.model_contributions and payload.model_performance:
        contrib_models = set(payload.model_contributions.keys())
        perf_models = set(payload.model_performance.keys())
        if contrib_models != perf_models:
            missing_in_perf = contrib_models - perf_models
            extra_in_perf = perf_models - contrib_models
            if missing_in_perf or extra_in_perf:
                warnings.append(
                    ValidatorWarning(
                        "model_contributions/model_performance",
                        f"Key mismatch: model_contributions keys {contrib_models} "
                        f"!= model_performance keys {perf_models}"
                    )
                )
    
    # ====== Optional: cv_scores alignment ======
    if payload.cv_scores:
        for model_name, scores in payload.cv_scores.items():
            if not isinstance(scores, (list, np.ndarray)):
                raise PayloadValidationError(
                    f"cv_scores['{model_name}'] must be list/array, got {type(scores)}"
                )
            if len(scores) == 0:
                warnings.append(
                    ValidatorWarning(
                        f"cv_scores[{model_name}]",
                        "Has zero CV folds"
                    )
                )
            # Convert to array and check finiteness
            scores_arr = np.asarray(scores, dtype=float).ravel()
            if not np.isfinite(scores_arr).all():
                n_bad = (~np.isfinite(scores_arr)).sum()
                raise PayloadValidationError(
                    f"cv_scores['{model_name}'] contains {n_bad} non-finite CV scores"
                )
    
    # ====== Advanced optional fields (light validation) ======
    if payload.per_model_oof_predictions is not None:
        for model_name, preds in payload.per_model_oof_predictions.items():
            validate_array_finite(preds, f"per_model_oof_predictions[{model_name}]")
            validate_arrays_aligned(
                preds, payload.y_test,
                f"per_model_oof_predictions[{model_name}]", "y_test"
            )
    
    if payload.per_model_holdout_predictions is not None:
        for model_name, preds in payload.per_model_holdout_predictions.items():
            validate_array_finite(preds, f"per_model_holdout_predictions[{model_name}]")
    
    if payload.per_model_feature_importance is not None:
        for model_name, importance in payload.per_model_feature_importance.items():
            validate_array_finite(importance, f"per_model_feature_importance[{model_name}]")
    
    if payload.cv_fold_val_years is not None:
        if not isinstance(payload.cv_fold_val_years, (list, np.ndarray)):
            raise PayloadValidationError(
                f"cv_fold_val_years must be list/array, got {type(payload.cv_fold_val_years)}"
            )
    
    return warnings


def validate_payload_for_figure(
    payload: ForecastVizPayload,
    required_fields: List[str],
) -> Tuple[bool, Optional[str]]:
    """
    Quick check: does payload have all fields required for a specific figure?
    
    Args:
        payload: Payload to check.
        required_fields: List of field names required by the figure.
                        Each name should be a valid ForecastVizPayload attribute.
    
    Returns:
        (is_valid, reason_if_invalid):
        - (True, None) if all required fields are present and non-None
        - (False, reason) if any required field is missing or None
    """
    for field_name in required_fields:
        if not hasattr(payload, field_name):
            return False, f"Unknown field: {field_name}"
        
        value = getattr(payload, field_name)
        
        # None check
        if value is None:
            return False, f"Required field {field_name} is None"
        
        # For collections (dict, list), check non-empty
        if isinstance(value, (dict, list)) and len(value) == 0:
            return False, f"Required field {field_name} is empty collection"
    
    return True, None


__all__ = [
    'PayloadValidationError',
    'ValidatorWarning',
    'validate_array_finite',
    'validate_arrays_aligned',
    'validate_dataframe_finite',
    'validate_dataframe_index_aligned',
    'validate_payload_essential',
    'validate_payload_for_figure',
]
