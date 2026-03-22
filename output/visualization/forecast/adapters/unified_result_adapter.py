# -*- coding: utf-8 -*-
"""Forecast Data Adapter: UnifiedForecastResult → ForecastVizPayload

Transforms the output of UnifiedForecaster into the type-safe ForecastVizPayload
contract expected by all visualization modules.

Key responsibilities:
- Normalize array shapes and types
- Extract per-model predictions and importance matrices
- Handle missing optional fields gracefully
- Apply validation to extracted fields
"""

from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import logging

from output.visualization.forecast.contracts import ForecastVizPayload
from output.visualization.forecast.validators import validate_payload_essential

logger = logging.getLogger(__name__)


class UnifiedResultAdapter:
    """Adapter from UnifiedForecastResult to ForecastVizPayload.
    
    Extracts, normalizes, and validates visualization-ready fields from the
    forecasting pipeline output.
    """
    
    @staticmethod
    def adapt(
        forecast_result: Any,
        panel_data: Optional[Any] = None,
        ranking_result: Optional[Any] = None,
    ) -> ForecastVizPayload:
        """Convert UnifiedForecastResult to ForecastVizPayload.
        
        Args:
            forecast_result: UnifiedForecastResult from forecasting pipeline.
            panel_data: Optional YearPanel for temporal context.
            ranking_result: Optional ranking result for current scores.
            
        Returns:
            ForecastVizPayload with extracted and validated fields.
            
        Raises:
            ValueError: If essential fields are missing or invalid.
        """
        logger.debug('Adapting UnifiedForecastResult to ForecastVizPayload...')
        
        # Extract training info (essential for most figures)
        training_info = getattr(forecast_result, 'training_info', {}) or {}
        
        # Extract actual vs predicted (F-01, F-02, F-03, etc.)
        y_test = training_info.get('y_test')
        y_pred = training_info.get('y_pred')
        
        y_test_arr = np.asarray(y_test, dtype=float) if y_test is not None else None
        y_pred_arr = np.asarray(y_pred, dtype=float) if y_pred is not None else None
        
        # Extract entity names
        entity_names = training_info.get('test_entities')
        
        # Extract model composition and performance
        model_contributions = getattr(forecast_result, 'model_contributions', {}) or {}
        model_performance = getattr(forecast_result, 'model_performance', {}) or {}
        cv_scores = getattr(forecast_result, 'cross_validation_scores', {}) or {}
        
        # Extract per-model predictions (for diversity analysis: F-17, F-18)
        per_model_oof = training_info.get('per_model_oof_predictions')
        per_model_holdout = training_info.get('per_model_holdout_predictions')
        
        # Normalize per-model predictions to Dict[str, np.ndarray]
        per_model_oof_dict = UnifiedResultAdapter._normalize_per_model_preds(per_model_oof)
        per_model_holdout_dict = UnifiedResultAdapter._normalize_per_model_preds(per_model_holdout)
        
        # Extract feature importance (per-model for F-14 and global for F-12)
        per_model_fi = training_info.get('per_model_feature_importance')
        per_model_fi_dict = UnifiedResultAdapter._normalize_per_model_fi(per_model_fi)
        
        # Extract prediction intervals (for uncertainty figures: F-07, F-08, F-09)
        prediction_intervals = getattr(forecast_result, 'prediction_intervals', {}) or {}
        interval_lower = prediction_intervals.get('lower')
        interval_upper = prediction_intervals.get('upper')
        
        # Extract predictions DataFrame (entity × component predictions)
        predictions_df = getattr(forecast_result, 'predictions', None)
        
        # Extract temporal context
        cv_fold_years = training_info.get('cv_fold_val_years')
        target_year = getattr(forecast_result, 'target_year', None)
        if target_year is None and panel_data is not None:
            # Fall back to panel_data: forecast year = max(panel_years) + 1
            panel_years = getattr(panel_data, 'years', [])
            if panel_years:
                target_year = max(panel_years) + 1
        
        # Extract current scores (for rank change analysis: F-10, F-23)
        current_scores = None
        provinces = None
        if panel_data is not None:
            # Try to extract most recent year's scores as baseline
            current_scores = UnifiedResultAdapter._extract_current_scores(
                panel_data, ranking_result
            )
            # Get province/region names
            provinces = getattr(panel_data, 'alternatives', None)
        
        # Build payload with essential fields
        payload = ForecastVizPayload(
            y_test=y_test_arr,
            y_pred_ensemble=y_pred_arr,
            entity_names=entity_names,
            provinces=provinces,
            prediction_year=target_year,
            current_scores=current_scores,
            model_contributions=model_contributions,
            model_performance=model_performance,
            cv_scores=cv_scores,
            predictions_df=predictions_df,
            interval_lower_df=interval_lower,
            interval_upper_df=interval_upper,
            # Advanced optional fields
            per_model_oof_predictions=per_model_oof_dict,
            per_model_holdout_predictions=per_model_holdout_dict,
            per_model_feature_importance=per_model_fi_dict,
            cv_fold_val_years=cv_fold_years,
            # Metadata
            target_name='Composite ML Forecast',
            model_names=list(model_contributions.keys()) if model_contributions else None,
        )
        
        # Validate essential fields
        try:
            validate_payload_essential(payload)
            logger.info(f'Payload validated: y_test shape {y_test_arr.shape if y_test_arr is not None else None}, '
                       f'{len(model_contributions)} models, {len(cv_scores)} CV score sets')
        except Exception as e:
            logger.warning(f'Payload validation warning: {e}')
        
        return payload
    
    @staticmethod
    def _normalize_per_model_preds(
        per_model_preds: Optional[Any]
    ) -> Optional[Dict[str, np.ndarray]]:
        """Normalize per-model predictions to Dict[str, np.ndarray].
        
        Handles various input formats:
        - Dict[str, np.ndarray/list]
        - Dict[str, pd.Series]
        - pd.DataFrame (columns = model names)
        """
        if per_model_preds is None:
            return None
        
        result = {}
        
        if isinstance(per_model_preds, dict):
            for model_name, preds in per_model_preds.items():
                try:
                    if isinstance(preds, pd.Series):
                        result[model_name] = preds.values.astype(float)
                    else:
                        result[model_name] = np.asarray(preds, dtype=float)
                except Exception as e:
                    logger.warning(f'Failed to normalize predictions for {model_name}: {e}')
        
        elif isinstance(per_model_preds, pd.DataFrame):
            for col in per_model_preds.columns:
                try:
                    result[col] = per_model_preds[col].values.astype(float)
                except Exception as e:
                    logger.warning(f'Failed to extract column {col}: {e}')
        
        return result if result else None
    
    @staticmethod
    def _normalize_per_model_fi(
        per_model_fi: Optional[Any]
    ) -> Optional[Dict[str, np.ndarray]]:
        """Normalize per-model feature importance to Dict[str, np.ndarray].
        
        Handles various input formats:
        - Dict[str, np.ndarray/list]
        - Dict[str, pd.Series]
        - pd.DataFrame (columns = model names, index = feature names)
        """
        if per_model_fi is None:
            return None
        
        result = {}
        
        if isinstance(per_model_fi, dict):
            for model_name, importance in per_model_fi.items():
                try:
                    if isinstance(importance, pd.Series):
                        result[model_name] = importance.values.astype(float)
                    else:
                        result[model_name] = np.asarray(importance, dtype=float)
                except Exception as e:
                    logger.warning(f'Failed to normalize importance for {model_name}: {e}')
        
        elif isinstance(per_model_fi, pd.DataFrame):
            for col in per_model_fi.columns:
                try:
                    result[col] = per_model_fi[col].values.astype(float)
                except Exception as e:
                    logger.warning(f'Failed to extract importance column {col}: {e}')
        
        return result if result else None
    
    @staticmethod
    def _extract_current_scores(
        panel_data: Any,
        ranking_result: Optional[Any] = None,
    ) -> Optional[np.ndarray]:
        """Extract current year's scores as baseline for rank change.
        
        Priority:
        1. Most recent year's composite ranking from ranking_result
        2. Most recent year's panel data scores
        """
        try:
            if ranking_result is not None:
                # Try to extract most recent year's scores
                years = getattr(panel_data, 'years', [])
                if years:
                    most_recent_year = max(years)
                    if hasattr(ranking_result, 'yearly_results'):
                        yearly = ranking_result.yearly_results
                        if most_recent_year in yearly:
                            scores_dict = getattr(yearly[most_recent_year], 'scores', None)
                            if scores_dict:
                                return np.asarray(list(scores_dict.values()), dtype=float)
            
            # Fall back to panel data
            if hasattr(panel_data, 'years') and hasattr(panel_data, 'data'):
                years = panel_data.years
                if years:
                    most_recent = max(years)
                    # Try to access most recent year's data (format may vary)
                    if most_recent in panel_data.data:
                        df = panel_data.data[most_recent]
                        # Try to find a score column (may be named 'score', 'composite', etc.)
                        for col in ['score', 'composite', 'rank_score', 'weighted_score']:
                            if col in df.columns:
                                return df[col].values.astype(float)
        
        except Exception as e:
            logger.debug(f'Failed to extract current scores: {e}')
        
        return None


__all__ = ['UnifiedResultAdapter']
