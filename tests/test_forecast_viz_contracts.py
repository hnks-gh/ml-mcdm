# -*- coding: utf-8 -*-
"""
Test: Forecast Visualization Contracts and Data Structures

Tests for ForecastVizPayload dataclass and related typed structures.
These tests verify that the payload contract is correctly defined and
that invalid payloads are detected early.
"""

import pytest
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ============================================================================
# Test Fixtures (Synthetic Data)
# ============================================================================

@pytest.fixture
def minimal_synthetic_payload():
    """Minimal valid ForecastVizPayload for essential figures."""
    return {
        'y_test': np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
        'y_pred_ensemble': np.array([0.51, 0.58, 0.72, 0.79, 0.88]),
        'entity_names': ['Entity_A', 'Entity_B', 'Entity_C', 'Entity_D', 'Entity_E'],
        'model_contributions': {'Model1': 0.5, 'Model2': 0.5},
        'model_performance': {
            'Model1': {'r2': 0.92, 'rmse': 0.05, 'mae': 0.04},
            'Model2': {'r2': 0.88, 'rmse': 0.07, 'mae': 0.06},
        },
        'cv_scores': {
            'Model1': [0.90, 0.91, 0.92],
            'Model2': [0.87, 0.88, 0.89],
        },
        'predictions_df': pd.DataFrame(
            {'pred_col': [0.51, 0.58, 0.72, 0.79, 0.88]},
            index=['Entity_A', 'Entity_B', 'Entity_C', 'Entity_D', 'Entity_E'],
        ),
        'interval_lower_df': pd.DataFrame(
            {'pred_col': [0.45, 0.52, 0.66, 0.73, 0.82]},
            index=['Entity_A', 'Entity_B', 'Entity_C', 'Entity_D', 'Entity_E'],
        ),
        'interval_upper_df': pd.DataFrame(
            {'pred_col': [0.57, 0.64, 0.78, 0.85, 0.94]},
            index=['Entity_A', 'Entity_B', 'Entity_C', 'Entity_D', 'Entity_E'],
        ),
        'current_scores': np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
        'provinces': ['Province_1', 'Province_2', 'Province_3', 'Province_4', 'Province_5'],
        'prediction_year': 2025,
    }


@pytest.fixture
def extended_synthetic_payload(minimal_synthetic_payload):
    """Extended payload with advanced optional fields."""
    payload = minimal_synthetic_payload.copy()
    payload.update({
        'per_model_oof_predictions': {
            'Model1': np.random.randn(5),
            'Model2': np.random.randn(5),
        },
        'per_model_holdout_predictions': {
            'Model1': np.random.randn(5),
            'Model2': np.random.randn(5),
        },
        'per_model_feature_importance': {
            'Model1': np.array([0.1, 0.2, 0.15, 0.25, 0.3]),
            'Model2': np.array([0.15, 0.18, 0.17, 0.28, 0.22]),
        },
        'cv_fold_val_years': [2021, 2022, 2023],
    })
    return payload


# ============================================================================
# Tests: Payload Structure (Phase 1 Placeholders)
# ============================================================================

class TestForecastVizPayloadStructure:
    """Test ForecastVizPayload contract definition and instantiation."""

    def test_payload_dataclass_exists(self):
        """ForecastVizPayload dataclass can be imported and instantiated."""
        from output.visualization.forecast.contracts import ForecastVizPayload
        payload = ForecastVizPayload(
            y_test=np.array([1.0, 2.0, 3.0]),
            y_pred_ensemble=np.array([1.1, 2.1, 3.1]),
        )
        assert payload is not None
        assert hasattr(payload, 'y_test')
        assert hasattr(payload, 'y_pred_ensemble')

    def test_required_fields_present(self, minimal_synthetic_payload):
        """Payload contains all required essential fields."""
        # TODO: Implement validation
        assert 'y_test' in minimal_synthetic_payload
        assert 'y_pred_ensemble' in minimal_synthetic_payload
        assert 'model_contributions' in minimal_synthetic_payload

    def test_optional_fields_allowed(self, extended_synthetic_payload):
        """Payload accepts optional advanced fields."""
        assert 'per_model_oof_predictions' in extended_synthetic_payload
        assert 'per_model_feature_importance' in extended_synthetic_payload

    def test_field_type_checking(self, minimal_synthetic_payload):
        """Fields have correct types (y_test is ndarray, predictions_df is DataFrame, etc.)."""
        assert isinstance(minimal_synthetic_payload['y_test'], np.ndarray)
        assert isinstance(minimal_synthetic_payload['predictions_df'], pd.DataFrame)
        assert isinstance(minimal_synthetic_payload['model_contributions'], dict)


class TestPayloadValidation:
    """Test validation of ForecastVizPayload requirements (Phase 1)."""

    def test_alignment_y_test_y_pred(self, minimal_synthetic_payload):
        """Validation ensures len(y_test) == len(y_pred_ensemble)."""
        from output.visualization.forecast.contracts import ForecastVizPayload
        from output.visualization.forecast.validators import validate_payload_essential
        
        payload = ForecastVizPayload(**minimal_synthetic_payload)
        warnings = validate_payload_essential(payload)  # should not raise
        assert isinstance(warnings, list)

    def test_entity_names_alignment(self, minimal_synthetic_payload):
        """Entity names length matches predictions if provided."""
        from output.visualization.forecast.contracts import ForecastVizPayload
        from output.visualization.forecast.validators import PayloadValidationError, validate_payload_essential
        
        # Valid case: matching lengths
        payload = ForecastVizPayload(**minimal_synthetic_payload)
        warnings = validate_payload_essential(payload)  # should not raise
        
        # Invalid case: mismatched lengths
        bad_payload_dict = minimal_synthetic_payload.copy()
        bad_payload_dict['entity_names'] = ['Entity_A', 'Entity_B']  # Only 2, but 5 predictions
        bad_payload = ForecastVizPayload(**bad_payload_dict)
        
        with pytest.raises(PayloadValidationError):
            validate_payload_essential(bad_payload)

    def test_interval_dataframe_index_consistency(self, minimal_synthetic_payload):
        """Interval DataFrames have consistent index."""
        lower_idx = set(minimal_synthetic_payload['interval_lower_df'].index)
        upper_idx = set(minimal_synthetic_payload['interval_upper_df'].index)
        assert lower_idx == upper_idx

    def test_missing_required_field_detected(self):
        """Payload missing required field raises validation error."""
        from output.visualization.forecast.contracts import ForecastVizPayload
        from output.visualization.forecast.validators import PayloadValidationError, validate_payload_essential
        
        # Create payload missing y_pred_ensemble
        payload = ForecastVizPayload(
            y_test=np.array([1.0, 2.0, 3.0]),
            y_pred_ensemble=None,  # Missing!
        )
        
        with pytest.raises(PayloadValidationError):
            validate_payload_essential(payload)


class TestPayloadFromUnifiedResult:
    """Test adapter that converts UnifiedForecastResult -> ForecastVizPayload (Phase 3)."""

    def test_adapter_extracts_all_fields(self):
        """UnifiedResultAdapter maps all UnifiedForecastResult fields correctly."""
        # TODO: Implement when unified_result_adapter.py is created
        # from output.visualization.forecast.adapters import UnifiedResultAdapter
        # from forecasting.unified import UnifiedForecastResult
        # adapter = UnifiedResultAdapter(unified_result_instance)
        # payload = adapter.to_payload()
        # assert payload.y_test is not None
        pytest.skip("Awaiting Phase 3 adapter implementation")

    def test_adapter_handles_missing_optional_fields(self):
        """Adapter gracefully handles absent optional fields in UnifiedForecastResult."""
        # TODO: Test adapter does not crash when per_model_oof_predictions is None
        pytest.skip("Awaiting Phase 3 adapter implementation")


# ============================================================================
# Tests: Dataclass Definition (Phase 1 Target)
# ============================================================================

class TestContractsPhase1:
    """Placeholder tests for core contract definitions in Phase 1."""

    def test_forecast_viz_payload_contract(self):
        """Contract: ForecastVizPayload contains all fields in specification."""
        # Expected fields from PLAN_forecast_visualization_refactor.md Section 7
        expected_essential = [
            'y_test', 'y_pred_ensemble', 'entity_names', 'model_contributions',
            'model_performance', 'cv_scores', 'predictions_df', 'interval_lower_df',
            'interval_upper_df', 'current_scores', 'provinces', 'prediction_year',
        ]
        expected_optional = [
            'per_model_oof_predictions', 'per_model_holdout_predictions',
            'per_model_feature_importance', 'cv_fold_val_years',
        ]
        # TODO: Once contracts.py exists:
        # from output.visualization.forecast.contracts import ForecastVizPayload
        # payload_fields = [f.name for f in fields(ForecastVizPayload)]
        # for field_name in expected_essential:
        #     assert field_name in payload_fields
        pytest.skip("Awaiting Phase 1 contracts.py implementation")


__all__ = [
    'TestForecastVizPayloadStructure',
    'TestPayloadValidation',
    'TestPayloadFromUnifiedResult',
    'TestContractsPhase1',
]
