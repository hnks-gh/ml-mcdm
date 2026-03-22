"""
Phase 4 Test: End-to-End Integration with VisualizationOrchestrator

Verifies that the ForecastPlotter facade integrates correctly with the existing
VisualizationOrchestrator and generates visualization artifacts without errors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import sys
import os
import tempfile

# Add workspace root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_mock_forecast_data() -> dict:
    """
    Create minimal mock forecast output for testing.
    Simulates structure of UnifiedForecastResult.
    """
    n_samples = 50
    n_models = 5
    
    y_test = np.random.randn(n_samples) * 10 + 50  # Mean 50, std 10
    y_pred_ensemble = y_test + np.random.randn(n_samples) * 2  # Slight noise
    
    # Mock per-model predictions
    per_model_preds = {}
    model_names = [f"model_{i}" for i in range(n_models)]
    for name in model_names:
        per_model_preds[name] = y_pred_ensemble + np.random.randn(n_samples) * 1
    
    # Mock model contributions (weights/metadata)
    model_contributions = {
        name: float(np.random.dirichlet([1] * n_models)[i])
        for i, name in enumerate(model_names)
    }
    
    # Mock model performance
    model_performance = {
        name: {
            'r2': float(np.random.uniform(0.5, 0.95)),
            'rmse': float(np.random.uniform(1, 5)),
            'mae': float(np.random.uniform(1, 4)),
        }
        for name in model_names
    }
    
    # Mock CV scores
    cv_scores = {
        name: list(np.random.uniform(0.5, 0.95, 5))
        for name in model_names
    }
    
    # Mock entity names (match sample size)
    entity_names = [f"entity_{i}" for i in range(n_samples)]
    
    # Mock feature importance
    feature_names = [f"feature_{j}" for j in range(10)]
    feature_importance = {
        name: float(val) / 10.0
        for name, val in zip(feature_names, np.random.exponential(1, 10))
    }
    
    return {
        'y_test': y_test,
        'y_pred_ensemble': y_pred_ensemble,
        'per_model_predictions': per_model_preds,
        'model_contributions': model_contributions,
        'model_performance': model_performance,
        'cv_scores': cv_scores,
        'entity_names': entity_names,
        'feature_importance': feature_importance,
    }


def test_basic_facade_methods():
    """Test a representative sample of facade methods with mock data."""
    
    from output.visualization.forecast_plots import ForecastPlotter
    
    mock_data = create_mock_forecast_data()
    plotter = ForecastPlotter(output_dir='./test_output', dpi=150)
    
    # Test 1: Accuracy method (F-01)
    try:
        result = plotter.plot_forecast_scatter(
            actual=mock_data['y_test'],
            predicted=mock_data['y_pred_ensemble'],
            entity_names=mock_data['entity_names']
        )
        # Result should be None (no matplotlib) or str (path)
        assert result is None or isinstance(result, str), \
            f"Expected None or str, got {type(result)}"
        print("[OK] plot_forecast_scatter works")
    except Exception as e:
        print("[FAIL] plot_forecast_scatter failed: {}".format(e))
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Ensemble method (F-04)
    try:
        result = plotter.plot_model_weights_donut(
            weights=mock_data['model_contributions']
        )
        assert result is None or isinstance(result, str)
        print("[OK] plot_model_weights_donut works")
    except Exception as e:
        print("[FAIL] plot_model_weights_donut failed: {}".format(e))
        return False
    
    # Test 3: Feature importance method (F-12)
    try:
        result = plotter.plot_feature_importance(
            importance=mock_data['feature_importance'],
            top_n=10
        )
        assert result is None or isinstance(result, str)
        print("[OK] plot_feature_importance works")
    except Exception as e:
        print("[FAIL] plot_feature_importance failed: {}".format(e))
        return False
    
    # Test 4: Ensemble method (F-05)
    try:
        result = plotter.plot_model_performance(
            model_metrics=mock_data['model_performance']
        )
        assert result is None or isinstance(result, str)
        print("[OK] plot_model_performance works")
    except Exception as e:
        print("[FAIL] plot_model_performance failed: {}".format(e))
        return False
    
    # Test 5: Ensemble method (F-06)
    try:
        result = plotter.plot_cv_boxplots(
            cv_scores=mock_data['cv_scores']
        )
        assert result is None or isinstance(result, str)
        print("[OK] plot_cv_boxplots works")
    except Exception as e:
        print("[FAIL] plot_cv_boxplots failed: {}".format(e))
        return False
    
    return True


def test_facade_delegation():
    """Verify that facade methods delegate to chart modules correctly."""
    
    from output.visualization.forecast_plots import ForecastPlotter
    from output.visualization.forecast.charts.accuracy import AccuracyCharts
    from output.visualization.forecast.charts.ensemble import EnsembleCharts
    
    plotter = ForecastPlotter(output_dir='./test_output', dpi=150)
    
    # Check that chart modules are correctly instantiated
    assert isinstance(plotter._accuracy, AccuracyCharts), "Accuracy module not initialized"
    assert isinstance(plotter._ensemble, EnsembleCharts), "Ensemble module not initialized"
    
    print("[OK] Chart module delegation verified")
    return True


def test_integration_with_visualization_orchestrator():
    """
    Test that ForecastPlotter works within VisualizationOrchestrator context.
    
    This is a simplified test - the real test would use actual forecast pipeline output.
    """
    
    from output.visualization import VisualizationOrchestrator
    
    # Create temporary output directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Initialize orchestrator (this creates ForecastPlotter internally)
            orchestrator = VisualizationOrchestrator(output_dir=tmpdir)
            assert hasattr(orchestrator, 'forecast'), "VisualizationOrchestrator missing forecast attribute"
            print("[OK] VisualizationOrchestrator initializes ForecastPlotter facade correctly")
            return True
        except Exception as e:
            print("[FAIL] VisualizationOrchestrator integration failed: {}".format(e))
            return False


def run_all_integration_tests():
    """Run all Phase 4 integration tests."""
    
    print("\n" + "="*70)
    print("PHASE 4: INTEGRATION TEST SUITE")
    print("="*70 + "\n")
    
    tests = [
        ("Facade Method Delegation", test_basic_facade_methods),
        ("Chart Module Routing", test_facade_delegation),
        ("VisualizationOrchestrator Integration", test_integration_with_visualization_orchestrator),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print("\n[FAIL] {}: Exception during execution".format(test_name))
            print("  {}: {}".format(type(e).__name__, e))
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print("{}: {}".format(status, test_name))
    
    print("\nTotal: {}/{} tests passed".format(passed, total))
    
    if passed == total:
        print("\nPHASE 4 INTEGRATION: ALL TESTS PASSED")
        return 0
    else:
        print("\nPHASE 4: {} test(s) failed".format(total - passed))
        return 1


if __name__ == '__main__':
    sys.exit(run_all_integration_tests())
