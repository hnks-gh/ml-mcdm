#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test to verify quick preview works in minimal UnifiedForecaster context."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths
sys.path.insert(0, str(Path(__file__).parent))

def test_unified_forecaster_quick_preview():
    """Test UnifiedForecaster with quick preview config."""
    
    print("\n" + "="*80)
    print("TEST: UnifiedForecaster with Quick Preview Config")
    print("="*80 + "\n")
    
    # Import config and forecaster
    from config import ForecastConfig
    from forecasting import UnifiedForecaster
    from data import PanelData
    
    # Create a ForecastConfig with quick_preview_mode=True
    print("1. Creating ForecastConfig with quick_preview_mode=True...")
    forecast_config = ForecastConfig()
    forecast_config.quick_preview_mode = True
    print(f"   quick_preview_mode = {forecast_config.quick_preview_mode}")
    print()
    
    # Create UnifiedForecaster with the config
    print("2. Creating UnifiedForecaster and passing config...")
    forecaster = UnifiedForecaster(
        conformal_method=forecast_config.conformal_method,
        conformal_alpha=forecast_config.conformal_alpha,
        cv_folds=forecast_config.cv_folds,
        cv_min_train_years=forecast_config.cv_min_train_years,
        random_state=forecast_config.random_state,
        verbose=forecast_config.verbose,
        target_level=forecast_config.forecast_level,
        config=forecast_config,  # CRITICAL: Pass config
    )
    print(f"   forecaster._config = {forecaster._config}")
    print(f"   forecaster._config.quick_preview_mode = {forecaster._config.quick_preview_mode}")
    print()
    
    # Create minimal test panel data
    print("3. Creating minimal PanelData for testing...")
    # Create simple test data
    n_provinces = 3
    n_years = 3
    n_subcriteria = 5
    
    years = [2021, 2022, 2023]
    provinces = [f'Province_{i}' for i in range(n_provinces)]
    subcriteria = [f'SC_{i}' for i in range(n_subcriteria)]
    
    # Create random panel data
    data = np.random.rand(n_provinces * n_years, n_subcriteria + 1)  # +1 for actual value
    index_data = []
    for year in years:
        for prov in provinces:
            index_data.append((prov, year))
    df = pd.DataFrame(data, columns=subcriteria + ['ACTUAL'])
    df.index = pd.MultiIndex.from_tuples(index_data, names=['Province', 'Year'])
    
    print(f"   Created test panel data: {df.shape}")
    print()
    
    # Create PanelData object (simplified - using direct assignment)
    print("4. Creating PanelData object...")
    from data.data_loader import PanelData as PanelDataClass
    
    panel_data = PanelDataClass(
        raw_df=df,
        hierarchy_mapping=None,
        year_contexts={year: None for year in years},
        base_year=2021,
        forecast_year=2024,
        target_col='ACTUAL'
    )
    print(f"   Panel data created: {panel_data.n_provinces} provinces, {panel_data.n_years} years")
    print()
    
    # Now test fit_predict
    print("5. Calling fit_predict with quick_preview_mode=True...")
    print("   (Should complete in milliseconds, not minutes)\n")
    
    import time
    start = time.time()
    try:
        result = forecaster.fit_predict(
            panel_data=panel_data,
            target_year=2024
        )
        elapsed = time.time() - start
        
        print(f"   ✓ Completed in {elapsed:.4f} seconds")
        print(f"   Result type: {type(result).__name__}")
        if result is not None:
            print(f"   Has predictions: {hasattr(result, 'predictions')} " +
                  f"(shape={result.predictions.shape if hasattr(result, 'predictions') else 'N/A'})")
        
        if elapsed > 5:
            print(f"\n   ⚠ WARNING: Took {elapsed:.2f}s - quick preview may not be active!")
            print("   (Quick preview should be < 500ms)")
        else:
            print(f"\n   ✓ QUICK PREVIEW WORKING: Completed in {elapsed:.4f}s")
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"   ✗ ERROR after {elapsed:.4f}s: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("="*80)

if __name__ == '__main__':
    try:
        test_unified_forecaster_quick_preview()
    except Exception as e:
        print(f"\n✗ SETUP ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
