#!/usr/bin/env python
# -*- coded: utf-8 -*-
"""Direct test of quick preview check in fit_predict."""

import sys
from pathlib import Path

# Set up paths
sys.path.insert(0, str(Path(__file__).parent))

def test_check():
    """Test that quick preview check works."""
    
    print("\n" + "="*80)
    print("DIRECT TEST: Quick Preview Check Logic")
    print("="*80 + "\n")
    
    # Import config and forecaster
    from config import ForecastConfig
    from forecasting import UnifiedForecaster
    
    # Create a ForecastConfig with quick_preview_mode=True
    print("1. Creating ForecastConfig with quick_preview_mode=True...")
    forecast_config = ForecastConfig()
    forecast_config.quick_preview_mode = True
    print(f"   quick_preview_mode = {forecast_config.quick_preview_mode}")
    print()
    
    # Create UnifiedForecaster WITH config
    print("2. Creating UnifiedForecaster and passing config...")
    forecaster = UnifiedForecaster(
        config=forecast_config,
    )
    print(f"   forecaster._config = {forecaster._config is not None}")
    print(f"   forecaster._config.quick_preview_mode = {forecaster._config.quick_preview_mode}")
    print()
    
    # Test the check logic directly
    print("3. Testing the check logic that happens in fit_predict()...")
    _config = forecaster._config
    _use_quick_preview = (_config is not None and _config.quick_preview_mode)
    print(f"   _config is not None: {_config is not None}")
    print(f"   _config.quick_preview_mode: {_config.quick_preview_mode}")
    print(f"   _use_quick_preview = {_use_quick_preview}")
    print()
    
    if _use_quick_preview:
        print("   ✓ SUCCESS: Quick preview check will activate")
    else:
        print("   ✗ FAILURE: Quick preview check will NOT activate")
    print()
    
    # Also test WITHOUT config
    print("4. Testing WITHOUT passing config (control)...")
    forecaster2 = UnifiedForecaster()
    _config2 = forecaster2._config
    _use_quick_preview2 = (_config2 is not None and getattr(_config2, 'quick_preview_mode', False))
    print(f"   forecaster2._config is not None: {_config2 is not None}")
    print(f"   _use_quick_preview2 = {_use_quick_preview2}")
    if not _use_quick_preview2:
        print("   ✓ Correct: Without config, quick preview is not active")
    print()
    print("="*80)

if __name__ == '__main__':
    try:
        test_check()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
