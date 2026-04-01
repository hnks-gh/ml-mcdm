#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal test of quick preview generator."""

import sys
import time

# Import the quick preview generator
from forecasting.quick_preview import QuickPreviewGenerator, QuickPreviewConfig

def test_quick_preview():
    """Test quick preview generator directly."""
    print("\n" + "="*80)
    print("QUICK PREVIEW GENERATOR - MINIMAL TEST")
    print("="*80 + "\n")
    
    # Parameters
    n_entities = 63  # provinces
    n_components = 29  # subcriteria
    target_year = 2024
    
    print(f"Parameters:")
    print(f"  Entities (provinces): {n_entities}")
    print(f"  Components (subcriteria): {n_components}")
    print(f"  Target year: {target_year}")
    print()
    
    # Create generator
    print("Creating QuickPreviewGenerator...")
    generator = QuickPreviewGenerator(
        n_entities=n_entities,
        n_components=n_components,
        target_year=target_year,
        random_state=42,
    )
    print("✓ Generator created successfully")
    print()
    
    # Generate results
    print("Generating synthetic forecast... (should be instant)")
    start = time.time()
    result = generator.generate()
    elapsed = time.time() - start
    
    print(f"✓ Mock forecast generated in {elapsed:.4f} seconds\n")
    
    # Print result info
    print("Result Properties:")
    print(f"  Type: {type(result).__name__}")
    print(f"  Has base_model_outputs: {result.base_model_outputs is not None}")
    if result.base_model_outputs:
        print(f"    Base models: {len(result.base_model_outputs)}")
        print(f"    Predictions shape: {result.base_model_outputs['CatBoost']['predictions'].shape if 'CatBoost' in result.base_model_outputs else 'N/A'}")
    print(f"  Has point_forecast: {result.point_forecast is not None}")
    if result.point_forecast is not None:
        print(f"    Shape: {result.point_forecast.shape}")
    print(f"  Has prediction_intervals: {result.prediction_intervals is not None}")
    if result.prediction_intervals is not None:
        print(f"    Shape: {result.prediction_intervals.shape}")
    print(f"  Has feature_importance_scores: {result.feature_importance_scores is not None}")
    if result.feature_importance_scores is not None:
        print(f"    Shape: {result.feature_importance_scores.shape}")
    print()
    
    print("="*80)
    print("✓ QUICK PREVIEW GENERATOR TEST PASSED")
    print("="*80)

if __name__ == '__main__':
    try:
        test_quick_preview()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
