#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal test of quick preview generator - attributes check."""

import sys
import time

# Import the quick preview generator
from forecasting.quick_preview import QuickPreviewGenerator

def test_quick_preview():
    """Test quick preview generator directly."""
    print("\n" + "="*80)
    print("QUICK PREVIEW GENERATOR - ATTRIBUTES TEST")
    print("="*80 + "\n")
    
    # Parameters
    n_entities = 63  # provinces
    n_components = 29  # subcriteria
    target_year = 2024
    
    # Create and generate
    generator = QuickPreviewGenerator(
        n_entities=n_entities,
        n_components=n_components,
        target_year=target_year,
        random_state=42,
    )
    result = generator.generate()
    
    print(f"Result type: {type(result).__name__}")
    print(f"\nResult attributes:")
    for attr in dir(result):
        if not attr.startswith('_'):
            try:
                value = getattr(result, attr)
                if not callable(value):
                    if hasattr(value, 'shape'):
                        print(f"  {attr}: {type(value).__name__} shape={value.shape}")
                    elif isinstance(value, dict) and len(str(value)) < 100:
                        print(f"  {attr}: {type(value).__name__} - {value}")
                    elif isinstance(value, (list, tuple)) and len(value) < 5:
                        print(f"  {attr}: {type(value).__name__} - {value}")
                    else:
                        val_str = str(value)[:80]
                        print(f"  {attr}: {type(value).__name__}")
            except Exception as e:
                print(f"  {attr}: ERROR - {e}")
    print()
    print("="*80)

if __name__ == '__main__':
    try:
        test_quick_preview()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
