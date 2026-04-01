#!/usr/bin/env python
"""Test improved quick preview diagnostics."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Generate mock forecast result with improved quality
from forecasting.quick_preview import QuickPreviewGenerator
from output.visualization.forecast_plots import ForecastPlotter

# Create generator and result
gen = QuickPreviewGenerator(
    n_entities=63,
    n_components=28,
    target_year=2025,
    random_state=42
)

result = gen.generate()

print("Mock Forecast Quality Check")
print("=" * 60)

# Extract data
residuals = result.forecast_residuals.values.flatten()
predictions = result.predictions.values.flatten()

# Compute quality metrics
mean_res = residuals.mean()
std_res = residuals.std()
skew_res = ((residuals - mean_res)**3).mean() / (std_res**3)

ss_tot = predictions.var() * len(predictions)
ss_res = (residuals**2).sum()
r2 = 1 - (ss_res / ss_tot)

print(f"Residuals Mean:       {mean_res:>8.7f}  (target: 0.0)")
print(f"Residuals Std:        {std_res:>8.6f}  (target: ~0.09-0.10)")
print(f"Residuals Skewness:   {skew_res:>8.5f}  (target: ~0.0)")
print(f"R² Score:             {r2:>8.4f}  (target: 0.70-0.75)")
print()
print(f"Predictions Mean:     {predictions.mean():>8.4f}  (target: ~0.65)")
print(f"Predictions Range:    [{predictions.min():.3f}, {predictions.max():.3f}]")
print()

# Check quality criteria
criteria_met = []
criteria_met.append(("Unbiased (mean ~0)", abs(mean_res) < 0.01))
criteria_met.append(("Good variance (std ~0.09-0.10)", 0.09 <= std_res <= 0.11))
criteria_met.append(("Normal distribution (skew ~0)", abs(skew_res) < 0.2))
criteria_met.append(("Target R² (0.70-0.75)", 0.70 <= r2 <= 0.75))

print("Quality Criteria:")
for name, met in criteria_met:
    status = "PASS" if met else "FAIL"
    print(f"  [{status}] {name}")

print()
if all(met for _, met in criteria_met):
    print("EXCELLENT: Mock forecast at moderate-high/good quality")
    print("  Ready for residual diagnostics visualization!")
else:
    print("Adjustments needed")

# Now test if we can generate diagnostics
print()
print("=" * 60)
print("Generating Residual Diagnostics Visualization...")

try:
    # Need to provide actual values (we'll use predictions as "actual" for this test)
    # In real scenario, actual would be the target values
    plotter = ForecastPlotter(output_dir="output/visualization")
    
    # For this test, we create synthetic "actual" values with known relationship to predicted
    # actual ≈ predicted + small random noise
    actual_for_viz = predictions + residuals  # This reconstructs the "actual" values
    
    output_path = plotter.plot_forecast_residuals(
        actual=actual_for_viz,
        predicted=predictions,
        save_name='test_improved_residuals.png'
    )
    
    if output_path:
        print(f"Success: Residuals visualization created!")
        print(f"  Saved to: {output_path}")
    else:
        print("Warning: Plot returned None")
        
except Exception as e:
    print(f"Error generating diagnostics: {e}")
    import traceback
    traceback.print_exc()

print()
print("Test complete. Check the visualization to verify:")
print("  - Residuals scatter: Should be balanced around y=0")
print("  - Distribution: Should be approximately bell-shaped")
print("  - Q-Q plot: Should follow the diagonal line")
print("  - No systematic patterns or bimodal behavior")

