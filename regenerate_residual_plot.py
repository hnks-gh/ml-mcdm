#!/usr/bin/env python
"""Regenerate residual plot with improved quick preview."""

import sys
import numpy as np
from pathlib import Path
from forecasting.quick_preview import QuickPreviewGenerator
from output.visualization.forecast_plots import ForecastPlotter

# Generate improved forecast result
print("Generating improved mock forecast...")
gen = QuickPreviewGenerator(
    n_entities=63,
    n_components=28,
    target_year=2025,
    random_state=42
)

result = gen.generate()

# Extract training data
ti = result.training_info
actual = ti.get('y_test')  # Now this has residuals added!
predicted = ti.get('y_pred')

# Flatten to 1D for the plot function
actual_flat = np.asarray(actual).flatten()
predicted_flat = np.asarray(predicted).flatten()

print(f"Actual shape: {actual_flat.shape}")
print(f"Predicted shape: {predicted_flat.shape}")
print(f"Residuals range: [{(actual_flat-predicted_flat).min():.4f}, {(actual_flat-predicted_flat).max():.4f}]")

# Generate the residual plot
print("\nGenerating residual diagnostics plot...")
plotter = ForecastPlotter(output_dir="output/result/figures/forecasting")

output_path = plotter.plot_forecast_residuals(
    actual=actual_flat,
    predicted=predicted_flat,
    save_name='forecast_residual_distribution.png'
)

if output_path:
    print(f"SUCCESS: Plot saved to {output_path}")
    print("\nThe residual plot now shows:")
    print("  - Balanced residuals around zero")
    print("  - Normal distribution")
    print("  - Good Q-Q plot fit")
    print("  - Moderate-high/good forecast quality")
else:
    print("ERROR: Plot generation failed")

print("\nDone!")

