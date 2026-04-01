#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rank Forecast Provinces using 6 MCDM Methods with CRITIC Weighting

Core calculation script:
1. Load forecast predictions
2. Calculate CRITIC weights
3. Apply 6 MCDM methods (TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, SAW)
4. Get top 10 provinces ranked by median score across methods
5. Output: CSV with provinces as rows, methods as columns
"""

import pandas as pd
import numpy as np
from pathlib import Path

from weighting.critic import CRITICWeightCalculator
from ranking.topsis import TOPSISCalculator
from ranking.vikor import VIKORCalculator
from ranking.promethee import PROMETHEECalculator
from ranking.copras import COPRASCalculator
from ranking.edas import EDASCalculator
from ranking.saw import SAWCalculator


def load_forecast_data(csv_path):
    """Load forecast predictions CSV."""
    df = pd.read_csv(csv_path, index_col=0)
    # Ensure all numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


def calculate_critic_weights(data):
    """Calculate CRITIC weights from the data."""
    calculator = CRITICWeightCalculator()
    result = calculator.calculate(data)
    return result.weights


def apply_mcdm_methods(data, weights):
    """
    Apply all 6 MCDM methods and return normalized scores (0-1, higher=better).
    
    Returns
    -------
    dict
        Maps method name to normalized final scores (pd.Series)
    """
    scores = {}
    
    # 1. TOPSIS (higher is better, typically 0-1)
    topsis = TOPSISCalculator()
    topsis_result = topsis.calculate(data, weights)
    scores['TOPSIS'] = topsis_result.final_scores
    
    # 2. VIKOR (lower is better, invert to 1 - Q to make higher=better)
    vikor = VIKORCalculator()
    vikor_result = vikor.calculate(data, weights)
    scores['VIKOR'] = 1.0 - vikor_result.final_scores
    
    # 3. PROMETHEE (net flow: higher is better, but can be negative; normalize)
    promethee = PROMETHEECalculator()
    promethee_result = promethee.calculate(data, weights)
    phi_net = promethee_result.final_scores
    # Normalize to 0-1: (x - min) / (max - min)
    if phi_net.max() - phi_net.min() > 1e-10:
        scores['PROMETHEE'] = (phi_net - phi_net.min()) / (phi_net.max() - phi_net.min())
    else:
        scores['PROMETHEE'] = pd.Series(0.5, index=phi_net.index)
    
    # 4. COPRAS (utility in 0-100, normalize to 0-1)
    copras = COPRASCalculator()
    copras_result = copras.calculate(data, weights)
    scores['COPRAS'] = copras_result.final_scores / 100.0
    
    # 5. EDAS (higher is better, typically 0-1)
    edas = EDASCalculator()
    edas_result = edas.calculate(data, weights)
    scores['EDAS'] = edas_result.final_scores
    
    # 6. SAW (higher is better)
    saw = SAWCalculator()
    saw_result = saw.calculate(data, weights)
    scores['SAW'] = saw_result.final_scores
    
    return scores


def main():
    # Paths
    input_csv = Path("output/result/csv/forecasting/forecast_predictions.csv")
    output_csv = Path("output/result/csv/ranking/forecast_top10_mcdm.csv")
    
    print("Loading forecast predictions...")
    data = load_forecast_data(input_csv)
    print(f"Data shape: {data.shape}")
    
    print("\nCalculating CRITIC weights...")
    weights = calculate_critic_weights(data)
    print(f"Weights calculated: {len(weights)} criteria")
    
    print("\nApplying 6 MCDM methods...")
    method_scores = apply_mcdm_methods(data, weights)
    
    # Combine all scores into DataFrame
    scores_df = pd.DataFrame(method_scores)
    print(f"\nScores computed for {len(scores_df)} provinces, {len(scores_df.columns)} methods")
    
    # Calculate median score per province (after normalization)
    scores_df['Median_Score'] = scores_df.median(axis=1)
    
    # Round all to 4 decimal places
    scores_df = scores_df.round(4)
    
    # Get top 10 by median
    top10 = scores_df.nlargest(10, 'Median_Score')
    
    # Prepare output: just top 10 with method scores and rank
    output_df = top10.copy()
    output_df['Rank'] = range(1, len(output_df) + 1)
    output_df = output_df[['Rank', 'TOPSIS', 'VIKOR', 'PROMETHEE', 'COPRAS', 'EDAS', 'SAW', 'Median_Score']]
    
    # Create output directory if needed
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving top 10 provinces to {output_csv}")
    output_df.to_csv(output_csv)
    
    print("\nTop 10 Provinces by Median MCDM Score (normalized 0-1, 4 decimals):")
    print(output_df.to_string())
    
    return output_df


if __name__ == '__main__':
    main()
