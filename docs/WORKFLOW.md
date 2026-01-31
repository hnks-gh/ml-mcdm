# ML-MCDM Workflow Guide

This document provides a detailed step-by-step description of the ML-MCDM analysis pipeline workflow.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Phase 1: Data Loading](#phase-1-data-loading)
4. [Phase 2: Weight Calculation](#phase-2-weight-calculation)
5. [Phase 3: MCDM Analysis](#phase-3-mcdm-analysis)
6. [Phase 4: ML Analysis](#phase-4-ml-analysis)
7. [Phase 5: Ensemble Integration](#phase-5-ensemble-integration)
8. [Phase 6: Advanced Analysis](#phase-6-advanced-analysis)
9. [Phase 7: Visualization](#phase-7-visualization)
10. [Phase 8: Output Generation](#phase-8-output-generation)
11. [Configuration Options](#configuration-options)
12. [Customization Guide](#customization-guide)

---

## Overview

The ML-MCDM pipeline is designed to analyze panel data (multiple entities across multiple time periods) using a combination of Multi-Criteria Decision Making (MCDM) methods and Machine Learning techniques. The pipeline is fully automated and produces comprehensive outputs including rankings, visualizations, and detailed reports.

### Key Design Principles

1. **Modularity**: Each phase is independent and can be customized
2. **Robustness**: Comprehensive error handling with graceful fallbacks
3. **Transparency**: Detailed logging and intermediate results
4. **Reproducibility**: Configurable random seeds and parameter management

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ML-MCDM Analysis Pipeline                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Phase 1     │───▶│  Phase 2     │───▶│  Phase 3     │               │
│  │  Data Load   │    │  Weights     │    │  MCDM        │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Panel Data  │    │  Entropy     │    │  TOPSIS      │               │
│  │  - Long      │    │  CRITIC      │    │  Dynamic     │               │
│  │  - Wide      │    │  Ensemble    │    │  VIKOR       │               │
│  │  - Cross-sec │    │              │    │  Fuzzy       │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                 │                        │
│                                                 ▼                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Phase 6     │◀───│  Phase 5     │◀───│  Phase 4     │               │
│  │  Analysis    │    │  Ensemble    │    │  ML          │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Convergence │    │  Stacking    │    │  Panel Reg   │               │
│  │  Sensitivity │    │  Borda       │    │  Random For  │               │
│  │              │    │  Copeland    │    │  LSTM        │               │
│  │              │    │              │    │  Rough Sets  │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                                          │
│         │                   │                   │                        │
│         └───────────────────┼───────────────────┘                        │
│                             ▼                                            │
│  ┌──────────────┐    ┌──────────────┐                                   │
│  │  Phase 7     │───▶│  Phase 8     │                                   │
│  │  Visualize   │    │  Output      │                                   │
│  └──────────────┘    └──────────────┘                                   │
│                             │                                            │
│                             ▼                                            │
│                    ┌──────────────┐                                      │
│                    │  Results     │                                      │
│                    │  - figures/  │                                      │
│                    │  - results/  │                                      │
│                    │  - reports/  │                                      │
│                    └──────────────┘                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Loading

### Purpose
Load and structure panel data for analysis.

### Input Options

1. **CSV File**: Load from `data/data.csv`
2. **Synthetic Data**: Auto-generate if no file provided

### Data Structures Created

```python
PanelData:
├── long: DataFrame         # Long format (Year, Province, C01...C20)
├── wide: Dict[year, DataFrame]  # Province × Components per year
├── cross_section: Dict[year, DataFrame]  # Year-specific cross-sections
├── provinces: List[str]    # Entity identifiers
├── years: List[int]        # Time periods
└── components: List[str]   # Criteria names
```

### Processing Steps

1. **Load raw data** from CSV or generate synthetic
2. **Validate structure** (required columns, data types)
3. **Create views**:
   - Long format for regression
   - Wide format for time-series
   - Cross-sectional for MCDM
4. **Feature engineering** (temporal features, lags, trends)

### Example Code

```python
from src.data_loader import PanelDataLoader

loader = PanelDataLoader(config)
panel_data = loader.load('data/data.csv')

# Or generate synthetic
panel_data = loader.generate_synthetic(
    n_provinces=64,
    n_years=5,
    n_components=20
)
```

### Output

```
Panel data loaded: 64 entities, 5 periods, 20 components
```

---

## Phase 2: Weight Calculation

### Purpose
Determine objective weights for each criterion using data-driven methods.

### Methods Applied

| Method | Description | Output |
|--------|-------------|--------|
| Entropy | Based on information content | `entropy_weights` |
| CRITIC | Based on contrast and correlation | `critic_weights` |
| Ensemble | Geometric mean of above | `ensemble_weights` |

### Processing Steps

1. **Get latest cross-section** (most recent year)
2. **Calculate Entropy weights**:
   - Normalize columns to proportions
   - Calculate Shannon entropy
   - Convert to weights
3. **Calculate CRITIC weights**:
   - Calculate standard deviation
   - Calculate correlation matrix
   - Compute information content
4. **Combine into Ensemble**:
   - Geometric mean of Entropy and CRITIC
   - Normalize to sum to 1

### Data Flow

```
Cross-section (latest year)
        │
        ├──▶ Entropy Calculator ──▶ entropy_weights
        │
        ├──▶ CRITIC Calculator ──▶ critic_weights
        │
        └──▶ Ensemble Calculator ──▶ ensemble_weights
```

### Output

```
Entropy weights range: [0.0312, 0.0723]
CRITIC weights range: [0.0289, 0.0815]
```

---

## Phase 3: MCDM Analysis

### Purpose
Calculate rankings using multiple MCDM methods.

### Methods Applied

| Method | Input | Output |
|--------|-------|--------|
| TOPSIS | Latest cross-section + weights | scores, rankings |
| Dynamic TOPSIS | Full panel data + weights | temporal scores |
| VIKOR | Latest cross-section + weights | S, R, Q values |
| Fuzzy TOPSIS | Panel data (for variance) + weights | fuzzy scores |

### Processing Steps

1. **Standard TOPSIS**:
   - Normalize decision matrix
   - Apply weights
   - Calculate ideal/anti-ideal solutions
   - Calculate distances and closeness coefficients
   
2. **Dynamic TOPSIS**:
   - Apply temporal discount factors
   - Calculate trajectory scores
   - Compute stability scores
   - Combine into dynamic score

3. **VIKOR**:
   - Calculate S (group utility)
   - Calculate R (individual regret)
   - Calculate Q (compromise)
   - Check acceptance conditions

4. **Fuzzy TOPSIS**:
   - Generate triangular fuzzy numbers from temporal variance
   - Apply fuzzy TOPSIS algorithm
   - Defuzzify results

### Data Flow

```
Panel Data + Ensemble Weights
        │
        ├──▶ TOPSIS ──▶ topsis_scores, topsis_rankings
        │
        ├──▶ Dynamic TOPSIS ──▶ dynamic_topsis_scores
        │
        ├──▶ VIKOR ──▶ S, R, Q, vikor_rankings
        │
        └──▶ Fuzzy TOPSIS ──▶ fuzzy_scores
```

### Output

```
TOPSIS: Top performer score = 0.7845
VIKOR: Best alternative Q = 0.0231
Fuzzy TOPSIS: Top performer = 0.8123
```

---

## Phase 4: ML Analysis

### Purpose
Apply machine learning methods for validation, feature importance, and forecasting.

### Methods Applied

| Method | Purpose | Output |
|--------|---------|--------|
| Panel Regression | Coefficient estimation | coefficients, R² |
| Random Forest | Feature importance | importance scores |
| LSTM | Future prediction | forecasts |
| Rough Sets | Attribute reduction | core attributes, reducts |

### Processing Steps

1. **Prepare Features**:
   - Create temporal features (lags, trends, rolling means)
   - Map MCDM scores to panel data
   
2. **Panel Regression**:
   - Fit Fixed Effects model
   - Estimate coefficients
   - Calculate significance tests

3. **Random Forest with TS-CV**:
   - Time-series cross-validation
   - Train final model
   - Extract feature importance

4. **LSTM Forecasting**:
   - Prepare sequences
   - Train LSTM model
   - Generate predictions

5. **Rough Set Reduction**:
   - Discretize attributes
   - Find core attributes
   - Identify reducts

### Data Flow

```
Panel Data + MCDM Scores
        │
        ├──▶ Panel Regression ──▶ coefficients, R²
        │
        ├──▶ Random Forest ──▶ feature_importance
        │
        ├──▶ LSTM ──▶ lstm_forecasts
        │
        └──▶ Rough Sets ──▶ core_attributes, reducts
```

### Error Handling

Each ML method has fallback handling:
- If Panel Regression fails → Skip with warning
- If Random Forest fails → Return empty importance
- If LSTM fails → Return null forecasts
- If Rough Sets fails → Return empty reduction

---

## Phase 5: Ensemble Integration

### Purpose
Combine results from multiple methods into a final ranking.

### Methods Applied

| Method | Input | Output |
|--------|-------|--------|
| Stacking | All score predictions | weighted predictions |
| Borda Count | All rankings | aggregated ranking |
| Copeland | All rankings | pairwise-based ranking |

### Processing Steps

1. **Prepare Base Predictions**:
   ```python
   base_predictions = {
       'topsis': topsis_scores,
       'dynamic_topsis': dynamic_scores,
       'vikor_q': vikor_Q,
       'fuzzy': fuzzy_scores,
       'rf_pred': rf_predictions  # if available
   }
   ```

2. **Stacking Ensemble**:
   - Train Ridge meta-learner
   - Generate weighted predictions
   - Calculate meta-model weights

3. **Rank Aggregation**:
   - Convert all scores to rankings
   - Apply Borda Count (positional voting)
   - Apply Copeland (pairwise comparisons)
   - Calculate Kendall's W (agreement)

### Data Flow

```
MCDM Scores + ML Predictions
        │
        ├──▶ Stacking ──▶ meta_weights, final_predictions
        │
        └──▶ Rank Aggregation
             ├──▶ Borda Count ──▶ borda_ranking
             └──▶ Copeland ──▶ copeland_ranking
                     │
                     ▼
             ┌──────────────────┐
             │ Final Aggregated │
             │     Ranking      │
             │ + Kendall's W    │
             └──────────────────┘
```

### Output

```
Kendall's W (agreement): 0.8234
Top 3: [Province_12, Province_45, Province_08]
```

---

## Phase 6: Advanced Analysis

### Purpose
Assess robustness and temporal dynamics of rankings.

### Methods Applied

| Method | Purpose | Output |
|--------|---------|--------|
| Convergence | Test if entities are converging | β, σ coefficients |
| Sensitivity | Test ranking robustness | sensitivity indices |

### Processing Steps

1. **Convergence Analysis**:
   - Prepare score time-series
   - Run β-convergence regression
   - Calculate σ-convergence trend
   - Identify convergence clubs (if any)

2. **Sensitivity Analysis**:
   - Define ranking function
   - Monte Carlo weight perturbation (1000 simulations)
   - Calculate weight sensitivity indices
   - Find critical weight ranges
   - Compute top-N stability

### Data Flow

```
MCDM Scores (all years)
        │
        ├──▶ Convergence Analysis
        │         │
        │         ├──▶ β coefficient (catch-up)
        │         ├──▶ σ by year (dispersion)
        │         └──▶ convergence clubs
        │
        └──▶ Sensitivity Analysis
                  │
                  ├──▶ weight_sensitivity
                  ├──▶ rank_stability
                  ├──▶ critical_weights
                  └──▶ overall_robustness
```

### Output

```
Beta convergence: -0.0234 (converging: YES)
Sigma trend: -0.0015 (converging: YES)
Overall robustness: 0.87
```

---

## Phase 7: Visualization

### Purpose
Generate high-resolution figures for all analyses.

### Visualizations Generated

| Chart | Description | File |
|-------|-------------|------|
| Score Evolution | Scores over time | `score_evolution.png` |
| Weight Comparison | Entropy vs CRITIC vs Ensemble | `weights_comparison.png` |
| MCDM Comparison | Method rankings comparison | `mcdm_comparison.png` |
| Feature Importance | RF feature importance | `feature_importance.png` |
| Convergence | β and σ convergence plots | `convergence.png` |
| Sensitivity | Weight sensitivity heatmap | `sensitivity.png` |
| Ranking Agreement | Method correlation matrix | `ranking_agreement.png` |

### Configuration

- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG (configurable)
- **Style**: Professional with consistent color scheme

### Output Directory

```
outputs/figures/
├── score_evolution.png
├── weights_comparison.png
├── mcdm_comparison.png
├── feature_importance.png
├── convergence.png
├── sensitivity.png
└── ranking_agreement.png
```

---

## Phase 8: Output Generation

### Purpose
Save all results in organized, accessible formats.

### Output Structure

```
outputs/
├── figures/                     # Visualizations
│   ├── score_evolution.png
│   ├── weights_comparison.png
│   └── ...
│
├── results/                     # Numerical data (CSV)
│   ├── final_rankings.csv      # Main ranking output
│   ├── weights_analysis.csv    # Criterion weights
│   ├── mcdm_scores_detailed.csv
│   ├── feature_importance.csv
│   ├── sensitivity_analysis.csv
│   ├── beta_convergence.csv
│   ├── sigma_convergence.csv
│   └── output_manifest.json    # File index
│
├── reports/                     # Text reports
│   └── analysis_report.txt     # Comprehensive summary
│
└── logs/                        # Execution logs
    └── pipeline.log
```

### Final Rankings CSV Format

```csv
Province,Final_Rank,Final_Score,TOPSIS_Rank,VIKOR_Rank,Fuzzy_Rank,Kendall_W
P12,1,0.8234,1,2,1,0.89
P45,2,0.7891,3,1,2,0.89
P08,3,0.7654,2,3,4,0.89
...
```

### Output Manifest (JSON)

```json
{
  "timestamp": "2024-01-31T10:30:00",
  "execution_time": 45.23,
  "files": {
    "rankings": "results/final_rankings.csv",
    "weights": "results/weights_analysis.csv",
    "figures": ["figures/score_evolution.png", ...],
    "report": "reports/analysis_report.txt"
  },
  "config": {
    "n_provinces": 64,
    "n_years": 5,
    "n_components": 20
  }
}
```

---

## Configuration Options

### Main Configuration (`src/config.py`)

```python
@dataclass
class Config:
    # Panel data
    panel: PanelDataConfig
        n_provinces: int = 64
        n_components: int = 20
        years: List[int] = [2020, 2021, 2022, 2023, 2024]
    
    # TOPSIS
    topsis: TOPSISConfig
        normalization: str = "vector"
        temporal_discount: float = 0.95
        trajectory_weight: float = 0.3
        stability_weight: float = 0.2
    
    # VIKOR
    vikor: VIKORConfig
        v: float = 0.5  # Group utility weight
    
    # Random Forest
    rf: RandomForestConfig
        n_estimators: int = 200
        max_depth: int = 10
        n_splits: int = 3
    
    # LSTM
    lstm: LSTMConfig
        sequence_length: int = 3
        hidden_units: int = 64
        epochs: int = 100
    
    # Sensitivity
    sensitivity: SensitivityConfig
        n_simulations: int = 1000
        perturbation_range: float = 0.2
```

### Quick Configuration via `main.py`

```python
CONFIG = {
    'data_path': 'data/data.csv',  # or None for synthetic
    'n_provinces': 64,
    'n_years': 5,
    'n_components': 20,
    'output_dir': 'outputs',
}
```

---

## Customization Guide

### Adding a New MCDM Method

1. Create new file in `src/mcdm/`:
```python
# src/mcdm/new_method.py
class NewMCDMMethod:
    def calculate(self, data, weights):
        # Implementation
        return NewMCDMResult(scores=scores, ranks=ranks)
```

2. Add to `src/mcdm/__init__.py`:
```python
from .new_method import NewMCDMMethod
```

3. Integrate in `src/main.py`:
```python
def _run_mcdm(self, panel_data, weights):
    # ... existing code ...
    new_method = NewMCDMMethod()
    new_result = new_method.calculate(df, weights_dict)
    results['new_method'] = new_result
```

### Adding a New ML Method

1. Create new file in `src/ml/`:
```python
# src/ml/new_ml.py
class NewMLMethod:
    def fit_predict(self, panel_data, target_col):
        # Implementation
        return NewMLResult(predictions=preds, metrics=metrics)
```

2. Add to `src/ml/__init__.py`

3. Integrate in `_run_ml()` method

### Custom Visualization

Add new visualization to `src/visualization.py`:

```python
def plot_custom_chart(self, data, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Custom plotting logic
    fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
    plt.close(fig)
```

### Custom Output Format

Modify `src/output_manager.py`:

```python
def save_custom_format(self, data, filename):
    # Custom format logic (Excel, JSON, etc.)
    pass
```

---

## Troubleshooting

### Common Issues

1. **Insufficient data for LSTM**:
   - Reduce `sequence_length` or
   - Increase number of time periods

2. **Panel regression fails**:
   - Check for missing values
   - Ensure sufficient variation in target

3. **Convergence not detected**:
   - May need more time periods (≥4)
   - Check if data has actual convergence pattern

4. **High memory usage**:
   - Reduce `n_simulations` in sensitivity analysis
   - Use smaller batch sizes for LSTM

### Logging

Enable debug logging:

```python
import logging
logging.getLogger('ml_topsis').setLevel(logging.DEBUG)
```

Check logs at: `outputs/logs/pipeline.log`

---

## Performance Tips

1. **Parallel Processing**: Monte Carlo simulations can be parallelized
2. **Caching**: Intermediate results are cached during pipeline execution
3. **Batch Processing**: For large datasets, process in entity batches
4. **Memory**: Use `del` to free large intermediate DataFrames

### Typical Execution Times

| Configuration | Time |
|--------------|------|
| 64 entities × 5 years × 20 criteria | ~45 seconds |
| 100 entities × 10 years × 30 criteria | ~3 minutes |
| 200 entities × 20 years × 50 criteria | ~15 minutes |

---

## Summary

The ML-MCDM pipeline provides a comprehensive, automated framework for multi-criteria decision analysis. By following this workflow, users can:

1. **Load** any panel data structure
2. **Weight** criteria objectively
3. **Rank** alternatives using multiple MCDM methods
4. **Validate** with machine learning
5. **Combine** results through ensemble methods
6. **Analyze** robustness and convergence
7. **Visualize** all results professionally
8. **Export** comprehensive outputs

For detailed method descriptions, see [METHODS.md](METHODS.md).
