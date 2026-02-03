# ML-MCDM Workflow Guide

This document provides a step-by-step description of the ML-MCDM analysis pipeline workflow.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Pipeline Phases](#3-pipeline-phases)
4. [Output Structure](#4-output-structure)
5. [Configuration](#5-configuration)
6. [Logging](#6-logging)

---

## 1. Overview

The ML-MCDM pipeline analyzes panel data (entities × time periods × criteria) using a combination of Multi-Criteria Decision Making (MCDM) methods and Machine Learning techniques.

### Key Features

- **Automated Pipeline**: Single entry point runs complete analysis
- **Modular Design**: Each phase is independent and configurable
- **Robust Error Handling**: Graceful fallbacks with detailed logging
- **High-Quality Outputs**: 300 DPI figures, comprehensive CSV results, detailed reports

---

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ML-MCDM Analysis Pipeline                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1        Phase 2        Phase 3        Phase 4                   │
│  ┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐                    │
│  │ Data │ ───► │Weight│ ───► │ MCDM │ ───► │  ML  │                    │
│  │ Load │      │ Calc │      │      │      │      │                    │
│  └──────┘      └──────┘      └──────┘      └──────┘                    │
│                                               │                          │
│  Phase 8        Phase 7        Phase 6        Phase 5                   │
│  ┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐                    │
│  │Output│ ◄─── │Visual│ ◄─── │Analys│ ◄─── │Ensem │ ◄──────────────────┘
│  │ Save │      │      │      │      │      │ ble  │                    │
│  └──────┘      └──────┘      └──────┘      └──────┘                    │
│      │                                                                   │
│      ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ outputs/                                                         │    │
│  │   ├── figures/    (22 high-resolution PNG charts)               │    │
│  │   ├── results/    (CSV data files + JSON metadata)              │    │
│  │   ├── reports/    (comprehensive analysis report)               │    │
│  │   └── logs/       (debug.log with detailed execution log)       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Pipeline Phases

### Phase 1: Data Loading

**Purpose:** Load and structure panel data for analysis.

| Input | Output |
|-------|--------|
| CSV file (`data/data.csv`) or synthetic generation | `PanelData` object with long, wide, and cross-section views |

**Data Structure:**
```
PanelData
├── long: DataFrame         # Long format (Year, Province, C01...C20)
├── wide: Dict[year, DataFrame]  # Province × Components per year
├── cross_section: Dict[year, DataFrame]
├── provinces: List[str]    # Entity identifiers (64 provinces)
├── years: List[int]        # Time periods (2020-2024)
└── components: List[str]   # Criteria names (C01-C20)
```

---

### Phase 2: Weight Calculation

**Purpose:** Calculate objective criteria weights using data-driven methods.

| Method | Description |
|--------|-------------|
| Entropy | Based on information content |
| CRITIC | Based on contrast + correlation |
| Ensemble | Geometric mean combination |

**Output:** `weights_analysis.csv`

---

### Phase 3: MCDM Analysis

**Purpose:** Calculate rankings using multiple MCDM methods.

| Method | Type | Output |
|--------|------|--------|
| TOPSIS | Traditional | Scores, Rankings |
| Dynamic TOPSIS | Temporal-aware | Dynamic scores |
| VIKOR | Traditional | Q, S, R values |
| Fuzzy TOPSIS | Uncertainty-aware | Fuzzy scores |

**Output:** `mcdm_scores_detailed.csv`, `final_rankings.csv`

---

### Phase 4: ML Analysis

**Purpose:** Feature importance analysis and validation using Random Forest.

| Component | Description |
|-----------|-------------|
| Random Forest TS | Time-series cross-validation |
| Feature Importance | Component contribution analysis |
| CV Scores | Model validation metrics |

**Output:** `feature_importance.csv`, `cv_scores.csv`, `rf_test_metrics.csv`

---

### Phase 5: Ensemble Integration

**Purpose:** Combine results from multiple methods into final ranking.

| Method | Description |
|--------|-------------|
| Stacking | Meta-learner combining MCDM predictions |
| Borda Count | Point-based rank aggregation |
| Kendall's W | Agreement measure |

**Output:** `stacking_weights.csv`, `aggregation_metadata.json`

---

### Phase 6: Advanced Analysis

**Purpose:** Assess robustness and temporal dynamics.

| Analysis | Description |
|----------|-------------|
| Sigma Convergence | Dispersion trend over time |
| Beta Convergence | Catch-up speed analysis |
| Sensitivity | Weight perturbation impact |

**Output:** `sigma_convergence.csv`, `beta_convergence.csv`, `sensitivity_analysis.csv`

---

### Phase 6.5: Future Prediction

**Purpose:** Predict next year (2025) rankings using ML forecasting.

| Component | Description |
|-----------|-------------|
| Predicted Components | Forecasted criterion values for 2025 |
| Predicted Rankings | TOPSIS/VIKOR rankings for 2025 |
| Uncertainty | Prediction confidence intervals |

**Output:** `predicted_rankings_2025.csv`, `predicted_components_2025.csv`, `prediction_uncertainty_2025.csv`

---

### Phase 7: Visualization

**Purpose:** Generate high-resolution figures (300 DPI).

| Category | Figures |
|----------|---------|
| Score Evolution | Top/bottom performers over time (01-02) |
| Weights | Comparison chart (03) |
| MCDM Results | TOPSIS, VIKOR, method agreement (04-07) |
| ML Analysis | Feature importance, CV progression (08, 16-22) |
| Analysis | Sensitivity, final ranking (09-12) |
| Predictions | Future predictions comparison (13) |

**Output:** 22 PNG files in `outputs/figures/`

---

### Phase 8: Output Generation

**Purpose:** Save all results in organized structure.

See [Output Structure](#4-output-structure) below.

---

## 4. Output Structure

```
outputs/
├── figures/                          # High-resolution visualizations
│   ├── 01_score_evolution_top.png
│   ├── 02_score_evolution_bottom.png
│   ├── 03_weights_comparison.png
│   ├── 04_topsis_scores.png
│   ├── 05_vikor_analysis.png
│   ├── 06_method_agreement.png
│   ├── 07_score_distribution.png
│   ├── 08_feature_importance.png
│   ├── 09_sensitivity_analysis.png
│   ├── 10_final_ranking.png
│   ├── 11_method_comparison.png
│   ├── 12_ensemble_weights.png
│   ├── 13_future_predictions.png
│   ├── 16_rf_feature_importance_detailed.png
│   ├── 17_rf_cv_progression.png
│   ├── 18_rf_actual_vs_predicted.png
│   ├── 19_rf_residual_analysis.png
│   ├── 20_rf_rank_correlation.png
│   ├── 21_ensemble_contribution.png
│   ├── 22_rf_model_performance.png
│   └── figure_manifest.json
│
├── results/                          # Numerical data
│   ├── final_rankings.csv            # Main ranking output
│   ├── weights_analysis.csv          # Criterion weights
│   ├── mcdm_scores_detailed.csv      # All MCDM scores
│   ├── feature_importance.csv        # RF feature importance
│   ├── cv_scores.csv                 # Cross-validation scores
│   ├── rf_test_metrics.csv           # RF test performance
│   ├── stacking_weights.csv          # Ensemble weights
│   ├── sensitivity_analysis.csv      # Sensitivity indices
│   ├── sigma_convergence.csv         # σ-convergence by year
│   ├── beta_convergence.csv          # β-convergence results
│   ├── predicted_rankings_2025.csv   # Future predictions
│   ├── predicted_components_2025.csv # Predicted values
│   ├── prediction_uncertainty_2025.csv
│   ├── forecast_summary_2025.json
│   ├── config_snapshot.json          # Run configuration
│   ├── execution_summary.json        # Timing info
│   └── output_manifest.json          # File index
│
├── reports/
│   └── analysis_report.txt           # Comprehensive text report
│
└── logs/
    └── debug.log                     # Detailed execution log
```

---

## 5. Configuration

### Running the Pipeline

```python
from src.pipeline import MLMCDMPipeline
from src.config import get_default_config

# With default config
pipeline = MLMCDMPipeline()
result = pipeline.run('data/data.csv')

# With custom config
config = get_default_config()
config.output_dir = 'custom_outputs'
pipeline = MLMCDMPipeline(config)
result = pipeline.run()
```

### Command Line

```bash
python run.py
```

### Key Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `output_dir` | `outputs` | Output directory |
| `n_estimators` | 200 | RF trees |
| `n_splits` | 5 | CV folds |
| `random_state` | 42 | Reproducibility |

---

## 6. Logging

### Console Output

- **Level:** INFO
- **Format:** Simple text (no colors)
- **Content:** Phase progress, key metrics

### Debug Log File

- **Location:** `outputs/logs/debug.log`
- **Level:** DEBUG (captures everything)
- **Format:** `timestamp | level | module | function:line | message`

### Example Console Output

```
12:30:45 | INFO     | ▶ Starting: Phase 1: Data Loading
12:30:46 | INFO     | ✓ Completed: Phase 1: Data Loading (0.82s)
12:30:46 | INFO     | ▶ Starting: Phase 2: Weight Calculation
12:30:46 | INFO     | ✓ Completed: Phase 2: Weight Calculation (0.15s)
...
12:31:30 | INFO     | Pipeline completed in 45.23 seconds
12:31:30 | INFO     | Outputs saved to: outputs
```

---

## Summary

The ML-MCDM pipeline provides:

1. **Automated Analysis**: Single command runs complete workflow
2. **Multiple Methods**: MCDM (5 traditional + 5 fuzzy) + ML (Random Forest)
3. **Future Predictions**: 2025 rankings with uncertainty
4. **Comprehensive Outputs**: Figures, CSVs, reports, logs
5. **Robust Design**: Error handling with detailed logging

For method details, see [METHODS.md](METHODS.md).
