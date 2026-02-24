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

The ML-MCDM pipeline analyzes panel data (entities × time periods × criteria) using **Intuitionistic Fuzzy Sets (IFS)** combined with **Evidential Reasoning (ER)** for robust multi-criteria ranking with uncertainty quantification.

### Core Methodology

- **12 MCDM Methods**: 6 Traditional + 6 IFS variants (TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, SAW)
- **Two-Stage Aggregation**: Within-criterion ER → Global ER
- **GTWC Weighting**: Game Theory Weight Combination (Entropy + CRITIC + MEREC + SD)
- **Bayesian Bootstrap**: 1000 iterations for weight uncertainty quantification
- **ML Forecasting**: 6-model ensemble + Super Learner + Conformal Prediction
- **Temporal Stability**: Split-half validation for robustness

### Key Features

- **Automated Pipeline**: Single entry point (`main.py`) runs complete analysis
- **Modular Design**: 7 independent phases with clean interfaces
- **State-of-the-Art Forecasting**: UnifiedForecaster with 6 diverse models
- **Robust Error Handling**: Adaptive zero-handling, graceful fallbacks with detailed logging
- **High-Quality Outputs**: 300 DPI figures, comprehensive CSV/JSON results, detailed reports

---

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│            ML-MCDM: IFS + Evidential Reasoning Pipeline                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1: Data Loading                                                  │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ Yearly CSVs → PanelData (63 provinces × 14 years × 29)  │           │
│  └────────────────────────┬─────────────────────────────────┘           │
│                           │                                              │
│  Phase 2: Weight Calculation                                            │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ GTWC: Entropy + CRITIC + MEREC + SD                     │           │
│  │ → Game Theory Combination → Bayesian Bootstrap (1000)    │           │
│  └────────────────────────┬─────────────────────────────────┘           │
│                           │                                              │
│  Phase 3: Hierarchical Ranking (IFS + ER)                               │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ Stage 1: Within-criterion (per 8 criteria)              │           │
│  │   • 12 MCDM methods (6 Traditional + 6 IFS)             │           │
│  │   • ER belief aggregation with adaptive zero-handling   │           │
│  │                                                          │           │
│  │ Stage 2: Global aggregation                             │           │
│  │   • Weighted ER across 8 criterion beliefs              │           │
│  │   • Final ranking with uncertainty (Kendall's W)        │           │
│  └────────────────────────┬─────────────────────────────────┘           │
│                           │                                              │
│  Phase 4: ML Forecasting (State-of-the-Art Ensemble)                   │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ 6 Models: GB + Bayesian + QuantileRF + PanelVAR + NAM  │           │
│  │ → Super Learner meta-ensemble + Conformal Prediction    │           │
│  │ → Aggregated feature importance across all models       │           │
│  └────────────────────────┬─────────────────────────────────┘           │
│                           │                                              │
│  Phase 5: Sensitivity Analysis                                          │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ Hierarchical multi-level sensitivity analysis:           │           │
│  │ • Subcriteria weight perturbation (±15%)                │           │
│  │ • Criteria weight perturbation (±15%)                   │           │
│  │ • IFS uncertainty (μ/ν ±10%)                            │           │
│  │ • Temporal stability (14-year correlation)              │           │
│  │ • Monte Carlo simulation (100+ iterations)              │           │
│  │ → Overall robustness score (0-1)                        │           │
│  └────────────────────────┬─────────────────────────────────┘           │
│                           │                                              │
│  Phase 6: Visualization                                                 │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ 5 high-resolution figures (300 DPI)                      │           │
│  │ • Ranking summary • Score distribution • Weights         │           │
│  │ • Sensitivity heatmap • Feature importance               │           │
│  └────────────────────────┬─────────────────────────────────┘           │
│                           │                                              │
│  Phase 7: Result Export                                                 │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ 14 output files:                                         │           │
│  │ • 9 CSV results (rankings, weights, scores, analysis)    │           │
│  │ • 3 JSON metadata (execution, config, manifest)          │           │
│  │ • 1 TXT report (comprehensive summary)                   │           │
│  │ • 1 debug.log (detailed execution trace)                │           │
│  └──────────────────────────────────────────────────────────┘           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Pipeline Phases

### Phase 1: Data Loading

**Purpose:** Load and structure panel data from yearly CSV files.

| Input | Output |
|-------|--------|
| Yearly CSV files (`data/2011.csv` ... `data/2024.csv`) | `PanelData` object with long, wide, and cross-section views |

**Data Structure:**
```
PanelData
├── long: DataFrame         # Long format (Year, Province, C01...C29)
├── wide: Dict[year, DataFrame]  # Province × Subcriteria per year
├── cross_section: Dict[year, DataFrame]
├── provinces: List[str]    # Entity identifiers (63 provinces + city)
├── years: List[int]        # Time periods (2011-2024, 14 years)
├── criteria: List[str]     # Criterion groups (C01-C08, 8 criteria)
└── subcriteria: List[str]  # Subcriteria names (29 subcriteria)
```

**Processing:**
- Loads each year's CSV independently
- Validates province codes (63 entities including cities)
- Maps 29 subcriteria to 8 hierarchical criteria using codebook
- Handles missing values via forward-fill
- Constructs long-format panel for temporal analysis

**Validation:**
- Shape: (63 provinces × 14 years × 29 subcriteria) = 24696 observations
- All values normalized to [0, 1] range
- No duplicate province-year combinations

---

### Phase 2: GTWC Weight Calculation

**Purpose:** Calculate objective criterion weights using Game Theory Weight Combination.

**Four Base Methods:**

| Method | Description | Formula |
|--------|-------------|--------|
| **Entropy** | Information content | $w_j = \frac{1 - E_j}{\sum_k (1 - E_k)}$ |
| **CRITIC** | Contrast + correlation | $w_j = \frac{\sigma_j \sum_k (1 - r_{jk})}{\sum_m \sigma_m \sum_k (1 - r_{mk})}$ |
| **MEREC** | Removal effects | $w_j = \frac{\text{Impact}_j}{\sum_k \text{Impact}_k}$ |
| **Std Dev** | Dispersion | $w_j = \frac{\sigma_j}{\sum_k \sigma_k}$ |

**Game Theory Combination:**

1. **Intra-Group Hybridization:**
   - Group A (Dispersion): Geometric mean of Entropy + Std Dev
   - Group B (Interaction): Harmonic mean of CRITIC + MEREC

2. **Cooperative Game Optimization:**
   $$
   \min L = \|\alpha_1 W_A + \alpha_2 W_B - W_A\|^2 + \|\alpha_1 W_A + \alpha_2 W_B - W_B\|^2
   $$

3. **Final Fusion:**
   $$
   W^* = \alpha_1 \cdot W_{\text{GroupA}} + \alpha_2 \cdot W_{\text{GroupB}}
   $$

**Bayesian Bootstrap (1000 iterations):**
- Uncertainty quantification via Dirichlet resampling
- 95% confidence intervals for each criterion weight
- Cosine similarity validation (should be > 0.95)

**Temporal Stability Check:**
- Split-half validation (first 7 vs last 7 years)
- Cosine similarity threshold: 0.85
- Flags unstable weights for review

**Output Files:**
- `criterion_weights.csv`: Mean weights ± bootstrap std
- `weights_analysis.csv`: Full 4-method breakdown + fusion coefficients

---

### Phase 3: Hierarchical Ranking (IFS + ER)

**Purpose:** Two-stage aggregation using Intuitionistic Fuzzy Sets and Evidential Reasoning.

#### 12 MCDM Methods (6 Traditional + 6 IFS)

| Method | Type | Key Innovation |
|--------|------|----------------|
| **TOPSIS** | Traditional | Ideal/anti-ideal distance |
| **VIKOR** | Traditional | Compromise solution |
| **PROMETHEE** | Traditional | Pairwise outranking |
| **COPRAS** | Traditional | Stepwise comparison |
| **EDAS** | Traditional | Distance from average |
| **SAW** | Traditional | Weighted sum |
| **IFS-TOPSIS** | Uncertainty | IFN distance measures |
| **IFS-VIKOR** | Uncertainty | IFN compromise |
| **IFS-PROMETHEE** | Uncertainty | IFN preference flows |
| **IFS-COPRAS** | Uncertainty | IFN weighted sums |
| **IFS-EDAS** | Uncertainty | IFN distance deviation |
| **IFS-SAW** | Uncertainty | IFN aggregation |

#### Stage 1: Within-Criterion Aggregation

For **each of 8 criteria** (C01 through C08):

1. **Run 12 MCDM methods** on subcriteria scores
2. **Adaptive Zero-Handling:**
   - Identify zero/missing values
   - Temporarily exclude from ranking
   - Restore after computation (assign worst rank)
3. **Normalize scores** to [0, 1]
4. **Construct IFS belief structure:**
   - Convert 12 method scores → 5-grade belief distribution
   - Grades: {Excellent, Good, Fair, Poor, Bad}
5. **ER combination** → single criterion belief per entity

#### Stage 2: Global Aggregation

1. **Inputs:** 8 criterion beliefs (one per C01-C08) + GTWC weights
2. **Weighted ER aggregation** using Yang & Xu (2002) algorithm:
   $$
   \beta_n = K \left[\beta_{1,n}\beta_{2,n} + \beta_{1,n}\beta_{2,H} + \beta_{1,H}\beta_{2,n}\right]
   $$
   Where K is normalization constant handling belief conflicts
3. **Utility calculation** from final belief distribution
4. **Final ranking** by utility scores (descending)

#### Validation

- **Kendall's W concordance coefficient** across 12 methods
- Expected: W > 0.7 (strong agreement)
- Actual: W ≈ 0.88 (very strong agreement)

**Output Files:**
- `final_rankings.csv`: Final ranks + ER utility scores
- `mcdm_scores_C01.csv` ... `mcdm_scores_C08.csv`: Per-criterion method scores (8 files)
- `mcdm_rank_comparison.csv`: Rank comparison across all 12 methods
- `prediction_uncertainty_er.csv`: Hesitancy degrees (π) per entity

---

### Phase 4: ML Forecasting

**Purpose:** Forecast future criterion values using state-of-the-art ensemble learning. Feature importance is computed as a by-product.

**Ensemble Architecture:**
- **6 Base Models:**
  1. **Gradient Boosting** — Tree-based with Huber loss
  2. **Bayesian Ridge** — Probabilistic linear model
  3. **Quantile Random Forest** — Distributional forecasting
  4. **Panel VAR** — Panel-specific dynamics
  5. **Hierarchical Bayesian** — Partial pooling across entities
  6. **Neural Additive Models (NAM)** — Interpretable non-linearity

- **Meta-Ensemble:** Super Learner (Ridge regression for optimal model weighting)
- **Uncertainty Quantification:** Conformal Prediction (distribution-free 95% intervals)

**Cross-Validation:**
- **Method:** TimeSeriesSplit (3 folds)
- **Respects temporal ordering** (no data leakage)
- **Metrics:** R², MAE, RMSE per fold per model
- **Expected performance:** Ensemble CV R² > 0.70

**Feature Importance (By-Product):**
- Each model computes its own feature importance:
  - GB: Gini importance
  - Bayesian: Absolute coefficients
  - Panel VAR: Coefficient magnitudes
  - NAM: Shape function variances
- **Aggregated** across all 6 models (averaged)
- Saved for exploratory analysis

**Output Files:**
- `forecast_predictions.csv`: Predicted values for target year
- `forecast_prediction_intervals.csv`: 95% confidence intervals
- `forecast_feature_importance.csv`: Aggregated importance (by-product)
- `forecast_model_weights.csv`: Super Learner optimal weights
- `forecast_cv_metrics.csv`: Cross-validation performance per model

---

### Phase 5: Sensitivity Analysis

**Purpose:** Assess robustness of rankings through hierarchical multi-level perturbation analysis.

---


### Phase 6: Visualization

**Purpose:** Generate high-resolution figures (300 DPI).

**Generated Figures:**
1. `01_final_ranking_summary.png` — Top 20 provinces with ER utility
2. `02_score_distribution.png` — Histogram + KDE
3. `03_weights_comparison.png` — GTWC weights (8 criteria)
4. `04_sensitivity_analysis.png` — Rank stability heatmap
5. `05_forecast_feature_importance.png` — Aggregated feature importance (if forecasting enabled)

**Output:** 5 PNG files in `result/figures/`

---

### Phase 7: Result Export

**Purpose:** Save all results in organized structure.

See [Output Structure](#4-output-structure) below.

---

## 4. Output Structure

```
result/
├── figures/                          # High-resolution visualizations (5 PNG files)
│   ├── final_ranking_summary.png      # Top 20 provinces with ER utility
│   ├── score_distribution.png         # Histogram + KDE
│   ├── weights_comparison.png         # GTWC weights (8 criteria)
│   ├── sensitivity_analysis.png       # Rank stability heatmap
│   └── forecast_feature_importance.png # Aggregated from 6 models (optional)
│
├── results/                          # Numerical data (14 files)
│   ├── final_rankings.csv             # Main output: rank + ER utility + province
│   │
│   ├── criterion_weights.csv          # GTWC weights with bootstrap CI
│   ├── weights_analysis.csv           # 4-method breakdown + fusion details
│   │
│   ├── mcdm_scores_C01.csv            # 12-method scores for C_01
│   ├── mcdm_scores_C02.csv            # ...
│   ├── mcdm_scores_C03.csv
│   ├── mcdm_scores_C04.csv
│   ├── mcdm_scores_C05.csv
│   ├── mcdm_scores_C06.csv
│   ├── mcdm_scores_C07.csv
│   ├── mcdm_scores_C08.csv            # 12-method scores for C_08
│   ├── mcdm_rank_comparison.csv       # Rank comparison across all methods
│   │
│   ├── feature_importance.csv         # Aggregated from 6 forecast models (if enabled)
│   │
│   ├── sensitivity_subcriteria.csv    # Subcriteria sensitivity (29 scores)
│   ├── sensitivity_criteria.csv       # Criteria sensitivity (8 scores)
│   ├── temporal_stability.csv         # Year-to-year correlations
│   ├── top_n_stability.csv            # Top-N ranking stability
│   ├── ifs_sensitivity.csv            # IFS μ/ν uncertainty
│   ├── robustness_summary.csv         # Overall robustness + confidence
│   ├── prediction_uncertainty_er.csv  # IFS hesitancy degrees (π)
│   │
│   ├── data_summary_statistics.csv    # Descriptive stats
│   ├── execution_summary.json         # Phase timings + metadata
│   └── config_snapshot.json           # Full configuration (reproducibility)
│
├── reports/
│   └── report.txt                     # Comprehensive text report
│
└── logs/
    └── debug.log                      # Detailed execution trace (DEBUG level)
```

**Total Output:** 5 figures + 17 data files + 1 report + 1 log = **24 files**

---

## 5. Configuration

### Running the Pipeline

**Command Line:**

```bash
python main.py                # Run with default configuration
```

**Python API:**

```python
from ml_mcdm.pipeline import MLMCDMPipeline
from ml_mcdm.config import get_default_config

config = get_default_config()

pipeline = MLMCDMPipeline(config)
result = pipeline.run()

# Access results
rankings = result.ranking.final_er_scores
print(f"Kendall's W: {result.ranking.kendall_w:.4f}")
print(f"Robustness: {result.analysis['sensitivity'].overall_robustness:.4f}")
```

### Key Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bootstrap_iterations` | 1000 | Bayesian bootstrap for weight uncertainty |
| `cv_folds` | 3 | Time-series CV folds for forecasting |
| `n_simulations` | 1000 | Monte Carlo sensitivity simulations |
| `random_state` | 42 | Reproducibility seed |
| `n_splits` | 5 | Time-series CV folds |
| `output_dir` | `result` | Result directory |
| `dpi` | 300 | Figure resolution |

### Performance

**Expected Runtime:** ~60-90 seconds on modern CPU

| Component | Time | Notes |
|-----------|------|-------|
| Weighting | ~90s | Bootstrap (1000 iterations) |
| Ranking | ~10s | 12 MCDM + 2-stage ER |
| ML Forecasting | ~15s | 6 models + Super Learner |
| Sensitivity Analysis | ~20s | Monte Carlo (1000 sims) |
| Visualization + Export | ~5s | 5 figures + results |

---

## 6. Logging

### Console Output

- **Level:** INFO
- **Format:** Simple text (no colors)
- **Content:** Phase progress, key metrics

### Debug Log File

- **Location:** `result/logs/debug.log`
- **Level:** DEBUG (captures everything)
- **Format:** `timestamp | level | module | function:line | message`

### Example Console Output

```
======================================================================
  ML-MCDM: IFS + Evidential Reasoning Hierarchical Ranking
======================================================================
  Provinces         : 63
  Years             : 2011-2024 (14 years)
  Subcriteria       : 29
  Criteria          : 8
  MCDM methods      : 12 (6 traditional + 6 IFS)
  Bootstrap iters   : 1000
  Sensitivity sims  : 1000
  Output            : result/
======================================================================

▶ Phase 1/7: Data Loading
  Loaded 63 provinces × 14 years × 29 subcriteria
  ✓ Completed in 0.91s

▶ Phase 2/7: Weight Calculation
  GTWC: Entropy + CRITIC + MEREC + SD with 1000 bootstrap
  Weights: [0.142, 0.118, 0.095, 0.158, 0.127, 0.109, 0.132, 0.119]
  Cosine similarity: 0.9915 (stable)
  ✓ Completed in 19.51s

▶ Phase 3/7: Hierarchical Ranking
  Stage 1: 12 MCDM × 8 criteria with adaptive zero-handling
  Stage 2: Weighted ER aggregation
  Kendall's W: 0.8786 (strong agreement)
  Top-ranked: P02 (utility = 0.8547)
  ✓ Completed in 7.02s

▶ Phase 4/7: ML Forecasting
  Ensemble: 6 models (GB, Bayesian, QRF, PanelVAR, HierBayes, NAM)
  Super Learner CV R²: 0.7355 ± 0.084
  Top feature: SC01 (importance = 0.082)
  ✓ Completed in 14.50s

▶ Phase 5/7: Sensitivity Analysis
  Hierarchical robustness testing (100+ simulations)
  Robustness: 0.9772 (HIGH confidence)
  Top-5 stability: 95.2%
  Temporal correlation: 0.94
  ✓ Completed in 4.50s

▶ Phase 6/7: Visualization
  Generated 5 figures (300 DPI)
  ✓ Completed in 3.50s

▶ Phase 7/7: Result Export
  Saved 14 output files
  ✓ Completed in 0.35s

======================================================================
Pipeline completed successfully in 32.82s
Outputs: result/
======================================================================
```

---

## Summary

The ML-MCDM pipeline provides:

1. **Rigorous Methodology**: IFS + two-stage ER with adaptive zero-handling
2. **Objective Weighting**: GTWC (4 methods) with Bayesian Bootstrap uncertainty
3. **Multi-Method Consensus**: 12 MCDM methods (6 traditional + 6 IFS)
4. **ML Forecasting**: 6-model ensemble (optional) with Super Learner
5. **Hierarchical Sensitivity**: Multi-level robustness analysis (1000 simulations)
6. **Comprehensive Outputs**: 5 high-resolution figures + 17+ data files + detailed report
7. **Production-Ready**: Single-command execution with scientifically validated defaults

**Key Metrics (typical values):**
- Kendall's W: 0.85-0.90 (strong method agreement)
- CV R²: 0.70-0.80 (good predictive power)
- Overall Robustness: 0.90-0.98 (very stable rankings)
- Weight Stability: > 0.95 (bootstrap cosine similarity)

For methodology details, see:
- [ranking.md](ranking.md) — IFS + ER hierarchical ranking
- [weighting.md](weighting.md) — GTWC weight calculation
- [objective.md](objective.md) — project objectives
