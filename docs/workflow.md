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

The ML-MCDM pipeline analyzes panel data (entities × time periods × criteria) using a hierarchical multi-criteria ranking approach for robust performance assessment.

### Core Methodology

- **5 Traditional MCDM Methods**: TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS
- **Raw Sum Baseline**: Transparent additive baseline (Stage 1 and 2)
- **Hierarchical Adaptive CRITIC**: Two-level weighting with year-regime analysis
- **ML Forecasting**: 4-model ensemble (CatBoost, Bayesian Ridge, SVR, ElasticNet) + Super Learner + Conformal Prediction
- **Quality Verification**: Temporal stability and sensitivity analysis for robustness

### Key Features

- **Automated Pipeline**: Single entry point (`main.py`) runs complete analysis
- **Modular Design**: 7 independent phases with clean interfaces
- **State-of-the-Art Forecasting**: UnifiedForecaster with 4 diverse models (CatBoost, Bayesian Ridge, SVR, ElasticNet) + Super Learner + Conformal Prediction
- **Robust Error Handling**: Adaptive zero-handling, graceful fallbacks with detailed logging
- **High-Quality Outputs**: 300 DPI figures, comprehensive CSV/JSON results, detailed reports

---

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│            ML-MCDM: Hierarchical Ranking Pipeline                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1: Data Loading                                                  │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ Yearly CSVs → PanelData (63 provinces × 14 years × 29)  │           │
│  └────────────────────────┬─────────────────────────────────┘           │
│                           │                                              │
│  Phase 2: Weight Calculation                                            │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ CRITIC Two-Level: Deterministic two-level CRITIC pipeline  │           │
│  └────────────────────────┬─────────────────────────────────┘           │
│                           │                                              │
│  Phase 3: Hierarchical Ranking                                          │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ Stage 1: Within-criterion (per 8 criteria)              │           │
│  │   • 5 MCDM methods (TOPSIS, VIKOR, PROMETHEE, COPRAS,   │           │
│  │     EDAS) + Raw Sum Baseline                            │           │
│  │                                                          │           │
│  │ Stage 2: Global aggregation                             │           │
│  │   • Weighted aggregation across 8 criteria               │           │
│  │   • Final ranking with concordance (Kendall's W)        │           │
│  └────────────────────────┬─────────────────────────────────┘           │
│                           │                                              │
│  Phase 4: ML Forecasting (TIER 3 Architecture)                         │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ 4 Models: CatBoost + Bayesian + SVR + ElasticNet        │           │
│  │ Impute panel → Super Learner meta-ensemble              │           │
│  │ → Conformal Prediction intervals                        │           │
│  └───────────────────────┬──────────────────────────────────┘           │
│                           │                                              │
│  Phase 5: Sensitivity Analysis                                          │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ Hierarchical multi-level sensitivity analysis:           │           │
│  │ • Subcriteria weight perturbation (±15%)                │           │
│  │ • Criteria weight perturbation (±15%)                   │           │
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

### Phase 2: CRITIC Two-Level Weight Calculation

**Purpose:** Calculate objective criterion weights using a deterministic two-level CRITIC pipeline.

**Method:**

| Method | Description | Formula |
|--------|-------------|--------|
| **CRITIC** | Contrast + correlation | $w_j = \frac{\sigma_j \sum_k (1 - r_{jk})}{\sum_m \sigma_m \sum_k (1 - r_{mk})}$ |

**Two-Level Procedure:**

1. **Level 1** — CRITIC per criterion group → local SC weights (sum to 1 within each group)
2. **Level 2** — CRITIC on criterion composite matrix → criterion weights (sum to 1 globally)
3. **Global SC weights** — $w_j = u_{k,j} \times v_k$ (Level 1 × Level 2 product)

**Temporal Stability Check:**
- Split-half validation (first 7 vs last 7 years)
- Cosine similarity threshold: 0.95
- Flags unstable weights for review

**Output Files:**
- `criterion_weights.csv`: CRITIC criterion-level weights
- `weights_analysis.csv`: Global + local weights per subcriteria

---

### Phase 3: Hierarchical Ranking

**Purpose:** Two-stage aggregation using hierarchical MCDM.

#### 5 Traditional MCDM Methods + Baseline

| Method | Key Innovation |
|--------|----------------|
| **TOPSIS** | Ideal/anti-ideal distance |
| **VIKOR** | Compromise solution |
| **PROMETHEE** | Pairwise outranking |
| **COPRAS** | Stepwise comparison |
| **EDAS** | Distance from average |
| **Base** | Raw sum baseline |

#### Stage 1: Within-Criterion Aggregation

For **each of 8 criteria** (C01 through C08):

1. **Run 6 MCDM methods** on subcriteria scores
2. **Adaptive Zero-Handling:**
   - Identify zero/missing values
   - Temporarily exclude from ranking
   - Restore after computation (assign worst rank)
3. **Normalize scores** to [0, 1]
4. **Aggregate scores** to produce criterion-level performance metrics.

#### Stage 2: Global Aggregation

1. **Inputs:** 8 criterion scores + weights
2. **Weighted aggregation** to combine 8 criteria into a global score.
3. **Final ranking** by global scores (descending)

#### Validation

- **Kendall's W concordance coefficient** across 6 methods
- Expected: W > 0.7 (strong agreement)
- Actual: W ≈ 0.88 (very strong agreement)

**Output Files:**
- `final_rankings.csv`: Final ranks + aggregate scores
- `mcdm_scores_C01.csv` ... `mcdm_scores_C08.csv`: Per-criterion method scores (8 files)
- `mcdm_rank_comparison.csv`: Rank comparison across all 6 methods

---

### Phase 4: ML Forecasting

**Purpose:** Forecast future criterion values using state-of-the-art ensemble learning. Feature importance is computed as a by-product.

**Pre-processing:** A fully ML-imputed copy of the panel (`build_ml_panel_data` — 3-stage: linear interpolation → ffill/bfill → median) is passed to the forecaster. The raw `panel_data` used by MCDM phases is never mutated.

**Ensemble Architecture (TIER 3):**
- **4 Core Base Models:**
  1. **CatBoost Gradient Boosting** — Joint multi-output tree-based model (MultiRMSE loss)
  2. **Bayesian Ridge** — Probabilistic linear model (PLS-compressed features)
  3. **Support Vector Regression** — ε-insensitive tube, RBF kernel
  4. **ElasticNet** — Regularized linear model (L1+L2)

- **Meta-Ensemble:** Super Learner (per-output meta-weights via `PanelWalkForwardCV`)
- **Uncertainty Quantification:** Conformal Prediction (distribution-free intervals, choice of Method)

**Cross-Validation:**
- **Method:** `PanelWalkForwardCV` — panel-aware walk-forward splitter (annual folds)
- **`min_train_years=5`**: first fold trains on the earliest 5 annual cohorts, validates on the next year
- **`cv_folds=5`** (default): up to 5 walk-forward folds
- **Respects temporal ordering** (no data leakage)

**Feature Importance (By-Product):**
- Each model computes its own feature importance:
  - GB: Gini importance
  - Bayesian: Absolute coefficients
- **Aggregated** across all 5 models (averaged)
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
1. `01_final_ranking_summary.png` — Top 20 provinces with final rankings
2. `02_score_distribution.png` — Histogram + KDE
3. `03_weights_comparison.png` — Ensemble weights (8 criteria)
4. `04_sensitivity_analysis.png` — Rank stability heatmap
5. `05_forecast_feature_importance.png` — Aggregated feature importance (if forecasting enabled)

**Output:** PNG files in `output/result/figures/<phase>/`

---

### Phase 7: Result Export

**Purpose:** Save all results in organized structure.

See [Output Structure](#4-output-structure) below.

---

## 4. Output Structure

```
output/result/
├── figures/                          # High-resolution visualizations, split by phase
│   ├── ranking/
│   │   ├── fig01_final_ranking.png
│   │   └── fig02_score_distribution.png
│   ├── weighting/
│   │   ├── fig03_weights_comparison.png
│   │   ├── fig04_weight_radar.png
│   │   └── fig05_weight_heatmap.png
│   ├── mcdm/
│   │   ├── fig06_method_agreement.png
│   │   ├── fig07_rank_parallel.png
│   │   └── fig08_<C##>_scores.png    # one per criterion
│   ├── sensitivity/
│   │   ├── fig09_criteria_sensitivity.png
│   │   ├── fig10_subcriteria_sensitivity.png
│   │   ├── fig11_top_n_stability.png
│   │   ├── fig12_temporal_stability.png
│   │   ├── fig13_rank_volatility.png
│   │   └── fig25_robustness_summary.png
│   ├── forecasting/
│   │   ├── fig18_feature_importance.png
│   │   ├── fig19_model_weights.png
│   │   ├── fig20_model_performance.png
│   │   ├── fig21_cv_boxplots.png
│   │   ├── fig22_prediction_intervals.png
│   │   └── fig23_rank_change_bubble.png
│   └── summary/
│       └── fig24_executive_dashboard.png
│
├── csv/                              # Numerical data, split by phase
│   ├── weighting/
│   │   ├── weights_analysis.csv       # Global + local weights per subcriteria
│   │   └── criterion_weights.csv      # Ensemble criterion-level weights
│   ├── ranking/
│   │   └── final_rankings.csv         # Main output: rank + score + province
│   ├── mcdm/
│   │   ├── mcdm_scores_C01.csv        # 6-method scores for C_01
│   │   ├── ...                        # one file per criterion
│   │   ├── mcdm_scores_C08.csv
│   │   └── mcdm_rank_comparison.csv   # Cross-method rank comparison matrix
│   ├── forecasting/
│   │   ├── forecast_predictions.csv
│   │   ├── forecast_summary.json
│   │   ├── model_contributions.csv
│   │   ├── model_performance.csv
│   │   ├── feature_importance.csv
│   │   ├── cross_validation_scores.csv
│   │   └── prediction_intervals.csv
│   ├── sensitivity/
│   │   ├── sensitivity_criteria.csv
│   │   ├── sensitivity_subcriteria.csv
│   │   ├── sensitivity_rank_stability.csv
│   │   ├── sensitivity_top_n_stability.csv
│   │   ├── sensitivity_temporal.csv
│   │   └── sensitivity_summary.json
│   └── summary/
│       ├── data_summary_statistics.csv
│       ├── execution_summary.json
│       └── config_snapshot.json
│
├── reports/
│   └── report.md                      # Comprehensive Markdown report
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
rankings = result.ranking.final_scores
print(f"Kendall's W: {result.ranking.kendall_w:.4f}")
print(f"Robustness: {result.analysis['sensitivity'].overall_robustness:.4f}")
```

### Key Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stability_threshold` | 0.95 | Minimum cosine similarity for temporal stability pass |
| `cv_folds` | 5 | Walk-forward CV folds for forecasting |
| `cv_min_train_years` | 5 | Minimum annual training cohorts before first val fold |
| `n_simulations` | 1000 | Monte Carlo sensitivity simulations |
| `random_state` | 42 | Reproducibility seed |
| `n_splits` | 5 | Time-series CV folds |
| `output_dir` | `result` | Result directory |
| `dpi` | 300 | Figure resolution |

### Performance

**Expected Runtime:** ~60-90 seconds on modern CPU

| Component | Time | Notes |
|-----------|------|-------|
| Weighting | ~2s | Deterministic CRITIC (two levels) |
| Ranking | ~10s | 6 MCDM + 2-stage ER |
| ML Forecasting | ~12-25s | 5 models + Super Learner + Conformal |
| Sensitivity Analysis | ~20s | Monte Carlo (1000 sims) |
| Visualization + Export | ~5s | 5 figures + results |

---

## 6. Logging

### Console Output

- **Level:** INFO
- **Format:** Simple text (no colors)
- **Content:** Phase progress, key metrics

### Debug Log File

- **Location:** `output/result/logs/debug.log`
- **Level:** DEBUG (captures everything)
- **Format:** `timestamp | level | module | function:line | message`

### Example Console Output

```
======================================================================
  ML-MCDM: Hierarchical Ranking Pipeline
======================================================================
  Provinces         : 63
  Years             : 2011-2024 (14 years)
  Subcriteria       : 29
  Criteria          : 8
  MCDM methods      : 6 (TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, SAW)
  MC simulations    : 2000
  Sensitivity sims  : 1000
  Output            : output/result/
======================================================================

▶ Phase 1/7: Data Loading
  Loaded 63 provinces × 14 years × 29 subcriteria
  ✓ Completed in 0.91s

▶ Phase 2/7: Weight Calculation
  CRITIC Two-Level: Deterministic two-level CRITIC pipeline
  Weights: [0.142, 0.118, 0.095, 0.158, 0.127, 0.109, 0.132, 0.119]
  Cosine similarity: 0.9915 (stable)
  ✓ Completed in 19.51s

▶ Phase 3/7: Hierarchical Ranking
  Stage 1: 6 MCDM × 8 criteria with adaptive zero-handling
  Stage 2: Weighted aggregation
  Kendall's W: 0.8786 (strong agreement)
  Top-ranked: P02 (score = 0.8547)
  ✓ Completed in 7.02s

▶ Phase 4/7: ML Forecasting
  Ensemble: 5 models (CatBoost, Bayesian, QRF, KRR, SVR)
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
Outputs: output/result/
======================================================================
```

---

## Summary

The ML-MCDM pipeline provides:

1. **Rigorous Methodology**: Two-stage hierarchical ranking with adaptive zero-handling
2. **Objective Weighting**: CRITIC Two-Level deterministic pipeline
3. **Multi-Method Consensus**: 6 MCDM methods (TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, SAW)
4. **ML Forecasting**: 5-model ensemble (CatBoost, Bayesian, QRF, KRR, SVR) + Super Learner
5. **Hierarchical Sensitivity**: Multi-level robustness analysis (1000 simulations)
6. **Comprehensive Outputs**: 5 high-resolution figures + 17+ data files + detailed report
7. **Production-Ready**: Single-command execution with scientifically validated defaults

**Key Metrics (typical values):**
- Kendall's W: 0.85-0.90 (strong method agreement)
- CV R²: 0.70-0.80 (good predictive power)
- Overall Robustness: 0.90-0.98 (very stable rankings)
- Weight Stability: > 0.95 (bootstrap cosine similarity)

For methodology details, see:
- [ranking.md](ranking.md) — Hierarchical ranking methodology
- [weighting.md](weighting.md) — CRITIC Two-Level weight calculation
- [objective.md](objective.md) — project objectives

