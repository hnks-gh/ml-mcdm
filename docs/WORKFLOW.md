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
- **Bayesian Bootstrap**: 999 iterations for weight uncertainty quantification
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
│  │ Yearly CSVs → PanelData (63 provinces × 14 years × 28)  │           │
│  └────────────────────────┬─────────────────────────────────┘           │
│                           │                                              │
│  Phase 2: Weight Calculation                                            │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ GTWC: Entropy + CRITIC + MEREC + SD                     │           │
│  │ → Game Theory Combination → Bayesian Bootstrap (999)    │           │
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
│  Phase 4: ML Feature Importance                                         │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ Random Forest (200 estimators, 5-fold time-series CV)   │           │
│  │ → Gini importance + CV R² validation                    │           │
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
├── long: DataFrame         # Long format (Year, Province, C01...C28)
├── wide: Dict[year, DataFrame]  # Province × Subcriteria per year
├── cross_section: Dict[year, DataFrame]
├── provinces: List[str]    # Entity identifiers (63 provinces + city)
├── years: List[int]        # Time periods (2011-2024, 14 years)
├── criteria: List[str]     # Criterion groups (C01-C08, 8 criteria)
└── subcriteria: List[str]  # Subcriteria names (28 components)
```

**Processing:**
- Loads each year's CSV independently
- Validates province codes (64 entities including cities)
- Maps 28 subcriteria to 8 hierarchical criteria using codebook
- Handles missing values via forward-fill
- Constructs long-format panel for temporal analysis

**Validation:**
- Shape: (64 provinces × 14 years × 28 subcriteria) = 25,088 observations
- All values normalized to [0, 1] range
- No duplicate province-year combinations

---

### Phase 2: GTWC Weight Calculation

**Purpose:** Calculate objective criterion weights using Game Theory Weight Combination.

**Four Base Methods:**

| Method | Description | Formula |
|--------|-------------|--------|
| **Entropy** | Information content | $w_j = \frac{1 - E_j}{\sum_k (1 - E_k)}$ |
| **CRITIC** | Contrast + correlation | $w_j = \frac{C_j(1 - \sum_k r_{jk})}{\sum_k C_k(1 - \sum_m r_{km})}$ |
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

**Bayesian Bootstrap (999 iterations):**
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

### Phase 4: ML Feature Importance

**Purpose:** Identify which subcriteria contribute most to final ranking using Random Forest.

**Model Configuration:**
- **Algorithm:** Random Forest Regressor
- **Target:** Final ER utility scores
- **Features:** 28 subcriteria values
- **Hyperparameters:**
  - `n_estimators=200` (production) or `30` (quick-test)
  - `max_depth=8`
  - `min_samples_split=10`
  - `random_state=42`

**Cross-Validation:**
- **Method:** TimeSeriesSplit (5 folds)
- **Respects temporal ordering** (no data leakage)
- **Metrics:** R², MAE, RMSE per fold
- **Expected performance:** CV R² > 0.70

**Feature Importance Extraction:**
- **Gini importance** (mean decrease in impurity)
- Normalized to sum to 1.0
- Ranked from most to least important

**Interpretation:**
- High importance → strong predictor of final ranking
- Validates criterion structure (do important features align with expectations?)
- Informs policy: which dimensions matter most for overall governance?

**Output Files:**
- `feature_importance.csv`: 28 subcriteria ranked by Gini importance
- `cv_scores.csv`: R², MAE, RMSE per fold
- `rf_test_metrics.csv`: Final test set performance

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

**Output:** 5-10 PNG files in `outputs/figures/`

**Purpose:** Save all results in organized structure.

See [Output Structure](#4-output-structure) below.

---

## 4. Output Structure

```
outputs/
├── figures/                          # High-resolution visualizations (5 PNG files)
│   ├── final_ranking_summary.png      # Top 20 provinces with ER utility
│   ├── score_distribution.png         # Histogram + KDE
│   ├── weights_comparison.png         # GTWC weights (8 criteria)
│   ├── sensitivity_analysis.png       # Rank stability heatmap
│   └── feature_importance.png         # RF Gini importance
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
│   ├── feature_importance.csv         # RF Gini importance (28 subcriteria)
│   ├── cv_scores.csv                  # Time-series CV results (5 folds)
│   │
│   ├── sensitivity_subcriteria.csv    # Subcriteria sensitivity (28 scores)
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
python main.py                # Quick-test mode  (29 bootstrap, 30 RF trees, 100 MC sims)
python main.py --production   # Production mode  (999 bootstrap, 200 RF trees, 1000 MC sims)
```

**Python API:**

```python
from ml_mcdm.pipeline import MLMCDMPipeline
from ml_mcdm.config import get_default_config

# Quick-test config
config = get_default_config()
config.weighting.bootstrap_iterations = 29
config.validation.n_simulations = 100

pipeline = MLMCDMPipeline(config)
result = pipeline.run()

# Access results
rankings = result.ranking.final_er_scores
print(f"Kendall's W: {result.ranking.kendall_w:.4f}")
print(f"Robustness: {result.analysis['sensitivity'].overall_robustness:.4f}")
```

### Key Configuration Options

| Parameter | Quick-Test | Production | Description |
|-----------|------------|------------|--------------|
| `bootstrap_iterations` | 29 | 999 | Bayesian bootstrap for weight uncertainty |
| `n_estimators` | 30 | 200 | Random Forest trees |
| `n_simulations` | 100 | 1000 | Monte Carlo sensitivity simulations |
| `random_state` | 42 | 42 | Reproducibility seed |
| `n_splits` | 5 | 5 | Time-series CV folds |
| `output_dir` | `outputs` | `outputs` | Result directory |
| `dpi` | 300 | 300 | Figure resolution |

### Performance

| Mode | Time | Bootstrap | RF Trees | MC Sims |
|------|------|-----------|----------|---------|
| **Quick-test** | ~30s | 29 | 30 | 100 |
| **Production** | ~3min | 999 | 200 | 1000 |

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
======================================================================
  ML-MCDM: IFS + Evidential Reasoning Hierarchical Ranking
======================================================================
  Mode              : QUICK TEST
  Provinces         : 63
  Years             : 2011-2024 (14 years)
  Subcriteria       : 28
  Criteria          : 8
  MCDM methods      : 12 (6 traditional + 6 IFS)
  Bootstrap iters   : 29
  Sensitivity sims  : 100
  Output            : outputs/
======================================================================

▶ Phase 1/7: Data Loading
  Loaded 63 provinces × 14 years × 28 subcriteria
  ✓ Completed in 0.91s

▶ Phase 2/7: Weight Calculation
  GTWC: Entropy + CRITIC + MEREC + SD with 29 bootstrap
  Weights: [0.142, 0.118, 0.095, 0.158, 0.127, 0.109, 0.132, 0.119]
  Cosine similarity: 0.9915 (stable)
  ✓ Completed in 19.51s

▶ Phase 3/7: Hierarchical Ranking
  Stage 1: 12 MCDM × 8 criteria with adaptive zero-handling
  Stage 2: Weighted ER aggregation
  Kendall's W: 0.8786 (strong agreement)
  Top-ranked: P02 (utility = 0.8547)
  ✓ Completed in 7.02s

▶ Phase 4/7: ML Feature Importance
  Random Forest (30 trees, 5-fold time-series CV)
  CV R²: 0.7355 ± 0.084
  Top feature: C_01_1 (importance = 0.082)
  ✓ Completed in 1.45s

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
Outputs: outputs/
======================================================================
```

---

## Summary

The ML-MCDM pipeline provides:

1. **Rigorous Methodology**: IFS + two-stage ER with adaptive zero-handling
2. **Objective Weighting**: GTWC (4 methods) with Bayesian Bootstrap uncertainty
3. **Multi-Method Consensus**: 12 MCDM methods (6 traditional + 6 IFS)
4. **ML Feature Importance**: Random Forest with time-series CV
5. **Hierarchical Sensitivity**: Multi-level robustness analysis (100+ simulations)
6. **Comprehensive Outputs**: 5 high-resolution figures + 17+ data files + detailed report
7. **Production-Ready**: Single-command execution with quick-test and production modes

**Key Metrics (typical values):**
- Kendall's W: 0.85-0.90 (strong method agreement)
- CV R²: 0.70-0.80 (good predictive power)
- Overall Robustness: 0.90-0.98 (very stable rankings)
- Weight Stability: > 0.95 (bootstrap cosine similarity)

For methodology details, see:
- [ranking.md](ranking.md) — IFS + ER hierarchical ranking
- [weighting.md](weighting.md) — GTWC weight calculation
- [objective.md](objective.md) — project objectives
