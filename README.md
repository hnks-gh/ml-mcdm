# ML-MCDM Framework

**A Hybrid Multi-Criteria Decision Making Framework with Evidential Reasoning and Ensemble Machine Learning**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production-brightgreen.svg)](https://github.com/hoangsonww/ml-mcdm)

## Overview

This framework combines Multi-Criteria Decision Making (MCDM) with machine learning to analyze and forecast multi-dimensional performance across entities. It integrates three major components:

1. **Objective Weighting** via CRITIC-based adaptive weighting (NaN-aware, two-level)
2. **Hierarchical Ranking** using 6 Traditional MCDM methods + Evidential Reasoning (ER)
3. **ML Forecasting (optional)** via 6-model ensemble + Super Learner + Conformal Prediction

**Application:** Vietnam PAPI (Provincial Governance and Public Administration Performance Index) analysis across 63 provinces over 14 years (2011-2024).

---

## Key Features

### Technical Summary
- **Pipeline orchestration**: `MLMCDMPipeline` drives seven phases (data load, weighting, ranking, forecasting, analysis, visualization, export) with phase-level metrics, timing, and configurable switches from a single `Config` dataclass tree.
- **Data model**: Yearly panel matrices (63 provinces, 8 criteria, 29 subcriteria) are loaded as `YearContext` objects with explicit missingness semantics (NaN = missing, 0.0 = valid score), dynamic exclusion of all-NaN entities, and optional MICE-based imputation with full audit logs.
- **Weighting**: Adaptive CRITIC weighting is applied at both subcriteria and criteria levels using contrast intensity and inter-criteria conflict $C_j = \sigma_j \sum_k (1 - r_{jk})$, followed by normalization $w_j = C_j / \sum_k C_k$. NaN-aware preprocessing ensures stable weights in sparse years.
- **Hierarchical ranking**: Six MCDM methods (TOPSIS, VIKOR, PROMETHEE II, COPRAS, EDAS, SAW) run per criterion; their scores are individually reported with Kendall’s $W$ concordance. The framework supports evidential reasoning (ER) fusion into belief distributions and weighted ER aggregation across criteria, but ER is **disabled by default** (`use_evidential_reasoning = False`).
- **Forecasting (optional)**: Five-model ensemble (CatBoost, Bayesian Ridge, Quantile Random Forest, Kernel Ridge Regression, Support Vector Regression) with Super Learner meta-ensemble, panel-aware temporal CV, and conformal prediction for distribution-free $1-\alpha$ intervals (default $\alpha=0.05$).
- **Analysis & validation**: Sensitivity analysis on weights, beliefs, and forecasts; bootstrap and perturbation uncertainty; diagnostics for belief completeness, entropy, residual behavior, and temporal stability.
- **Outputs & visualization**: Phase-scoped CSV/JSON artifacts, 300 DPI figures, and text reports with reproducible directory layout under `output/result/` and full debug logs.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Panel Data (N provinces × T years × p criteria)              │
└────────────────┬────────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐   ┌─────────────────┐
│  WEIGHTING   │   │    RANKING      │
│              │   │                 │
│  CRITIC      │   │ Stage 1: Within │
│  Adaptive    │──►│  - 6 MCDM Mtds  │
│  Weighting   │   │  - ER Combine   │
│ (NaN-aware)  │   │ Stage 2: Global │
│              │   │  - ER Aggregate │
│              │   │  - Final Rank   │
└──────────────┘   └────────┬────────┘
                            │
               ┌────────────┼────────────┐
               ▼            ▼            ▼
        ┌───────────┐ ┌───────────┐ ┌───────────┐
        │ML FORECAST│ │ ANALYSIS  │ │ VISUALISE │
        │ (OPTIONAL)│ │           │ │ & EXPORT  │
        │           │ │• Sensitiv.│ │           │
        │• 5 Models │ │• Robust.  │ │• 7+ charts│
        │• Super L  │ │• Kendall W│ │• 14 files │
        └───────────┘ └───────────┘ └───────────┘
```

---

## Project Structure

```
ml-mcdm/
├── main.py                 # Entry point
├── pyproject.toml          # Package configuration & dependencies
│
├── data/                   # Input data
│   ├── csv/               # Yearly panel files (2011.csv ... 2024.csv)
│   ├── codebook/          # Variable descriptions
│   ├── data_loader.py     # Data I/O and validation
│   ├── missing_data.py    # ML panel imputation (build_ml_panel_data)
│   └── imputation/        # MICE imputation modules
│
├── pipeline.py            # Main orchestrator
├── loggers/               # Structured console + debug logging
├── output/                # Results export + report writers
│
├── weighting/             # Weight calculation
│   ├── critic.py          # CRITIC weighting
│   ├── adaptive.py        # NaN-aware adaptive weights
│   ├── bootstrap.py       # Bayesian bootstrap utilities
│   ├── normalization.py   # Min-max/vector/z-score normalization
│   └── base.py            # Weighting entry points + result types
│
├── ranking/               # MCDM methods + ER aggregation
│   ├── topsis.py
│   ├── vikor.py
│   ├── promethee.py
│   ├── copras.py
│   ├── edas.py
│   ├── saw.py
│   └── evidential_reasoning/
│       ├── base.py
│       └── hierarchical_er.py
│
├── ranking/               # Ranking orchestrator + ER aggregation
│   └── hierarchical_pipeline.py
│
├── analysis/              # Production-ready analysis
│   ├── sensitivity.py     # Hierarchical sensitivity (565 lines)
│   └── validation.py      # Comprehensive validation (533 lines)
│
├── forecasting/           # Machine learning (optional)
│   ├── base.py
│   ├── features.py        # 12-block temporal feature engineering
│   ├── preprocessing.py   # Scaling, transforms, splits
│   ├── catboost_forecaster.py # CatBoost (joint MultiRMSE boosting)
│   ├── bayesian.py        # Bayesian Ridge
│   ├── quantile_forest.py # Quantile RF
│   ├── kernel_ridge.py    # Kernel Ridge Regression
│   ├── svr.py             # Support Vector Regression
│   ├── panel_mice.py      # PanelSequentialMICE (opt-in)
│   ├── augmentation.py    # ConditionalPanelAugmenter (opt-in)
│   ├── shift_detection.py # MMD² covariate shift detection (opt-in)
│   ├── incremental_update.py # IncrementalEnsembleUpdater (opt-in)
│   ├── super_learner.py   # Meta-ensemble (OOF + Dirichlet stacking)
│   ├── conformal.py       # Conformal prediction (split / CV+ / ACI)
│   └── unified.py         # Ensemble orchestrator (6-stage pipeline)
│
├── tests/                 # Test suite (400+ tests)
│   ├── test_mcdm_traditional.py
│   ├── test_mcdm_textbook.py
│   ├── test_evidential_reasoning.py
│   ├── test_ranking_pipeline_nan.py
│   ├── test_forecasting.py
│   ├── test_missing_data.py
│   ├── test_sensitivity.py
│   ├── test_validation.py
│   ├── test_weighting.py
│   └── test_output.py
│
├── output/result/         # Generated results (git-ignored)
│   ├── figures/          # PNG charts (300 DPI)
│   ├── results/          # CSV files
│   ├── reports/          # Text reports
│   └── logs/             # Debug logs
│
└── docs/                  # Documentation
    ├── objective.md       # Project objectives
    ├── dataset_description.md  # Data description
    ├── workflow.md        # Pipeline workflow
    ├── weighting.md       # Weight calculation details
    ├── ranking.md         # ER ranking methodology
    └── forecast.md        # ML forecasting methods
```

---

## Documentation

### Core Documentation

| Document | Description |
|----------|-------------|
| [objective.md](docs/objective.md) | Project objectives and research questions |
| [dataset_description.md](docs/dataset_description.md) | Data structure and variables |
| [workflow.md](docs/workflow.md) | Pipeline workflow and execution |

### Technical Documentation

| Document | Description |
|----------|-------------|
| [weighting.md](docs/weighting.md) | Monte Carlo Entropy–CRITIC Ensemble methodology |
| [ranking.md](docs/ranking.md) | Hierarchical MCDM + Evidential Reasoning details |
| [forecast.md](docs/forecast.md) | Ensemble ML forecasting architecture |

---

## Methodology Highlights

### Evidential Reasoning (ER)

> **Note:** ER aggregation is **disabled by default** (`use_evidential_reasoning = False`). The framework supports ER fusion but it is not active in the current pipeline configuration.

The framework supports combining multiple assessments into belief distributions over evaluation grades:

$$
\text{Belief} = \{(\text{Excellent}, \beta_E), (\text{Good}, \beta_G), (\text{Fair}, \beta_F), (\text{Poor}, \beta_P), (\text{Bad}, \beta_B), (H, \beta_H)\}
$$

**Pairwise combination:**
$$
\beta_n = K \left[\beta_{1,n}\beta_{2,n} + \beta_{1,n}\beta_{2,H} + \beta_{1,H}\beta_{2,n}\right]
$$

Where K is normalization constant handling conflicts.

**Two-stage architecture (available but disabled by default):**
1. **Stage 1**: Within each criterion, combine 6 method scores via ER
2. **Stage 2**: Combine 8 criterion beliefs via weighted ER

**Reference:** Yang, J.B., & Xu, D.L. (2002). On the evidential reasoning algorithm. *IEEE Trans. SMC-A*, 32(3), 289-304.

---

### CRITIC-Based Adaptive Weighting

The weighting module uses the CRITIC method with NaN-aware preprocessing and two-level aggregation:

1. **Contrast intensity**: standard deviation $\sigma_j$ captures variability of criterion $j$
2. **Conflict**: $(1 - r_{jk})$ measures disagreement with other criteria
3. **CRITIC score**: $C_j = \sigma_j \sum_k (1 - r_{jk})$
4. **Normalization**: $w_j = C_j / \sum_k C_k$
5. **Two-level weights**: subcriteria weights roll up into criterion-level weights for ER aggregation

---

### ML Forecasting

The pipeline integrates a five-model ensemble (CatBoost, Bayesian Ridge,
Quantile Random Forest, Kernel Ridge Regression, Support Vector Regression). A Super Learner
meta-ensemble optimizes per-output model weights from out-of-fold predictions, while
conformal prediction provides distribution-free uncertainty intervals. An ML-imputed
copy of the panel (`build_ml_panel_data`) is passed to the forecaster so the MCDM
phases remain on observed data.

---

## Output Files

### Results (CSV)

| Phase | Example Files | Description |
|------|----------------|-------------|
| **weighting/** | `weights_analysis.csv`, `critic_weights_YYYY.csv`, `sc_global_weights_all_years.csv` | CRITIC diagnostics and per-year weights |
| **mcdm/** | `mcdm_scores_C01-C08.csv`, `mcdm_scores_composite.csv` | Per-criterion scores from 6 MCDM methods |
| **ranking/** | `mcdm_criteria_C01-C08_ranking.csv`, `mcdm_scores_composite_ranking.csv` | ER aggregation results and method comparisons |
| **forecasting/** | `forecast_predictions_target_year.csv`, `forecast_model_comparison.csv`, `forecast_cv_metrics.csv` | Forecast outputs and CV metrics (optional) |
| **sensitivity/** | `sensitivity_summary.json` | Robustness summaries and perturbation diagnostics |

### Figures (PNG, 300 DPI)

- Final ranking summary chart
- Score distribution across provinces
- Weight comparison across criteria
- Sensitivity analysis heatmap
- Feature importance bar chart

### Reports (TXT)

- `report.txt`: Comprehensive analysis summary
- `debug.log`: Detailed execution log

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## References

### Core Methodologies

1. **Yang, J.B., & Xu, D.L.** (2002). On the evidential reasoning algorithm for multiple attribute decision analysis under uncertainty. *IEEE Transactions on Systems, Man, and Cybernetics—Part A*, 32(3), 289-304.

2. **Hwang, C.L., & Yoon, K.** (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer.

3. **Diakoulaki, D., Mavrotas, G., & Papayannakis, L.** (1995). Determining objective weights in multiple criteria problems: The CRITIC method. *Computers & Operations Research*, 22(7), 763-770.

4. **Friedman, J.H.** (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.

5. **Breiman, L.** (2001). Random forests. *Machine Learning*, 45(1), 5-32.
