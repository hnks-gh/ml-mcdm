# ML-MCDM Framework

**A Hybrid Multi-Criteria Decision Making Framework with Evidential Reasoning and Ensemble Machine Learning**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production-brightgreen.svg)](https://github.com/hoangsonww/ml-mcdm)

## Overview

This framework combines state-of-the-art Multi-Criteria Decision Making (MCDM) methods with Machine Learning to analyze and forecast multi-dimensional performance across entities. It integrates three major components:

1. **Objective Weighting** via Monte Carlo Entropy–CRITIC Ensemble
2. **Hierarchical Ranking** using 5 Traditional MCDM methods + Evidential Reasoning (ER)
3. **ML Forecasting** via 6-model ensemble + Super Learner + Conformal Prediction

**Application:** Vietnam PAPI (Provincial Governance and Public Administration Performance Index) analysis across 63 provinces over 14 years (2011-2024).

---

## Key Features

### 🎯 Hierarchical Ranking System
- **5 MCDM Methods**: TOPSIS, VIKOR, PROMETHEE II, COPRAS, EDAS
- **Two-Stage Architecture**: Within-criterion combination → Global aggregation
- **Evidential Reasoning**: Rigorous belief combination (Yang & Xu, 2002)
- **Adaptive NaN Handling**: Automatic exclusion of inactive sub-criteria per year

### ⚖️ Objective Weight Calculation
- **2 Complementary Methods**: Shannon Entropy + CRITIC
- **Beta-Blended MC Ensemble**: Beta(α_a, α_b)-sampled blend, 64-point grid tuning
- **Uncertainty Quantification**: Bayesian Bootstrap (200 iterations)
- **Temporal Stability**: Split-half validation

### 🤖 Machine Learning Forecasting
- **State-of-the-Art Ensemble**: 5 diverse models optimized for N<1000
  - Gradient Boosting (Huber loss)
  - Bayesian Ridge (uncertainty quantification)
  - Quantile Random Forest (distributional forecasting)
  - Panel VAR (panel-specific dynamics)
  - Neural Additive Models (interpretable non-linearity)
- **Super Learner**: Automatic optimal model weighting via meta-learning
- **Conformal Prediction**: Distribution-free 95% prediction intervals
- **Feature Importance**: Aggregated across all 5 models

### 📊 Analysis & Validation
- **Hierarchical Sensitivity Analysis**: Multi-level robustness testing
  - Subcriteria & criteria weight perturbation (±15%)
  - Temporal stability (year-to-year correlation)
  - Monte Carlo simulation (100+ iterations)
  - Forecast robustness testing
- **Comprehensive Validation**: End-to-end pipeline validation
  - Cross-level consistency checking
  - Weight scheme robustness (temporal + method agreement)
  - Forecast quality metrics
- **Convergence Analysis**: Kendall's W concordance coefficient
- **Bootstrap Uncertainty**: Bayesian Bootstrap confidence intervals

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
│ • Entropy    │   │ Stage 1: Within │
│ • CRITIC     │──►│  - 5 MCDM Mtds  │
│              │   │  - ER Combine   │
│ MC Ensemble  │   │                 │
│ (Beta blend) │   │ Stage 2: Global │
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
        │• 6 Models │ │• Robust.  │ │• 5 charts │
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
│   ├── 2011-2024.csv      # Historical panel data
│   └── codebook/          # Variable descriptions
│
├── pipeline.py            # Main orchestrator
├── config.py              # Configuration management
├── data_loader.py         # Data I/O and validation
├── logger.py              # Logging system
├── output_manager.py      # Results export
├── visualization.py       # Chart generation (300 DPI)
│
├── weighting/             # Weight calculation
│   ├── entropy.py
│   ├── critic.py
│   ├── adaptive.py        # NaN-aware adaptive weights
│   ├── bootstrap.py       # Bayesian Bootstrap
│   ├── normalization.py   # Global min-max normalization
│   └── hybrid_weighting.py  # Main interface (MC ensemble)
│
├── mcdm/                  # MCDM methods
│   ├── traditional/       # Traditional MCDM
│   │   ├── topsis.py
│   │   ├── vikor.py
│   │   ├── promethee.py
│   │   ├── copras.py
│   │   ├── edas.py
│   │   └── saw.py
│
├── evidential_reasoning/  # ER aggregation
│   ├── base.py            # BeliefDistribution, ER engine
│   └── hierarchical_er.py # Two-stage hierarchical ER
│
├── ranking/               # Ranking orchestrator
│   └── pipeline.py        # Hierarchical ranking pipeline
│
├── analysis/              # Production-ready analysis
│   ├── sensitivity.py     # Hierarchical sensitivity (565 lines)
│   └── validation.py      # Comprehensive validation (533 lines)
│
├── forecasting/           # Machine learning (experimental)
│   ├── base.py
│   ├── features.py        # Feature engineering
│   ├── tree_ensemble.py   # GB, RF, ET
│   ├── linear.py          # Bayesian, Huber, Ridge
│   ├── neural.py          # MLP, Attention
│   └── unified.py         # Ensemble orchestrator
│
├── tests/                 # Test suite
│   └── weighting/         # Weighting module tests
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

Combines multiple assessments into belief distributions over evaluation grades:

$$
\text{Belief} = \{(\text{Excellent}, \beta_E), (\text{Good}, \beta_G), (\text{Fair}, \beta_F), (\text{Poor}, \beta_P), (\text{Bad}, \beta_B), (H, \beta_H)\}
$$

**Pairwise combination:**
$$
\beta_n = K \left[\beta_{1,n}\beta_{2,n} + \beta_{1,n}\beta_{2,H} + \beta_{1,H}\beta_{2,n}\right]
$$

Where K is normalization constant handling conflicts.

**Two-stage architecture:**
1. **Stage 1**: Within each criterion, combine 5 method scores via ER
2. **Stage 2**: Combine 8 criterion beliefs via weighted ER

**Reference:** Yang, J.B., & Xu, D.L. (2002). On the evidential reasoning algorithm. *IEEE Trans. SMC-A*, 32(3), 289-304.

---

### Monte Carlo Entropy–CRITIC Ensemble

Combines two complementary objective weighting methods:

1. **Shannon Entropy**: captures dispersion of normalized scores across provinces
2. **CRITIC**: captures inter-criteria conflict via standard deviation + correlation
3. **Beta-blend**: $w = \beta \cdot w_E + (1-\beta) \cdot w_C$, $\beta \sim \text{Beta}(\alpha_a, \alpha_b)$
4. **Hyperparameter tuning**: 64-point grid search over $(\alpha_a, \alpha_b)$; optional Bayesian GP
5. **Bayesian Bootstrap**: 200 iterations for uncertainty quantification

---

### ML Forecasting

The pipeline integrates state-of-the-art ensemble forecasting with 6 diverse
models (Gradient Boosting, Bayesian Ridge, Quantile Forest, Panel VAR,
Neural Additive Models). Super Learner meta-ensemble
automatically optimizes model weights, with conformal prediction providing
distribution-free uncertainty intervals.

---

## Output Files

### Results (CSV)

| File | Description |
|------|-------------|
| `final_rankings.csv` | Final province rankings with ER scores |
| `criterion_weights.csv` | MC ensemble weights with bootstrap uncertainty |
| `mcdm_scores_C01–C08.csv` | Per-criterion scores from 5 MCDM methods |
| `mcdm_rank_comparison.csv` | Rank comparison across MCDM methods |
| `weights_analysis.csv` | Weight derivation details |
| `forecast_feature_importance.csv` | Aggregated from 6 forecast models (optional) |
| `forecast_cv_metrics.csv` | Cross-validation performance (optional) |
| **`sensitivity_subcriteria.csv`** | **29 subcriteria sensitivity scores** |
| **`sensitivity_criteria.csv`** | **8 criteria sensitivity scores** |
| **`temporal_stability.csv`** | **Year-to-year rank correlations** |
| **`top_n_stability.csv`** | **Top-N ranking stability metrics** |
| `robustness_summary.csv` | Overall robustness + confidence level |
| `prediction_uncertainty_er.csv` | ER belief-structure uncertainty |
| `data_summary_statistics.csv` | Descriptive statistics of input data |
| `execution_summary.json` | Pipeline timing and metadata |
| `config_snapshot.json` | Full configuration used |

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