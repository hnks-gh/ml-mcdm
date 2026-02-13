# ML-MCDM Framework

**A Hybrid Multi-Criteria Decision Making Framework with Intuitionistic Fuzzy Sets, Evidential Reasoning, and Ensemble Machine Learning**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production-brightgreen.svg)](https://github.com/hoangsonww/ml-mcdm)

## Overview

This framework combines state-of-the-art Multi-Criteria Decision Making (MCDM) methods with Machine Learning to analyze and forecast multi-dimensional performance across entities. It integrates three major components:

1. **Objective Weighting** via Game Theory Weight Combination (GTWC)
2. **Hierarchical Ranking** using Intuitionistic Fuzzy Sets (IFS) + Evidential Reasoning (ER)
3. **Ensemble Forecasting** with multiple machine learning models

**Application:** Vietnam PAPI (Provincial Governance and Public Administration Performance Index) analysis across 63 provinces over 14 years (2011-2024).

---

## Key Features

### 🎯 Hierarchical Ranking System
- **12 MCDM Methods**: 6 Traditional + 6 IFS variants
  - Traditional: TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, SAW
  - IFS Extensions: Handles uncertainty via Atanassov's Intuitionistic Fuzzy Sets
- **Two-Stage Architecture**: Within-criterion combination → Global aggregation
- **Evidential Reasoning**: Rigorous belief combination (Yang & Xu, 2002)
- **Adaptive Zero-Handling**: Automatic exclusion of missing/zero data with restoration

### ⚖️ Objective Weight Calculation
- **4 Complementary Methods**: Entropy, CRITIC, MEREC, Standard Deviation
- **Game Theory Combination**: Intra-group hybridization + cooperative optimization
- **Uncertainty Quantification**: Bayesian Bootstrap (999 iterations)
- **Temporal Stability**: Split-half validation

### 🤖 Machine Learning Forecasting
- **7 Model Types**: 
  - Tree-based: Gradient Boosting, Random Forest, Extra Trees
  - Linear: Bayesian Ridge, Huber, Ridge
  - Neural: MLP, Self-Attention (optional)
- **Performance-Based Weighting**: Softmax over CV R² scores
- **Uncertainty Quantification**: Prediction intervals + model disagreement
- **Rich Feature Engineering**: Lag, rolling stats, momentum, trends

### 📊 Analysis & Validation
- **Convergence Analysis**: Kendall's W concordance coefficient
- **Sensitivity Analysis**: Weight perturbation studies
- **Cross-Validation**: Time-series CV with proper temporal ordering
- **Robustness Testing**: Bootstrap confidence intervals

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
│ • CRITIC     │──►│  - Traditional  │
│ • MEREC      │   │  - IFS-MCDM     │
│ • Std Dev    │   │  - ER Combine   │
│              │   │                 │
│ Game Theory  │   │ Stage 2: Global │
│ Combination  │   │  - ER Aggregate │
│              │   │  - Final Rank   │
└──────────────┘   └────────┬────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │  FORECASTING   │
                   │                │
                   │ • Features Eng │
                   │ • 7 ML Models  │
                   │ • Ensemble     │
                   │ • Uncertainty  │
                   └────────┬───────┘
                            │
                            ▼
                   ┌────────────────┐
                   │   ANALYSIS     │
                   │                │
                   │ • Validation   │
                   │ • Sensitivity  │
                   │ • Robustness   │
                   └────────────────┘
```

---

## Project Structure

```
ml-mcdm/
├── run.py                  # Entry point
├── pyproject.toml          # Package configuration
├── requirements.txt        # Dependencies
│
├── data/                   # Input data
│   ├── 2011-2024.csv      # Historical panel data
│   └── codebook/          # Variable descriptions
│
├── src/                    # Source code
│   ├── pipeline.py        # Main orchestrator
│   ├── config.py          # Configuration management
│   ├── data_loader.py     # Data I/O and validation
│   ├── logger.py          # Logging system
│   ├── output_manager.py  # Results export
│   ├── visualization.py   # Chart generation (300 DPI)
│   │
│   ├── weighting/         # Weight calculation
│   │   ├── entropy.py
│   │   ├── critic.py
│   │   ├── merec.py
│   │   ├── standard_deviation.py
│   │   ├── fusion.py      # Game Theory Combination
│   │   └── hybrid_weighting.py  # Main interface
│   │
│   ├── ranking/           # Ranking methods
│   │   ├── pipeline.py    # Hierarchical ranking orchestrator
│   │   ├── ifs/           # Intuitionistic Fuzzy Sets
│   │   │   ├── base.py
│   │   │   ├── topsis.py
│   │   │   ├── vikor.py
│   │   │   ├── promethee.py
│   │   │   ├── copras.py
│   │   │   ├── edas.py
│   │   │   └── saw.py
│   │   ├── evidential_reasoning/  # ER aggregation
│   │   │   ├── base.py
│   │   │   └── hierarchical.py
│   │   └── traditional/   # Traditional MCDM
│   │       ├── topsis.py
│   │       ├── vikor.py
│   │       ├── promethee.py
│   │       ├── copras.py
│   │       ├── edas.py
│   │       └── saw.py
│   │
│   ├── ml/                # Machine learning
│   │   └── forecasting/
│   │       ├── base.py
│   │       ├── features.py          # Feature engineering
│   │       ├── tree_ensemble.py     # GB, RF, ET
│   │       ├── linear.py            # Bayesian, Huber, Ridge
│   │       ├── neural.py            # MLP, Attention
│   │       └── unified.py           # Ensemble orchestrator
│   │
│   └── analysis/          # Analysis tools
│       ├── sensitivity.py
│       └── validation.py
│
├── outputs/               # Generated results
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
    ├── ranking.md         # IFS+ER ranking methodology
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
| [weighting.md](docs/weighting.md) | Game Theory Weight Combination (GTWC) methodology |
| [ranking.md](docs/ranking.md) | IFS-MCDM + Evidential Reasoning details |
| [forecast.md](docs/forecast.md) | Ensemble ML forecasting architecture |

---

## Methodology Highlights

### Intuitionistic Fuzzy Sets (IFS)

Extends classical fuzzy sets by introducing independent non-membership:

$$
\text{IFN} = (\mu, \nu, \pi)
$$

Where:
- **μ (mu)**: Membership degree ∈ [0, 1]
- **ν (nu)**: Non-membership degree ∈ [0, 1]
- **π (pi)**: Hesitancy = 1 - μ - ν
- **Constraint**: μ + ν ≤ 1

**Construction from temporal data:**
- μ: Normalized current value
- ν: Temporal variance (historical std)
- π: Unexplained uncertainty

**Reference:** Atanassov, K.T. (1986). Intuitionistic fuzzy sets. *Fuzzy Sets and Systems*, 20(1), 87-96.

---

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
1. **Stage 1**: Within each criterion, combine 12 method scores via ER
2. **Stage 2**: Combine 8 criterion beliefs via weighted ER

**Reference:** Yang, J.B., & Xu, D.L. (2002). On the evidential reasoning algorithm. *IEEE Trans. SMC-A*, 32(3), 289-304.

---

### Game Theory Weight Combination (GTWC)

Combines 4 weighting methods through:

1. **Intra-Group Hybridization:**
   - Group A (Dispersion): Geometric mean of Entropy + Std Dev
   - Group B (Interaction): Harmonic mean of CRITIC + MEREC

2. **Cooperative Game Optimization:**
   $$
   \min L = \|α_1W_A + α_2W_B - W_A\|^2 + \|α_1W_A + α_2W_B - W_B\|^2
   $$

3. **Final Aggregation:**
   $$
   W^* = α_1 \cdot W_{\text{GroupA}} + α_2 \cdot W_{\text{GroupB}}
   $$

4. **Bayesian Bootstrap:** 999 iterations for uncertainty quantification

---

### Ensemble Machine Learning

**7 Model Types:**

| Model | Type | Key Feature |
|-------|------|-------------|
| Gradient Boosting | Tree | Huber loss (outlier robust) |
| Random Forest | Tree | Natural uncertainty from tree variance |
| Extra Trees | Tree | Extra randomization, lower variance |
| Bayesian Ridge | Linear | Posterior predictive uncertainty |
| Huber Regression | Linear | Robust to outliers |
| Ridge Regression | Linear | Fast L2 regularization |
| MLP | Neural | SELU activation, self-normalizing |
| Self-Attention | Neural | Learned feature importance |

**Performance-Based Weighting:**
$$
w_i = \frac{\exp(5 \cdot R^2_i)}{\sum_j \exp(5 \cdot R^2_j)}
$$

**Uncertainty Quantification:**
$$
\sigma^2_{\text{total}} = \underbrace{\sum_i w_i \sigma^2_i}_{\text{within-model}} + \underbrace{\sum_i w_i (\hat{y}_i - \bar{\hat{y}})^2}_{\text{disagreement}}
$$

---

## Output Files

### Results (CSV)

| File | Description |
|------|-------------|
| `final_rankings.csv` | Current year rankings with scores |
| `predicted_rankings_2025.csv` | Forecasted rankings |
| `mcdm_scores_detailed.csv` | Scores from all 12 methods |
| `weights_analysis.csv` | Criterion weights with uncertainty |
| `cv_scores.csv` | Cross-validation performance |
| `feature_importance.csv` | ML feature importance scores |
| `prediction_uncertainty_2025.csv` | Forecast uncertainty estimates |

### Figures (PNG, 300 DPI)

- `current_rankings_map.png`: Geographic visualization
- `forecast_comparisons.png`: Current vs predicted
- `method_convergence.png`: MCDM method agreement
- `feature_importance.png`: Top predictive features
- `uncertainty_intervals.png`: Prediction confidence bands

### Reports (TXT)

- `report.txt`: Comprehensive analysis summary
- `debug.log`: Detailed execution log

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## References

### Core Methodologies

1. **Atanassov, K.T.** (1986). Intuitionistic fuzzy sets. *Fuzzy Sets and Systems*, 20(1), 87-96.

2. **Yang, J.B., & Xu, D.L.** (2002). On the evidential reasoning algorithm for multiple attribute decision analysis under uncertainty. *IEEE Transactions on Systems, Man, and Cybernetics—Part A*, 32(3), 289-304.

3. **Hwang, C.L., & Yoon, K.** (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer.

4. **Keshavarz-Ghorabaee, M., et al.** (2021). Determination of Objective Weights Using a New Method Based on the Removal Effects of Criteria (MEREC). *Symmetry*, 13(4), 525.

5. **Diakoulaki, D., Mavrotas, G., & Papayannakis, L.** (1995). Determining objective weights in multiple criteria problems: The CRITIC method. *Computers & Operations Research*, 22(7), 763-770.

6. **Friedman, J.H.** (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.

7. **Breiman, L.** (2001). Random forests. *Machine Learning*, 45(1), 5-32.