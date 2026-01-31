# ML-MCDM

Machine Learning enhanced Multi-Criteria Decision Making for panel data analysis.

## Overview

This framework integrates **MCDM methods** with **Machine Learning** to rank entities (provinces, companies, etc.) across multiple time periods and criteria. It combines traditional decision-making techniques with predictive modeling for comprehensive analysis.

### Technical Approach

```
Panel Data (entities × time × criteria)
         │
         ├─► Weight Calculation (Entropy, CRITIC, Ensemble)
         │
         ├─► MCDM Ranking
         │     ├─ Traditional: TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS
         │     └─ Fuzzy: Handles uncertainty via triangular fuzzy numbers
         │
         ├─► ML Analysis
         │     ├─ Panel Regression (Fixed/Random Effects)
         │     ├─ Random Forest with Time-Series CV
         │     ├─ Gradient Boosting Ensemble
         │     └─ Neural Networks with Attention
         │
         ├─► Ensemble Integration (Stacking, Borda, Copeland)
         │
         └─► Analysis (Convergence, Sensitivity, Validation)
```

## Project Structure

```
ml-topsis-vikor/
├── run.py                  # Entry point
├── pyproject.toml          # Package config
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration dataclasses
│   ├── pipeline.py         # Main orchestrator
│   ├── data_loader.py      # Panel data I/O
│   ├── output_manager.py   # Results export
│   ├── visualization.py    # Charts (300 DPI)
│   ├── logger.py
│   │
│   ├── mcdm/               # Decision methods
│   │   ├── weights.py      # Entropy, CRITIC, Ensemble
│   │   ├── topsis.py       # Distance to ideal solution
│   │   ├── vikor.py        # Compromise ranking
│   │   ├── promethee.py    # Outranking flows
│   │   ├── copras.py       # Proportional assessment
│   │   ├── edas.py         # Distance from average
│   │   ├── fuzzy_base.py   # Triangular fuzzy numbers
│   │   └── fuzzy_*.py      # Fuzzy variants
│   │
│   ├── ml/                 # Machine learning
│   │   ├── panel_regression.py     # FE/RE/Pooled OLS
│   │   ├── random_forest_ts.py     # RF with temporal CV
│   │   ├── advanced_forecasting.py # Gradient boosting
│   │   ├── neural_forecasting.py   # MLP + Attention
│   │   ├── unified_forecasting.py  # Ensemble orchestrator
│   │   ├── lstm_forecast.py
│   │   └── rough_sets.py           # Feature reduction
│   │
│   ├── ensemble/           # Aggregation
│   │   ├── stacking.py     # Meta-learner
│   │   └── aggregation.py  # Borda, Copeland
│   │
│   └── analysis/           # Validation
│       ├── convergence.py  # Beta/Sigma convergence
│       ├── sensitivity.py  # Weight perturbation
│       └── validation.py   # Bootstrap, CV
│
├── data/
│   └── data.csv            # Input: Year, Province, C01-C20
│
├── outputs/
│   ├── figures/            # PNG charts
│   ├── results/            # CSV files
│   └── reports/            # Analysis reports
│
├── tests/
│   └── test_core.py
│
└── docs/
    ├── METHODS.md          # Mathematical formulations
    └── WORKFLOW.md         # Pipeline phases
```

## Methods

| Category | Methods |
|----------|---------|
| **Weighting** | Entropy (information content), CRITIC (contrast + correlation), Ensemble |
| **MCDM** | TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS — each with fuzzy variant |
| **ML** | Panel Regression, Random Forest, Gradient Boosting, Neural Networks |
| **Ensemble** | Stacking meta-learner, Borda Count, Copeland |
| **Analysis** | β/σ convergence, Monte Carlo sensitivity, Bootstrap validation |