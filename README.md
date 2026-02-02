# ML-MCDM Framework v2.0

Machine Learning Enhanced Multi-Criteria Decision Making with Multiple Methods for Panel Data Analysis.

## Overview

This framework integrates **10 MCDM methods** (5 traditional + 5 fuzzy variants) with **advanced Machine Learning** to rank entities (provinces) across multiple time periods and criteria. It combines multiple decision-making techniques with ensemble forecasting for comprehensive analysis and robust future predictions.

```
Panel Data (64 provinces × 5 years × 20 criteria)
         │
         ├─► Weight Calculation (Entropy, CRITIC, Ensemble)
         │
         ├─► MCDM Ranking
         │     ├─ Traditional: TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS
         │     └─ Fuzzy: Handles uncertainty via triangular fuzzy numbers
         │
         ├─► ML Forecasting (7 Models: GB, RF, ET, Bayesian, Huber, MLP, Attention)
         │     ├─ Unified weighted predictions
         │     └─ Future Predictions with uncertainty
         │
         ├─► Ensemble Integration (Stacking, Borda, Copeland, Kemeny-Young)
         │
         └─► Analysis (Convergence, Sensitivity, Validation)
```

## Project Structure

```
ml-mcdm/
├── run.py                  # Entry point
├── pyproject.toml          # Package config
├── requirements.txt
│
├── src/
│   ├── config.py           # Configuration
│   ├── pipeline.py         # Main orchestrator
│   ├── data_loader.py      # Panel data I/O
│   ├── output_manager.py   # Results export
│   ├── visualization.py    # Charts (300 DPI)
│   ├── logger.py           # Logging system
│   │
│   ├── weighting/          # Weight calculation
│   │   ├── entropy.py
│   │   ├── critic.py
│   │   ├── ensemble.py
│   │   └── docs/README.md
│   │
│   ├── mcdm/               # Decision methods
│   │   ├── traditional/    # TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS
│   │   │   └── docs/README.md
│   │   └── fuzzy/          # Fuzzy variants with TFN
│   │       └── docs/README.md
│   │
│   ├── ml/                 # Machine learning
│   │   └── forecasting/    # 7 models: GB, RF, ET, Bayesian, Huber, MLP, Attention
│   │       └── docs/README.md
│   │
│   ├── ensemble/           # Aggregation
│   │   └── aggregation/    # Stacking, Borda, Copeland, Kemeny-Young
│   │       └── docs/README.md
│   │
│   └── analysis/           # Validation
│       ├── sensitivity.py
│       ├── validation.py
│       └── docs/README.md
│
├── data/
│   └── data.csv            # Input: Year, Province, C01-C20
│
├── outputs/
│   ├── figures/            # PNG charts (300 DPI)
│   ├── results/            # CSV files
│   ├── reports/            # Analysis reports
│   └── logs/               # debug.log
│
├── tests/
│   └── test_core.py
│
└── docs/
    ├── METHODS.md          # Methods summary
    └── WORKFLOW.md         # Pipeline workflow
```