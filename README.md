# ML-MCDM: Machine Learning Enhanced Multi-Criteria Decision Making Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive framework combining **Multi-Criteria Decision Making (MCDM)** methods with **Machine Learning** for panel data analysis and sustainability assessment.

## Overview

ML-MCDM provides an integrated econometric-ML hybrid approach for multi-criteria decision making with panel data. The framework is designed for analyzing entities (e.g., provinces, companies, countries) across multiple time periods and criteria.

### Key Capabilities

- **Multiple MCDM Methods**: TOPSIS, Dynamic TOPSIS, VIKOR, Fuzzy TOPSIS
- **Objective Weighting**: Entropy weights, CRITIC weights, Ensemble weighting
- **Machine Learning**: Panel Regression (FE/RE), Random Forest with time-series CV, LSTM forecasting, Rough Set attribute reduction
- **Ensemble Methods**: Stacking meta-learning, Borda Count, Copeland rank aggregation
- **Advanced Analysis**: Beta/Sigma convergence analysis, Monte Carlo sensitivity analysis
- **Comprehensive Outputs**: High-resolution visualizations, CSV results, detailed reports

## Features

- ✅ Panel data support (entities × time periods × criteria)
- ✅ Multiple objective weighting methods (Entropy, CRITIC, Ensemble)
- ✅ Four MCDM methods with automatic rank aggregation
- ✅ ML-based validation and feature importance analysis
- ✅ Time-series forecasting with LSTM neural networks
- ✅ Dimensionality reduction with Rough Set Theory
- ✅ Convergence and sensitivity analysis
- ✅ Professional high-resolution visualizations (300 DPI)
- ✅ Automated comprehensive report generation
- ✅ Production-ready with full error handling

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/your-repo/ML-MCDM.git
cd ML-MCDM

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install package
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Project Structure

```
ML-MCDM/
├── main.py                 # Entry point - run this to start analysis
├── pyproject.toml          # Package configuration
├── requirements.txt        # Dependencies
├── README.md               # This file
│
├── src/                    # Source code
│   ├── main.py             # Pipeline orchestrator (MLTOPSISPipeline)
│   ├── config.py           # Configuration dataclasses
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── logger.py           # Logging utilities
│   ├── output_manager.py   # Output file management
│   ├── visualization.py    # Visualization generators
│   │
│   ├── mcdm/               # Multi-Criteria Decision Making methods
│   │   ├── topsis.py       # TOPSIS and Dynamic TOPSIS
│   │   ├── vikor.py        # VIKOR method
│   │   ├── fuzzy_topsis.py # Fuzzy TOPSIS with triangular numbers
│   │   └── weights.py      # Entropy, CRITIC, Ensemble weights
│   │
│   ├── ml/                 # Machine Learning methods
│   │   ├── panel_regression.py  # Fixed/Random Effects regression
│   │   ├── random_forest_ts.py  # Random Forest with TS cross-validation
│   │   ├── lstm_forecast.py     # LSTM neural network forecasting
│   │   └── rough_sets.py        # Rough Set attribute reduction
│   │
│   ├── ensemble/           # Ensemble and aggregation methods
│   │   ├── stacking.py     # Stacking meta-learner
│   │   └── aggregation.py  # Borda Count, Copeland aggregation
│   │
│   └── analysis/           # Analysis tools
│       ├── convergence.py  # Beta/Sigma convergence analysis
│       ├── sensitivity.py  # Monte Carlo sensitivity analysis
│       └── validation.py   # Cross-validation utilities
│
├── data/                   # Data files
│   └── data.csv            # Panel data (Year, Province, C01-C20)
│
├── docs/                   # Documentation
│   ├── METHODS.md          # Detailed method descriptions
│   └── WORKFLOW.md         # Step-by-step workflow guide
│
├── tests/                  # Test suite
│   └── test_core.py        # Core functionality tests
│
└── outputs/                # Generated outputs
    ├── figures/            # High-resolution charts (300 DPI)
    ├── results/            # CSV numerical results
    ├── reports/            # Analysis reports
    └── logs/               # Execution logs
```

## Quick Start

### Basic Usage

Simply run the main script:

```bash
python main.py
```

This will:
1. Generate synthetic panel data (64 entities × 5 years × 20 criteria)
2. Run the complete analysis pipeline
3. Save results to `outputs/` directory

### Using Custom Data

```bash
python main.py data/data.csv
```

### Configuration

Edit the `CONFIG` dictionary in `main.py`:

```python
CONFIG = {
    'data_path': None,           # Path to CSV or None for synthetic data
    'n_provinces': 64,           # Number of entities
    'n_years': 5,                # Number of time periods
    'n_components': 20,          # Number of criteria/components
    'output_dir': 'outputs',     # Output directory
}
```

### Python API

```python
from src import MLTOPSISPipeline, get_default_config

# Configure
config = get_default_config()
config.panel.n_provinces = 30
config.panel.years = [2020, 2021, 2022, 2023, 2024]

# Run pipeline
pipeline = MLTOPSISPipeline(config)
result = pipeline.run('data/data.csv')

# Access results
print(f"Top entity: {result.panel_data.entities[result.topsis_rankings.argmin()]}")
print(f"Kendall's W: {result.aggregated_ranking.kendall_w:.4f}")
print(f"Execution time: {result.execution_time:.2f}s")

# Get ranking DataFrame
rankings = result.get_final_ranking_df()
print(rankings.head(10))
```

## Input Data Format

Panel data CSV with the following structure:

```csv
Year,Province,C01,C02,C03,...,C20
2020,P01,0.75,0.82,0.68,...,0.71
2020,P02,0.62,0.74,0.81,...,0.65
2021,P01,0.78,0.85,0.70,...,0.73
...
```

| Column | Description |
|--------|-------------|
| `Year` | Time period identifier (integer) |
| `Province` | Entity identifier (string) |
| `C01-C20` | Criteria/component values (0-1 normalized recommended) |

## Output Files

### Results (`outputs/results/`)

| File | Description |
|------|-------------|
| `final_rankings.csv` | Aggregated entity rankings with all scores |
| `weights_analysis.csv` | Criterion weights (Entropy, CRITIC, Ensemble) |
| `mcdm_scores_detailed.csv` | Detailed scores from all MCDM methods |
| `feature_importance.csv` | Random Forest feature importance |
| `sensitivity_analysis.csv` | Weight sensitivity indices |
| `beta_convergence.csv` | Beta convergence analysis results |
| `sigma_convergence.csv` | Sigma convergence by year |

### Figures (`outputs/figures/`)

- Score evolution over time
- MCDM method comparison
- Weight distribution charts
- Convergence analysis plots
- Sensitivity heatmaps
- Feature importance visualization

### Reports (`outputs/reports/`)

- `analysis_report.txt` - Comprehensive analysis summary

## Documentation

For detailed information about the methods and workflow, see:

- [**METHODS.md**](docs/METHODS.md) - Detailed description of all methods (TOPSIS, VIKOR, LSTM, etc.)
- [**WORKFLOW.md**](docs/WORKFLOW.md) - Step-by-step workflow and pipeline phases

## Pipeline Phases

| Phase | Description |
|-------|-------------|
| 1. Data Loading | Load/generate panel data, create cross-sectional views |
| 2. Weight Calculation | Compute Entropy, CRITIC, and Ensemble weights |
| 3. MCDM Analysis | Run TOPSIS, Dynamic TOPSIS, VIKOR, Fuzzy TOPSIS |
| 4. ML Analysis | Panel regression, Random Forest, LSTM, Rough Sets |
| 5. Ensemble Integration | Stacking meta-learner, rank aggregation |
| 6. Advanced Analysis | Convergence and sensitivity analysis |
| 7. Visualization | Generate all high-resolution figures |
| 8. Output | Save results, reports, and manifest |

## Requirements

### Core Dependencies

```
numpy >= 1.21.0
pandas >= 1.3.0
scipy >= 1.7.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

### Optional Dependencies

```
shap >= 0.40.0          # For SHAP explainability
factor-analyzer >= 0.4.0 # For factor analysis
```

## Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{ml_mcdm,
  title={ML-MCDM: Machine Learning Enhanced Multi-Criteria Decision Making Framework},
  author={ML-MCDM Development Team},
  year={2024},
  url={https://github.com/your-repo/ML-MCDM}
}
```

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## Acknowledgments

This framework builds upon established MCDM theory and modern machine learning techniques for comprehensive decision support analysis.
