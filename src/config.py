# -*- coding: utf-8 -*-
"""Configuration management for ML-MCDM pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Literal
from enum import Enum
import json


class NormalizationType(Enum):
    """Supported normalization methods."""
    VECTOR = "vector"
    MINMAX = "minmax"
    ZSCORE = "zscore"
    MAX = "max"


class WeightMethod(Enum):
    """Supported weighting methods."""
    ENTROPY = "entropy"
    CRITIC = "critic"
    ENSEMBLE = "ensemble"
    EQUAL = "equal"


class AggregationType(Enum):
    """Supported rank aggregation methods."""
    BORDA = "borda"
    COPELAND = "copeland"
    KEMENY = "kemeny"
    STACKING = "stacking"


@dataclass
class PathConfig:
    """File and directory paths configuration."""
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    
    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"
    
    @property
    def output_dir(self) -> Path:
        return self.base_dir / "outputs"
    
    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"
    
    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"
    
    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"
    
    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"
    
    @property
    def data_file(self) -> Path:
        return self.data_dir / "data.csv"
    
    def ensure_directories(self) -> None:
        """Create all necessary directories."""
        for d in [self.data_dir, self.output_dir, self.figures_dir, 
                  self.reports_dir, self.results_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class PanelDataConfig:
    """Panel data structure configuration."""
    n_provinces: int = 64
    n_components: int = 20
    years: List[int] = field(default_factory=lambda: [2020, 2021, 2022, 2023, 2024])
    province_col: str = "Province"
    year_col: str = "Year"
    component_prefix: str = "C"
    
    @property
    def n_years(self) -> int:
        return len(self.years)
    
    @property
    def n_observations(self) -> int:
        return self.n_provinces * self.n_years
    
    @property
    def component_cols(self) -> List[str]:
        return [f"{self.component_prefix}{i+1:02d}" for i in range(self.n_components)]
    
    @property
    def train_years(self) -> List[int]:
        return self.years[:-1]
    
    @property
    def test_year(self) -> int:
        return self.years[-1]


@dataclass
class RandomConfig:
    """Random state configuration for reproducibility."""
    seed: int = 42
    n_bootstrap: int = 1000
    cv_folds: int = 5


@dataclass
class TOPSISConfig:
    """TOPSIS method configuration."""
    normalization: NormalizationType = NormalizationType.VECTOR
    weight_method: WeightMethod = WeightMethod.ENSEMBLE
    benefit_criteria: Optional[List[str]] = None
    cost_criteria: Optional[List[str]] = None
    temporal_discount: float = 0.9
    trajectory_weight: float = 0.3
    stability_weight: float = 0.2


@dataclass
class VIKORConfig:
    """VIKOR method configuration."""
    v: float = 0.5
    v_sensitivity: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    acceptance_threshold: float = 0.25


@dataclass
class FuzzyTOPSISConfig:
    """Fuzzy TOPSIS configuration."""
    use_temporal_variance: bool = True
    alpha_cut: float = 0.0
    spread_factor: float = 1.0
    linguistic_scales: Dict[str, tuple] = field(default_factory=lambda: {
        "VL": (0.0, 0.0, 0.2), "L": (0.0, 0.2, 0.4),
        "M": (0.2, 0.5, 0.8), "H": (0.6, 0.8, 1.0), "VH": (0.8, 1.0, 1.0)
    })


@dataclass
class PanelRegressionConfig:
    """Panel regression configuration."""
    model_type: Literal["fe", "re", "pooled"] = "fe"
    robust_se: bool = True
    time_effects: bool = True
    hausman_test: bool = True


@dataclass
class RandomForestConfig:
    """Random Forest configuration with time-series CV."""
    n_estimators: int = 200
    max_depth: Optional[int] = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    n_splits: int = 2  # Reduced from 3 to work with 5-year panels after lag features
    gap: int = 0
    use_lags: bool = True
    n_lags: int = 2
    use_rolling_features: bool = True
    rolling_window: int = 2


@dataclass
class NeuralConfig:
    """
    Neural network configuration (MLP and Attention-based forecasters).
    
    Note: This configures the MLP/Attention neural network forecasters,
    NOT true LSTM/RNN models. Neural networks are DISABLED by default
    because the current panel data (5 years, 64 provinces) is insufficient
    for effective neural network training.
    
    The neural forecasting uses:
    - MLP (Multi-Layer Perceptron) with modern architecture
    - Attention-based networks for temporal weighting
    
    Enable only if you have sufficient data (100+ samples recommended).
    """
    enabled: bool = False  # Disabled - insufficient data for neural networks
    hidden_units: int = 64
    n_layers: int = 2
    dropout: float = 0.2
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    patience: int = 15
    attention: bool = False


@dataclass
class EnsembleConfig:
    """Ensemble and meta-learning configuration."""
    # All 10 MCDM methods (5 traditional + 5 fuzzy)
    base_methods: List[str] = field(default_factory=lambda: [
        # Traditional MCDM
        "topsis", "dynamic_topsis", "vikor", "promethee", "copras", "edas",
        # Fuzzy MCDM
        "fuzzy_topsis", "fuzzy_vikor", "fuzzy_promethee", "fuzzy_copras", "fuzzy_edas"
    ])
    meta_learner: Literal["ridge", "bayesian", "xgboost"] = "ridge"
    alpha: float = 1.0  # Regularization strength for ridge
    meta_cv_folds: int = 5
    aggregation_method: AggregationType = AggregationType.BORDA
    method_weights: Optional[Dict[str, float]] = None


@dataclass
class ConvergenceConfig:
    """Convergence analysis configuration."""
    beta_convergence: bool = True
    conditional_vars: Optional[List[str]] = None
    sigma_convergence: bool = True
    club_convergence: bool = True
    n_clubs: int = 4
    markov_chains: bool = True
    n_quantiles: int = 4


@dataclass
class ValidationConfig:
    """Validation and robustness configuration."""
    n_bootstrap: int = 1000
    n_simulations: int = 1000  # Alias for n_bootstrap
    bootstrap_ci: float = 0.95
    weight_perturbation: float = 0.1
    n_sensitivity_scenarios: int = 100
    drop_one_year: bool = True
    drop_one_component: bool = True
    alternative_weights: bool = True
    rank_correlation_threshold: float = 0.85


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    figsize: tuple = (12, 8)
    dpi: int = 300
    style: str = "seaborn-v0_8-whitegrid"
    palette: str = "viridis"
    heatmap_cmap: str = "RdYlGn"
    diverging_cmap: str = "RdBu_r"
    save_formats: List[str] = field(default_factory=lambda: ["png"])
    animate_temporal: bool = True
    animation_fps: int = 2


@dataclass
class Config:
    """Master configuration combining all sub-configurations."""
    paths: PathConfig = field(default_factory=PathConfig)
    panel: PanelDataConfig = field(default_factory=PanelDataConfig)
    random: RandomConfig = field(default_factory=RandomConfig)
    topsis: TOPSISConfig = field(default_factory=TOPSISConfig)
    vikor: VIKORConfig = field(default_factory=VIKORConfig)
    fuzzy: FuzzyTOPSISConfig = field(default_factory=FuzzyTOPSISConfig)
    panel_regression: PanelRegressionConfig = field(default_factory=PanelRegressionConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    def __post_init__(self):
        self.paths.ensure_directories()
    
    @property
    def output_dir(self) -> str:
        """Get output directory path as string."""
        return str(self.paths.output_dir)
    
    @property
    def n_simulations(self) -> int:
        """Alias for validation n_bootstrap."""
        return self.validation.n_bootstrap
    
    def to_dict(self) -> Dict:
        def _to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: _to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, list):
                return [_to_dict(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            return obj
        return _to_dict(self)
    
    def save(self, filepath: Path) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def summary(self) -> str:
        return f"""
{'='*80}
CONFIGURATION SUMMARY - Panel Data Econometric-ML Hybrid Framework
{'='*80}

PANEL DATA:
  Provinces: {self.panel.n_provinces}
  Components: {self.panel.n_components}  
  Years: {self.panel.years}
  Total observations: {self.panel.n_observations}

MCDM METHODS:
  TOPSIS normalization: {self.topsis.normalization.value}
  TOPSIS weights: {self.topsis.weight_method.value}
  VIKOR v parameter: {self.vikor.v}
  Fuzzy temporal variance: {self.fuzzy.use_temporal_variance}

ML METHODS:
  Random Forest estimators: {self.random_forest.n_estimators}
  Neural hidden units: {self.neural.hidden_units} (disabled by default)
  Neural epochs: {self.neural.epochs}

ENSEMBLE:
  Base methods: {len(self.ensemble.base_methods)}
  Meta-learner: {self.ensemble.meta_learner}
  Aggregation: {self.ensemble.aggregation_method.value}

VALIDATION:
  Bootstrap iterations: {self.validation.n_bootstrap}
  Sensitivity scenarios: {self.validation.n_sensitivity_scenarios}
{'='*80}
"""


_config: Optional[Config] = None

def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config

def get_default_config() -> Config:
    """Get a fresh default configuration."""
    return Config()

def set_config(config: Config) -> None:
    global _config
    _config = config

def reset_config() -> None:
    global _config
    _config = Config()
