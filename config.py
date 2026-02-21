# -*- coding: utf-8 -*-
"""
Centralised Configuration for ML-MCDM Pipeline
================================================

All configurable parameters are defined here as typed dataclasses.
The master ``Config`` class composes every sub-config and provides
serialisation, summary printing, and global singleton management.

Configuration Groups
--------------------
- PathConfig               — directory structure
- PanelDataConfig          — entity / time / hierarchy dimensions
- RandomConfig             — reproducibility seeds
- TOPSISConfig             — TOPSIS method parameters
- VIKORConfig              — VIKOR compromise parameter
- IFSConfig                — Intuitionistic Fuzzy Set parameters
- WeightingConfig          — GTWC + Bayesian Bootstrap
- EvidentialReasoningConfig— two-stage ER aggregation
- ForecastConfig           — ML forecasting ensemble settings
- ValidationConfig         — sensitivity / robustness analysis
- VisualizationConfig      — figure appearance defaults
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Literal
from enum import Enum
import json


# =========================================================================
# Enumerations
# =========================================================================

class NormalizationType(Enum):
    """Supported normalization methods for traditional MCDM."""
    VECTOR = "vector"
    MINMAX = "minmax"
    ZSCORE = "zscore"
    MAX = "max"


class WeightMethod(Enum):
    """Supported weighting method families."""
    ENTROPY = "entropy"
    CRITIC = "critic"
    MEREC = "merec"
    STD_DEV = "std_dev"
    ENSEMBLE = "ensemble"
    EQUAL = "equal"


class AggregationType(Enum):
    """Supported global rank aggregation."""
    EVIDENTIAL_REASONING = "evidential_reasoning"


# =========================================================================
# Path Configuration
# =========================================================================

@dataclass
class PathConfig:
    """File and directory paths, all derived from *base_dir*."""
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

    def ensure_directories(self) -> None:
        """Create every output directory if missing."""
        for d in [self.data_dir, self.output_dir, self.figures_dir,
                  self.reports_dir, self.results_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)


# =========================================================================
# Data Configuration
# =========================================================================

@dataclass
class PanelDataConfig:
    """Panel data dimensions and naming conventions."""
    n_provinces: int = 63
    years: List[int] = field(default_factory=lambda: list(range(2011, 2025)))
    province_col: str = "Province"
    year_col: str = "Year"

    # Hierarchy
    n_subcriteria: int = 28
    n_criteria: int = 8
    subcriteria_prefix: str = "SC"
    criteria_prefix: str = "C"

    @property
    def n_years(self) -> int:
        return len(self.years)

    @property
    def n_observations(self) -> int:
        return self.n_provinces * self.n_years

    @property
    def subcriteria_cols(self) -> List[str]:
        return [f"{self.subcriteria_prefix}{i+1:02d}"
                for i in range(self.n_subcriteria)]

    @property
    def criteria_cols(self) -> List[str]:
        return [f"{self.criteria_prefix}{i+1:02d}"
                for i in range(self.n_criteria)]

    @property
    def train_years(self) -> List[int]:
        return self.years[:-1]

    @property
    def test_year(self) -> int:
        return self.years[-1]


# =========================================================================
# Reproducibility
# =========================================================================

@dataclass
class RandomConfig:
    """Random-state and resampling defaults."""
    seed: int = 42


# =========================================================================
# MCDM Method Parameters
# =========================================================================

@dataclass
class TOPSISConfig:
    """TOPSIS configuration."""
    normalization: NormalizationType = NormalizationType.VECTOR
    benefit_criteria: Optional[List[str]] = None   # None → all benefit
    cost_criteria: Optional[List[str]] = None


@dataclass
class VIKORConfig:
    """VIKOR compromise parameter."""
    v: float = 0.5


@dataclass
class IFSConfig:
    """Intuitionistic Fuzzy Set configuration (Atanassov, 1986).

    Parameters
    ----------
    spread_factor : float
        Controls mapping of temporal σ to IFS hesitancy π.
    n_grades : int
        Linguistic assessment grades for ER belief distributions.
    use_temporal_variance : bool
        Derive hesitancy from inter-annual variance.
    """
    spread_factor: float = 1.0
    n_grades: int = 5
    use_temporal_variance: bool = True


# =========================================================================
# Weighting Configuration
# =========================================================================

@dataclass
class WeightingConfig:
    """Robust Global Hybrid Weighting (GTWC) configuration.

    Pipeline steps
    --------------
    1. Global Min–Max Normalisation
    2. Entropy / CRITIC / MEREC / Std-Dev weight vectors
    3. Game-Theory Weight Combination (cooperative Nash equilibrium)
    4. Bayesian Bootstrap uncertainty quantification
    5. Split-half temporal stability verification
    """
    bootstrap_iterations: int = 199
    stability_threshold: float = 0.95
    epsilon: float = 1e-10


# =========================================================================
# Evidential Reasoning Configuration
# =========================================================================

@dataclass
class EvidentialReasoningConfig:
    """Two-stage ER aggregation (Yang & Xu, 2002).

    Stage 1 — Within each criterion, combine 12 MCDM method scores.
    Stage 2 — Combine 8 criterion beliefs with criterion weights.
    """
    n_grades: int = 5
    method_weight_scheme: Literal["equal", "rank_based"] = "equal"
    base_methods: List[str] = field(default_factory=lambda: [
        # Traditional
        "topsis", "vikor", "promethee", "copras", "edas", "saw",
        # IFS
        "ifs_topsis", "ifs_vikor", "ifs_promethee",
        "ifs_copras", "ifs_edas", "ifs_saw",
    ])


# =========================================================================
# ML Forecasting (State-of-the-Art Ensemble)
# =========================================================================

@dataclass
class ForecastConfig:
    """
    State-of-the-art forecasting configuration for UnifiedForecaster.
    
    Uses 6 diverse models + Super Learner + Conformal Prediction.
    Optimized for small-to-medium panel data (N < 1000).
    """
    enabled: bool = True
    target_year: Optional[int] = None  # Auto-set to latest_year + 1
    
    # Conformal prediction settings
    conformal_method: str = 'CQR'  # 'split', 'cv_plus', 'CQR'
    conformal_alpha: float = 0.05  # 95% coverage
    
    # Cross-validation
    cv_folds: int = 3
    
    # Ensemble settings (6 models fixed)
    random_state: int = 42
    verbose: bool = True


# =========================================================================
# Validation & Sensitivity
# =========================================================================

@dataclass
class ValidationConfig:
    """Sensitivity analysis and robustness testing."""
    n_simulations: int = 1000
    weight_perturbation: float = 0.1
    n_sensitivity_scenarios: int = 100
    rank_correlation_threshold: float = 0.85


# =========================================================================
# Visualisation
# =========================================================================

@dataclass
class VisualizationConfig:
    """Figure appearance defaults."""
    figsize: tuple = (12, 8)
    dpi: int = 300
    style: str = "seaborn-v0_8-whitegrid"
    palette: str = "viridis"
    heatmap_cmap: str = "RdYlGn"
    diverging_cmap: str = "RdBu_r"
    save_formats: List[str] = field(default_factory=lambda: ["png"])


# =========================================================================
# Master Configuration
# =========================================================================

@dataclass
class Config:
    """Master configuration composing every sub-config."""
    paths: PathConfig = field(default_factory=PathConfig)
    panel: PanelDataConfig = field(default_factory=PanelDataConfig)
    random: RandomConfig = field(default_factory=RandomConfig)

    # MCDM
    topsis: TOPSISConfig = field(default_factory=TOPSISConfig)
    vikor: VIKORConfig = field(default_factory=VIKORConfig)
    ifs: IFSConfig = field(default_factory=IFSConfig)

    # Weighting
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    er: EvidentialReasoningConfig = field(default_factory=EvidentialReasoningConfig)

    # ML Forecasting
    forecast: ForecastConfig = field(default_factory=ForecastConfig)

    # Analysis
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self):
        self.paths.ensure_directories()

    # --- convenience properties ---

    @property
    def output_dir(self) -> str:
        return str(self.paths.output_dir)

    # --- serialisation ---

    def to_dict(self) -> Dict:
        def _cvt(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: _cvt(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, (list, tuple)):
                return [_cvt(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _cvt(v) for k, v in obj.items()}
            return obj
        return _cvt(self)

    def save(self, filepath: Path) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        return (
            f"\n{'='*72}\n"
            f"  ML-MCDM Configuration Summary\n"
            f"{'='*72}\n\n"
            f"  PANEL DATA\n"
            f"    Provinces       : {self.panel.n_provinces}\n"
            f"    Subcriteria     : {self.panel.n_subcriteria}  (SC01-SC28)\n"
            f"    Criteria        : {self.panel.n_criteria}  (C01-C08)\n"
            f"    Years           : {self.panel.years[0]}-{self.panel.years[-1]}"
            f"  ({self.panel.n_years} years)\n"
            f"    Observations    : {self.panel.n_observations}\n\n"
            f"  MCDM METHODS\n"
            f"    TOPSIS norm     : {self.topsis.normalization.value}\n"
            f"    VIKOR v         : {self.vikor.v}\n"
            f"    IFS spread      : {self.ifs.spread_factor}\n"
            f"    IFS grades      : {self.ifs.n_grades}\n\n"
            f"  WEIGHTING\n"
            f"    Strategy        : GTWC (Entropy+CRITIC+MEREC+SD)\n"
            f"    Bootstrap iters : {self.weighting.bootstrap_iterations}\n"
            f"    Stability thr   : {self.weighting.stability_threshold}\n\n"
            f"  EVIDENTIAL REASONING\n"
            f"    Base methods    : {len(self.er.base_methods)}\n"
            f"    Method weights  : {self.er.method_weight_scheme}\n"
            f"    Grades          : {self.er.n_grades}\n\n"
            f"  VALIDATION\n"
            f"    Sensitivity sim : {self.validation.n_simulations}\n"
            f"    Weight perturb  : {self.validation.weight_perturbation}\n"
            f"{'='*72}\n"
        )


# =========================================================================
# Global Config Singleton
# =========================================================================

_config: Optional[Config] = None


def get_config() -> Config:
    """Return global config (create default on first call)."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def get_default_config() -> Config:
    """Return a *fresh* default Config instance."""
    return Config()


def set_config(config: Config) -> None:
    """Replace the global config singleton."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset to a fresh default Config."""
    global _config
    _config = Config()
