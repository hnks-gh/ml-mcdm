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
- WeightingConfig          — Hybrid MC Ensemble + Bayesian Bootstrap
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
    CRITIC = "critic"
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
        """Root of the data/ directory (codebook, csv/ sub-dir)."""
        return self.base_dir / "data"

    @property
    def data_csv_dir(self) -> Path:
        """Input CSV files directory (``data/csv/YYYY.csv``)."""
        return self.data_dir / "csv"

    @property
    def output_dir(self) -> Path:
        return self.base_dir / "output" / "result"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def csv_dir(self) -> Path:
        return self.output_dir / "csv"

    # Keep old name as alias for backward compatibility
    @property
    def results_dir(self) -> Path:
        return self.csv_dir

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    def ensure_directories(self) -> None:
        """Create every output directory if missing."""
        phases = ["weighting", "ranking", "mcdm", "forecasting", "sensitivity", "summary"]
        dirs = [
            self.data_dir, self.output_dir,
            self.reports_dir, self.logs_dir,
        ]
        # figures phase subdirs
        for phase in phases:
            dirs.append(self.figures_dir / phase)
        # csv phase subdirs
        for phase in phases:
            dirs.append(self.csv_dir / phase)
        for d in dirs:
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
    n_subcriteria: int = 29
    n_criteria: int = 8
    subcriteria_prefix: str = "SC"
    criteria_prefix: str = "C"

    @property
    def n_years(self) -> int:
        return len(self.years)

    @property
    def n_observations(self) -> int:
        return self.n_provinces * self.n_years

    # Number of sub-criteria per criterion, in C01..C08 order.
    # C01:4, C02:4, C03:3, C04:4, C05:4, C06:4, C07:3, C08:3  → 29 total
    _subcriteria_per_criterion: List[int] = field(
        default_factory=lambda: [4, 4, 3, 4, 4, 4, 3, 3],
        init=False,
        repr=False,
    )

    @property
    def subcriteria_cols(self) -> List[str]:
        """Return SC codes in dataset order: SC11–SC14, SC21–SC24, …, SC81–SC83."""
        codes: List[str] = []
        for crit_idx, n_sub in enumerate(self._subcriteria_per_criterion, start=1):
            for sub_idx in range(1, n_sub + 1):
                codes.append(f"{self.subcriteria_prefix}{crit_idx}{sub_idx}")
        return codes

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


# =========================================================================
# Weighting Configuration
# =========================================================================

@dataclass
class WeightingConfig:
    """CRITIC Weighting configuration.

    Two-level deterministic design
    ──────────────────────────────
    Level 1 : CRITIC on each criterion group (m × n_k matrices)
              → local SC weights summing to 1 within each group
    Level 2 : CRITIC on criterion composite matrix
              → criterion weights summing to 1 globally
    Global  : global_SC_weight = local_SC_weight × criterion_weight

    Fully deterministic — no Monte Carlo, no Beta blending, no tuning grid.
    Temporal stability analysis is handled separately in ``analysis/``.
    """
    # ── Numerics ────────────────────────────────────────────────────────
    epsilon: float = 1e-10

    # ── Stability (analytical, not MC-based) ────────────────────────────
    stability_threshold:     float = 0.95
    perform_stability_check: bool  = True


# =========================================================================
# Evidential Reasoning Configuration
# =========================================================================

@dataclass
class EvidentialReasoningConfig:
    """Two-stage ER aggregation (Yang & Xu, 2002).

    Stage 1 — Within each criterion, combine 6 traditional MCDM method scores.
    Stage 2 — Combine 8 criterion beliefs with criterion weights.
    """
    n_grades: int = 5
    method_weight_scheme: Literal["equal", "rank_based"] = "equal"
    base_methods: List[str] = field(default_factory=lambda: [
        # Traditional
        "topsis", "vikor", "promethee", "copras", "edas", "saw",
    ])


# =========================================================================
# ML Forecasting (State-of-the-Art Ensemble)
# =========================================================================

@dataclass
class ForecastConfig:
    """
    State-of-the-art forecasting configuration for UnifiedForecaster.

    Uses 5 diverse base models + Super Learner meta-ensemble + Conformal
    Prediction calibration.  Optimised for small-to-medium panel data (N < 1000).

    Model hyperparameters
    ---------------------
    gb_max_depth / gb_n_estimators
        GradientBoostingForecaster (Huber-loss sequential trees).
        ``max_depth=5`` is the principled midpoint between the underfitting
        depth-4 and the overfitting depth-6 at n≈756: depth-5 yields
        32 leaves ≈ 24 samples/leaf, providing a healthy bias-variance
        trade-off.  ``n_estimators=200`` aligns the ensemble build with
        the standalone class default (previously hardcoded to 100 in
        ``_create_models``).

    nam_n_basis / nam_n_iterations
        NeuralAdditiveForecaster Random Fourier Feature (RFF) basis count
        and backfitting iterations.  With 60 PCA components in the reduced
        space, a basis of 10 yields only 20 effective parameters per shape
        function; 30 gives 60 effective parameters (parity with the PCA
        dimensionality) while remaining computationally fast.

    pvar_lag_selection_method
        PanelVARForecaster lag-order selection.  The only valid value is
        ``"cv"`` (hold-out CV MSE).  Passing ``"bic"`` or ``"aic"`` raises
        a ``DeprecationWarning`` inside ``PanelVARForecaster`` and maps
        silently to ``"cv"``; classic penalised-likelihood criteria are
        invalid under Ridge regularisation because the effective
        degrees-of-freedom ``tr(X(X'X+λI)⁻¹X') ≪ raw parameter count``.
    """
    enabled: bool = True
    target_year: Optional[int] = None  # Auto-set to latest_year + 1

    # ── Conformal prediction ─────────────────────────────────────────────
    conformal_method: str = 'cv_plus'  # 'split', 'cv_plus', 'adaptive'
    conformal_alpha: float = 0.05      # 95% joint coverage (Bonferroni across components)

    # ── Cross-validation ─────────────────────────────────────────────────
    cv_folds: int = 3

    # ── Reproducibility ──────────────────────────────────────────────────
    random_state: int = 42
    verbose: bool = True

    # ── GradientBoosting hyperparameters ─────────────────────────────────
    gb_max_depth: int = 5
    """Tree depth. 5 ≈ 32 leaves → ~24 samples/leaf at n=756."""
    gb_n_estimators: int = 200
    """Number of boosting stages; aligned with GradientBoostingForecaster default."""

    # ── Forecast target level ─────────────────────────────────────────────
    forecast_level: str = "criteria"
    """Forecast target granularity.
    'criteria'    — predict 8 aggregated criterion composites (C01–C08).
    'subcriteria' — predict all 29 raw sub-criterion values (SC11–SC83).
    """

    # ── NeuralAdditiveModel hyperparameters ──────────────────────────────
    nam_n_basis: int = 30
    """RFF basis functions per shape function (30→60 effective params in 60-dim PCA space)."""
    nam_n_iterations: int = 10
    """Backfitting passes; 5 is insufficient for 60 PCA dimensions."""

    # ── PanelVAR lag selection ────────────────────────────────────────────
    pvar_lag_selection_method: str = "cv"
    """Hold-out CV MSE lag selection. Only valid method for Ridge-regularised VAR."""


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
# Ranking Configuration
# =========================================================================

@dataclass
class RankingConfig:
    """
    Hierarchical ranking configuration.

    run_all_years
        When True, the ranking pipeline is executed for every year in the
        panel (not just the target year) to enable temporal charts.
        Execution is parallelised via ThreadPoolExecutor.

    max_parallel_years
        Maximum number of year-ranking threads to run concurrently.
        Defaults to min(n_years, cpu_count).  Set to 1 to force sequential
        execution (debugging / memory-constrained environments).

    expose_mc_province_stats
        When True, per-province Monte Carlo rank distribution statistics
        (mean_rank, std_rank, prob_top1, prob_topK) are carried through
        the details dict so CSV writers can persist them.
    """
    run_all_years: bool = True
    max_parallel_years: Optional[int] = None   # None → auto (cpu_count)
    expose_mc_province_stats: bool = True


# =========================================================================
# Visualisation
# =========================================================================

@dataclass
class VisualizationConfig:
    """Figure appearance defaults."""
    figsize: tuple = field(default_factory=lambda: (12, 8))
    dpi: int = 300
    style: str = "seaborn-v0_8-whitegrid"
    palette: str = "viridis"
    heatmap_cmap: str = "RdYlGn"
    diverging_cmap: str = "RdBu_r"
    save_formats: List[str] = field(default_factory=lambda: ["png"])
    # Ranking chart options
    ranking_top_n: int = 20      # how many provinces to show in top-N ranking charts
    ranking_top_n_options: List[int] = field(
        default_factory=lambda: [5, 10, 15, 20]
    )   # additional cut-offs to generate separate charts for


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

    # Weighting
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    er: EvidentialReasoningConfig = field(default_factory=EvidentialReasoningConfig)

    # Ranking
    ranking: RankingConfig = field(default_factory=RankingConfig)

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
                d = {k: _cvt(v) for k, v in obj.__dict__.items()}
                # Include computed properties that are not stored in __dict__
                if isinstance(obj, PanelDataConfig):
                    d['subcriteria_cols'] = obj.subcriteria_cols
                    d['criteria_cols'] = obj.criteria_cols
                    d['train_years'] = obj.train_years
                    d['test_year'] = obj.test_year
                    d['n_observations'] = obj.n_observations
                return d
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
            f"    Subcriteria     : {self.panel.n_subcriteria}  (SC11-SC83)\n"
            f"    Criteria        : {self.panel.n_criteria}  (C01-C08)\n"
            f"    Years           : {self.panel.years[0]}-{self.panel.years[-1]}"
            f"  ({self.panel.n_years} years)\n"
            f"    Observations    : {self.panel.n_observations}\n\n"
            f"  MCDM METHODS\n"
            f"    TOPSIS norm     : {self.topsis.normalization.value}\n"
            f"    VIKOR v         : {self.vikor.v}\n\n"
            f"  WEIGHTING\n"
            f"    Strategy        : Deterministic CRITIC\n"
            f"    MC simulations  : {self.weighting.mc_n_simulations}\n"
            f"    Stability thr   : {self.weighting.stability_threshold}\n\n"
            f"  RANKING\n"
            f"    Run all years   : {self.ranking.run_all_years}\n"
            f"    Max workers     : {self.ranking.max_parallel_years or 'auto'}\n\n"
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
