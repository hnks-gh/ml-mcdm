# -*- coding: utf-8 -*-
"""Forecast Visualization Figure Manifest

Registry of all forecast figures with metadata, dependencies, and conditional
execution rules.

Design principle:
- Each figure has an ID (F-01, F-02, ..., F-22)
- Each figure declares required and optional input fields
- Orchestrator uses manifest to determine execution order and skip conditions
- Manifest enables extensibility and conditional generation
"""

from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Callable, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FigureCategory(Enum):
    """Figure categorization for organization and filtering."""
    ACCURACY = "Predictive Accuracy and Error Structure"
    ENSEMBLE = "Ensemble Composition and Model Comparison"
    UNCERTAINTY = "Uncertainty Quantification and Calibration"
    IMPACT = "Forecast Impact and Business Metrics"
    INTERPRETABILITY = "Feature Importance and Model Internals"
    DIVERSITY = "Model Diversity and Prediction Analysis"
    TEMPORAL = "Temporal Reliability and Entity-Level Analysis"


@dataclass
class FigureSpec:
    """Metadata for a single forecast figure.
    
    Attributes:
        figure_id: String ID (F-01, F-02, etc.)
        title: Human readable title
        category: FigureCategory enum
        module: Python module path (e.g., 'accuracy', 'ensemble')
        method_name: Method name in chart class
        save_name: Default PNG filename
        description: Detailed description and interpretation guide
        
        # Field dependencies
        required_fields: Payload fields that must be non-None to execute
        optional_fields: Fields that enable enhanced versions of the figure
        
        # Execution control
        is_essential: True if figure is in mandatory essential suite
        enable_by_default: True if should be generated in normal runs
        conditional_skip: Optional predicate for runtime skip decision
        
        # Integration metadata
        references: Related figures
        dependencies: Other figures that should execute first
        skip_reason_code: Code logged if skipped (for debugging)
    """
    
    figure_id: str
    title: str
    category: FigureCategory
    module: str
    method_name: str
    save_name: str
    description: str
    
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    
    is_essential: bool = True
    enable_by_default: bool = True
    conditional_skip: Optional[Callable[[Any], bool]] = None
    
    references: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    skip_reason_code: str = "NOT_REQUIRED"
    
    def can_execute(self, payload: Any) -> tuple[bool, str]:
        """Check if figure can execute given payload.
        
        Returns:
            (can_execute: bool, reason: str)
        """
        # Check required fields
        for field_name in self.required_fields:
            field_value = getattr(payload, field_name, None)
            if field_value is None:
                return False, f"MISSING_REQUIRED:{field_name}"
        
        # Check conditional skip
        if self.conditional_skip is not None:
            try:
                if self.conditional_skip(payload):
                    return False, "CONDITIONAL_SKIP"
            except Exception as e:
                logger.warning(f'Conditional skip check failed for {self.figure_id}: {e}')
        
        return True, "OK"


class ForecastFigureManifest:
    """Registry and query interface for all forecast figures."""
    
    def __init__(self):
        self._figures: Dict[str, FigureSpec] = {}
        self._by_category: Dict[FigureCategory, List[str]] = {}
        self._by_module: Dict[str, List[str]] = {}
        self._initialize_manifest()
    
    def _initialize_manifest(self):
        """Define all forecast figures."""
        
        # ================== ACCURACY (F-01, F-02, F-03) ==================
        self.register(FigureSpec(
            figure_id="F-01",
            title="Actual vs Predicted Scatter",
            category=FigureCategory.ACCURACY,
            module="accuracy",
            method_name="plot_forecast_scatter",
            save_name="fig01_forecast_scatter.png",
            description="Scatter plot of actual vs predicted with identity line, fitted trend, residual coloring, and statistical annotations.",
            required_fields=["y_test", "y_pred_ensemble"],
            is_essential=True,
        ))
        
        self.register(FigureSpec(
            figure_id="F-02",
            title="Residual Diagnostics Panel",
            category=FigureCategory.ACCURACY,
            module="accuracy",
            method_name="plot_forecast_residuals",
            save_name="fig02_forecast_residuals.png",
            description="4-panel residual diagnostics: residual-vs-predicted, histogram+KDE, Q-Q plot, top errors with entity labels.",
            required_fields=["y_test", "y_pred_ensemble"],
            is_essential=True,
        ))
        
        self.register(FigureSpec(
            figure_id="F-03",
            title="Holdout Comparison",
            category=FigureCategory.ACCURACY,
            module="accuracy",
            method_name="plot_holdout_comparison",
            save_name="fig03_holdout_comparison.png",
            description="4-panel holdout comparison by model and ensemble: scatter, residual distribution, heatmap, ranking consistency.",
            required_fields=["y_test", "y_pred_ensemble"],
            optional_fields=["entity_names", "per_model_holdout_predictions"],
            is_essential=True,
        ))
        
        # ================== ENSEMBLE (F-04, F-05, F-06, F-22) ==================
        self.register(FigureSpec(
            figure_id="F-04",
            title="Model Contribution Weights",
            category=FigureCategory.ENSEMBLE,
            module="ensemble",
            method_name="plot_model_weights_donut",
            save_name="fig04_model_weights_donut.png",
            description="Donut pie chart of meta-learner weight distribution with deterministic model ordering.",
            required_fields=["model_contributions"],
            is_essential=True,
        ))
        
        self.register(FigureSpec(
            figure_id="F-05",
            title="Per-Model Performance Comparison",
            category=FigureCategory.ENSEMBLE,
            module="ensemble",
            method_name="plot_model_performance",
            save_name="fig05_model_performance.png",
            description="Grouped bar chart comparing models across shared metrics (R², RMSE, MAE) with ranking badges.",
            required_fields=["model_performance"],
            is_essential=True,
        ))
        
        self.register(FigureSpec(
            figure_id="F-06",
            title="Cross-Validation Score Distributions",
            category=FigureCategory.ENSEMBLE,
            module="ensemble",
            method_name="plot_cv_boxplots",
            save_name="fig06_cv_boxplots.png",
            description="Box plots of CV R² scores per model showing fold-wise variability and model stability rankings.",
            required_fields=["cv_scores"],
            is_essential=True,
        ))
        
        self.register(FigureSpec(
            figure_id="F-22",
            title="Ensemble Architecture Flowchart",
            category=FigureCategory.ENSEMBLE,
            module="ensemble",
            method_name="plot_ensemble_architecture",
            save_name="fig22_ensemble_architecture.png",
            description="3-tier data-driven architecture diagram: base models → meta-learner → predictions with actual model roster.",
            required_fields=["model_contributions"],
            is_essential=True,
        ))
        
        # ================== UNCERTAINTY (F-07, F-08, F-09, F-16) ==================
        self.register(FigureSpec(
            figure_id="F-07",
            title="Prediction Intervals for Entities",
            category=FigureCategory.UNCERTAINTY,
            module="uncertainty",
            method_name="plot_prediction_intervals",
            save_name="fig07_prediction_intervals.png",
            description="Time series with prediction intervals (upper/lower bands) for top-N entities with coverage annotation.",
            required_fields=["interval_lower_df", "interval_upper_df"],
            optional_fields=["entity_names"],
            is_essential=True,
        ))
        
        self.register(FigureSpec(
            figure_id="F-08",
            title="Conformal Coverage Calibration",
            category=FigureCategory.UNCERTAINTY,
            module="uncertainty",
            method_name="plot_conformal_coverage",
            save_name="fig08_conformal_coverage.png",
            description="Calibration curve showing nominal vs empirical coverage with 45° line and confidence shading.",
            required_fields=["y_test", "y_pred_ensemble", "interval_lower_df", "interval_upper_df"],
            is_essential=True,
        ))
        
        self.register(FigureSpec(
            figure_id="F-09",
            title="Interval Calibration Scatter",
            category=FigureCategory.UNCERTAINTY,
            module="uncertainty",
            method_name="plot_interval_calibration_scatter",
            save_name="fig09_interval_calibration_scatter.png",
            description="Scatter of interval width vs historical absolute error with regression line and alignment diagnostics.",
            required_fields=["y_test", "y_pred_ensemble", "interval_lower_df", "interval_upper_df"],
            optional_fields=["entity_names"],
            is_essential=True,
        ))
        
        self.register(FigureSpec(
            figure_id="F-16",
            title="Bootstrap Metric Confidence Intervals",
            category=FigureCategory.UNCERTAINTY,
            module="uncertainty",
            method_name="plot_bootstrap_ci",
            save_name="fig16_bootstrap_ci.png",
            description="Bootstrap confidence intervals for R², RMSE, and MAE with stability estimates and null-hypothesis bands.",
            required_fields=["y_test", "y_pred_ensemble"],
            is_essential=True,
        ))
        
        # ================== INTERPRETABILITY (F-12, F-14) ==================
        self.register(FigureSpec(
            figure_id="F-12",
            title="Global Feature Importance",
            category=FigureCategory.INTERPRETABILITY,
            module="interpretability",
            method_name="plot_feature_importance",
            save_name="fig12_feature_importance.png",
            description="Lollipop chart of top-N global feature importance with aggregated importance across models.",
            required_fields=[],  # Can work with empty if no importance data available
            optional_fields=["per_model_feature_importance", "feature_names"],
            is_essential=False,
            skip_reason_code="NO_IMPORTANCE_DATA",
        ))
        
        self.register(FigureSpec(
            figure_id="F-14",
            title="Per-Model Feature Importance Heatmap",
            category=FigureCategory.INTERPRETABILITY,
            module="interpretability",
            method_name="plot_per_model_importance_heatmap",
            save_name="fig14_per_model_importance_heatmap.png",
            description="Heatmap of feature importance across models with normalization-by-model and hierarchical clustering.",
            required_fields=[],
            optional_fields=["per_model_feature_importance", "feature_names"],
            is_essential=False,
            skip_reason_code="NO_IMPORTANCE_DATA",
        ))
        
        # ================== IMPACT (F-10, F-11, F-21) ==================
        self.register(FigureSpec(
            figure_id="F-10",
            title="Rank Change Bubble Chart",
            category=FigureCategory.IMPACT,
            module="impact",
            method_name="plot_rank_change_bubble",
            save_name="fig10_rank_change_bubble.png",
            description="Bubble chart showing current vs forecast rank, with size/color encoding performance and confidence.",
            required_fields=["y_pred_ensemble", "provinces"],
            optional_fields=["current_scores", "prediction_year"],
            is_essential=True,
        ))
        
        self.register(FigureSpec(
            figure_id="F-11",
            title="Province Comparison (Current vs Forecast)",
            category=FigureCategory.IMPACT,
            module="impact",
            method_name="plot_province_comparison",
            save_name="fig11_province_comparison.png",
            description="Grouped bar chart comparing current vs forecast scores for top-N provinces with CI bounds.",
            required_fields=["y_pred_ensemble", "provinces"],
            optional_fields=["current_scores", "interval_lower_df", "interval_upper_df"],
            is_essential=True,
        ))
        
        self.register(FigureSpec(
            figure_id="F-21",
            title="Score Trajectory with Forecast Fan",
            category=FigureCategory.IMPACT,
            module="impact",
            method_name="plot_score_trajectory",
            save_name="fig21_score_trajectory.png",
            description="Line plot of historical scores + forecast with confidence interval fan for selected entities.",
            required_fields=["y_pred_ensemble"],
            optional_fields=["provinces", "interval_lower_df", "interval_upper_df", "prediction_year"],
            is_essential=True,
        ))
        
        # ================== DIVERSITY (F-17, F-18) ==================
        self.register(FigureSpec(
            figure_id="F-17",
            title="Prediction Correlation Heatmap",
            category=FigureCategory.DIVERSITY,
            module="diversity",
            method_name="plot_prediction_correlation_heatmap",
            save_name="fig17_prediction_correlation_heatmap.png",
            description="Clustered correlation heatmap of per-model predictions with dendrograms and diversity metrics annotation.",
            required_fields=[],
            optional_fields=["per_model_oof_predictions"],
            is_essential=False,
            skip_reason_code="INSUFFICIENT_MODELS",
        ))
        
        self.register(FigureSpec(
            figure_id="F-18",
            title="Prediction Scatter Matrix",
            category=FigureCategory.DIVERSITY,
            module="diversity",
            method_name="plot_prediction_scatter_matrix",
            save_name="fig18_prediction_scatter_matrix.png",
            description="Pairwise scatter matrix of base model OOF predictions with marginal distributions and correlation coefficients.",
            required_fields=[],
            optional_fields=["per_model_oof_predictions"],
            is_essential=False,
            skip_reason_code="INSUFFICIENT_MODELS",
        ))
        
        # ================== TEMPORAL (F-19, F-20) ==================
        self.register(FigureSpec(
            figure_id="F-19",
            title="Entity Error Analysis",
            category=FigureCategory.TEMPORAL,
            module="temporal",
            method_name="plot_entity_error_analysis",
            save_name="fig19_entity_error_analysis.png",
            description="Horizontal bar chart of entity-wise errors sorted by magnitude with signed bias annotation and top/bottom highlighting.",
            required_fields=["y_test", "y_pred_ensemble"],
            optional_fields=["entity_names"],
            is_essential=True,
        ))
        
        self.register(FigureSpec(
            figure_id="F-20",
            title="Temporal Training Curve",
            category=FigureCategory.TEMPORAL,
            module="temporal",
            method_name="plot_temporal_training_curve",
            save_name="fig20_temporal_training_curve.png",
            description="Walk-forward temporal curve showing R² vs validation year/fold with fold stratification bands.",
            required_fields=["cv_scores"],
            optional_fields=["cv_fold_val_years"],
            is_essential=True,
        ))
    
    def register(self, spec: FigureSpec):
        """Register a figure specification."""
        self._figures[spec.figure_id] = spec
        
        # Index by category
        if spec.category not in self._by_category:
            self._by_category[spec.category] = []
        self._by_category[spec.category].append(spec.figure_id)
        
        # Index by module
        if spec.module not in self._by_module:
            self._by_module[spec.module] = []
        self._by_module[spec.module].append(spec.figure_id)
    
    def get_figure(self, figure_id: str) -> Optional[FigureSpec]:
        """Get figure spec by ID."""
        return self._figures.get(figure_id)
    
    def get_all_figures(self) -> List[FigureSpec]:
        """Get all registered figures in ID order."""
        return [self._figures[fid] for fid in sorted(self._figures.keys())]
    
    def get_essential_figures(self) -> List[FigureSpec]:
        """Get essential figures (mandatory suite)."""
        return [f for f in self.get_all_figures() if f.is_essential]
    
    def get_optional_figures(self) -> List[FigureSpec]:
        """Get optional/advanced figures."""
        return [f for f in self.get_all_figures() if not f.is_essential]
    
    def get_by_category(self, category: FigureCategory) -> List[FigureSpec]:
        """Get all figures in a category."""
        figure_ids = self._by_category.get(category, [])
        return [self._figures[fid] for fid in figure_ids]
    
    def get_by_module(self, module: str) -> List[FigureSpec]:
        """Get all figures from a chart module."""
        figure_ids = self._by_module.get(module, [])
        return [self._figures[fid] for fid in figure_ids]
    
    def get_execution_order(self) -> List[FigureSpec]:
        """Get figures in recommended execution order (respecting dependencies)."""
        # Simple topological sort: figures without dependencies first
        # For now, return in ID order since dependencies are mostly cross-module
        return self.get_all_figures()
    
    def create_global_manifest(self) -> 'ForecastFigureManifest':
        """Return self (implements singleton pattern)."""
        return self


# Global manifest instance
_GLOBAL_MANIFEST: Optional[ForecastFigureManifest] = None


def get_manifest() -> ForecastFigureManifest:
    """Get or create the global figure manifest."""
    global _GLOBAL_MANIFEST
    if _GLOBAL_MANIFEST is None:
        _GLOBAL_MANIFEST = ForecastFigureManifest()
    return _GLOBAL_MANIFEST


__all__ = [
    'FigureCategory',
    'FigureSpec',
    'ForecastFigureManifest',
    'get_manifest',
]
