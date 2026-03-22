# -*- coding: utf-8 -*-
"""
Forecast Visualization: Ensemble Architecture and Model Comparison

Figures F-04, F-05, F-06, F-22 covering model weights, performance,
cross-validation distributions, and ensemble pipeline architecture.

This module is extracted from the monolithic forecast_plots.py and
refactored to accept ForecastVizPayload for type-safe data flow.
"""

from typing import Optional, Dict, List
import numpy as np

from output.visualization.base import (
    BasePlotter, HAS_MATPLOTLIB, HAS_SCIPY,
    PALETTE, CATEGORICAL_COLORS, plt, sp_stats,
)
from output.visualization.forecast.contracts import ForecastVizPayload
from output.visualization.forecast.metrics import r2_score, rmse_score, mae_score


def _mcolor(name: str, fallback: str = '#7F8C8D') -> str:
    """Return a display colour for name by matching family fragments."""
    _FAMILY_COLOR = {
        'catboost': '#E74C3C', 'xgboost': '#E74C3C',
        'random': '#3498DB', 'rf': '#3498DB',
        'neural': '#9B59B6', 'nam': '#9B59B6',
        'bayesian': '#F39C12', 'bayes': '#F39C12',
        'var': '#1ABC9C', 'panel': '#1ABC9C',
        'meta': '#E67E22', 'learner': '#E67E22',
        'quantile': '#E91E63', 'qrf': '#E91E63',
        'kernel': '#8E44AD', 'svr': '#C0392B',
        'ridge': '#D35400',
    }
    lo = name.lower()
    for frag, col in _FAMILY_COLOR.items():
        if frag in lo:
            return col
    return fallback


class EnsembleCharts(BasePlotter):
    """
    Ensemble architecture and model comparison diagnostics.
    
    Figures:
    - F-04: Model contribution weights donut chart
    - F-05: Per-model performance comparison
    - F-06: Cross-validation score distributions
    - F-22: Ensemble architecture flowchart (3-tier pipeline)
    """
    
    # ====================================================================
    # F-04: Model Weights Donut
    # ====================================================================
    
    def plot_model_weights_donut(
        self,
        payload: ForecastVizPayload,
        save_name: str = 'fig04_model_weights_donut.png',
    ) -> Optional[str]:
        """
        Donut pie chart of meta-learner weight distribution.
        
        Args:
            payload: ForecastVizPayload with model_contributions field.
            save_name: Output PNG filename.
        
        Returns:
            Path to saved figure or None if matplotlib unavailable.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if not payload.model_contributions:
            return None
        
        labels = list(payload.model_contributions.keys())
        sizes = list(payload.model_contributions.values())
        colors = CATEGORICAL_COLORS[:len(labels)]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=colors, pctdistance=0.8,
            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
        )
        
        for t in autotexts:
            t.set_fontsize(10)
            t.set_fontweight('bold')
        for t in texts:
            t.set_fontsize(10)
        
        ax.set_title('Meta-Learner — Base Model Weights', fontsize=14, pad=20)
        centre = plt.Circle((0, 0), 0.55, fc='white')
        ax.add_artist(centre)
        ax.text(0, 0, 'Meta-\nLearner', ha='center', va='center',
                fontsize=14, fontweight='bold', color='#333333')
        
        return self._save(fig, save_name)
    
    # ====================================================================
    # F-05: Model Performance Comparison
    # ====================================================================
    
    def plot_model_performance(
        self,
        payload: ForecastVizPayload,
        save_name: str = 'fig05_model_performance.png',
    ) -> Optional[str]:
        """
        Grouped bar chart of model metrics across all models.
        
        Args:
            payload: ForecastVizPayload with model_performance field.
            save_name: Output PNG filename.
        
        Returns:
            Path to saved figure or None if matplotlib unavailable.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if not payload.model_performance:
            return None
        
        models = list(payload.model_performance.keys())
        all_metrics = sorted({m for d in payload.model_performance.values() for m in d})
        n_models = len(models)
        n_metrics = len(all_metrics)
        
        if n_metrics == 0:
            return None
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 7))
        if n_metrics == 1:
            axes = [axes]
        
        colors = CATEGORICAL_COLORS[:n_models]
        
        for mi, metric in enumerate(all_metrics):
            ax = axes[mi]
            vals = [payload.model_performance[m].get(metric, 0) for m in models]
            
            bars = ax.bar(range(n_models), vals, color=colors, alpha=0.8,
                         edgecolor='white', linewidth=1)
            
            for i, (bar, v) in enumerate(zip(bars, vals)):
                ax.text(i, v + max(vals) * 0.02, f'{v:.4f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_xticks(range(n_models))
            ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
            ax.set_ylabel(metric.upper(), fontsize=10)
            ax.set_title(f'{metric.upper()} Comparison', fontsize=11, fontweight='bold')
            ax.set_ylim(0, max(vals) * 1.15 if max(vals) > 0 else 1)
        
        fig.suptitle('Model Performance Comparison', fontsize=13, fontweight='bold', y=1.00)
        plt.tight_layout()
        return self._save(fig, save_name)
    
    # ====================================================================
    # F-06: Cross-Validation Boxplots
    # ====================================================================
    
    def plot_cv_boxplots(
        self,
        payload: ForecastVizPayload,
        save_name: str = 'fig06_cv_boxplots.png',
    ) -> Optional[str]:
        """
        Box plots of cross-validation score distributions per model.
        
        Args:
            payload: ForecastVizPayload with cv_scores field.
            save_name: Output PNG filename.
        
        Returns:
            Path to saved figure or None if matplotlib unavailable.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if not payload.cv_scores:
            return None
        
        metrics = list(payload.cv_scores.keys())
        n = len(metrics)
        
        fig, ax = plt.subplots(figsize=(max(8, n * 2.5), 7))
        data = [payload.cv_scores[m] for m in metrics]
        colors = CATEGORICAL_COLORS[:n]
        
        bp = ax.boxplot(
            data, patch_artist=True, notch=True,
            boxprops=dict(linewidth=1.2),
            medianprops=dict(color='black', lw=2),
            whiskerprops=dict(color='gray'),
            flierprops=dict(marker='D', ms=5, alpha=0.5),
        )
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.65)
        
        for i, (m, sc) in enumerate(payload.cv_scores.items()):
            jitter = np.random.default_rng(42).normal(i + 1, 0.06, len(sc))
            ax.scatter(jitter, sc, s=50, color='black', alpha=0.5, zorder=5)
            mean_v = np.mean(sc)
            std_v = np.std(sc)
            ylim = ax.get_ylim()
            ax.text(i + 1, max(sc) + 0.03 * (ylim[1] - ylim[0]),
                    f'{mean_v:.3f}±{std_v:.3f}', ha='center', fontsize=9,
                    fontweight='bold')
        
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylabel('Cross-Validation Score')
        ax.set_title('Cross-Validation Score Distributions', pad=12)
        
        return self._save(fig, save_name)
    
    # ====================================================================
    # F-22: Ensemble Architecture Flowchart
    # ====================================================================
    
    def plot_ensemble_architecture(
        self,
        payload: ForecastVizPayload,
        save_name: str = 'fig22_ensemble_architecture.png',
    ) -> Optional[str]:
        """
        3-tier ensemble pipeline flowchart.
        
        Shows data flow: Base Models → Meta-Learner → Conformal Prediction → Outputs.
        Uses active model roster from payload.model_names or model_contributions.
        
        Args:
            payload: ForecastVizPayload with model_names or model_contributions.
            save_name: Output PNG filename.
        
        Returns:
            Path to saved figure or None if matplotlib unavailable.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        try:
            from matplotlib.patches import FancyBboxPatch
        except ImportError:
            return None
        
        fig, ax = plt.subplots(figsize=(16, 11))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 11)
        ax.axis('off')
        
        # Color scheme
        tier_colors = {
            'data': '#EBF5FB', 'base': '#D6EAF8', 'ensemble': '#D5F5E3',
            'conformal': '#FDEBD0', 'output': '#F9EBEA',
        }
        border_colors = {
            'data': '#2E86C1', 'base': '#1A5276', 'ensemble': '#1E8449',
            'conformal': '#CA6F1E', 'output': '#922B21',
        }
        
        def _box(x, y, w, h, label, sublabel='', kind='base', fontsize=9):
            fc = tier_colors[kind]
            ec = border_colors[kind]
            patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                   boxstyle='round,pad=0.08',
                                   fc=fc, ec=ec, lw=2, zorder=3)
            ax.add_patch(patch)
            ax.text(x, y + (0.12 if sublabel else 0), label,
                    ha='center', va='center', fontsize=fontsize,
                    fontweight='bold', color=ec, zorder=4)
            if sublabel:
                ax.text(x, y - 0.22, sublabel,
                        ha='center', va='center', fontsize=7,
                        color='#555555', style='italic', zorder=4)
        
        def _arrow(x0, y0, x1, y1):
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='#555555',
                                       lw=1.8))
        
        # Data source
        _box(8, 10.1, 5.5, 0.85, 'Panel Data (Provinces × Years × Criteria)',
             kind='data', fontsize=9.5)
        
        # Base models tier
        ax.text(8, 9.35, 'Tier 1 — Base Estimators', ha='center',
                fontsize=9, color='#1A5276', style='italic')
        
        model_names = payload.model_names or list(payload.model_contributions.keys()) \
            if payload.model_contributions else ['CatBoost', 'BayesianRidge', 'KernelRidge', 'SVR', 'QuantileRF']
        
        n_base = len(model_names)
        _bw = min(2.2, max(1.0, 13.6 / n_base - 0.3))
        xs_base = np.linspace(1.2, 14.8, n_base)
        
        for xi, mname in zip(xs_base, model_names):
            _box(xi, 8.5, _bw, 1.15, self._truncate(mname, 12),
                 kind='base', fontsize=max(6.5, 8.5 - max(0, n_base - 6) * 0.4))
            _arrow(8, 9.68, xi, 9.08)
        
        # Meta-learner tier
        ax.text(8, 7.5, 'Tier 2 — Stacked Generalization', ha='center',
                fontsize=9, color='#1E8449', style='italic')
        _box(8, 6.85, 5.8, 0.9, 'Meta-Learner',
             'Optimal linear combination', kind='ensemble')
        for xi in xs_base:
            _arrow(xi, 7.93, 8, 7.30)
        
        # Conformal tier
        ax.text(8, 5.7, 'Tier 3 — Uncertainty Calibration', ha='center',
                fontsize=9, color='#CA6F1E', style='italic')
        _box(8, 5.05, 5.8, 0.9, 'Conformal Prediction',
             'Distribution-free coverage guarantee (95%)', kind='conformal')
        _arrow(8, 6.40, 8, 5.50)
        
        # Outputs
        _arrow(8, 4.60, 8, 3.90)
        out_cols = [
            ('Point\nForecast', 'Scores'), ('Lower CI\n(2.5%)', 'Bound'),
            ('Upper CI\n(97.5%)', 'Bound'), ('Uncertainty', 'Width'),
        ]
        xs_out = np.linspace(3.5, 12.5, 4)
        for xi, (name, sub) in zip(xs_out, out_cols):
            _box(xi, 3.3, 2.2, 1.1, name, sub, kind='output', fontsize=8)
            _arrow(8, 3.90, xi, 3.85)
        
        ax.text(8, 2.7, '↓ Feeds into MCDM Ranking Pipeline', ha='center',
                fontsize=10, fontweight='bold', color='#555555')
        
        fig.suptitle('ML Ensemble Forecasting Architecture — 3-Tier Pipeline',
                     fontsize=13, y=0.98, fontweight='bold')
        plt.tight_layout()
        return self._save(fig, save_name)
    
    # ====================================================================
    # F-20b: Model Contribution Dots (Weight vs CV R²)
    # ====================================================================
    
    def plot_model_contribution_dots(
        self,
        payload: ForecastVizPayload,
        save_name: str = 'fig20b_model_contribution_dots.png',
    ) -> Optional[str]:
        """
        Bubble chart of model contribution weight vs Cross-Validation R².
        
        Bubble size represents weight, position shows weight (x) vs CV R² (y).
        
        Args:
            payload: ForecastVizPayload with model_contributions, cv_scores.
            save_name: Output PNG filename.
        
        Returns:
            Path to saved figure or None if matplotlib unavailable or data missing.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        # TODO: Phase 2 implementation
        # This method will display a bubble chart showing:
        # X-axis: Model contribution weight
        # Y-axis: Cross-validation R² score
        # Bubble size: Model weight (scaled for visibility)
        # Color: Model family (using _mcolor function)
        return None
    
    # ====================================================================
    # F-15: Model Metric Radar/Spider Chart
    # ====================================================================
    
    def plot_model_metric_radar(
        self,
        payload: ForecastVizPayload,
        metrics: Optional[List[str]] = None,
        use_bar_fallback: bool = True,
        save_name: str = 'fig15_model_metric_radar.png',
    ) -> Optional[str]:
        """
        Radar/spider chart of model metrics (R², RMSE, MAE, etc.).
        Falls back to grouped bar chart if radar plot not available.
        
        Args:
            payload: ForecastVizPayload with model_performance field.
            metrics: Metrics to include (default: R², RMSE, MAE, Bias).
            use_bar_fallback: If True, use grouped bar chart fallback.
            save_name: Output PNG filename.
        
        Returns:
            Path to saved figure or None if matplotlib unavailable or data missing.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        # TODO: Phase 2 implementation
        # Will create radar/polar plot with:
        # - One axis per metric
        # - One line per model
        # - Normalized values (0-1 scale)
        # Fallback: If polar plot fails or metric is incompatible, use grouped bar chart
        return None


__all__ = ['EnsembleCharts']
