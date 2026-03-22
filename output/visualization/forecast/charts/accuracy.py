# -*- coding: utf-8 -*-
"""
Forecast Visualization: Accuracy Diagnostics

Figures F-01, F-02, F-03 covering actual-vs-predicted scatter,
residual diagnostics, and holdout comparison.

This module is extracted from the monolithic forecast_plots.py and
refactored to:
- Accept ForecastVizPayload for type-safe data flow
- Use centralized metrics from metrics.py
- Maintain visualization quality (300 DPI, publication-grade styling)
"""

from typing import Optional
import numpy as np
import pandas as pd

from output.visualization.base import (
    BasePlotter, HAS_MATPLOTLIB, HAS_SCIPY,
    PALETTE, CATEGORICAL_COLORS, plt, sp_stats,
)
from output.visualization.forecast.contracts import ForecastVizPayload
from output.visualization.forecast.metrics import (
    r2_score, rmse_score, mae_score, mape_score,
)


class AccuracyCharts(BasePlotter):
    """
    Accuracy and error structure diagnostics for ensemble forecasting.
    
    Figures:
    - F-01: Actual vs Predicted scatter with fit and confidence interval
    - F-02: 4-panel residual diagnostics (vs predicted, hist+KDE, Q-Q, top errors)
    - F-03: Holdout comparison (scatter, residual distribution, heatmap, ranking)
    """
    
    # ====================================================================
    # F-01: Actual vs Predicted
    # ====================================================================
    
    def plot_forecast_scatter(
        self,
        payload: ForecastVizPayload,
        save_name: str = 'fig01_forecast_scatter.png',
    ) -> Optional[str]:
        """
        Actual vs Predicted scatter with residual coloring, fit line, and stats.
        
        Args:
            payload: ForecastVizPayload with y_test, y_pred_ensemble.
            save_name: Output PNG filename.
        
        Returns:
            Path to saved figure or None if matplotlib unavailable.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        y_true = payload.y_test
        y_pred = payload.y_pred_ensemble
        entity_names = payload.entity_names
        
        fig, ax = plt.subplots(figsize=(12, 11))
        
        # Color by absolute residual
        residuals = np.abs(y_true - y_pred)
        scatter = ax.scatter(
            y_true, y_pred, c=residuals, cmap='RdYlGn_r',
            s=100, alpha=0.8, edgecolors='black', linewidths=0.5, zorder=4,
        )
        
        # Perfect prediction line
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                'k--', lw=2, label='Perfect', alpha=0.7, zorder=2)
        
        # Polynomial fit
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        xs = np.linspace(lo - margin, hi + margin, 200)
        ax.plot(xs, p(xs), color=PALETTE['royal_blue'], lw=2,
                label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}', zorder=3)
        
        # Confidence band
        ci = 1.96 * np.std(y_true - y_pred)
        ax.fill_between(xs, p(xs) - ci, p(xs) + ci, alpha=0.12,
                        color=PALETTE['royal_blue'], label='95% CI')
        
        # Compute metrics using centralized functions
        r2 = r2_score(y_true, y_pred)
        mae = mae_score(y_true, y_pred)
        rmse = rmse_score(y_true, y_pred)
        
        stats = (f'R² = {r2:.6f}\nMAE = {mae:.6f}\n'
                 f'RMSE = {rmse:.6f}\nN = {len(y_true)}')
        ax.text(0.03, 0.97, stats, transform=ax.transAxes, fontsize=10,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', fc='white',
                          ec='#CCCCCC', alpha=0.95))
        
        # Annotate worst 5 errors
        worst = np.argsort(residuals)[-5:]
        for wi in worst:
            lbl = entity_names[wi] if entity_names else f'#{wi}'
            ax.annotate(lbl, (y_true[wi], y_pred[wi]),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=8, alpha=0.85,
                        arrowprops=dict(arrowstyle='->', alpha=0.5))
        
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('|Residual|', fontsize=10)
        
        ax.set_xlabel('Actual Score')
        ax.set_ylabel('Predicted Score')
        ax.set_title('Forecast Performance — Actual vs Predicted', pad=12)
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        
        return self._save(fig, save_name)
    
    # ====================================================================
    # F-02: Residual Diagnostics (4-panel)
    # ====================================================================
    
    def plot_forecast_residuals(
        self,
        payload: ForecastVizPayload,
        save_name: str = 'fig02_forecast_residuals.png',
    ) -> Optional[str]:
        """
        4-panel residual diagnostics: vs predicted, histogram+KDE, Q-Q, top errors.
        
        Args:
            payload: ForecastVizPayload with y_test, y_pred_ensemble.
            save_name: Output PNG filename.
        
        Returns:
            Path to saved figure or None if matplotlib unavailable.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        y_true = payload.y_test
        y_pred = payload.y_pred_ensemble
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Panel 1 – Residuals vs Predicted
        ax = axes[0, 0]
        colors = ['#17B169' if r >= 0 else '#E5625E' for r in residuals]
        ax.scatter(y_pred, residuals, c=colors, s=70, alpha=0.7,
                   edgecolors='black', linewidths=0.4)
        ax.axhline(0, color='black', lw=1.5)
        
        std_r = np.std(residuals) if np.std(residuals) > 0 else 1
        for mult, style, col in [(1, '--', PALETTE['gold']),
                                  (2, ':', PALETTE['crimson'])]:
            ax.axhline(mult * std_r, ls=style, color=col, lw=1.2)
            ax.axhline(-mult * std_r, ls=style, color=col, lw=1.2)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residual')
        ax.set_title('Residuals vs Predicted')
        
        # Panel 2 – Residual histogram + KDE
        ax = axes[0, 1]
        ax.hist(residuals, bins=25, density=True, color=PALETTE['royal_blue'],
                alpha=0.7, edgecolor='white')
        if HAS_SCIPY and len(residuals) > 5:
            xg = np.linspace(residuals.min() - std_r,
                             residuals.max() + std_r, 200)
            kde = sp_stats.gaussian_kde(residuals)
            ax.plot(xg, kde(xg), color=PALETTE['crimson'], lw=2, label='KDE')
            ax.plot(xg, sp_stats.norm.pdf(xg, residuals.mean(), std_r),
                    '--', color='gray', lw=1.5, label='Normal')
        ax.set_xlabel('Residual')
        ax.set_ylabel('Density')
        ax.set_title('Residual Distribution')
        ax.legend(fontsize=9)
        
        # Panel 3 – QQ plot
        ax = axes[1, 0]
        if HAS_SCIPY:
            (osm, osr), (slope, intercept, _) = sp_stats.probplot(
                residuals, dist='norm',
            )
            ax.scatter(osm, osr, s=50, alpha=0.6, color=PALETTE['royal_blue'],
                       edgecolors='white')
            ax.plot(osm, slope * np.array(osm) + intercept, 'r-', lw=2)
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Ordered Values')
        ax.set_title('Q-Q Plot (Normality Check)')
        
        # Panel 4 – Largest errors
        ax = axes[1, 1]
        abs_e = np.abs(residuals)
        order = np.argsort(abs_e)[::-1][:15]
        ax.barh(
            range(len(order)), abs_e[order],
            color=plt.colormaps['Reds'](np.linspace(0.35, 0.9, len(order))),
            edgecolor='white',
        )
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f'#{i}' for i in order], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('|Residual|')
        ax.set_title('Largest Prediction Errors')
        
        fig.suptitle('Forecast Residual Diagnostics', fontsize=14, y=1.01)
        plt.tight_layout()
        return self._save(fig, save_name)
    
    # ====================================================================
    # F-03: Holdout Comparison (4-panel)
    # ====================================================================
    
    def plot_holdout_comparison(
        self,
        payload: ForecastVizPayload,
        save_name: str = 'fig03_holdout_comparison.png',
    ) -> Optional[str]:
        """
        Comprehensive holdout evaluation: per-model scatter, residual distribution,
        province heatmap, and R² ranking.
        
        Args:
            payload: ForecastVizPayload with y_test, y_pred_ensemble,
                     per_model_holdout_predictions (optional), entity_names,
                     model_contributions, model_performance.
            save_name: Output PNG filename.
        
        Returns:
            Path to saved figure or None if matplotlib unavailable.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        y_test = payload.y_test
        y_pred_ensemble = payload.y_pred_ensemble
        n = len(y_test)
        
        if n < 3:
            return None
        
        # Build model dict (include ensemble)
        model_preds = {}
        if payload.per_model_holdout_predictions:
            for name, arr in payload.per_model_holdout_predictions.items():
                model_preds[name] = np.asarray(arr, dtype=float).ravel()[:n]
        model_preds['Meta-Learner'] = y_pred_ensemble
        
        model_names = list(model_preds.keys())
        entity_names = payload.entity_names
        contributions = payload.model_contributions or {}
        
        # Model color mapping
        def _model_color(name: str) -> str:
            color_map = {
                'xgboost': '#E74C3C', 'xgb': '#E74C3C', 'catboost': '#E74C3C',
                'random': '#3498DB', 'rf': '#3498DB', 'forest': '#3498DB',
                'neural': '#9B59B6', 'nam': '#9B59B6', 'additive': '#9B59B6',
                'bayesian': '#F39C12', 'bayes': '#F39C12', 'hierarchical': '#F39C12',
                'var': '#1ABC9C', 'panel': '#1ABC9C',
                'meta': '#E67E22', 'learner': '#E67E22', 'ensemble': '#E67E22',
                'quantile': '#E91E63', 'qrf': '#E91E63',
            }
            nl = name.lower()
            for key, col in color_map.items():
                if key in nl:
                    return col
            return '#95A5A6'
        
        colors = [_model_color(m) for m in model_names]
        n_models = len(model_names)
        
        fig = plt.figure(figsize=(22, 18))
        gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.30)
        
        # ── Panel 1: Predicted vs Actual scatter per model ───────────
        ax1 = fig.add_subplot(gs[0, 0])
        lo = min(y_test.min(), min(p.min() for p in model_preds.values()))
        hi = max(y_test.max(), max(p.max() for p in model_preds.values()))
        margin = (hi - lo) * 0.05
        ax1.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                 'k--', lw=1.5, alpha=0.6, label='Perfect', zorder=1)
        
        for i, (name, pred) in enumerate(model_preds.items()):
            col = colors[i]
            is_ensemble = (name == 'Meta-Learner')
            marker = '*' if is_ensemble else 'o'
            ms = 80 if is_ensemble else 40
            zorder = 6 if is_ensemble else 3
            
            r2 = r2_score(y_test, pred)
            
            ax1.scatter(y_test, pred, s=ms, c=col, marker=marker,
                        alpha=0.80, edgecolors='white', linewidths=0.6,
                        zorder=zorder,
                        label=f'{self._truncate(name, 14)} (R²={r2:.4f})')
        
        ax1.set_xlabel('Actual Score (Holdout)')
        ax1.set_ylabel('Predicted Score')
        ax1.set_title('Predicted vs Actual — All Models', fontsize=12,
                       fontweight='bold', pad=8)
        ax1.legend(fontsize=7.5, loc='upper left', ncol=1, framealpha=0.9)
        ax1.set_xlim(lo - margin, hi + margin)
        ax1.set_ylim(lo - margin, hi + margin)
        ax1.set_aspect('equal', adjustable='box')
        ax1.grid(alpha=0.25)
        
        # ── Panel 2: Residual violin / box per model ─────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        residuals_per_model = [y_test - model_preds[name] for name in model_names]
        
        parts = ax2.violinplot(residuals_per_model, positions=range(n_models),
                               showmedians=False, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.35)
        
        bp = ax2.boxplot(residuals_per_model, positions=range(n_models),
                         patch_artist=True, widths=0.35,
                         boxprops=dict(linewidth=1.0),
                         medianprops=dict(color='black', lw=2),
                         whiskerprops=dict(color='gray'),
                         flierprops=dict(marker='D', ms=3, alpha=0.5))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.55)
        
        for i, resid in enumerate(residuals_per_model):
            rmse = rmse_score(y_test, y_test - resid)
            mae = mae_score(y_test, y_test - resid)
            ax2.text(i, ax2.get_ylim()[0], f'RMSE={rmse:.4f}\nMAE={mae:.4f}',
                     ha='center', va='top', fontsize=6.5, color=colors[i],
                     fontweight='bold')
        
        ax2.axhline(0, color='black', lw=1.2, ls='-', zorder=0)
        ax2.set_xticks(range(n_models))
        ax2.set_xticklabels([self._truncate(m, 12) for m in model_names],
                            rotation=40, ha='right', fontsize=8)
        ax2.set_ylabel('Residual (Actual − Predicted)')
        ax2.set_title('Residual Distribution per Model', fontsize=12,
                       fontweight='bold', pad=8)
        ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax2.set_axisbelow(True)
        
        # ── Panel 3: Province × Model residual heatmap ───────────────
        ax3 = fig.add_subplot(gs[1, 0])
        
        prov_labels = entity_names if entity_names else [f'#{i}' for i in range(n)]
        
        # Sort by ensemble error magnitude
        ensemble_resid = y_test - y_pred_ensemble
        sort_idx = np.argsort(np.abs(ensemble_resid))[::-1]
        show_n = min(30, n)
        show_idx = sort_idx[:show_n]
        
        resid_matrix = np.column_stack([
            (y_test - model_preds[m])[show_idx] for m in model_names
        ])
        
        vabs = max(abs(resid_matrix.min()), abs(resid_matrix.max()))
        if vabs < 1e-12:
            vabs = 1.0
        im = ax3.imshow(resid_matrix, aspect='auto', cmap='RdBu_r',
                        vmin=-vabs, vmax=vabs)
        
        ax3.set_xticks(range(n_models))
        ax3.set_xticklabels([self._truncate(m, 10) for m in model_names],
                            rotation=40, ha='right', fontsize=7.5)
        ax3.set_yticks(range(show_n))
        ax3.set_yticklabels([self._truncate(str(prov_labels[i]), 14)
                             for i in show_idx], fontsize=7)
        
        # Annotate cells
        for ri in range(show_n):
            for ci in range(n_models):
                val = resid_matrix[ri, ci]
                txt_col = 'white' if abs(val) > vabs * 0.55 else 'black'
                ax3.text(ci, ri, f'{val:.3f}', ha='center', va='center',
                         fontsize=5.5, color=txt_col)
        
        cbar = fig.colorbar(im, ax=ax3, shrink=0.7, pad=0.02)
        cbar.set_label('Residual (Actual − Predicted)', fontsize=9)
        ax3.set_title(f'Province × Model Residual Heatmap (Top {show_n} by |error|)',
                       fontsize=12, fontweight='bold', pad=8)
        
        # ── Panel 4: Model R² ranking + contribution weight ──────────
        ax4 = fig.add_subplot(gs[1, 1])
        
        r2_scores = [r2_score(y_test, model_preds[name]) for name in model_names]
        
        sort_order = np.argsort(r2_scores)[::-1]
        sorted_names = [model_names[i] for i in sort_order]
        sorted_r2 = [r2_scores[i] for i in sort_order]
        sorted_colors = [colors[i] for i in sort_order]
        sorted_weights = [contributions.get(model_names[i], 0.0) for i in sort_order]
        
        max_w = max(max(sorted_weights), 1e-9)
        alphas = [0.40 + 0.60 * (w / max_w) for w in sorted_weights]
        
        bars = ax4.barh(range(len(sorted_names)), sorted_r2,
                        color=sorted_colors, edgecolor='white',
                        linewidth=0.5, zorder=2)
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)
        
        for i, (r2, w) in enumerate(zip(sorted_r2, sorted_weights)):
            label = f'R²={r2:.4f}'
            if w > 0:
                label += f'  (w={w:.3f})'
            ax4.text(max(r2 + 0.005, 0.01), i, label,
                     va='center', fontsize=8, color='#333333',
                     fontweight='bold')
        
        ax4.set_yticks(range(len(sorted_names)))
        ax4.set_yticklabels(
            [self._truncate(n, 16) for n in sorted_names], fontsize=9)
        ax4.invert_yaxis()
        ax4.set_xlabel('Holdout R²', fontsize=11)
        ax4.set_title('Model Ranking — Holdout R² (bar opacity ∝ weight)',
                       fontsize=12, fontweight='bold', pad=8)
        ax4.xaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax4.set_axisbelow(True)
        ax4.axvline(0, color='gray', lw=0.8)
        
        # Overall stats
        r2_ens = r2_score(y_test, y_pred_ensemble)
        mae_ens = mae_score(y_test, y_pred_ensemble)
        rmse_ens = rmse_score(y_test, y_pred_ensemble)
        mape_ens = mape_score(y_test, y_pred_ensemble)
        
        stats_text = (
            f'Ensemble Holdout Metrics\n'
            f'{"─" * 28}\n'
            f'R²   = {r2_ens:.6f}\n'
            f'MAE  = {mae_ens:.6f}\n'
            f'RMSE = {rmse_ens:.6f}\n'
            f'MAPE = {mape_ens:.2f}%\n'
            f'N    = {n}'
        )
        fig.text(0.99, 0.01, stats_text, fontsize=9,
                 fontfamily='monospace', va='bottom', ha='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='#F8F8FF',
                           ec='#CCCCCC', alpha=0.95))
        
        fig.suptitle('Holdout Forecast Comparison — Predicted vs Test Set',
                     fontsize=15, fontweight='bold', y=0.99)
        return self._save(fig, save_name)
    
    # ====================================================================
    # F-13: Per-Model Residual Distributions
    # ====================================================================
    
    def plot_residual_distributions(
        self,
        payload: ForecastVizPayload,
        max_models: int = 12,
        save_name: str = 'fig13_residual_distributions.png',
    ) -> Optional[str]:
        """
        Per-model residual distribution panel (histograms + ensemble overlay).
        
        Note: This requires per_model_oof_predictions in the payload.
        
        Args:
            payload: ForecastVizPayload with y_test, per_model_oof_predictions.
            max_models: Maximum number of models to display.
            save_name: Output PNG filename.
        
        Returns:
            Path to saved figure or None if matplotlib unavailable or data missing.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        # TODO: Phase 2 implementation
        # This method will display per-model residual histograms
        # with ensemble residuals overlaid for comparison
        return None


__all__ = ['AccuracyCharts']
