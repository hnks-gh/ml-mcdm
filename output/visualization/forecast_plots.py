# -*- coding: utf-8 -*-
"""
Forecasting Plots (fig16–fig23)
===============================

Figures covering ML forecasting diagnostics: actual-vs-predicted scatter,
residual panels, feature importance, model weight donut, performance
comparison, CV box plots, prediction intervals, and rank-change bubble.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .base import (
    BasePlotter, HAS_MATPLOTLIB, HAS_SCIPY,
    PALETTE, CATEGORICAL_COLORS, plt, sp_stats,
)

# ── Module-level model-family colour map (shared across all new figures) ──── #
_FAMILY_COLOR: Dict[str, str] = {
    'catboost': '#E74C3C', 'random':   '#3498DB',
    'neural':   '#9B59B6', 'bayesian': '#F39C12', 'var':       '#1ABC9C',
    'meta':     '#E67E22', 'quantile': '#E91E63', 'kernel':    '#8E44AD',
    'svr':      '#C0392B', 'ridge':    '#D35400',
}


def _mcolor(name: str, fallback: str = '#7F8C8D') -> str:
    """Return a display colour for *name* by matching family fragments."""
    lo = name.lower()
    for frag, col in _FAMILY_COLOR.items():
        if frag in lo:
            return col
    return fallback


class ForecastPlotter(BasePlotter):
    """Figures for the ML-forecasting pipeline phase."""

    # ==================================================================
    #  FIG 16 – Forecast Actual vs Predicted
    # ==================================================================

    def plot_forecast_scatter(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        entity_names: Optional[List[str]] = None,
        save_name: str = 'fig16_forecast_scatter.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)

        fig, ax = plt.subplots(figsize=(12, 11))

        residuals = np.abs(actual - predicted)
        scatter = ax.scatter(
            actual, predicted, c=residuals, cmap='RdYlGn_r',
            s=100, alpha=0.8, edgecolors='black', linewidths=0.5, zorder=4,
        )

        lo = min(actual.min(), predicted.min())
        hi = max(actual.max(), predicted.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                'k--', lw=2, label='Perfect', alpha=0.7, zorder=2)

        z = np.polyfit(actual, predicted, 1)
        p = np.poly1d(z)
        xs = np.linspace(lo - margin, hi + margin, 200)
        ax.plot(xs, p(xs), color=PALETTE['royal_blue'], lw=2,
                label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}', zorder=3)

        ci = 1.96 * np.std(actual - predicted)
        ax.fill_between(xs, p(xs) - ci, p(xs) + ci, alpha=0.12,
                        color=PALETTE['royal_blue'], label='95% CI')

        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        stats = (f'R² = {r2:.6f}\nMAE = {mae:.6f}\n'
                 f'RMSE = {rmse:.6f}\nN = {len(actual)}')
        ax.text(0.03, 0.97, stats, transform=ax.transAxes, fontsize=10,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', fc='white',
                          ec='#CCCCCC', alpha=0.95))

        worst = np.argsort(residuals)[-5:]
        for wi in worst:
            lbl = entity_names[wi] if entity_names else f'#{wi}'
            ax.annotate(lbl, (actual[wi], predicted[wi]),
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

    # ==================================================================
    #  FIG 17 – Forecast Residual Diagnostics (4-panel)
    # ==================================================================

    def plot_forecast_residuals(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        save_name: str = 'fig17_forecast_residuals.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)
        residuals = actual - predicted

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Panel 1 – Residuals vs Predicted
        ax = axes[0, 0]
        colors = ['#17B169' if r >= 0 else '#E5625E' for r in residuals]
        ax.scatter(predicted, residuals, c=colors, s=70, alpha=0.7,
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

    # ==================================================================
    #  FIG 18 – Feature Importance (lollipop)
    # ==================================================================

    def plot_feature_importance(
        self,
        importance: Dict[str, float],
        top_n: int = 20,
        title: str = 'Feature Importance — Top Features',
        save_name: str = 'fig18_feature_importance.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names = [k for k, _ in items]
        vals = [v for _, v in items]
        n = len(names)

        fig, ax = plt.subplots(figsize=(12, max(7, n * 0.38)))
        colors = plt.colormaps['cool'](np.linspace(0.2, 0.85, n))

        ax.hlines(range(n), 0, vals, color=colors, linewidth=2.5, zorder=2)
        ax.scatter(vals, range(n), s=100, color=colors, edgecolors='black',
                   linewidths=0.6, zorder=3)
        for i, v in enumerate(vals):
            ax.text(v + 0.003, i, f'{v:.4f}', va='center', fontsize=9)

        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [self._truncate(nm, 25) for nm in names], fontsize=9,
        )
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(title, pad=12)
        return self._save(fig, save_name)

    # Backward-compatible alias
    def plot_feature_importance_single(self, importance_dict, **kw):
        return self.plot_feature_importance(importance_dict, **kw)

    # ==================================================================
    #  FIG 19 – Model Weights Donut
    # ==================================================================

    def plot_model_weights_donut(
        self,
        weights: Dict[str, float],
        save_name: str = 'fig19_model_weights.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        labels = list(weights.keys())
        sizes = list(weights.values())
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

    # ==================================================================
    #  FIG 20 – Model Performance Comparison
    # ==================================================================

    def plot_model_performance(
        self,
        model_metrics: Dict[str, Dict[str, float]],
        save_name: str = 'fig20_model_performance.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        models = list(model_metrics.keys())
        all_metrics = sorted({m for d in model_metrics.values() for m in d})
        n_models = len(models)
        n_metrics = len(all_metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 7))
        if n_metrics == 1:
            axes = [axes]

        colors = CATEGORICAL_COLORS[:n_models]

        for mi, metric in enumerate(all_metrics):
            ax = axes[mi]
            vals = [model_metrics[m].get(metric, 0) for m in models]
            bars = ax.bar(range(n_models), vals, color=colors,
                          edgecolor='black', linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                        f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
            ax.set_xticks(range(n_models))
            ax.set_xticklabels(
                [self._truncate(m, 12) for m in models],
                rotation=45, ha='right', fontsize=9,
            )
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

        fig.suptitle('Forecasting Model Performance Comparison',
                     fontsize=14, y=1.02)
        plt.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 21 – CV Box Plots
    # ==================================================================

    def plot_cv_boxplots(
        self,
        cv_scores: Dict[str, List[float]],
        save_name: str = 'fig21_cv_boxplots.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        metrics = list(cv_scores.keys())
        n = len(metrics)

        fig, ax = plt.subplots(figsize=(max(8, n * 2.5), 7))
        data = [cv_scores[m] for m in metrics]
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

        for i, (m, sc) in enumerate(cv_scores.items()):
            jitter = np.random.default_rng(42).normal(i + 1, 0.06, len(sc))
            ax.scatter(jitter, sc, s=50, color='black', alpha=0.5, zorder=5)
            mean_v = np.mean(sc)
            std_v = np.std(sc)
            ylim = ax.get_ylim()
            ax.text(i + 1, max(sc) + 0.03 * (ylim[1] - ylim[0]),
                    f'{mean_v:.3f}±{std_v:.3f}', ha='center', fontsize=9,
                    fontweight='bold')

        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylabel('Score')
        ax.set_title('Cross-Validation Score Distributions', pad=12)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 22 – Prediction Intervals
    # ==================================================================

    def plot_prediction_intervals(
        self,
        predictions: pd.DataFrame,
        lower: pd.DataFrame,
        upper: pd.DataFrame,
        top_n: int = 20,
        save_name: str = 'fig22_prediction_intervals.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        col = predictions.columns[0] if len(predictions.columns) > 0 else None
        if col is None:
            return None

        pred_vals = predictions[col].values
        sorted_idx = np.argsort(pred_vals)[::-1][:top_n]
        names = [predictions.index[i] for i in sorted_idx]

        pred_show = pred_vals[sorted_idx]
        low_show = (lower[col].values[sorted_idx]
                    if col in lower.columns else pred_show * 0.9)
        high_show = (upper[col].values[sorted_idx]
                     if col in upper.columns else pred_show * 1.1)

        fig, ax = plt.subplots(figsize=(13, max(8, top_n * 0.38)))
        y = np.arange(top_n)
        ax.barh(y, high_show - low_show, left=low_show, height=0.5,
                color=PALETTE['royal_blue'], alpha=0.2, edgecolor='none',
                label='95% CI')
        ax.scatter(pred_show, y, s=80, color=PALETTE['royal_blue'],
                   edgecolors='black', linewidths=0.6, zorder=4,
                   label='Point Estimate')
        ax.hlines(y, low_show, high_show, color=PALETTE['royal_blue'],
                  lw=1.5, zorder=3)

        ax.set_yticks(y)
        ax.set_yticklabels(
            [self._truncate(str(n), 18) for n in names], fontsize=9,
        )
        ax.invert_yaxis()
        ax.set_xlabel('Predicted Score')
        ax.set_title(
            f'Prediction with 95% Conformal Intervals — Top {top_n}', pad=12,
        )
        ax.legend(loc='lower right', fontsize=9)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 23 – Rank Change Bubble Chart
    # ==================================================================

    def plot_rank_change_bubble(
        self,
        provinces: List[str],
        current_scores: np.ndarray,
        predicted_scores: np.ndarray,
        prediction_year: int,
        save_name: str = 'fig23_rank_change_bubble.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None
        current_scores = np.asarray(current_scores)
        predicted_scores = np.asarray(predicted_scores)

        curr_rank = np.argsort(np.argsort(-current_scores)) + 1
        pred_rank = np.argsort(np.argsort(-predicted_scores)) + 1
        rank_change = curr_rank.astype(int) - pred_rank.astype(int)

        fig, ax = plt.subplots(figsize=(13, 11))

        sizes = (np.abs(rank_change) + 1) * 30
        scatter = ax.scatter(
            curr_rank, pred_rank, s=sizes, c=rank_change,
            cmap='RdYlGn', alpha=0.8, edgecolors='black',
            linewidths=0.5, zorder=4,
        )

        n = len(provinces)
        ax.plot([1, n], [1, n], 'k--', lw=1.5, alpha=0.5, label='No Change')

        abs_change = np.abs(rank_change)
        top_movers = np.argsort(abs_change)[-8:]
        for idx in top_movers:
            ax.annotate(
                f'{provinces[idx]}\n({rank_change[idx]:+d})',
                (curr_rank[idx], pred_rank[idx]),
                xytext=(8, 8), textcoords='offset points', fontsize=8,
                arrowprops=dict(arrowstyle='->', alpha=0.5),
            )

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Rank Change (+improved)', fontsize=10)

        ax.set_xlabel('Current Rank (2024)')
        ax.set_ylabel(f'Predicted Rank ({prediction_year})')
        ax.set_title(
            f'Rank Change Analysis — Current vs Predicted ({prediction_year})',
            pad=12,
        )
        ax.legend(fontsize=9)
        ax.invert_xaxis()
        ax.invert_yaxis()
        return self._save(fig, save_name)


    # ==================================================================
    #  FIG 16c – Holdout Forecast Comparison (Predicted vs Test, all models)
    # ==================================================================

    def plot_holdout_comparison(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        per_model_predictions: Optional[Dict[str, np.ndarray]] = None,
        entity_names: Optional[List[str]] = None,
        model_contributions: Optional[Dict[str, float]] = None,
        save_name: str = 'fig16c_holdout_comparison.png',
    ) -> Optional[str]:
        """Comprehensive holdout forecast comparison: per-model predicted vs actual.

        This is the primary diagnostic for evaluating how well each base model
        and the Meta-Learner ensemble predict scores on the temporal holdout
        set (the last observed year used as a pseudo-test split).

        The figure contains four panels:

        1. **Top-left — Predicted vs Actual scatter per model** with perfect-line
           reference along with R² and MAE annotations per model.
        2. **Top-right — Residual box/violin per model** showing the error
           distribution (median, IQR, outliers) for every model + ensemble.
        3. **Bottom-left — Per-province residual heatmap** (Model × Province)
           showing signed residuals so analysts can spot which provinces
           each model struggles with.
        4. **Bottom-right — Model ranking bar chart** of holdout R² with
           contribution weight encoded as bar opacity (heavier models darker).
        """
        if not HAS_MATPLOTLIB:
            return None

        y_test = np.asarray(y_test, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        n = len(y_test)
        if n < 3:
            return None

        # Build model dict (include ensemble)
        model_preds: Dict[str, np.ndarray] = {}
        if per_model_predictions:
            for name, arr in per_model_predictions.items():
                model_preds[name] = np.asarray(arr, dtype=float).ravel()[:n]
        model_preds['Meta-Learner'] = y_pred

        model_names = list(model_preds.keys())
        n_models = len(model_names)

        # Contribution weights (for bar opacity)
        contribs = model_contributions or {}

        # Model family colour mapping
        _family_color = {
            'xgboost': '#E74C3C', 'xgb': '#E74C3C', 'catboost': '#E74C3C',
            'random': '#3498DB', 'rf': '#3498DB', 'forest': '#3498DB',
            'neural': '#9B59B6', 'nam': '#9B59B6', 'additive': '#9B59B6',
            'bayesian': '#F39C12', 'bayes': '#F39C12', 'hierarchical': '#F39C12',
            'var': '#1ABC9C', 'panel': '#1ABC9C',
            'meta': '#E67E22', 'learner': '#E67E22', 'ensemble': '#E67E22',
            'quantile': '#E91E63', 'qrf': '#E91E63',
        }
        def _model_color(name: str) -> str:
            nl = name.lower()
            for key, col in _family_color.items():
                if key in nl:
                    return col
            return '#95A5A6'

        colors = [_model_color(m) for m in model_names]

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

            ss_res = np.sum((y_test - pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            mae = np.mean(np.abs(y_test - pred))

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
        residuals_per_model = []
        for name in model_names:
            residuals_per_model.append(y_test - model_preds[name])

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

        # Annotate RMSE & MAE under each box
        for i, resid in enumerate(residuals_per_model):
            rmse = float(np.sqrt(np.mean(resid ** 2)))
            mae = float(np.mean(np.abs(resid)))
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

        if entity_names and len(entity_names) == n:
            prov_labels = entity_names
        else:
            prov_labels = [f'#{i}' for i in range(n)]

        # Sort provinces by ensemble residual magnitude
        ensemble_resid = y_test - y_pred
        sort_idx = np.argsort(np.abs(ensemble_resid))[::-1]
        # Show top 30 at most
        show_n = min(30, n)
        show_idx = sort_idx[:show_n]

        resid_matrix = np.column_stack([
            (y_test - model_preds[m])[show_idx] for m in model_names
        ])  # shape: (show_n, n_models)

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

        r2_scores = []
        for name in model_names:
            pred = model_preds[name]
            ss_res = np.sum((y_test - pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            r2_scores.append(1 - ss_res / (ss_tot + 1e-12))

        # Sort by R²
        sort_order = np.argsort(r2_scores)[::-1]
        sorted_names = [model_names[i] for i in sort_order]
        sorted_r2 = [r2_scores[i] for i in sort_order]
        sorted_colors = [colors[i] for i in sort_order]
        sorted_weights = [contribs.get(model_names[i], 0.0)
                          for i in sort_order]

        # Normalize alphas: ensemble weight → opacity
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

        # Overall stats annotation
        ss_res_ens = np.sum((y_test - y_pred) ** 2)
        ss_tot_ens = np.sum((y_test - y_test.mean()) ** 2)
        r2_ens = 1 - ss_res_ens / (ss_tot_ens + 1e-12)
        mae_ens = np.mean(np.abs(y_test - y_pred))
        rmse_ens = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mape_safe = np.mean(
            np.abs((y_test - y_pred) / np.where(np.abs(y_test) < 1e-9, 1, y_test))
        ) * 100

        stats_text = (
            f'Ensemble Holdout Metrics\n'
            f'{"─" * 28}\n'
            f'R²   = {r2_ens:.6f}\n'
            f'MAE  = {mae_ens:.6f}\n'
            f'RMSE = {rmse_ens:.6f}\n'
            f'MAPE = {mape_safe:.2f}%\n'
            f'N    = {n}'
        )
        fig.text(0.99, 0.01, stats_text, fontsize=9,
                 fontfamily='monospace', va='bottom', ha='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='#F8F8FF',
                           ec='#CCCCCC', alpha=0.95))

        fig.suptitle('Holdout Forecast Comparison — Predicted vs Test Set',
                     fontsize=15, fontweight='bold', y=0.99)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 16b – Ensemble Architecture Flowchart
    # ==================================================================

    def plot_ensemble_architecture(
        self,
        model_names: Optional[List[str]] = None,
        model_contributions: Optional[Dict[str, float]] = None,
        save_name: str = 'fig16b_ensemble_architecture.png',
    ) -> Optional[str]:
        """Data-driven matplotlib flowchart of the 3-tier ensemble pipeline.

        The base-estimator tier is built from the live ``model_names`` list so
        the figure always reflects the actual active model set.

        Tier 1 — Base Estimators (dynamic, derived from ``model_names``)
        Tier 2 — Meta-Learner (stacked generalisation, NNLS weights)
        Tier 3 — Conformal Prediction (distribution-free coverage guarantee)

        Parameters
        ----------
        model_names : list of str, optional
            Ordered list of active base model keys (e.g. ``['CatBoost',
            'BayesianRidge', 'KernelRidge', 'SVR', 'QuantileRF']``).
            If None, derived from ``model_contributions`` keys; if that is also
            None, falls back to the default 5-model roster.
        model_contributions : dict, optional
            ``{model_name: weight}`` from the fitted Meta-Learner.  Used to
            annotate top-3 contributors in the Meta-Learner box and to derive
            ``model_names`` when that argument is not provided.
        save_name : str
            Output filename (relative to the configured figures directory).
        """
        if not HAS_MATPLOTLIB:
            return None

        try:
            from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        except ImportError:
            return None

        fig, ax = plt.subplots(figsize=(16, 11))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 11)
        ax.axis('off')

        # ── colour scheme ──────────────────────────────────────────
        tier_colors = {
            'data':      '#EBF5FB',
            'base':      '#D6EAF8',
            'ensemble':  '#D5F5E3',
            'conformal': '#FDEBD0',
            'output':    '#F9EBEA',
        }
        border_colors = {
            'data':      '#2E86C1',
            'base':      '#1A5276',
            'ensemble':  '#1E8449',
            'conformal': '#CA6F1E',
            'output':    '#922B21',
        }

        def _box(x, y, w, h, label, sublabel='', kind='base', fontsize=9):
            fc = tier_colors[kind]
            ec = border_colors[kind]
            patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                   boxstyle='round,pad=0.08',
                                   fc=fc, ec=ec, lw=2, zorder=3,
                                   transform=ax.transData)
            ax.add_patch(patch)
            ax.text(x, y + (0.12 if sublabel else 0), label,
                    ha='center', va='center', fontsize=fontsize,
                    fontweight='bold', color=ec, zorder=4)
            if sublabel:
                ax.text(x, y - 0.22, sublabel,
                        ha='center', va='center', fontsize=7,
                        color='#555555', style='italic', zorder=4)

        def _arrow(x0, y0, x1, y1, color='#555555'):
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color=color,
                                       lw=1.8, connectionstyle='arc3,rad=0.0'),
                        zorder=2)

        # ── Data source ────────────────────────────────────────────
        _box(8, 10.1, 5.5, 0.85, 'Panel Data (63 Provinces × 14 Years × 29 SC)',
             kind='data', fontsize=9.5)

        # ── Tier 1 – Base Models ───────────────────────────────────
        ax.text(8, 9.35, 'Tier 1 — Base Estimators', ha='center',
                fontsize=9, color='#1A5276', style='italic')

        # Resolve model names: explicit arg > model_contributions keys > default
        _DEFAULT_MODELS = [
            'CatBoost', 'BayesianRidge',
            'KernelRidge', 'SVR', 'QuantileRF',
        ]
        _MODEL_LABELS: Dict[str, tuple] = {
            'CatBoost':     ('CatBoost',    'Symmetric GB'),
            'BayesianRidge':('Bayesian\nRidge', 'Group Lasso'),
            'KernelRidge':  ('Kernel\nRidge', 'RBF L2'),
            'SVR':          ('SVR',          'ε-insensitive'),
            'QuantileRF':   ('Quantile\nRF', 'Distributional'),
        }
        if model_names is not None:
            _active = list(model_names)
        elif model_contributions:
            _active = list(model_contributions.keys())
        else:
            _active = _DEFAULT_MODELS

        n_base = max(len(_active), 1)
        # Dynamic box width: shrink when more models but never exceed 2.2
        _bw = min(2.2, max(1.0, 13.6 / n_base - 0.3))
        xs_base = np.linspace(1.2, 14.8, n_base)

        for xi, mkey in zip(xs_base, _active):
            _lbl, _sub = _MODEL_LABELS.get(mkey, (mkey, ''))
            _box(xi, 8.5, _bw, 1.15, _lbl, _sub, kind='base',
                 fontsize=max(6.5, 8.5 - max(0, n_base - 6) * 0.4))
            _arrow(8, 9.68, xi, 9.08)

        # ── Tier 2 – Meta-Learner ─────────────────────────────────
        ax.text(8, 7.5, 'Tier 2 — Stacked Generalisation', ha='center',
                fontsize=9, color='#1E8449', style='italic')

        _box(8, 6.85, 5.8, 0.9, 'Meta-Learner', 'Optimal linear combination', kind='ensemble')
        for xi in xs_base:
            _arrow(xi, 7.93, 8, 7.30)

        # ── Weights annotation ─────────────────────────────────────
        if model_contributions:
            top3 = sorted(model_contributions.items(), key=lambda x: x[1], reverse=True)[:3]
            w_txt = '  '.join(f'{self._truncate(m,12)}: {w:.2f}' for m, w in top3)
            ax.text(8, 6.35, f'Top contributors: {w_txt}',
                    ha='center', fontsize=7.5, color='#1E8449', style='italic')

        # ── Tier 3 – Conformal Prediction ─────────────────────────
        ax.text(8, 5.7, 'Tier 3 — Uncertainty Calibration', ha='center',
                fontsize=9, color='#CA6F1E', style='italic')

        _box(8, 5.05, 5.8, 0.9,
             'Conformal Prediction', 'Distribution-free coverage guarantee (95%)',
             kind='conformal')
        _arrow(8, 6.40, 8, 5.50)

        # ── Output ─────────────────────────────────────────────────
        _arrow(8, 4.60, 8, 3.90)

        out_cols = [
            ('Point\nForecast', 'Province scores\nfor target year'),
            ('Lower CI\n(2.5%)', 'Conformal lower\nbound'),
            ('Upper CI\n(97.5%)', 'Conformal upper\nbound'),
            ('Uncertainty\nScore', 'Interval width\n/ entropy'),
        ]
        xs_out = np.linspace(3.5, 12.5, 4)
        for xi, (name, sub) in zip(xs_out, out_cols):
            _box(xi, 3.3, 2.2, 1.1, name, sub, kind='output', fontsize=8)
            _arrow(8, 3.90, xi, 3.85)

        # ── MCDM feed ──────────────────────────────────────────────
        ax.text(8, 2.7, '↓ Feeds into MCDM Ranking Pipeline', ha='center',
                fontsize=10, fontweight='bold', color='#555555')

        # ── Title ──────────────────────────────────────────────────
        ax.text(8, 11.0, '', ha='center')
        fig.suptitle('ML Ensemble Forecasting Architecture — 3-Tier Pipeline',
                     fontsize=13, y=0.98, fontweight='bold')
        plt.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 19b – Model Contribution Dot Plot (weight vs CV score)
    # ==================================================================

    def plot_model_contribution_dots(
        self,
        model_contributions: Dict[str, float],
        cross_validation_scores: Optional[Dict[str, List[float]]] = None,
        model_performance: Optional[Dict[str, Dict[str, float]]] = None,
        save_name: str = 'fig19b_model_contribution_dots.png',
    ) -> Optional[str]:
        """Bubble chart: Meta-Learner weight (x) vs CV R² score (y).

        Bubble size encodes the contribution weight; color encodes model family.
        Annotates each model to show the weight–performance trade-off made by
        the Meta-Learner during stacking.
        """
        if not HAS_MATPLOTLIB:
            return None
        if not model_contributions:
            return None

        # Derive CV R² mean for each model
        cv_means: Dict[str, float] = {}
        if cross_validation_scores:
            for m, scores in cross_validation_scores.items():
                if scores:
                    cv_means[m] = float(np.nanmean(scores))
        if not cv_means and model_performance:
            for m, metrics in model_performance.items():
                if 'R2' in metrics:
                    cv_means[m] = float(metrics['R2'])
                elif 'r2' in metrics:
                    cv_means[m] = float(metrics['r2'])

        models = sorted(model_contributions.keys())
        weights = np.array([model_contributions[m] for m in models])
        r2_vals = np.array([cv_means.get(m, 0.0) for m in models])

        # Model family colour mapping
        _family_color = {
            'catboost': '#E74C3C', 'xgboost': '#E74C3C', 'xgb': '#E74C3C',
            'random': '#3498DB', 'rf': '#3498DB', 'forest': '#3498DB',
            'quantile': '#E91E63', 'qrf': '#E91E63',
            'neural': '#9B59B6', 'nam': '#9B59B6', 'additive': '#9B59B6',
            'bayesian': '#F39C12', 'bayes': '#F39C12', 'hierarchical': '#F39C12',
            'var': '#1ABC9C', 'panel': '#1ABC9C',
            'meta': '#E67E22', 'learner': '#E67E22', 'kernel': '#8E44AD',
            'svr': '#C0392B', 'ridge': '#D35400',
        }
        colors_ = []
        for m in models:
            ml = m.lower()
            c = '#95A5A6'
            for key, col in _family_color.items():
                if key in ml:
                    c = col
                    break
            colors_.append(c)

        fig, ax = plt.subplots(figsize=(11, 8))

        sizes = weights / max(weights.max(), 1e-9) * 1200 + 80
        sc = ax.scatter(weights, r2_vals, s=sizes, c=colors_, alpha=0.82,
                        edgecolors='white', linewidths=1.5, zorder=4)

        # Quadrant lines
        mw = float(np.median(weights))
        mr = float(np.median(r2_vals)) if r2_vals.max() > 0 else 0.5
        ax.axvline(mw, color='gray', lw=1.0, ls='--', alpha=0.6)
        ax.axhline(mr, color='gray', lw=1.0, ls='--', alpha=0.6)

        # Quadrant labels
        xlo, xhi = ax.get_xlim() if ax.get_xlim() != (0, 1) else (weights.min(), weights.max())
        for lbl, xi, yi, col in [
            ('High weight\nHigh CV',   0.75, 0.75, '#27AE60'),
            ('Low weight\nHigh CV',    0.25, 0.75, '#2980B9'),
            ('High weight\nLow CV',    0.75, 0.25, '#E67E22'),
            ('Low weight\nLow CV',     0.25, 0.25, '#C0392B'),
        ]:
            ax.text(xi, yi, lbl, ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color=col,
                    alpha=0.40, style='italic', fontweight='bold')

        for i, m in enumerate(models):
            ax.annotate(
                self._truncate(m, 16),
                (weights[i], r2_vals[i]),
                xytext=(6, 6), textcoords='offset points',
                fontsize=8, color='#333333',
            )

        ax.set_xlabel('Meta-Learner Contribution Weight', fontsize=11)
        ax.set_ylabel('Mean Cross-Validation R²', fontsize=11)
        ax.set_title('Model Contribution vs CV Performance — Meta-Learner Diagnostics',
                     pad=12)

        # Size legend
        for w_leg in [0.1, 0.3, 0.5]:
            ax.scatter([], [], s=w_leg / max(weights.max(), 1e-9) * 1200 + 80,
                       c='gray', alpha=0.6, label=f'Weight = {w_leg:.1f}')
        ax.legend(title='Bubble size → weight', fontsize=9,
                  title_fontsize=9, loc='lower right')

        plt.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 22b – Conformal Coverage Calibration Curve
    # ==================================================================

    def plot_conformal_coverage(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        alpha_levels: Optional[List[float]] = None,
        save_name: str = 'fig22b_conformal_coverage.png',
    ) -> Optional[str]:
        """Reliability calibration curve for conformal prediction intervals.

        For each miscoverage level α, derives the conformal interval half-width
        as the (1−α) quantile of |residuals| on the test set, then measures
        empirical coverage.  A well-calibrated conformal predictor hugs the y=x
        diagonal (nominal = empirical coverage).
        """
        if not HAS_MATPLOTLIB:
            return None

        y_test = np.asarray(y_test, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        if len(y_test) < 4:
            return None

        residuals = np.abs(y_test - y_pred)
        if alpha_levels is None:
            alpha_levels = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25,
                            0.30, 0.35, 0.40, 0.50, 0.60, 0.70]

        nominal_cov, empirical_cov, interval_widths = [], [], []
        for alpha in alpha_levels:
            nominal = 1.0 - alpha
            # Conformal guarantee: use ceiling quantile (finite-sample correction)
            n = len(residuals)
            q_idx = min(int(np.ceil((n + 1) * nominal)) - 1, n - 1)
            q_val = float(np.sort(residuals)[q_idx])
            emp_cov = float(np.mean(residuals <= q_val))
            nominal_cov.append(nominal)
            empirical_cov.append(emp_cov)
            interval_widths.append(2.0 * q_val)

        nominal_cov = np.array(nominal_cov)
        empirical_cov = np.array(empirical_cov)
        interval_widths = np.array(interval_widths)

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # — Panel 1: Coverage calibration curve ———————————————————
        ax = axes[0]
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Perfect calibration')
        ax.fill_between([0, 1], [0, 1], [0.05, 1.05], alpha=0.07,
                        color='green', label='±5% band')
        ax.fill_between([0, 1], [-0.05, 0.95], [0, 1], alpha=0.07, color='green')

        sc_ = ax.scatter(nominal_cov, empirical_cov, c=interval_widths,
                         cmap='YlOrRd', s=120, zorder=4,
                         edgecolors='black', linewidths=0.7)
        ax.plot(nominal_cov, empirical_cov, color=PALETTE.get('royal_blue', '#4472C4'),
                lw=2.0, alpha=0.8, zorder=3)

        # Mark the standard 95% point
        idx_95 = np.argmin(np.abs(nominal_cov - 0.95))
        ax.scatter([nominal_cov[idx_95]], [empirical_cov[idx_95]],
                   s=220, color='red', zorder=5, marker='*',
                   label=f'95% CI → emp. {empirical_cov[idx_95]:.1%}')

        cb = fig.colorbar(sc_, ax=ax, shrink=0.7)
        cb.set_label('Interval Width (2·q)', fontsize=9)

        ax.set_xlabel('Nominal Coverage (1 − α)', fontsize=11)
        ax.set_ylabel('Empirical Coverage', fontsize=11)
        ax.set_title('Conformal Prediction — Coverage Calibration Curve', pad=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # — Panel 2: Interval width vs nominal coverage —————————————
        ax2 = axes[1]
        ax2.fill_between(nominal_cov, 0, interval_widths,
                         alpha=0.25, color=PALETTE.get('royal_blue', '#4472C4'))
        ax2.plot(nominal_cov, interval_widths,
                 color=PALETTE.get('royal_blue', '#4472C4'), lw=2.0, marker='o',
                 ms=7, mec='black', mew=0.7)
        ax2.axvline(0.95, color='red', ls='--', lw=1.5, label='95% nominal')

        for nc, iw in zip(nominal_cov, interval_widths):
            ax2.text(nc, iw + 0.002, f'{iw:.3f}', ha='center', fontsize=7,
                     color='#333333')

        ax2.set_xlabel('Nominal Coverage (1 − α)', fontsize=11)
        ax2.set_ylabel('Interval Width', fontsize=11)
        ax2.set_title('Conformal Interval Width vs. Coverage Level', pad=10)
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        fig.suptitle('Conformal Prediction Diagnostics', fontsize=13, y=1.01)
        plt.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 23b – Province Score Comparison: Actual vs Forecast
    # ==================================================================

    def plot_province_forecast_comparison(
        self,
        provinces: List[str],
        current_scores: np.ndarray,
        pred_df: pd.DataFrame,
        intervals: Optional[Dict[str, pd.DataFrame]] = None,
        top_n: int = 20,
        prediction_year: int = 2025,
        save_name: str = 'fig23b_province_forecast_comparison.png',
    ) -> Optional[str]:
        """Grouped bar chart: current score vs forecast for top-N provinces.

        Each province gets two bars (current = steel-blue, predicted = teal)
        with conformal interval error bars on the predicted bar.
        """
        if not HAS_MATPLOTLIB:
            return None
        current_scores = np.asarray(current_scores, dtype=float)
        if pred_df is None or pred_df.empty:
            return None

        # Resolve prediction column
        pred_col = pred_df.columns[0]
        pred_scores = pred_df[pred_col].values

        # Align provinces with predictions index
        prov_to_curr: Dict[str, float] = {p: s for p, s in zip(provinces, current_scores)}
        prov_to_pred: Dict[str, float] = {}
        for idx_val, score in zip(pred_df.index, pred_scores):
            prov_to_pred[str(idx_val)] = float(score)

        common_prov = [p for p in provinces if p in prov_to_pred]
        if not common_prov:
            # Fallback: positional alignment
            n_align = min(len(provinces), len(pred_scores))
            common_prov = provinces[:n_align]
            prov_to_pred = {p: float(pred_scores[i])
                            for i, p in enumerate(common_prov)}

        # Sort by predicted score, take top-N
        common_prov = sorted(common_prov,
                             key=lambda p: prov_to_pred.get(p, 0), reverse=True)[:top_n]
        n_show = len(common_prov)

        curr_vals = np.array([prov_to_curr.get(p, 0.0) for p in common_prov])
        pred_vals = np.array([prov_to_pred.get(p, 0.0) for p in common_prov])

        # Error bars from prediction intervals
        pred_lo = pred_hi = None
        if intervals:
            lower_df = intervals.get('lower')
            upper_df = intervals.get('upper')
            if lower_df is not None and not lower_df.empty:
                lo_col = lower_df.columns[0] if pred_col not in lower_df.columns else pred_col
                pred_lo = np.array([
                    lower_df.loc[p, lo_col] if p in lower_df.index else pred_vals[i]
                    for i, p in enumerate(common_prov)
                ], dtype=float)
            if upper_df is not None and not upper_df.empty:
                hi_col = upper_df.columns[0] if pred_col not in upper_df.columns else pred_col
                pred_hi = np.array([
                    upper_df.loc[p, hi_col] if p in upper_df.index else pred_vals[i]
                    for i, p in enumerate(common_prov)
                ], dtype=float)

        x = np.arange(n_show)
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(13, n_show * 0.75), 8))

        color_curr = PALETTE.get('royal_blue', '#4472C4')
        color_pred = PALETTE.get('emerald', '#17B169')

        bars_curr = ax.bar(x - width / 2, curr_vals, width, color=color_curr,
                           alpha=0.75, edgecolor='white', linewidth=0.6,
                           label='Current Score (2024)', zorder=3)
        err_low = (pred_vals - pred_lo) if pred_lo is not None else None
        err_hi = (pred_hi - pred_vals) if pred_hi is not None else None
        yerr_ = (np.array([err_low, err_hi]) if err_low is not None and err_hi is not None
                 else None)
        bars_pred = ax.bar(x + width / 2, pred_vals, width, color=color_pred,
                           alpha=0.75, edgecolor='white', linewidth=0.6,
                           label=f'Predicted Score ({prediction_year})',
                           yerr=yerr_,
                           error_kw=dict(ecolor='#E74C3C', capsize=4, lw=1.5),
                           zorder=3)

        # Score change arrows / annotations
        for i, (cv, pv) in enumerate(zip(curr_vals, pred_vals)):
            delta = pv - cv
            sym = '▲' if delta > 0 else '▼'
            col = '#27AE60' if delta > 0 else '#E74C3C'
            ax.text(x[i], max(cv, pv) + 0.015, f'{sym}{abs(delta):.3f}',
                    ha='center', fontsize=7.5, color=col, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([self._truncate(p, 14) for p in common_prov],
                           rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Score')
        ax.set_title(
            f'Province Score Comparison — Current (2024) vs. Forecast ({prediction_year})',
            pad=12)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 23c – Score Trajectory: Historical + Forecast
    # ==================================================================

    def plot_score_trajectory(
        self,
        provinces: List[str],
        scores: np.ndarray,
        pred_df: pd.DataFrame,
        intervals: Optional[Dict[str, pd.DataFrame]] = None,
        panel_years: Optional[List[int]] = None,
        target_year: int = 2025,
        top_n: int = 10,
        save_name: str = 'fig23c_score_trajectory.png',
    ) -> Optional[str]:
        """Province score trajectories across time with forecast CI extension.

        Shows the top-N provinces (by predicted score) as individual lines.
        The final (forecast) point is shown with a shaded CI fan.  A dashed
        vertical separator marks the observed/forecast boundary.
        """
        if not HAS_MATPLOTLIB:
            return None
        if pred_df is None or pred_df.empty:
            return None

        pred_col = pred_df.columns[0]
        pred_scores_all = pred_df[pred_col].to_dict()

        # Identify top-N by predicted score
        all_prov_pred = {str(k): float(v) for k, v in pred_scores_all.items()}
        avail = [p for p in provinces if p in all_prov_pred]
        if not avail:
            avail = provinces
            avail_pred = {p: float(pred_scores_all.get(p, 0.0)) for p in avail}
        else:
            avail_pred = {p: all_prov_pred[p] for p in avail}

        top_prov = sorted(avail_pred.keys(), key=lambda p: avail_pred[p],
                          reverse=True)[:top_n]
        if not top_prov:
            return None

        # Map province to current score (last observed year)
        prov_to_curr = {p: float(s) for p, s in zip(provinces, scores)}

        # Prediction interval values
        prov_lo: Dict[str, float] = {}
        prov_hi: Dict[str, float] = {}
        if intervals:
            lower_df = intervals.get('lower')
            upper_df = intervals.get('upper')
            if lower_df is not None and not lower_df.empty:
                lo_col = lower_df.columns[0]
                for p in top_prov:
                    if p in lower_df.index:
                        prov_lo[p] = float(lower_df.loc[p, lo_col])
            if upper_df is not None and not upper_df.empty:
                hi_col = upper_df.columns[0]
                for p in top_prov:
                    if p in upper_df.index:
                        prov_hi[p] = float(upper_df.loc[p, hi_col])

        # Build timeline
        last_year = (max(panel_years) if panel_years else target_year - 1)
        years_line = [last_year, target_year]

        cmap_t = plt.colormaps['tab10']
        fig, ax = plt.subplots(figsize=(12, 8))

        # Vertical separator: observed vs forecast
        ax.axvline(last_year + 0.5, color='gray', lw=1.5, ls='--', alpha=0.6)
        ax.text(last_year + 0.55, 0.98, 'Forecast →', transform=ax.get_xaxis_transform(),
                fontsize=9, color='gray', style='italic', va='top')
        ax.text(last_year + 0.45, 0.98, '← Observed', transform=ax.get_xaxis_transform(),
                fontsize=9, color='gray', style='italic', va='top', ha='right')

        for ri, prov in enumerate(top_prov):
            color = cmap_t(ri / max(top_n, 1))
            curr_s = prov_to_curr.get(prov, 0.0)
            pred_s = avail_pred.get(prov, curr_s)

            # Line from last observed to forecast
            ax.plot(years_line, [curr_s, pred_s],
                    color=color, lw=2.2, alpha=0.85,
                    marker='o', ms=7, mec='white', mew=1.2, zorder=3)

            # Forecast CI fan
            lo_s = prov_lo.get(prov, pred_s)
            hi_s = prov_hi.get(prov, pred_s)
            ax.fill_between([last_year, target_year],
                            [curr_s, lo_s], [curr_s, hi_s],
                            alpha=0.12, color=color, zorder=1)

            # End labels
            ax.text(target_year + 0.1, pred_s, self._truncate(prov, 14),
                    va='center', fontsize=8, color=color)

        ax.set_xticks(years_line)
        ax.set_xticklabels([str(y) for y in years_line], fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(
            f'Score Trajectory — Top {top_n} Provinces by Forecast ({target_year})',
            pad=12)

        # Legend
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=cmap_t(i / max(top_n, 1)),
                          lw=2, label=self._truncate(p, 16))
                   for i, p in enumerate(top_prov)]
        ax.legend(handles=handles, fontsize=8, loc='lower right',
                  ncol=2, framealpha=0.85)
        plt.tight_layout()
        return self._save(fig, save_name)

    # ================================================================
    # FIG 17b – Residual Distributions per Model + Ensemble
    # ================================================================

    def plot_residual_distributions(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        per_model_predictions: Optional[Dict[str, np.ndarray]] = None,
        save_name: str = 'fig17b_residual_distributions.png',
    ) -> Optional[str]:
        """Histogram + KDE + Normal reference for OOF residuals per model.

        Each panel covers one base model (from *per_model_predictions*) plus a
        final panel for the Meta-Learner ensemble.  The Normal reference uses
        the empirical mean and standard deviation of that model's residuals,
        making systematic bias and heavy tails immediately visible.

        Args:
            y_test: Raveled 1-D array of OOF actual governance-index values.
            y_pred: Raveled 1-D array of OOF Meta-Learner ensemble predictions
                (must be same length as *y_test*).
            per_model_predictions: Optional mapping {model_name →
                prediction array of shape (n_oof, n_outputs)}.  Each model
                gets its own panel; the ensemble is always added last.
            save_name: Output filename.

        Returns:
            Absolute path to the saved PNG, or *None* if matplotlib is
            unavailable.
        """
        if not HAS_MATPLOTLIB:
            return None

        y_t = np.asarray(y_test, dtype=float).ravel()
        y_p = np.asarray(y_pred, dtype=float).ravel()

        # Ordered list of (label, colour, 1-D prediction vector)
        panels: List[tuple] = []
        if per_model_predictions:
            for nm, preds in per_model_predictions.items():
                p = np.asarray(preds, dtype=float).ravel()
                if len(p) == len(y_t):
                    panels.append((nm, _mcolor(nm), p))
        if len(y_p) == len(y_t):
            panels.append(('Meta-Learner', PALETTE['amber'], y_p))

        n = len(panels)
        if n == 0:
            return None

        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 4.2 * nrows),
            squeeze=False,
        )
        axes_flat = axes.ravel()

        for idx, (nm, color, pred) in enumerate(panels):
            ax = axes_flat[idx]
            res = (y_t - pred)
            ok  = np.isfinite(res)
            res = res[ok]
            if len(res) < 5:
                ax.set_visible(False)
                continue

            mu    = float(res.mean())
            sigma = float(res.std())
            skew  = float(
                ((res - mu) ** 3).mean() / (sigma ** 3 + 1e-15)
            )
            n_bins = min(35, max(8, len(res) // 8))

            ax.hist(
                res, bins=n_bins, density=True,
                alpha=0.45, color=color,
                edgecolor='white', linewidth=0.5, label='Residuals',
            )
            kde_x = np.linspace(res.min() - sigma, res.max() + sigma, 250)
            if HAS_SCIPY and sp_stats is not None and len(res) > 3:
                try:
                    kde = sp_stats.gaussian_kde(res, bw_method='scott')
                    ax.plot(kde_x, kde(kde_x), color=color,
                            lw=2.2, label='KDE')
                except Exception:
                    pass
                if sigma > 0:
                    ax.plot(
                        kde_x,
                        sp_stats.norm.pdf(kde_x, mu, sigma),
                        'k--', lw=1.5, alpha=0.75, label='Normal ref',
                    )
            ax.axvline(0, color='#555555', ls=':', lw=1.2, alpha=0.6)
            ax.set_title(
                f'{nm}\nμ={mu:.3f}   σ={sigma:.3f}   skew={skew:.2f}',
                fontsize=9, fontweight='bold',
            )
            ax.set_xlabel('Residual (actual − predicted)', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.tick_params(labelsize=7)
            if idx == 0:
                ax.legend(fontsize=7, framealpha=0.7)

        for idx in range(n, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(
            'OOF Residual Distributions — Base Models + Meta-Learner',
            fontsize=13, fontweight='bold', y=1.01,
        )
        plt.tight_layout()
        return self._save(fig, save_name)

    # ================================================================
    # FIG 18b – Per-Model Feature Importance Heatmap
    # ================================================================

    def plot_per_model_importance_heatmap(
        self,
        per_model_importance: Dict[str, np.ndarray],
        feature_names: Optional[List[str]] = None,
        top_n: int = 20,
        save_name: str = 'fig18b_per_model_importance_heatmap.png',
    ) -> Optional[str]:
        """Heatmap of feature importances with rows = features, columns = models.

        Each model's column is normalised to [0, 1] so within-model relative
        importance is directly comparable across models with different scales.
        Rows are the top-*top_n* features ranked by **maximum importance across
        any single model**, ensuring the features that matter most to at least
        one model are always represented.

        Args:
            per_model_importance: Mapping {model_name →
                ndarray of shape (n_original_features, n_outputs)}.  Arrays
                are produced by the reducer-aware C1 logic in stage6.
            feature_names: Optional list of length n_original_features.  When
                omitted, indices are used for row labels.
            top_n: Number of features to show (default 20).
            save_name: Output filename.

        Returns:
            Absolute path to the saved PNG, or *None* if matplotlib is
            unavailable or *per_model_importance* is empty.
        """
        if not HAS_MATPLOTLIB or not per_model_importance:
            return None

        model_names = list(per_model_importance.keys())
        # Aggregate per-model importance to 1-D (|mean| across outputs)
        agg: Dict[str, np.ndarray] = {}
        for nm, arr in per_model_importance.items():
            a = np.asarray(arr, dtype=float)
            if a.ndim == 1:
                agg[nm] = np.abs(a)
            elif a.ndim == 2:
                agg[nm] = np.abs(a).mean(axis=1)
            else:
                continue

        if not agg:
            return None

        n_feat = len(next(iter(agg.values())))
        if feature_names is None or len(feature_names) != n_feat:
            feature_names = [f'Feature {i}' for i in range(n_feat)]

        # Select top_n by maximum importance across any model
        imp_matrix = np.column_stack([agg[nm] for nm in model_names
                                       if nm in agg])   # (n_feat, n_models)
        row_max = imp_matrix.max(axis=1)
        top_idx = np.argsort(row_max)[::-1][:top_n]
        top_feats = [self._truncate(feature_names[i], 28) for i in top_idx]
        sub = imp_matrix[top_idx, :]   # (top_n, n_models)

        # Column-normalise so each model spans [0, 1]
        col_min = sub.min(axis=0)
        col_rng = sub.max(axis=0) - col_min
        col_rng[col_rng < 1e-15] = 1.0
        sub_norm = (sub - col_min) / col_rng

        n_show   = sub_norm.shape[0]
        n_models = sub_norm.shape[1]
        fig, ax  = plt.subplots(
            figsize=(max(8, n_models * 1.8), max(6, n_show * 0.42))
        )

        im = ax.imshow(
            sub_norm, aspect='auto', cmap='YlOrRd',
            vmin=0, vmax=1, interpolation='nearest',
        )
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(
            [nm for nm in model_names if nm in agg],
            rotation=35, ha='right', fontsize=9,
        )
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(top_feats, fontsize=8)

        # Annotate cells
        thresh = 0.55
        for r in range(n_show):
            for c in range(n_models):
                val = sub_norm[r, c]
                txt_col = 'white' if val > thresh else '#333333'
                ax.text(
                    c, r, f'{val:.2f}',
                    ha='center', va='center',
                    fontsize=7, color=txt_col,
                )
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label('Normalised importance (per-model)', fontsize=8)
        ax.set_title(
            f'Feature Importance Heatmap — Top-{n_show} Features',
            fontsize=12, fontweight='bold', pad=12,
        )
        plt.tight_layout()
        return self._save(fig, save_name)

    # ================================================================
    # FIG 20b – Multi-Metric Radar Chart (Spider / Polar)
    # ================================================================

    def plot_model_metric_radar(
        self,
        model_performance: Dict[str, Dict[str, float]],
        per_model_predictions: Optional[Dict[str, np.ndarray]] = None,
        y_test: Optional[np.ndarray] = None,
        save_name: str = 'fig20b_model_metric_radar.png',
    ) -> Optional[str]:
        """Radar (spider) chart comparing models across five performance metrics.

        When *per_model_predictions* and *y_test* are provided, the five axes
        are: **R²**, **Pearson r**, **1 − norm(RMSE)**, **1 − norm(MAE)**,
        and **1 − norm(|bias|)** — all normalised to [0, 1] so 1 is always
        better.  When predictions are unavailable, the chart falls back to a
        bar chart of ``mean_r2`` from *model_performance*.

        Args:
            model_performance: Dict mapping model_name → metric dict (must
                contain at least ``'mean_r2'``).
            per_model_predictions: Optional {model_name → (n_oof, n_outputs)
                ndarray}.  Enables full five-axis radar.
            y_test: Raveled 1-D OOF actual values aligned with each model's
                raveled OOF predictions.  Required for full radar.
            save_name: Output filename.

        Returns:
            Absolute path to the saved PNG, or *None*.
        """
        if not HAS_MATPLOTLIB:
            return None
        if not model_performance:
            return None

        models = [
            nm for nm in model_performance
            if not np.isnan(model_performance[nm].get('mean_r2', np.nan))
        ]
        if not models:
            return None

        # ── Full radar when predictions + actuals are available ───────────
        if (
            per_model_predictions is not None
            and y_test is not None
            and len(y_test) > 0
        ):
            y_t = np.asarray(y_test, dtype=float)
            metric_data: Dict[str, Dict[str, float]] = {}
            for nm in models:
                if nm not in per_model_predictions:
                    continue
                p = np.asarray(
                    per_model_predictions[nm], dtype=float
                ).ravel()
                if len(p) != len(y_t):
                    continue
                ok = np.isfinite(p) & np.isfinite(y_t)
                if ok.sum() < 5:
                    continue
                yt_ok, yp_ok = y_t[ok], p[ok]
                residuals = yt_ok - yp_ok
                try:
                    from sklearn.metrics import (
                        r2_score as _r2,
                        mean_squared_error as _mse,
                        mean_absolute_error as _mae,
                    )
                    r2   = float(_r2(yt_ok, yp_ok))
                    rmse = float(np.sqrt(_mse(yt_ok, yp_ok)))
                    mae  = float(_mae(yt_ok, yp_ok))
                except Exception:
                    r2   = float(1 - np.var(residuals) / (np.var(yt_ok) + 1e-15))
                    rmse = float(np.sqrt(np.mean(residuals ** 2)))
                    mae  = float(np.mean(np.abs(residuals)))
                if HAS_SCIPY and sp_stats is not None:
                    try:
                        corr, _ = sp_stats.pearsonr(yt_ok, yp_ok)
                    except Exception:
                        corr = float(np.corrcoef(yt_ok, yp_ok)[0, 1])
                else:
                    corr = float(np.corrcoef(yt_ok, yp_ok)[0, 1])
                bias = float(residuals.mean())
                metric_data[nm] = {
                    'r2': r2, 'rmse': rmse, 'mae': mae,
                    'corr': corr, 'absbias': abs(bias),
                }

            if len(metric_data) < 2:
                # Fall through to simple bar chart below
                per_model_predictions = None
            else:
                metric_names = ['R²', 'Pearson r', '1-RMSE', '1-MAE', '1-|Bias|']
                n_ax = len(metric_names)
                angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False)
                angles_c = np.append(angles, angles[0])

                def _norm_inv(vals: np.ndarray) -> np.ndarray:
                    mn, mx = np.nanmin(vals), np.nanmax(vals)
                    if mx == mn:
                        return np.full_like(vals, 0.5)
                    return 1.0 - (vals - mn) / (mx - mn)

                def _norm(vals: np.ndarray) -> np.ndarray:
                    mn, mx = np.nanmin(vals), np.nanmax(vals)
                    if mx == mn:
                        return np.full_like(vals, 0.5)
                    return (vals - mn) / (mx - mn)

                nm_list   = list(metric_data.keys())
                r2_vals   = _norm(np.array([metric_data[m]['r2']      for m in nm_list]))
                cr_vals   = _norm(np.array([metric_data[m]['corr']    for m in nm_list]))
                rm_vals   = _norm_inv(np.array([metric_data[m]['rmse']    for m in nm_list]))
                ma_vals   = _norm_inv(np.array([metric_data[m]['mae']     for m in nm_list]))
                bi_vals   = _norm_inv(np.array([metric_data[m]['absbias'] for m in nm_list]))

                fig, ax = plt.subplots(
                    figsize=(9, 9),
                    subplot_kw={'projection': 'polar'},
                )
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_xticks(angles)
                ax.set_xticklabels(metric_names, fontsize=11)
                ax.set_ylim(0, 1)
                ax.set_yticks([0.25, 0.5, 0.75, 1.0])
                ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'],
                                   fontsize=7, color='#555555')
                ax.grid(color='#CCCCCC', linestyle='--', linewidth=0.5)

                cmap_r = plt.get_cmap('tab10')
                for i, nm in enumerate(nm_list):
                    vals = np.array([r2_vals[i], cr_vals[i],
                                     rm_vals[i], ma_vals[i], bi_vals[i]])
                    vals_c = np.append(vals, vals[0])
                    color  = _mcolor(nm, cmap_r(i % 10))
                    ax.plot(angles_c, vals_c, lw=2, color=color, label=nm)
                    ax.fill(angles_c, vals_c, alpha=0.10, color=color)

                ax.legend(
                    loc='upper right',
                    bbox_to_anchor=(1.35, 1.15),
                    fontsize=9,
                )
                ax.set_title(
                    'Multi-Metric Performance Radar',
                    fontsize=13, fontweight='bold', pad=20,
                )
                return self._save(fig, save_name)

        # ── Fallback: simple bar chart of mean R² ─────────────────────────
        r2s   = [model_performance[nm].get('mean_r2', np.nan) for nm in models]
        stds  = [model_performance[nm].get('std_r2',  0.0)    for nm in models]
        colors = [_mcolor(nm, CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)])
                  for i, nm in enumerate(models)]

        fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.6), 6))
        xpos = np.arange(len(models))
        ax.bar(xpos, r2s, color=colors, alpha=0.8,
               yerr=stds, capsize=5, error_kw={'elinewidth': 1.5})
        ax.set_xticks(xpos)
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Mean R²', fontsize=10)
        ax.set_title('Model Cross-Validation R² (walk-forward OOF)',
                     fontsize=12, fontweight='bold')
        ax.axhline(0, color='#555555', lw=0.8, ls='--')
        ax.set_ylim(min(0, min(r2s) - 0.05), max(r2s) * 1.15 + 0.05)
        plt.tight_layout()
        return self._save(fig, save_name)

    # ================================================================
    # FIG 20c – Bootstrap Metric Confidence Intervals
    # ================================================================

    def plot_bootstrap_metric_ci(
        self,
        y_test: np.ndarray,
        per_model_predictions: Optional[Dict[str, np.ndarray]] = None,
        y_pred_ensemble: Optional[np.ndarray] = None,
        n_bootstrap: int = 200,
        metric: str = 'r2',
        save_name: str = 'fig20c_bootstrap_metric_ci.png',
    ) -> Optional[str]:
        """Horizontal error-bar chart showing bootstrap 95 % CI for each model.

        For each model (and the Meta-Learner ensemble), *n_bootstrap* bootstrap
        resamples of the OOF (y_test, predictions) pairs are drawn and the
        chosen *metric* is computed.  The result reveals whether apparent
        performance differences are statistically meaningful.

        Args:
            y_test: Raveled 1-D OOF actual values.
            per_model_predictions: Optional {model_name → (n_oof, n_outputs)
                ndarray}.
            y_pred_ensemble: Raveled 1-D ensemble OOF predictions.  If
                *per_model_predictions* is also given, the ensemble is added
                as an extra entry.
            n_bootstrap: Number of bootstrap resamples (default 200).
            metric: One of ``'r2'``, ``'rmse'``, or ``'mae'``.
            save_name: Output filename.

        Returns:
            Absolute path to the saved PNG, or *None*.
        """
        if not HAS_MATPLOTLIB:
            return None

        y_t = np.asarray(y_test, dtype=float).ravel()
        n   = len(y_t)
        if n < 5:
            return None

        # Build {name: pred_1d} dict
        all_preds: Dict[str, np.ndarray] = {}
        if per_model_predictions:
            for nm, preds in per_model_predictions.items():
                p = np.asarray(preds, dtype=float).ravel()
                if len(p) == n:
                    all_preds[nm] = p
        if y_pred_ensemble is not None:
            p_ens = np.asarray(y_pred_ensemble, dtype=float).ravel()
            if len(p_ens) == n:
                all_preds['Meta-Learner'] = p_ens

        if not all_preds:
            return None

        def _compute(yt: np.ndarray, yp: np.ndarray) -> float:
            ok = np.isfinite(yt) & np.isfinite(yp)
            if ok.sum() < 3:
                return np.nan
            a, b = yt[ok], yp[ok]
            if metric == 'r2':
                ss_res = np.sum((a - b) ** 2)
                ss_tot = np.sum((a - np.mean(a)) ** 2)
                return float(1 - ss_res / (ss_tot + 1e-15))
            elif metric == 'rmse':
                return float(np.sqrt(np.mean((a - b) ** 2)))
            else:
                return float(np.mean(np.abs(a - b)))

        rng = np.random.RandomState(42)
        results: Dict[str, tuple] = {}
        for nm, pred in all_preds.items():
            boot = np.zeros(n_bootstrap)
            for k in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                boot[k] = _compute(y_t[idx], pred[idx])
            valid = boot[np.isfinite(boot)]
            if len(valid) == 0:
                continue
            results[nm] = (
                float(valid.mean()),
                float(np.percentile(valid, 2.5)),
                float(np.percentile(valid, 97.5)),
            )

        if not results:
            return None

        # Sort by mean metric (descending for r2, ascending for error)
        key_fn  = (lambda x: x[1][0]) if metric == 'r2' else (lambda x: -x[1][0])
        ordered = sorted(results.items(), key=key_fn, reverse=False)
        names   = [x[0] for x in ordered]
        means   = np.array([x[1][0] for x in ordered])
        lo      = np.array([x[1][1] for x in ordered])
        hi      = np.array([x[1][2] for x in ordered])

        fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.55 + 2)))
        ypos    = np.arange(len(names))
        colors  = [_mcolor(nm, CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)])
                   for i, nm in enumerate(names)]

        ax.barh(ypos, means, color=colors, alpha=0.75, height=0.6)
        ax.errorbar(
            means, ypos,
            xerr=[means - lo, hi - means],
            fmt='none', color='#444444',
            elinewidth=1.8, capsize=5, capthick=1.5,
        )
        for i, (m, l_, h_) in enumerate(zip(means, lo, hi)):
            ax.text(
                h_ + (hi.max() - lo.min()) * 0.01,
                i,
                f'{m:.3f} [{l_:.3f}, {h_:.3f}]',
                va='center', fontsize=8,
            )
        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=9)
        lbl = {'r2': 'R²', 'rmse': 'RMSE', 'mae': 'MAE'}.get(metric, metric.upper())
        ax.set_xlabel(f'{lbl} (bootstrap mean ± 95 % CI)', fontsize=10)
        ax.set_title(
            f'Bootstrap Performance CI — {lbl}  (B={n_bootstrap})',
            fontsize=12, fontweight='bold',
        )
        if metric == 'r2':
            ax.axvline(0, color='#555555', lw=0.8, ls='--')
        plt.tight_layout()
        return self._save(fig, save_name)

    # ================================================================
    # FIG 22c – Interval Calibration Scatter
    # ================================================================

    def plot_interval_calibration_scatter(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        lower: 'pd.DataFrame',
        upper: 'pd.DataFrame',
        entity_names: Optional[List[str]] = None,
        save_name: str = 'fig22c_interval_calibration_scatter.png',
    ) -> Optional[str]:
        """Scatter of prediction interval width vs. entity historical error.

        For each unique entity, the historical **mean absolute OOF error** is
        plotted against the **mean prediction interval width** for that entity
        in the target year.  Points lying below the 45° reference line are
        over-confident (interval narrower than prediction error); points above
        are conservative.  A polynomial trend line summarises the relationship.

        Args:
            y_test: Raveled 1-D OOF actual values.
            y_pred: Raveled 1-D OOF ensemble predictions (same length).
            lower: DataFrame (entities × outputs) of lower conformal bounds.
            upper: DataFrame with the same index as *lower*, upper bounds.
            entity_names: Optional list of entity labels aligned with
                *y_test* / *y_pred* (length = len(y_test)).  When provided,
                errors are averaged per entity before plotting.
            save_name: Output filename.

        Returns:
            Absolute path to the saved PNG, or *None*.
        """
        if not HAS_MATPLOTLIB:
            return None

        y_t = np.asarray(y_test, dtype=float).ravel()
        y_p = np.asarray(y_pred, dtype=float).ravel()
        ok  = np.isfinite(y_t) & np.isfinite(y_p)
        if ok.sum() < 5:
            return None

        # ── Per-entity mean absolute OOF error ───────────────────────────
        abs_err_all = np.abs(y_t[ok] - y_p[ok])
        if entity_names is not None and len(entity_names) == len(y_t):
            from collections import defaultdict as _dd
            ent_arr = np.asarray(entity_names)[ok]
            ent_err: Dict[str, list] = _dd(list)
            for ent, err in zip(ent_arr, abs_err_all):
                ent_err[ent].append(err)
            shared_ents = sorted(ent_err.keys())
            ent_mean_err = np.array([np.mean(ent_err[e]) for e in shared_ents])
        else:
            # Treat each sample as its own "entity"
            shared_ents = [str(i) for i in range(len(abs_err_all))]
            ent_mean_err = abs_err_all

        # ── Per-entity mean interval width from prediction bounds ─────────
        if not isinstance(lower, pd.DataFrame) or not isinstance(upper, pd.DataFrame):
            return None
        try:
            widths = (upper - lower)
            ent_width = widths.mean(axis=1)   # Series: entity → mean width
        except Exception:
            return None

        # Intersect with entities that have OOF errors
        common = [
            e for e in shared_ents
            if e in ent_width.index
        ]
        if len(common) < 5:
            return None

        x_vals = np.array([float(ent_width.loc[e]) for e in common])
        y_vals = np.array([
            float(ent_mean_err[shared_ents.index(e)]) for e in common
        ])
        fin = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_vals, y_vals = x_vals[fin], y_vals[fin]
        if len(x_vals) < 5:
            return None

        # Coverage fractions
        above   = (y_vals > x_vals).mean()
        covered = 1.0 - above

        fig, ax = plt.subplots(figsize=(9, 8))
        sc = ax.scatter(
            x_vals, y_vals,
            c=y_vals - x_vals, cmap='RdYlGn_r',
            alpha=0.75, s=55, edgecolors='white', linewidths=0.5,
        )
        plt.colorbar(sc, ax=ax, label='Error − Width  (< 0 = well-covered)', fraction=0.03)

        # 45° reference line
        lim = max(x_vals.max(), y_vals.max()) * 1.1
        ax.plot([0, lim], [0, lim], color='#888888', ls='--',
                lw=1.5, label='45° reference')

        # Polynomial (degree-2) trend
        try:
            coeffs = np.polyfit(x_vals, y_vals, deg=2)
            xs_trend = np.linspace(x_vals.min(), x_vals.max(), 200)
            ax.plot(xs_trend, np.polyval(coeffs, xs_trend),
                    color=PALETTE['coral'], lw=2.2, label='Trend (poly-2)')
        except Exception:
            pass

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Mean Prediction Interval Width (target year)', fontsize=11)
        ax.set_ylabel('Mean Absolute OOF Error (historical)', fontsize=11)
        ax.set_title(
            'Interval Calibration: Width vs. Historical Error\n'
            f'Coverage (error < width): {covered * 100:.1f} %  '
            f'({len(common)} entities)',
            fontsize=12, fontweight='bold',
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        return self._save(fig, save_name)

    # ================================================================
    # FIG 23d – Entity Error Analysis
    # ================================================================

    def plot_entity_error_analysis(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        entity_names: Optional[List[str]] = None,
        top_n: int = 40,
        save_name: str = 'fig23d_entity_error_analysis.png',
    ) -> Optional[str]:
        """Horizontal bar chart of per-entity prediction errors, sorted by magnitude.

        Blue bars indicate **under-prediction** (predicted < actual), red bars
        indicate **over-prediction** (predicted > actual).  Bar length encodes
        the mean absolute error; signed value is shown as a text annotation.

        When *entity_names* are provided, OOF samples are grouped and averaged
        per entity before plotting.  Without names, each sample is treated as
        its own observation.

        Args:
            y_test: Raveled 1-D OOF actual values.
            y_pred: Raveled 1-D OOF ensemble predictions (same length).
            entity_names: Optional entity labels aligned with *y_test* /
                *y_pred* (one per sample, may repeat across outputs).
            top_n: Maximum number of entities to display (default 40).
            save_name: Output filename.

        Returns:
            Absolute path to the saved PNG, or *None*.
        """
        if not HAS_MATPLOTLIB:
            return None

        y_t = np.asarray(y_test, dtype=float).ravel()
        y_p = np.asarray(y_pred, dtype=float).ravel()
        ok  = np.isfinite(y_t) & np.isfinite(y_p)
        if ok.sum() < 3:
            return None
        res = y_t[ok] - y_p[ok]

        if entity_names is not None and len(entity_names) == len(y_t):
            from collections import defaultdict as _dd
            ent_arr = np.asarray(entity_names)[ok]
            ent_res: Dict[str, list] = _dd(list)
            for ent, r in zip(ent_arr, res):
                ent_res[ent].append(r)
            entities      = sorted(ent_res.keys())
            signed_errors = np.array([np.mean(ent_res[e]) for e in entities])
        else:
            entities      = [str(i) for i in range(len(res))]
            signed_errors = res

        abs_errors = np.abs(signed_errors)
        sort_idx   = np.argsort(abs_errors)[::-1][:top_n]
        ent_show   = [entities[i]      for i in sort_idx]
        ae_show    = abs_errors[sort_idx]
        se_show    = signed_errors[sort_idx]

        # Sort within display by signed error for cleaner colour palette
        bar_colors = [
            PALETTE['coral']    if se > 0 else PALETTE['royal_blue']
            for se in se_show
        ]

        n_show = len(ent_show)
        fig, ax = plt.subplots(figsize=(10, max(5, n_show * 0.38)))
        ypos = np.arange(n_show)
        ax.barh(ypos, ae_show, color=bar_colors, alpha=0.82, height=0.72)

        for i, (ae, se) in enumerate(zip(ae_show, se_show)):
            sign_str = f'+{se:.3f}' if se > 0 else f'{se:.3f}'
            ax.text(ae + ae_show.max() * 0.01, i, sign_str,
                    va='center', fontsize=7.5)

        ax.set_yticks(ypos)
        ax.set_yticklabels(
            [self._truncate(e, 22) for e in ent_show], fontsize=8
        )
        ax.set_xlabel('Mean Absolute Error', fontsize=10)
        ax.set_title(
            f'Entity Error Analysis — Top-{n_show} by |Error|\n'
            f'Red = over-predicted  ·  Blue = under-predicted',
            fontsize=12, fontweight='bold',
        )
        ax.invert_yaxis()

        from matplotlib.patches import Patch
        leg_els = [
            Patch(facecolor=PALETTE['coral'],      label='Over-predicted  (actual > pred)'),
            Patch(facecolor=PALETTE['royal_blue'], label='Under-predicted  (actual < pred)'),
        ]
        ax.legend(handles=leg_els, fontsize=8, loc='lower right')
        plt.tight_layout()
        return self._save(fig, save_name)

    # ================================================================
    # FIG 23e – Temporal Training Curve (Walk-Forward CV)
    # ================================================================

    def plot_temporal_training_curve(
        self,
        cv_scores: Dict[str, List[float]],
        cv_fold_val_years: Optional[List[int]] = None,
        save_name: str = 'fig23e_temporal_training_curve.png',
    ) -> Optional[str]:
        """Per-fold walk-forward CV R² vs. validation year for each model.

        Shows how each model's OOF performance changes as the training window
        expands (more historical data = typically better performance).  The
        thick black line is the **model-averaged ensemble R²** per fold.  A
        shaded band spans the min–max range across all models per fold,
        highlighting variance across the ensemble.

        Args:
            cv_scores: Dict mapping model_name → list of per-fold R² values
                (one entry per walk-forward fold, earliest → latest).
            cv_fold_val_years: Optional list of integer calendar years
                corresponding to each fold's validation set.  When *None*,
                fold indices ``[1, 2, …, n_folds]`` are used.
            save_name: Output filename.

        Returns:
            Absolute path to the saved PNG, or *None*.
        """
        if not HAS_MATPLOTLIB or not cv_scores:
            return None

        # Filter to models with at least one valid fold score
        clean: Dict[str, np.ndarray] = {}
        for nm, scores in cv_scores.items():
            arr = np.asarray(scores, dtype=float)
            if arr.size > 0 and not np.all(np.isnan(arr)):
                clean[nm] = arr

        if not clean:
            return None

        # Align all models to the same fold count (pad short lists with NaN)
        n_folds = max(len(v) for v in clean.values())
        aligned: Dict[str, np.ndarray] = {}
        for nm, arr in clean.items():
            if len(arr) < n_folds:
                arr = np.pad(arr.astype(float), (0, n_folds - len(arr)),
                             constant_values=np.nan)
            aligned[nm] = arr[:n_folds]

        x_vals = (
            list(cv_fold_val_years[:n_folds])
            if cv_fold_val_years is not None and len(cv_fold_val_years) >= n_folds
            else list(range(1, n_folds + 1))
        )
        x_label = 'Validation Year' if cv_fold_val_years is not None else 'Walk-Forward Fold'

        # Ensemble mean (nanmean across models per fold)
        mat       = np.vstack(list(aligned.values()))   # (n_models, n_folds)
        ens_mean  = np.nanmean(mat, axis=0)
        ens_min   = np.nanmin(mat, axis=0)
        ens_max   = np.nanmax(mat, axis=0)

        fig, ax = plt.subplots(figsize=(max(9, n_folds * 0.85), 6))

        # Shade min–max envelope
        ax.fill_between(x_vals, ens_min, ens_max,
                        alpha=0.12, color=PALETTE['slate'],
                        label='Model range (min–max)')

        # Per-model lines
        cmap_r = plt.get_cmap('tab10')
        for i, (nm, vals) in enumerate(aligned.items()):
            color = _mcolor(nm, cmap_r(i % 10))
            ax.plot(x_vals, vals, marker='o', ms=4,
                    lw=1.5, color=color, alpha=0.75, label=nm)

        # Ensemble mean — bold
        ax.plot(x_vals, ens_mean, marker='D', ms=6,
                lw=2.8, color='#1A1A1A', zorder=5,
                label='Ensemble mean')

        ax.axhline(0, color='#888888', lw=0.8, ls='--')
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel('OOF R²', fontsize=11)
        ax.set_xticks(x_vals)
        ax.set_xticklabels(
            [str(x) for x in x_vals],
            rotation=35 if cv_fold_val_years is not None else 0,
            ha='right' if cv_fold_val_years is not None else 'center',
        )
        ax.set_title(
            'Walk-Forward CV: Model R² vs. Validation Year\n'
            '(Training window expands left → right)',
            fontsize=12, fontweight='bold',
        )
        ax.legend(fontsize=8, ncol=2, framealpha=0.85)
        plt.tight_layout()
        return self._save(fig, save_name)

    # ================================================================
    # FIG 24a – Pairwise Prediction Correlation Heatmap
    # ================================================================

    def plot_prediction_correlation_heatmap(
        self,
        per_model_oof_predictions: Dict[str, np.ndarray],
        save_name: str = 'fig24a_prediction_correlation_heatmap.png',
    ) -> Optional[str]:
        """Annotated heatmap of pairwise Pearson correlations between model OOF preds.

        High off-diagonal correlations indicate redundant models (low ensemble
        diversity); low correlations indicate complementary models whose errors
        partially cancel in the meta-learner combination.  Rows and columns are
        reordered by hierarchical clustering (average linkage on correlation
        distance ``1 − |r|``) so correlated models cluster together.

        Args:
            per_model_oof_predictions: Dict {model_name → ndarray of shape
                (n_oof, n_outputs)}.  All arrays must have the same n_oof.
            save_name: Output filename.

        Returns:
            Absolute path to the saved PNG, or *None*.
        """
        if not HAS_MATPLOTLIB or not per_model_oof_predictions:
            return None

        model_names = list(per_model_oof_predictions.keys())
        if len(model_names) < 2:
            return None

        # Ravel each model to 1-D (n_oof × n_outputs)
        vecs: Dict[str, np.ndarray] = {}
        for nm, arr in per_model_oof_predictions.items():
            a = np.asarray(arr, dtype=float).ravel()
            if a.size > 0:
                vecs[nm] = a

        if len(vecs) < 2:
            return None

        nm_list = list(vecs.keys())
        n_m     = len(nm_list)
        corr    = np.full((n_m, n_m), np.nan)

        for i in range(n_m):
            for j in range(n_m):
                a, b = vecs[nm_list[i]], vecs[nm_list[j]]
                # Pairwise complete-case correlation
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 5:
                    corr[i, j] = np.nan
                    continue
                if i == j:
                    corr[i, j] = 1.0
                    continue
                if HAS_SCIPY and sp_stats is not None:
                    try:
                        r, _ = sp_stats.pearsonr(a[ok], b[ok])
                    except Exception:
                        r = float(np.corrcoef(a[ok], b[ok])[0, 1])
                else:
                    r = float(np.corrcoef(a[ok], b[ok])[0, 1])
                corr[i, j] = r

        # ── Hierarchical clustering reorder ───────────────────────────────
        order = list(range(n_m))
        try:
            from scipy.cluster import hierarchy as _sch
            dist_mat = 1.0 - np.abs(np.nan_to_num(corr))
            np.fill_diagonal(dist_mat, 0.0)
            # pdist expects condensed distance matrix
            from scipy.spatial.distance import squareform as _sq
            condensed = _sq(dist_mat)
            linkage   = _sch.linkage(condensed, method='average')
            dendro    = _sch.dendrogram(linkage, no_plot=True)
            order     = dendro['leaves']
        except Exception:
            pass

        corr_ord = corr[np.ix_(order, order)]
        labels   = [nm_list[i] for i in order]

        fig, ax  = plt.subplots(figsize=(max(7, n_m * 1.3), max(6, n_m * 1.2)))
        im = ax.imshow(
            corr_ord, cmap='RdBu_r', vmin=-1, vmax=1,
            aspect='auto', interpolation='nearest',
        )
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                     label='Pearson r')

        ax.set_xticks(range(n_m))
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
        ax.set_yticks(range(n_m))
        ax.set_yticklabels(labels, fontsize=9)

        # Annotate cells
        for i in range(n_m):
            for j in range(n_m):
                val = corr_ord[i, j]
                if not np.isnan(val):
                    txt_col = 'white' if abs(val) > 0.6 else '#333333'
                    ax.text(j, i, f'{val:.2f}',
                            ha='center', va='center',
                            fontsize=8, color=txt_col)

        ax.set_title(
            'Pairwise OOF Prediction Correlation\n'
            '(hierarchical clustering order  ·  high off-diagonal = low diversity)',
            fontsize=12, fontweight='bold',
        )
        plt.tight_layout()
        return self._save(fig, save_name)

    # ================================================================
    # FIG 24b – Pairwise Prediction Scatter Matrix
    # ================================================================

    def plot_prediction_scatter_matrix(
        self,
        per_model_oof_predictions: Dict[str, np.ndarray],
        max_models: int = 7,
        save_name: str = 'fig24b_prediction_scatter_matrix.png',
    ) -> Optional[str]:
        """Scatter matrix of pairwise OOF predictions across base models.

        **Upper triangle**: scatter plot of Model A predictions vs. Model B
        predictions on the OOF set (regions of disagreement are immediately
        visible).
        **Diagonal**: histogram of each model's OOF predictions.
        **Lower triangle**: Pearson r value formatted as text.

        Args:
            per_model_oof_predictions: Dict {model_name → ndarray of shape
                (n_oof, n_outputs)}.
            max_models: Cap the display to at most this many models to avoid
                excessive figure size (default 7).  Models beyond the cap are
                silently dropped.
            save_name: Output filename.

        Returns:
            Absolute path to the saved PNG, or *None*.
        """
        if not HAS_MATPLOTLIB or not per_model_oof_predictions:
            return None

        nm_list = list(per_model_oof_predictions.keys())[:max_models]
        if len(nm_list) < 2:
            return None

        vecs: Dict[str, np.ndarray] = {
            nm: np.asarray(per_model_oof_predictions[nm], dtype=float).ravel()
            for nm in nm_list
        }
        n_m   = len(nm_list)
        panel = 2.8
        fig, axes = plt.subplots(
            n_m, n_m,
            figsize=(panel * n_m, panel * n_m),
            squeeze=False,
        )

        for row in range(n_m):
            for col in range(n_m):
                ax  = axes[row][col]
                a   = vecs[nm_list[row]]
                b   = vecs[nm_list[col]]
                ok  = np.isfinite(a) & np.isfinite(b)

                if row == col:
                    # Diagonal — histogram
                    data = a[np.isfinite(a)]
                    if len(data) > 2:
                        ax.hist(data, bins=20, color=_mcolor(nm_list[row]),
                                alpha=0.7, edgecolor='white', linewidth=0.3)
                    ax.set_title(nm_list[row], fontsize=8, fontweight='bold')

                elif row > col:
                    # Lower triangle — Pearson r text
                    ax.set_facecolor('#F8F9FA')
                    if ok.sum() >= 5:
                        if HAS_SCIPY and sp_stats is not None:
                            try:
                                r, p = sp_stats.pearsonr(a[ok], b[ok])
                            except Exception:
                                r = float(np.corrcoef(a[ok], b[ok])[0, 1])
                                p = np.nan
                        else:
                            r = float(np.corrcoef(a[ok], b[ok])[0, 1])
                            p = np.nan
                        sig = '***' if (not np.isnan(p) and p < 0.001) else \
                              '**'  if (not np.isnan(p) and p < 0.01)  else \
                              '*'   if (not np.isnan(p) and p < 0.05)  else ''
                        ax.text(
                            0.5, 0.5,
                            f'r = {r:.3f}{sig}',
                            ha='center', va='center', transform=ax.transAxes,
                            fontsize=10,
                            color=PALETTE['crimson'] if abs(r) > 0.7
                            else PALETTE['teal'] if abs(r) < 0.4
                            else '#444444',
                            fontweight='bold',
                        )
                    ax.set_xticks([])
                    ax.set_yticks([])

                else:
                    # Upper triangle — scatter
                    if ok.sum() >= 5:
                        ax.scatter(a[ok], b[ok],
                                   s=8, alpha=0.4,
                                   color=_mcolor(nm_list[col]),
                                   rasterized=True)
                        # Identity line
                        lim_lo = min(a[ok].min(), b[ok].min())
                        lim_hi = max(a[ok].max(), b[ok].max())
                        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                                color='#888888', lw=0.8, ls='--')

                # Minimal axis labels on borders only
                if row == n_m - 1:
                    ax.set_xlabel(nm_list[col], fontsize=7)
                if col == 0:
                    ax.set_ylabel(nm_list[row], fontsize=7)

                ax.tick_params(labelsize=6)

        fig.suptitle(
            'OOF Prediction Scatter Matrix — Ensemble Diversity',
            fontsize=13, fontweight='bold', y=1.01,
        )
        plt.tight_layout()
        return self._save(fig, save_name)


__all__ = ['ForecastPlotter']
