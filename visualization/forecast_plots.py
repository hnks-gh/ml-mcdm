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

        ax.set_title('Super Learner — Base Model Weights', fontsize=14, pad=20)
        centre = plt.Circle((0, 0), 0.55, fc='white')
        ax.add_artist(centre)
        ax.text(0, 0, 'Super\nLearner', ha='center', va='center',
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


__all__ = ['ForecastPlotter']
