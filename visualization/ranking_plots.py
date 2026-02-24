# -*- coding: utf-8 -*-
"""
Ranking & ER Plots (fig01–fig02)
================================

Publication-quality figures for the Evidential Reasoning ranking
phase: lollipop ranking chart and score distribution histogram.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from .base import (
    BasePlotter, HAS_MATPLOTLIB, HAS_SCIPY,
    PALETTE, CATEGORICAL_COLORS, plt, sp_stats,
)


class RankingPlotter(BasePlotter):
    """Figures related to final ER ranking and score distribution."""

    # ==================================================================
    #  FIG 01 – Final ER Ranking  (horizontal lollipop + gradient fill)
    # ==================================================================

    def plot_final_ranking(
        self,
        provinces: List[str],
        scores: np.ndarray,
        ranks: np.ndarray,
        title: str = 'Hierarchical ER Final Ranking',
        save_name: str = 'fig01_final_er_ranking.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None
        scores = np.asarray(scores)
        ranks = np.asarray(ranks)
        order = np.argsort(ranks)
        n = len(provinces)

        fig, ax = plt.subplots(figsize=(14, max(10, n * 0.28)))

        norm = plt.Normalize(scores[order].min(), scores[order].max())
        cmap = plt.colormaps['RdYlGn']

        for i, idx in enumerate(order):
            colour = cmap(norm(scores[idx]))
            ax.barh(i, scores[idx], height=0.65, color=colour,
                    edgecolor='white', linewidth=0.4, zorder=2)
            ax.plot(scores[idx], i, 'o', color='black', markersize=5, zorder=3)
            ax.text(scores[idx] + 0.003, i, f'{scores[idx]:.4f}',
                    va='center', fontsize=8, color='#333333')

        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [f'{int(ranks[idx]):>2d}. {provinces[idx]}' for idx in order],
            fontsize=9,
        )
        ax.invert_yaxis()
        ax.set_xlabel('Evidential Reasoning Composite Score')
        ax.set_title(title, fontsize=14, pad=12)
        ax.axvline(np.median(scores), ls=':', color='gray', lw=1.2,
                   label=f'Median = {np.median(scores):.4f}')
        ax.legend(loc='lower right', fontsize=9)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.01)
        cbar.set_label('ER Score', fontsize=10)
        return self._save(fig, save_name)

    # Backward-compatible alias
    def plot_final_ranking_summary(self, provinces, scores, ranks, **kw):
        return self.plot_final_ranking(provinces, scores, ranks, **kw)

    # ==================================================================
    #  FIG 02 – ER Score Distribution  (histogram + KDE + rug)
    # ==================================================================

    def plot_score_distribution(
        self,
        scores: np.ndarray,
        title: str = 'Distribution of ER Scores',
        save_name: str = 'fig02_score_distribution.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None
        scores = np.asarray(scores)

        fig, ax = plt.subplots(figsize=self.figsize)

        n_bins = min(30, max(10, len(scores) // 3))
        n_vals, bins, patches = ax.hist(
            scores, bins=n_bins, density=True,
            alpha=0.75, edgecolor='white', linewidth=0.6,
        )
        cm = plt.colormaps['viridis']
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        _ptp = bin_centers.max() - bin_centers.min()
        rng = _ptp if _ptp > 0 else 1
        col = (bin_centers - bin_centers.min()) / rng
        for c, p in zip(col, patches):
            p.set_facecolor(cm(c))

        # KDE overlay
        if HAS_SCIPY and len(scores) > 5:
            kde = sp_stats.gaussian_kde(scores)
            x_grid = np.linspace(
                scores.min() - scores.std(), scores.max() + scores.std(), 300,
            )
            ax.plot(x_grid, kde(x_grid), color=PALETTE['crimson'], lw=2.5,
                    label='KDE', zorder=4)

        # Rug plot
        ax.plot(scores, np.zeros_like(scores) - 0.02 * max(n_vals.max(), 1),
                '|', color='black', ms=8, alpha=0.5, zorder=5)

        mean, med, std = scores.mean(), np.median(scores), scores.std()
        ax.axvline(mean, ls='--', color=PALETTE['coral'], lw=1.8,
                   label=f'Mean = {mean:.4f}')
        ax.axvline(med, ls='-.', color=PALETTE['royal_blue'], lw=1.8,
                   label=f'Median = {med:.4f}')

        # Normality test
        norm_text = ''
        if HAS_SCIPY and 3 <= len(scores) <= 5000:
            sw_stat, sw_p = sp_stats.shapiro(scores)
            norm_text = f'Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f}'

        skew_v = float(sp_stats.skew(scores)) if HAS_SCIPY else 0
        kurt_v = float(sp_stats.kurtosis(scores)) if HAS_SCIPY else 0
        stats_box = (
            f'N = {len(scores)}\n'
            f'Mean = {mean:.4f}\n'
            f'Median = {med:.4f}\n'
            f'Std = {std:.4f}\n'
            f'Skew = {skew_v:.3f}\n'
            f'Kurt = {kurt_v:.3f}\n'
            + norm_text
        )
        ax.text(0.97, 0.97, stats_box, transform=ax.transAxes, fontsize=9,
                va='top', ha='right', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor='#CCCCCC', alpha=0.95))

        ax.set_xlabel('ER Composite Score')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend(loc='upper left', fontsize=9)
        return self._save(fig, save_name)


__all__ = ['RankingPlotter']
