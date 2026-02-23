# -*- coding: utf-8 -*-
"""
MCDM Method-Agreement Plots (fig06–fig08)
==========================================

Figures for cross-method comparison: Spearman agreement heatmap,
rank parallel-coordinates, and per-criterion score panels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .base import (
    BasePlotter, HAS_MATPLOTLIB, HAS_SCIPY,
    PALETTE, CATEGORICAL_COLORS, plt, sp_stats,
)


class MCDMPlotter(BasePlotter):
    """Figures for MCDM inter-method agreement and criterion-level scores."""

    # ==================================================================
    #  FIG 06 – MCDM Method Agreement Matrix (Spearman heatmap)
    # ==================================================================

    def plot_method_agreement_matrix(
        self,
        rankings_dict: Dict[str, np.ndarray],
        title: str = 'MCDM Method Rank Agreement',
        save_name: str = 'fig06_method_agreement.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB or not HAS_SCIPY:
            return None

        methods = list(rankings_dict.keys())
        n = len(methods)
        corr = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r, _ = sp_stats.spearmanr(
                    np.asarray(rankings_dict[methods[i]]),
                    np.asarray(rankings_dict[methods[j]]),
                )
                corr[i, j] = corr[j, i] = r

        fig, ax = plt.subplots(
            figsize=(max(8, n * 0.75 + 2), max(7, n * 0.65 + 1)),
        )
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        labels = [self._truncate(m, 14) for m in methods]
        ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        for i in range(n):
            for j in range(n):
                txt_col = 'white' if abs(corr[i, j]) > 0.65 else 'black'
                ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                        fontsize=max(5, min(8, 120 // n)),
                        color=txt_col, fontweight='bold')

        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label('Spearman ρ', fontsize=10)
        ax.set_title(title, pad=12)

        avg_corr = (corr.sum() - n) / max(n * (n - 1), 1)
        ax.text(0.5, -0.08, f'Average pairwise ρ = {avg_corr:.4f}',
                transform=ax.transAxes, ha='center', fontsize=10, style='italic')
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 07 – MCDM Rank Comparison Parallel Coordinates
    # ==================================================================

    def plot_rank_parallel_coordinates(
        self,
        rankings_dict: Dict[str, np.ndarray],
        entity_names: List[str],
        top_n: int = 25,
        save_name: str = 'fig07_rank_parallel.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        methods = list(rankings_dict.keys())
        n_m = len(methods)
        first_ranks = np.asarray(rankings_dict[methods[0]])
        top_idx = np.argsort(first_ranks)[:top_n]

        fig, ax = plt.subplots(figsize=(max(12, n_m * 1.5), 10))
        cmap = plt.colormaps['viridis']
        norm = plt.Normalize(0, top_n - 1)

        for pos, idx in enumerate(top_idx):
            y_vals = [np.asarray(rankings_dict[m])[idx] for m in methods]
            colour = cmap(norm(pos))
            ax.plot(range(n_m), y_vals, '-o', color=colour, alpha=0.75,
                    lw=1.8, ms=7, markeredgecolor='white', markeredgewidth=0.8,
                    label=entity_names[idx] if pos < 10 else '')

        ax.set_xticks(range(n_m))
        ax.set_xticklabels(
            [self._truncate(m, 14) for m in methods],
            rotation=30, ha='right', fontsize=9,
        )
        ax.set_ylabel('Rank (1 = best)')
        ax.invert_yaxis()
        ax.set_title(
            f'Rank Trajectories Across {n_m} MCDM Methods (Top {top_n})',
            pad=12,
        )
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8,
                  title='Province', title_fontsize=9)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 08 – Per-Criterion Method Scores
    # ==================================================================

    def plot_criterion_scores(
        self,
        scores: Dict[str, pd.Series],
        criterion_name: str,
        top_n: int = 20,
        save_name: str = 'fig08_criterion_scores.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        n_methods = len(scores)
        fig, axes = plt.subplots(
            1, min(n_methods, 6),
            figsize=(4.5 * min(n_methods, 6), max(8, top_n * 0.32)),
            sharey=True,
        )
        if min(n_methods, 6) == 1:
            axes = [axes]

        for idx, (method, series) in enumerate(list(scores.items())[:6]):
            ax = axes[idx]
            sorted_s = series.sort_values(ascending=False).head(top_n)
            cmap_ = plt.colormaps['viridis']
            vmin, vmax = sorted_s.min(), sorted_s.max()
            norm_ = (plt.Normalize(vmin, vmax)
                     if vmax > vmin else plt.Normalize(0, 1))
            bar_colors = [cmap_(norm_(v)) for v in sorted_s.values]

            ax.barh(range(len(sorted_s)), sorted_s.values, color=bar_colors,
                    edgecolor='white', linewidth=0.3)
            ax.set_yticks(range(len(sorted_s)))
            ax.set_yticklabels(
                [self._truncate(str(n), 15) for n in sorted_s.index],
                fontsize=8,
            )
            ax.invert_yaxis()
            ax.set_xlabel('Score', fontsize=9)
            ax.set_title(f'{method}', fontsize=11, fontweight='bold')

        fig.suptitle(
            f'{criterion_name} — Per-Method Scores (Top {top_n})',
            fontsize=14, y=1.02,
        )
        plt.tight_layout()
        return self._save(fig, save_name)


__all__ = ['MCDMPlotter']
