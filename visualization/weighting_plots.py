# -*- coding: utf-8 -*-
"""
Weighting Plots (fig03–fig05)
=============================

Figures for subcriteria weight comparison: grouped bar chart, radar
(spider) diagram, and heatmap.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from .base import (
    BasePlotter, HAS_MATPLOTLIB,
    CATEGORICAL_COLORS, plt,
)


class WeightingPlotter(BasePlotter):
    """Figures for the objective-weighting phase."""

    # ==================================================================
    #  FIG 03 – Weight Comparison (grouped bar + dot overlay)
    # ==================================================================

    def plot_weights_comparison(
        self,
        weights: Dict[str, np.ndarray],
        component_names: List[str],
        title: str = 'Subcriteria Weight Comparison',
        save_name: str = 'fig03_weights_comparison.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        methods = list(weights.keys())
        n_m = len(methods)
        n_c = len(component_names)
        x = np.arange(n_c)
        width = 0.78 / n_m

        fig, ax = plt.subplots(figsize=(max(14, n_c * 0.65), 9))
        colors = CATEGORICAL_COLORS[:n_m]

        for i, (method, w) in enumerate(weights.items()):
            offset = (i - n_m / 2 + 0.5) * width
            ax.bar(x + offset, w, width, label=method,
                   color=colors[i], edgecolor='white', linewidth=0.4)
            ax.scatter(x + offset, w, s=18, color='black', zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [self._truncate(c, 12) for c in component_names],
            rotation=55, ha='right', fontsize=8,
        )
        ax.set_ylabel('Weight')
        ax.set_title(title, pad=12)
        ax.legend(loc='upper right', ncol=min(n_m, 3), fontsize=9)
        ax.set_xlim(-0.6, n_c - 0.4)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 04 – Weight Radar Chart (spider diagram)
    # ==================================================================

    def plot_weight_radar(
        self,
        weights: Dict[str, np.ndarray],
        component_names: List[str],
        save_name: str = 'fig04_weight_radar.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        n = len(component_names)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(11, 11), subplot_kw={'projection': 'polar'})
        colors = CATEGORICAL_COLORS

        for i, (method, w) in enumerate(weights.items()):
            vals = list(w) + [w[0]]
            ax.plot(angles, vals, 'o-', lw=2, label=method,
                    color=colors[i % len(colors)], markersize=5)
            ax.fill(angles, vals, alpha=0.08, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [self._truncate(c, 10) for c in component_names], fontsize=8,
        )
        ax.set_title('Weight Profiles — Radar Comparison', y=1.08, fontsize=14)
        ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.08), fontsize=9)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 05 – Weight Heatmap (method × subcriteria)
    # ==================================================================

    def plot_weight_heatmap(
        self,
        weights: Dict[str, np.ndarray],
        component_names: List[str],
        save_name: str = 'fig05_weight_heatmap.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        methods = list(weights.keys())
        data = np.array([weights[m] for m in methods])

        fig, ax = plt.subplots(
            figsize=(max(12, len(component_names) * 0.55), 5),
        )
        im = ax.imshow(data, aspect='auto', cmap='YlOrRd')

        ax.set_xticks(range(len(component_names)))
        ax.set_xticklabels(
            [self._truncate(c, 10) for c in component_names],
            rotation=55, ha='right', fontsize=8,
        )
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=10)

        for i in range(len(methods)):
            for j in range(len(component_names)):
                val = data[i, j]
                txt_col = ('white'
                           if val > data.mean() + 0.5 * data.std()
                           else 'black')
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=7, color=txt_col, fontweight='bold')

        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label('Weight', fontsize=10)
        ax.set_title(
            'Weight Values Heatmap (Method × Subcriteria)', pad=12,
        )
        return self._save(fig, save_name)


__all__ = ['WeightingPlotter']
