# -*- coding: utf-8 -*-
"""
Summary / Executive Dashboard Plots (fig24)
============================================

High-level summary visualisation combining KPIs, top-10 bar,
weight distribution, robustness text, and optional agreement heatmap.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any

from .base import (
    BasePlotter, HAS_MATPLOTLIB,
    PALETTE, CATEGORICAL_COLORS, plt,
)

try:
    from matplotlib.patches import FancyBboxPatch
except ImportError:
    FancyBboxPatch = None


class SummaryPlotter(BasePlotter):
    """Executive dashboard and high-level summary figures."""

    # ==================================================================
    #  FIG 24 – Executive Dashboard
    # ==================================================================

    def plot_executive_dashboard(
        self,
        results: Dict[str, Any],
        save_name: str = 'fig24_executive_dashboard.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB or FancyBboxPatch is None:
            return None

        fig = plt.figure(figsize=(22, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

        fig.suptitle(
            'ML-MCDM Analysis — Executive Dashboard',
            fontsize=18, fontweight='bold', y=0.99,
        )

        # KPI cards
        kpi_data = results.get('kpis', {})
        for i, (label, value) in enumerate(list(kpi_data.items())[:4]):
            ax = fig.add_subplot(gs[0, i])
            ax.axis('off')
            ax.add_patch(FancyBboxPatch(
                (0.05, 0.1), 0.9, 0.8,
                boxstyle='round,pad=0.05',
                facecolor=CATEGORICAL_COLORS[i], alpha=0.15,
                edgecolor=CATEGORICAL_COLORS[i], linewidth=2,
                transform=ax.transAxes,
            ))
            ax.text(0.5, 0.65, str(value), ha='center', va='center',
                    fontsize=26, fontweight='bold',
                    color=CATEGORICAL_COLORS[i],
                    transform=ax.transAxes)
            ax.text(0.5, 0.25, label, ha='center', va='center',
                    fontsize=11, color='#333333', transform=ax.transAxes)

        # Top 10 bar
        ax = fig.add_subplot(gs[1, :2])
        if 'top_10' in results:
            items_10 = results['top_10'][:10]
            names, scs = zip(*items_10)
            cmap_arr = plt.colormaps['viridis'](
                np.linspace(0.85, 0.25, len(items_10)),
            )
            ax.barh(range(len(items_10)), scs, color=cmap_arr,
                    edgecolor='white', linewidth=0.5)
            ax.set_yticks(range(len(items_10)))
            ax.set_yticklabels(
                [f'{i+1}. {n}' for i, n in enumerate(names)], fontsize=10,
            )
            ax.invert_yaxis()
            ax.set_xlabel('ER Score')
            ax.set_title('Top 10 Provinces', fontsize=12)

        # Top 15 weights
        ax = fig.add_subplot(gs[1, 2:])
        if 'fused_weights' in results:
            w = results['fused_weights']
            names_w = results.get(
                'subcriteria_names', [f'SC{i}' for i in range(len(w))],
            )
            order = np.argsort(w)[::-1][:15]
            ax.barh(
                range(len(order)), w[order],
                color=plt.colormaps['YlOrRd'](
                    np.linspace(0.3, 0.9, len(order)),
                ),
                edgecolor='white',
            )
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(
                [self._truncate(names_w[i], 14) for i in order], fontsize=9,
            )
            ax.invert_yaxis()
            ax.set_xlabel('Fused Weight')
            ax.set_title('Top 15 Subcriteria Weights', fontsize=12)

        # Robustness text
        ax = fig.add_subplot(gs[2, :2])
        ax.axis('off')
        if 'robustness_text' in results and results['robustness_text']:
            ax.text(
                0.05, 0.95, results['robustness_text'],
                transform=ax.transAxes, fontsize=10, va='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', fc='#F8F8F8', ec='#CCCCCC'),
            )

        # Method agreement mini-heatmap
        ax = fig.add_subplot(gs[2, 2:])
        if 'agreement_matrix' in results:
            am = results['agreement_matrix']
            im = ax.imshow(am, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title('Method Agreement', fontsize=12)
            fig.colorbar(im, ax=ax, shrink=0.6)
        else:
            ax.axis('off')

        return self._save(fig, save_name)


__all__ = ['SummaryPlotter']
