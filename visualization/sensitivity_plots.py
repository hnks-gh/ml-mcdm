# -*- coding: utf-8 -*-
"""
Sensitivity & Robustness Plots (fig09–fig15, fig25)
====================================================

Figures for sensitivity analysis: tornado charts, subcriteria bars,
top-N stability, temporal stability, rank volatility, IFS sensitivity,
ER uncertainty distribution, and the composite robustness summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from .base import (
    BasePlotter, HAS_MATPLOTLIB,
    PALETTE, CATEGORICAL_COLORS, plt,
)

try:
    from matplotlib.patches import FancyBboxPatch
except ImportError:
    FancyBboxPatch = None


class SensitivityPlotter(BasePlotter):
    """Figures for sensitivity, robustness, and ER uncertainty."""

    # ==================================================================
    #  FIG 09 – Criteria Sensitivity Tornado
    # ==================================================================

    def plot_sensitivity_tornado(
        self,
        sensitivity: Dict[str, float],
        title: str = 'Criteria Sensitivity Analysis',
        save_name: str = 'fig09_criteria_sensitivity.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        items = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
        names = [k for k, _ in items]
        vals = [v for _, v in items]
        n = len(names)

        fig, ax = plt.subplots(figsize=(12, max(6, n * 0.42)))

        vmin, vmax = min(vals), max(vals)
        norm = plt.Normalize(vmin, vmax) if vmax > vmin else plt.Normalize(0, 1)
        cmap = plt.colormaps['RdYlGn_r']
        colors = [cmap(norm(v)) for v in vals]

        bars = ax.barh(range(n), vals, color=colors, edgecolor='black',
                       linewidth=0.5, height=0.7, zorder=2)
        for bar, val in zip(bars, vals):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9)

        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Sensitivity Index')
        ax.set_title(title, pad=12)

        mean_v = np.mean(vals)
        std_v = np.std(vals)
        high_ = sum(1 for v in vals if v > mean_v + std_v)
        ax.text(0.97, 0.03, f'High sensitivity criteria: {high_}/{n}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec='#CCCCCC'))
        return self._save(fig, save_name)

    # Backward-compatible alias
    def plot_sensitivity_analysis(self, sensitivity, **kw):
        return self.plot_sensitivity_tornado(sensitivity, **kw)

    # ==================================================================
    #  FIG 10 – Subcriteria Sensitivity (Top 20)
    # ==================================================================

    def plot_subcriteria_sensitivity(
        self,
        sensitivity: Dict[str, float],
        top_n: int = 20,
        save_name: str = 'fig10_subcriteria_sensitivity.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        items = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names = [k for k, _ in items]
        vals = [v for _, v in items]

        fig, ax = plt.subplots(figsize=(12, max(7, len(names) * 0.38)))
        colors = plt.colormaps['magma'](np.linspace(0.25, 0.85, len(names)))

        ax.barh(range(len(names)), vals, color=colors, edgecolor='white',
                linewidth=0.4, height=0.65)
        for i, v in enumerate(vals):
            ax.text(v + 0.003, i, f'{v:.4f}', va='center', fontsize=8)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(
            [self._truncate(n, 22) for n in names], fontsize=9,
        )
        ax.invert_yaxis()
        ax.set_xlabel('Sensitivity Index')
        ax.set_title(f'Subcriteria Weight Sensitivity (Top {top_n})', pad=12)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 11 – Top-N Stability Bar
    # ==================================================================

    def plot_top_n_stability(
        self,
        stability: Dict[int, float],
        save_name: str = 'fig11_top_n_stability.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        ns = sorted(stability.keys())
        vals = [stability[n] for n in ns]

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = [
            PALETTE['emerald'] if v >= 0.8
            else PALETTE['gold'] if v >= 0.5
            else PALETTE['crimson']
            for v in vals
        ]

        bars = ax.bar(range(len(ns)), vals, color=colors, edgecolor='black',
                      linewidth=0.7, width=0.55, zorder=2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                    f'{v:.1%}', ha='center', fontsize=11, fontweight='bold')

        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([f'Top-{n}' for n in ns], fontsize=11)
        ax.set_ylabel('Stability (overlap ratio)')
        ax.set_ylim(0, 1.12)
        ax.set_title('Ranking Stability — Top-N Overlap Under Perturbation', pad=12)
        ax.axhline(0.8, ls=':', color='gray', lw=1, label='80% threshold')
        ax.legend(fontsize=9)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 12 – Temporal Stability (year-to-year rank correlation)
    # ==================================================================

    def plot_temporal_stability(
        self,
        temporal: Dict[str, float],
        save_name: str = 'fig12_temporal_stability.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        pairs = sorted(temporal.keys())
        vals = [temporal[p] for p in pairs]

        fig, ax = plt.subplots(figsize=(max(10, len(pairs) * 0.8), 7))
        colors = [
            PALETTE['emerald'] if v >= 0.9
            else PALETTE['gold'] if v >= 0.7
            else PALETTE['crimson']
            for v in vals
        ]

        ax.bar(range(len(pairs)), vals, color=colors, edgecolor='black',
               linewidth=0.5, width=0.6, zorder=2)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8,
                    fontweight='bold')

        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(pairs, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Spearman Rank Correlation')
        ax.set_ylim(0, 1.1)
        ax.set_title('Temporal Stability — Year-to-Year Rank Correlations', pad=12)
        ax.axhline(np.mean(vals), ls='--', color='gray', lw=1.2,
                   label=f'Mean = {np.mean(vals):.4f}')
        ax.legend(fontsize=9)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 13 – Rank Stability per Province
    # ==================================================================

    def plot_rank_volatility(
        self,
        rank_stability: Dict[str, float],
        save_name: str = 'fig13_rank_volatility.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        items = sorted(rank_stability.items(), key=lambda x: x[1])
        names = [k for k, _ in items]
        vals = [v for _, v in items]
        n = len(names)

        fig, ax = plt.subplots(figsize=(13, max(10, n * 0.25)))
        vmin, vmax = min(vals), max(vals)
        norm = plt.Normalize(vmin, vmax) if vmax > vmin else plt.Normalize(0, 1)
        cmap = plt.colormaps['RdYlGn']
        colors = [cmap(norm(v)) for v in vals]

        ax.barh(range(n), vals, color=colors, edgecolor='white',
                linewidth=0.3, height=0.7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [self._truncate(nm, 20) for nm in names], fontsize=8,
        )
        ax.set_xlabel('Rank Stability Score')
        ax.set_title('Province Rank Stability (higher = more stable)', pad=12)
        ax.axvline(np.mean(vals), ls='--', color='gray', lw=1.2,
                   label=f'Mean = {np.mean(vals):.3f}')
        ax.legend(fontsize=9)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 14 – IFS Sensitivity (comparative bar)
    # ==================================================================

    def plot_ifs_sensitivity(
        self,
        mu_sens: float,
        nu_sens: float,
        save_name: str = 'fig14_ifs_sensitivity.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        fig, ax = plt.subplots(figsize=(10, 7))

        categories = ['Membership (μ)', 'Non-Membership (ν)']
        values = [mu_sens, nu_sens]
        colors = [PALETTE['royal_blue'], PALETTE['magenta']]

        bars = ax.bar(categories, values, color=colors, edgecolor='black',
                      linewidth=0.8, width=0.45, zorder=2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f'{v:.4f}', ha='center', fontsize=14, fontweight='bold')

        ax.set_ylabel('Sensitivity Index')
        ax.set_title('IFS Uncertainty Sensitivity Analysis', pad=12)
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)

        interpretation = (
            'Low sensitivity — robust' if max(values) < 0.2
            else 'Moderate sensitivity' if max(values) < 0.5
            else 'High sensitivity — caution'
        )
        ax.text(0.97, 0.97, interpretation, transform=ax.transAxes,
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec='#CCCCCC'))
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 15 – ER Uncertainty Distribution
    # ==================================================================

    def plot_er_uncertainty(
        self,
        uncertainty_df: pd.DataFrame,
        provinces: List[str],
        top_n: int = 30,
        save_name: str = 'fig15_er_uncertainty.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None
        if uncertainty_df.shape[1] < 2:
            return None

        fig, ax = plt.subplots(
            figsize=(14, max(8, min(top_n, len(uncertainty_df)) * 0.35)),
        )

        n_show = min(top_n, len(uncertainty_df))
        data = uncertainty_df.iloc[:n_show].values

        ax.boxplot(
            data.T, vert=False, patch_artist=True,
            boxprops=dict(facecolor=PALETTE['royal_blue'], alpha=0.6),
            medianprops=dict(color='black', lw=2),
            whiskerprops=dict(color='gray'),
            flierprops=dict(marker='o', ms=4, alpha=0.4),
        )

        ax.set_yticks(range(1, n_show + 1))
        labels = (
            uncertainty_df.index[:n_show].tolist()
            if hasattr(uncertainty_df.index, 'tolist')
            else provinces[:n_show]
        )
        ax.set_yticklabels(
            [self._truncate(str(l), 18) for l in labels], fontsize=9,
        )
        ax.set_xlabel('Belief Degree')
        ax.set_title('ER Aggregation Uncertainty Distribution', pad=12)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 25 – Robustness Summary Infographic (4-panel)
    # ==================================================================

    def plot_robustness_summary(
        self,
        overall_robustness: float,
        confidence_level: float,
        criteria_sens: Dict[str, float],
        top_n_stab: Dict[int, float],
        mu_sens: float,
        nu_sens: float,
        save_name: str = 'fig25_robustness_summary.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB or FancyBboxPatch is None:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 13))

        # Panel 1: Overall gauge
        ax = axes[0, 0]
        ax.axis('off')
        color = (
            PALETTE['emerald'] if overall_robustness >= 0.8
            else PALETTE['gold'] if overall_robustness >= 0.5
            else PALETTE['crimson']
        )
        ax.add_patch(FancyBboxPatch(
            (0.1, 0.2), 0.8, 0.6, boxstyle='round,pad=0.08',
            fc=color, alpha=0.15, ec=color, lw=3,
            transform=ax.transAxes,
        ))
        ax.text(0.5, 0.6, f'{overall_robustness:.4f}', ha='center',
                va='center', fontsize=42, fontweight='bold', color=color,
                transform=ax.transAxes)
        ax.text(0.5, 0.35, 'Overall Robustness Score', ha='center',
                fontsize=14, transform=ax.transAxes)
        ax.text(0.5, 0.22, f'Confidence Level: {confidence_level:.0%}',
                ha='center', fontsize=11, color='gray', transform=ax.transAxes)

        # Panel 2: Criteria sensitivity bars
        ax = axes[0, 1]
        if criteria_sens:
            items = sorted(criteria_sens.items(), key=lambda x: x[1], reverse=True)
            names_ = [k for k, _ in items]
            vals_ = [v for _, v in items]
            vmin, vmax = min(vals_), max(vals_)
            norm_ = plt.Normalize(vmin, vmax) if vmax > vmin else plt.Normalize(0, 1)
            colors_ = [plt.colormaps['RdYlGn_r'](norm_(v)) for v in vals_]
            ax.barh(range(len(names_)), vals_, color=colors_,
                    edgecolor='black', lw=0.5)
            ax.set_yticks(range(len(names_)))
            ax.set_yticklabels(names_, fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel('Sensitivity')
            ax.set_title('Criteria Sensitivity', fontsize=12)

        # Panel 3: Top-N stability
        ax = axes[1, 0]
        if top_n_stab:
            ns = sorted(top_n_stab.keys())
            vals_s = [top_n_stab[n] for n in ns]
            bar_colors = [
                PALETTE['emerald'] if v >= 0.8
                else PALETTE['gold'] if v >= 0.5
                else PALETTE['crimson']
                for v in vals_s
            ]
            ax.bar(range(len(ns)), vals_s, color=bar_colors,
                   edgecolor='black', lw=0.7)
            for i, v in enumerate(vals_s):
                ax.text(i, v + 0.015, f'{v:.1%}', ha='center',
                        fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(ns)))
            ax.set_xticklabels([f'Top-{n}' for n in ns])
            ax.set_ylim(0, 1.15)
            ax.set_ylabel('Stability')
            ax.set_title('Ranking Stability', fontsize=12)
            ax.axhline(0.8, ls=':', color='gray', lw=1)

        # Panel 4: IFS sensitivity
        ax = axes[1, 1]
        bars = ax.bar(
            ['Membership (μ)', 'Non-Member. (ν)'], [mu_sens, nu_sens],
            color=[PALETTE['royal_blue'], PALETTE['magenta']],
            edgecolor='black', lw=0.7, width=0.5,
        )
        for bar, v in zip(bars, [mu_sens, nu_sens]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sensitivity')
        ax.set_title('IFS Uncertainty Sensitivity', fontsize=12)

        fig.suptitle('Robustness & Sensitivity Summary', fontsize=15, y=1.01)
        plt.tight_layout()
        return self._save(fig, save_name)


__all__ = ['SensitivityPlotter']
