# -*- coding: utf-8 -*-
"""
Sensitivity, Robustness, and Uncertainty Visualizations.

This module provides the `SensitivityPlotter` class, which generates 
publication-quality diagnostic plots to evaluate the stability of MCDM 
and hierarchical rankings. It includes tornado charts for weight sensitivity, 
temporal stability analysis, and Monte Carlo rank-volatility distributions.

Key Figures
-----------
- **fig09/09b (Tornado Charts)**: Criteria and subcriteria weight 
  sensitivity analysis.
- **fig11 (Top-N Stability)**: Overlap ratios indicating how many 
  alternatives remain in the top-N tier under perturbation.
- **fig12 (Temporal Stability)**: Year-to-year Spearman rank correlations.
- **fig14/14b (Rank Volatility)**: Quadrant analysis and violin plots 
  showing rank deviation distributions.
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
    """
    Generator for sensitivity and robustness visualizations.

    Handles the rendering of weight sensitivity indices, temporal rank 
    consistency, and stochastic uncertainty profiles.
    """

    # ==================================================================
    #  FIG 09 – Criteria Sensitivity Tornado
    # ==================================================================

    def plot_sensitivity_tornado(
        self,
        sensitivity: Dict[str, float],
        title: str = 'Criteria Sensitivity Analysis',
        save_name: str = 'fig09_criteria_sensitivity.png',
    ) -> Optional[str]:
        """
        Produce a tornado chart of criteria weight sensitivity.

        Identifies which criteria most significantly impact the final 
        ranking when their weights are perturbed.

        Parameters
        ----------
        sensitivity : Dict[str, float]
            Mapping of criterion name to its sensitivity index.
        title : str, default='Criteria Sensitivity Analysis'
            The plot title.
        save_name : str, default='fig09_criteria_sensitivity.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
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
        """
        Legacy alias for historical compatibility.

        See Also
        --------
        plot_sensitivity_tornado
        """
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
        """
        Produce a horizontal bar chart of subcriteria sensitivity.

        Focuses on the top-N most sensitive subcriteria to guide 
        targeted data quality audits.

        Parameters
        ----------
        sensitivity : Dict[str, float]
            Mapping of subcriterion name to sensitivity index.
        top_n : int, default=20
            The number of subcriteria to display.
        save_name : str, default='fig10_subcriteria_sensitivity.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
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
        """
        Produce a stability chart for top-N tiers.

        Measures the overlap ratio of alternatives in the top performance 
        bands under stochastic perturbation.

        Parameters
        ----------
        stability : Dict[int, float]
            Mapping of N (tier size) to stability ratio (0-1).
        save_name : str, default='fig11_top_n_stability.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
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
        """
        Produce a temporal stability bar chart.

        Calculates year-to-year Spearman rank correlations to assess 
        the longitudinal consistency of provincial rankings.

        Parameters
        ----------
        temporal : Dict[str, float]
            Mapping of year-pair (e.g. '2022-2023') to correlation.
        save_name : str, default='fig12_temporal_stability.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
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
        """
        Produce a horizontal bar chart of rank stability per province.

        Identifies provinces with the most (and least) stable rankings 
        across simulation trials.

        Parameters
        ----------
        rank_stability : Dict[str, float]
            Mapping of province name to stability score (0-1).
        save_name : str, default='fig13_rank_volatility.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
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
    #  FIG 15 – Ranking Uncertainty Distribution
    # ==================================================================

    # ==================================================================
    #  FIG 09b – Butterfly Tornado (diverging criteria sensitivity)
    # ==================================================================

    def plot_tornado_butterfly(
        self,
        criteria_sensitivity: Dict[str, float],
        perturbation_analysis: Optional[Dict[str, Any]] = None,
        save_name: str = 'fig09b_tornado_butterfly.png',
    ) -> Optional[str]:
        """
        Produce a two-sided butterfly tornado chart.

        Compares raw sensitivity (destabilizing potential) against stability 
        contribution, making high-impact criteria visually distinct.

        Parameters
        ----------
        criteria_sensitivity : Dict[str, float]
            Mapping of criterion name to sensitivity index.
        perturbation_analysis : Dict[str, Any], optional
            Detailed results from stochastic simulations.
        save_name : str, default='fig09b_tornado_butterfly.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
        if not HAS_MATPLOTLIB:
            return None
        if not criteria_sensitivity:
            return None

        items = sorted(criteria_sensitivity.items(), key=lambda x: x[1])
        names = [k for k, _ in items]
        vals = np.array([v for _, v in items], dtype=float)
        n = len(names)

        mean_val = float(np.mean(vals))
        # Right side: raw sensitivity  | Left side: deviation below mean (floored at 0)
        right_bars = vals
        left_bars = np.clip(mean_val - vals, 0, None)

        # Categorise by sensitivity relative to overall mean+std
        std_val = float(np.std(vals))
        high_thresh = mean_val + std_val
        low_thresh = mean_val - std_val

        right_colors, left_colors = [], []
        for v in vals:
            if v >= high_thresh:
                right_colors.append('#C73E1D')   # High sensitivity — red
                left_colors.append('#F5C6BB')
            elif v >= mean_val:
                right_colors.append('#E07B39')   # Above average — orange
                left_colors.append('#FAD9C5')
            else:
                right_colors.append('#2E86AB')   # Below average — blue
                left_colors.append('#C2DFF0')

        fig, ax = plt.subplots(figsize=(13, max(6, n * 0.48)))
        y = np.arange(n)

        ax.barh(y, right_bars, left=0, height=0.55, color=right_colors,
                edgecolor='white', linewidth=0.5, label='Sensitivity Index', zorder=3)
        ax.barh(y, -left_bars, left=0, height=0.55, color=left_colors,
                edgecolor='white', linewidth=0.5, label='Stability Contribution', zorder=3)

        # Value labels
        for i, (rv, lv, rc) in enumerate(zip(right_bars, left_bars, right_colors)):
            ax.text(rv + 0.003, i, f'{rv:.4f}', va='center', fontsize=8.5,
                    color=rc, fontweight='bold')
            if lv > 0:
                ax.text(-lv - 0.003, i, f'{lv:.4f}', va='center', fontsize=8,
                        ha='right', color='gray')

        # Centre spine / mean reference
        ax.axvline(0, color='black', lw=1.2, zorder=4)
        ax.axvline(mean_val, color='gray', lw=1.0, ls='--', alpha=0.7,
                   label=f'Mean = {mean_val:.4f}')

        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('← Stability Contribution  |  Sensitivity Index →')
        ax.set_title('Criteria Sensitivity — Butterfly Tornado Chart', pad=12)
        ax.legend(fontsize=9, loc='lower right')

        # Quadrant annotation
        n_high = int(np.sum(vals >= high_thresh))
        ax.text(0.98, 0.02, f'High-sensitivity: {n_high}/{n}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='#FFF4E5', ec='#E07B39'))

        plt.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 09c – Subcriteria Dot/Strip Plot (grouped by criterion)
    # ==================================================================

    def plot_subcriteria_dotstrip(
        self,
        subcriteria_sensitivity: Dict[str, float],
        save_name: str = 'fig09c_subcriteria_dotstrip.png',
    ) -> Optional[str]:
        """
        Produce a jittered dot/strip plot of subcriteria sensitivity.

        Groups subcriteria by their parent criterion to reveal patterns of 
        instability within specific thematic areas.

        Parameters
        ----------
        subcriteria_sensitivity : Dict[str, float]
            Mapping of subcriterion ID to sensitivity index.
        save_name : str, default='fig09c_subcriteria_dotstrip.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
        if not HAS_MATPLOTLIB:
            return None
        if not subcriteria_sensitivity:
            return None

        import re

        def _parent(sc: str) -> str:
            m = re.match(r'SC(\d)', sc, re.IGNORECASE)
            return f'C{m.group(1)}' if m else sc[:2]

        items = list(subcriteria_sensitivity.items())
        groups: Dict[str, List] = {}
        for sc, val in items:
            g = _parent(sc)
            groups.setdefault(g, []).append((sc, val))
        group_keys = sorted(groups.keys())
        n_groups = len(group_keys)

        cmap_g = plt.colormaps['tab10']
        group_colors = {g: cmap_g(i / max(n_groups, 1)) for i, g in enumerate(group_keys)}

        fig, ax = plt.subplots(figsize=(14, max(7, n_groups * 1.1)))

        rng_j = np.random.RandomState(0)
        all_vals: List[float] = []
        xtick_pos, xtick_labels = [], []

        for gi, g in enumerate(group_keys):
            sc_items = sorted(groups[g], key=lambda x: x[0])
            n_sc = len(sc_items)
            jitter = rng_j.uniform(-0.3, 0.3, n_sc)
            vals_g = np.array([v for _, v in sc_items])
            all_vals.extend(vals_g)
            color = group_colors[g]

            # Shaded criterion band (alternating)
            if gi % 2 == 0:
                ax.axhspan(gi - 0.5, gi + 0.5, facecolor='#F8F8F8',
                           alpha=0.4, zorder=0)

            ax.scatter(vals_g + jitter * 0.01, [gi] * n_sc,
                       c=[color] * n_sc, s=70, alpha=0.85,
                       edgecolors='white', linewidths=0.6, zorder=3)

            # Mean reference bar
            mean_g = float(np.mean(vals_g))
            ax.plot([mean_g, mean_g], [gi - 0.35, gi + 0.35],
                    color=color, lw=2.5, zorder=4)

            # SC point labels
            for (sc_name, _), xv in zip(sc_items, vals_g):
                ax.text(xv, gi + 0.22, self._truncate(sc_name, 8),
                        ha='center', va='bottom', fontsize=6.5,
                        color='#444444')

            xtick_pos.append(gi)
            xtick_labels.append(g)

        grand_mean = float(np.mean(all_vals)) if all_vals else 0.0
        ax.axvline(grand_mean, color='gray', lw=1.2, ls='--', alpha=0.7,
                   label=f'Overall mean = {grand_mean:.4f}')

        ax.set_yticks(xtick_pos)
        ax.set_yticklabels(xtick_labels, fontsize=11)
        ax.set_xlabel('Sensitivity Index')
        ax.set_title('Subcriteria Sensitivity — Grouped Dot/Strip Plot', pad=12)
        ax.legend(fontsize=9)
        plt.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 13b – Province Stability with Line + CI Bands
    # ==================================================================

    def plot_stability_line_ci(
        self,
        rank_stability: Dict[str, float],
        ci_width: float = 0.05,
        save_name: str = 'fig13b_stability_line_ci.png',
    ) -> Optional[str]:
        """
        Produce a stability line chart with confidence intervals.

        Visualizes provincial stability scores across a sorted plateau, 
        emphasizing zones of volatility vs. robustness.

        Parameters
        ----------
        rank_stability : Dict[str, float]
            Mapping of province name to stability score.
        ci_width : float, default=0.05
            The half-width of the confidence band.
        save_name : str, default='fig13b_stability_line_ci.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
        if not HAS_MATPLOTLIB:
            return None
        if not rank_stability:
            return None

        items = sorted(rank_stability.items(), key=lambda x: x[1], reverse=True)
        names = [k for k, _ in items]
        vals = np.array([v for _, v in items], dtype=float)
        n = len(names)
        x = np.arange(n)

        # ±ci band; cap at [0, 1]
        lo = np.clip(vals - ci_width, 0, 1)
        hi = np.clip(vals + ci_width, 0, 1)

        zone_defs = [
            (0.85, 1.01, '#D4EDDA', 'Very Stable ≥ 0.85'),
            (0.70, 0.85, '#FFF3CD', 'Stable 0.70–0.85'),
            (0.50, 0.70, '#FFE4CC', 'Moderate 0.50–0.70'),
            (0.00, 0.50, '#F8D7DA', 'Volatile < 0.50'),
        ]

        fig, ax = plt.subplots(figsize=(max(14, n * 0.22), 8))

        # Zone shading (horizontal bands)
        for zlo, zhi, zc, zlabel in zone_defs:
            ax.axhspan(zlo, zhi, facecolor=zc, alpha=0.45, zorder=0,
                       label=zlabel)

        # CI ribbon
        ax.fill_between(x, lo, hi, alpha=0.25, color=PALETTE.get('royal_blue', '#4472C4'),
                        label=f'±{ci_width:.0%} CI band', zorder=1)
        # Stability line
        ax.plot(x, vals, color=PALETTE.get('royal_blue', '#4472C4'), lw=1.8,
                zorder=2, label='Stability Score')
        ax.scatter(x, vals, c=vals, cmap='RdYlGn', vmin=0, vmax=1,
                   s=28, zorder=3, edgecolors='none')

        # Mean reference
        mean_v = float(np.mean(vals))
        ax.axhline(mean_v, color='gray', lw=1.2, ls='--', alpha=0.8,
                   label=f'Mean = {mean_v:.3f}')

        # Annotate n-counts per zone
        for zlo, zhi, zc, _ in zone_defs:
            cnt = int(np.sum((vals >= zlo) & (vals < zhi)))
            if cnt:
                mid_x = n * 0.98
                mid_y = (zlo + min(zhi, 1.0)) / 2
                ax.text(mid_x, mid_y, f'n={cnt}', ha='right', va='center',
                        fontsize=9, color='#333333',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7,
                                  ec='none'))

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x[::max(1, n // 20)])
        ax.set_xticklabels(
            [self._truncate(names[i], 12)
             for i in range(0, n, max(1, n // 20))],
            rotation=45, ha='right', fontsize=7,
        )
        ax.set_ylabel('Rank Stability Score')
        ax.set_title('Province Rank Stability — Line + CI Bands', pad=12)
        ax.legend(loc='lower left', fontsize=8, ncol=2)
        plt.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 14 – Rank vs Stability Scatter (quadrant analysis)
    # ==================================================================

    def plot_rank_stability_scatter(
        self,
        final_ranking: Dict[str, int],
        rank_stability: Dict[str, float],
        top_n_label: int = 10,
        save_name: str = 'fig14_rank_stability_scatter.png',
    ) -> Optional[str]:
        """
        Produce a rank vs. stability quadrant scatter plot.

        Identifies 'Elite' leaders (high rank, high stability) vs. 'At-risk' 
        leaders (high rank, low stability).

        Parameters
        ----------
        final_ranking : Dict[str, int]
            Map of province to its final integer rank.
        rank_stability : Dict[str, float]
            Map of province to its stability score.
        top_n_label : int, default=10
            Number of extreme points to label.
        save_name : str, default='fig14_rank_stability_scatter.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
        if not HAS_MATPLOTLIB:
            return None
        if not final_ranking or not rank_stability:
            return None

        common = sorted(set(final_ranking) & set(rank_stability))
        if len(common) < 2:
            return None

        ranks = np.array([final_ranking[p] for p in common], dtype=float)
        stabs = np.array([rank_stability[p] for p in common], dtype=float)

        med_rank = float(np.median(ranks))
        med_stab = float(np.median(stabs))

        quadrant_colors = {
            'Elite':                  ('#17B169', 'Good rank + High stability'),
            'Stable under-performers': ('#2E86AB', 'Poor rank + High stability'),
            'At-risk leaders':         ('#E07B39', 'Good rank + Volatile'),
            'Underperformers':         ('#C73E1D', 'Poor rank + Volatile'),
        }

        def _qcolor(r: float, s: float) -> str:
            if r <= med_rank and s >= med_stab:
                return '#17B169'
            if r > med_rank and s >= med_stab:
                return '#2E86AB'
            if r <= med_rank and s < med_stab:
                return '#E07B39'
            return '#C73E1D'

        colors_ = [_qcolor(r, s) for r, s in zip(ranks, stabs)]

        fig, ax = plt.subplots(figsize=(12, 9))

        # Quadrant shading
        ax.axvspan(0, med_rank, ymin=0, ymax=1, facecolor='#EAFBF1', alpha=0.25)
        ax.axhspan(med_stab, 1.0, facecolor='#E8F4FD', alpha=0.25)

        ax.axvline(med_rank, color='gray', lw=1.2, ls='--', alpha=0.7)
        ax.axhline(med_stab, color='gray', lw=1.2, ls='--', alpha=0.7)

        sc = ax.scatter(ranks, stabs, c=colors_, s=90, alpha=0.85,
                        edgecolors='white', linewidths=0.7, zorder=3)

        # Label extremes
        sort_idx = np.argsort(stabs)
        label_idx = set(
            sort_idx[:top_n_label // 2].tolist() +
            sort_idx[-top_n_label // 2:].tolist() +
            np.argsort(ranks)[:top_n_label // 2].tolist()
        )
        for i in label_idx:
            if i < len(common):
                ax.annotate(
                    self._truncate(common[i], 14),
                    (ranks[i], stabs[i]),
                    xytext=(5, 4), textcoords='offset points',
                    fontsize=7, color='#333333',
                    arrowprops=dict(arrowstyle='->', color='gray',
                                   lw=0.8, shrinkA=3, shrinkB=3),
                )

        # Quadrant labels
        for (label, (c, desc)), (rx, ry) in zip(quadrant_colors.items(), [
            (med_rank * 0.5, med_stab + (1 - med_stab) * 0.7),
            (med_rank + (max(ranks) - med_rank) * 0.5, med_stab + (1 - med_stab) * 0.7),
            (med_rank * 0.5, med_stab * 0.3),
            (med_rank + (max(ranks) - med_rank) * 0.5, med_stab * 0.3),
        ]):
            ax.text(rx, ry, label, ha='center', va='center',
                    fontsize=10, fontweight='bold', color=c, alpha=0.55,
                    style='italic')

        ax.set_xlabel('Final Rank (lower = better)', fontsize=11)
        ax.set_ylabel('Rank Stability Score (higher = more stable)', fontsize=11)
        ax.set_title('Rank vs. Stability — Quadrant Analysis', pad=12)

        # Legend proxy
        from matplotlib.lines import Line2D
        proxies = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                   markersize=9, label=f'{lbl}\n({desc})')
            for lbl, (c, desc) in quadrant_colors.items()
        ]
        ax.legend(handles=proxies, fontsize=8, loc='upper right',
                  framealpha=0.9, title='Quadrant', title_fontsize=9)

        plt.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 14b – Rank-Change Violin (per criterion)
    # ==================================================================

    def plot_rank_change_violin(
        self,
        perturbation_analysis: Dict[str, Any],
        provinces: Optional[List[str]] = None,
        save_name: str = 'fig14b_rank_change_violin.png',
    ) -> Optional[str]:
        """
        Produce violin plots of rank-change distributions.

        Visualizes the full probability distribution of rank deviations 
        for the most volatile provinces across Monte Carlo trials.

        Parameters
        ----------
        perturbation_analysis : Dict[str, Any]
            Output from the stochastic simulation engine.
        provinces : List[str], optional
            Ordered list of province names for indexing.
        save_name : str, default='fig14b_rank_change_violin.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
        if not HAS_MATPLOTLIB:
            return None
        if not perturbation_analysis:
            return None

        sim = perturbation_analysis.get('simulated_rankings')
        mean_r = perturbation_analysis.get('mean_rank')
        std_r = perturbation_analysis.get('std_rank')

        if sim is None or not hasattr(sim, 'shape'):
            return None

        sim = np.asarray(sim, dtype=float)
        if sim.ndim != 2:
            return None

        n_sim, n_prov = sim.shape

        # Identify provinces to display (top_n by std of rank change)
        if mean_r is not None:
            mean_r = np.asarray(mean_r, dtype=float)
            dev = sim - mean_r[np.newaxis, :]
        else:
            mean_r = sim.mean(axis=0)
            dev = sim - mean_r[np.newaxis, :]

        if std_r is not None:
            std_r = np.asarray(std_r, dtype=float)
        else:
            std_r = dev.std(axis=0)

        top_n = min(20, n_prov)
        top_idx = np.argsort(std_r)[-top_n:][::-1]  # highest std first

        if provinces is not None and len(provinces) == n_prov:
            labels_ = [self._truncate(provinces[i], 14) for i in top_idx]
        else:
            labels_ = [f'P{i+1}' for i in top_idx]

        data_list = [dev[:, i] for i in top_idx]

        fig, ax = plt.subplots(figsize=(max(12, top_n * 0.7), 8))

        parts = ax.violinplot(data_list, positions=range(top_n),
                              showmedians=True, showextrema=True,
                              widths=0.65)

        # Colour by std magnitude
        stds_shown = std_r[top_idx]
        vmax_s = float(stds_shown.max()) if stds_shown.max() > 0 else 1.0
        cmap_v = plt.colormaps['YlOrRd']

        for i, body in enumerate(parts['bodies']):
            c = cmap_v(stds_shown[i] / vmax_s)
            body.set_facecolor(c)
            body.set_alpha(0.7)
            body.set_edgecolor('white')

        for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if key in parts:
                parts[key].set_color('black')
                parts[key].set_linewidth(1.2)

        # Overlay box stats
        for i, d in enumerate(data_list):
            q1, med, q3 = np.percentile(d, [25, 50, 75])
            ax.scatter([i], [med], color='white', s=30, zorder=5)

        ax.axhline(0, color='gray', lw=1.2, ls='--', alpha=0.7,
                   label='Zero rank change')
        ax.set_xticks(range(top_n))
        ax.set_xticklabels(labels_, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Rank Deviation from Mean (positions)')
        ax.set_title(
            f'Monte Carlo Rank-Change Distributions — Top {top_n} Most Volatile Provinces',
            pad=12,
        )
        ax.legend(fontsize=9)

        # Colourbar for std axis
        sm = plt.cm.ScalarMappable(
            cmap=cmap_v, norm=plt.Normalize(0, vmax_s),
        )
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
        cb.set_label('Rank Std Dev', fontsize=9)

        plt.tight_layout()
        return self._save(fig, save_name)


__all__ = ['SensitivityPlotter']
