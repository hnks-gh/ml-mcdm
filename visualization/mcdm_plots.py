# -*- coding: utf-8 -*-
"""
MCDM Method-Agreement Plots (fig06 – fig08)
=============================================

Publication-quality figures for cross-method comparison and criterion-level
analysis.

fig06  – Spearman method-agreement heatmap (clustered)
fig06b – Kendall's W / avg-Spearman bar chart per criterion
fig07  – 2×4 grid of per-criterion parallel-coord panels (one per criterion)
fig08  – Per-criterion method score panels (horizontal bar, top-N)
fig08b – MCDM composite vs ER final score scatter / bubble comparison
fig08c – Province × Criterion ER utility heatmap (Stage-1 belief avg utility)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    BasePlotter, HAS_MATPLOTLIB, HAS_SCIPY,
    PALETTE, CATEGORICAL_COLORS, GRADIENT_CMAPS, plt, sp_stats,
)

# Five traditional MCDM methods used in this project
_METHODS = ['TOPSIS', 'VIKOR', 'PROMETHEE', 'COPRAS', 'EDAS']
_METHOD_COLORS = {m: CATEGORICAL_COLORS[i] for i, m in enumerate(_METHODS)}

_logger = logging.getLogger(__name__)


class MCDMPlotter(BasePlotter):
    """Figures for MCDM inter-method agreement and criterion-level scores."""

    # ==================================================================
    #  FIG 06 – MCDM Method Agreement Matrix (Spearman heatmap, clustered)
    # ==================================================================

    def plot_method_agreement_matrix(
        self,
        rankings_dict: Dict[str, np.ndarray],
        title: str = 'MCDM Method Rank Agreement (Spearman ρ)',
        save_name: str = 'fig06_method_agreement.png',
    ) -> Optional[str]:
        """
        Spearman rank-correlation heatmap across all criterion-method pairs.
        When scipy is available the columns/rows are reordered by hierarchical
        clustering so groups of agreeing methods cluster together.
        """
        if not HAS_MATPLOTLIB or not HAS_SCIPY:
            return None

        methods = list(rankings_dict.keys())
        n = len(methods)
        if n < 2:
            return None

        # Build symmetric Spearman matrix
        corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                r, _ = sp_stats.spearmanr(
                    np.asarray(rankings_dict[methods[i]]),
                    np.asarray(rankings_dict[methods[j]]),
                )
                corr[i, j] = corr[j, i] = float(r)

        # Hierarchical clustering for column/row order
        col_order = list(range(n))
        if HAS_SCIPY and n >= 3:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
            dist = 1 - corr
            np.fill_diagonal(dist, 0)
            dist = np.clip(dist, 0, None)
            Z = linkage(squareform(dist), method='average')
            col_order = list(leaves_list(Z))

        c_ord = corr[np.ix_(col_order, col_order)]
        labels = [self._truncate(methods[i], 16) for i in col_order]

        # Layout: heatmap + dendrogram strip on top
        if HAS_SCIPY and n >= 3:
            fig = plt.figure(figsize=(max(9, n * 0.75 + 2),
                                      max(8, n * 0.65 + 2)))
            gs = fig.add_gridspec(2, 2,
                                  height_ratios=[0.8, n * 0.65 + 1],
                                  width_ratios=[n * 0.75 + 2, 0.6],
                                  hspace=0.02, wspace=0.04)
            ax_dend = fig.add_subplot(gs[0, 0])
            ax_heat = fig.add_subplot(gs[1, 0])
            ax_cbar = fig.add_subplot(gs[1, 1])

            from scipy.cluster.hierarchy import dendrogram
            dendrogram(Z, ax=ax_dend, no_labels=True,
                       color_threshold=0,
                       above_threshold_color='#777777',
                       link_color_func=lambda _: '#555555')
            ax_dend.set_axis_off()
        else:
            fig, (ax_heat, ax_cbar) = plt.subplots(
                1, 2,
                figsize=(max(9, n * 0.75 + 2), max(8, n * 0.65 + 1)),
                gridspec_kw={'width_ratios': [n * 0.75 + 2, 0.6]},
            )

        im = ax_heat.imshow(c_ord, cmap='RdBu_r', vmin=-1, vmax=1,
                            aspect='auto')
        ax_heat.set_xticks(range(n))
        ax_heat.set_yticks(range(n))
        ax_heat.set_xticklabels(labels, rotation=55, ha='right', fontsize=8)
        ax_heat.set_yticklabels(labels, fontsize=8)

        fs = max(5, min(8, 110 // n))
        for i in range(n):
            for j in range(n):
                txt_col = 'white' if abs(c_ord[i, j]) > 0.65 else 'black'
                ax_heat.text(j, i, f'{c_ord[i, j]:.2f}',
                             ha='center', va='center',
                             fontsize=fs, color=txt_col, fontweight='bold')

        avg_corr = (corr.sum() - n) / max(n * (n - 1), 1)
        ax_heat.set_title(
            f'{title}\nAverage pairwise ρ = {avg_corr:.4f}',
            pad=10, fontsize=12, fontweight='bold')

        cb = fig.colorbar(im, cax=ax_cbar, orientation='vertical')
        cb.set_label('Spearman ρ', fontsize=10)

        fig.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 06b – Kendall's W / Avg Spearman per Criterion
    # ==================================================================

    def plot_method_agreement_per_criterion(
        self,
        ranking_result: Any,
        save_name: str = 'fig06b_agreement_per_criterion.png',
    ) -> Optional[str]:
        """
        Bar chart of Spearman inter-method agreement (average pairwise ρ)
        computed separately within each of the 8 criteria, with overall
        Kendall's W overlaid as a reference line.
        """
        if not HAS_MATPLOTLIB or not HAS_SCIPY:
            return None

        try:
            crit_method_ranks = ranking_result.criterion_method_ranks
            if not crit_method_ranks:
                return None

            crit_ids = sorted(crit_method_ranks.keys())
            avg_rhos: List[float] = []

            for crit_id in crit_ids:
                method_ranks = crit_method_ranks[crit_id]
                methods = list(method_ranks.keys())
                nm = len(methods)
                if nm < 2:
                    avg_rhos.append(1.0)
                    continue
                rho_sum, count = 0.0, 0
                for i in range(nm):
                    for j in range(i + 1, nm):
                        r_i = np.asarray(
                            method_ranks[methods[i]].values
                            if hasattr(method_ranks[methods[i]], 'values')
                            else method_ranks[methods[i]])
                        r_j = np.asarray(
                            method_ranks[methods[j]].values
                            if hasattr(method_ranks[methods[j]], 'values')
                            else method_ranks[methods[j]])
                        rho, _ = sp_stats.spearmanr(r_i, r_j)
                        rho_sum += float(rho)
                        count += 1
                avg_rhos.append(rho_sum / count if count else 1.0)

            global_kw = getattr(ranking_result, 'kendall_w', None)
            n = len(crit_ids)
            x = np.arange(n)

            fig, ax = plt.subplots(figsize=(max(9, n * 1.1), 6))

            # Colour bars by agreement level
            bar_colors = [
                '#1A6B3C' if r > 0.80 else
                ('#2E86AB' if r > 0.60 else
                 ('#F4A100' if r > 0.40 else '#C73E1D'))
                for r in avg_rhos
            ]
            bars = ax.bar(x, avg_rhos, color=bar_colors,
                          edgecolor='white', linewidth=0.5,
                          width=0.6, zorder=2)

            # Value annotations
            for xi, val in zip(x, avg_rhos):
                ax.text(xi, val + 0.01, f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9.5,
                        fontweight='bold', color='#222222')

            # Kendall's W reference
            if global_kw is not None:
                ax.axhline(global_kw, ls='--', lw=1.6,
                           color=PALETTE['crimson'],
                           label=f"Kendall's W (global) = {global_kw:.4f}")

            # Agreement threshold bands
            for thresh, label, col in [
                (0.80, 'Strong ρ > 0.80', '#1A6B3C'),
                (0.60, 'Moderate ρ > 0.60', '#2E86AB'),
                (0.40, 'Weak ρ > 0.40',   '#F4A100'),
            ]:
                ax.axhline(thresh, ls=':', lw=0.9, color=col, alpha=0.55)

            # Custom legend for thresholds
            from matplotlib.patches import Patch
            legend_patches = [
                Patch(color='#1A6B3C', label='Strong (ρ > 0.80)'),
                Patch(color='#2E86AB', label='Moderate (0.60 < ρ ≤ 0.80)'),
                Patch(color='#F4A100', label='Weak (0.40 < ρ ≤ 0.60)'),
                Patch(color='#C73E1D', label='Poor (ρ ≤ 0.40)'),
            ]
            handles, existing_labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + legend_patches,
                      labels=existing_labels + [p.get_label() for p in legend_patches],
                      fontsize=9, loc='lower right', ncol=2)

            ax.set_xticks(x)
            ax.set_xticklabels(crit_ids, fontsize=11)
            ax.set_ylabel('Average Pairwise Spearman ρ', fontsize=11)
            ax.set_ylim(0, 1.08)
            ax.set_title('Inter-Method Rank Agreement per Criterion\n'
                         '(average pairwise Spearman ρ across 5 MCDM methods)',
                         fontsize=13, fontweight='bold', pad=10)
            ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
            fig.tight_layout()
            return self._save(fig, save_name)
        except Exception as _exc:
            _logger.warning('plot_method_agreement_bars failed: %s', _exc)
            return None

    # ==================================================================
    #  FIG 07 – 2×4 Grid of Per-Criterion Parallel-Coord Panels
    # ==================================================================

    def plot_rank_parallel_coordinates(
        self,
        rankings_dict: Dict[str, np.ndarray],
        entity_names: List[str],
        top_n: int = 25,
        save_name: str = 'fig07_rank_parallel.png',
    ) -> Optional[str]:
        """
        Backward-compatible single-panel parallel-coord chart used when
        criterion-level data is not available (e.g. direct call).
        """
        return self._plot_parallel_single(
            rankings_dict, entity_names, top_n, save_name)

    def plot_criterion_parallel_grid(
        self,
        ranking_result: Any,
        provinces: List[str],
        top_n: int = 20,
        save_name: str = 'fig07_criterion_parallel_grid.png',
    ) -> Optional[str]:
        """
        2×4 grid of parallel-coordinate panels — one per criterion.
        Each panel shows the five MCDM-method ranks for the top-N provinces
        within that criterion, colour-coded by province position.
        """
        if not HAS_MATPLOTLIB:
            return None

        try:
            crit_method_ranks = ranking_result.criterion_method_ranks
            crit_ids = sorted(crit_method_ranks.keys())
            n_crit = len(crit_ids)
            if n_crit == 0:
                return None

            n_cols = 4
            n_rows = int(np.ceil(n_crit / n_cols))
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(n_cols * 5.5, n_rows * 5),
                squeeze=False,
            )

            cmap = plt.colormaps['tab20']

            for ci, crit_id in enumerate(crit_ids):
                row, col = divmod(ci, n_cols)
                ax = axes[row][col]
                method_ranks = crit_method_ranks[crit_id]
                methods = [m for m in _METHODS if m in method_ranks]
                if not methods:
                    methods = list(method_ranks.keys())
                n_m = len(methods)

                # Determine top-N provinces by first-method rank
                first_r = np.asarray(
                    method_ranks[methods[0]].values
                    if hasattr(method_ranks[methods[0]], 'values')
                    else method_ranks[methods[0]])
                top_idx = np.argsort(first_r)[:top_n]

                for pos, idx in enumerate(top_idx):
                    prov = provinces[idx] if idx < len(provinces) else str(idx)
                    y_vals = []
                    for m in methods:
                        arr = np.asarray(
                            method_ranks[m].values
                            if hasattr(method_ranks[m], 'values')
                            else method_ranks[m])
                        y_vals.append(float(arr[idx]) if idx < len(arr) else np.nan)
                    color = cmap(pos / max(top_n - 1, 1))
                    ax.plot(range(n_m), y_vals, '-o',
                            color=color, alpha=0.70, lw=1.4,
                            ms=4, markeredgecolor='white',
                            markeredgewidth=0.5,
                            label=prov if pos < 5 else '_')

                ax.set_xticks(range(n_m))
                ax.set_xticklabels(
                    [self._truncate(m, 8) for m in methods],
                    rotation=30, ha='right', fontsize=7.5)
                ax.set_ylabel('Rank', fontsize=8)
                ax.invert_yaxis()
                ax.set_title(crit_id, fontsize=11, fontweight='bold')
                ax.yaxis.grid(True, linestyle='--', alpha=0.3)

            # Hide unused panels
            for ci in range(n_crit, n_rows * n_cols):
                row, col = divmod(ci, n_cols)
                axes[row][col].set_visible(False)

            fig.suptitle(
                f'Per-Criterion MCDM Rank Trajectories — Top {top_n} Provinces',
                fontsize=14, fontweight='bold', y=1.01,
            )
            fig.tight_layout()
            return self._save(fig, save_name)
        except Exception as _exc:
            _logger.warning('plot_rank_parallel_coordinates (grid) failed: %s', _exc)
            return None

    def _plot_parallel_single(
        self,
        rankings_dict: Dict[str, np.ndarray],
        entity_names: List[str],
        top_n: int,
        save_name: str,
    ) -> Optional[str]:
        """Internal: single-panel parallel-coord (backward-compat fallback)."""
        methods = list(rankings_dict.keys())
        n_m = len(methods)
        first_ranks = np.asarray(rankings_dict[methods[0]])
        top_idx = np.argsort(first_ranks)[:top_n]

        fig, ax = plt.subplots(figsize=(max(12, n_m * 1.5), 10))
        cmap = plt.colormaps['viridis']
        norm = plt.Normalize(0, top_n - 1)

        for pos, idx in enumerate(top_idx):
            y_vals = [np.asarray(rankings_dict[m])[idx] for m in methods]
            color = cmap(norm(pos))
            ax.plot(range(n_m), y_vals, '-o', color=color, alpha=0.75,
                    lw=1.8, ms=7, markeredgecolor='white',
                    markeredgewidth=0.8,
                    label=entity_names[idx] if pos < 10 else '')

        ax.set_xticks(range(n_m))
        ax.set_xticklabels(
            [self._truncate(m, 14) for m in methods],
            rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Rank (1 = best)')
        ax.invert_yaxis()
        ax.set_title(
            f'Rank Trajectories Across {n_m} MCDM Methods (Top {top_n})',
            pad=12)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8,
                  title='Province', title_fontsize=9)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 08 – Per-Criterion Grouped Bar (Top 5, All Methods in One)
    # ==================================================================

    def plot_criterion_scores(
        self,
        scores: Dict[str, pd.Series],
        criterion_name: str,
        top_n: int = 5,
        save_name: str = 'fig08_criterion_scores.png',
    ) -> Optional[str]:
        """
        Grouped horizontal bar chart showing the top-5 provinces for a
        criterion across all MCDM methods in a single panel.

        Provinces are ranked by their *mean normalised score* across the
        five methods.  Each province gets one horizontal bar per method,
        colour-coded by the method.  This provides a compact 5-in-1 view
        that replaces the former five-panel layout.
        """
        if not HAS_MATPLOTLIB:
            return None

        methods = [m for m in _METHODS if m in scores]
        if not methods:
            methods = list(scores.keys())
        n_methods = len(methods)
        if n_methods == 0:
            return None

        # --- Determine top-N provinces by mean score across methods ------
        # Build province -> list-of-scores; compute mean
        all_provinces: set = set()
        for m in methods:
            s = scores[m]
            all_provinces |= set(
                s.index.tolist() if hasattr(s, 'index') else range(len(s))
            )

        prov_means: Dict[str, float] = {}
        for prov in all_provinces:
            vals = []
            for m in methods:
                s = scores[m]
                if hasattr(s, 'index') and prov in s.index:
                    vals.append(float(s.loc[prov]))
                elif isinstance(prov, int) and prov < len(s):
                    vals.append(float(s.iloc[prov]))
            prov_means[prov] = float(np.mean(vals)) if vals else 0.0

        top_provs = sorted(prov_means, key=prov_means.get, reverse=True)[:top_n]
        n_show = len(top_provs)
        if n_show == 0:
            return None

        # --- Build data matrix: (n_show, n_methods) ----------------------
        data = np.zeros((n_show, n_methods))
        for mi, m in enumerate(methods):
            s = scores[m]
            for pi, prov in enumerate(top_provs):
                if hasattr(s, 'index') and prov in s.index:
                    data[pi, mi] = float(s.loc[prov])
                elif isinstance(prov, int) and prov < len(s):
                    data[pi, mi] = float(s.iloc[prov])

        # --- Plot ---------------------------------------------------------
        bar_h = 0.14
        y = np.arange(n_show)
        offsets = np.linspace(-(n_methods - 1) / 2,
                              (n_methods - 1) / 2,
                              n_methods) * bar_h

        fig, ax = plt.subplots(figsize=(12, max(5, n_show * 1.2)))

        for mi, method in enumerate(methods):
            color = _METHOD_COLORS.get(method, CATEGORICAL_COLORS[mi])
            bars = ax.barh(
                y + offsets[mi], data[:, mi], bar_h,
                label=method, color=color,
                edgecolor='white', linewidth=0.5, alpha=0.88, zorder=3,
            )
            # Value labels
            for bi, val in enumerate(data[:, mi]):
                ax.text(val + 0.003, y[bi] + offsets[mi],
                        f'{val:.3f}', va='center', fontsize=7.5,
                        color='#333333')

        # Mean marker (diamond) for each province
        for pi in range(n_show):
            mean_v = prov_means[top_provs[pi]]
            ax.plot(mean_v, y[pi], 'D', color='black', markersize=6,
                    zorder=5, label='Mean' if pi == 0 else '_')

        ax.set_yticks(y)
        ax.set_yticklabels(
            [self._truncate(str(p), 18) for p in top_provs], fontsize=10,
        )
        ax.invert_yaxis()
        ax.set_xlabel('Normalised Score', fontsize=11)
        ax.set_title(
            f'{criterion_name} — Top {n_show} Provinces (All Methods)',
            fontsize=13, fontweight='bold', pad=10,
        )
        ax.legend(fontsize=9, loc='lower right', ncol=min(n_methods + 1, 3))
        ax.xaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 08b – MCDM Composite vs ER Final Score Scatter
    # ==================================================================

    def plot_mcdm_composite_scatter(
        self,
        ranking_result: Any,
        provinces: List[str],
        save_name: str = 'fig08b_mcdm_composite_scatter.png',
    ) -> Optional[str]:
        """
        Scatter matrix: each MCDM method's composite score (averaged across
        criteria) plotted against the final ER score, one panel per method.
        Point size ∝ rank agreement (inverse of rank-range across methods).
        A diagonal reference line highlights perfect agreement with ER.
        """
        if not HAS_MATPLOTLIB or not HAS_SCIPY:
            return None

        try:
            active_provs = (
                list(ranking_result.final_ranking.index)
                if hasattr(ranking_result.final_ranking, 'index')
                else provinces
            )
            er_scores = np.asarray(
                ranking_result.final_scores.reindex(active_provs).values)

            # Build composite per method (mean score across criteria)
            method_composites: Dict[str, np.ndarray] = {}
            for crit_id, method_scores in ranking_result.criterion_method_scores.items():
                for method, series in method_scores.items():
                    arr = np.asarray(
                        series.reindex(active_provs).values
                        if hasattr(series, 'reindex')
                        else series)
                    if method not in method_composites:
                        method_composites[method] = arr.copy()
                    else:
                        method_composites[method] = method_composites[method] + arr

            n_crit = max(len(ranking_result.criterion_method_scores), 1)
            for m in method_composites:
                method_composites[m] /= n_crit

            methods = [m for m in _METHODS if m in method_composites]
            if not methods:
                methods = list(method_composites.keys())
            n_m = len(methods)
            if n_m == 0:
                return None

            n_cols = min(n_m, 3)
            n_rows = int(np.ceil(n_m / n_cols))
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(5.5 * n_cols, 5 * n_rows),
                squeeze=False,
            )

            er_ranks = np.asarray(
                ranking_result.final_ranking.reindex(active_provs).values,
                dtype=float)
            n_prov = len(active_provs)
            sizes = 15 + 60 * (n_prov - er_ranks) / max(n_prov - 1, 1)

            for idx, method in enumerate(methods):
                row, col = divmod(idx, n_cols)
                ax = axes[row][col]
                x = method_composites[method]
                y = er_scores
                color = _METHOD_COLORS.get(method, CATEGORICAL_COLORS[idx])

                ax.scatter(x, y, s=sizes, c=color,
                           alpha=0.72, edgecolors='black',
                           linewidths=0.4, zorder=3)

                # Diagonal reference (perfect agreement)
                lo = min(x.min(), y.min())
                hi = max(x.max(), y.max())
                ax.plot([lo, hi], [lo, hi], '--', color='gray',
                        lw=1.2, alpha=0.7, label='y = x')

                # Spearman ρ
                rho, pval = sp_stats.spearmanr(x, y)
                ax.text(0.05, 0.95,
                        f'Spearman ρ = {rho:.3f}\np = {pval:.3f}',
                        transform=ax.transAxes, fontsize=9,
                        va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor='#CCCCCC',
                                  alpha=0.9))

                # Regression line
                if len(x) > 2:
                    m_reg, b_reg = np.polyfit(x, y, 1)
                    x_fit = np.linspace(x.min(), x.max(), 200)
                    ax.plot(x_fit, m_reg * x_fit + b_reg,
                            color=color, lw=1.5, alpha=0.6)

                ax.set_xlabel(f'{method} Composite Score', fontsize=10)
                ax.set_ylabel('ER Final Score', fontsize=10)
                ax.set_title(method, fontsize=12, fontweight='bold')
                ax.xaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
                ax.yaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
                ax.set_axisbelow(True)

            # Hide unused panels
            for idx in range(n_m, n_rows * n_cols):
                row, col = divmod(idx, n_cols)
                axes[row][col].set_visible(False)

            fig.suptitle(
                'MCDM Composite Score vs ER Final Score\n'
                '(marker size ∝ final ER rank quality)',
                fontsize=14, fontweight='bold', y=1.02,
            )
            fig.tight_layout()
            return self._save(fig, save_name)
        except Exception as _exc:
            _logger.warning('plot_mcdm_composite_vs_er failed: %s', _exc)
            return None

    # ==================================================================
    #  FIG 08c – Province × Criterion ER Utility Heatmap
    # ==================================================================

    def plot_criterion_er_utility_heatmap(
        self,
        ranking_result: Any,
        provinces: List[str],
        save_name: str = 'fig08c_criterion_er_utility.png',
    ) -> Optional[str]:
        """
        Province × Criterion heatmap of Stage-1 ER average utility
        (the expected utility of each province's criterion-level belief
        distribution).  Provinces are sorted by final ER rank; criteria
        are sorted by mean utility standard deviation (most discriminating
        criteria on the right).
        """
        if not HAS_MATPLOTLIB:
            return None

        try:
            er_res = getattr(ranking_result, 'er_result', None)
            crit_beliefs = getattr(er_res, 'criterion_beliefs', {}) \
                if er_res else {}
            if not crit_beliefs:
                return None

            # Infer criterion ordering
            first_prov = next(iter(crit_beliefs))
            crit_ids = sorted(crit_beliefs[first_prov].keys())

            # Sort provinces by final ER rank
            ranks = ranking_result.final_ranking
            sorted_provs = [p for p in
                            ranks.sort_values().index.tolist()
                            if p in crit_beliefs]

            # Build Province × Criterion matrix of avg utility
            data = np.full((len(sorted_provs), len(crit_ids)), np.nan)
            for pi, prov in enumerate(sorted_provs):
                for ci, crit_id in enumerate(crit_ids):
                    bd = crit_beliefs[prov].get(crit_id)
                    if bd is not None:
                        data[pi, ci] = float(bd.average_utility())

            # Sort criteria by discriminating power (std across provinces)
            crit_std = np.nanstd(data, axis=0)
            crit_order = np.argsort(crit_std)[::-1]
            data   = data[:, crit_order]
            crit_labels = [crit_ids[i] for i in crit_order]

            n_prov, n_crit = data.shape
            fig, ax = plt.subplots(
                figsize=(max(8, n_crit * 1.3),
                         max(10, n_prov * 0.28)))

            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
            im = ax.imshow(data, aspect='auto', cmap='RdYlGn',
                           vmin=vmin, vmax=vmax)

            ax.set_xticks(range(n_crit))
            ax.set_xticklabels(crit_labels, fontsize=10, fontweight='bold')
            ax.set_yticks(range(n_prov))
            rank_vals = ranks.reindex(sorted_provs).values
            ax.set_yticklabels(
                [f'{int(rank_vals[i]):>2d}. {sorted_provs[i]}'
                 for i in range(n_prov)],
                fontsize=8,
            )

            # Annotate cells
            threshold = vmin + 0.60 * (vmax - vmin)
            for i in range(n_prov):
                for j in range(n_crit):
                    val = data[i, j]
                    if not np.isnan(val):
                        txt_col = 'white' if val > threshold else 'black'
                        ax.text(j, i, f'{val:.2f}',
                                ha='center', va='center',
                                fontsize=6, color=txt_col)

            # Criterion-std annotation on x-axis
            for ci, (std_v, lbl) in enumerate(
                    zip(crit_std[crit_order], crit_labels)):
                ax.text(ci, -0.7, f'σ={std_v:.3f}',
                        ha='center', va='top', fontsize=7,
                        color='#555555', style='italic')

            cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.02)
            cbar.set_label('ER Stage-1 Avg Utility', fontsize=10)

            ax.set_title(
                'Province × Criterion ER Average Utility\n'
                '(Stage-1 belief distributions | criteria sorted by σ)',
                fontsize=13, fontweight='bold', pad=10)
            fig.tight_layout()
            return self._save(fig, save_name)
        except Exception as _exc:
            _logger.warning('plot_criterion_er_utility_heatmap failed: %s', _exc)
            return None


__all__ = ['MCDMPlotter']
