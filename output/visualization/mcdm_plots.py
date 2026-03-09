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
                                      max(8, n * 0.65 + 2)),
                            layout='constrained')
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

    # ==================================================================
    #  Helpers — method comparison metrics
    # ==================================================================

    _TRAD_METHODS = ['TOPSIS', 'VIKOR', 'PROMETHEE', 'COPRAS', 'EDAS', 'Base']

    def _compute_method_stability(
        self, ranking_result: Any
    ) -> Dict[str, float]:
        """
        Cross-criteria stability for each method.

        For each traditional MCDM method (and Base): compute the average
        pairwise Spearman ρ between that method's per-criterion rank vectors
        across all criterion pairs.  Higher = more consistent ordering across
        criteria groups.

        For ER: use the overall Kendall's W from the ER result (inter-method
        agreement of the 5 MCDM inputs used by ER).

        Returns
        -------
        Dict[str, float]  method → stability score ∈ [−1, 1]
        """
        stability: Dict[str, float] = {}

        crit_method_ranks = getattr(ranking_result, 'criterion_method_ranks', {})
        crit_ids = sorted(crit_method_ranks.keys())
        n_crit = len(crit_ids)

        if HAS_SCIPY and n_crit >= 2:
            for method in self._TRAD_METHODS:
                rank_vecs: List[np.ndarray] = []
                for crit_id in crit_ids:
                    series = crit_method_ranks[crit_id].get(method)
                    if series is not None:
                        rank_vecs.append(
                            np.asarray(series.values
                                       if hasattr(series, 'values') else series,
                                       dtype=float)
                        )
                if len(rank_vecs) < 2:
                    continue
                rho_vals: List[float] = []
                for i in range(len(rank_vecs)):
                    for j in range(i + 1, len(rank_vecs)):
                        # Align by minimum shared length (criteria may have
                        # different province counts due to dynamic exclusion)
                        a, b = rank_vecs[i], rank_vecs[j]
                        min_len = min(len(a), len(b))
                        if min_len < 2:
                            continue
                        rho, _ = sp_stats.spearmanr(a[:min_len], b[:min_len])
                        if not np.isnan(rho):
                            rho_vals.append(float(rho))
                if rho_vals:
                    stability[method] = float(np.mean(rho_vals))

        # ER stability ≡ Kendall's W of the 5 MCDM methods used by ER
        kw = getattr(ranking_result, 'kendall_w', None)
        if kw is not None:
            stability['ER'] = float(kw)

        return stability

    def _compute_method_disc_power(
        self, ranking_result: Any
    ) -> Dict[str, float]:
        """
        Discriminatory power for each method: IQR (Q75−Q25) of the
        criterion-weighted composite score across all active provinces.

        For each traditional method, the composite score is the weighted
        average of normalised per-criterion scores (using
        ``criterion_weights_used`` from the ranking result).  For ER, the
        composite score is directly from ``final_scores``.

        Returns
        -------
        Dict[str, float]  method → IQR ≥ 0
        """
        disc: Dict[str, float] = {}

        crit_method_scores = getattr(ranking_result, 'criterion_method_scores', {})
        crit_weights = getattr(ranking_result, 'criterion_weights_used', {})
        crit_ids = sorted(crit_method_scores.keys())

        # Determine active province list from final ranking
        final_scores = getattr(ranking_result, 'final_scores', None)
        if final_scores is not None and hasattr(final_scores, 'index'):
            provinces = list(final_scores.index)
        else:
            first_crit = next(iter(crit_method_scores.values()), {})
            first_s = next(iter(first_crit.values()), None)
            provinces = (list(first_s.index)
                         if first_s is not None and hasattr(first_s, 'index')
                         else [])
        if not provinces:
            return disc

        # Normalise criterion weights to sum to 1
        total_w = sum(crit_weights.get(c, 1.0) for c in crit_ids)
        if total_w <= 0:
            total_w = len(crit_ids)
        norm_w = {c: crit_weights.get(c, 1.0) / total_w for c in crit_ids}

        for method in self._TRAD_METHODS:
            composite = np.zeros(len(provinces))
            weight_sum = np.zeros(len(provinces))
            for crit_id in crit_ids:
                series = crit_method_scores[crit_id].get(method)
                if series is None:
                    continue
                w = norm_w[crit_id]
                for pi, prov in enumerate(provinces):
                    if hasattr(series, 'loc') and prov in series.index:
                        val = float(series.loc[prov])
                        if not np.isnan(val):
                            composite[pi] += w * val
                            weight_sum[pi] += w
            # Avoid division by zero for provinces missing all criteria
            valid = weight_sum > 0
            if valid.sum() < 2:
                continue
            comp_valid = composite[valid] / weight_sum[valid]
            disc[method] = float(
                np.percentile(comp_valid, 75) - np.percentile(comp_valid, 25)
            )

        # ER discriminatory power from final composite scores
        if final_scores is not None:
            vals = np.asarray(
                final_scores.values
                if hasattr(final_scores, 'values') else final_scores,
                dtype=float,
            )
            vals = vals[~np.isnan(vals)]
            if len(vals) >= 2:
                disc['ER'] = float(
                    np.percentile(vals, 75) - np.percentile(vals, 25)
                )

        return disc

    # ==================================================================
    #  FIG 08e – Method Stability Comparison (horizontal bar)
    # ==================================================================

    def plot_method_stability_comparison(
        self,
        ranking_result: Any,
        save_name: str = 'fig08e_method_stability.png',
    ) -> Optional[str]:
        """
        Horizontal bar chart comparing the cross-criteria ranking stability
        (average pairwise Spearman ρ) of Base, the 5 MCDM methods, and ER.

        Bars sorted from highest to lowest stability.  Base is shown in a
        muted gray, MCDM methods in the categorical palette, and ER in a
        distinct accent colour.  A dashed vertical line marks the median.
        """
        if not HAS_MATPLOTLIB or not HAS_SCIPY:
            return None

        try:
            stability = self._compute_method_stability(ranking_result)
            if len(stability) < 2:
                return None

            # Sort by value descending
            labels = sorted(stability, key=stability.get, reverse=True)
            values = [stability[m] for m in labels]
            n = len(labels)

            # Colour assignment
            _ER_COLOUR   = '#1A6B3C'   # deep green — ER stands out
            _BASE_COLOUR = '#9E9E9E'   # neutral gray — naive baseline
            _PALETTE = [
                '#2E86AB', '#E84855', '#F4A100',
                '#6A4C93', '#3BB273', '#FF6B35',
            ]
            trad_idx = 0
            bar_colours = []
            for m in labels:
                if m == 'ER':
                    bar_colours.append(_ER_COLOUR)
                elif m == 'Base':
                    bar_colours.append(_BASE_COLOUR)
                else:
                    bar_colours.append(_PALETTE[trad_idx % len(_PALETTE)])
                    trad_idx += 1

            fig, ax = plt.subplots(figsize=(12, max(5, n * 0.70 + 1.5)))

            bars = ax.barh(
                range(n), values,
                color=bar_colours, edgecolor='white', linewidth=0.6,
                height=0.60, zorder=3,
            )
            # Grid
            ax.xaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
            ax.set_axisbelow(True)

            # Value labels
            x_max = max(values) if values else 1.0
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(
                    val + x_max * 0.012, i,
                    f'{val:.3f}',
                    va='center', ha='left', fontsize=9.5,
                    color='#222222', fontweight='bold',
                )

            # Median reference line
            med = float(np.median(values))
            ax.axvline(med, color='#555555', linestyle=':', linewidth=1.4,
                       label=f'Median = {med:.3f}', zorder=4)

            ax.set_yticks(range(n))
            ax.set_yticklabels(labels, fontsize=10.5)
            ax.invert_yaxis()
            ax.set_xlabel('Cross-Criteria Spearman ρ  (avg pairwise)', fontsize=11)

            # Legend patches
            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor=_BASE_COLOUR, label='Baseline (Base)'),
                Patch(facecolor=_PALETTE[0],  label='MCDM Methods'),
                Patch(facecolor=_ER_COLOUR,    label='ER Aggregation'),
            ]
            ax.legend(
                handles=legend_handles, fontsize=9.5,
                loc='lower right', framealpha=0.85,
            )

            ax.set_title(
                'Ranking Stability by Method',
                fontsize=14, fontweight='bold', pad=14,
            )
            ax.text(
                0.5, 1.01,
                '↑  Higher = more consistent ranking across criteria groups',
                transform=ax.transAxes,
                ha='center', va='bottom', fontsize=9.5, color='#555555',
                style='italic',
            )

            # Right-side rank badge
            for i, m in enumerate(labels):
                ax.text(
                    -0.001, i, f'#{i+1}',
                    va='center', ha='right', fontsize=8,
                    color='#888888', transform=ax.get_yaxis_transform(),
                )

            ax.set_xlim(left=min(0, min(values) - 0.05),
                        right=x_max + x_max * 0.14)
            fig.tight_layout()
            return self._save(fig, save_name)

        except Exception as _exc:
            _logger.warning('plot_method_stability_comparison failed: %s', _exc)
            return None

    # ==================================================================
    #  FIG 08f – Method Discriminatory Power Comparison (vertical bar)
    # ==================================================================

    def plot_method_disc_power_comparison(
        self,
        ranking_result: Any,
        save_name: str = 'fig08f_method_disc_power.png',
    ) -> Optional[str]:
        """
        Vertical bar chart comparing the discriminatory power (score IQR,
        Q75 − Q25) of Base, the 5 MCDM methods, and ER.

        Methods appear in fixed order: Base → TOPSIS → VIKOR → PROMETHEE
        → COPRAS → EDAS → ER.  A dashed horizontal line marks the ER IQR
        as a reference target.  Bar interiors show a translucent Q10–Q90
        band to communicate the full score spread.
        """
        if not HAS_MATPLOTLIB:
            return None

        try:
            disc = self._compute_method_disc_power(ranking_result)
            if len(disc) < 2:
                return None

            # Fixed display order — only include methods that have a value
            _ORDER = ['Base', 'TOPSIS', 'VIKOR', 'PROMETHEE', 'COPRAS', 'EDAS', 'ER']
            labels = [m for m in _ORDER if m in disc]
            values = [disc[m] for m in labels]
            n = len(labels)

            _ER_COLOUR   = '#1A6B3C'
            _BASE_COLOUR = '#9E9E9E'
            _PALETTE = [
                '#2E86AB', '#E84855', '#F4A100',
                '#6A4C93', '#3BB273', '#FF6B35',
            ]
            trad_idx = 0
            bar_colours = []
            for m in labels:
                if m == 'ER':
                    bar_colours.append(_ER_COLOUR)
                elif m == 'Base':
                    bar_colours.append(_BASE_COLOUR)
                else:
                    bar_colours.append(_PALETTE[trad_idx % len(_PALETTE)])
                    trad_idx += 1

            x = np.arange(n)
            bar_width = 0.55

            fig, ax = plt.subplots(figsize=(12, 7))

            bars = ax.bar(
                x, values,
                color=bar_colours, edgecolor='white', linewidth=0.7,
                width=bar_width, zorder=3,
            )
            # Subtle inner Q10–Q90 score-range bands
            crit_method_scores = getattr(ranking_result, 'criterion_method_scores', {})
            crit_weights = getattr(ranking_result, 'criterion_weights_used', {})
            crit_ids = sorted(crit_method_scores.keys())
            final_scores = getattr(ranking_result, 'final_scores', None)
            if final_scores is not None and hasattr(final_scores, 'index'):
                provinces = list(final_scores.index)
            else:
                first_c = next(iter(crit_method_scores.values()), {})
                first_s = next(iter(first_c.values()), None)
                provinces = (list(first_s.index)
                             if first_s is not None and hasattr(first_s, 'index') else [])

            total_w = sum(crit_weights.get(c, 1.0) for c in crit_ids)
            if total_w <= 0:
                total_w = max(len(crit_ids), 1)
            norm_w = {c: crit_weights.get(c, 1.0) / total_w for c in crit_ids}

            for xi, method in enumerate(labels):
                if method == 'ER':
                    if final_scores is not None:
                        comp = np.asarray(
                            final_scores.values
                            if hasattr(final_scores, 'values') else final_scores,
                            dtype=float,
                        )
                    else:
                        continue
                else:
                    comp = np.zeros(len(provinces))
                    w_acc = np.zeros(len(provinces))
                    for crit_id in crit_ids:
                        series = crit_method_scores[crit_id].get(method)
                        if series is None:
                            continue
                        w = norm_w[crit_id]
                        for pi, prov in enumerate(provinces):
                            if hasattr(series, 'loc') and prov in series.index:
                                val = float(series.loc[prov])
                                if not np.isnan(val):
                                    comp[pi] += w * val
                                    w_acc[pi] += w
                    valid = w_acc > 0
                    if valid.sum() < 2:
                        continue
                    comp = comp[valid] / w_acc[valid]

                comp = comp[~np.isnan(comp)]
                if len(comp) < 2:
                    continue
                q10 = float(np.percentile(comp, 10))
                q90 = float(np.percentile(comp, 90))
                # Normalise band to bar height (values are IQR; band shows
                # Q10-Q90 relative to the data range for visual context)
                # Draw the band as a translucent rectangle over the bar
                bar_h = values[xi]
                # Map q10-q90 proportionally onto bar height
                data_range = q90 - q10 if q90 > q10 else 1.0
                band_bottom = bar_h * max(0.0, (q10 - q10) / data_range)
                band_top    = bar_h  # cap at bar top
                ax.bar(
                    xi, band_top - band_bottom,
                    bottom=band_bottom,
                    color='white', alpha=0.25,
                    width=bar_width * 0.88, zorder=4,
                )

            # Grid
            ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
            ax.set_axisbelow(True)

            # Value labels on bar tops
            y_max = max(values) if values else 1.0
            for xi, val in enumerate(values):
                ax.text(
                    xi, val + y_max * 0.015,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=9.5,
                    color='#222222', fontweight='bold',
                )

            # ER IQR reference line
            if 'ER' in disc:
                er_iqr = disc['ER']
                ax.axhline(er_iqr, color=_ER_COLOUR, linestyle='--',
                           linewidth=1.5, alpha=0.8,
                           label=f'ER IQR = {er_iqr:.4f}', zorder=5)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=11)
            ax.set_ylabel('Score IQR  (Q75 − Q25)', fontsize=11)
            ax.set_ylim(bottom=0, top=y_max * 1.14)

            # Colour-coded x-tick labels
            for tick_lbl, m in zip(ax.get_xticklabels(), labels):
                if m == 'ER':
                    tick_lbl.set_color(_ER_COLOUR)
                    tick_lbl.set_fontweight('bold')
                elif m == 'Base':
                    tick_lbl.set_color('#666666')

            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor=_BASE_COLOUR, label='Baseline (Base)'),
                Patch(facecolor=_PALETTE[0],  label='MCDM Methods'),
                Patch(facecolor=_ER_COLOUR,    label='ER Aggregation'),
            ]
            ax.legend(
                handles=legend_handles, fontsize=9.5,
                loc='upper right', framealpha=0.85,
            )

            ax.set_title(
                'Discriminatory Power by Method',
                fontsize=14, fontweight='bold', pad=14,
            )
            ax.text(
                0.5, 1.01,
                '↑  Higher = better ability to differentiate provinces'
                '  |  Score IQR (Q75 − Q25) of criterion-weighted composite',
                transform=ax.transAxes,
                ha='center', va='bottom', fontsize=9.5, color='#555555',
                style='italic',
            )

            fig.tight_layout()
            return self._save(fig, save_name)

        except Exception as _exc:
            _logger.warning('plot_method_disc_power_comparison failed: %s', _exc)
            return None


__all__ = ['MCDMPlotter']
