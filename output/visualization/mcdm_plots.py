"""
MCDM Method-Agreement and Stability Visualizations.

This module provides the `MCDMPlotter` class, which generates 
publication-quality figures for comparing multiple MCDM methods. It 
focuses on rank agreement (Spearman matrices), method stability across 
criteria groups, and discriminatory power (score distribution spread).

Key Figures
-----------
- **fig06 (Agreement Matrix)**: Clustered Spearman rank-correlation heatmap 
  identifies groups of similar methods.
- **fig08e (Stability Comparison)**: Evaluates how consistently a method 
  orders alternatives across different criteria subsets.
- **fig08f (Discriminatory Power)**: Uses score IQR to measure how 
  effectively a method differentiates between top and bottom performers.
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
    """
    Generator for MCDM comparative analysis visualizations.

    Provides tools to assess inter-method consensus, structural stability of 
    rankings, and the statistical resolution of different aggregation 
    algorithms.
    """

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
        Produce a clustered Spearman rank-correlation heatmap.

        Automatically applies hierarchical clustering to reorder methods, 
        making cliques of agreeing techniques visually apparent.

        Parameters
        ----------
        rankings_dict : Dict[str, np.ndarray]
            Dictionary mapping method names to their ranking vectors.
        title : str, default='MCDM Method Rank Agreement (Spearman ρ)'
            The plot title.
        save_name : str, default='fig06_method_agreement.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
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
    #  Helpers — method comparison metrics
    # ==================================================================

    _TRAD_METHODS = ['TOPSIS', 'VIKOR', 'PROMETHEE', 'COPRAS', 'EDAS', 'SAW']

    def _compute_method_stability(
        self, ranking_result: Any
    ) -> Dict[str, float]:
        """
        Calculate cross-criteria ranking stability for each method.

        Stablity is defined as the average pairwise Spearman ρ between 
        a method's per-criterion rank vectors.

        Parameters
        ----------
        ranking_result : RankingResult
            Aggregated results containing criterion-level method rankings.

        Returns
        -------
        Dict[str, float]
            Mapping of method name to stability score in range [-1, 1].
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

        return stability

    def _compute_method_disc_power(
        self, ranking_result: Any
    ) -> Dict[str, float]:
        """
        Calculate the discriminatory power (score IQR) for each method.

        Measures the spread of composite scores to assess how well a method 
        separates alternatives.

        Parameters
        ----------
        ranking_result : RankingResult
            The results containing criterion-weighted composite scores.

        Returns
        -------
        Dict[str, float]
            Mapping of method name to its Inter-Quartile Range (IQR).
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
        Produce a horizontal bar chart of cross-criteria ranking stability.

        Enables comparison of 'Base' (naive) results against traditional 
        MCDM and fused ER rankings.

        Parameters
        ----------
        ranking_result : RankingResult
            The output of the aggregation engine.
        save_name : str, default='fig08e_method_stability.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
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
                Patch(facecolor=_PALETTE[0],  label='MCDM Methods'),
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
        Produce a vertical bar chart of method discriminatory power.

        Compares the IQR of scores across methods to identify which 
        algorithms provide the clearest separation of performance tiers.

        Parameters
        ----------
        ranking_result : RankingResult
            The output of the aggregation engine.
        save_name : str, default='fig08f_method_disc_power.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
        if not HAS_MATPLOTLIB:
            return None

        try:
            disc = self._compute_method_disc_power(ranking_result)
            if len(disc) < 2:
                return None

            # Fixed display order — only include methods that have a value
            _ORDER = ['TOPSIS', 'VIKOR', 'PROMETHEE', 'COPRAS', 'EDAS', 'SAW']
            labels = [m for m in _ORDER if m in disc]
            values = [disc[m] for m in labels]
            n = len(labels)

            _PALETTE = [
                '#2E86AB', '#A23B72', '#F18F01',
                '#17B169', '#C73E1D', '#7B68EE',
            ]
            bar_colours = [_PALETTE[i % len(_PALETTE)] for i in range(n)]

            x = np.arange(n)
            bar_width = 0.55

            fig, ax = plt.subplots(figsize=(12, 7))

            bars = ax.bar(
                x, values,
                color=bar_colours, edgecolor='white', linewidth=0.7,
                width=bar_width, zorder=3, alpha=0.85,
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

            # Median reference line
            med_iqr = float(np.median(values))
            ax.axhline(med_iqr, color='#555555', linestyle='--',
                       linewidth=1.5, alpha=0.7,
                       label=f'Median IQR = {med_iqr:.4f}', zorder=5)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
            ax.set_ylabel('Score IQR  (Q75 − Q25)', fontsize=11, fontweight='bold')
            ax.set_ylim(bottom=0, top=y_max * 1.14)
            ax.set_title(
                'Method Discriminatory Power Comparison\n'
                '(Higher IQR = better discrimination between alternatives)',
                fontsize=13, fontweight='bold', pad=12,
            )
            ax.legend(fontsize=10, loc='upper right')

            fig.tight_layout()
            return self._save(fig, save_name)

        except Exception as _exc:
            _logger.warning('plot_method_disc_power_comparison failed: %s', _exc)
            return None


__all__ = ['MCDMPlotter']
