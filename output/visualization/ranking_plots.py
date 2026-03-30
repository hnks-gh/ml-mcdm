"""
Ranking and Evidential Reasoning Visualizations.

This module provides the `RankingPlotter` class, which generates 
publication-quality diagnostic plots for the final ranking phase. It 
includes multi-year trajectories (slopegraphs), belief distribution 
heatmaps, and uncertainty scatter plots to evaluate the robustness of 
provincial rankings.

Key Figures
-----------
- **fig01c (Slopegraph)**: Multi-year rank trajectories for top-N provinces.
- **fig01d (Belief Heatmap)**: Distribution of fused belief degrees across 
  grades for each province.
- **fig01e (Uncertainty Scatter)**: Rank vs. entropy analysis with 
  quadrant-based risk assessment.
- **fig02b (MC Rank Uncertainty)**: Error-bar chart showing Monte Carlo 
  mean ranks and P(Top-K) probabilities.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from .base import (
    BasePlotter, HAS_MATPLOTLIB, HAS_SCIPY,
    PALETTE, CATEGORICAL_COLORS, GRADIENT_CMAPS, plt, sp_stats,
)

_logger = logging.getLogger(__name__)


class RankingPlotter(BasePlotter):
    """
    Generator for ranking and Evidential Reasoning visualizations.

    Handles the rendering of longitudinal trajectories, belief structures, 
    and Monte Carlo uncertainty metrics for prioritized analytical insight.
    """

    # ==================================================================
    #  FIG 01 – Final ER Ranking  (horizontal lollipop + gradient fill)
    # ==================================================================

    def plot_final_ranking_summary(self, provinces, scores, ranks, **kw):
        """
        Legacy alias for historical compatibility.

        Returns
        -------
        None
        """
        return None

    # ==================================================================
    #  FIG 01c – Multi-Year Slope Graph (Top-N Bumpchart)
    # ==================================================================

    def plot_multiyear_slopegraph(
        self,
        multi_year_results: Dict[int, Any],
        top_n: int = 20,
        save_name: str = 'fig01c_multiyear_slopegraph.png',
    ) -> Optional[str]:
        """
        Produce a multi-year slopegraph (bumpchart) for top-N provinces.

        Tracks the rank evolution of the highest-performing provinces over 
        the entire temporal span of the panel data.

        Parameters
        ----------
        multi_year_results : Dict[int, Any]
            Dictionary mapping year to ranking results.
        top_n : int, default=20
            The number of provinces to include (ordered by mean rank).
        save_name : str, default='fig01c_multiyear_slopegraph.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
        if not HAS_MATPLOTLIB:
            return None
        if not multi_year_results:
            return None

        years = sorted(multi_year_results.keys())
        if len(years) < 2:
            return None

        # Collect rank Series keyed by year
        year_ranks: Dict[int, pd.Series] = {}
        for yr in years:
            res = multi_year_results[yr]
            fr = getattr(res, 'final_ranking', None)
            if fr is not None:
                year_ranks[yr] = fr

        if len(year_ranks) < 2:
            return None

        # Build Province × Year rank DataFrame
        rank_df = pd.DataFrame(year_ranks)
        rank_df.index.name = 'Province'

        # Select top_n by mean rank (lowest = best)
        mean_rank = rank_df.mean(axis=1)
        top_provinces = mean_rank.nsmallest(top_n).index.tolist()
        df_top = rank_df.loc[top_provinces]

        n_prov = len(top_provinces)
        colors = [CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
                  for i in range(n_prov)]

        fig, ax = plt.subplots(figsize=(max(10, len(years) * 1.4),
                                        max(9, n_prov * 0.35)))

        for i, prov in enumerate(top_provinces):
            vals = df_top.loc[prov, years].values.astype(float)
            col  = colors[i]
            ax.plot(years, vals, '-o', color=col, lw=2, markersize=6,
                    alpha=0.85, zorder=3)
            # Left label
            ax.text(years[0] - 0.15, vals[0], prov,
                    va='center', ha='right', fontsize=7.5,
                    color=col, fontweight='bold')
            # Right label with final rank
            ax.text(years[-1] + 0.15, vals[-1],
                    f'{prov} (#{int(vals[-1])})',
                    va='center', ha='left', fontsize=7.5, color=col)

        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], fontsize=9)
        ax.set_ylabel('Rank  (1 = Best)', fontsize=11)
        ax.invert_yaxis()
        ax.set_title(f'Multi-Year Rank Trajectories — Top {top_n} Provinces',
                     fontsize=13, fontweight='bold', pad=12)
        ax.yaxis.grid(True, linestyle='--', alpha=0.35)
        ax.set_xlim(years[0] - 0.8, years[-1] + 0.8)
        fig.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 01d – Final ER Belief Distribution Heatmap
    # ==================================================================

    def plot_belief_heatmap(
        self,
        ranking_result: Any,
        provinces: List[str],
        save_name: str = 'fig01d_belief_heatmap.png',
    ) -> Optional[str]:
        """
        Render a heatmap of fused belief degrees.

        Visualizes the probability mass allocated to each evaluation grade 
        for all active provinces, providing transparency into the ER 
        aggregation logic.

        Parameters
        ----------
        ranking_result : RankingResult
            Aggregated results containing ER belief matrices.
        provinces : List[str]
            List of province names for indexing.
        save_name : str, default='fig01d_belief_heatmap.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
        if not HAS_MATPLOTLIB:
            return None

        try:
            er_res = getattr(ranking_result, 'er_result', None)
            final_beliefs = getattr(er_res, 'final_beliefs', {}) if er_res else {}
            if not final_beliefs:
                return None

            # Determine grade count from first entry
            first_bd    = next(iter(final_beliefs.values()))
            n_grades    = len(first_bd.beliefs)
            grade_labels = [f'Grade {g+1}' for g in range(n_grades)]

            # Sort provinces by ER rank
            ranks = ranking_result.final_ranking
            sorted_provs = [p for p in
                            ranks.sort_values().index.tolist()
                            if p in final_beliefs]

            data = np.array([
                final_beliefs[p].beliefs for p in sorted_provs
            ])   # shape (n_prov, n_grades)

            n_prov = len(sorted_provs)
            fig, ax = plt.subplots(
                figsize=(max(7, n_grades * 1.4), max(10, n_prov * 0.28)))

            im = ax.imshow(data, aspect='auto', cmap='RdYlGn',
                           vmin=0, vmax=data.max())

            ax.set_xticks(range(n_grades))
            ax.set_xticklabels(grade_labels, fontsize=10)
            ax.set_yticks(range(n_prov))
            rank_vals = ranks.reindex(sorted_provs).values
            ax.set_yticklabels(
                [f'{int(rank_vals[i]):>2d}. {sorted_provs[i]}'
                 for i in range(n_prov)],
                fontsize=8,
            )

            # Annotate cells
            threshold = data.max() * 0.55
            for i in range(n_prov):
                for j in range(n_grades):
                    val = data[i, j]
                    if val > 0.005:   # skip near-zero cells
                        txt_col = 'white' if val > threshold else 'black'
                        ax.text(j, i, f'{val:.3f}',
                                ha='center', va='center',
                                fontsize=6.5, color=txt_col)

            cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.02)
            cbar.set_label('Belief Degree', fontsize=10)
            ax.set_title('Final ER Belief Distribution per Province '
                         '(Grade 1 = Excellent … Grade 5 = Bad)',
                         fontsize=13, fontweight='bold', pad=10)
            fig.tight_layout()
            return self._save(fig, save_name)
        except Exception as _exc:
            _logger.warning('plot_belief_heatmap failed: %s', _exc)
            return None

    # ==================================================================
    #  FIG 01e – Rank vs Uncertainty Scatter
    # ==================================================================

    def plot_rank_uncertainty_scatter(
        self,
        ranking_result: Any,
        provinces: List[str],
        save_name: str = 'fig01e_rank_uncertainty_scatter.png',
    ) -> Optional[str]:
        """
        Render a scatter plot of rank vs. belief-entropy uncertainty.

        Identifies 'at-risk' provinces where high ranks might be unstable 
        due to high evidence conflict (entropy).

        Parameters
        ----------
        ranking_result : RankingResult
            The results containing uncertainty data.
        provinces : List[str]
            List of province names for iteration.
        save_name : str, default='fig01e_rank_uncertainty_scatter.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
        if not HAS_MATPLOTLIB:
            return None

        try:
            er_res = getattr(ranking_result, 'er_result', None)
            unc    = getattr(er_res, 'uncertainty', None) if er_res else None
            if unc is None or unc.empty:
                return None

            ranks  = ranking_result.final_ranking.reindex(provinces).values
            scores = ranking_result.final_scores.reindex(provinces).values
            entropy_vals = unc['belief_entropy'].reindex(provinces).values \
                if 'belief_entropy' in unc.columns else np.zeros(len(provinces))
            width_vals   = unc['utility_interval_width'].reindex(provinces).values \
                if 'utility_interval_width' in unc.columns else np.ones(len(provinces))

            valid = ~(np.isnan(ranks) | np.isnan(entropy_vals))
            ranks  = ranks[valid].astype(float)
            ent    = entropy_vals[valid]
            width  = width_vals[valid]
            provs  = [p for p, v in zip(provinces, valid) if v]

            fig, ax = plt.subplots(figsize=(12, 8))

            # Quadrant shading
            med_rank = np.median(ranks)
            med_ent  = np.median(ent)
            ax.axhspan(med_ent, ent.max() * 1.05, xmin=0, xmax=0.5,
                       color='#FFF3CD', alpha=0.45, zorder=0,
                       label='High rank + high uncertainty')
            ax.axhspan(med_ent, ent.max() * 1.05, xmin=0.5, xmax=1,
                       color='#F8D7DA', alpha=0.35, zorder=0)
            ax.axhspan(ent.min() * 0.95, med_ent, xmin=0, xmax=0.5,
                       color='#D4EDDA', alpha=0.35, zorder=0,
                       label='Good rank + low uncertainty')
            ax.axhline(med_ent, ls='--', lw=0.9, color='gray', alpha=0.6)
            ax.axvline(med_rank, ls='--', lw=0.9, color='gray', alpha=0.6)

            # Normalise marker size by width
            w_min, w_max = width.min(), width.max()
            sizes = 30 + 220 * (width - w_min) / (w_max - w_min + 1e-12)

            scatter = ax.scatter(
                ranks, ent, s=sizes, c=ranks,
                cmap='RdYlGn_r', edgecolors='black',
                linewidths=0.4, alpha=0.82, zorder=3,
            )

            # Label top-10 and bottom-10 by rank
            label_set = set(sorted(range(len(provs)),
                                   key=lambda i: ranks[i])[:10]
                            + sorted(range(len(provs)),
                                     key=lambda i: ranks[i])[-10:])
            for i in label_set:
                ax.annotate(
                    provs[i],
                    (ranks[i], ent[i]),
                    textcoords='offset points',
                    xytext=(5, 3),
                    fontsize=7,
                    color='#333333',
                )

            cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label('ER Rank (lower = better)', fontsize=10)

            ax.set_xlabel('ER Rank  (1 = best)', fontsize=11)
            ax.set_ylabel('Belief Entropy  (higher = more uncertain)', fontsize=11)
            ax.set_title('Rank vs Belief-Entropy Uncertainty\n'
                         '(marker size ∝ utility interval width)',
                         fontsize=13, fontweight='bold', pad=10)
            ax.legend(fontsize=9, loc='upper left')
            fig.tight_layout()
            return self._save(fig, save_name)
        except Exception as _exc:
            _logger.warning('plot_rank_uncertainty_scatter failed: %s', _exc)
            return None

    # ==================================================================
    #  FIG 02b – MC Rank-Uncertainty Horizontal Error-Bar Chart
    # ==================================================================

    def plot_mc_rank_uncertainty(
        self,
        mc_province_stats: Dict[str, Any],
        top_n: int = 40,
        save_name: str = 'fig02b_mc_rank_uncertainty.png',
    ) -> Optional[str]:
        """
        Render a horizontal error-bar chart for Monte Carlo rank uncertainty.

        Shows mean ranks and standard deviations across thousands of 
        simulations, with color indicating the probability of the province 
        remaining in the top quartile (P(Top-K)).

        Parameters
        ----------
        mc_province_stats : Dict[str, Any]
            Dictionary containing 'province_mean_rank', 'province_std_rank', 
            and 'province_prob_topK'.
        top_n : int, default=40
            Number of top provinces to include in the plot.
        save_name : str, default='fig02b_mc_rank_uncertainty.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
        if not HAS_MATPLOTLIB:
            return None
        if not mc_province_stats:
            return None

        try:
            mean_r  = mc_province_stats.get('province_mean_rank', {})
            std_r   = mc_province_stats.get('province_std_rank', {})
            prob_tk = mc_province_stats.get('province_prob_topK', {})

            if not mean_r:
                return None

            # Sort by mean rank ascending (best first)
            sorted_provs = sorted(mean_r.keys(), key=lambda p: mean_r[p])
            if len(sorted_provs) > top_n:
                sorted_provs = sorted_provs[:top_n]

            means  = np.array([mean_r.get(p, 0) for p in sorted_provs])
            stds   = np.array([std_r.get(p, 0)  for p in sorted_provs])
            prob   = np.array([prob_tk.get(p, 0) for p in sorted_provs])
            n = len(sorted_provs)

            fig, ax = plt.subplots(figsize=(12, max(8, n * 0.28)))

            # Colour by P(Top-K)
            cmap   = plt.colormaps['RdYlGn']
            colors = [cmap(float(p)) for p in prob]

            for i, (prov, m, s, col) in enumerate(
                    zip(sorted_provs, means, stds, colors)):
                ax.barh(i, m, height=0.55,
                        color=col, alpha=0.75, zorder=2, edgecolor='white')
                ax.errorbar(m, i, xerr=s,
                            fmt='none', ecolor='#444444',
                            elinewidth=1.4, capsize=3, zorder=4)
                ax.plot(m, i, 'D', color='black', markersize=4, zorder=5)

            ax.set_yticks(range(n))
            ax.set_yticklabels(sorted_provs, fontsize=8.5)
            ax.invert_yaxis()
            ax.set_xlabel('MC Mean Rank  (lower = better)', fontsize=11)
            ax.set_title(
                f'Monte-Carlo Rank Uncertainty — Top {n} Provinces\n'
                '(error bars = ±1 SD; colour = P(Top-K))',
                fontsize=13, fontweight='bold', pad=10)

            # Colourbar for P(Top-K)
            import matplotlib as mpl
            sm = mpl.cm.ScalarMappable(
                cmap=cmap,
                norm=mpl.colors.Normalize(vmin=0, vmax=max(prob.max(), 1e-6)))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.02)
            cbar.set_label('P(Top-K)', fontsize=10)

            ax.xaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
            fig.tight_layout()
            return self._save(fig, save_name)
        except Exception as _exc:
            _logger.warning('plot_mc_rank_uncertainty failed: %s', _exc)
            return None


__all__ = ['RankingPlotter']
