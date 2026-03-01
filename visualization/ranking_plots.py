# -*- coding: utf-8 -*-
"""
Ranking & ER Plots (fig01 – fig02)
====================================

Publication-quality figures for the Evidential Reasoning ranking phase.

fig01  – Horizontal lollipop ranking chart with gradient fill
fig01b – Tier-band chart with percentile band overlays
fig01c – Multi-year slope graph (bumpchart) for top-N provinces
fig01d – Final ER belief distribution heatmap (Province × Grade)
fig01e – Rank-uncertainty scatter (rank vs entropy + interval width)
fig02  – ER score distribution: histogram + KDE + rug
fig02b – MC rank-uncertainty horizontal error-bar chart
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

    # ==================================================================
    #  FIG 01b – Tier-Band Lollipop with Percentile Overlays
    # ==================================================================

    def plot_tier_ranking(
        self,
        provinces: List[str],
        scores: np.ndarray,
        ranks: np.ndarray,
        save_name: str = 'fig01b_tier_ranking.png',
    ) -> Optional[str]:
        """
        Horizontal lollipop chart identical to fig01 but with horizontal
        colour-shaded bands marking the five performance tiers
        (Elite top-10 %, High 10–25 %, Upper-Mid 25–50 %,
        Lower-Mid 50–75 %, Low 75–100 %) and province counts per band.
        """
        if not HAS_MATPLOTLIB:
            return None
        scores = np.asarray(scores)
        ranks  = np.asarray(ranks)
        n      = len(provinces)
        order  = np.argsort(ranks)

        TIERS = [
            ('Elite (Top 10 %)',      0,        n * 0.10, '#1A6B3C'),
            ('High (10–25 %)',         n * 0.10, n * 0.25, '#2E86AB'),
            ('Upper-Mid (25–50 %)',   n * 0.25, n * 0.50, '#F4A100'),
            ('Lower-Mid (50–75 %)',   n * 0.50, n * 0.75, '#E5625E'),
            ('Low (75–100 %)',        n * 0.75, n,        '#901E3B'),
        ]

        fig, ax = plt.subplots(figsize=(14, max(10, n * 0.28)))

        # Shaded tier bands (horizontal spans over y positions)
        for label, y0, y1, col in TIERS:
            ax.axhspan(y0 - 0.5, min(y1, n) - 0.5,
                       color=col, alpha=0.07, zorder=0)
            mid = (y0 + min(y1, n)) / 2 - 0.5
            count = int(min(y1, n) - y0)
            ax.text(-0.001, mid,
                    f'  {label}\n  n={count}',
                    va='center', ha='right', fontsize=7,
                    color=col, fontweight='bold',
                    transform=ax.get_yaxis_transform())

        norm = plt.Normalize(scores[order].min(), scores[order].max())
        cmap = plt.colormaps['RdYlGn']

        for i, idx in enumerate(order):
            col = cmap(norm(scores[idx]))
            ax.barh(i, scores[idx], height=0.60,
                    color=col, edgecolor='white', linewidth=0.4, zorder=2)
            ax.plot(scores[idx], i, 'o', color='black',
                    markersize=4.5, zorder=3)
            ax.text(scores[idx] + 0.002, i, f'{scores[idx]:.4f}',
                    va='center', fontsize=7.5, color='#333333')

        # Tier boundary lines
        for _, _, y1, col in TIERS[:-1]:
            ax.axhline(min(y1, n) - 0.5, color=col, lw=0.8,
                       linestyle='--', alpha=0.6, zorder=4)

        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [f'{int(ranks[idx]):>2d}. {provinces[idx]}' for idx in order],
            fontsize=8.5,
        )
        ax.invert_yaxis()
        ax.set_xlabel('Evidential Reasoning Composite Score')
        ax.set_title('Hierarchical ER Final Ranking — Performance Tiers',
                     fontsize=14, pad=12)
        ax.axvline(np.median(scores), ls=':', color='gray', lw=1.2,
                   label=f'Median = {np.median(scores):.4f}')
        ax.legend(loc='lower right', fontsize=9)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.01)
        cbar.set_label('ER Score', fontsize=10)
        fig.tight_layout()
        return self._save(fig, save_name)

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
        Bumpchart / slopegraph showing how the rank of the *top_n* provinces
        (ranked by their mean rank across all years) evolves year by year.
        Each province is a coloured line; rank 1 is at the top.
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
        Heatmap of the final fused belief degrees (Grade 1–5) for every
        province, sorted by ER rank.  Warmer colours indicate stronger
        belief in higher (better) grades.
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
            n_grades    = len(first_bd.degrees)
            grade_labels = [f'Grade {g+1}' for g in range(n_grades)]

            # Sort provinces by ER rank
            ranks = ranking_result.final_ranking
            sorted_provs = [p for p in
                            ranks.sort_values().index.tolist()
                            if p in final_beliefs]

            data = np.array([
                final_beliefs[p].degrees for p in sorted_provs
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
        Scatter plot of ER rank (x) vs belief entropy (y) with point size
        proportional to utility interval width.  Quadrant shading lets
        analysts identify provinces with high rank *and* high uncertainty
        (top-right quadrant) — a flag for additional scrutiny.
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
        Horizontal error-bar chart showing the Monte-Carlo mean rank ± 1 SD
        for every province (or top *top_n* by mean rank).  Marker colour
        encodes P(Top-K) — the empirical probability of appearing in the top
        quartile across 10 000 MC draws.
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
