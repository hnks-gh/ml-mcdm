# -*- coding: utf-8 -*-
"""
Weighting Plots (fig03 – fig05)
================================

Professional publication-quality figures for the objective-weighting phase.
Six figures are produced:

fig03  – Three-method grouped bar chart (Entropy / CRITIC / Hybrid) with MC CI
fig03b – MC weight uncertainty: per-SC error bars with CI ribbons
fig03c – Criterion-level three-method horizontal grouped bar + donut
fig03d – Diverging deviation: Entropy / CRITIC vs Hybrid baseline
fig04  – Radar (spider) with three methods overlay
fig05  – Annotated weight heatmap with optional dendrogram clustering
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, List, Optional

from .base import (
    BasePlotter, HAS_MATPLOTLIB, HAS_SCIPY,
    CATEGORICAL_COLORS, PALETTE, GRADIENT_CMAPS, plt,
)

# Method display names and shared colour assignments
_METHOD_COLORS = {
    'Entropy': '#2E86AB',   # royal blue
    'CRITIC':  '#C73E1D',   # crimson
    'Hybrid':  '#17B169',   # emerald
}
_METHOD_ORDER = ['Entropy', 'CRITIC', 'Hybrid']

_logger = logging.getLogger(__name__)


class WeightingPlotter(BasePlotter):
    """Figures for the objective-weighting phase."""

    # ==================================================================
    #  FIG 03 – Three-method grouped bar with MC CI ribbons
    # ==================================================================

    def plot_weights_comparison(
        self,
        weights: Dict[str, np.ndarray],
        component_names: List[str],
        title: str = 'Subcriteria Weight Comparison — Entropy / CRITIC / Hybrid',
        save_name: str = 'fig03_weights_comparison.png',
        ci_lower: Optional[np.ndarray] = None,
        ci_upper: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """
        Grouped bar chart comparing up to three weighting methods.  SCs are
        sorted by the Hybrid weight descending so the most influential
        sub-criteria appear on the left.  Optional MC 95 % CI ribbons are
        drawn on Hybrid bars when *ci_lower* / *ci_upper* are supplied.
        """
        if not HAS_MATPLOTLIB:
            return None

        n_c = len(component_names)
        # Canonical order: Entropy first, CRITIC second, Hybrid last
        methods = [m for m in _METHOD_ORDER if m in weights]
        if not methods:
            methods = list(weights.keys())
        n_m = len(methods)

        # Sort sub-criteria by Hybrid (or last-present method) weight desc
        sort_key = 'Hybrid' if 'Hybrid' in weights else methods[-1]
        order = np.argsort(weights[sort_key])[::-1]
        sorted_names = [component_names[i] for i in order]

        bar_width = min(0.22, 0.72 / n_m)
        x = np.arange(n_c)
        offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * bar_width

        fig, ax = plt.subplots(figsize=(max(16, n_c * 0.65), 8))

        for idx, method in enumerate(methods):
            w_sorted = weights[method][order]
            color = _METHOD_COLORS.get(method, CATEGORICAL_COLORS[idx])
            ax.bar(
                x + offsets[idx], w_sorted, bar_width,
                label=method, color=color,
                edgecolor='white', linewidth=0.5, alpha=0.88, zorder=3,
            )
            ax.scatter(x + offsets[idx], w_sorted, s=20,
                       color=color, edgecolors='black', linewidths=0.4,
                       zorder=5)

            # MC 95 % CI overlay on Hybrid bars only
            if method == 'Hybrid' and ci_lower is not None and ci_upper is not None:
                lo = ci_lower[order]
                hi = ci_upper[order]
                ax.fill_between(x + offsets[idx], lo, hi,
                                alpha=0.20, color=color, zorder=2,
                                label='95 % MC CI')
                ax.errorbar(
                    x + offsets[idx], w_sorted,
                    yerr=[w_sorted - lo, hi - w_sorted],
                    fmt='none', ecolor=color, elinewidth=1.0, capsize=2,
                    zorder=6,
                )

        # Faint vertical separators at criterion-group boundaries
        self._draw_criterion_separators(ax, component_names, order, n_c)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [self._truncate(sorted_names[i], 12) for i in range(n_c)],
            rotation=55, ha='right', fontsize=8,
        )
        ax.set_ylabel('Weight', fontsize=11)
        ax.set_title(title, pad=14, fontsize=13, fontweight='bold')

        # De-duplicate legend (CI ribbon adds a duplicate entry)
        handles, labels = ax.get_legend_handles_labels()
        seen: Dict[str, Any] = {}
        for h, l in zip(handles, labels):
            seen.setdefault(l, h)
        ax.legend(seen.values(), seen.keys(),
                  loc='upper right', ncol=min(n_m + 1, 4), fontsize=9)

        ax.set_xlim(-0.6, n_c - 0.4)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 03b – MC Weight Uncertainty (error-bar + CI ribbon per SC)
    # ==================================================================

    def plot_mc_weight_uncertainty(
        self,
        subcriteria: List[str],
        mc_means: np.ndarray,
        mc_stds: np.ndarray,
        ci_lower: np.ndarray,
        ci_upper: np.ndarray,
        criteria_groups: Optional[Dict[str, List[str]]] = None,
        save_name: str = 'fig03b_mc_weight_uncertainty.png',
    ) -> Optional[str]:
        """
        Error-bar chart showing Monte-Carlo weight mean ± 1 SD and 95 % CI
        band for every sub-criterion.  Alternating background shading groups
        sub-criteria by their parent criterion, aiding readability.
        """
        if not HAS_MATPLOTLIB:
            return None

        n = len(subcriteria)
        order = np.argsort(mc_means)[::-1]
        sc_sorted   = [subcriteria[i] for i in order]
        mean_sorted = mc_means[order]
        std_sorted  = mc_stds[order]
        lo_sorted   = ci_lower[order]
        hi_sorted   = ci_upper[order]

        x = np.arange(n)
        fig, ax = plt.subplots(figsize=(max(16, n * 0.65), 7))

        # Alternating background stripes by criterion group
        if criteria_groups:
            sc_to_crit = {sc: cid for cid, scs in criteria_groups.items()
                          for sc in scs}
            crit_order = list(dict.fromkeys(
                sc_to_crit.get(sc, '') for sc in sc_sorted))
            stripe_col = ['#F8F8FF', '#EEF4FF']
            sc_x_map = {sc: i for i, sc in enumerate(sc_sorted)}
            for ci_idx, crit_id in enumerate(
                    c for c in crit_order if c):
                crit_scs = [sc for sc in sc_sorted
                            if sc_to_crit.get(sc) == crit_id]
                if not crit_scs:
                    continue
                xs = [sc_x_map[sc] for sc in crit_scs]
                left, right = min(xs) - 0.4, max(xs) + 0.4
                ax.axvspan(left, right, alpha=0.35,
                           color=stripe_col[ci_idx % 2], zorder=0)
                ax.text((left + right) / 2, hi_sorted.max() * 1.02,
                        crit_id, ha='center', va='bottom',
                        fontsize=7.5, color='#555555', style='italic')

        # 95 % CI ribbon
        ax.fill_between(x, lo_sorted, hi_sorted,
                        alpha=0.25, color=_METHOD_COLORS['Hybrid'],
                        label='95 % MC CI', zorder=1)

        # Mean ± 1 SD error bars
        ax.errorbar(x, mean_sorted, yerr=std_sorted,
                    fmt='o', color=_METHOD_COLORS['Hybrid'],
                    ecolor='#555555', elinewidth=1.2, capsize=3,
                    markersize=5, zorder=4, label='Mean ± 1 SD')

        # Equal-weight reference line
        ax.axhline(1.0 / n, ls=':', lw=1, color='gray',
                   label=f'Equal weight (1/{n})')

        ax.set_xticks(x)
        ax.set_xticklabels(
            [self._truncate(s, 11) for s in sc_sorted],
            rotation=55, ha='right', fontsize=8,
        )
        ax.set_ylabel('Weight', fontsize=11)
        ax.set_title(
            'Monte-Carlo Weight Uncertainty per Sub-Criterion (10 000 draws)',
            pad=12, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.set_xlim(-0.6, n - 0.4)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 03c – Criterion-level three-method horizontal grouped bar
    # ==================================================================

    def plot_criterion_weights_comparison(
        self,
        entropy_crit: Dict[str, float],
        critic_crit: Dict[str, float],
        hybrid_crit: Dict[str, float],
        save_name: str = 'fig03c_criterion_weights.png',
    ) -> Optional[str]:
        """
        Horizontal grouped bar (Entropy / CRITIC / Hybrid) for the 8 criteria
        plus a Hybrid donut chart for at-a-glance weight shares.
        """
        if not HAS_MATPLOTLIB:
            return None

        crit_ids = sorted(
            set(hybrid_crit) | set(entropy_crit) | set(critic_crit))
        crit_ids_sorted = sorted(
            crit_ids, key=lambda c: hybrid_crit.get(c, 0), reverse=True)
        n = len(crit_ids_sorted)
        if n == 0:
            return None

        methods_data: Dict[str, List[float]] = {
            'Entropy': [entropy_crit.get(c, 0) for c in crit_ids_sorted],
            'CRITIC':  [critic_crit.get(c, 0) for c in crit_ids_sorted],
            'Hybrid':  [hybrid_crit.get(c, 0) for c in crit_ids_sorted],
        }

        bar_h = 0.22
        offsets = np.array([-1, 0, 1]) * bar_h
        y = np.arange(n)

        fig, (ax_bar, ax_pie) = plt.subplots(
            1, 2, figsize=(14, max(5, n * 0.65)),
            gridspec_kw={'width_ratios': [3, 1]})

        for idx, method in enumerate(_METHOD_ORDER):
            vals = methods_data[method]
            color = _METHOD_COLORS[method]
            ax_bar.barh(
                y + offsets[idx], vals, bar_h,
                label=method, color=color,
                edgecolor='white', linewidth=0.4, alpha=0.88,
            )
            for yi, v in zip(y + offsets[idx], vals):
                ax_bar.text(v + 0.001, yi, f'{v:.3f}', va='center',
                            fontsize=7.5, color='#333333')

        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels(crit_ids_sorted, fontsize=10)
        ax_bar.set_xlabel('Criterion Weight', fontsize=11)
        ax_bar.set_title('Criterion-Level Weight Comparison',
                         fontsize=13, fontweight='bold', pad=10)
        ax_bar.legend(loc='lower right', fontsize=9)
        ax_bar.axvline(1.0 / n, ls=':', lw=1, color='gray')
        vmax = max(max(v) for v in methods_data.values())
        ax_bar.set_xlim(0, vmax * 1.28)
        ax_bar.xaxis.grid(True, linestyle='--', alpha=0.4)
        ax_bar.set_axisbelow(True)

        # Hybrid donut (right panel)
        hybrid_vals = methods_data['Hybrid']
        wedge_colors = [CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
                        for i in range(n)]
        ax_pie.pie(
            hybrid_vals, labels=crit_ids_sorted,
            colors=wedge_colors, autopct='%1.1f%%',
            startangle=90, pctdistance=0.75,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.2},
            textprops={'fontsize': 8},
        )
        circle = plt.Circle((0, 0), 0.55, color='white')
        ax_pie.add_patch(circle)
        ax_pie.set_title('Hybrid\nWeight Share', fontsize=10,
                         fontweight='bold', pad=6)

        fig.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 03d – Diverging deviation: Entropy / CRITIC vs Hybrid baseline
    # ==================================================================

    def plot_weight_deviation(
        self,
        entropy_sc: Dict[str, float],
        critic_sc: Dict[str, float],
        hybrid_sc: Dict[str, float],
        subcriteria: List[str],
        save_name: str = 'fig03d_weight_deviation.png',
    ) -> Optional[str]:
        """
        Diverging bar chart where each bar shows the signed deviation of
        Entropy and CRITIC from the Hybrid baseline.  Sub-criteria are sorted
        by |Entropy – Hybrid| descending so the most discrepant SCs come first.
        """
        if not HAS_MATPLOTLIB:
            return None

        from matplotlib.patches import Patch

        n = len(subcriteria)
        entropy_dev = np.array([entropy_sc.get(sc, 0) - hybrid_sc.get(sc, 0)
                                for sc in subcriteria])
        critic_dev  = np.array([critic_sc.get(sc, 0)  - hybrid_sc.get(sc, 0)
                                for sc in subcriteria])

        order = np.argsort(np.abs(entropy_dev))[::-1]
        sc_sorted = [subcriteria[i] for i in order]
        ed_sorted = entropy_dev[order]
        cd_sorted = critic_dev[order]

        x = np.arange(n)
        bw = 0.35

        fig, ax = plt.subplots(figsize=(max(16, n * 0.65), 7))

        # Entropy deviation — blue tones
        for xi, val in zip(x - bw / 2, ed_sorted):
            col = '#2E86AB' if val >= 0 else '#A8D5EA'
            ax.bar(xi, val, bw, color=col, edgecolor='white',
                   linewidth=0.4, alpha=0.88)

        # CRITIC deviation — red tones
        for xi, val in zip(x + bw / 2, cd_sorted):
            col = '#C73E1D' if val >= 0 else '#EEA090'
            ax.bar(xi, val, bw, color=col, edgecolor='white',
                   linewidth=0.4, alpha=0.88)

        ax.axhline(0, color='black', lw=1.2, zorder=5)

        legend_elems = [
            Patch(facecolor='#2E86AB', label='Entropy > Hybrid'),
            Patch(facecolor='#A8D5EA', label='Entropy < Hybrid'),
            Patch(facecolor='#C73E1D', label='CRITIC > Hybrid'),
            Patch(facecolor='#EEA090', label='CRITIC < Hybrid'),
        ]
        ax.legend(handles=legend_elems, loc='upper right', ncol=2, fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [self._truncate(s, 11) for s in sc_sorted],
            rotation=55, ha='right', fontsize=8,
        )
        ax.set_ylabel('Deviation from Hybrid Weight', fontsize=11)
        ax.set_title('Method Deviation from Hybrid Baseline (per Sub-Criterion)',
                     pad=12, fontsize=13, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.6, n - 0.4)
        fig.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 04 – Weight Radar Chart (up to 3 methods)
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
        angles += angles[:1]   # close the polygon

        methods = [m for m in _METHOD_ORDER if m in weights]
        if not methods:
            methods = list(weights.keys())

        fig, ax = plt.subplots(figsize=(11, 11), subplot_kw={'projection': 'polar'})

        for i, method in enumerate(methods):
            w = weights[method]
            vals = list(w) + [w[0]]
            color = _METHOD_COLORS.get(method, CATEGORICAL_COLORS[i])
            ax.plot(angles, vals, 'o-', lw=2.2, label=method,
                    color=color, markersize=5, alpha=0.9)
            ax.fill(angles, vals, alpha=0.08, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [self._truncate(c, 10) for c in component_names], fontsize=8,
        )
        ax.set_rlabel_position(30)
        ax.tick_params(axis='y', labelsize=7)
        ax.set_title('Weight Profiles — Radar Comparison', y=1.08,
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.12), fontsize=10)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 05 – Annotated Heatmap with optional dendrogram clustering
    # ==================================================================

    def plot_weight_heatmap(
        self,
        weights: Dict[str, np.ndarray],
        component_names: List[str],
        save_name: str = 'fig05_weight_heatmap.png',
    ) -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        methods = [m for m in _METHOD_ORDER if m in weights]
        if not methods:
            methods = list(weights.keys())

        n_m = len(methods)
        n_c = len(component_names)
        data = np.array([weights[m] for m in methods])

        # Hierarchical clustering on the SC axis when scipy is available
        col_order = np.arange(n_c)
        if HAS_SCIPY and n_c >= 4:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import pdist
            dist = pdist(data.T, metric='euclidean')
            Z = linkage(dist, method='average')
            col_order = leaves_list(Z)

            fig = plt.figure(figsize=(max(14, n_c * 0.6), n_m + 4))
            gs = fig.add_gridspec(
                2, 2, height_ratios=[1.3, 4],
                width_ratios=[n_c, 1],
                hspace=0.02, wspace=0.02,
            )
            dend_ax = fig.add_subplot(gs[0, 0])
            heat_ax = fig.add_subplot(gs[1, 0])
            cbar_ax = fig.add_subplot(gs[1, 1])

            from scipy.cluster.hierarchy import dendrogram
            dendrogram(Z, ax=dend_ax, no_labels=True,
                       color_threshold=0,
                       above_threshold_color='#888888',
                       link_color_func=lambda _: '#555555')
            dend_ax.set_axis_off()
            dendrogrammed = True
        else:
            fig, (heat_ax, cbar_ax) = plt.subplots(
                1, 2,
                figsize=(max(14, n_c * 0.6), max(4, n_m + 2)),
                gridspec_kw={'width_ratios': [n_c, 1]},
            )
            dendrogrammed = False

        data_ord = data[:, col_order]
        names_ord = [component_names[i] for i in col_order]

        import matplotlib as mpl
        cmap = plt.get_cmap('YlOrRd')
        heat_ax.imshow(data_ord, aspect='auto', cmap=cmap, vmin=0)

        heat_ax.set_xticks(range(n_c))
        heat_ax.set_xticklabels(
            [self._truncate(s, 10) for s in names_ord],
            rotation=55, ha='right', fontsize=8,
        )
        heat_ax.set_yticks(range(n_m))
        heat_ax.set_yticklabels(methods, fontsize=10)

        vmax = data_ord.max()
        vmin = data_ord.min()
        threshold = vmin + 0.65 * (vmax - vmin)
        for i in range(n_m):
            for j in range(n_c):
                val = data_ord[i, j]
                txt_col = 'white' if val > threshold else 'black'
                heat_ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                             fontsize=7, color=txt_col, fontweight='bold')

        heat_ax.set_title(
            'Weight Heatmap — Method × Sub-Criterion'
            + (' (clustered)' if dendrogrammed else ''),
            pad=10, fontsize=13, fontweight='bold',
        )

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                                  orientation='vertical')
        cbar_ax.set_ylabel('Weight', fontsize=10)

        fig.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  Internal helpers
    # ==================================================================

    @staticmethod
    def _draw_criterion_separators(
        ax, component_names: List[str], order: np.ndarray, n_c: int
    ) -> None:
        """
        Draw faint dotted vertical separators at criterion-group boundaries.
        Relies on the SC naming convention SC{criterion_digit}{index}.
        """
        try:
            prev_crit = None
            for i, idx in enumerate(order):
                sc = component_names[idx]
                crit = sc[2] if len(sc) >= 3 else '?'
                if prev_crit is not None and crit != prev_crit:
                    ax.axvline(i - 0.5, color='#AAAAAA', lw=0.8,
                               linestyle=':', alpha=0.7, zorder=4)
                prev_crit = crit
        except Exception as _exc:
            _logger.debug('criterion separator drawing failed: %s', _exc)


__all__ = ['WeightingPlotter']
