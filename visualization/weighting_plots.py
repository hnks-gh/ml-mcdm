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
        """Polygon radar (all sub-criteria, all methods).

        Uses the shared ``_draw_radar`` helper so the polygon grid,
        north-start, and clockwise spoke order are applied.
        """
        if not HAS_MATPLOTLIB:
            return None

        methods = [m for m in _METHOD_ORDER if m in weights]
        if not methods:
            methods = list(weights.keys())

        return self._draw_radar(
            weights, component_names, methods,
            title='Weight Profiles — Radar Comparison',
            save_name=save_name,
        )

    # ==================================================================
    #  FIG 05 – Annotated Weight Heatmap (sequential SC order, no dendrogram)
    # ==================================================================

    def plot_weight_heatmap(
        self,
        weights: Dict[str, np.ndarray],
        component_names: List[str],
        save_name: str = 'fig05_weight_heatmap.png',
    ) -> Optional[str]:
        """
        Annotated weight heatmap with sub-criteria sorted sequentially
        (SC11, SC12, …, SC83).  Criterion-group boundaries are marked with
        a dashed vertical separator for clean readability.  No clustering
        or dendrogram is applied.
        """
        if not HAS_MATPLOTLIB:
            return None

        methods = [m for m in _METHOD_ORDER if m in weights]
        if not methods:
            methods = list(weights.keys())

        n_m = len(methods)
        n_c = len(component_names)
        data = np.array([weights[m] for m in methods])

        # Sort columns sequentially by SC name: SC11 → SC12 → … → SC83
        def _sc_sort_key(name: str) -> tuple:
            try:
                return (int(name[2]), int(name[3:]))
            except Exception:
                return (99, 99)

        col_order = np.array(
            sorted(range(n_c), key=lambda i: _sc_sort_key(component_names[i])))
        data_ord  = data[:, col_order]
        names_ord = [component_names[i] for i in col_order]

        import matplotlib as mpl
        fig, (heat_ax, cbar_ax) = plt.subplots(
            1, 2,
            figsize=(max(14, n_c * 0.62), max(4, n_m + 2)),
            gridspec_kw={'width_ratios': [n_c, 1]},
        )

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

        # Clean dashed separators at criterion-group boundaries + group labels
        prev_c: Optional[int] = None
        first_j_of_group: int = 0
        for j, sc in enumerate(names_ord):
            try:
                c = int(sc[2])
            except Exception:
                c = -1
            if prev_c is not None and c != prev_c:
                heat_ax.axvline(j - 0.5, color='#555555', lw=1.6,
                                linestyle='--', alpha=0.75)
                # label the completed group at its horizontal mid-point
                mid = (first_j_of_group + j - 1) / 2
                heat_ax.text(mid, -0.70, f'C0{prev_c}',
                             ha='center', va='top', fontsize=7.5,
                             color='#333333', fontstyle='italic')
                first_j_of_group = j
            prev_c = c

        # Label the last group
        if prev_c is not None:
            mid = (first_j_of_group + n_c - 1) / 2
            heat_ax.text(mid, -0.70, f'C0{prev_c}',
                         ha='center', va='top', fontsize=7.5,
                         color='#333333', fontstyle='italic')

        heat_ax.set_title(
            'Weight Heatmap — Method × Sub-Criterion (sequential order)',
            pad=10, fontsize=13, fontweight='bold',
        )

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                                  orientation='vertical')
        cbar_ax.set_ylabel('Weight', fontsize=10)

        fig.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 04a + 04b – Additional Radar Charts (Criteria + Group radars)
    # ==================================================================

    def plot_weight_radar_grouped(
        self,
        weights: Dict[str, np.ndarray],
        component_names: List[str],
        save_prefix: str = 'fig04',
    ) -> List[Optional[str]]:
        """
        Generate 9 additional radar charts:

        * **fig04a_weight_radar_criteria.png** – one radar for the 8 criteria
          (weights aggregated by summing SC global weights per group).
        * **fig04b_C01_radar.png … fig04b_C08_radar.png** – one radar per
          criterion group using sub-criteria weights *locally normalised*
          (summing to 1 within each group).  All radars overlay Entropy,
          CRITIC and Hybrid.
        """
        if not HAS_MATPLOTLIB:
            return []

        methods = [m for m in _METHOD_ORDER if m in weights]
        if not methods:
            methods = list(weights.keys())
        if not methods:
            return []

        # ── parse SC groups ────────────────────────────────────────────
        groups: Dict[int, List[int]] = {}
        for i, sc in enumerate(component_names):
            try:
                groups.setdefault(int(sc[2]), []).append(i)
            except Exception:
                pass
        if not groups:
            return []

        crit_digits = sorted(groups.keys())           # [1, 2, … 8]
        crit_labels = [f'C0{d}' for d in crit_digits]

        # ── criteria-level weights ─────────────────────────────────────
        crit_w: Dict[str, np.ndarray] = {}
        for method in methods:
            crit_w[method] = np.array(
                [weights[method][np.array(groups[d])].sum()
                 for d in crit_digits])

        saved: List[Optional[str]] = []

        # fig04a – criteria-level radar (all three methods)
        saved.append(self._draw_radar(
            crit_w, crit_labels, methods,
            title='Criterion-Level Weight Radar — Entropy / CRITIC / Hybrid',
            save_name=f'{save_prefix}a_weight_radar_criteria.png',
        ))

        # fig04a-entropy – criteria-level radar (Entropy only)
        if 'Entropy' in crit_w:
            saved.append(self._draw_radar(
                {'Entropy': crit_w['Entropy']}, crit_labels, ['Entropy'],
                title='Criterion-Level Weight Radar — Entropy',
                save_name=f'{save_prefix}a_weight_radar_criteria_entropy.png',
            ))

        # fig04a-critic – criteria-level radar (CRITIC only)
        if 'CRITIC' in crit_w:
            saved.append(self._draw_radar(
                {'CRITIC': crit_w['CRITIC']}, crit_labels, ['CRITIC'],
                title='Criterion-Level Weight Radar — CRITIC',
                save_name=f'{save_prefix}a_weight_radar_criteria_critic.png',
            ))

        # fig04b_C0X – group radars with locally-normalised weights
        for d in crit_digits:
            idxs     = np.array(groups[d])
            sc_names = [component_names[i] for i in idxs]
            local_w: Dict[str, np.ndarray] = {}
            for method in methods:
                raw   = weights[method][idxs]
                total = raw.sum()
                local_w[method] = raw / total if total > 1e-12 else raw
            saved.append(self._draw_radar(
                local_w, sc_names, methods,
                title=(f'C0{d} Sub-Criterion Weights (locally normalised) '
                       '— Entropy / CRITIC / Hybrid'),
                save_name=f'{save_prefix}b_C0{d}_radar.png',
            ))

        return saved

    def _draw_radar(
        self,
        weights: Dict[str, np.ndarray],
        labels: List[str],
        methods: List[str],
        title: str,
        save_name: str,
    ) -> Optional[str]:
        """Low-level helper: draw a single polar radar chart and save it.

        Grid lines are drawn as polygons (octagon for 8 spokes, etc.)
        instead of the default circular grid.  Spoke order starts at
        north (top, π/2) and proceeds clockwise so that C01 appears at
        the 12-o'clock position when the labels are C01–C08.
        """
        n = len(labels)
        if n < 3:
            return None

        # Angles: start at π/2 (north), go clockwise (negative direction).
        angles = [(np.pi / 2 - i * 2 * np.pi / n) % (2 * np.pi)
                  for i in range(n)]
        angles_closed = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(9, 9),
                               subplot_kw={'projection': 'polar'})

        # --- polygon grid lines instead of circular ones ----------------
        ax.set_yticklabels([])            # hide default radial labels
        ax.yaxis.grid(False)              # hide default circular grid
        ax.xaxis.grid(False)              # hide default spoke grid
        ax.spines['polar'].set_visible(False)

        # Determine nice radial ticks
        all_vals = np.concatenate([weights[m] for m in methods])
        r_max = float(np.max(all_vals)) * 1.12
        n_rings = 5
        r_ticks = np.linspace(0, r_max, n_rings + 1)[1:]

        for r in r_ticks:
            polygon_pts = [(a, r) for a in angles] + [(angles[0], r)]
            theta_pts = [p[0] for p in polygon_pts]
            r_pts     = [p[1] for p in polygon_pts]
            ax.plot(theta_pts, r_pts, color='#CCCCCC', lw=0.7,
                    linestyle='-', zorder=0)
            # Radial tick label on first spoke
            ax.text(angles[0], r + r_max * 0.015, f'{r:.3f}',
                    ha='center', va='bottom', fontsize=6.5,
                    color='#888888')

        # Spoke lines from centre to max ring
        for a in angles:
            ax.plot([a, a], [0, r_max], color='#CCCCCC', lw=0.5, zorder=0)

        # --- data polygons ------------------------------------------------
        for i, method in enumerate(methods):
            w    = weights[method]
            vals = list(w) + [w[0]]
            color = _METHOD_COLORS.get(method, CATEGORICAL_COLORS[i])
            ax.plot(angles_closed, vals, 'o-', lw=2.2, label=method,
                    color=color, markersize=5, alpha=0.9)
            ax.fill(angles_closed, vals, alpha=0.10, color=color)

        ax.set_xticks(angles)
        ax.set_xticklabels(
            [self._truncate(lb, 10) for lb in labels], fontsize=9)
        ax.set_rlabel_position(15)
        ax.tick_params(axis='y', labelsize=0)  # keep ticks invisible
        ax.set_rlim(0, r_max)
        ax.set_title(title, y=1.10, fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.12), fontsize=10)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 04c – Hierarchical Weight Rose (Coxcomb) Chart  [creative]
    # ==================================================================

    def plot_weight_hierarchical_rose(
        self,
        weights: Dict[str, np.ndarray],
        component_names: List[str],
        save_name: str = 'fig04c_weight_hierarchical_rose.png',
    ) -> Optional[str]:
        """
        Hierarchical Coxcomb / Rose chart — one panel per weighting method.

        Each petal maps to one criterion; its radial length equals the
        criterion's aggregated global weight.  The petal is angularly
        subdivided into sub-criterion slices proportional to their *local*
        (within-criterion) normalised weights.  Slice shading darkens
        progressively for sub-criterion identity.
        """
        if not HAS_MATPLOTLIB:
            return None

        import matplotlib as mpl

        methods = [m for m in _METHOD_ORDER if m in weights]
        if not methods:
            return None

        # -- parse groups --
        groups: Dict[int, List[int]] = {}
        for i, sc in enumerate(component_names):
            try:
                groups.setdefault(int(sc[2]), []).append(i)
            except Exception:
                pass
        crit_digits = sorted(groups.keys())
        n_crit = len(crit_digits)
        if n_crit == 0:
            return None

        n_m      = len(methods)
        sector_w = 2 * np.pi / n_crit
        gap      = 0.05

        palette = [
            '#2E86AB', '#C73E1D', '#17B169', '#F4A261',
            '#9B5DE5', '#F15BB5', '#00BBF9', '#06D6A0',
        ]
        group_colors = palette[:n_crit]

        fig, axes = plt.subplots(
            1, n_m, figsize=(7 * n_m, 7.5),
            subplot_kw={'projection': 'polar'},
        )
        if n_m == 1:
            axes = [axes]

        for ax, method in zip(axes, methods):
            w = weights[method]

            for ci, d in enumerate(crit_digits):
                idxs       = np.array(groups[d])
                sc_ws      = w[idxs]
                crit_ht    = float(sc_ws.sum())
                local_sum  = crit_ht
                local_frac = (sc_ws / local_sum
                              if local_sum > 1e-12
                              else np.ones(len(idxs)) / len(idxs))

                base_col  = group_colors[ci % len(group_colors)]
                rgba_base = mpl.colors.to_rgba(base_col)

                theta_start = ci * sector_w + gap / 2
                theta_avail = sector_w - gap
                bar_offset  = 0.0

                for si, (frac, _sc_w) in enumerate(zip(local_frac, sc_ws)):
                    t0 = theta_start + bar_offset * theta_avail
                    dt = frac * theta_avail
                    shade     = 0.35 + 0.55 * (si / max(len(idxs) - 1, 1))
                    slice_col = tuple(
                        min(1.0, c * (0.5 + shade))
                        for c in rgba_base[:3]
                    ) + (0.85,)

                    ax.bar(
                        t0 + dt / 2, crit_ht,
                        width=dt, bottom=0,
                        color=slice_col,
                        edgecolor='white', linewidth=0.5,
                    )

                    # Sub-criterion tip label
                    theta_mid = t0 + dt / 2
                    rot_deg   = np.degrees(theta_mid)
                    if 90 < rot_deg < 270:
                        rot_deg -= 180
                    ax.text(
                        theta_mid, crit_ht * 1.08,
                        component_names[idxs[si]],
                        ha='center', va='bottom',
                        fontsize=6, rotation=rot_deg,
                        rotation_mode='anchor',
                        color='#222222',
                    )
                    bar_offset += frac

                # Criterion label at petal centre
                theta_mid = theta_start + theta_avail / 2
                ax.text(
                    theta_mid, crit_ht * 0.50,
                    f'C0{d}',
                    ha='center', va='center',
                    fontsize=9.5, fontweight='bold', color='white',
                )

            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.spines['polar'].set_visible(False)
            ax.set_title(
                method, y=1.08, fontsize=13, fontweight='bold',
                color=_METHOD_COLORS.get(method, 'black'),
            )

        fig.suptitle(
            'Hierarchical Weight Rose — Criteria & Sub-Criteria Breakdown',
            y=1.02, fontsize=15, fontweight='bold',
        )
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
