"""
Weighting Phase Visualizations.

This module provides the `WeightingPlotter` class, which generates 
publication-quality diagnostic plots for the objective criteria weighting 
phase. It covers sub-criteria weight comparisons, uncertainty analysis via 
Monte Carlo simulations, and hierarchical breakdowns using radar and rose 
charts.

Key Figures
-----------
- **fig03 (Weights Comparison)**: Grouped bar chart comparing weighting 
  methods with optional confidence intervals.
- **fig03b (MC Uncertainty)**: Error-bar chart showing Monte Carlo means, 
  standard deviations, and 95% CIs.
- **fig03c (Criterion Weights)**: Horizontal bar and donut chart for 
  higher-level criteria importance.
- **fig04 (Weight Radar)**: Multi-axis radar plots for sub-criteria 
  profiles across years.
- **fig04c (Hierarchical Rose)**: Coxcomb/Rose charts for nested weight 
  distributions.
- **fig05 (Weight Heatmap)**: Annotated heatmaps of weights over time or 
  across methods.
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
    'CRITIC': '#C73E1D',   # crimson
}
_METHOD_ORDER = ['CRITIC']

_logger = logging.getLogger(__name__)


class WeightingPlotter(BasePlotter):
    """
    Generator for criteria weighting visualizations.

    Handles the rendering of bar, radar, rose, and heatmap charts to 
    illustrate the distribution and stability of sub-criteria weights.
    """

    # ==================================================================
    #  FIG 03 – Three-method grouped bar with MC CI ribbons
    # ==================================================================

    def plot_weights_comparison(
        self,
        weights: Dict[str, np.ndarray],
        component_names: List[str],
        title: str = 'Subcriteria Weight Comparison — CRITIC',
        save_name: str = 'fig03_weights_comparison.png',
        ci_lower: Optional[np.ndarray] = None,
        ci_upper: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """
        Produce a grouped bar chart comparing weighting methods.

        Sub-criteria are sorted by weight magnitude (descending) to 
        highlight the most influential factors. 

        Parameters
        ----------
        weights : Dict[str, np.ndarray]
            Dictionary mapping method names to weight arrays.
        component_names : List[str]
            Names of the sub-criteria (e.g., 'SC11').
        title : str, default='Subcriteria Weight Comparison — CRITIC'
            The plot title.
        save_name : str, default='fig03_weights_comparison.png'
            The output filename.
        ci_lower : np.ndarray, optional
            Lower bound of the 95% confidence interval for CRITIC.
        ci_upper : np.ndarray, optional
            Upper bound of the 95% confidence interval for CRITIC.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
        if not HAS_MATPLOTLIB:
            return None

        n_c = len(component_names)
        # Canonical order: CRITIC first
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

            # MC 95 % CI overlay on CRITIC bars when ci arrays supplied
            if method == 'CRITIC' and ci_lower is not None and ci_upper is not None:
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
        Visualize Monte-Carlo weight uncertainty with error bars and CI ribbons.

        Uses alternating vertical background stripes to group sub-criteria 
        by their parent criteria, providing clear hierarchical context.

        Parameters
        ----------
        subcriteria : List[str]
            List of sub-criteria IDs.
        mc_means : np.ndarray
            Mean weights across Monte-Carlo simulations.
        mc_stds : np.ndarray
            Standard deviation of weights across simulations.
        ci_lower : np.ndarray
            95% confidence interval lower bound.
        ci_upper : np.ndarray
            95% confidence interval upper bound.
        criteria_groups : Dict[str, List[str]], optional
            Mapping of criteria IDs to their constituent sub-criteria.
        save_name : str, default='fig03b_mc_weight_uncertainty.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
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
                        alpha=0.25, color=_METHOD_COLORS['CRITIC'],
                        label='95 % CI', zorder=1)

        # Mean ± 1 SD error bars
        ax.errorbar(x, mean_sorted, yerr=std_sorted,
                    fmt='o', color=_METHOD_COLORS['CRITIC'],
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
        critic_crit: Dict[str, float],
        save_name: str = 'fig03c_criterion_weights.png',
    ) -> Optional[str]:
        """
        Plot criterion-level weights using horizontal bars and a donut chart.

        Parameters
        ----------
        critic_crit : Dict[str, float]
            Mapping of criterion IDs to their aggregated CRITIC weights.
        save_name : str, default='fig03c_criterion_weights.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
        if not HAS_MATPLOTLIB:
            return None

        crit_ids_sorted = sorted(
            critic_crit, key=lambda c: critic_crit.get(c, 0), reverse=True)
        n = len(crit_ids_sorted)
        if n == 0:
            return None

        vals = [critic_crit.get(c, 0) for c in crit_ids_sorted]
        y = np.arange(n)
        color = _METHOD_COLORS['CRITIC']

        fig, (ax_bar, ax_pie) = plt.subplots(
            1, 2, figsize=(14, max(5, n * 0.65)),
            gridspec_kw={'width_ratios': [3, 1]})

        ax_bar.barh(y, vals, 0.6, color=color,
                    edgecolor='white', linewidth=0.4, alpha=0.88)
        for yi, v in zip(y, vals):
            ax_bar.text(v + 0.001, yi, f'{v:.3f}', va='center',
                        fontsize=7.5, color='#333333')

        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels(crit_ids_sorted, fontsize=10)
        ax_bar.set_xlabel('Criterion Weight', fontsize=11)
        ax_bar.set_title('Criterion-Level CRITIC Weights',
                         fontsize=13, fontweight='bold', pad=10)
        ax_bar.axvline(1.0 / n, ls=':', lw=1, color='gray')
        ax_bar.set_xlim(0, max(vals) * 1.28)
        ax_bar.xaxis.grid(True, linestyle='--', alpha=0.4)
        ax_bar.set_axisbelow(True)

        # Donut (right panel)
        wedge_colors = [CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
                        for i in range(n)]
        ax_pie.pie(
            vals, labels=crit_ids_sorted,
            colors=wedge_colors, autopct='%1.1f%%',
            startangle=90, pctdistance=0.75,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.2},
            textprops={'fontsize': 8},
        )
        circle = plt.Circle((0, 0), 0.55, color='white')
        ax_pie.add_patch(circle)
        ax_pie.set_title('CRITIC\nWeight Share', fontsize=10,
                         fontweight='bold', pad=6)

        fig.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 03d – removed (required Hybrid baseline)
    # ==================================================================

    def plot_weight_deviation(
        self,
        critic_sc: Dict[str, float],
        subcriteria: List[str],
        save_name: str = 'fig03d_weight_deviation.png',
    ) -> Optional[str]:
        """
        Legacy method for weight deviation plots. (Currently disabled).

        Returns
        -------
        None
        """
        return None

    # ==================================================================
    #  FIG 04 – Weight Radar Chart
    # ==================================================================

    def plot_weight_radar(
        self,
        weights: Dict[str, np.ndarray],
        component_names: List[str],
        weight_all_years: Optional[Dict[int, Any]] = None,
        save_name: str = 'fig04_weight_radar.png',
    ) -> List[Optional[str]]:
        """
        Produce a radar chart of sub-criterion weights.

        If longitudinal data is provided, generates a multi-panel grid 
        spanning all years, plus 14 individual standalone figures.

        Parameters
        ----------
        weights : Dict[str, np.ndarray]
            Current weights to plot.
        component_names : List[str]
            Sub-criteria labels.
        weight_all_years : Dict[int, Any], optional
            Longitudinal weight data for multi-year grids.
        save_name : str, default='fig04_weight_radar.png'
            The output filename.

        Returns
        -------
        List[Optional[str]]
            Paths to the saved figures.
        """
        if not HAS_MATPLOTLIB:
            return []

        saved_paths: List[Optional[str]] = []

        if weight_all_years:
            # 1. Combined 14-panel figure (default 2x7)
            saved_paths.append(self._plot_weight_radar_multiyear(
                weight_all_years, component_names, save_name))

            # 1b. Additional 14-panel figure (4x4 grid, centered bottom row)
            saved_paths.append(self._plot_weight_radar_multiyear_4x4(
                weight_all_years, component_names, save_name))

            # 2. Individual 14 standalone figures (one per year)
            years = sorted(weight_all_years.keys())
            for yr in years:
                gw = weight_all_years[yr].get('global_sc_weights', {})
                w_arr = np.array([gw.get(sc, 0.0) for sc in component_names])

                # Append year to filename: fig04_weight_radar_2011.png
                base_name = save_name.rsplit('.', 1)[0]
                yr_save_name = f"{base_name}_{yr}.png"

                saved_paths.append(self._draw_radar(
                    {'CRITIC': w_arr}, component_names, ['CRITIC'],
                    title=f'CRITIC Weight Radar — {yr}',
                    save_name=yr_save_name,
                ))
            return saved_paths

        methods = [m for m in _METHOD_ORDER if m in weights]
        if not methods:
            methods = list(weights.keys())

        saved_paths.append(self._draw_radar(
            weights, component_names, methods,
            title='Weight Profiles — Radar Comparison',
            save_name=save_name,
        ))
        return saved_paths

    def _draw_radar_to_ax(
        self,
        ax,
        w_vals: np.ndarray,
        labels: List[str],
        r_max: float,
        color: str = '#C73E1D',
        title: str = '',
    ) -> None:
        """Draw a single radar onto an existing polar *ax* (no save)."""
        n = len(labels)
        if n < 3:
            return
        angles = [(np.pi / 2 - i * 2 * np.pi / n) % (2 * np.pi)
                  for i in range(n)]
        angles_closed = angles + [angles[0]]

        ax.set_yticklabels([])
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        ax.spines['polar'].set_visible(False)

        # Polygon ring grid
        n_rings = 4
        r_ticks = np.linspace(0, r_max, n_rings + 1)[1:]
        for r in r_ticks:
            pts = angles + [angles[0]]
            ax.plot(pts, [r] * (n + 1),
                    color='#CCCCCC', lw=0.5, linestyle='-', zorder=0)
        for a in angles:
            ax.plot([a, a], [0, r_max], color='#CCCCCC', lw=0.4, zorder=0)

        # Data polygon
        vals = list(w_vals) + [w_vals[0]]
        ax.plot(angles_closed, vals, 'o-', lw=1.5, color=color,
                markersize=2.5, alpha=0.9)
        ax.fill(angles_closed, vals, alpha=0.15, color=color)

        ax.set_xticks(angles)
        ax.set_xticklabels(
            [self._truncate(lb, 6) for lb in labels], fontsize=5.0)
        ax.set_rlim(0, r_max)
        ax.tick_params(axis='y', labelsize=0)
        if title:
            ax.set_title(title, fontsize=9, fontweight='bold', pad=4)

    def _plot_weight_radar_multiyear(
        self,
        weight_all_years: Dict[int, Any],
        component_names: List[str],
        save_name: str,
    ) -> Optional[str]:
        """14-panel radar grid — one panel per year, shared radial scale."""
        years = sorted(weight_all_years.keys())
        n_years = len(years)
        ncols = 7
        nrows = (n_years + ncols - 1) // ncols

        # Build per-year weight arrays aligned to component_names
        year_weights: Dict[int, np.ndarray] = {}
        for yr in years:
            gw = weight_all_years[yr].get('global_sc_weights', {})
            year_weights[yr] = np.array([gw.get(sc, 0.0) for sc in component_names])

        # Shared r_max
        r_max = float(np.max(np.concatenate(list(year_weights.values())))) * 1.12
        color = _METHOD_COLORS['CRITIC']

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 3.6, nrows * 4.2),
            subplot_kw={'projection': 'polar'},
        )
        axes = np.array(axes).reshape(nrows, ncols)

        for idx, yr in enumerate(years):
            row, col = divmod(idx, ncols)
            self._draw_radar_to_ax(
                axes[row, col], year_weights[yr], component_names,
                r_max, color=color, title=str(yr),
            )

        # Hide unused subplots
        for idx in range(n_years, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        fig.suptitle(
            'CRITIC Sub-Criterion Weight Radars — 2011–2024',
            fontsize=14, fontweight='bold', y=1.01,
        )
        fig.tight_layout()
        return self._save(fig, save_name)

    def _plot_weight_radar_multiyear_4x4(
        self,
        weight_all_years: Dict[int, Any],
        component_names: List[str],
        save_name: str,
    ) -> Optional[str]:
        """
        14-panel radar grid (4 rows x 4 columns) as requested.
        Rows 1-3 have 4 panels; Row 4 has 2 panels centered horizontally.
        """
        import matplotlib.gridspec as gridspec

        years = sorted(weight_all_years.keys())
        n_years = len(years)

        # Build per-year weight arrays aligned to component_names
        year_weights: Dict[int, np.ndarray] = {}
        for yr in years:
            gw = weight_all_years[yr].get('global_sc_weights', {})
            year_weights[yr] = np.array([gw.get(sc, 0.0) for sc in component_names])

        # Shared r_max across all panels
        all_vals = np.concatenate(list(year_weights.values()))
        r_max = float(np.max(all_vals)) * 1.12
        color = _METHOD_COLORS['CRITIC']

        fig = plt.figure(figsize=(16, 18))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.25)

        for idx, yr in enumerate(years):
            if idx < 12:
                # Top 3 rows: 4 per row
                row, col = divmod(idx, 4)
            else:
                # Bottom row: index 12 -> col 1, index 13 -> col 2 (centered in row of 4)
                row = 3
                col = (idx - 12) + 1

            ax = fig.add_subplot(gs[row, col], projection='polar')
            self._draw_radar_to_ax(
                ax, year_weights[yr], component_names,
                r_max, color=color, title=str(yr),
            )

        fig.suptitle(
            'CRITIC Sub-Criterion Weights Comparison — 2011–2024 (4x4 Grid)',
            fontsize=16, fontweight='bold', y=0.96,
        )
        # Manual adjustment to avoid tight_layout warnings with custom GridSpec
        fig.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95)

        # Save as a separate file to distinguish from the 2x7 version
        base_name = save_name.rsplit('.', 1)[0]
        grouped_save_name = f"{base_name}_grouped_4x4.png"
        return self._save(fig, grouped_save_name)

    # ==================================================================
    #  FIG 05 – Annotated Weight Heatmap
    # ==================================================================

    def plot_weight_heatmap(
        self,
        weights: Dict[str, np.ndarray],
        component_names: List[str],
        weight_all_years: Optional[Dict[int, Any]] = None,
        save_name: str = 'fig05_weight_heatmap.png',
    ) -> Optional[str]:
        """
        Render an annotated heatmap of sub-criteria weights.

        Parameters
        ----------
        weights : Dict[str, np.ndarray]
            Weights mapped by method name.
        component_names : List[str]
            Sub-criteria names.
        weight_all_years : Dict[int, Any], optional
            Historical weights for Year x SC heatmap.
        save_name : str, default='fig05_weight_heatmap.png'
            The output filename.

        Returns
        -------
        str, optional
            The absolute path to the saved figure, or None if failed.
        """
        if not HAS_MATPLOTLIB:
            return None

        if weight_all_years:
            return self._plot_weight_heatmap_multiyear(
                weight_all_years, component_names, save_name)

        # ── Single-year fallback ─────────────────────────────────────────
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
                heat_ax.text(mid, -1.10, f'C0{prev_c}',
                             ha='center', va='top', fontsize=7.5,
                             color='#333333', fontstyle='italic')
                first_j_of_group = j
            prev_c = c

        # Label the last group
        if prev_c is not None:
            mid = (first_j_of_group + n_c - 1) / 2
            heat_ax.text(mid, -1.10, f'C0{prev_c}',
                         ha='center', va='top', fontsize=7.5,
                         color='#333333', fontstyle='italic')

        heat_ax.set_title(
            'Weight Heatmap — Method × Sub-Criterion (sequential order)',
            pad=30, fontsize=13, fontweight='bold',
        )

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                                  orientation='vertical')
        cbar_ax.set_ylabel('Weight', fontsize=10)

        fig.tight_layout()
        return self._save(fig, save_name)

    def _plot_weight_heatmap_multiyear(
        self,
        weight_all_years: Dict[int, Any],
        component_names: List[str],
        save_name: str,
    ) -> Optional[str]:
        """Years × sub-criteria annotated heatmap covering all 14 years."""
        import matplotlib as mpl

        years = sorted(weight_all_years.keys())
        n_years = len(years)
        n_c = len(component_names)

        def _sc_sort_key(name: str) -> tuple:
            try:
                return (int(name[2]), int(name[3:]))
            except Exception:
                return (99, 99)

        col_order = np.array(
            sorted(range(n_c), key=lambda i: _sc_sort_key(component_names[i])))
        names_ord = [component_names[i] for i in col_order]

        # Build data matrix: rows = years, cols = sorted SCs
        data = np.zeros((n_years, n_c))
        for yr_idx, yr in enumerate(years):
            gw = weight_all_years[yr].get('global_sc_weights', {})
            for sc_idx, sc in enumerate(names_ord):
                data[yr_idx, sc_idx] = gw.get(sc, 0.0)

        fig, (heat_ax, cbar_ax) = plt.subplots(
            1, 2,
            figsize=(max(18, n_c * 0.65), max(7, n_years * 0.58)),
            gridspec_kw={'width_ratios': [n_c, 1]},
        )

        cmap = plt.get_cmap('YlOrRd')
        vmin, vmax = data.min(), data.max()
        heat_ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

        heat_ax.set_xticks(range(n_c))
        heat_ax.set_xticklabels(
            [self._truncate(s, 10) for s in names_ord],
            rotation=55, ha='right', fontsize=8,
        )
        heat_ax.set_yticks(range(n_years))
        heat_ax.set_yticklabels([str(yr) for yr in years], fontsize=9)

        # Annotate cells
        threshold = vmin + 0.65 * (vmax - vmin)
        for i in range(n_years):
            for j in range(n_c):
                val = data[i, j]
                txt_col = 'white' if val > threshold else 'black'
                heat_ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                             fontsize=5.5, color=txt_col, fontweight='bold')

        # Criterion-group dashed separators and labels
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
                mid = (first_j_of_group + j - 1) / 2
                heat_ax.text(mid, -1.20, f'C0{prev_c}',
                             ha='center', va='top', fontsize=8,
                             color='#333333', fontstyle='italic')
                first_j_of_group = j
            prev_c = c
        if prev_c is not None:
            mid = (first_j_of_group + n_c - 1) / 2
            heat_ax.text(mid, -1.20, f'C0{prev_c}',
                         ha='center', va='top', fontsize=8,
                         color='#333333', fontstyle='italic')

        heat_ax.set_title(
            'CRITIC Sub-Criterion Weights — Year × Sub-Criterion (2011–2024)',
            pad=35, fontsize=13, fontweight='bold',
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
          (summing to 1 within each group).
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

        # fig04a – criteria-level radar (CRITIC)
        saved.append(self._draw_radar(
            crit_w, crit_labels, methods,
            title='Criterion-Level Weight Radar — CRITIC',
            save_name=f'{save_prefix}a_weight_radar_criteria.png',
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
                       '— CRITIC'),
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
