# -*- coding: utf-8 -*-
"""
Publication-Quality Visualization Suite for ML-MCDM Analysis
=============================================================

Produces a comprehensive set of 25+ sophisticated, high-resolution figures
covering every pipeline phase: weighting, MCDM ranking, ER aggregation,
sensitivity analysis, ML forecasting, and validation diagnostics.

Design principles:
  - Publication-quality (300 DPI, vector-ready layouts)
  - Rich colour palettes with perceptual uniformity
  - Statistical annotations on every chart
  - Dual-encoded information (colour + position + size)
  - Consistent professional styling across all figures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as mticker
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch, Polygon, Circle
    from matplotlib.collections import PatchCollection
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    from scipy import stats as sp_stats
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =========================================================================
# Style constants
# =========================================================================

_PALETTE = {
    'deep_blue':   '#1B2838',
    'royal_blue':  '#2E86AB',
    'teal':        '#0E7C7B',
    'emerald':     '#17B169',
    'gold':        '#F4A100',
    'amber':       '#F18F01',
    'coral':       '#E5625E',
    'crimson':     '#C73E1D',
    'magenta':     '#A23B72',
    'lavender':    '#7B68EE',
    'slate':       '#626D71',
    'light_gray':  '#F0F0F0',
    'medium_gray': '#C0C0C0',
    'white':       '#FFFFFF',
}

_CATEGORICAL_COLORS = [
    '#2E86AB', '#A23B72', '#F18F01', '#17B169', '#C73E1D',
    '#7B68EE', '#0E7C7B', '#F4A100', '#E5625E', '#626D71',
    '#1B9AAA', '#D81159', '#8F2D56', '#218380', '#FBB13C',
]

_GRADIENT_CMAPS = {
    'ranking':     'RdYlGn',
    'weights':     'YlOrRd',
    'correlation': 'RdBu_r',
    'sequential':  'viridis',
    'diverging':   'coolwarm',
    'heat':        'magma',
}


def _apply_style() -> None:
    """Apply a consistent publication style to all figures."""
    if not HAS_MATPLOTLIB:
        return
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
        'font.size': 10,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.labelweight': 'bold',
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.grid.which': 'major',
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#CCCCCC',
        'figure.titlesize': 15,
        'figure.titleweight': 'bold',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
    })


# =========================================================================
# Visualizer
# =========================================================================

class PanelVisualizer:
    """
    Publication-quality visualization suite for ML-MCDM panel-data analysis.

    Generates 25+ individual high-resolution PNG figures, each focused on
    a single analytical dimension.  Figures are numbered for easy reference
    in the companion report document.
    """

    def __init__(self,
                 output_dir: str = 'outputs/figures',
                 dpi: int = 300,
                 figsize: Tuple[int, int] = (14, 9),
                 style: str = 'seaborn-v0_8-whitegrid'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        self.generated_figures: List[str] = []

        if HAS_MATPLOTLIB:
            _apply_style()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save(self, fig, name: str) -> Optional[str]:
        try:
            path = self.output_dir / name
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none', format='png')
            plt.close(fig)
            self.generated_figures.append(str(path))
            return str(path)
        except Exception:
            try:
                path = self.output_dir / name
                fig.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                self.generated_figures.append(str(path))
                return str(path)
            except Exception:
                plt.close(fig)
                return None

    # Legacy alias
    def _save_figure(self, fig, name: str) -> Optional[str]:
        return self._save(fig, name)

    @staticmethod
    def _truncate(label: str, n: int = 18) -> str:
        return label if len(label) <= n else label[:n-1] + '…'

    def get_generated_figures(self) -> List[str]:
        return list(self.generated_figures)

    # ==================================================================
    #  FIG 01 – Final ER Ranking  (horizontal lollipop + gradient fill)
    # ==================================================================

    def plot_final_ranking(self,
                           provinces: List[str],
                           scores: np.ndarray,
                           ranks: np.ndarray,
                           title: str = 'Hierarchical ER Final Ranking',
                           save_name: str = 'fig01_final_er_ranking.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None
        scores = np.asarray(scores); ranks = np.asarray(ranks)
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
        ax.set_yticklabels([f'{int(ranks[idx]):>2d}. {provinces[idx]}' for idx in order],
                           fontsize=9)
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
    def plot_final_ranking_summary(self, provinces, scores, ranks, **kwargs):
        return self.plot_final_ranking(provinces, scores, ranks, **kwargs)

    # ==================================================================
    #  FIG 02 – ER Score Distribution  (histogram + KDE + rug)
    # ==================================================================

    def plot_score_distribution(self,
                                scores: np.ndarray,
                                title: str = 'Distribution of ER Scores',
                                save_name: str = 'fig02_score_distribution.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None
        scores = np.asarray(scores)

        fig, ax = plt.subplots(figsize=self.figsize)

        n_bins = min(30, max(10, len(scores) // 3))
        n_vals, bins, patches = ax.hist(scores, bins=n_bins, density=True,
                                        alpha=0.75, edgecolor='white', linewidth=0.6)
        cm = plt.colormaps['viridis']
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        rng = np.ptp(bin_centers) if np.ptp(bin_centers) > 0 else 1
        col = (bin_centers - bin_centers.min()) / rng
        for c, p in zip(col, patches):
            p.set_facecolor(cm(c))

        # KDE overlay
        if HAS_SCIPY and len(scores) > 5:
            kde = sp_stats.gaussian_kde(scores)
            x_grid = np.linspace(scores.min() - scores.std(), scores.max() + scores.std(), 300)
            ax.plot(x_grid, kde(x_grid), color=_PALETTE['crimson'], lw=2.5,
                    label='KDE', zorder=4)

        # Rug plot
        ax.plot(scores, np.zeros_like(scores) - 0.02 * max(n_vals.max(), 1),
                '|', color='black', ms=8, alpha=0.5, zorder=5)

        mean, med, std = scores.mean(), np.median(scores), scores.std()
        ax.axvline(mean, ls='--', color=_PALETTE['coral'], lw=1.8,
                   label=f'Mean = {mean:.4f}')
        ax.axvline(med, ls='-.', color=_PALETTE['royal_blue'], lw=1.8,
                   label=f'Median = {med:.4f}')

        # Normality test
        norm_text = ''
        if HAS_SCIPY and 3 <= len(scores) <= 5000:
            sw_stat, sw_p = sp_stats.shapiro(scores)
            norm_text = f'Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f}'

        skew_v = float(sp_stats.skew(scores)) if HAS_SCIPY else 0
        kurt_v = float(sp_stats.kurtosis(scores)) if HAS_SCIPY else 0
        stats_box = (f'N = {len(scores)}\n'
                     f'Mean = {mean:.4f}\n'
                     f'Median = {med:.4f}\n'
                     f'Std = {std:.4f}\n'
                     f'Skew = {skew_v:.3f}\n'
                     f'Kurt = {kurt_v:.3f}\n'
                     + norm_text)
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
    #  FIG 03 – Weight Comparison  (grouped bar + dot overlay)
    # ==================================================================

    def plot_weights_comparison(self,
                                weights: Dict[str, np.ndarray],
                                component_names: List[str],
                                title: str = 'Subcriteria Weight Comparison',
                                save_name: str = 'fig03_weights_comparison.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        methods = list(weights.keys())
        n_m = len(methods)
        n_c = len(component_names)
        x = np.arange(n_c)
        width = 0.78 / n_m

        fig, ax = plt.subplots(figsize=(max(14, n_c * 0.65), 9))
        colors = _CATEGORICAL_COLORS[:n_m]

        for i, (method, w) in enumerate(weights.items()):
            offset = (i - n_m / 2 + 0.5) * width
            bars = ax.bar(x + offset, w, width, label=method,
                          color=colors[i], edgecolor='white', linewidth=0.4)
            ax.scatter(x + offset, w, s=18, color='black', zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([self._truncate(c, 12) for c in component_names],
                           rotation=55, ha='right', fontsize=8)
        ax.set_ylabel('Weight')
        ax.set_title(title, pad=12)
        ax.legend(loc='upper right', ncol=min(n_m, 3), fontsize=9)
        ax.set_xlim(-0.6, n_c - 0.4)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 04 – Weight Radar Chart (spider diagram)
    # ==================================================================

    def plot_weight_radar(self,
                          weights: Dict[str, np.ndarray],
                          component_names: List[str],
                          save_name: str = 'fig04_weight_radar.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        n = len(component_names)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(11, 11), subplot_kw={'projection': 'polar'})
        colors = _CATEGORICAL_COLORS

        for i, (method, w) in enumerate(weights.items()):
            vals = list(w) + [w[0]]
            ax.plot(angles, vals, 'o-', lw=2, label=method,
                    color=colors[i % len(colors)], markersize=5)
            ax.fill(angles, vals, alpha=0.08, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([self._truncate(c, 10) for c in component_names], fontsize=8)
        ax.set_title('Weight Profiles — Radar Comparison', y=1.08, fontsize=14)
        ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.08), fontsize=9)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 05 – Weight Heatmap (method × subcriteria)
    # ==================================================================

    def plot_weight_heatmap(self,
                            weights: Dict[str, np.ndarray],
                            component_names: List[str],
                            save_name: str = 'fig05_weight_heatmap.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        methods = list(weights.keys())
        data = np.array([weights[m] for m in methods])

        fig, ax = plt.subplots(figsize=(max(12, len(component_names) * 0.55), 5))
        im = ax.imshow(data, aspect='auto', cmap='YlOrRd')

        ax.set_xticks(range(len(component_names)))
        ax.set_xticklabels([self._truncate(c, 10) for c in component_names],
                           rotation=55, ha='right', fontsize=8)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=10)

        for i in range(len(methods)):
            for j in range(len(component_names)):
                val = data[i, j]
                txt_col = 'white' if val > data.mean() + 0.5 * data.std() else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=7, color=txt_col, fontweight='bold')

        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label('Weight', fontsize=10)
        ax.set_title('Weight Values Heatmap (Method × Subcriteria)', pad=12)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 06 – MCDM Method Agreement Matrix (Spearman heatmap)
    # ==================================================================

    def plot_method_agreement_matrix(self,
                                     rankings_dict: Dict[str, np.ndarray],
                                     title: str = 'MCDM Method Rank Agreement',
                                     save_name: str = 'fig06_method_agreement.png') -> Optional[str]:
        if not HAS_MATPLOTLIB or not HAS_SCIPY:
            return None

        methods = list(rankings_dict.keys())
        n = len(methods)
        corr = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                r, _ = sp_stats.spearmanr(
                    np.asarray(rankings_dict[methods[i]]),
                    np.asarray(rankings_dict[methods[j]]))
                corr[i, j] = corr[j, i] = r

        fig, ax = plt.subplots(figsize=(max(8, n * 0.75 + 2), max(7, n * 0.65 + 1)))
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        labels = [self._truncate(m, 14) for m in methods]
        ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        for i in range(n):
            for j in range(n):
                txt_col = 'white' if abs(corr[i, j]) > 0.65 else 'black'
                ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                        fontsize=max(5, min(8, 120 // n)), color=txt_col, fontweight='bold')

        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label('Spearman ρ', fontsize=10)
        ax.set_title(title, pad=12)

        avg_corr = (corr.sum() - n) / max(n * (n - 1), 1)
        ax.text(0.5, -0.08, f'Average pairwise ρ = {avg_corr:.4f}',
                transform=ax.transAxes, ha='center', fontsize=10, style='italic')
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 07 – MCDM Rank Comparison Parallel Coordinates
    # ==================================================================

    def plot_rank_parallel_coordinates(self,
                                       rankings_dict: Dict[str, np.ndarray],
                                       entity_names: List[str],
                                       top_n: int = 25,
                                       save_name: str = 'fig07_rank_parallel.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        methods = list(rankings_dict.keys())
        n_m = len(methods)
        first_ranks = np.asarray(rankings_dict[methods[0]])
        top_idx = np.argsort(first_ranks)[:top_n]

        fig, ax = plt.subplots(figsize=(max(12, n_m * 1.5), 10))
        cmap = plt.colormaps['viridis']
        norm = plt.Normalize(0, top_n - 1)

        for pos, idx in enumerate(top_idx):
            y_vals = [np.asarray(rankings_dict[m])[idx] for m in methods]
            colour = cmap(norm(pos))
            ax.plot(range(n_m), y_vals, '-o', color=colour, alpha=0.75,
                    lw=1.8, ms=7, markeredgecolor='white', markeredgewidth=0.8,
                    label=entity_names[idx] if pos < 10 else '')

        ax.set_xticks(range(n_m))
        ax.set_xticklabels([self._truncate(m, 14) for m in methods],
                           rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Rank (1 = best)')
        ax.invert_yaxis()
        ax.set_title(f'Rank Trajectories Across {n_m} MCDM Methods (Top {top_n})', pad=12)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8,
                  title='Province', title_fontsize=9)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 08 – Per-Criterion Method Scores
    # ==================================================================

    def plot_criterion_scores(self,
                              scores: Dict[str, pd.Series],
                              criterion_name: str,
                              top_n: int = 20,
                              save_name: str = 'fig08_criterion_scores.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        n_methods = len(scores)
        fig, axes = plt.subplots(1, min(n_methods, 6),
                                 figsize=(4.5 * min(n_methods, 6), max(8, top_n * 0.32)),
                                 sharey=True)
        if min(n_methods, 6) == 1:
            axes = [axes]

        for idx, (method, series) in enumerate(list(scores.items())[:6]):
            ax = axes[idx]
            sorted_s = series.sort_values(ascending=False).head(top_n)
            cmap_ = plt.colormaps['viridis']
            vmin, vmax = sorted_s.min(), sorted_s.max()
            norm_ = plt.Normalize(vmin, vmax) if vmax > vmin else plt.Normalize(0, 1)
            bar_colors = [cmap_(norm_(v)) for v in sorted_s.values]

            ax.barh(range(len(sorted_s)), sorted_s.values, color=bar_colors,
                    edgecolor='white', linewidth=0.3)
            ax.set_yticks(range(len(sorted_s)))
            ax.set_yticklabels([self._truncate(str(n), 15) for n in sorted_s.index],
                               fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel('Score', fontsize=9)
            ax.set_title(f'{method}', fontsize=11, fontweight='bold')

        fig.suptitle(f'{criterion_name} — Per-Method Scores (Top {top_n})',
                     fontsize=14, y=1.02)
        plt.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 09 – Criteria Sensitivity Tornado
    # ==================================================================

    def plot_sensitivity_tornado(self,
                                 sensitivity: Dict[str, float],
                                 title: str = 'Criteria Sensitivity Analysis',
                                 save_name: str = 'fig09_criteria_sensitivity.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        items = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
        names = [k for k, _ in items]
        vals  = [v for _, v in items]
        n = len(names)

        fig, ax = plt.subplots(figsize=(12, max(6, n * 0.42)))

        vmin, vmax = min(vals), max(vals)
        norm = plt.Normalize(vmin, vmax) if vmax > vmin else plt.Normalize(0, 1)
        cmap = plt.colormaps['RdYlGn_r']
        colors = [cmap(norm(v)) for v in vals]

        bars = ax.barh(range(n), vals, color=colors, edgecolor='black', linewidth=0.5,
                       height=0.7, zorder=2)
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
    def plot_sensitivity_analysis(self, sensitivity, **kwargs):
        return self.plot_sensitivity_tornado(sensitivity, **kwargs)

    # ==================================================================
    #  FIG 10 – Subcriteria Sensitivity (Top 20)
    # ==================================================================

    def plot_subcriteria_sensitivity(self,
                                     sensitivity: Dict[str, float],
                                     top_n: int = 20,
                                     save_name: str = 'fig10_subcriteria_sensitivity.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        items = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names = [k for k, _ in items]
        vals  = [v for _, v in items]

        fig, ax = plt.subplots(figsize=(12, max(7, len(names) * 0.38)))
        colors = plt.colormaps['magma'](np.linspace(0.25, 0.85, len(names)))

        ax.barh(range(len(names)), vals, color=colors, edgecolor='white', linewidth=0.4,
                height=0.65)
        for i, v in enumerate(vals):
            ax.text(v + 0.003, i, f'{v:.4f}', va='center', fontsize=8)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([self._truncate(n, 22) for n in names], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Sensitivity Index')
        ax.set_title(f'Subcriteria Weight Sensitivity (Top {top_n})', pad=12)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 11 – Top-N Stability Bar
    # ==================================================================

    def plot_top_n_stability(self,
                              stability: Dict[int, float],
                              save_name: str = 'fig11_top_n_stability.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        ns = sorted(stability.keys())
        vals = [stability[n] for n in ns]

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = [_PALETTE['emerald'] if v >= 0.8 else
                  _PALETTE['gold'] if v >= 0.5 else _PALETTE['crimson']
                  for v in vals]

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

    def plot_temporal_stability(self,
                                temporal: Dict[str, float],
                                save_name: str = 'fig12_temporal_stability.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        pairs = sorted(temporal.keys())
        vals = [temporal[p] for p in pairs]

        fig, ax = plt.subplots(figsize=(max(10, len(pairs) * 0.8), 7))
        colors = [_PALETTE['emerald'] if v >= 0.9 else
                  _PALETTE['gold'] if v >= 0.7 else _PALETTE['crimson']
                  for v in vals]

        ax.bar(range(len(pairs)), vals, color=colors, edgecolor='black',
               linewidth=0.5, width=0.6, zorder=2)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

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

    def plot_rank_volatility(self,
                              rank_stability: Dict[str, float],
                              save_name: str = 'fig13_rank_volatility.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        items = sorted(rank_stability.items(), key=lambda x: x[1])
        names = [k for k, _ in items]
        vals  = [v for _, v in items]
        n = len(names)

        fig, ax = plt.subplots(figsize=(13, max(10, n * 0.25)))
        vmin, vmax = min(vals), max(vals)
        norm = plt.Normalize(vmin, vmax) if vmax > vmin else plt.Normalize(0, 1)
        cmap = plt.colormaps['RdYlGn']
        colors = [cmap(norm(v)) for v in vals]

        ax.barh(range(n), vals, color=colors, edgecolor='white', linewidth=0.3,
                height=0.7)
        ax.set_yticks(range(n))
        ax.set_yticklabels([self._truncate(nm, 20) for nm in names], fontsize=8)
        ax.set_xlabel('Rank Stability Score')
        ax.set_title('Province Rank Stability (higher = more stable)', pad=12)
        ax.axvline(np.mean(vals), ls='--', color='gray', lw=1.2,
                   label=f'Mean = {np.mean(vals):.3f}')
        ax.legend(fontsize=9)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 14 – IFS Sensitivity (comparative bar)
    # ==================================================================

    def plot_ifs_sensitivity(self,
                              mu_sens: float,
                              nu_sens: float,
                              save_name: str = 'fig14_ifs_sensitivity.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        fig, ax = plt.subplots(figsize=(10, 7))

        categories = ['Membership (μ)', 'Non-Membership (ν)']
        values = [mu_sens, nu_sens]
        colors = [_PALETTE['royal_blue'], _PALETTE['magenta']]

        bars = ax.bar(categories, values, color=colors, edgecolor='black',
                      linewidth=0.8, width=0.45, zorder=2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f'{v:.4f}', ha='center', fontsize=14, fontweight='bold')

        ax.set_ylabel('Sensitivity Index')
        ax.set_title('IFS Uncertainty Sensitivity Analysis', pad=12)
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)

        interpretation = 'Low sensitivity — robust' if max(values) < 0.2 else \
                         'Moderate sensitivity' if max(values) < 0.5 else 'High sensitivity — caution'
        ax.text(0.97, 0.97, interpretation, transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec='#CCCCCC'))
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 15 – ER Uncertainty Distribution
    # ==================================================================

    def plot_er_uncertainty(self,
                            uncertainty_df: pd.DataFrame,
                            provinces: List[str],
                            top_n: int = 30,
                            save_name: str = 'fig15_er_uncertainty.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        if uncertainty_df.shape[1] < 2:
            return None

        fig, ax = plt.subplots(figsize=(14, max(8, min(top_n, len(uncertainty_df)) * 0.35)))

        n_show = min(top_n, len(uncertainty_df))
        data = uncertainty_df.iloc[:n_show].values

        bp = ax.boxplot(data.T, vert=False, patch_artist=True,
                        boxprops=dict(facecolor=_PALETTE['royal_blue'], alpha=0.6),
                        medianprops=dict(color='black', lw=2),
                        whiskerprops=dict(color='gray'),
                        flierprops=dict(marker='o', ms=4, alpha=0.4))

        ax.set_yticks(range(1, n_show + 1))
        labels = (uncertainty_df.index[:n_show].tolist()
                  if hasattr(uncertainty_df.index, 'tolist')
                  else provinces[:n_show])
        ax.set_yticklabels([self._truncate(str(l), 18) for l in labels], fontsize=9)
        ax.set_xlabel('Belief Degree')
        ax.set_title('ER Aggregation Uncertainty Distribution', pad=12)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 16 – Forecast Actual vs Predicted
    # ==================================================================

    def plot_forecast_scatter(self,
                               actual: np.ndarray,
                               predicted: np.ndarray,
                               lower: Optional[np.ndarray] = None,
                               upper: Optional[np.ndarray] = None,
                               entity_names: Optional[List[str]] = None,
                               save_name: str = 'fig16_forecast_scatter.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None
        actual = np.asarray(actual); predicted = np.asarray(predicted)

        fig, ax = plt.subplots(figsize=(12, 11))

        residuals = np.abs(actual - predicted)
        scatter = ax.scatter(actual, predicted, c=residuals, cmap='RdYlGn_r',
                             s=100, alpha=0.8, edgecolors='black', linewidths=0.5,
                             zorder=4)

        lo = min(actual.min(), predicted.min())
        hi = max(actual.max(), predicted.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                'k--', lw=2, label='Perfect', alpha=0.7, zorder=2)

        z = np.polyfit(actual, predicted, 1)
        p = np.poly1d(z)
        xs = np.linspace(lo - margin, hi + margin, 200)
        ax.plot(xs, p(xs), color=_PALETTE['royal_blue'], lw=2,
                label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}', zorder=3)

        ci = 1.96 * np.std(actual - predicted)
        ax.fill_between(xs, p(xs) - ci, p(xs) + ci, alpha=0.12,
                        color=_PALETTE['royal_blue'], label='95% CI')

        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        stats = (f'R² = {r2:.6f}\n'
                 f'MAE = {mae:.6f}\n'
                 f'RMSE = {rmse:.6f}\n'
                 f'N = {len(actual)}')
        ax.text(0.03, 0.97, stats, transform=ax.transAxes, fontsize=10,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#CCCCCC', alpha=0.95))

        worst = np.argsort(residuals)[-5:]
        for wi in worst:
            lbl = entity_names[wi] if entity_names else f'#{wi}'
            ax.annotate(lbl, (actual[wi], predicted[wi]),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=8, alpha=0.85,
                        arrowprops=dict(arrowstyle='->', alpha=0.5))

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('|Residual|', fontsize=10)

        ax.set_xlabel('Actual Score')
        ax.set_ylabel('Predicted Score')
        ax.set_title('Forecast Performance — Actual vs Predicted', pad=12)
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 17 – Forecast Residual Diagnostics (4-panel)
    # ==================================================================

    def plot_forecast_residuals(self,
                                actual: np.ndarray,
                                predicted: np.ndarray,
                                save_name: str = 'fig17_forecast_residuals.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None
        actual = np.asarray(actual); predicted = np.asarray(predicted)
        residuals = actual - predicted

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1 – Residuals vs Predicted
        ax = axes[0, 0]
        colors = ['#17B169' if r >= 0 else '#E5625E' for r in residuals]
        ax.scatter(predicted, residuals, c=colors, s=70, alpha=0.7,
                   edgecolors='black', linewidths=0.4)
        ax.axhline(0, color='black', lw=1.5)
        std_r = np.std(residuals) if np.std(residuals) > 0 else 1
        for mult, style, col in [(1, '--', _PALETTE['gold']), (2, ':', _PALETTE['crimson'])]:
            ax.axhline(mult * std_r, ls=style, color=col, lw=1.2)
            ax.axhline(-mult * std_r, ls=style, color=col, lw=1.2)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Residual')
        ax.set_title('Residuals vs Predicted')

        # 2 – Residual histogram + KDE
        ax = axes[0, 1]
        ax.hist(residuals, bins=25, density=True, color=_PALETTE['royal_blue'],
                alpha=0.7, edgecolor='white')
        if HAS_SCIPY and len(residuals) > 5:
            xg = np.linspace(residuals.min() - std_r, residuals.max() + std_r, 200)
            kde = sp_stats.gaussian_kde(residuals)
            ax.plot(xg, kde(xg), color=_PALETTE['crimson'], lw=2, label='KDE')
            ax.plot(xg, sp_stats.norm.pdf(xg, residuals.mean(), std_r),
                    '--', color='gray', lw=1.5, label='Normal')
        ax.set_xlabel('Residual'); ax.set_ylabel('Density')
        ax.set_title('Residual Distribution')
        ax.legend(fontsize=9)

        # 3 – QQ plot
        ax = axes[1, 0]
        if HAS_SCIPY:
            (osm, osr), (slope, intercept, _) = sp_stats.probplot(residuals, dist='norm')
            ax.scatter(osm, osr, s=50, alpha=0.6, color=_PALETTE['royal_blue'],
                       edgecolors='white')
            ax.plot(osm, slope * np.array(osm) + intercept, 'r-', lw=2)
        ax.set_xlabel('Theoretical Quantiles'); ax.set_ylabel('Ordered Values')
        ax.set_title('Q-Q Plot (Normality Check)')

        # 4 – Largest errors
        ax = axes[1, 1]
        abs_e = np.abs(residuals)
        order = np.argsort(abs_e)[::-1][:15]
        ax.barh(range(len(order)), abs_e[order],
                color=plt.colormaps['Reds'](np.linspace(0.35, 0.9, len(order))),
                edgecolor='white')
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f'#{i}' for i in order], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('|Residual|')
        ax.set_title('Largest Prediction Errors')

        fig.suptitle('Forecast Residual Diagnostics', fontsize=14, y=1.01)
        plt.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 18 – Feature Importance (lollipop)
    # ==================================================================

    def plot_feature_importance(self,
                                importance: Dict[str, float],
                                top_n: int = 20,
                                title: str = 'Feature Importance — Top Features',
                                save_name: str = 'fig18_feature_importance.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names = [k for k, _ in items]
        vals  = [v for _, v in items]
        n = len(names)

        fig, ax = plt.subplots(figsize=(12, max(7, n * 0.38)))
        colors = plt.colormaps['cool'](np.linspace(0.2, 0.85, n))

        ax.hlines(range(n), 0, vals, color=colors, linewidth=2.5, zorder=2)
        ax.scatter(vals, range(n), s=100, color=colors, edgecolors='black',
                   linewidths=0.6, zorder=3)
        for i, v in enumerate(vals):
            ax.text(v + 0.003, i, f'{v:.4f}', va='center', fontsize=9)

        ax.set_yticks(range(n))
        ax.set_yticklabels([self._truncate(nm, 25) for nm in names], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(title, pad=12)
        return self._save(fig, save_name)

    # Backward-compatible alias
    def plot_feature_importance_single(self, importance_dict, **kwargs):
        return self.plot_feature_importance(importance_dict, **kwargs)

    # ==================================================================
    #  FIG 19 – Model Weights Donut
    # ==================================================================

    def plot_model_weights_donut(self,
                                  weights: Dict[str, float],
                                  save_name: str = 'fig19_model_weights.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        labels = list(weights.keys())
        sizes = list(weights.values())
        colors = _CATEGORICAL_COLORS[:len(labels)]

        fig, ax = plt.subplots(figsize=(10, 10))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=colors, pctdistance=0.8,
            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))

        for t in autotexts:
            t.set_fontsize(10); t.set_fontweight('bold')
        for t in texts:
            t.set_fontsize(10)

        ax.set_title('Super Learner — Base Model Weights', fontsize=14, pad=20)
        centre = plt.Circle((0, 0), 0.55, fc='white')
        ax.add_artist(centre)
        ax.text(0, 0, 'Super\nLearner', ha='center', va='center',
                fontsize=14, fontweight='bold', color='#333333')
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 20 – Model Performance Comparison
    # ==================================================================

    def plot_model_performance(self,
                                model_metrics: Dict[str, Dict[str, float]],
                                save_name: str = 'fig20_model_performance.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        models = list(model_metrics.keys())
        all_metrics = sorted({m for d in model_metrics.values() for m in d.keys()})
        n_models = len(models)
        n_metrics = len(all_metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 7))
        if n_metrics == 1:
            axes = [axes]

        colors = _CATEGORICAL_COLORS[:n_models]

        for mi, metric in enumerate(all_metrics):
            ax = axes[mi]
            vals = [model_metrics[m].get(metric, 0) for m in models]
            bars = ax.bar(range(n_models), vals, color=colors, edgecolor='black',
                          linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                        f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
            ax.set_xticks(range(n_models))
            ax.set_xticklabels([self._truncate(m, 12) for m in models],
                               rotation=45, ha='right', fontsize=9)
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

        fig.suptitle('Forecasting Model Performance Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 21 – CV Box Plots
    # ==================================================================

    def plot_cv_boxplots(self,
                          cv_scores: Dict[str, List[float]],
                          save_name: str = 'fig21_cv_boxplots.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        metrics = list(cv_scores.keys())
        n = len(metrics)

        fig, ax = plt.subplots(figsize=(max(8, n * 2.5), 7))
        data = [cv_scores[m] for m in metrics]
        colors = _CATEGORICAL_COLORS[:n]

        bp = ax.boxplot(data, patch_artist=True, notch=True,
                        boxprops=dict(linewidth=1.2),
                        medianprops=dict(color='black', lw=2),
                        whiskerprops=dict(color='gray'),
                        flierprops=dict(marker='D', ms=5, alpha=0.5))
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col); patch.set_alpha(0.65)

        for i, (m, sc) in enumerate(cv_scores.items()):
            jitter = np.random.default_rng(42).normal(i + 1, 0.06, len(sc))
            ax.scatter(jitter, sc, s=50, color='black', alpha=0.5, zorder=5)
            mean_v = np.mean(sc); std_v = np.std(sc)
            ylim = ax.get_ylim()
            ax.text(i + 1, max(sc) + 0.03 * (ylim[1] - ylim[0]),
                    f'{mean_v:.3f}±{std_v:.3f}', ha='center', fontsize=9, fontweight='bold')

        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylabel('Score')
        ax.set_title('Cross-Validation Score Distributions', pad=12)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 22 – Prediction Intervals
    # ==================================================================

    def plot_prediction_intervals(self,
                                   predictions: pd.DataFrame,
                                   lower: pd.DataFrame,
                                   upper: pd.DataFrame,
                                   top_n: int = 20,
                                   save_name: str = 'fig22_prediction_intervals.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        col = predictions.columns[0] if len(predictions.columns) > 0 else None
        if col is None:
            return None

        pred_vals = predictions[col].values
        sorted_idx = np.argsort(pred_vals)[::-1][:top_n]
        names = [predictions.index[i] for i in sorted_idx]

        pred_show = pred_vals[sorted_idx]
        low_show = lower[col].values[sorted_idx] if col in lower.columns else pred_show * 0.9
        high_show = upper[col].values[sorted_idx] if col in upper.columns else pred_show * 1.1

        fig, ax = plt.subplots(figsize=(13, max(8, top_n * 0.38)))

        y = np.arange(top_n)
        ax.barh(y, high_show - low_show, left=low_show, height=0.5,
                color=_PALETTE['royal_blue'], alpha=0.2, edgecolor='none', label='95% CI')
        ax.scatter(pred_show, y, s=80, color=_PALETTE['royal_blue'],
                   edgecolors='black', linewidths=0.6, zorder=4, label='Point Estimate')
        ax.hlines(y, low_show, high_show, color=_PALETTE['royal_blue'], lw=1.5, zorder=3)

        ax.set_yticks(y)
        ax.set_yticklabels([self._truncate(str(n), 18) for n in names], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Predicted Score')
        ax.set_title(f'Prediction with 95% Conformal Intervals — Top {top_n}', pad=12)
        ax.legend(loc='lower right', fontsize=9)
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 23 – Rank Change Bubble Chart
    # ==================================================================

    def plot_rank_change_bubble(self,
                                 provinces: List[str],
                                 current_scores: np.ndarray,
                                 predicted_scores: np.ndarray,
                                 prediction_year: int,
                                 save_name: str = 'fig23_rank_change_bubble.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None
        current_scores = np.asarray(current_scores)
        predicted_scores = np.asarray(predicted_scores)

        curr_rank = np.argsort(np.argsort(-current_scores)) + 1
        pred_rank = np.argsort(np.argsort(-predicted_scores)) + 1
        rank_change = curr_rank.astype(int) - pred_rank.astype(int)

        fig, ax = plt.subplots(figsize=(13, 11))

        sizes = (np.abs(rank_change) + 1) * 30
        scatter = ax.scatter(curr_rank, pred_rank, s=sizes, c=rank_change,
                             cmap='RdYlGn', alpha=0.8,
                             edgecolors='black', linewidths=0.5, zorder=4)

        n = len(provinces)
        ax.plot([1, n], [1, n], 'k--', lw=1.5, alpha=0.5, label='No Change')

        abs_change = np.abs(rank_change)
        top_movers = np.argsort(abs_change)[-8:]
        for idx in top_movers:
            ax.annotate(f'{provinces[idx]}\n({rank_change[idx]:+d})',
                        (curr_rank[idx], pred_rank[idx]),
                        xytext=(8, 8), textcoords='offset points', fontsize=8,
                        arrowprops=dict(arrowstyle='->', alpha=0.5))

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Rank Change (+improved)', fontsize=10)

        ax.set_xlabel('Current Rank (2024)')
        ax.set_ylabel(f'Predicted Rank ({prediction_year})')
        ax.set_title(f'Rank Change Analysis — Current vs Predicted ({prediction_year})', pad=12)
        ax.legend(fontsize=9)
        ax.invert_xaxis(); ax.invert_yaxis()
        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 24 – Executive Dashboard
    # ==================================================================

    def plot_executive_dashboard(self,
                                  results: Dict[str, Any],
                                  save_name: str = 'fig24_executive_dashboard.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        fig = plt.figure(figsize=(22, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

        fig.suptitle('ML-MCDM Analysis — Executive Dashboard',
                     fontsize=18, fontweight='bold', y=0.99)

        # KPI cards
        kpi_data = results.get('kpis', {})
        for i, (label, value) in enumerate(list(kpi_data.items())[:4]):
            ax = fig.add_subplot(gs[0, i])
            ax.axis('off')
            ax.add_patch(FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                         boxstyle='round,pad=0.05',
                         facecolor=_CATEGORICAL_COLORS[i], alpha=0.15,
                         edgecolor=_CATEGORICAL_COLORS[i], linewidth=2,
                         transform=ax.transAxes))
            ax.text(0.5, 0.65, str(value), ha='center', va='center',
                    fontsize=26, fontweight='bold', color=_CATEGORICAL_COLORS[i],
                    transform=ax.transAxes)
            ax.text(0.5, 0.25, label, ha='center', va='center',
                    fontsize=11, color='#333333', transform=ax.transAxes)

        # Top 10 bar
        ax = fig.add_subplot(gs[1, :2])
        if 'top_10' in results:
            items_10 = results['top_10'][:10]
            names, scs = zip(*items_10)
            cmap_arr = plt.colormaps['viridis'](np.linspace(0.85, 0.25, len(items_10)))
            ax.barh(range(len(items_10)), scs, color=cmap_arr, edgecolor='white', linewidth=0.5)
            ax.set_yticks(range(len(items_10)))
            ax.set_yticklabels([f'{i+1}. {n}' for i, n in enumerate(names)], fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel('ER Score')
            ax.set_title('Top 10 Provinces', fontsize=12)

        # Top 15 weights
        ax = fig.add_subplot(gs[1, 2:])
        if 'fused_weights' in results:
            w = results['fused_weights']
            names_w = results.get('subcriteria_names', [f'SC{i}' for i in range(len(w))])
            order = np.argsort(w)[::-1][:15]
            ax.barh(range(len(order)), w[order],
                    color=plt.colormaps['YlOrRd'](np.linspace(0.3, 0.9, len(order))),
                    edgecolor='white')
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([self._truncate(names_w[i], 14) for i in order], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Fused Weight')
            ax.set_title('Top 15 Subcriteria Weights', fontsize=12)

        # Robustness text
        ax = fig.add_subplot(gs[2, :2])
        ax.axis('off')
        if 'robustness_text' in results and results['robustness_text']:
            ax.text(0.05, 0.95, results['robustness_text'], transform=ax.transAxes,
                    fontsize=10, va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', fc='#F8F8F8', ec='#CCCCCC'))

        # Method agreement mini-heatmap
        ax = fig.add_subplot(gs[2, 2:])
        if 'agreement_matrix' in results:
            am = results['agreement_matrix']
            im = ax.imshow(am, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title('Method Agreement', fontsize=12)
            fig.colorbar(im, ax=ax, shrink=0.6)
        else:
            ax.axis('off')

        return self._save(fig, save_name)

    # ==================================================================
    #  FIG 25 – Robustness Summary Infographic
    # ==================================================================

    def plot_robustness_summary(self,
                                 overall_robustness: float,
                                 confidence_level: float,
                                 criteria_sens: Dict[str, float],
                                 top_n_stab: Dict[int, float],
                                 mu_sens: float,
                                 nu_sens: float,
                                 save_name: str = 'fig25_robustness_summary.png') -> Optional[str]:
        if not HAS_MATPLOTLIB:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 13))

        # Panel 1: Overall gauge
        ax = axes[0, 0]
        ax.axis('off')
        color = (_PALETTE['emerald'] if overall_robustness >= 0.8 else
                 _PALETTE['gold'] if overall_robustness >= 0.5 else _PALETTE['crimson'])
        ax.add_patch(FancyBboxPatch((0.1, 0.2), 0.8, 0.6,
                     boxstyle='round,pad=0.08', fc=color, alpha=0.15,
                     ec=color, lw=3, transform=ax.transAxes))
        ax.text(0.5, 0.6, f'{overall_robustness:.4f}', ha='center', va='center',
                fontsize=42, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.5, 0.35, 'Overall Robustness Score', ha='center', fontsize=14,
                transform=ax.transAxes)
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
            ax.barh(range(len(names_)), vals_, color=colors_, edgecolor='black', lw=0.5)
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
            bar_colors = [_PALETTE['emerald'] if v >= 0.8 else
                          _PALETTE['gold'] if v >= 0.5 else _PALETTE['crimson'] for v in vals_s]
            ax.bar(range(len(ns)), vals_s, color=bar_colors, edgecolor='black', lw=0.7)
            for i, v in enumerate(vals_s):
                ax.text(i, v + 0.015, f'{v:.1%}', ha='center', fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(ns)))
            ax.set_xticklabels([f'Top-{n}' for n in ns])
            ax.set_ylim(0, 1.15)
            ax.set_ylabel('Stability')
            ax.set_title('Ranking Stability', fontsize=12)
            ax.axhline(0.8, ls=':', color='gray', lw=1)

        # Panel 4: IFS sensitivity
        ax = axes[1, 1]
        bars = ax.bar(['Membership (μ)', 'Non-Member. (ν)'], [mu_sens, nu_sens],
                      color=[_PALETTE['royal_blue'], _PALETTE['magenta']],
                      edgecolor='black', lw=0.7, width=0.5)
        for bar, v in zip(bars, [mu_sens, nu_sens]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sensitivity')
        ax.set_title('IFS Uncertainty Sensitivity', fontsize=12)

        fig.suptitle('Robustness & Sensitivity Summary', fontsize=15, y=1.01)
        plt.tight_layout()
        return self._save(fig, save_name)

    # ==================================================================
    #  Convenience: generate_all (single entry point from pipeline)
    # ==================================================================

    def generate_all(self, pipeline_result: Any,
                      panel_data: Any,
                      weights: Dict[str, Any],
                      ranking_result: Any,
                      analysis_results: Dict[str, Any],
                      forecast_result: Any = None) -> int:
        """
        Generate every applicable figure. Returns count of figures produced.
        """
        count = 0

        def _inc(path):
            nonlocal count
            if path:
                count += 1

        provinces = panel_data.provinces
        scores = np.asarray(ranking_result.final_scores.values
                            if hasattr(ranking_result.final_scores, 'values')
                            else ranking_result.final_scores)
        ranks = np.asarray(ranking_result.final_ranking.values
                           if hasattr(ranking_result.final_ranking, 'values')
                           else ranking_result.final_ranking)
        subcriteria = weights['subcriteria']

        # Core ranking
        _inc(self.plot_final_ranking(provinces, scores, ranks))
        _inc(self.plot_score_distribution(scores))

        # Weights
        w_dict = {
            'Entropy': weights['entropy'],
            'CRITIC': weights['critic'],
            'MEREC': weights['merec'],
            'Std Dev': weights['std_dev'],
            'Fused': weights['fused'],
        }
        _inc(self.plot_weights_comparison(w_dict, subcriteria))
        _inc(self.plot_weight_radar(w_dict, subcriteria))
        _inc(self.plot_weight_heatmap(w_dict, subcriteria))

        # Method agreement
        all_method_ranks = {}
        for crit_id, method_ranks in ranking_result.criterion_method_ranks.items():
            for method, rank_series in method_ranks.items():
                col = f'{crit_id}_{method}'
                all_method_ranks[col] = (rank_series.values
                                          if hasattr(rank_series, 'values')
                                          else np.asarray(rank_series))
        if all_method_ranks:
            _inc(self.plot_method_agreement_matrix(all_method_ranks))
            _inc(self.plot_rank_parallel_coordinates(all_method_ranks, provinces))

        # Per-criterion scores
        for ci, (crit_id, method_scores) in enumerate(
                ranking_result.criterion_method_scores.items()):
            _inc(self.plot_criterion_scores(
                method_scores, crit_id, top_n=20,
                save_name=f'fig08_{crit_id}_scores.png'))

        # Sensitivity
        sens = analysis_results.get('sensitivity')
        if sens is not None:
            if hasattr(sens, 'criteria_sensitivity') and sens.criteria_sensitivity:
                _inc(self.plot_sensitivity_tornado(sens.criteria_sensitivity))
            if hasattr(sens, 'subcriteria_sensitivity') and sens.subcriteria_sensitivity:
                _inc(self.plot_subcriteria_sensitivity(sens.subcriteria_sensitivity))
            if hasattr(sens, 'top_n_stability') and sens.top_n_stability:
                _inc(self.plot_top_n_stability(sens.top_n_stability))
            if hasattr(sens, 'temporal_stability') and sens.temporal_stability:
                _inc(self.plot_temporal_stability(sens.temporal_stability))
            if hasattr(sens, 'rank_stability') and sens.rank_stability:
                _inc(self.plot_rank_volatility(sens.rank_stability))
            if hasattr(sens, 'ifs_membership_sensitivity'):
                _inc(self.plot_ifs_sensitivity(
                    sens.ifs_membership_sensitivity,
                    getattr(sens, 'ifs_nonmembership_sensitivity', 0)))
            if hasattr(sens, 'overall_robustness'):
                _inc(self.plot_robustness_summary(
                    sens.overall_robustness,
                    getattr(sens, 'confidence_level', 0.95),
                    getattr(sens, 'criteria_sensitivity', {}),
                    getattr(sens, 'top_n_stability', {}),
                    getattr(sens, 'ifs_membership_sensitivity', 0),
                    getattr(sens, 'ifs_nonmembership_sensitivity', 0),
                ))

        # ER uncertainty
        try:
            unc = ranking_result.er_result.uncertainty
            _inc(self.plot_er_uncertainty(unc, provinces))
        except Exception:
            pass

        # Forecast
        if forecast_result is not None:
            try:
                if hasattr(forecast_result, 'training_info'):
                    ti = forecast_result.training_info
                    actual = ti.get('y_test')
                    predicted = ti.get('y_pred')
                    if actual is not None and predicted is not None:
                        ent = ti.get('test_entities')
                        _inc(self.plot_forecast_scatter(
                            np.asarray(actual), np.asarray(predicted), entity_names=ent))
                        _inc(self.plot_forecast_residuals(
                            np.asarray(actual), np.asarray(predicted)))

                if hasattr(forecast_result, 'feature_importance'):
                    imp = forecast_result.feature_importance
                    if hasattr(imp, 'to_dict'):
                        imp_dict = (imp['Importance'].to_dict() if 'Importance' in imp.columns
                                    else imp.iloc[:, 0].to_dict())
                    else:
                        imp_dict = imp
                    _inc(self.plot_feature_importance(imp_dict))

                if hasattr(forecast_result, 'model_contributions') and forecast_result.model_contributions:
                    _inc(self.plot_model_weights_donut(forecast_result.model_contributions))

                if hasattr(forecast_result, 'model_performance') and forecast_result.model_performance:
                    _inc(self.plot_model_performance(forecast_result.model_performance))

                if hasattr(forecast_result, 'cross_validation_scores') and forecast_result.cross_validation_scores:
                    _inc(self.plot_cv_boxplots(forecast_result.cross_validation_scores))

                if (hasattr(forecast_result, 'prediction_intervals')
                        and forecast_result.prediction_intervals):
                    preds = forecast_result.predictions
                    intervals = forecast_result.prediction_intervals
                    lower = intervals.get('lower')
                    upper = intervals.get('upper')
                    if lower is not None and upper is not None:
                        _inc(self.plot_prediction_intervals(preds, lower, upper))

                if hasattr(forecast_result, 'predictions') and forecast_result.predictions is not None:
                    pred_df = forecast_result.predictions
                    if len(pred_df.columns) > 0:
                        pred_col = pred_df.columns[0]
                        pred_scores = pred_df[pred_col].values
                        _inc(self.plot_rank_change_bubble(
                            provinces, scores, pred_scores,
                            prediction_year=getattr(forecast_result, 'target_year',
                                                     max(panel_data.years) + 1),
                        ))
            except Exception:
                pass

        # Executive dashboard
        try:
            top10_idx = np.argsort(ranks)[:10]
            top10 = [(provinces[i], scores[i]) for i in top10_idx]

            kpis = {
                'Provinces': len(provinces),
                'Years': len(panel_data.years),
                'Subcriteria': panel_data.n_subcriteria,
                'MCDM Methods': len(ranking_result.methods_used),
            }

            rob_text = ''
            if sens and hasattr(sens, 'overall_robustness'):
                rob_text = (
                    f'Overall Robustness : {sens.overall_robustness:.4f}\n'
                    f'Confidence Level   : {getattr(sens, "confidence_level", 0.95):.0%}\n'
                    f'IFS mu Sensitivity : {getattr(sens, "ifs_membership_sensitivity", 0):.4f}\n'
                    f'IFS nu Sensitivity : {getattr(sens, "ifs_nonmembership_sensitivity", 0):.4f}\n'
                )

            _inc(self.plot_executive_dashboard({
                'kpis': kpis,
                'top_10': top10,
                'fused_weights': weights['fused'],
                'subcriteria_names': subcriteria,
                'robustness_text': rob_text,
            }))
        except Exception:
            pass

        return count


# =========================================================================
# Factory
# =========================================================================

def create_visualizer(output_dir: str = 'outputs/figures') -> PanelVisualizer:
    return PanelVisualizer(output_dir=output_dir)
