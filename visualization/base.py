# -*- coding: utf-8 -*-
"""
Visualization Shared Utilities
==============================

Constants (palettes, colormaps), styling helper, and ``BasePlotter``
base class shared by all phase-specific plotters.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

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
    sp_stats = None

# =========================================================================
# Color constants
# =========================================================================

PALETTE = {
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

CATEGORICAL_COLORS = [
    '#2E86AB', '#A23B72', '#F18F01', '#17B169', '#C73E1D',
    '#7B68EE', '#0E7C7B', '#F4A100', '#E5625E', '#626D71',
    '#1B9AAA', '#D81159', '#8F2D56', '#218380', '#FBB13C',
]

GRADIENT_CMAPS = {
    'ranking':     'RdYlGn',
    'weights':     'YlOrRd',
    'correlation': 'RdBu_r',
    'sequential':  'viridis',
    'diverging':   'coolwarm',
    'heat':        'magma',
}


def apply_style() -> None:
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
# BasePlotter
# =========================================================================

class BasePlotter:
    """
    Shared functionality for all plotter subclasses.

    Subclasses call ``self._save(fig, name)`` and ``self._truncate(label)``
    without duplicating logic.
    """

    def __init__(self,
                 output_dir: str = 'outputs/figures',
                 dpi: int = 300,
                 figsize: Tuple[int, int] = (14, 9)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        self.generated_figures: List[str] = []

        if HAS_MATPLOTLIB:
            apply_style()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save(self, fig, name: str) -> Optional[str]:
        """Save *fig* to *output_dir/name*, record it, close it."""
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

    @staticmethod
    def _truncate(label: str, n: int = 18) -> str:
        return label if len(label) <= n else label[:n - 1] + '…'

    def get_generated_figures(self) -> List[str]:
        return list(self.generated_figures)


__all__ = [
    'HAS_MATPLOTLIB', 'HAS_SCIPY',
    'PALETTE', 'CATEGORICAL_COLORS', 'GRADIENT_CMAPS',
    'apply_style', 'BasePlotter',
    'plt', 'np', 'sp_stats',
]
