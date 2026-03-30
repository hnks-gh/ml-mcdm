"""
Visualization Infrastructure and Shared Utilities.

This module provides the foundational components for all visualization 
tasks in the ML-MCDM pipeline. It defines a centralized design system 
(colors, fonts, styles) and the `BasePlotter` abstract class to ensure 
visual consistency across weighting, ranking, and forecasting plots.

Key Features
------------
- **Design Tokens**: Standardized color palettes (`PALETTE`) and colormaps 
  (`GRADIENT_CMAPS`) for categorical and sequential data.
- **Global Styling**: The `apply_style` function configures Matplotlib 
  rcParams for high-DPI, publication-ready outputs.
- **Automated Watermarking**: `BasePlotter` injects execution metadata 
  (timestamps, config hashes, git commits) into every saved figure to 
  guarantee auditability.
- **Robust Persistence**: Implements fail-safe figure saving with 
  automatic DPI fallback and error logging.
"""

from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

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

_logger = logging.getLogger(__name__)

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
    """
    Apply a consistent publication style to all figures.

    Configures global `matplotlib.rcParams` to ensure high-quality, 
    consistent font choice, grid visibility, and layout padding. 
    Does nothing if Matplotlib is not installed.
    """
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
    Abstract foundation for all phase-specific plotters.

    Provides core utilities for directory management, metadata watermarking, 
    and fail-safe figure persistence.

    Attributes
    ----------
    output_dir : Path
        The directory where generated figures are saved.
    dpi : int
        The resolution for saved PNG files.
    figsize : Tuple[int, int]
        Default dimensions for new figures.
    generated_figures : List[str]
        A running inventory of all files successfully saved to disk.
    """

    def __init__(self,
                 output_dir: str = 'output/result/figures',
                 dpi: int = 300,
                 figsize: Tuple[int, int] = (14, 9)):
        """
        Initialize the base plotter.

        Parameters
        ----------
        output_dir : str, default='output/result/figures'
            Target path for plot storage. Created if missing.
        dpi : int, default=300
            Resolution for high-quality export.
        figsize : Tuple[int, int], default=(14, 9)
            Default (width, height) in inches.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        self.generated_figures: List[str] = []
        self._metadata: Optional[Dict[str, str]] = None

        if HAS_MATPLOTLIB:
            apply_style()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_metadata(self, metadata: Dict[str, str]) -> None:
        """
        Set execution metadata for figure watermarking.

        Parameters
        ----------
        metadata : Dict[str, str]
            Dictionary containing 'config_hash', 'git_commit', and 
            other audit identifiers.
        """
        self._metadata = metadata

    def _save(self, fig, name: str) -> Optional[str]:
        """
        Save a figure to disk with automatic watermarking.

        Implements a two-tier saving strategy: first at target DPI, then 
        at 150 DPI if the first attempt fails (e.g., due to memory 
        constraints or complex SVG rendering).

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object to save.
        name : str
            The filename (including extension, e.g., 'rank_heatmap.png').

        Returns
        -------
        str, optional
            The absolute path to the saved file, or None if saving failed.
        """
        try:
            if self._metadata and hasattr(fig, 'text'):
                from datetime import datetime
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cfg_hash = self._metadata.get('config_hash', 'N/A')
                git_commit = self._metadata.get('git_commit', 'N/A')
                watermark = f"Timestamp: {ts} | Config: {cfg_hash} | Commit: {git_commit}"
                fig.text(0.99, 0.01, watermark, 
                         fontsize=6, color='gray', ha='right', va='bottom', alpha=0.6)
                         
            path = self.output_dir / name
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none', format='png')
            plt.close(fig)
            self.generated_figures.append(str(path))
            return str(path)
        except Exception as _exc:
            _logger.warning('savefig failed for %s (retrying at 150 dpi): %s',
                            name, _exc)
            try:
                path = self.output_dir / name
                fig.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                self.generated_figures.append(str(path))
                return str(path)
            except Exception as _exc2:
                _logger.warning('savefig retry also failed for %s: %s',
                                name, _exc2)
                plt.close(fig)
                return None

    @staticmethod
    def _truncate(label: str, n: int = 18) -> str:
        """
        Truncate long strings with an ellipsis for plotting.

        Parameters
        ----------
        label : str
            The input string.
        n : int, default=18
            The maximum number of characters.

        Returns
        -------
        str
            The truncated string.
        """
        return label if len(label) <= n else label[:n - 1] + '…'

    def get_generated_figures(self) -> List[str]:
        """
        Get the list of all files saved by this plotter.

        Returns
        -------
        List[str]
            Absolute paths to saved figures.
        """
        return list(self.generated_figures)


__all__ = [
    'HAS_MATPLOTLIB', 'HAS_SCIPY',
    'PALETTE', 'CATEGORICAL_COLORS', 'GRADIENT_CMAPS',
    'apply_style', 'BasePlotter',
    'plt', 'np', 'sp_stats',
]
