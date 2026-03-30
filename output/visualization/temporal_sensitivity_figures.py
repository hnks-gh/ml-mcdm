"""
Temporal Stability and Sensitivity Figures.

This module provides the `TemporalSensitivityFigureGenerator` class, 
which produces high-resolution illustrations for temporal stability and 
CRITIC-based sensitivity analysis. These figures are optimized for 
LaTeX integration and provide fine-grained diagnostics of ranking 
robustness across time windows and perturbation tiers.

Key Figures
-----------
- **Temporal Stability Timeline**: Rolling Spearman's ρ and Kendall's W 
  consensus across time windows.
- **Sensitivity Heatmap**: Analysis of rank disruption across criteria and 
  perturbation tiers (conservative, moderate, aggressive).
- **Robustness Comparison**: Tier-wise aggregate stability scores.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


class TemporalSensitivityFigureGenerator:
    """
    Generator for publication-quality temporal stability and sensitivity figures.

    Produces high-resolution PNG exports with consistent styling for 
    academic reporting and executive diagnostics.
    """

    def __init__(self, output_dir: str = 'output/result/figures', dpi: int = 300):
        """
        Initialize the figure generator.

        Parameters
        ----------
        output_dir : str, default='output/result/figures'
            Directory for PNG exports.
        dpi : int, default=300
            Resolution for saved files (dots per inch).
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                'matplotlib not available; install via: pip install matplotlib seaborn'
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        # Set publication-grade styling
        sns.set_style('whitegrid')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10
        plt.rcParams['legend.framealpha'] = 0.95

    def plot_temporal_stability_timeline(
        self,
        temporal_result: Any,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 5),
    ) -> str:
        """
        Render a timeline of temporal stability across rolling windows.

        Visualizes pairwise Spearman's ρ between consecutive windows along 
        with an omnibus Kendall's W reference.

        Parameters
        ----------
        temporal_result : TemporalStabilityResult
            Output from the WindowedTemporalStabilityAnalyzer.
        title : str, optional
            The plot title.
        figsize : Tuple[int, int], default=(8, 5)
            Figure dimensions (width, height) in inches.

        Returns
        -------
        str
            Absolute path to the saved figure.
        """
        if not temporal_result or not temporal_result.rolling_timeline:
            logger.warning('Cannot plot temporal stability: empty timeline')
            return ''

        fig, ax = plt.subplots(figsize=figsize)

        # Extract data
        timeline = temporal_result.rolling_timeline
        window_labels = [item.get('window_label', f"W{i}") for i, item in enumerate(timeline)]
        rho_values = [item.get('rho_to_next', 0.0) for item in timeline[:-1]]  # Pairs, so n-1
        window_indices = np.arange(len(rho_values))

        # Plot Spearman's rho as primary line
        ax.plot(
            window_indices,
            rho_values,
            marker='o',
            linestyle='-',
            linewidth=2.5,
            markersize=8,
            color='#1f77b4',  # Blue
            label=f"Spearman's ρ (mean={temporal_result.spearman_rho_mean:.3f})",
            zorder=3,
        )

        # Add Kendall's W reference line
        ax.axhline(
            temporal_result.kendalls_w,
            color='#ff7f0e',
            linestyle='--',
            linewidth=2.0,
            label=f"Kendall's W (omnibus={temporal_result.kendalls_w:.3f})",
            zorder=2,
        )

        # Add stability threshold reference (0.70)
        ax.axhline(0.70, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, zorder=1)
        ax.text(
            -0.5, 0.70, 'Stability(0.70)',
            fontsize=9, va='bottom', ha='right', color='gray', alpha=0.7
        )

        # Formatting
        ax.set_xlabel('Window Pair', fontsize=11, fontweight='bold')
        ax.set_ylabel("Correlation Coefficient", fontsize=11, fontweight='bold')
        ax.set_title(
            title or 'Temporal Stability Analysis: Window-Based Spearman\'s Rho',
            fontsize=12, fontweight='bold', pad=15
        )
        ax.set_xticks(window_indices)
        ax.set_xticklabels([f'P{i+1}' for i in window_indices], fontsize=9)
        ax.set_ylim([min(0, min(rho_values) - 0.1), 1.05])
        ax.grid(True, alpha=0.3, zorder=0)
        ax.legend(loc='lower left', fontsize=10, framealpha=0.95)

        # Save
        path = self.output_dir / 'temporal_stability_timeline.png'
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f'Saved: {path.name} (300 DPI, {figsize[0]}×{figsize[1]} inches)')
        return str(path)

    def plot_sensitivity_heatmap(
        self,
        sensitivity_result: Any,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> str:
        """
        Render a sensitivity heatmap of criteria across perturbation tiers.

        Visualizes the probability of rank disruption for each criterion 
        under conservative, moderate, and aggressive weight shifts.

        Parameters
        ----------
        sensitivity_result : SensitivityResult
            Output from the CRITICSensitivityAnalyzer.
        title : str, optional
            The plot title.
        figsize : Tuple[int, int], default=(10, 6)
            Figure dimensions (width, height) in inches.

        Returns
        -------
        str
            Absolute path to the saved figure.
        """
        if not sensitivity_result or not sensitivity_result.per_criterion_sensitivity:
            logger.warning('Cannot plot sensitivity: empty results')
            return ''

        # Build heatmap matrix: criteria × tiers
        criteria = sorted(sensitivity_result.per_criterion_sensitivity.keys())
        tiers = ['conservative', 'moderate', 'aggressive']

        # Sensitivity scores (fraction of perturbations causing rank change)
        sensitivity_matrix = np.zeros((len(criteria), len(tiers)))

        for i, criterion in enumerate(criteria):
            for j, tier in enumerate(tiers):
                sensitivity_matrix[i, j] = (
                    sensitivity_result.per_criterion_sensitivity[criterion].get(tier, 0.0)
                )

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        # Use diverging colormap: low sensitivity (cool) → high sensitivity (warm)
        cmap = sns.color_palette('RdYlGn_r', as_cmap=True)  # Red=high, Green=low
        im = ax.imshow(sensitivity_matrix, cmap=cmap, aspect='auto', vmin=0.0, vmax=1.0)

        # Axes labels
        ax.set_xticks(np.arange(len(tiers)))
        ax.set_yticks(np.arange(len(criteria)))
        ax.set_xticklabels([t.capitalize() for t in tiers], fontsize=10)
        ax.set_yticklabels(criteria, fontsize=10)

        # Rotate x labels for clarity
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

        # Add text annotations
        for i in range(len(criteria)):
            for j in range(len(tiers)):
                value = sensitivity_matrix[i, j]
                # Text color based on background
                text_color = 'white' if value > 0.5 else 'black'
                ax.text(
                    j, i, f'{value:.2f}',
                    ha='center', va='center',
                    color=text_color, fontsize=9, fontweight='bold'
                )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Sensitivity Score (0=robust, 1=very sensitive)', fontsize=10)

        # Title and labels
        ax.set_xlabel('Perturbation Tier', fontsize=11, fontweight='bold', labelpad=10)
        ax.set_ylabel('Criterion', fontsize=11, fontweight='bold', labelpad=10)
        ax.set_title(
            title or 'Sensitivity Analysis: Per-Criterion Rank Disruption by Perturbation Tier',
            fontsize=12, fontweight='bold', pad=15
        )

        # Grid
        ax.set_xticks(np.arange(len(tiers)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(criteria)) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

        # Save
        path = self.output_dir / 'sensitivity_heatmap.png'
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f'Saved: {path.name} (300 DPI, {figsize[0]}×{figsize[1]} inches)')
        return str(path)

    def plot_robustness_comparison(
        self,
        sensitivity_result: Any,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 5),
    ) -> str:
        """
        Render a bar chart comparing robustness across perturbation tiers.

        Provides a high-level summary of how the model performs as weight 
        variance increases.

        Parameters
        ----------
        sensitivity_result : SensitivityResult
            Output from the CRITICSensitivityAnalyzer.
        title : str, optional
            The plot title.
        figsize : Tuple[int, int], default=(8, 5)
            Figure dimensions (width, height) in inches.

        Returns
        -------
        str
            Absolute path to the saved figure.
        """
        if not sensitivity_result or not sensitivity_result.tier_robustness:
            logger.warning('Cannot plot robustness: empty results')
            return ''

        fig, ax = plt.subplots(figsize=figsize)

        # Extract robustness scores
        tiers = list(sensitivity_result.tier_robustness.keys())
        robustness = list(sensitivity_result.tier_robustness.values())

        # Color gradient: conservative (green) → aggressive (red)
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red

        # Create bar chart
        bars = ax.bar(tiers, robustness, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

        # Add value labels on bars
        for bar, rob in zip(bars, robustness):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f'{rob:.3f}',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold'
            )

        # Add robustness threshold reference (0.90)
        ax.axhline(0.90, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(
            -0.4, 0.90, 'High (0.90)',
            fontsize=9, va='bottom', ha='right', color='gray', alpha=0.7
        )

        # Formatting
        ax.set_ylabel('Robustness Score', fontsize=11, fontweight='bold')
        ax.set_title(
            title or 'Weight Robustness Across Perturbation Tiers',
            fontsize=12, fontweight='bold', pad=15
        )
        ax.set_ylim([0, 1.1])
        ax.set_xticklabels([t.capitalize() for t in tiers], fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

        # Save
        path = self.output_dir / 'robustness_comparison.png'
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f'Saved: {path.name} (300 DPI, {figsize[0]}×{figsize[1]} inches)')
        return str(path)
