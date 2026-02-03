# -*- coding: utf-8 -*-
"""High-resolution visualization for MCDM analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Import matplotlib with non-interactive backend fallback
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Polygon
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


class PanelVisualizer:
    """
    Professional high-resolution visualization for panel data MCDM analysis.
    
    All visualizations are saved as individual high-resolution PNG files.
    Each figure is a single, focused chart for maximum clarity.
    """
    
    def __init__(self,
                 output_dir: str = 'outputs/figures',
                 style: str = 'seaborn-v0_8-whitegrid',
                 figsize: Tuple[int, int] = (12, 8),
                 dpi: int = 300):
        """
        Initialize visualizer with high-resolution settings.
        
        Parameters
        ----------
        output_dir : str
            Directory for saving figures
        style : str
            Matplotlib style
        figsize : Tuple[int, int]
            Default figure size (width, height) in inches
        dpi : int
            Figure resolution (300 DPI for publication quality)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi  # High resolution: 300 DPI
        
        if HAS_MATPLOTLIB:
            try:
                plt.style.use(style)
            except:
                plt.style.use('default')
            
            # Set global parameters for high-quality output
            plt.rcParams['figure.dpi'] = dpi
            plt.rcParams['savefig.dpi'] = dpi
            plt.rcParams['font.size'] = 11
            plt.rcParams['axes.titlesize'] = 13
            plt.rcParams['axes.labelsize'] = 11
            plt.rcParams['xtick.labelsize'] = 10
            plt.rcParams['ytick.labelsize'] = 10
            plt.rcParams['legend.fontsize'] = 10
            plt.rcParams['figure.titlesize'] = 14
        
        # Professional color palette
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'tertiary': '#F18F01',
            'quaternary': '#C73E1D',
            'success': '#2E7D32',
            'warning': '#F57C00',
            'info': '#1976D2',
            'light': '#E0E0E0',
            'dark': '#212121'
        }
        
        self.cmap = 'viridis'
        
        # Track generated figures
        self.generated_figures = []
    
    def _save_figure(self, fig, save_name: str) -> str:
        """Save figure with high resolution settings and error handling."""
        try:
            save_path = self.output_dir / save_name
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none',
                       format='png', transparent=False)
            plt.close(fig)
            self.generated_figures.append(str(save_path))
            return str(save_path)
        except Exception as e:
            # Try saving with lower DPI as fallback
            try:
                save_path = self.output_dir / save_name
                fig.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                self.generated_figures.append(str(save_path))
                return str(save_path)
            except Exception:
                plt.close(fig)
                return None
    
    def get_generated_figures(self) -> List[str]:
        """Return list of all generated figure paths."""
        return self.generated_figures
    
    # =========================================================================
    # INDIVIDUAL CHART METHODS - Each produces a single high-resolution figure
    # =========================================================================
    
    def plot_score_evolution_top(self,
                                 scores: pd.DataFrame,
                                 entity_col: str = 'Province',
                                 time_col: str = 'Year',
                                 score_col: str = 'score',
                                 top_n: int = 10,
                                 title: str = 'Top Performers Score Evolution',
                                 save_name: str = '01_score_evolution_top.png') -> Optional[str]:
        """Plot score evolution for top performers as single chart."""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Pivot data
        wide = scores.pivot(index=time_col, columns=entity_col, values=score_col)
        
        # Get top performers (by mean score)
        mean_scores = wide.mean().sort_values(ascending=False)
        top_entities = mean_scores.head(top_n).index
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, top_n))
        
        for i, entity in enumerate(top_entities):
            ax.plot(wide.index, wide[entity], marker='o', linewidth=2.5, 
                   label=f'{entity}', markersize=8, color=colors[i])
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(wide.index)
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_score_evolution_bottom(self,
                                    scores: pd.DataFrame,
                                    entity_col: str = 'Province',
                                    time_col: str = 'Year',
                                    score_col: str = 'score',
                                    bottom_n: int = 10,
                                    title: str = 'Bottom Performers Score Evolution',
                                    save_name: str = '02_score_evolution_bottom.png') -> Optional[str]:
        """Plot score evolution for bottom performers as single chart."""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Pivot data
        wide = scores.pivot(index=time_col, columns=entity_col, values=score_col)
        
        # Get bottom performers
        mean_scores = wide.mean().sort_values(ascending=False)
        bottom_entities = mean_scores.tail(bottom_n).index
        
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, bottom_n))
        
        for i, entity in enumerate(bottom_entities):
            ax.plot(wide.index, wide[entity], marker='s', linewidth=2.5, 
                   label=f'{entity}', markersize=8, linestyle='--', color=colors[i])
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(wide.index)
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)

    def plot_score_evolution(self,
                            scores: pd.DataFrame,
                            entity_col: str = 'province',
                            time_col: str = 'year',
                            score_col: str = 'score',
                            top_n: int = 10,
                            bottom_n: int = 5,
                            title: str = 'Score Evolution Over Time',
                            save_name: str = 'score_evolution.png') -> Optional[str]:
        """
        Plot score evolution for top and bottom performers.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1] * 0.7))
        
        # Pivot data
        wide = scores.pivot(index=time_col, columns=entity_col, values=score_col)
        
        # Get top and bottom performers (by mean score)
        mean_scores = wide.mean().sort_values(ascending=False)
        top_entities = mean_scores.head(top_n).index
        bottom_entities = mean_scores.tail(bottom_n).index
        
        # Plot top performers
        ax = axes[0]
        for entity in top_entities:
            ax.plot(wide.index, wide[entity], marker='o', linewidth=2, 
                   label=entity[:15], markersize=6)
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'Top {top_n} Performers', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot bottom performers
        ax = axes[1]
        for entity in bottom_entities:
            ax.plot(wide.index, wide[entity], marker='s', linewidth=2,
                   label=entity[:15], markersize=6, linestyle='--')
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'Bottom {bottom_n} Performers', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_rank_heatmap(self,
                         rankings: pd.DataFrame,
                         entity_col: str = 'province',
                         time_col: str = 'year',
                         rank_col: str = 'rank',
                         title: str = 'Ranking Evolution Heatmap',
                         save_name: str = 'rank_heatmap.png') -> Optional[str]:
        """
        Plot heatmap of rankings over time.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        # Pivot to wide format
        wide = rankings.pivot(index=entity_col, columns=time_col, values=rank_col)
        
        # Sort by mean rank
        wide = wide.loc[wide.mean(axis=1).sort_values().index]
        
        # Limit to top 30 for readability
        if len(wide) > 30:
            wide = wide.head(30)
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], len(wide) * 0.4 + 2))
        
        # Create heatmap
        im = ax.imshow(wide.values, aspect='auto', cmap='RdYlGn_r')
        
        # Labels
        ax.set_yticks(range(len(wide)))
        ax.set_yticklabels([str(e)[:20] for e in wide.index], fontsize=9)
        ax.set_xticks(range(len(wide.columns)))
        ax.set_xticklabels(wide.columns, fontsize=10)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Rank (1=Best)', fontsize=10)
        
        # Add text annotations
        for i in range(len(wide)):
            for j in range(len(wide.columns)):
                rank = wide.iloc[i, j]
                if not np.isnan(rank):
                    text_color = 'white' if rank > len(wide) / 2 else 'black'
                    ax.text(j, i, f'{int(rank)}', ha='center', va='center',
                           color=text_color, fontsize=8)
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Province', fontsize=11)
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_method_comparison(self,
                              rankings_dict: Dict[str, np.ndarray],
                              entity_names: List[str],
                              title: str = 'MCDM Method Comparison',
                              save_name: str = 'method_comparison.png') -> Optional[str]:
        """
        Compare rankings from different MCDM methods.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        methods = list(rankings_dict.keys())
        n_methods = len(methods)
        n_entities = len(entity_names)
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], max(8, n_entities * 0.3)))
        
        # Create rank matrix
        rank_matrix = np.column_stack([rankings_dict[m] for m in methods])
        
        # Sort by first method
        sort_idx = np.argsort(rank_matrix[:, 0])[:30]  # Top 30
        rank_matrix = rank_matrix[sort_idx]
        sorted_names = [entity_names[i][:20] for i in sort_idx]
        
        # Parallel coordinates style
        x_positions = np.arange(n_methods)
        
        for i, name in enumerate(sorted_names):
            ranks = rank_matrix[i]
            color = plt.cm.viridis(i / len(sorted_names))
            ax.plot(x_positions, ranks, '-o', color=color, alpha=0.7,
                   linewidth=1.5, markersize=6, label=name if i < 10 else '')
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(methods, fontsize=11, rotation=15)
        ax.set_ylabel('Rank', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.invert_yaxis()  # Lower rank is better
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_convergence(self,
                        sigma_by_year: Dict[int, float],
                        beta_coefficient: float,
                        half_life: float,
                        title: str = 'Convergence Analysis',
                        save_name: str = 'convergence.png') -> Optional[str]:
        """
        Plot convergence analysis results.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Sigma convergence
        ax = axes[0]
        years = list(sigma_by_year.keys())
        sigmas = list(sigma_by_year.values())
        
        ax.plot(years, sigmas, 'o-', linewidth=2, markersize=8,
               color=self.colors['primary'])
        
        # Trend line
        z = np.polyfit(range(len(years)), sigmas, 1)
        p = np.poly1d(z)
        ax.plot(years, p(range(len(years))), '--', 
               color=self.colors['secondary'], linewidth=2, label='Trend')
        
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Coefficient of Variation (σ)', fontsize=11)
        ax.set_title('Sigma Convergence', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Beta convergence info
        ax = axes[1]
        ax.axis('off')
        
        convergence_type = 'CONVERGING' if beta_coefficient < 0 else 'DIVERGING'
        color = self.colors['success'] if beta_coefficient < 0 else self.colors['quaternary']
        
        info_text = f"""
        Beta Convergence Analysis
        ─────────────────────────
        
        β coefficient: {beta_coefficient:.4f}
        
        Interpretation:
        {"Provinces are converging" if beta_coefficient < 0 else "Provinces are diverging"}
        {"(poorer catching up to richer)" if beta_coefficient < 0 else "(gap is widening)"}
        
        Convergence Speed: {abs(beta_coefficient):.4f}
        
        Half-life: {half_life:.1f} years
        {"(time to halve the gap)" if half_life < 100 else "(very slow convergence)"}
        
        Status: {convergence_type}
        """
        
        ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='center', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_weight_sensitivity(self,
                               weight_sensitivity: Dict[str, float],
                               title: str = 'Weight Sensitivity Analysis',
                               save_name: str = 'weight_sensitivity.png') -> Optional[str]:
        """
        Plot weight sensitivity bar chart.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] * 0.6))
        
        criteria = list(weight_sensitivity.keys())
        sensitivities = list(weight_sensitivity.values())
        
        # Sort by sensitivity
        sorted_idx = np.argsort(sensitivities)[::-1]
        criteria = [criteria[i] for i in sorted_idx]
        sensitivities = [sensitivities[i] for i in sorted_idx]
        
        # Color gradient
        colors = [plt.cm.RdYlGn_r(s) for s in sensitivities]
        
        bars = ax.barh(range(len(criteria)), sensitivities, color=colors)
        
        ax.set_yticks(range(len(criteria)))
        ax.set_yticklabels([c[:25] for c in criteria], fontsize=10)
        ax.set_xlabel('Sensitivity Index', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for bar, val in zip(bars, sensitivities):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9)
        
        ax.set_xlim(0, max(sensitivities) * 1.15)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_feature_importance(self,
                               importance_dict: Dict[str, float],
                               title: str = 'Feature Importance',
                               save_name: str = 'feature_importance.png',
                               top_n: int = 20) -> Optional[str]:
        """
        Plot feature importance from ML models.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] * 0.7))
        
        # Sort and take top N
        sorted_items = sorted(importance_dict.items(), 
                             key=lambda x: x[1], reverse=True)[:top_n]
        features = [item[0] for item in sorted_items]
        importances = [item[1] for item in sorted_items]
        
        # Color gradient
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(features)))[::-1]
        
        bars = ax.barh(range(len(features)), importances, color=colors)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([f[:30] for f in features], fontsize=10)
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for bar, val in zip(bars, importances):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9)
        
        ax.set_xlim(0, max(importances) * 1.15)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_radar_chart(self,
                        values: Dict[str, np.ndarray],
                        categories: List[str],
                        title: str = 'Multi-Criteria Profile',
                        save_name: str = 'radar_chart.png') -> Optional[str]:
        """
        Plot radar chart for multi-criteria comparison.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        n_categories = len(categories)
        angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(values)))
        
        for (name, vals), color in zip(values.items(), colors):
            vals = list(vals) + [vals[0]]  # Close
            ax.plot(angles, vals, 'o-', linewidth=2, label=name[:20], color=color)
            ax.fill(angles, vals, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c[:15] for c in categories], fontsize=10)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_ensemble_weights(self,
                             weights: Dict[str, float],
                             title: str = 'Ensemble Model Weights',
                             save_name: str = 'ensemble_weights.png') -> Optional[str]:
        """
        Plot pie chart of ensemble weights.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = list(weights.keys())
        sizes = list(weights.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                          colors=colors, startangle=90,
                                          explode=[0.02] * len(labels))
        
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def create_dashboard(self,
                        results: Dict[str, Any],
                        save_name: str = 'dashboard.png') -> Optional[str]:
        """
        Create comprehensive results dashboard.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('ML-MCDM Panel Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Score distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if 'scores' in results:
            ax1.hist(results['scores'], bins=30, color=self.colors['primary'],
                    edgecolor='white', alpha=0.7)
            ax1.set_title('Score Distribution', fontsize=11, fontweight='bold')
            ax1.set_xlabel('Score')
            ax1.set_ylabel('Frequency')
        
        # 2. Top 10 bar chart
        ax2 = fig.add_subplot(gs[0, 1])
        if 'top_10' in results:
            names, scores = zip(*results['top_10'])
            colors = plt.cm.Greens(np.linspace(0.3, 0.9, 10))[::-1]
            ax2.barh(range(10), scores, color=colors)
            ax2.set_yticks(range(10))
            ax2.set_yticklabels([n[:15] for n in names], fontsize=9)
            ax2.set_title('Top 10 Provinces', fontsize=11, fontweight='bold')
            ax2.invert_yaxis()
        
        # 3. Method agreement
        ax3 = fig.add_subplot(gs[0, 2])
        if 'agreement_matrix' in results:
            im = ax3.imshow(results['agreement_matrix'], cmap='RdYlGn', 
                           vmin=-1, vmax=1)
            ax3.set_title('Method Agreement', fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax3, shrink=0.8)
        
        # 4-6. More visualizations based on available results
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.text(0.5, 0.5, 'Panel\nAnalysis\nSummary', ha='center', va='center',
                fontsize=14, transform=ax4.transAxes)
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.text(0.5, 0.5, 'Temporal\nTrends', ha='center', va='center',
                fontsize=14, transform=ax5.transAxes)
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.text(0.5, 0.5, 'Convergence\nStatus', ha='center', va='center',
                fontsize=14, transform=ax6.transAxes)
        ax6.axis('off')
        
        # 7-9. Bottom row
        ax7 = fig.add_subplot(gs[2, :])
        if 'summary_table' in results:
            ax7.axis('off')
            # Create summary table
            table_data = results['summary_table']
            table = ax7.table(cellText=table_data['values'],
                             colLabels=table_data['columns'],
                             rowLabels=table_data['rows'],
                             loc='center',
                             cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
        else:
            ax7.text(0.5, 0.5, 'Summary Statistics', ha='center', va='center',
                    fontsize=14, transform=ax7.transAxes)
            ax7.axis('off')
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)


    def plot_ml_training_progress(self,
                                  train_losses: List[float],
                                  val_losses: Optional[List[float]] = None,
                                  metrics_history: Optional[Dict[str, List[float]]] = None,
                                  title: str = 'ML Training Progress',
                                  save_name: str = 'ml_training_progress.png') -> Optional[str]:
        """
        Plot machine learning training progress with loss curves and metrics.
        
        Parameters
        ----------
        train_losses : List[float]
            Training loss values per epoch
        val_losses : List[float], optional
            Validation loss values per epoch
        metrics_history : Dict[str, List[float]], optional
            Additional metrics to plot (e.g., {'R²': [...], 'MAE': [...]})
        """
        if not HAS_MATPLOTLIB:
            return None
        
        n_plots = 1 + (1 if metrics_history else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        
        if n_plots == 1:
            axes = [axes]
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        ax = axes[0]
        ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
        if val_losses:
            ax.plot(epochs, val_losses, 'r--', linewidth=2, label='Validation Loss', marker='s', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Highlight minimum loss
        min_train_idx = np.argmin(train_losses)
        ax.axvline(x=min_train_idx + 1, color='blue', linestyle=':', alpha=0.5)
        ax.annotate(f'Best: {train_losses[min_train_idx]:.4f}', 
                   xy=(min_train_idx + 1, train_losses[min_train_idx]),
                   xytext=(10, 10), textcoords='offset points', fontsize=9)
        
        if val_losses:
            min_val_idx = np.argmin(val_losses)
            ax.axvline(x=min_val_idx + 1, color='red', linestyle=':', alpha=0.5)
        
        # Additional metrics
        if metrics_history and len(axes) > 1:
            ax = axes[1]
            colors = plt.cm.Set2(np.linspace(0, 1, len(metrics_history)))
            for (metric_name, values), color in zip(metrics_history.items(), colors):
                ax.plot(range(1, len(values) + 1), values, '-o', linewidth=2, 
                       label=metric_name, color=color, markersize=4)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Metric Value', fontsize=11)
            ax.set_title('Training Metrics', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_model_comparison(self,
                             model_results: Dict[str, Dict[str, float]],
                             metrics: List[str] = ['R²', 'MAE', 'RMSE'],
                             title: str = 'Model Performance Comparison',
                             save_name: str = 'model_comparison.png') -> Optional[str]:
        """
        Compare multiple ML models across different metrics.
        
        Parameters
        ----------
        model_results : Dict[str, Dict[str, float]]
            Model name -> {metric: value}
        metrics : List[str]
            Metrics to compare
        """
        if not HAS_MATPLOTLIB:
            return None
        
        models = list(model_results.keys())
        n_metrics = len(metrics)
        n_models = len(models)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_models))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [model_results[m].get(metric, 0) for m in models]
            bars = ax.bar(range(n_models), values, color=colors, edgecolor='black', linewidth=0.5)
            
            ax.set_xticks(range(n_models))
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_prediction_analysis(self,
                                 y_actual: np.ndarray,
                                 y_predicted: np.ndarray,
                                 entity_names: Optional[List[str]] = None,
                                 title: str = 'Prediction Analysis',
                                 save_name: str = 'prediction_analysis.png') -> Optional[str]:
        """
        Plot actual vs predicted values with residual analysis.
        
        Parameters
        ----------
        y_actual : np.ndarray
            Actual values
        y_predicted : np.ndarray
            Predicted values
        entity_names : List[str], optional
            Names of entities for labeling
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Actual vs Predicted scatter plot
        ax = axes[0, 0]
        ax.scatter(y_actual, y_predicted, alpha=0.6, c=self.colors['primary'], 
                  edgecolors='white', s=80)
        
        # Perfect prediction line
        min_val = min(y_actual.min(), y_predicted.min())
        max_val = max(y_actual.max(), y_predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Regression line
        z = np.polyfit(y_actual, y_predicted, 1)
        p = np.poly1d(z)
        ax.plot(np.sort(y_actual), p(np.sort(y_actual)), 'g-', linewidth=2, 
               label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        
        ax.set_xlabel('Actual Values', fontsize=11)
        ax.set_ylabel('Predicted Values', fontsize=11)
        ax.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Calculate R²
        ss_res = np.sum((y_actual - y_predicted) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Residual plot
        ax = axes[0, 1]
        residuals = y_actual - y_predicted
        ax.scatter(y_predicted, residuals, alpha=0.6, c=self.colors['secondary'], 
                  edgecolors='white', s=80)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Values', fontsize=11)
        ax.set_ylabel('Residuals', fontsize=11)
        ax.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add residual statistics
        ax.text(0.05, 0.95, f'Mean: {residuals.mean():.4f}\nStd: {residuals.std():.4f}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Residual histogram
        ax = axes[1, 0]
        ax.hist(residuals, bins=30, color=self.colors['tertiary'], edgecolor='white', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.axvline(x=residuals.mean(), color='blue', linestyle='-', linewidth=2, 
                  label=f'Mean: {residuals.mean():.4f}')
        ax.set_xlabel('Residual', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 4. Prediction errors by entity (top/bottom)
        ax = axes[1, 1]
        abs_errors = np.abs(residuals)
        sorted_idx = np.argsort(abs_errors)[::-1]
        
        # Show top 15 largest errors
        n_show = min(15, len(abs_errors))
        top_errors = abs_errors[sorted_idx[:n_show]]
        
        if entity_names:
            labels = [entity_names[i][:15] for i in sorted_idx[:n_show]]
        else:
            labels = [f'Entity {sorted_idx[i]}' for i in range(n_show)]
        
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, n_show))
        bars = ax.barh(range(n_show), top_errors, color=colors)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Absolute Error', fontsize=11)
        ax.set_title('Top Prediction Errors', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_cv_results(self,
                       cv_scores: Dict[str, List[float]],
                       fold_names: Optional[List[str]] = None,
                       title: str = 'Cross-Validation Results',
                       save_name: str = 'cv_results.png') -> Optional[str]:
        """
        Plot cross-validation results across folds.
        
        Parameters
        ----------
        cv_scores : Dict[str, List[float]]
            Metric name -> scores per fold
        fold_names : List[str], optional
            Names for folds (e.g., ['2020', '2021', '2022'])
        """
        if not HAS_MATPLOTLIB:
            return None
        
        metrics = list(cv_scores.keys())
        n_folds = len(list(cv_scores.values())[0])
        n_metrics = len(metrics)
        
        if fold_names is None:
            fold_names = [f'Fold {i+1}' for i in range(n_folds)]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Bar chart per fold
        ax = axes[0]
        x = np.arange(n_folds)
        width = 0.8 / n_metrics
        colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))
        
        for i, (metric, scores) in enumerate(cv_scores.items()):
            offset = (i - n_metrics/2 + 0.5) * width
            bars = ax.bar(x + offset, scores, width, label=metric, color=colors[i], 
                         edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Fold', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Scores by Fold', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(fold_names, fontsize=10)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Box plot summary
        ax = axes[1]
        data = [cv_scores[m] for m in metrics]
        bp = ax.boxplot(data, labels=metrics, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add individual points
        for i, (metric, scores) in enumerate(cv_scores.items()):
            x_jitter = np.random.normal(i + 1, 0.05, len(scores))
            ax.scatter(x_jitter, scores, color='black', alpha=0.6, s=50, zorder=3)
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Score Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean ± std annotations
        for i, (metric, scores) in enumerate(cv_scores.items()):
            mean_val = np.mean(scores)
            std_val = np.std(scores)
            ax.annotate(f'{mean_val:.3f}±{std_val:.3f}', 
                       xy=(i + 1, max(scores) + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                       ha='center', fontsize=9, fontweight='bold')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_neural_forecast(self,
                          actual: np.ndarray,
                          predicted: np.ndarray,
                          entity_names: List[str],
                          train_loss: List[float],
                          val_loss: Optional[List[float]] = None,
                          title: str = 'Neural Network Forecast Results',
                          save_name: str = 'neural_forecast.png') -> Optional[str]:
        """
        Plot neural network forecasting results with training curves and predictions.
        
        Parameters
        ----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray  
            Predicted values
        entity_names : List[str]
            Names of entities
        train_loss : List[float]
            Training loss history
        val_loss : List[float], optional
            Validation loss history
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Training curves
        ax = axes[0, 0]
        epochs = range(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=3)
        if val_loss:
            ax.plot(epochs, val_loss, 'r--', linewidth=2, label='Val Loss', marker='s', markersize=3)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss (MSE)', fontsize=11)
        ax.set_title('Neural Network Training Progress', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 2. Actual vs Predicted
        ax = axes[0, 1]
        ax.scatter(actual, predicted, alpha=0.6, c=self.colors['primary'], 
                  edgecolors='white', s=80)
        min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('Actual', fontsize=11)
        ax.set_ylabel('Predicted', fontsize=11)
        ax.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Calculate metrics
        mse = np.mean((actual - predicted) ** 2)
        mae = np.mean(np.abs(actual - predicted))
        ax.text(0.05, 0.95, f'MSE: {mse:.4f}\nMAE: {mae:.4f}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Prediction comparison bars
        ax = axes[1, 0]
        n_show = min(20, len(entity_names))
        sorted_idx = np.argsort(actual)[::-1][:n_show]
        
        x = np.arange(n_show)
        width = 0.35
        
        ax.barh(x - width/2, actual[sorted_idx], width, label='Actual', 
               color=self.colors['primary'], alpha=0.8)
        ax.barh(x + width/2, predicted[sorted_idx], width, label='Predicted',
               color=self.colors['tertiary'], alpha=0.8)
        
        ax.set_yticks(x)
        ax.set_yticklabels([entity_names[i][:12] for i in sorted_idx], fontsize=9)
        ax.set_xlabel('Score', fontsize=11)
        ax.set_title('Top Predictions Comparison', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # 4. Rank comparison
        ax = axes[1, 1]
        actual_ranks = np.argsort(np.argsort(-actual)) + 1
        predicted_ranks = np.argsort(np.argsort(-predicted)) + 1
        
        ax.scatter(actual_ranks, predicted_ranks, alpha=0.6, c=self.colors['secondary'],
                  edgecolors='white', s=80)
        ax.plot([1, len(actual)], [1, len(actual)], 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('Actual Rank', fontsize=11)
        ax.set_ylabel('Predicted Rank', fontsize=11)
        ax.set_title('Rank Agreement', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Calculate rank correlation
        from scipy.stats import spearmanr
        corr, _ = spearmanr(actual_ranks, predicted_ranks)
        ax.text(0.05, 0.95, f'Spearman ρ: {corr:.4f}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_rf_analysis(self,
                        feature_importance: Dict[str, float],
                        cv_scores: Dict[str, List[float]],
                        test_actual: np.ndarray,
                        test_predicted: np.ndarray,
                        entity_names: List[str],
                        title: str = 'Random Forest Analysis',
                        save_name: str = 'rf_analysis.png') -> Optional[str]:
        """
        Comprehensive Random Forest analysis visualization.
        
        Parameters
        ----------
        feature_importance : Dict[str, float]
            Feature importance scores
        cv_scores : Dict[str, List[float]]
            Cross-validation scores
        test_actual : np.ndarray
            Test set actual values
        test_predicted : np.ndarray
            Test set predictions
        entity_names : List[str]
            Entity names
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Feature importance
        ax = axes[0, 0]
        sorted_items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        features = [item[0][:20] for item in sorted_items]
        importances = [item[1] for item in sorted_items]
        
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(features)))[::-1]
        bars = ax.barh(range(len(features)), importances, color=colors)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title('Feature Importance (Top 15)', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar, val in zip(bars, importances):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=8)
        
        # 2. CV Scores boxplot
        ax = axes[0, 1]
        cv_metrics = list(cv_scores.keys())
        data = [cv_scores[m] for m in cv_metrics]
        bp = ax.boxplot(data, labels=cv_metrics, patch_artist=True)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(cv_metrics)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Cross-Validation Performance', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean labels
        for i, metric in enumerate(cv_metrics):
            mean_val = np.mean(cv_scores[metric])
            ax.annotate(f'{mean_val:.3f}', xy=(i + 1, mean_val),
                       xytext=(5, 0), textcoords='offset points', fontsize=9)
        
        # 3. Actual vs Predicted
        ax = axes[1, 0]
        ax.scatter(test_actual, test_predicted, alpha=0.6, c=self.colors['primary'],
                  edgecolors='white', s=80)
        
        min_val, max_val = min(test_actual.min(), test_predicted.min()), max(test_actual.max(), test_predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        # Regression line
        z = np.polyfit(test_actual, test_predicted, 1)
        p = np.poly1d(z)
        ax.plot(np.sort(test_actual), p(np.sort(test_actual)), 'g-', linewidth=2, label='Fit')
        
        ax.set_xlabel('Actual', fontsize=11)
        ax.set_ylabel('Predicted', fontsize=11)
        ax.set_title('Test Set: Actual vs Predicted', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # R² and metrics
        ss_res = np.sum((test_actual - test_predicted) ** 2)
        ss_tot = np.sum((test_actual - np.mean(test_actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        mae = np.mean(np.abs(test_actual - test_predicted))
        ax.text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.4f}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Residual analysis
        ax = axes[1, 1]
        residuals = test_actual - test_predicted
        ax.scatter(test_predicted, residuals, alpha=0.6, c=self.colors['tertiary'],
                  edgecolors='white', s=80)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Residuals', fontsize=11)
        ax.set_title('Residual Analysis', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax.text(0.05, 0.95, f'Mean: {residuals.mean():.4f}\nStd: {residuals.std():.4f}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_ensemble_model_analysis(self,
                                     base_predictions: Dict[str, np.ndarray],
                                     meta_predictions: np.ndarray,
                                     actual: np.ndarray,
                                     weights: Dict[str, float],
                                     entity_names: List[str],
                                     title: str = 'Ensemble Model Analysis',
                                     save_name: str = 'ensemble_model_analysis.png') -> Optional[str]:
        """
        Visualize ensemble model performance and contributions.
        
        Parameters
        ----------
        base_predictions : Dict[str, np.ndarray]
            Predictions from base models
        meta_predictions : np.ndarray
            Final ensemble predictions
        actual : np.ndarray
            Actual values
        weights : Dict[str, float]
            Model weights in ensemble
        entity_names : List[str]
            Entity names
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Model weights pie chart
        ax = axes[0, 0]
        labels = list(weights.keys())
        sizes = list(weights.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                          colors=colors, startangle=90,
                                          explode=[0.02] * len(labels))
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        ax.set_title('Base Model Weights', fontsize=12, fontweight='bold')
        
        # 2. Base models vs Ensemble performance
        ax = axes[0, 1]
        models = list(base_predictions.keys()) + ['Ensemble']
        r2_scores = []
        
        for model, preds in base_predictions.items():
            ss_res = np.sum((actual - preds) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2_scores.append(1 - (ss_res / ss_tot) if ss_tot > 0 else 0)
        
        ss_res = np.sum((actual - meta_predictions) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2_scores.append(1 - (ss_res / ss_tot) if ss_tot > 0 else 0)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
        bars = ax.bar(range(len(models)), r2_scores, color=colors, edgecolor='black')
        
        # Highlight ensemble
        bars[-1].set_edgecolor('red')
        bars[-1].set_linewidth(2)
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('R² Score', fontsize=11)
        ax.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, r2_scores):
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', fontsize=9, fontweight='bold')
        
        # 3. Predictions correlation heatmap
        ax = axes[1, 0]
        all_preds = {**base_predictions, 'Ensemble': meta_predictions, 'Actual': actual}
        pred_df = pd.DataFrame(all_preds)
        corr = pred_df.corr()
        
        im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(corr.columns, fontsize=10)
        
        # Add correlation values
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                text_color = 'white' if abs(corr.iloc[i, j]) > 0.5 else 'black'
                ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                       color=text_color, fontsize=9)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation', fontsize=10)
        ax.set_title('Predictions Correlation', fontsize=12, fontweight='bold')
        
        # 4. Ensemble vs Actual scatter
        ax = axes[1, 1]
        ax.scatter(actual, meta_predictions, alpha=0.6, c=self.colors['primary'],
                  edgecolors='white', s=80)
        
        min_val, max_val = min(actual.min(), meta_predictions.min()), max(actual.max(), meta_predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        ax.set_xlabel('Actual', fontsize=11)
        ax.set_ylabel('Ensemble Prediction', fontsize=11)
        ax.set_title('Ensemble: Actual vs Predicted', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        ss_res = np.sum((actual - meta_predictions) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        ax.text(0.05, 0.95, f'Ensemble R² = {r2:.4f}', 
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_ml_summary_dashboard(self,
                                  results: Dict[str, Any],
                                  title: str = 'ML Analysis Dashboard',
                                  save_name: str = 'ml_dashboard.png') -> Optional[str]:
        """
        Create comprehensive ML analysis dashboard.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Dictionary containing:
            - 'rf_importance': feature importance dict
            - 'rf_cv_scores': CV scores dict
            - 'neural_train_loss': training loss list (optional, neural networks disabled by default)
            - 'neural_val_loss': validation loss list (optional)
            - 'model_metrics': Dict[model_name, Dict[metric, value]]
            - 'predictions': Dict[model_name, (actual, predicted)]
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Feature importance (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'rf_importance' in results and results['rf_importance']:
            sorted_items = sorted(results['rf_importance'].items(), 
                                 key=lambda x: x[1], reverse=True)[:10]
            features = [item[0][:15] for item in sorted_items]
            importances = [item[1] for item in sorted_items]
            
            colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(features)))[::-1]
            ax1.barh(range(len(features)), importances, color=colors)
            ax1.set_yticks(range(len(features)))
            ax1.set_yticklabels(features, fontsize=9)
            ax1.set_xlabel('Importance', fontsize=10)
            ax1.set_title('Top 10 Features', fontsize=11, fontweight='bold')
            ax1.invert_yaxis()
            ax1.grid(True, alpha=0.3, axis='x')
        else:
            ax1.text(0.5, 0.5, 'No RF Data', ha='center', va='center', fontsize=12)
            ax1.axis('off')
        
        # 2. Training curves (top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'neural_train_loss' in results and results['neural_train_loss']:
            epochs = range(1, len(results['neural_train_loss']) + 1)
            ax2.plot(epochs, results['neural_train_loss'], 'b-', linewidth=2, 
                    label='Train', marker='o', markersize=3)
            if 'neural_val_loss' in results and results['neural_val_loss']:
                ax2.plot(epochs, results['neural_val_loss'], 'r--', linewidth=2,
                        label='Val', marker='s', markersize=3)
            ax2.set_xlabel('Epoch', fontsize=10)
            ax2.set_ylabel('Loss', fontsize=10)
            ax2.set_title('Neural Network Training', fontsize=11, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Neural Data', ha='center', va='center', fontsize=12)
            ax2.axis('off')
        
        # 3. Model comparison (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        if 'model_metrics' in results and results['model_metrics']:
            models = list(results['model_metrics'].keys())
            r2_scores = [results['model_metrics'][m].get('R²', 0) for m in models]
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
            bars = ax3.bar(range(len(models)), r2_scores, color=colors, edgecolor='black')
            ax3.set_xticks(range(len(models)))
            ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
            ax3.set_ylabel('R² Score', fontsize=10)
            ax3.set_title('Model Comparison', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, r2_scores):
                ax3.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', fontsize=8, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Metrics', ha='center', va='center', fontsize=12)
            ax3.axis('off')
        
        # 4-6. Prediction scatter plots (middle row)
        if 'predictions' in results:
            for i, (model_name, (actual, predicted)) in enumerate(list(results['predictions'].items())[:3]):
                ax = fig.add_subplot(gs[1, i])
                ax.scatter(actual, predicted, alpha=0.5, c=self.colors['primary'], s=40)
                min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                ax.set_xlabel('Actual', fontsize=10)
                ax.set_ylabel('Predicted', fontsize=10)
                ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                ss_res = np.sum((actual - predicted) ** 2)
                ss_tot = np.sum((actual - np.mean(actual)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                ax.text(0.05, 0.95, f'R²={r2:.3f}', transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 7. CV Scores (bottom-left)
        ax7 = fig.add_subplot(gs[2, 0])
        if 'rf_cv_scores' in results and results['rf_cv_scores']:
            metrics = list(results['rf_cv_scores'].keys())
            data = [results['rf_cv_scores'][m] for m in metrics]
            bp = ax7.boxplot(data, labels=metrics, patch_artist=True)
            colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax7.set_ylabel('Score', fontsize=10)
            ax7.set_title('RF Cross-Validation', fontsize=11, fontweight='bold')
            ax7.grid(True, alpha=0.3, axis='y')
        else:
            ax7.text(0.5, 0.5, 'No CV Data', ha='center', va='center', fontsize=12)
            ax7.axis('off')
        
        # 8-9. Summary statistics (bottom-middle and right)
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('off')
        
        summary_text = "ML ANALYSIS SUMMARY\n" + "=" * 50 + "\n\n"
        
        if 'model_metrics' in results:
            for model, metrics in results['model_metrics'].items():
                summary_text += f"{model}:\n"
                for metric, value in metrics.items():
                    summary_text += f"  {metric}: {value:.4f}\n"
                summary_text += "\n"
        
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)

    # =========================================================================
    # ADDITIONAL INDIVIDUAL HIGH-RESOLUTION CHARTS
    # =========================================================================
    
    def plot_weights_comparison(self,
                                weights: Dict[str, np.ndarray],
                                component_names: List[str],
                                title: str = 'Criteria Weights Comparison',
                                save_name: str = '03_weights_comparison.png') -> Optional[str]:
        """Plot comparison of weights from different methods."""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        n_components = len(component_names)
        n_methods = len(weights)
        x = np.arange(n_components)
        width = 0.8 / n_methods
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_methods))
        
        for i, (method, w) in enumerate(weights.items()):
            offset = (i - n_methods/2 + 0.5) * width
            bars = ax.bar(x + offset, w, width, label=method.title(), 
                         color=colors[i], edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Component', fontsize=12, fontweight='bold')
        ax.set_ylabel('Weight', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([c[:10] for c in component_names], rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_topsis_scores_bar(self,
                               scores: np.ndarray,
                               entity_names: List[str],
                               title: str = 'MCDM Scores Ranking',
                               save_name: str = '04_mcdm_scores.png') -> Optional[str]:
        """
        Plot MCDM scores as horizontal bar chart.
        
        Works for any MCDM method (TOPSIS, VIKOR, PROMETHEE, etc.).
        Method name kept as 'plot_topsis_scores_bar' for compatibility.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(12, max(10, len(entity_names) * 0.3)))
        
        # Sort by score
        sorted_idx = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_idx]
        sorted_names = [entity_names[i] for i in sorted_idx]
        
        # Show top 30
        n_show = min(30, len(sorted_scores))
        colors = plt.cm.viridis(np.linspace(0.9, 0.2, n_show))
        
        bars = ax.barh(range(n_show), sorted_scores[:n_show], color=colors, 
                      edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(sorted_names[:n_show], fontsize=10)
        ax.set_xlabel('MCDM Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, sorted_scores[:n_show]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_vikor_analysis(self,
                            Q: np.ndarray,
                            S: np.ndarray,
                            R: np.ndarray,
                            entity_names: List[str],
                            title: str = 'VIKOR Analysis Results',
                            save_name: str = '05_vikor_analysis.png') -> Optional[str]:
        """Plot VIKOR Q, S, R values."""
        if not HAS_MATPLOTLIB:
            return None
        
        # Convert to numpy arrays if needed
        Q = np.array(Q) if hasattr(Q, 'values') else np.asarray(Q)
        S = np.array(S) if hasattr(S, 'values') else np.asarray(S)
        R = np.array(R) if hasattr(R, 'values') else np.asarray(R)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sort by Q value
        sorted_idx = np.argsort(Q)
        n_show = min(20, len(Q))
        
        x = np.arange(n_show)
        width = 0.25
        
        ax.bar(x - width, Q[sorted_idx][:n_show], width, label='Q', color=self.colors['primary'])
        ax.bar(x, S[sorted_idx][:n_show], width, label='S', color=self.colors['secondary'])
        ax.bar(x + width, R[sorted_idx][:n_show], width, label='R', color=self.colors['tertiary'])
        
        ax.set_xlabel('Entity', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([entity_names[i][:10] for i in sorted_idx[:n_show]], 
                          rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_method_agreement_matrix(self,
                                     rankings_dict: Dict[str, np.ndarray],
                                     title: str = 'MCDM Methods Agreement Matrix',
                                     save_name: str = '06_method_agreement.png') -> Optional[str]:
        """Plot correlation matrix between different MCDM methods."""
        if not HAS_MATPLOTLIB:
            return None
        
        from scipy.stats import spearmanr
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Convert to numpy arrays if needed
        rankings_np = {}
        for k, v in rankings_dict.items():
            rankings_np[k] = np.array(v) if hasattr(v, 'values') else np.asarray(v)
        
        methods = list(rankings_np.keys())
        n = len(methods)
        corr_matrix = np.zeros((n, n))
        
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                corr, _ = spearmanr(rankings_np[m1], rankings_np[m2])
                corr_matrix[i, j] = corr
        
        im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
        
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(methods, fontsize=11)
        
        # Add correlation values
        for i in range(n):
            for j in range(n):
                text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                ax.text(j, i, f'{corr_matrix[i, j]:.3f}', ha='center', va='center',
                       color=text_color, fontsize=11, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Spearman Correlation', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_score_distribution(self,
                                scores: np.ndarray,
                                title: str = 'Score Distribution',
                                save_name: str = '07_score_distribution.png') -> Optional[str]:
        """Plot histogram of score distribution."""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        n, bins, patches = ax.hist(scores, bins=30, color=self.colors['primary'],
                                   edgecolor='white', alpha=0.8)
        
        # Color gradient
        cm = plt.colormaps['viridis']
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        
        # Add statistics
        ax.axvline(scores.mean(), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {scores.mean():.4f}')
        ax.axvline(np.median(scores), color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(scores):.4f}')
        
        ax.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics box
        stats_text = f'N = {len(scores)}\nStd = {scores.std():.4f}\nMin = {scores.min():.4f}\nMax = {scores.max():.4f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_sigma_convergence(self,
                               sigma_by_year: Dict[int, float],
                               title: str = 'Sigma Convergence Analysis',
                               save_name: str = '08_sigma_convergence.png') -> Optional[str]:
        """Plot sigma convergence over time."""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        years = list(sigma_by_year.keys())
        sigmas = list(sigma_by_year.values())
        
        ax.plot(years, sigmas, 'o-', linewidth=2.5, markersize=10,
               color=self.colors['primary'], markeredgecolor='white', markeredgewidth=2)
        
        # Trend line
        z = np.polyfit(range(len(years)), sigmas, 1)
        p = np.poly1d(z)
        ax.plot(years, p(range(len(years))), '--', 
               color=self.colors['secondary'], linewidth=2, label=f'Trend (slope={z[0]:.4f})')
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Coefficient of Variation (σ)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks(years)
        
        # Interpretation
        trend = "Converging" if z[0] < 0 else "Diverging"
        ax.text(0.95, 0.95, f'Trend: {trend}', transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightgreen' if z[0] < 0 else 'lightyellow', alpha=0.7))
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_beta_convergence_info(self,
                                   beta_coefficient: float,
                                   half_life: float,
                                   title: str = 'Beta Convergence Analysis',
                                   save_name: str = '09_beta_convergence.png') -> Optional[str]:
        """Plot beta convergence information panel."""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.axis('off')
        
        convergence_type = 'CONVERGING' if beta_coefficient < 0 else 'DIVERGING'
        color = self.colors['success'] if beta_coefficient < 0 else self.colors['quaternary']
        
        info_text = f"""
╔══════════════════════════════════════════════════════════════╗
║                    BETA CONVERGENCE ANALYSIS                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   β Coefficient:     {beta_coefficient:>12.6f}                         ║
║                                                              ║
║   Interpretation:                                            ║
║   {"Provinces are converging (catching up)    " if beta_coefficient < 0 else "Provinces are diverging (gap widening)   "}     ║
║                                                              ║
║   Convergence Speed: {abs(beta_coefficient):>12.6f}                         ║
║                                                              ║
║   Half-Life:         {half_life:>12.1f} years                       ║
║   {"(time to halve development gap)           " if half_life < 100 else "(very slow convergence rate)              "}     ║
║                                                              ║
║   Status:            {convergence_type:>12s}                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        
        ax.text(0.5, 0.5, info_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='center', horizontalalignment='center',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_feature_importance_single(self,
                                       importance_dict: Dict[str, float],
                                       title: str = 'Feature Importance Analysis',
                                       save_name: str = '10_feature_importance.png',
                                       top_n: int = 20) -> Optional[str]:
        """Plot feature importance as single chart."""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))
        
        # Sort and take top N
        sorted_items = sorted(importance_dict.items(), 
                             key=lambda x: x[1], reverse=True)[:top_n]
        features = [item[0] for item in sorted_items]
        importances = [item[1] for item in sorted_items]
        
        # Color gradient
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(features)))[::-1]
        
        bars = ax.barh(range(len(features)), importances, color=colors,
                      edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, importances):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_sensitivity_analysis(self,
                                  weight_sensitivity: Dict[str, float],
                                  title: str = 'Criteria Sensitivity Analysis',
                                  save_name: str = '11_sensitivity_analysis.png') -> Optional[str]:
        """Plot weight sensitivity analysis."""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        criteria = list(weight_sensitivity.keys())
        sensitivities = list(weight_sensitivity.values())
        
        # Sort by sensitivity
        sorted_idx = np.argsort(sensitivities)[::-1]
        criteria = [criteria[i] for i in sorted_idx]
        sensitivities = [sensitivities[i] for i in sorted_idx]
        
        # Color gradient based on sensitivity
        norm = plt.Normalize(min(sensitivities), max(sensitivities))
        colors = plt.cm.RdYlGn_r(norm(sensitivities))
        
        bars = ax.barh(range(len(criteria)), sensitivities, color=colors,
                      edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(range(len(criteria)))
        ax.set_yticklabels([c[:25] for c in criteria], fontsize=10)
        ax.set_xlabel('Sensitivity Index', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, sensitivities):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=9)
        
        # Add interpretation
        high_sens = sum(1 for s in sensitivities if s > 0.5)
        ax.text(0.95, 0.05, f'High sensitivity criteria: {high_sens}/{len(criteria)}',
               transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_final_ranking_summary(self,
                                   entities: List[str],
                                   final_scores: np.ndarray,
                                   final_ranking: np.ndarray,
                                   title: str = 'Final Ranking Summary',
                                   save_name: str = '12_final_ranking.png') -> Optional[str]:
        """Plot final ranking summary."""
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(14, max(10, len(entities) * 0.25)))
        
        # Sort by ranking
        sorted_idx = np.argsort(final_ranking)
        n_show = min(40, len(entities))
        
        sorted_scores = final_scores[sorted_idx][:n_show]
        sorted_names = [entities[i] for i in sorted_idx[:n_show]]
        
        # Color gradient
        colors = plt.cm.RdYlGn(np.linspace(0.9, 0.2, n_show))
        
        bars = ax.barh(range(n_show), sorted_scores, color=colors,
                      edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(range(n_show))
        ax.set_yticklabels([f'{i+1}. {name}' for i, name in enumerate(sorted_names)], fontsize=10)
        ax.set_xlabel('Final Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, sorted_scores)):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_future_predictions(self,
                                entities: List[str],
                                current_scores: np.ndarray,
                                predicted_scores: np.ndarray,
                                prediction_year: int,
                                title: str = 'Future Predictions Comparison',
                                save_name: str = '15_future_predictions.png') -> Optional[str]:
        """
        Plot comparison of current vs predicted future rankings.
        
        Parameters
        ----------
        entities : List[str]
            Entity names
        current_scores : np.ndarray
            Current year MCDM scores (from any method)
        predicted_scores : np.ndarray
            Predicted future year MCDM scores
        prediction_year : int
            The year being predicted
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        
        # 1. Current vs Predicted Scatter
        ax = axes[0]
        ax.scatter(current_scores, predicted_scores, alpha=0.7, 
                   c=self.colors['primary'], s=80, edgecolor='white', linewidth=0.5)
        
        # Perfect correlation line
        min_val = min(current_scores.min(), predicted_scores.min())
        max_val = max(current_scores.max(), predicted_scores.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Perfect Correlation', alpha=0.7)
        
        # Correlation
        corr = np.corrcoef(current_scores, predicted_scores)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=ax.transAxes,
                fontsize=11, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Current MCDM Score (2024)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Predicted MCDM Score ({prediction_year})', fontsize=11, fontweight='bold')
        ax.set_title('Score Stability Analysis', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 2. Top 20 Predicted Rankings
        ax = axes[1]
        sorted_idx = np.argsort(predicted_scores)[::-1][:20]
        
        y_pos = np.arange(20)
        pred_vals = predicted_scores[sorted_idx]
        curr_vals = current_scores[sorted_idx]
        names = [entities[i] for i in sorted_idx]
        
        width = 0.35
        ax.barh(y_pos - width/2, pred_vals, width, label=f'Predicted ({prediction_year})',
                color=self.colors['primary'], edgecolor='black', linewidth=0.5)
        ax.barh(y_pos + width/2, curr_vals, width, label='Current (2024)',
                color=self.colors['secondary'], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'{i+1}. {n}' for i, n in enumerate(names)], fontsize=10)
        ax.set_xlabel('MCDM Score', fontsize=11, fontweight='bold')
        ax.set_title(f'Top 20 Predicted Rankings for {prediction_year}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    # =========================================================================
    # SINGLE-CHART ML PROGRESS VISUALIZATIONS
    # =========================================================================
    
    def plot_neural_training_curve(self,
                                 train_loss: List[float],
                                 val_loss: Optional[List[float]] = None,
                                 title: str = 'Neural Network Training Loss Progression',
                                 save_name: str = 'ml_neural_training_curve.png') -> Optional[str]:
        """
        Single chart showing neural network training and validation loss over epochs.
        Includes early stopping point, best epoch marker, and convergence analysis.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        epochs = np.arange(1, len(train_loss) + 1)
        
        # Plot training loss with gradient fill
        ax.plot(epochs, train_loss, 'b-', linewidth=2.5, label='Training Loss', 
                marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1.5)
        ax.fill_between(epochs, train_loss, alpha=0.2, color='blue')
        
        if val_loss:
            ax.plot(epochs, val_loss, 'r-', linewidth=2.5, label='Validation Loss',
                    marker='s', markersize=4, markerfacecolor='white', markeredgewidth=1.5)
            ax.fill_between(epochs, val_loss, alpha=0.2, color='red')
            
            # Find best epoch (lowest validation loss)
            best_epoch = np.argmin(val_loss) + 1
            best_val = min(val_loss)
            ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7,
                       label=f'Best Epoch: {best_epoch}')
            ax.scatter([best_epoch], [best_val], s=200, c='green', marker='*', 
                       zorder=5, edgecolors='black', linewidths=1)
            ax.annotate(f'Best: {best_val:.6f}', xy=(best_epoch, best_val),
                        xytext=(15, 15), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='green'),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
            
            # Calculate overfitting indicator
            if len(val_loss) > 10:
                recent_val = np.mean(val_loss[-5:])
                early_val = np.mean(val_loss[5:10])
                if recent_val > early_val * 1.1:
                    ax.text(0.98, 0.98, '⚠️ Potential Overfitting Detected', 
                            transform=ax.transAxes, fontsize=11, 
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Training statistics box
        stats_text = f"Training Statistics:\n"
        stats_text += f"  Initial Loss: {train_loss[0]:.6f}\n"
        stats_text += f"  Final Loss: {train_loss[-1]:.6f}\n"
        stats_text += f"  Reduction: {((train_loss[0] - train_loss[-1]) / train_loss[0] * 100):.1f}%\n"
        stats_text += f"  Total Epochs: {len(train_loss)}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, len(train_loss) + 1)
        
        # Add minor gridlines
        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.1)
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_rf_cv_progression(self,
                               cv_scores: Dict[str, List[float]],
                               title: str = 'Random Forest Cross-Validation Progression',
                               save_name: str = 'ml_rf_cv_progression.png') -> Optional[str]:
        """
        Single chart showing CV scores across folds with detailed statistics.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        metrics = list(cv_scores.keys())
        n_folds = len(list(cv_scores.values())[0])
        folds = np.arange(1, n_folds + 1)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(metrics)))
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
        
        for i, (metric, values) in enumerate(cv_scores.items()):
            marker = markers[i % len(markers)]
            ax.plot(folds, values, '-', linewidth=2.5, label=f'{metric}',
                    marker=marker, markersize=12, markerfacecolor='white',
                    markeredgewidth=2, color=colors[i])
            
            # Add mean line
            mean_val = np.mean(values)
            ax.axhline(y=mean_val, color=colors[i], linestyle='--', alpha=0.5, linewidth=1.5)
            
            # Annotate each point
            for fold, val in zip(folds, values):
                ax.annotate(f'{val:.4f}', xy=(fold, val),
                            xytext=(0, 8), textcoords='offset points',
                            fontsize=9, ha='center', fontweight='bold')
        
        # Statistics table
        stats_lines = ["Performance Statistics:"]
        for metric, values in cv_scores.items():
            mean_v = np.mean(values)
            std_v = np.std(values)
            stats_lines.append(f"  {metric}: {mean_v:.4f} ± {std_v:.4f}")
        
        ax.text(0.02, 0.02, '\n'.join(stats_lines), transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        ax.set_xlabel('Cross-Validation Fold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(folds)
        ax.set_xticklabels([f'Fold {i}' for i in folds], fontsize=11)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_rf_feature_importance_detailed(self,
                                            feature_importance: Dict[str, float],
                                            title: str = 'Random Forest Feature Importance Analysis',
                                            save_name: str = 'ml_rf_feature_importance.png') -> Optional[str]:
        """
        Single detailed chart showing feature importance with cumulative contribution.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Sort features by importance
        sorted_items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features = [item[0] for item in sorted_items]
        importances = [item[1] for item in sorted_items]
        
        # Color gradient based on importance
        norm_imp = np.array(importances) / max(importances)
        colors = plt.cm.Blues(0.3 + norm_imp * 0.6)
        
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importances, color=colors, edgecolor='navy', linewidth=0.8)
        
        # Add value labels
        for bar, val in zip(bars, importances):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
        
        # Cumulative importance line on secondary axis
        ax2 = ax.twiny()
        cumulative = np.cumsum(importances)
        cumulative_pct = cumulative / cumulative[-1] * 100
        ax2.plot(cumulative_pct, y_pos, 'r-', linewidth=2.5, marker='D', 
                 markersize=6, label='Cumulative %')
        ax2.set_xlabel('Cumulative Importance (%)', fontsize=11, color='red')
        ax2.tick_params(axis='x', colors='red')
        ax2.set_xlim(0, 105)
        
        # Mark 80% threshold
        idx_80 = np.argmax(cumulative_pct >= 80)
        ax.axhline(y=idx_80, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(max(importances) * 0.5, idx_80 - 0.5, 
                f'80% explained by top {idx_80 + 1} features',
                fontsize=10, color='green', fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # Statistics box
        stats_text = f"Total Features: {len(features)}\n"
        stats_text += f"Top Feature: {features[0]} ({importances[0]:.4f})\n"
        stats_text += f"Mean Importance: {np.mean(importances):.4f}\n"
        stats_text += f"Std Importance: {np.std(importances):.4f}"
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_actual_vs_predicted(self,
                                 actual: np.ndarray,
                                 predicted: np.ndarray,
                                 model_name: str = 'Model',
                                 entity_names: Optional[List[str]] = None,
                                 title: str = 'Actual vs Predicted Analysis',
                                 save_name: str = 'ml_actual_vs_predicted.png') -> Optional[str]:
        """
        Single detailed scatter plot with regression analysis.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Calculate metrics
        residuals = actual - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        # Color by residual magnitude
        residual_mag = np.abs(residuals)
        colors = plt.cm.RdYlGn_r(residual_mag / (residual_mag.max() + 1e-10))
        
        scatter = ax.scatter(actual, predicted, c=residual_mag, cmap='RdYlGn_r',
                             s=120, alpha=0.7, edgecolors='black', linewidths=0.5)
        
        # Perfect prediction line
        min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin],
                'k--', linewidth=2.5, label='Perfect Prediction', alpha=0.8)
        
        # Regression line
        z = np.polyfit(actual, predicted, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val - margin, max_val + margin, 100)
        ax.plot(x_line, p(x_line), 'b-', linewidth=2, 
                label=f'Fit: y = {z[0]:.3f}x + {z[1]:.3f}', alpha=0.8)
        
        # Confidence interval (95%)
        y_pred_line = p(x_line)
        ci = 1.96 * np.std(residuals)
        ax.fill_between(x_line, y_pred_line - ci, y_pred_line + ci, 
                        alpha=0.15, color='blue', label='95% CI')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('|Residual|', fontsize=11)
        
        # Annotate worst predictions
        worst_idx = np.argsort(residual_mag)[-5:]
        for idx in worst_idx:
            label = entity_names[idx] if entity_names else f'#{idx}'
            ax.annotate(label, xy=(actual[idx], predicted[idx]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=9, alpha=0.8,
                        arrowprops=dict(arrowstyle='->', alpha=0.5))
        
        # Statistics box
        stats_text = f"{model_name} Performance Metrics:\n"
        stats_text += f"━━━━━━━━━━━━━━━━━━━━━━\n"
        stats_text += f"  R² Score:     {r2:.6f}\n"
        stats_text += f"  MAE:          {mae:.6f}\n"
        stats_text += f"  RMSE:         {rmse:.6f}\n"
        stats_text += f"  Slope:        {z[0]:.6f}\n"
        stats_text += f"  Intercept:    {z[1]:.6f}\n"
        stats_text += f"  N Samples:    {len(actual)}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(min_val - margin, max_val + margin)
        ax.set_ylim(min_val - margin, max_val + margin)
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_residual_analysis(self,
                               actual: np.ndarray,
                               predicted: np.ndarray,
                               model_name: str = 'Model',
                               title: str = 'Residual Analysis',
                               save_name: str = 'ml_residual_analysis.png') -> Optional[str]:
        """
        Single detailed residual analysis chart.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        residuals = actual - predicted
        
        # Color by sign
        colors = ['green' if r >= 0 else 'red' for r in residuals]
        
        scatter = ax.scatter(predicted, residuals, c=colors, s=100, alpha=0.6,
                             edgecolors='black', linewidths=0.5)
        
        # Zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
        
        # ±1 std lines
        std_res = np.std(residuals)
        ax.axhline(y=std_res, color='orange', linestyle='--', linewidth=1.5, 
                   label=f'+1σ ({std_res:.4f})')
        ax.axhline(y=-std_res, color='orange', linestyle='--', linewidth=1.5,
                   label=f'-1σ ({-std_res:.4f})')
        ax.axhline(y=2*std_res, color='red', linestyle=':', linewidth=1.5,
                   label=f'+2σ ({2*std_res:.4f})')
        ax.axhline(y=-2*std_res, color='red', linestyle=':', linewidth=1.5,
                   label=f'-2σ ({-2*std_res:.4f})')
        
        # Fill bands
        ax.fill_between(ax.get_xlim(), -std_res, std_res, alpha=0.1, color='green')
        ax.fill_between(ax.get_xlim(), -2*std_res, 2*std_res, alpha=0.05, color='yellow')
        
        # LOESS smoothing trend
        try:
            from scipy.ndimage import uniform_filter1d
            sorted_idx = np.argsort(predicted)
            smoothed = uniform_filter1d(residuals[sorted_idx], size=max(5, len(residuals)//10))
            ax.plot(predicted[sorted_idx], smoothed, 'b-', linewidth=2.5, 
                    label='Trend (smoothed)', alpha=0.8)
        except:
            pass
        
        # Statistics
        n_outliers = np.sum(np.abs(residuals) > 2 * std_res)
        stats_text = f"Residual Statistics:\n"
        stats_text += f"  Mean:       {np.mean(residuals):.6f}\n"
        stats_text += f"  Std:        {std_res:.6f}\n"
        stats_text += f"  Min:        {np.min(residuals):.6f}\n"
        stats_text += f"  Max:        {np.max(residuals):.6f}\n"
        stats_text += f"  Outliers:   {n_outliers} ({n_outliers/len(residuals)*100:.1f}%)"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_rank_correlation_analysis(self,
                                       actual: np.ndarray,
                                       predicted: np.ndarray,
                                       entity_names: List[str],
                                       model_name: str = 'Model',
                                       title: str = 'Rank Prediction Analysis',
                                       save_name: str = 'ml_rank_correlation.png') -> Optional[str]:
        """
        Single chart analyzing ranking prediction accuracy.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Calculate ranks
        actual_ranks = np.argsort(np.argsort(-actual)) + 1
        predicted_ranks = np.argsort(np.argsort(-predicted)) + 1
        rank_diff = actual_ranks - predicted_ranks
        
        # Spearman correlation
        from scipy.stats import spearmanr
        corr, pvalue = spearmanr(actual_ranks, predicted_ranks)
        
        # Color by rank difference
        abs_diff = np.abs(rank_diff)
        colors = plt.cm.RdYlGn_r(abs_diff / (abs_diff.max() + 1))
        
        scatter = ax.scatter(actual_ranks, predicted_ranks, c=abs_diff, cmap='RdYlGn_r',
                             s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
        
        # Perfect prediction line
        max_rank = len(actual)
        ax.plot([1, max_rank], [1, max_rank], 'k--', linewidth=2.5, 
                label='Perfect Ranking', alpha=0.8)
        
        # ±5 rank tolerance bands
        ax.fill_between([1, max_rank], [1-5, max_rank-5], [1+5, max_rank+5],
                        alpha=0.1, color='green', label='±5 Rank Tolerance')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('|Rank Difference|', fontsize=11)
        
        # Annotate biggest rank changes
        worst_idx = np.argsort(abs_diff)[-5:]
        for idx in worst_idx:
            ax.annotate(f'{entity_names[idx]}\nΔ={rank_diff[idx]:+d}',
                        xy=(actual_ranks[idx], predicted_ranks[idx]),
                        xytext=(15, 15), textcoords='offset points',
                        fontsize=9, alpha=0.9,
                        arrowprops=dict(arrowstyle='->', alpha=0.6),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Statistics
        exact_match = np.sum(rank_diff == 0)
        within_3 = np.sum(abs_diff <= 3)
        within_5 = np.sum(abs_diff <= 5)
        
        stats_text = f"Ranking Performance ({model_name}):\n"
        stats_text += f"━━━━━━━━━━━━━━━━━━━━━━\n"
        stats_text += f"  Spearman ρ:    {corr:.4f}\n"
        stats_text += f"  P-value:       {pvalue:.2e}\n"
        stats_text += f"  Exact Match:   {exact_match}/{len(actual)} ({exact_match/len(actual)*100:.1f}%)\n"
        stats_text += f"  Within ±3:     {within_3}/{len(actual)} ({within_3/len(actual)*100:.1f}%)\n"
        stats_text += f"  Within ±5:     {within_5}/{len(actual)} ({within_5/len(actual)*100:.1f}%)\n"
        stats_text += f"  Max |Δ Rank|:  {abs_diff.max()}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        ax.set_xlabel('Actual Rank', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Rank', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, max_rank + 1)
        ax.set_ylim(0, max_rank + 1)
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_model_convergence_analysis(self,
                                        train_history: Dict[str, List[float]],
                                        title: str = 'Model Convergence Analysis',
                                        save_name: str = 'ml_convergence_analysis.png') -> Optional[str]:
        """
        Single chart showing detailed convergence behavior with multiple metrics.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(train_history)))
        
        for i, (metric_name, values) in enumerate(train_history.items()):
            epochs = np.arange(1, len(values) + 1)
            
            # Normalize for comparison
            norm_values = (np.array(values) - min(values)) / (max(values) - min(values) + 1e-10)
            
            ax.plot(epochs, norm_values, '-', linewidth=2.5, label=metric_name,
                    color=colors[i], marker='o', markersize=4, markerfacecolor='white')
            
            # Convergence point detection
            if len(values) > 10:
                recent_std = np.std(values[-5:])
                overall_std = np.std(values)
                if recent_std < overall_std * 0.1:
                    conv_epoch = len(values) - 5
                    ax.axvline(x=conv_epoch, color=colors[i], linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Metric Value', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add epoch markers
        ax.axvspan(0, len(list(train_history.values())[0]) * 0.1, alpha=0.1, color='yellow',
                   label='Warm-up Phase')
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)
    
    def plot_ensemble_contribution_analysis(self,
                                            base_predictions: Dict[str, np.ndarray],
                                            weights: Dict[str, float],
                                            actual: np.ndarray,
                                            title: str = 'Ensemble Model Contribution Analysis',
                                            save_name: str = 'ml_ensemble_contribution.png') -> Optional[str]:
        """
        Single chart showing contribution of each base model to the ensemble.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        models = list(base_predictions.keys())
        n_models = len(models)
        
        # Calculate R² for each model
        r2_scores = []
        for model_name, preds in base_predictions.items():
            ss_res = np.sum((actual - preds) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r2_scores.append(r2)
        
        # Weighted ensemble
        weighted_preds = np.zeros_like(actual)
        for model_name, preds in base_predictions.items():
            weighted_preds += weights[model_name] * preds
        
        ss_res = np.sum((actual - weighted_preds) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        ensemble_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        models_with_ensemble = models + ['Ensemble']
        r2_with_ensemble = r2_scores + [ensemble_r2]
        weights_with_ensemble = [weights.get(m, 0) for m in models] + [1.0]
        
        x_pos = np.arange(len(models_with_ensemble))
        
        # Create bars with color based on R²
        colors = plt.cm.RdYlGn(np.array(r2_with_ensemble))
        bars = ax.bar(x_pos, r2_with_ensemble, color=colors, edgecolor='black', linewidth=1.5)
        
        # Highlight ensemble bar
        bars[-1].set_edgecolor('gold')
        bars[-1].set_linewidth(3)
        
        # Add weight annotations
        for i, (bar, weight, r2) in enumerate(zip(bars, weights_with_ensemble, r2_with_ensemble)):
            # R² value
            ax.annotate(f'R²={r2:.4f}', xy=(bar.get_x() + bar.get_width()/2, r2),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=10, fontweight='bold')
            # Weight value
            if i < len(models):
                ax.annotate(f'w={weight:.3f}', xy=(bar.get_x() + bar.get_width()/2, 0),
                            xytext=(0, 5), textcoords='offset points',
                            ha='center', fontsize=9, color='gray')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models_with_ensemble, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim(0, max(r2_with_ensemble) * 1.15)
        
        # Improvement indicator
        best_base_r2 = max(r2_scores)
        if ensemble_r2 > best_base_r2:
            improvement = (ensemble_r2 - best_base_r2) / best_base_r2 * 100
            ax.annotate(f'↑ {improvement:.1f}% vs best base',
                        xy=(len(models), ensemble_r2),
                        xytext=(20, 20), textcoords='offset points',
                        fontsize=11, color='green', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='green'))
        
        plt.tight_layout()
        return self._save_figure(fig, save_name)


def create_visualizer(output_dir: str = 'outputs/figures') -> PanelVisualizer:
    """Factory function to create visualizer."""
    return PanelVisualizer(output_dir=output_dir)
