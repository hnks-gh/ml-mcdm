# -*- coding: utf-8 -*-
"""
Convergence Analysis
====================

Beta and Sigma convergence tests for panel data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConvergenceResult:
    """Result container for convergence analysis."""
    # Beta convergence
    beta_coefficient: float
    beta_std_error: float
    beta_t_stat: float
    beta_p_value: float
    convergence_speed: float      # Speed of convergence
    half_life: float              # Years to halve the gap
    beta_converging: bool         # True if significant negative beta
    
    # Sigma convergence
    sigma_by_year: Dict[int, float]
    sigma_trend: float            # Linear trend coefficient
    sigma_converging: bool        # True if declining dispersion
    
    # Conditional convergence (with controls)
    conditional_beta: Optional[float] = None
    conditional_r2: Optional[float] = None
    
    # Club convergence
    clubs: Optional[Dict[str, List[str]]] = None
    n_clubs: int = 0
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "CONVERGENCE ANALYSIS RESULTS",
            f"{'='*60}",
            f"\n{'─'*30}",
            "BETA (β) CONVERGENCE",
            f"{'─'*30}",
            f"Beta coefficient: {self.beta_coefficient:.4f}",
            f"Standard error: {self.beta_std_error:.4f}",
            f"t-statistic: {self.beta_t_stat:.4f}",
            f"p-value: {self.beta_p_value:.4f}",
            f"Convergence speed: {self.convergence_speed:.4f}",
            f"Half-life (years): {self.half_life:.2f}",
            f"Converging: {'YES ✓' if self.beta_converging else 'NO ✗'}",
            f"\n{'─'*30}",
            "SIGMA (σ) CONVERGENCE",
            f"{'─'*30}",
            f"Sigma by year:"
        ]
        for year, sigma in self.sigma_by_year.items():
            lines.append(f"  {year}: {sigma:.4f}")
        lines.extend([
            f"Sigma trend: {self.sigma_trend:.6f}",
            f"Converging: {'YES ✓' if self.sigma_converging else 'NO ✗'}"
        ])
        
        if self.conditional_beta is not None:
            lines.extend([
                f"\n{'─'*30}",
                "CONDITIONAL CONVERGENCE",
                f"{'─'*30}",
                f"Conditional beta: {self.conditional_beta:.4f}",
                f"R²: {self.conditional_r2:.4f}"
            ])
        
        if self.clubs:
            lines.extend([
                f"\n{'─'*30}",
                "CLUB CONVERGENCE",
                f"{'─'*30}",
                f"Number of clubs: {self.n_clubs}"
            ])
            for club_name, members in self.clubs.items():
                lines.append(f"  {club_name}: {len(members)} provinces")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class ConvergenceAnalysis:
    """
    Convergence analysis for panel data sustainability scores.
    
    Tests for:
    - Absolute beta convergence (unconditional)
    - Conditional beta convergence
    - Sigma convergence
    - Club convergence
    """
    
    def __init__(self,
                 significance_level: float = 0.05,
                 min_periods: int = 3):
        """
        Initialize convergence analysis.
        
        Parameters
        ----------
        significance_level : float
            Significance level for hypothesis tests
        min_periods : int
            Minimum number of time periods required
        """
        self.significance_level = significance_level
        self.min_periods = min_periods
    
    def analyze(self,
               scores: pd.DataFrame,
               entity_col: str = 'province',
               time_col: str = 'year',
               score_col: str = 'score',
               control_cols: Optional[List[str]] = None) -> ConvergenceResult:
        """
        Perform comprehensive convergence analysis.
        
        Parameters
        ----------
        scores : pd.DataFrame
            Panel data with sustainability scores
        entity_col : str
            Entity identifier column
        time_col : str
            Time period column
        score_col : str
            Score column to analyze
        control_cols : List[str], optional
            Control variables for conditional convergence
        """
        # Normalize column names for case-insensitive matching
        col_mapping = {c.lower(): c for c in scores.columns}
        entity_col_actual = col_mapping.get(entity_col.lower(), entity_col)
        time_col_actual = col_mapping.get(time_col.lower(), time_col)
        score_col_actual = col_mapping.get(score_col.lower(), score_col)
        
        # Validate columns exist
        if entity_col_actual not in scores.columns:
            # Try common alternatives
            for alt in ['Province', 'province', 'Entity', 'entity', 'ID', 'id']:
                if alt in scores.columns:
                    entity_col_actual = alt
                    break
            else:
                raise ValueError(f"Entity column '{entity_col}' not found in data")
        
        if time_col_actual not in scores.columns:
            for alt in ['Year', 'year', 'Time', 'time', 'Period', 'period']:
                if alt in scores.columns:
                    time_col_actual = alt
                    break
            else:
                raise ValueError(f"Time column '{time_col}' not found in data")
        
        if score_col_actual not in scores.columns:
            for alt in ['Score', 'score', 'Value', 'value']:
                if alt in scores.columns:
                    score_col_actual = alt
                    break
            else:
                raise ValueError(f"Score column '{score_col}' not found in data")
        
        # Reshape to wide format for analysis
        try:
            wide = scores.pivot(index=entity_col_actual, columns=time_col_actual, values=score_col_actual)
        except Exception as e:
            # Handle duplicate entries by aggregating
            wide = scores.groupby([entity_col_actual, time_col_actual])[score_col_actual].mean().unstack()
        
        years = sorted(wide.columns)
        
        if len(years) < self.min_periods:
            # Return default result with available data
            sigma_result = self._sigma_convergence(wide, years) if len(years) >= 2 else {'sigma_by_year': {}, 'trend': 0, 'converging': False}
            return ConvergenceResult(
                beta_coefficient=0.0,
                beta_std_error=0.0,
                beta_t_stat=0.0,
                beta_p_value=1.0,
                convergence_speed=0.0,
                half_life=float('inf'),
                beta_converging=False,
                sigma_by_year=sigma_result['sigma_by_year'],
                sigma_trend=sigma_result['trend'],
                sigma_converging=sigma_result['converging'],
                conditional_beta=None,
                conditional_r2=None,
                clubs=None,
                n_clubs=0
            )
        
        # Beta convergence
        try:
            beta_result = self._beta_convergence(wide, years)
        except Exception as e:
            beta_result = {'beta': 0.0, 'std_error': 0.0, 't_stat': 0.0, 'p_value': 1.0,
                          'speed': 0.0, 'half_life': float('inf'), 'converging': False}
        
        # Sigma convergence
        try:
            sigma_result = self._sigma_convergence(wide, years)
        except Exception as e:
            sigma_result = {'sigma_by_year': {y: 0 for y in years}, 'trend': 0, 'converging': False}
        
        # Conditional convergence
        if control_cols:
            try:
                cond_result = self._conditional_convergence(
                    scores, entity_col_actual, time_col_actual, score_col_actual, control_cols
                )
            except Exception:
                cond_result = (None, None)
        else:
            cond_result = (None, None)
        
        # Club convergence
        try:
            clubs, n_clubs = self._club_convergence(wide, years)
        except Exception:
            clubs, n_clubs = None, 0
        
        return ConvergenceResult(
            beta_coefficient=beta_result['beta'],
            beta_std_error=beta_result['std_error'],
            beta_t_stat=beta_result['t_stat'],
            beta_p_value=beta_result['p_value'],
            convergence_speed=beta_result['speed'],
            half_life=beta_result['half_life'],
            beta_converging=beta_result['converging'],
            sigma_by_year=sigma_result['sigma_by_year'],
            sigma_trend=sigma_result['trend'],
            sigma_converging=sigma_result['converging'],
            conditional_beta=cond_result[0],
            conditional_r2=cond_result[1],
            clubs=clubs,
            n_clubs=n_clubs
        )
    
    def _beta_convergence(self, wide: pd.DataFrame, 
                         years: List) -> Dict:
        """
        Test for absolute beta convergence.
        
        Model: g_i = α + β * ln(y_{i,0}) + ε_i
        where g_i = average growth rate
        
        Negative beta indicates convergence.
        """
        T = len(years)
        y_initial = wide[years[0]].values
        y_final = wide[years[-1]].values
        
        # Log transformation (add small constant to avoid log(0))
        ln_y0 = np.log(y_initial + 0.001)
        
        # Average annual growth rate
        growth_rate = (y_final - y_initial) / (T - 1)
        
        # OLS regression: growth ~ ln(initial)
        X = np.column_stack([np.ones(len(ln_y0)), ln_y0])
        y = growth_rate
        
        # Remove any NaN/inf
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        
        # OLS estimation
        XtX_inv = np.linalg.inv(X.T @ X + 1e-8 * np.eye(2))
        coeffs = XtX_inv @ X.T @ y
        
        alpha, beta = coeffs
        
        # Residuals and standard errors
        y_pred = X @ coeffs
        residuals = y - y_pred
        n = len(y)
        
        sigma2 = np.sum(residuals ** 2) / (n - 2)
        var_beta = sigma2 * XtX_inv
        std_errors = np.sqrt(np.diag(var_beta))
        
        # t-statistic and p-value
        t_stat = beta / (std_errors[1] + 1e-10)
        
        # Two-tailed p-value from t-distribution
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        # Convergence speed: β = -(1 - e^{-λT})/T
        # Solving for λ: λ = -ln(1 + βT) / T
        if beta < 0:
            speed = -np.log(1 + beta * T) / T if (1 + beta * T) > 0 else 0.02
        else:
            speed = 0
        
        # Half-life: t* = ln(2) / λ
        half_life = np.log(2) / speed if speed > 0 else np.inf
        
        # Test for convergence (significant negative beta)
        converging = (beta < 0) and (p_value < self.significance_level)
        
        return {
            'beta': beta,
            'std_error': std_errors[1],
            't_stat': t_stat,
            'p_value': p_value,
            'speed': speed,
            'half_life': half_life,
            'converging': converging
        }
    
    def _sigma_convergence(self, wide: pd.DataFrame, 
                          years: List) -> Dict:
        """
        Test for sigma convergence.
        
        Measures: Decline in cross-sectional dispersion over time.
        """
        # Calculate coefficient of variation for each year
        sigma_by_year = {}
        sigmas = []
        
        for year in years:
            values = wide[year].dropna().values
            if len(values) > 1:
                # Coefficient of variation
                cv = np.std(values) / (np.mean(values) + 1e-10)
                sigma_by_year[year] = cv
                sigmas.append(cv)
        
        # Linear trend in sigma
        if len(sigmas) > 1:
            time_idx = np.arange(len(sigmas))
            trend = np.polyfit(time_idx, sigmas, 1)[0]
        else:
            trend = 0
        
        # Converging if negative trend
        converging = trend < 0
        
        return {
            'sigma_by_year': sigma_by_year,
            'trend': trend,
            'converging': converging
        }
    
    def _conditional_convergence(self, scores: pd.DataFrame,
                                 entity_col: str,
                                 time_col: str,
                                 score_col: str,
                                 control_cols: List[str]) -> Tuple[float, float]:
        """
        Test for conditional convergence with control variables.
        
        Model: g_i = α + β * ln(y_{i,0}) + γ'X_i + ε_i
        """
        # Get wide format and control averages
        wide = scores.pivot(index=entity_col, columns=time_col, values=score_col)
        years = sorted(wide.columns)
        
        T = len(years)
        y_initial = wide[years[0]].values
        y_final = wide[years[-1]].values
        
        ln_y0 = np.log(y_initial + 0.001)
        growth_rate = (y_final - y_initial) / (T - 1)
        
        # Average control variables per entity
        controls_avg = scores.groupby(entity_col)[control_cols].mean()
        
        # Align indices
        common_idx = wide.index.intersection(controls_avg.index)
        
        # Build design matrix
        X_base = ln_y0
        X_controls = controls_avg.loc[common_idx].values
        X = np.column_stack([np.ones(len(common_idx)), X_base[:len(common_idx)], X_controls])
        y = growth_rate[:len(common_idx)]
        
        # Remove NaN
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        
        # OLS
        XtX_inv = np.linalg.inv(X.T @ X + 1e-8 * np.eye(X.shape[1]))
        coeffs = XtX_inv @ X.T @ y
        
        conditional_beta = coeffs[1]  # Coefficient on ln(y0)
        
        # R²
        y_pred = X @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        
        return conditional_beta, r2
    
    def _club_convergence(self, wide: pd.DataFrame, 
                         years: List) -> Tuple[Dict[str, List], int]:
        """
        Test for club convergence using clustering.
        
        Identifies groups (clubs) of provinces converging to different
        steady states.
        """
        # Use growth trajectories for clustering
        growth_matrix = wide.diff(axis=1).iloc[:, 1:].values
        
        # Remove NaN rows
        mask = ~np.isnan(growth_matrix).any(axis=1)
        clean_growth = growth_matrix[mask]
        entities = wide.index[mask].tolist()
        
        if len(entities) < 4:
            return None, 0
        
        # Simple k-means clustering
        n_clusters = min(4, len(entities) // 5)
        n_clusters = max(2, n_clusters)
        
        clubs = self._kmeans_clustering(clean_growth, entities, n_clusters)
        
        return clubs, len(clubs)
    
    def _kmeans_clustering(self, data: np.ndarray, 
                          labels: List[str],
                          k: int) -> Dict[str, List[str]]:
        """Simple k-means clustering."""
        n_samples = data.shape[0]
        
        # Initialize centroids randomly
        np.random.seed(42)
        idx = np.random.choice(n_samples, k, replace=False)
        centroids = data[idx]
        
        # Iterate
        for _ in range(50):
            # Assign clusters
            distances = np.zeros((n_samples, k))
            for i in range(k):
                distances[:, i] = np.sqrt(np.sum((data - centroids[i]) ** 2, axis=1))
            
            cluster_assignments = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = cluster_assignments == i
                if mask.sum() > 0:
                    new_centroids[i] = data[mask].mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        # Create club dictionary
        clubs = {}
        for i in range(k):
            mask = cluster_assignments == i
            members = [labels[j] for j in range(n_samples) if mask[j]]
            if members:
                clubs[f"Club_{i+1}"] = members
        
        return clubs


def test_convergence(scores: pd.DataFrame,
                    entity_col: str = 'province',
                    time_col: str = 'year',
                    score_col: str = 'score') -> ConvergenceResult:
    """Convenience function for convergence analysis."""
    analyzer = ConvergenceAnalysis()
    return analyzer.analyze(scores, entity_col, time_col, score_col)
