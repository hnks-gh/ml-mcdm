# -*- coding: utf-8 -*-
"""
Enhanced Sensitivity Analysis for ML-MCDM Framework
====================================================

State-of-the-art sensitivity analysis tailored for the hierarchical IFS+ER+Forecasting pipeline.

Features:
- Hierarchical sensitivity (subcriteria → criteria → final ER aggregation)
- IFS uncertainty analysis (membership/non-membership perturbation)
- Temporal stability analysis across years
- Forecast prediction robustness
- Multi-level Monte Carlo simulation
- Belief distribution sensitivity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
import warnings
import functools
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing


def _silence_warnings(func):
    """Scope all warning filters to the duration of *func* only."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return func(*args, **kwargs)
    return wrapper


@dataclass
class SensitivityResult:
    """Comprehensive sensitivity analysis results for the hierarchical MCDM system."""
    
    # Weight sensitivity at different hierarchy levels
    subcriteria_sensitivity: Dict[str, float]  # Per subcriterion
    criteria_sensitivity: Dict[str, float]      # Per criterion
    
    # Rank stability
    rank_stability: Dict[str, float]            # Per province
    top_n_stability: Dict[int, float]           # Top-3, Top-5, Top-10 stability
    
    # Temporal analysis
    temporal_stability: Dict[str, float]        # Year-to-year rank correlation
    temporal_rank_volatility: Dict[str, float]  # Province rank volatility over time
    
    # IFS-specific sensitivity
    ifs_membership_sensitivity: float           # Sensitivity to μ perturbation
    ifs_nonmembership_sensitivity: float        # Sensitivity to ν perturbation
    
    # Forecast robustness
    forecast_sensitivity: Optional[Dict[str, float]] = None  # Feature importance stability
    
    # Overall metrics
    overall_robustness: float = 0.0
    confidence_level: float = 0.95
    
    # Detailed perturbation analysis
    perturbation_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate comprehensive summary report."""
        lines = [
            f"\n{'='*70}",
            "ENHANCED SENSITIVITY ANALYSIS RESULTS",
            f"{'='*70}",
            f"\nOverall Robustness Score: {self.overall_robustness:.4f}",
            f"Confidence Level: {self.confidence_level * 100:.0f}%",
            f"\n{'-'*70}",
            "HIERARCHICAL WEIGHT SENSITIVITY",
            f"{'-'*70}",
            "\nCRITERIA LEVEL (higher = more sensitive):"
        ]
        
        # Criteria sensitivity
        if self.criteria_sensitivity:
            sorted_crit = sorted(self.criteria_sensitivity.items(), 
                                key=lambda x: x[1], reverse=True)
            for criterion, sens in sorted_crit[:5]:
                bar = '[' + '=' * int(sens * 20) + ' ' * (20 - int(sens * 20)) + ']'
                lines.append(f"  {criterion}: {sens:.4f} {bar}")
        
        # Subcriteria sensitivity (top 10)
        lines.extend([
            f"\nSUBCRITERIA LEVEL (top 10 most sensitive):"
        ])
        if self.subcriteria_sensitivity:
            sorted_sub = sorted(self.subcriteria_sensitivity.items(), 
                               key=lambda x: x[1], reverse=True)
            for sub, sens in sorted_sub[:10]:
                lines.append(f"  {sub}: {sens:.4f}")
        
        # IFS sensitivity
        lines.extend([
            f"\n{'-'*70}",
            "IFS UNCERTAINTY SENSITIVITY",
            f"{'-'*70}",
            f"  Membership (μ) perturbation:     {self.ifs_membership_sensitivity:.4f}",
            f"  Non-membership (ν) perturbation: {self.ifs_nonmembership_sensitivity:.4f}",
        ])
        
        # Temporal stability
        if self.temporal_stability:
            lines.extend([
                f"\n{'-'*70}",
                "TEMPORAL STABILITY (year-to-year rank correlation)",
                f"{'-'*70}"
            ])
            for transition, corr in sorted(self.temporal_stability.items()):
                lines.append(f"  {transition}: {corr:.4f}")
        
        # Top-N stability
        lines.extend([
            f"\n{'-'*70}",
            "TOP-N RANKING STABILITY",
            f"{'-'*70}"
        ])
        if self.top_n_stability:
            for n, stability in sorted(self.top_n_stability.items()):
                lines.append(f"  Top-{n:2d}: {stability:6.1%} stable")
        
        # Forecast sensitivity
        if self.forecast_sensitivity:
            lines.extend([
                f"\n{'-'*70}",
                "FORECAST FEATURE IMPORTANCE STABILITY",
                f"{'-'*70}"
            ])
            sorted_feat = sorted(self.forecast_sensitivity.items(), 
                                key=lambda x: x[1], reverse=True)
            for feat, sens in sorted_feat[:5]:
                lines.append(f"  {feat}: {sens:.4f}")
        
        lines.append("=" * 70)
        return "\n".join(lines)


class SensitivityAnalysis:
    """
    State-of-the-art sensitivity analysis for hierarchical IFS+ER MCDM system.
    
    This analyzer understands the full pipeline structure:
    - Subcriteria → Criteria → Final ER aggregation
    - IFS membership/non-membership uncertainty
    - Temporal panel data structure
    - ML forecasting stability
    """
    
    def __init__(self,
                 n_simulations: int = 1000,
                 perturbation_range: float = 0.15,
                 ifs_perturbation: float = 0.10,
                 confidence_level: float = 0.95,
                 seed: int = 42,
                 n_jobs: int = -1):
        """
        Initialize enhanced sensitivity analyzer.
        
        Parameters
        ----------
        n_simulations : int
            Number of Monte Carlo simulations
        perturbation_range : float
            Maximum weight perturbation (±15% default)
        ifs_perturbation : float
            IFS membership/non-membership perturbation (±10% default)
        confidence_level : float
            Confidence level for statistical tests
        seed : int
            Random seed for reproducibility
        n_jobs : int
            Number of parallel jobs (-1 uses all CPU cores)
        """
        self.n_simulations = n_simulations
        self.perturbation_range = perturbation_range
        self.ifs_perturbation = ifs_perturbation
        self.confidence_level = confidence_level
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Determine number of parallel workers
        if n_jobs == -1:
            self.n_jobs = max(1, multiprocessing.cpu_count() - 1)
        elif n_jobs <= 0:
            self.n_jobs = max(1, multiprocessing.cpu_count() + n_jobs)
        else:
            self.n_jobs = min(n_jobs, multiprocessing.cpu_count())
        
        self.use_parallel = self.n_jobs > 1

    def _weights_to_dict(self, weights: Dict, perturbed_array: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Convert weights dict/array to subcriteria Dict[str, float] for ranking pipeline."""
        subcriteria = weights['subcriteria']
        if perturbed_array is not None:
            return dict(zip(subcriteria, perturbed_array))
        return weights['fused_dict']

    @_silence_warnings
    def analyze_full_pipeline(self,
                              panel_data: Any,
                              ranking_pipeline: Any,
                              weights: Dict[str, np.ndarray],
                              ranking_result: Any,
                              forecast_result: Optional[Any] = None) -> SensitivityResult:
        """
        Comprehensive sensitivity analysis of the entire ML-MCDM pipeline.
        
        Parameters
        ----------
        panel_data : PanelData
            Full panel dataset
        ranking_pipeline : HierarchicalRankingPipeline
            Ranking system instance
        weights : Dict
            Weight dictionary with 'fused', 'entropy', etc.
        ranking_result : HierarchicalRankingResult
            Current ranking results
        forecast_result : UnifiedForecastResult, optional
            Forecasting results if available
        
        Returns
        -------
        SensitivityResult
            Comprehensive sensitivity analysis results
        """
        import logging
        _logger = logging.getLogger('ml_mcdm')
        
        if self.use_parallel:
            _logger.info(f"Sensitivity analysis: Using parallel execution with {self.n_jobs} workers")
        else:
            _logger.info("Sensitivity analysis: Using sequential execution")
        
        self.rng = np.random.RandomState(self.seed)  # Reset RNG
        
        # 1. Hierarchical weight sensitivity
        subcriteria_sens, criteria_sens = self._hierarchical_weight_sensitivity(
            panel_data, ranking_pipeline, weights
        )
        
        # 2. Rank stability (Monte Carlo on full pipeline)
        rank_stability, top_n_stability, perturbation_results = self._monte_carlo_pipeline(
            panel_data, ranking_pipeline, weights
        )
        
        # 3. Temporal stability analysis
        temporal_stability, temporal_volatility = self._temporal_stability_analysis(
            panel_data, ranking_pipeline, weights
        )
        
        # 4. IFS sensitivity
        ifs_mu_sens, ifs_nu_sens = self._ifs_uncertainty_sensitivity(
            panel_data, ranking_pipeline, weights
        )
        
        # 5. Forecast sensitivity (if available)
        forecast_sens = None
        if forecast_result is not None:
            forecast_sens = self._forecast_robustness(forecast_result)
        
        # Calculate overall robustness
        overall_robustness = self._calculate_overall_robustness(
            rank_stability, temporal_stability, top_n_stability
        )
        
        # Detailed perturbation analysis
        perturbation_analysis = {
            'simulated_rankings': perturbation_results,
            'mean_rank': perturbation_results.mean(axis=0),
            'std_rank': perturbation_results.std(axis=0),
            'rank_range': perturbation_results.max(axis=0) - perturbation_results.min(axis=0),
        }
        
        return SensitivityResult(
            subcriteria_sensitivity=subcriteria_sens,
            criteria_sensitivity=criteria_sens,
            rank_stability=rank_stability,
            top_n_stability=top_n_stability,
            temporal_stability=temporal_stability,
            temporal_rank_volatility=temporal_volatility,
            ifs_membership_sensitivity=ifs_mu_sens,
            ifs_nonmembership_sensitivity=ifs_nu_sens,
            forecast_sensitivity=forecast_sens,
            overall_robustness=overall_robustness,
            confidence_level=self.confidence_level,
            perturbation_analysis=perturbation_analysis
        )
    
    def _hierarchical_weight_sensitivity(self,
                                         panel_data: Any,
                                         ranking_pipeline: Any,
                                         weights: Dict) -> Tuple[Dict, Dict]:
        """Analyze sensitivity at both hierarchy levels."""
        from ranking import HierarchicalRankingPipeline
        
        base_year = max(panel_data.years)
        base_er_result = ranking_pipeline.rank(
            panel_data, self._weights_to_dict(weights), target_year=base_year
        )
        base_ranking = base_er_result.final_ranking.rank().values

        # Precompute MCDM scores once — fast-path reruns only re-run ER
        _precomp = getattr(base_er_result, 'criterion_method_scores', None)
        _can_fast = (_precomp is not None) and hasattr(ranking_pipeline, 'rank_fast')
        _hier = panel_data.hierarchy
        # Use year-context active provinces (dynamic exclusion): only provinces
        # that have valid data in the base year participate in sensitivity analysis.
        _base_ctx = getattr(panel_data, 'year_contexts', {}).get(base_year)
        _alts = (
            list(_base_ctx.active_provinces)
            if _base_ctx is not None
            else list(panel_data.provinces)
        )
        _n_alts = len(_alts)

        # Criteria-level sensitivity
        criteria_sens = {}
        fused_weights = weights['fused']
        criteria = panel_data.hierarchy.all_criteria
        # Use the active subcriteria list (same ordering as fused_weights array)
        active_subcriteria_list = weights['subcriteria']
        active_sc_index = {sc: j for j, sc in enumerate(active_subcriteria_list)}

        for i, criterion in enumerate(criteria):
            # Find indices within `active_subcriteria_list` for this criterion
            subcrit_indices = []
            for sub in panel_data.hierarchy.criteria_to_subcriteria.get(criterion, []):
                if sub in active_sc_index:
                    subcrit_indices.append(active_sc_index[sub])
            
            if not subcrit_indices:
                continue
            
            rank_changes = []
            
            # Perturb criterion weight (affects all its subcriteria)
            for delta in self.rng.uniform(-self.perturbation_range,
                                          self.perturbation_range, 5):
                if abs(delta) < 0.01:
                    continue

                perturbed_weights = fused_weights.copy()
                for idx in subcrit_indices:
                    perturbed_weights[idx] *= (1 + delta)

                # Renormalize
                perturbed_weights = perturbed_weights / perturbed_weights.sum()
                perts_dict = self._weights_to_dict(weights, perturbed_weights)

                # Fast-path: only re-run ER aggregation (skip 12 MCDM methods)
                try:
                    if _can_fast:
                        _er = ranking_pipeline.rank_fast(
                            precomputed_scores=_precomp,
                            subcriteria_weights=perts_dict,
                            hierarchy=_hier,
                            alternatives=_alts,
                        )
                        new_ranking = _er.final_ranking.rank().values
                    else:
                        perturbed_result = ranking_pipeline.rank(
                            panel_data, perts_dict, target_year=base_year
                        )
                        new_ranking = perturbed_result.final_ranking.rank().values

                    # Measure rank change
                    rank_change = np.mean(np.abs(new_ranking - base_ranking))
                    rank_changes.append(rank_change)
                except Exception:
                    continue
            
            if rank_changes:
                criteria_sens[criterion] = np.mean(rank_changes) / max(_n_alts, 1)
            else:
                criteria_sens[criterion] = 0.0
        
        # Normalize criteria sensitivity
        max_sens = max(criteria_sens.values()) if criteria_sens else 1.0
        if max_sens > 0:
            criteria_sens = {k: v / max_sens for k, v in criteria_sens.items()}
        
        # Subcriteria-level sensitivity (one-at-a-time)
        # Use weights['subcriteria'] — the globally-active subset after
        # dynamic exclusion in the weighting pipeline — not the full
        # hierarchy list which may include all-NaN sub-criteria.
        subcriteria_sens = {}
        subcriteria = weights['subcriteria']
        
        for i, subcriterion in enumerate(subcriteria):
            rank_changes = []

            for delta in self.rng.uniform(-self.perturbation_range,
                                          self.perturbation_range, 3):
                if abs(delta) < 0.01:
                    continue

                perturbed_weights = fused_weights.copy()
                perturbed_weights[i] *= (1 + delta)
                perturbed_weights = perturbed_weights / perturbed_weights.sum()
                perts_dict2 = self._weights_to_dict(weights, perturbed_weights)

                # Fast-path: only re-run ER aggregation
                try:
                    if _can_fast:
                        _er2 = ranking_pipeline.rank_fast(
                            precomputed_scores=_precomp,
                            subcriteria_weights=perts_dict2,
                            hierarchy=_hier,
                            alternatives=_alts,
                        )
                        new_ranking = _er2.final_ranking.rank().values
                    else:
                        perturbed_result = ranking_pipeline.rank(
                            panel_data, perts_dict2, target_year=base_year
                        )
                        new_ranking = perturbed_result.final_ranking.rank().values
                    rank_change = np.mean(np.abs(new_ranking - base_ranking))
                    rank_changes.append(rank_change)
                except Exception:
                    continue
            
            if rank_changes:
                subcriteria_sens[subcriterion] = np.mean(rank_changes) / max(_n_alts, 1)
            else:
                subcriteria_sens[subcriterion] = 0.0
        
        # Normalize subcriteria sensitivity
        max_sens = max(subcriteria_sens.values()) if subcriteria_sens else 1.0
        if max_sens > 0:
            subcriteria_sens = {k: v / max_sens for k, v in subcriteria_sens.items()}
        
        return subcriteria_sens, criteria_sens
    
    def _monte_carlo_pipeline(self,
                              panel_data: Any,
                              ranking_pipeline: Any,
                              weights: Dict) -> Tuple[Dict, Dict, np.ndarray]:
        """Monte Carlo simulation with simultaneous weight perturbations (parallelized)."""
        base_year = max(panel_data.years)
        # Dynamic exclusion: use only active provinces for the base year
        _mc_ctx    = getattr(panel_data, 'year_contexts', {}).get(base_year)
        _alts_mc   = (
            list(_mc_ctx.active_provinces)
            if _mc_ctx is not None
            else list(panel_data.provinces)
        )
        n_provinces = len(_alts_mc)

        # Get base ranking
        base_result = ranking_pipeline.rank(panel_data, self._weights_to_dict(weights), target_year=base_year)
        base_ranking = base_result.final_ranking.rank().values

        # Precompute MCDM scores for fast ER-only reruns
        _precomp_mc  = getattr(base_result, 'criterion_method_scores', None)
        _can_fast_mc = (_precomp_mc is not None) and hasattr(ranking_pipeline, 'rank_fast')
        _hier_mc     = panel_data.hierarchy

        # Monte Carlo simulations — honour the user-specified count in full.
        # The old `min(..., 100)` silently discarded 90% of requested simulations.
        n_sims = max(1, self.n_simulations)
        simulated_rankings = np.zeros((n_sims, n_provinces))
        
        if self.use_parallel:
            # Parallel execution using ThreadPoolExecutor (better for I/O-bound tasks)
            def run_simulation(sim_idx):
                # Create independent RNG for this simulation
                rng = np.random.RandomState(self.seed + sim_idx)
                perturbation = 1 + rng.uniform(
                    -self.perturbation_range,
                    self.perturbation_range,
                    len(weights['fused'])
                )
                perturbed_weights = weights['fused'] * perturbation
                perturbed_weights = perturbed_weights / perturbed_weights.sum()
                perts_mc_par = self._weights_to_dict(weights, perturbed_weights)

                try:
                    if _can_fast_mc:
                        _er_par = ranking_pipeline.rank_fast(
                            precomputed_scores=_precomp_mc,
                            subcriteria_weights=perts_mc_par,
                            hierarchy=_hier_mc,
                            alternatives=_alts_mc,
                        )
                        return sim_idx, _er_par.final_ranking.rank().values
                    else:
                        perturbed_result = ranking_pipeline.rank(
                            panel_data, perts_mc_par, target_year=base_year
                        )
                        return sim_idx, perturbed_result.final_ranking.rank().values
                except Exception as _e:
                    import logging as _log
                    _log.getLogger('ml_mcdm').debug(
                        'MC sim %d failed: %s', sim_idx, _e)
                    return sim_idx, base_ranking  # Fallback to base
            
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {executor.submit(run_simulation, sim): sim for sim in range(n_sims)}
                for future in as_completed(futures):
                    sim_idx, ranking = future.result()
                    simulated_rankings[sim_idx] = ranking
        else:
            # Sequential execution
            _n_failures = 0
            import logging as _log_seq
            _logger_seq = _log_seq.getLogger('ml_mcdm')
            for sim in range(n_sims):
                perturbation = 1 + self.rng.uniform(
                    -self.perturbation_range,
                    self.perturbation_range,
                    len(weights['fused'])
                )
                perturbed_weights = weights['fused'] * perturbation
                perturbed_weights = perturbed_weights / perturbed_weights.sum()

                try:
                    perts_mc_seq = self._weights_to_dict(weights, perturbed_weights)
                    if _can_fast_mc:
                        _er_seq = ranking_pipeline.rank_fast(
                            precomputed_scores=_precomp_mc,
                            subcriteria_weights=perts_mc_seq,
                            hierarchy=_hier_mc,
                            alternatives=_alts_mc,
                        )
                        simulated_rankings[sim] = _er_seq.final_ranking.rank().values
                    else:
                        perturbed_result = ranking_pipeline.rank(
                            panel_data, perts_mc_seq, target_year=base_year
                        )
                        simulated_rankings[sim] = perturbed_result.final_ranking.rank().values
                except Exception as _exc:
                    _n_failures += 1
                    _logger_seq.debug('MC sim %d failed: %s', sim, _exc)
                    simulated_rankings[sim] = base_ranking  # Fallback to base

            if _n_failures > 0:
                _failure_rate = _n_failures / n_sims
                if _failure_rate > 0.20:
                    import warnings as _w
                    _w.warn(
                        f'MC simulation: {_n_failures}/{n_sims} runs failed '
                        f'({_failure_rate:.0%}). Results may be unreliable. '
                        'Enable DEBUG logging for details.',
                        RuntimeWarning, stacklevel=2,
                    )
        
        # Calculate rank stability per province (over active provinces only)
        rank_stability = {}
        for i, province in enumerate(_alts_mc):
            ranks = simulated_rankings[:, i]

            # Stability = 1 - (normalised std dev)
            max_std   = max(n_provinces / 2, 1)
            actual_std = ranks.std()
            stability  = max(0.0, 1.0 - actual_std / max_std)

            rank_stability[province] = stability
        
        # Top-N stability
        top_n_stability = {}
        for n in [3, 5, 10]:
            base_top_n = set(np.argsort(base_ranking)[:n])
            matches = 0
            
            for sim_ranking in simulated_rankings:
                sim_top_n = set(np.argsort(sim_ranking)[:n])
                if sim_top_n == base_top_n:
                    matches += 1
            
            top_n_stability[n] = matches / len(simulated_rankings)
        
        return rank_stability, top_n_stability, simulated_rankings
    
    def _temporal_stability_analysis(self,
                                     panel_data: Any,
                                     ranking_pipeline: Any,
                                     weights: Dict) -> Tuple[Dict, Dict]:
        """Analyze ranking stability across years."""
        years = sorted(panel_data.years)
        
        if len(years) < 3:
            return {}, {}
        
        # Get rankings for multiple years (subsample for efficiency).
        # Store the full final_ranking Series (indexed by province) so we can
        # later align to the common-province intersection before computing
        # Spearman correlations — different active-province counts per year
        # would otherwise cause dimension mismatches in spearmanr().
        year_ranking_series: Dict[int, pd.Series] = {}
        sample_years = years[-5:] if len(years) > 5 else years

        for year in sample_years:
            try:
                result = ranking_pipeline.rank(
                    panel_data, self._weights_to_dict(weights), target_year=year
                )
                year_ranking_series[year] = result.final_ranking  # pd.Series
            except Exception:
                continue

        if len(year_ranking_series) < 2:
            return {}, {}

        sorted_years = sorted(year_ranking_series.keys())

        # ------------------------------------------------------------------
        # Compute common-province intersection first — used for both the
        # Spearman correlation (size alignment) and volatility analysis.
        # ------------------------------------------------------------------
        common_provinces = None
        for yr in sorted_years:
            ctx_yr = getattr(panel_data, 'year_contexts', {}).get(yr)
            yr_provs = (
                set(ctx_yr.active_provinces)
                if ctx_yr is not None
                else set(panel_data.provinces)
            )
            common_provinces = (
                yr_provs if common_provinces is None
                else common_provinces & yr_provs
            )
        common_provinces = sorted(common_provinces or panel_data.provinces)

        # Helper: map a Series to a fixed province list (NaN for absent)
        def _reindex_ranking(ranking_series: pd.Series,
                             target_provinces: List[str]) -> np.ndarray:
            """Reindex a ranking Series to a fixed province list."""
            return np.array([
                float(ranking_series.get(p, np.nan))
                for p in target_provinces
            ])

        # Build aligned rank arrays once, reused for both correlation and volatility
        aligned_rankings: Dict[int, np.ndarray] = {
            yr: _reindex_ranking(year_ranking_series[yr], common_provinces)
            for yr in sorted_years
        }

        # Year-to-year rank correlation — Spearman on common-province-aligned
        # arrays so both inputs always have identical length.
        from scipy.stats import spearmanr
        temporal_stability = {}
        for i in range(len(sorted_years) - 1):
            y1, y2 = sorted_years[i], sorted_years[i + 1]
            a1 = aligned_rankings[y1]
            a2 = aligned_rankings[y2]
            # Mask provinces missing in either year
            valid_mask = ~(np.isnan(a1) | np.isnan(a2))
            if valid_mask.sum() >= 3:
                corr, _ = spearmanr(a1[valid_mask], a2[valid_mask])
            else:
                corr = 0.0
            temporal_stability[f"{y1}-{y2}"] = float(np.nan_to_num(corr))

        # Province-level rank volatility over time
        temporal_volatility = {}
        ranking_matrix = np.array(
            [aligned_rankings.get(yr, np.full(len(common_provinces), np.nan))
             for yr in sorted_years]
        )
        max_volatility = max(len(common_provinces) / 2, 1)
        for i, province in enumerate(common_provinces):
            prov_ranks = ranking_matrix[:, i]
            valid_ranks = prov_ranks[~np.isnan(prov_ranks)]
            if len(valid_ranks) >= 2:
                volatility = valid_ranks.std() / max_volatility
            else:
                volatility = 0.0
            temporal_volatility[province] = float(volatility)

        return temporal_stability, temporal_volatility
    
    def _ifs_uncertainty_sensitivity(self,
                                     panel_data: Any,
                                     ranking_pipeline: Any,
                                     weights: Dict) -> Tuple[float, float]:
        """
        Test sensitivity to IFS membership/non-membership uncertainty.

        Performs Monte Carlo perturbation of the IFS decision matrices
        and measures the resulting rank displacement.

        For each simulation:
        1. Build the base IFS matrices (one per criterion group).
        2. Perturb μ values by U(-δ_μ, δ_μ) → re-rank → record displacement.
        3. Perturb ν values by U(-δ_ν, δ_ν) → re-rank → record displacement.

        Sensitivity = mean(|Δrank|) / n_provinces, normalised to [0, 1].

        Returns
        -------
        mu_sensitivity : float
            Normalised mean rank displacement under μ perturbation.
        nu_sensitivity : float
            Normalised mean rank displacement under ν perturbation.
        """
        import logging
        _logger = logging.getLogger('ml_mcdm')

        base_year    = max(panel_data.years)
        _ifs_ctx     = getattr(panel_data, 'year_contexts', {}).get(base_year)
        n_provinces  = len(
            _ifs_ctx.active_provinces
            if _ifs_ctx is not None
            else panel_data.provinces
        )
        n_sims = min(self.n_simulations, 15)  # cap for cost
        delta  = self.ifs_perturbation  # default 0.10

        # ── Obtain base ranking & base IFS matrices ──
        try:
            base_result  = ranking_pipeline.rank(
                panel_data, self._weights_to_dict(weights), target_year=base_year)
            base_ranking = base_result.final_ranking.rank().values
        except Exception as e:
            _logger.warning(f"IFS sensitivity: base ranking failed ({e}), "
                            "returning zero sensitivity.")
            return 0.0, 0.0

        # Build base IFS matrices per criterion using clean criterion matrices
        from mcdm.ifs.base import IFSDecisionMatrix
        hierarchy      = panel_data.hierarchy
        historical_std = ranking_pipeline._compute_historical_std(panel_data)
        global_range   = ranking_pipeline._compute_global_range(panel_data)

        base_ifs: Dict[str, IFSDecisionMatrix] = {}
        for crit_id in sorted(hierarchy.all_criteria):
            # Use get_criterion_matrix for NaN-free data (respects YearContext)
            df_crit = panel_data.get_criterion_matrix(base_year, crit_id)
            if df_crit.empty:
                continue
            subcrit_cols = df_crit.columns.tolist()
            cost_local   = [c for c in subcrit_cols
                            if c in ranking_pipeline.cost_criteria]
            df_norm      = ranking_pipeline._minmax_normalize(
                df_crit, cost_criteria=cost_local)
            std_crit = (historical_std[subcrit_cols].copy()
                        if all(sc in historical_std.columns
                               for sc in subcrit_cols)
                        else pd.DataFrame(
                            0.0, index=df_crit.index, columns=subcrit_cols))
            range_crit = (global_range[subcrit_cols]
                          if all(sc in global_range.index
                                 for sc in subcrit_cols)
                          else pd.Series(1.0, index=subcrit_cols))

            base_ifs[crit_id] = IFSDecisionMatrix.from_temporal_variance(
                current_data=df_norm,
                historical_std=std_crit,
                global_range=range_crit,
                spread_factor=ranking_pipeline.ifs_spread_factor,
            )

        if not base_ifs:
            _logger.warning("IFS sensitivity: no IFS matrices built.")
            return 0.0, 0.0

        # ── Monte Carlo: μ perturbation (parallelized) ──
        mu_displacements: List[float] = []
        
        if self.use_parallel:
            def run_mu_sim(sim_idx):
                rng = np.random.RandomState(self.seed + sim_idx)
                overrides = {
                    cid: mat.perturb(delta_mu=delta, delta_nu=0.0, rng=rng)
                    for cid, mat in base_ifs.items()
                }
                try:
                    res = ranking_pipeline.rank(
                        panel_data, self._weights_to_dict(weights),
                        target_year=base_year,
                        ifs_overrides=overrides)
                    new_ranking = res.final_ranking.rank().values
                    return np.mean(np.abs(new_ranking - base_ranking))
                except Exception:
                    return None
            
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(run_mu_sim, i) for i in range(n_sims)]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        mu_displacements.append(result)
        else:
            for sim_idx in range(n_sims):
                rng = np.random.RandomState(self.seed + sim_idx)
                overrides = {
                    cid: mat.perturb(delta_mu=delta, delta_nu=0.0, rng=rng)
                    for cid, mat in base_ifs.items()
                }
                try:
                    res = ranking_pipeline.rank(
                        panel_data, self._weights_to_dict(weights),
                        target_year=base_year,
                        ifs_overrides=overrides)
                    new_ranking = res.final_ranking.rank().values
                    mu_displacements.append(
                        np.mean(np.abs(new_ranking - base_ranking)))
                except Exception:
                    continue

        # ── Monte Carlo: ν perturbation (parallelized) ──
        nu_displacements: List[float] = []
        
        if self.use_parallel:
            def run_nu_sim(sim_idx):
                rng = np.random.RandomState(self.seed + sim_idx + 1000)  # Different seed offset
                overrides = {
                    cid: mat.perturb(delta_mu=0.0, delta_nu=delta, rng=rng)
                    for cid, mat in base_ifs.items()
                }
                try:
                    res = ranking_pipeline.rank(
                        panel_data, self._weights_to_dict(weights),
                        target_year=base_year,
                        ifs_overrides=overrides)
                    new_ranking = res.final_ranking.rank().values
                    return np.mean(np.abs(new_ranking - base_ranking))
                except Exception:
                    return None
            
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(run_nu_sim, i) for i in range(n_sims)]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        nu_displacements.append(result)
        else:
            for sim_idx in range(n_sims):
                rng = np.random.RandomState(self.seed + sim_idx + 1000)
                overrides = {
                    cid: mat.perturb(delta_mu=0.0, delta_nu=delta, rng=rng)
                    for cid, mat in base_ifs.items()
                }
                try:
                    res = ranking_pipeline.rank(
                        panel_data, self._weights_to_dict(weights),
                        target_year=base_year,
                        ifs_overrides=overrides)
                    new_ranking = res.final_ranking.rank().values
                    nu_displacements.append(
                        np.mean(np.abs(new_ranking - base_ranking)))
                except Exception:
                    continue

        # Normalise: displacement / (n_provinces / 2)  → [0, 1]
        max_disp = n_provinces / 2.0
        mu_sensitivity = (float(np.mean(mu_displacements)) / max_disp
                          if mu_displacements else 0.0)
        nu_sensitivity = (float(np.mean(nu_displacements)) / max_disp
                          if nu_displacements else 0.0)

        _logger.info(f"IFS sensitivity ({n_sims} sims): "
                     f"μ={mu_sensitivity:.4f}, ν={nu_sensitivity:.4f}")

        return mu_sensitivity, nu_sensitivity
    
    def _forecast_robustness(self, forecast_result: Any) -> Dict[str, float]:
        """Analyze forecast feature importance stability via bootstrap resampling.
        
        Bootstraps the feature importance scores across models to measure
        the coefficient of variation (CV) of each feature's importance.
        Lower CV means more stable/reliable importance.
        """
        if not hasattr(forecast_result, 'feature_importance'):
            return {}
        
        feature_importance = forecast_result.feature_importance
        if isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
            # Feature importance DataFrame: rows=features, cols=components or models
            # Bootstrap resample columns to estimate importance stability
            n_cols = feature_importance.shape[1]
            if n_cols < 2:
                # Not enough columns to bootstrap — return mean importance
                return feature_importance.iloc[:, 0].to_dict()
            
            n_bootstrap = min(100, self.n_simulations)
            bootstrap_means = []
            for _ in range(n_bootstrap):
                # Resample columns with replacement
                col_indices = self.rng.choice(n_cols, size=n_cols, replace=True)
                resampled = feature_importance.iloc[:, col_indices]
                bootstrap_means.append(resampled.mean(axis=1).values)
            
            bootstrap_matrix = np.array(bootstrap_means)  # (n_bootstrap, n_features)
            overall_mean = np.mean(bootstrap_matrix, axis=0)
            overall_std = np.std(bootstrap_matrix, axis=0)
            
            # CV = std / mean (coefficient of variation)
            feature_names = feature_importance.index.tolist()
            feature_cv = {}
            for i, feat in enumerate(feature_names):
                if abs(overall_mean[i]) > 1e-10:
                    feature_cv[feat] = float(overall_std[i] / abs(overall_mean[i]))
                else:
                    feature_cv[feat] = 0.0
            
            return feature_cv
        
        return {}
    
    def _calculate_overall_robustness(self,
                                      rank_stability: Dict,
                                      temporal_stability: Dict,
                                      top_n_stability: Dict) -> float:
        """Calculate composite robustness score."""
        scores = []
        
        # Weight different stability components
        if rank_stability:
            scores.append(np.mean(list(rank_stability.values())))
        
        if temporal_stability:
            scores.append(np.mean(list(temporal_stability.values())))
        
        if top_n_stability:
            scores.append(top_n_stability.get(5, 0.5))  # Top-5 stability
        
        if scores:
            return np.mean(scores)
        else:
            return 0.5  # Neutral robustness if no data


def run_sensitivity_analysis(
    panel_data: Any,
    ranking_pipeline: Any,
    weights: Dict,
    ranking_result: Any,
    forecast_result: Optional[Any] = None,
    n_simulations: int = 1000
) -> SensitivityResult:
    """
    Convenience function for enhanced sensitivity analysis.
    
    Parameters
    ----------
    panel_data : PanelData
        Panel dataset
    ranking_pipeline : HierarchicalRankingPipeline
        Ranking system
    weights : Dict
        Weight dictionary
    ranking_result : HierarchicalRankingResult
        Current ranking
    forecast_result : UnifiedForecastResult, optional
        Forecast results
    n_simulations : int
        Number of Monte Carlo simulations
    
    Returns
    -------
    SensitivityResult
        Comprehensive sensitivity analysis
    """
    analyzer = SensitivityAnalysis(n_simulations=n_simulations)
    return analyzer.analyze_full_pipeline(
        panel_data, ranking_pipeline, weights, ranking_result, forecast_result
    )

