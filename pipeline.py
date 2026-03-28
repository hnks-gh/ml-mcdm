# -*- coding: utf-8 -*-
"""
ML-MCDM Pipeline Orchestrator
==============================

Seven-phase production pipeline:

  Phase 1  Data Loading
  Phase 2  Weight Calculation       (Hybrid Weighting — two-level MC Ensemble)
  Phase 3  Hierarchical Ranking     (6 traditional MCDM methods)
  Phase 4  ML Forecasting           (base models + Meta-Learner + Conformal)
  Phase 5  Sensitivity Analysis     (Hierarchical multi-level robustness)
  Phase 6  Visualization            (high-resolution figures)
  Phase 7  Result Export            (CSV / JSON / text report)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import warnings
import time
import json
import traceback

# Internal imports (support both package and direct execution)
try:
    from .config import Config, get_default_config
    from .loggers import setup_logging
    from .loggers.context import PhaseMetrics
    from .data import DataLoader, PanelData, HierarchyMapping, YearContext
    from .ranking import TOPSISCalculator
    from .ranking import HierarchicalRankingPipeline, HierarchicalRankingResult
    from .analysis import MLSensitivityAnalysis, CombinedSensitivityResult
    from .output.visualization import VisualizationOrchestrator
    from .output import OutputOrchestrator
except ImportError:
    from config import Config, get_default_config
    from loggers import setup_logging
    from loggers.context import PhaseMetrics
    from data import DataLoader, PanelData, HierarchyMapping, YearContext
    from ranking import TOPSISCalculator
    from ranking import HierarchicalRankingPipeline, HierarchicalRankingResult
    from analysis import MLSensitivityAnalysis, CombinedSensitivityResult
    from output.visualization import VisualizationOrchestrator
    from output import OutputOrchestrator


def _to_array(x) -> np.ndarray:
    """Convert Series / list / array to plain ndarray."""
    if x is None:
        return np.array([])
    if hasattr(x, 'values'):
        return x.values
    return np.asarray(x)


# =========================================================================
# Result container
# =========================================================================

@dataclass
class PipelineResult:
    """Container for all pipeline results."""
    # Data
    panel_data: PanelData
    decision_matrix: np.ndarray   # latest-year snapshot (avail_provs × avail_scs);
                                  # may contain NaN for Type 1/3 structural gaps —
                                  # no imputation applied.  MCDM phases use
                                  # panel_data directly and are unaffected.

    # Weights (new — two-level hierarchical MC ensemble)
    sc_weights: np.ndarray               # global SC weights array (p,)
    criterion_weights_dict: Dict[str, float]  # {C01..C08: float}, sums to 1
    weight_details: Dict[str, Any]       # full WeightResult.details dict

    # Hierarchical Ranking (Traditional MCDM)
    # Optional so the pipeline can return partial results if this phase fails
    ranking_result: Optional[HierarchicalRankingResult] = None

    # Multi-year rankings — {year: HierarchicalRankingResult} for all panel years.
    # Present when config.ranking.run_all_years is True.
    multi_year_results: Dict[int, Any] = None  # type: ignore[assignment]

    # ML Forecasting (UnifiedForecaster)
    forecast_result: Optional[Any] = None

    # Analysis
    sensitivity_result: Any = None

    # MC diagnostics (Level 2 mc_diagnostics from weight_details)
    mc_ensemble_diagnostics: Optional[Dict] = None

    # Meta
    execution_time: float = 0.0
    config: Optional[Config] = None

    def __post_init__(self):
        if self.multi_year_results is None:
            self.multi_year_results = {}

    # ---- convenience accessors ----

    def get_final_ranking_df(self) -> pd.DataFrame:
        """Sorted DataFrame with province, rank, score."""
        if self.ranking_result is None or getattr(self.ranking_result, 'final_ranking', None) is None:
            return pd.DataFrame(columns=['Province', 'Score', 'Rank'])
            
        return pd.DataFrame({
            'Province': self.ranking_result.final_ranking.index,
            'Score': self.ranking_result.final_scores.values,
            'Rank':  self.ranking_result.final_ranking.values,
        }).sort_values('Rank').reset_index(drop=True)


# =========================================================================
# Pipeline
# =========================================================================

class MLMCDMPipeline:
    """
    Production-grade ML-MCDM pipeline for panel data analysis.

    Integrates
    ----------
    * Two-Level Deterministic CRITIC Weighting (NaN-aware, adaptive)
    * 6 Traditional MCDM Methods per Criterion Group:
        TOPSIS, VIKOR, PROMETHEE II, COPRAS, EDAS, SAW
    * Criterion-Weighted Aggregation with Composite Scoring
    * ML Forecasting: 4-Model Ensemble (CatBoost, Bayesian Ridge, SVR, ElasticNet)
        + Super Learner Meta-Ensemble + Conformal Prediction (optional)
    * Hierarchical Sensitivity Analysis & Temporal Stability Assessment
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_default_config()

        # Output directories
        self._setup_output_directory()

        # Logging (console + debug JSON)
        self.console, self.debug_log = setup_logging(self.config.output_dir)
        # stdlib-compatible logger used by submodules
        import logging as _logging
        self.logger = _logging.getLogger('ml_mcdm')

        # Output orchestrator (CSV + Markdown report)
        self.output_orch = OutputOrchestrator(
            base_output_dir=self.config.output_dir,
        )

        # Visualization orchestrator
        self.visualizer = VisualizationOrchestrator(
            output_dir=str(Path(self.config.output_dir) / 'figures'),
            dpi=self.config.visualization.dpi,
            ranking_top_n=self.config.visualization.ranking_top_n,
        )

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------

    def _setup_output_directory(self) -> None:
        out = Path(self.config.output_dir)
        phases = ('weighting', 'ranking', 'mcdm', 'forecasting', 'sensitivity')
        for top in ('figures', 'csv', 'reports', 'logs'):
            (out / top).mkdir(parents=True, exist_ok=True)
        for phase in phases:
            (out / 'figures' / phase).mkdir(parents=True, exist_ok=True)
            (out / 'csv' / phase).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    def run(self, data_path: Optional[str] = None) -> PipelineResult:
        """Execute full analysis pipeline and return results."""
        start_time = time.time()

        try:
            self.console.banner('ML-MCDM')

            # Phase 1: Data Loading
            with self.console.phase('Data Loading') as ph:
                panel_data = self._load_data()
                ph.detail(f'{len(panel_data.provinces)} provinces, '
                          f'{len(panel_data.years)} years, '
                          f'{panel_data.n_subcriteria} subcriteria')

            # Phase 2: Weight Calculation
            with self.console.phase('CRITIC Weight Calculation') as ph:
                weights = self._calculate_weights(panel_data)

                # Per-year weights (cross-section, independent per year)
                weight_all_years: Dict[int, Any] = {}
                try:
                    weight_all_years = self._calculate_weights_all_years(panel_data)
                    ph.detail(
                        f'Per-year CRITIC: {len(weight_all_years)}/{len(panel_data.years)} '
                        f'years computed'
                    )
                except Exception as _wt_exc:
                    self.logger.warning(
                        'Per-year CRITIC weights failed (non-fatal): %s', _wt_exc)

            # Phase 3: Hierarchical Ranking (6 MCDM methods)
            multi_year_results: Dict[int, Any] = {}
            ranking_result: Optional[HierarchicalRankingResult] = None
            with self.console.phase('Hierarchical Ranking') as ph:
                # PL-1: guard against primary ranking failure so the pipeline
                # can still save partial results (weights, data) rather than
                # crashing entirely — consistent with phases 4-7.
                try:
                    self.logger.info(
                        f"\n[DEBUG] Starting primary ranking for latest year: {max(panel_data.years)}"
                    )
                    ranking_result = self._run_hierarchical_ranking(panel_data, weights)
                    self.logger.info(
                        f"[DEBUG] Primary ranking complete: result type={type(ranking_result).__name__}"
                    )
                    if ranking_result is not None:
                        self.logger.info(
                            f"[DEBUG] Ranking result has: "
                            f"final_ranking={ranking_result.final_ranking is not None}, "
                            f"final_scores={ranking_result.final_scores is not None}, "
                            f"criterion_method_scores={len(getattr(ranking_result, 'criterion_method_scores', {}))} criteria"
                        )
                except Exception as _rank_exc:
                    self.logger.error(
                        f"[DEBUG] Primary ranking FAILED with exception: {_rank_exc}", exc_info=True
                    )
                    ph.warning(f'Hierarchical ranking failed: {_rank_exc}')
                    self.logger.error(
                        'Phase 3 (Hierarchical Ranking) failed — partial results will '
                        'be saved (weights and data are available): %s',
                        _rank_exc,
                    )
                    self.logger.debug(traceback.format_exc())

                # Multi-year: run all panel years for temporal visualisations
                if ranking_result is not None and getattr(
                    getattr(self.config, 'ranking', None), 'run_all_years', True
                ):
                    try:
                        self.logger.info(
                            f"[DEBUG] Starting multi-year ranking for {len(panel_data.years)} years"
                        )
                        multi_year_results = self._run_hierarchical_ranking_all_years(
                            panel_data, weights
                        )
                        self.logger.info(
                            f"[DEBUG] Multi-year ranking complete: {len(multi_year_results)} years succeeded"
                        )
                        for yr, result in multi_year_results.items():
                            n_crit = len(getattr(result, 'criterion_method_scores', {}))
                            self.logger.info(
                                f"[DEBUG] Year {yr}: {n_crit} criteria with method scores"
                            )
                        ph.detail(
                            f'Multi-year: {len(multi_year_results)}/{len(panel_data.years)} '
                            f'years ranked in parallel'
                        )
                    except Exception as _myr_exc:
                        self.logger.error(
                            f"[DEBUG] Multi-year ranking FAILED: {_myr_exc}", exc_info=True
                        )
                        self.logger.warning(
                            f'Multi-year ranking failed (non-fatal): {_myr_exc}'
                        )
                else:
                    if ranking_result is None:
                        self.logger.info("[DEBUG] Skipping multi-year ranking (primary ranking failed)")
                    else:
                        self.logger.info("[DEBUG] Skipping multi-year ranking (run_all_years=False)")

            # Phase 4: ML Forecasting (4 base models + Super Learner + Conformal Prediction)
            forecast_result = None
            with self.console.phase('Ensemble ML Forecasting') as ph:
                if not self.config.forecast.enabled:
                    ph.detail('Forecasting disabled (config.forecast.enabled=False)')
                    self.logger.info('PHASE 4: Ensemble ML Forecasting DISABLED (skipped)')
                else:
                    try:
                        forecast_result = self._run_forecasting(panel_data)
                    except Exception as e:
                        ph.warning(f'Forecasting skipped: {e}')
                        self.logger.debug(traceback.format_exc())

            # Phase 5: Sensitivity Analysis & Validation
            analysis_results: Dict[str, Any] = {'sensitivity': None, 'validation': None}
            with self.console.phase('Sensitivity Analysis & Validation') as ph:
                if ranking_result is not None:
                    try:
                        analysis_results = self._run_analysis(
                            panel_data, ranking_result, weights, forecast_result)
                    except Exception as e:
                        ph.warning(f'Sensitivity analysis skipped: {e}')
                else:
                    ph.warning('Sensitivity analysis skipped (ranking unavailable)')

                try:
                    analysis_results['validation'] = self._run_validation(
                        panel_data, weights, ranking_result, forecast_result)
                except Exception as e:
                    ph.warning(f'Validation skipped: {e}')

            execution_time = time.time() - start_time

            # Phase 6: Visualizations
            with self.console.phase('Generating Figures') as ph:
                try:
                    fig_count = self.visualizer.generate_all(
                        panel_data=panel_data,
                        weights=weights,
                        ranking_result=ranking_result,
                        analysis_results=analysis_results,
                        forecast_result=forecast_result,
                        multi_year_results=multi_year_results,
                        weight_all_years=weight_all_years,
                    )
                    ph.metric('Figures', fig_count)
                except Exception as e:
                    ph.warning(f'Visualization failed: {e}')

            # Phase 7: Save Results
            with self.console.phase('Saving Results') as ph:
                try:
                    figure_paths = self.visualizer.get_generated_figures()
                    self.output_orch.save_all(
                        panel_data=panel_data,
                        weights=weights,
                        ranking_result=ranking_result,
                        forecast_result=forecast_result,
                        analysis_results=analysis_results,
                        execution_time=execution_time,
                        figure_paths=figure_paths,
                        config=self.config,
                        multi_year_results=multi_year_results,
                        weight_all_years=weight_all_years,
                    )
                except Exception as e:
                    ph.warning(f'Result saving failed: {e}')
                    self.logger.debug(traceback.format_exc())

            self.console.separator()
            self.console.info(f'Pipeline completed in {execution_time:.2f}s')
            self.console.info(f'Outputs → {self.config.output_dir}')
            self.console.separator()


            # Build result object
            subcriteria  = weights['subcriteria']   # already filtered to active SCs
            latest_year  = max(panel_data.years)
            latest_ctx   = panel_data.year_contexts.get(latest_year)
            # Use the latest year's active provinces for the decision matrix
            if latest_ctx is not None:
                active_provs = latest_ctx.active_provinces
            else:
                active_provs = panel_data.provinces
            cs = panel_data.subcriteria_cross_section[latest_year]
            avail_scs = [sc for sc in subcriteria if sc in cs.columns]
            avail_provs = [p for p in active_provs if p in cs.index]

            # Build the display snapshot for the latest year.
            # NaN cells (Type 1 structural gaps, Type 3 partial province data)
            # are preserved — no back-fill or median imputation is applied.
            # MCDM ranking and weighting phases use panel_data directly via
            # YearContext and are entirely unaffected by NaN in this field.
            dm_df = cs.loc[avail_provs, avail_scs].copy()
            nan_count = int(dm_df.isna().sum().sum())
            if nan_count > 0:
                self.logger.info(
                    'Decision matrix for year %d: %d NaN cell(s) across %d '
                    'province-SC pair(s). NaN values preserved — no imputation '
                    'applied (strategy: complete-case exclusion).',
                    latest_year, nan_count,
                    int(dm_df.isna().any(axis=1).sum()),
                )

            decision_matrix = dm_df.values

            return PipelineResult(
                panel_data              = panel_data,
                decision_matrix         = decision_matrix,
                sc_weights              = weights['sc_array'],
                criterion_weights_dict  = weights['criterion_weights'],
                weight_details          = weights.get('details', {}),
                ranking_result          = ranking_result,
                multi_year_results      = multi_year_results,
                forecast_result         = forecast_result,
                sensitivity_result      = analysis_results.get('sensitivity'),
                mc_ensemble_diagnostics = (
                    weights.get('details', {})
                    .get('level2', {})
                    .get('mc_diagnostics')
                ),
                execution_time  = execution_time,
                config          = self.config,
            )
        finally:
            # Always flush & close the debug log — even if a phase raises
            self.debug_log.close()

    # -----------------------------------------------------------------
    # Phase 1: Data Loading
    # -----------------------------------------------------------------

    def _load_data(self) -> PanelData:
        loader = DataLoader(self.config)
        panel_data = loader.load()
        self.logger.info(
            f"Loaded: {len(panel_data.provinces)} provinces, "
            f"{len(panel_data.years)} years, "
            f"{panel_data.n_subcriteria} subcriteria, "
            f"{panel_data.n_criteria} criteria"
        )
        return panel_data

    # -----------------------------------------------------------------
    # Phase 2: Weight Calculation
    # -----------------------------------------------------------------

    def _calculate_weights(self, panel_data: PanelData) -> Dict[str, Any]:
        """Run two-level deterministic CRITIC weighting on the full panel.

        Dynamic exclusion rules applied **before** weighting:

        1. Sub-criteria columns that are all-NaN across the *entire* panel are
           dropped (``CRITICWeightingCalculator`` also guards internally).
        2. Province-year rows where **any** *year-active* SC is NaN are
           dropped.  A SC is year-active if it has ≥1 valid value for that
           specific year (per ``YearContext``).  Rows from years with
           structural Type-1 SC gaps (e.g. SC71-SC83 absent 2011-2017, SC52
           absent 2021-2024) are **kept** — those rows carry NaN only for the
           absent SCs, which the downstream CRITIC calculator handles via
           column-mean imputation.  Using a naïve global
           ``notna().all()`` across all 29 SCs would reduce the effective
           weighting panel to ~2 years and severely bias the weights.

        The resulting weight vector spans only *active* sub-criteria.
        """
        from .weighting import CRITICWeightingCalculator

        subcriteria = panel_data.hierarchy.all_subcriteria
        panel_df    = panel_data.subcriteria_long.copy()

        # ── Dynamic exclusion Step 1: drop all-NaN sub-criteria ──────────
        active_scs  = [sc for sc in subcriteria
                       if sc in panel_df.columns and panel_df[sc].notna().any()]
        dropped_scs = [sc for sc in subcriteria if sc not in active_scs]
        if dropped_scs:
            self.logger.info(
                "  Weighting: %d sub-criteria excluded (all-NaN): %s",
                len(dropped_scs), dropped_scs,
            )
        subcriteria = active_scs

        # ── Dynamic exclusion Step 2: year-aware valid rows ───────────────
        # Each row is valid if the SCs that are *active in its year*
        # (per YearContext) are all non-NaN.  This preserves entire yearly
        # cohorts from years with structural Type-1 SC gaps (e.g. SC71-SC83
        # absent 2011-2017, SC52 absent 2021-2024) that a naïve global
        # notna().all() across all 29 SCs would incorrectly discard,
        # reducing the effective panel to only ~2 years.
        year_ctxs = panel_data.year_contexts
        if year_ctxs:
            valid_rows = pd.Series(False, index=panel_df.index)
            for _yr, _ctx in year_ctxs.items():
                _yr_mask   = panel_df['Year'] == _yr
                _yr_active = [sc for sc in _ctx.active_subcriteria
                              if sc in panel_df.columns]
                if _yr_active:
                    _yr_valid  = panel_df.loc[_yr_mask, _yr_active].notna().all(axis=1)
                    valid_rows |= _yr_mask & _yr_valid
        else:
            # Fallback: no YearContext available (should not occur in normal use)
            valid_rows = panel_df[subcriteria].notna().all(axis=1)
        n_dropped  = int((~valid_rows).sum())
        if n_dropped > 0:
            dropped_prov_years = (
                panel_df.loc[~valid_rows, ['Year', 'Province']].drop_duplicates()
            )
            by_year = (
                dropped_prov_years.groupby('Year')['Province']
                .apply(list).to_dict()
            )
            self.logger.info(
                "  Weighting: %d observations excluded (%d unique prov-year pairs)",
                n_dropped, len(dropped_prov_years),
            )
            for yr, provs in sorted(by_year.items()):
                self.logger.info(
                    "    %d: %d province(s) skipped (%s%s)",
                    yr, len(provs), ', '.join(provs[:5]),
                    '\u2026' if len(provs) > 5 else '',
                )
        panel_df = panel_df[valid_rows].copy()

        # ── Build active criteria_groups ──────────────────────────────────
        active_groups: Dict[str, Any] = {}
        for crit_id, sc_list in panel_data.hierarchy.criteria_to_subcriteria.items():
            active = [sc for sc in sc_list if sc in subcriteria]
            if active:
                active_groups[crit_id] = active

        self.logger.info(
            "CRITIC Weighting: %d obs, %d active SCs, %d criterion groups",
            len(panel_df), len(subcriteria), len(active_groups),
        )

        calc   = CRITICWeightingCalculator(config=self.config.weighting)
        result = calc.calculate(
            panel_df        = panel_df,
            criteria_groups = active_groups,
            entity_col      = 'Province',
            time_col        = 'Year',
        )

        # ── Extract weights ───────────────────────────────────────────────
        global_sc_weights  = result.details['global_sc_weights']
        critic_sc_weights  = result.details.get('critic_sc_weights', {})
        criterion_weights  = result.details['level2']['criterion_weights']
        critic_criterion_w = result.details.get('critic_criterion_weights', {})
        sc_arr = np.array([global_sc_weights.get(sc, 0.0) for sc in subcriteria])

        self.logger.info(
            "  Global SC weights: [%.4f, %.4f]", sc_arr.min(), sc_arr.max()
        )
        self.logger.info(
            "  Criterion weights (CRITIC): %s",
            ', '.join(f"{k}={v:.3f}" for k, v in sorted(criterion_weights.items())),
        )

        return {
            'global_sc_weights':        global_sc_weights,
            'critic_sc_weights':        critic_sc_weights,
            'criterion_weights':        criterion_weights,
            'critic_criterion_weights': critic_criterion_w,
            'sc_array':                 sc_arr,
            'subcriteria':              subcriteria,
            'criteria_groups':          active_groups,
            'details':                  result.details,
        }

    def _calculate_weights_all_years(
        self, panel_data: PanelData
    ) -> Dict[int, Dict[str, Any]]:
        """Compute per-year CRITIC weights from each year's cross-section.

        Each year is treated independently: only the provinces that reported
        data in that specific year (per ``YearContext``) are included.  The
        two-level CRITIC computation is identical to ``_calculate_weights``
        but operates on a single-year slice rather than the full panel.

        Returns
        -------
        Dict[int, Dict]
            Keyed by year.  Each value has the same structure as the dict
            returned by ``_calculate_weights``:
            ``global_sc_weights``, ``criterion_weights``,
            ``critic_criterion_weights``, ``sc_array``,
            ``subcriteria``, ``criteria_groups``, ``details``.
            Failed years are omitted and logged as warnings.
        """
        from .weighting import CRITICWeightingCalculator

        all_sc       = panel_data.hierarchy.all_subcriteria
        panel_long   = panel_data.subcriteria_long
        year_ctxs    = panel_data.year_contexts
        results: Dict[int, Dict[str, Any]] = {}

        for year in sorted(panel_data.years):
            try:
                # ── Slice to this year only ───────────────────────────────
                yr_df = panel_long[panel_long['Year'] == year].copy()
                if yr_df.empty:
                    self.logger.warning(
                        '  Per-year weights: year %d — no rows, skipped', year)
                    continue

                # ── Year-active sub-criteria (from YearContext if available) ──
                ctx = year_ctxs.get(year)
                if ctx is not None:
                    active_scs = [sc for sc in ctx.active_subcriteria
                                  if sc in yr_df.columns and yr_df[sc].notna().any()]
                else:
                    active_scs = [sc for sc in all_sc
                                  if sc in yr_df.columns and yr_df[sc].notna().any()]

                if not active_scs:
                    self.logger.warning(
                        '  Per-year weights: year %d — no active SCs, skipped', year)
                    continue

                # ── Drop rows missing any active SC ───────────────────────
                yr_df = yr_df[yr_df[active_scs].notna().all(axis=1)].copy()
                if yr_df.empty:
                    self.logger.warning(
                        '  Per-year weights: year %d — all rows dropped, skipped', year)
                    continue

                # ── Build active criteria groups for this year ─────────────
                active_groups: Dict[str, Any] = {}
                for crit_id, sc_list in panel_data.hierarchy.criteria_to_subcriteria.items():
                    active = [sc for sc in sc_list if sc in active_scs]
                    if active:
                        active_groups[crit_id] = active

                # ── Run two-level CRITIC on the year slice ─────────────────
                calc = CRITICWeightingCalculator(config=self.config.weighting)
                res  = calc.calculate(
                    panel_df        = yr_df,
                    criteria_groups = active_groups,
                    entity_col      = 'Province',
                    time_col        = 'Year',
                )

                global_sc_w  = res.details['global_sc_weights']
                criterion_w  = res.details['level2']['criterion_weights']
                critic_crit_w = res.details.get('critic_criterion_weights', {})
                sc_arr = np.array([global_sc_w.get(sc, 0.0) for sc in active_scs])

                results[year] = {
                    'global_sc_weights':        global_sc_w,
                    'critic_sc_weights':        global_sc_w,
                    'criterion_weights':        criterion_w,
                    'critic_criterion_weights': critic_crit_w,
                    'sc_array':                 sc_arr,
                    'subcriteria':              active_scs,
                    'criteria_groups':          active_groups,
                    'details':                  res.details,
                }

            except Exception as _exc:
                self.logger.warning(
                    '  Per-year weights: year %d failed — %s', year, _exc)

        self.logger.info(
            'Per-year CRITIC weights: %d/%d years computed',
            len(results), len(panel_data.years),
        )
        return results

    # -----------------------------------------------------------------
    # Phase 3: Hierarchical Ranking
    # -----------------------------------------------------------------

    def _run_hierarchical_ranking(
        self,
        panel_data: PanelData,
        weights: Dict[str, Any],
    ) -> HierarchicalRankingResult:
        pipeline = HierarchicalRankingPipeline()
        return pipeline.rank(
            panel_data          = panel_data,
            subcriteria_weights = weights['global_sc_weights'],
            criterion_weights   = weights['criterion_weights'],
        )

    def _run_hierarchical_ranking_all_years(
        self,
        panel_data: PanelData,
        weights: Dict[str, Any],
    ) -> Dict[int, HierarchicalRankingResult]:
        """
        Run hierarchical ranking for every year in *panel_data.years* using
        thread-level parallelism.

        Each year-ranking is independent (its own ``YearContext`` governs which
        provinces and sub-criteria are active), so they can run concurrently.
        ``ThreadPoolExecutor`` is used rather than ``ProcessPoolExecutor`` to
        avoid pickling issues with complex dataclass objects;  numpy operations
        release the GIL so meaningful parallelism is achieved for the
        matrix-heavy MCDM stages.

        Returns
        -------
        Dict[int, HierarchicalRankingResult]
            Keyed by year.  Missing years (errors) are silently omitted and
            logged as warnings.
        """
        import concurrent.futures as _cf
        import os as _os

        years = sorted(panel_data.years)
        sc_weights   = weights['global_sc_weights']
        crit_weights = weights['criterion_weights']

        # Resolve max_parallel_years: config → cpu_count → len(years)
        _cfg_max = getattr(getattr(self.config, 'ranking', None),
                           'max_parallel_years', None)
        max_workers = min(
            len(years),
            _cfg_max if _cfg_max is not None else (_os.cpu_count() or 4),
        )

        self.logger.info(
            "Multi-year ranking: %d years (%d–%d), max_workers=%d",
            len(years), years[0], years[-1], max_workers,
        )

        def _rank_year(year: int):
            """Worker: create a fresh pipeline instance and rank one year."""
            try:
                _pl = HierarchicalRankingPipeline()
                result = _pl.rank(
                    panel_data          = panel_data,
                    subcriteria_weights = sc_weights,
                    criterion_weights   = crit_weights,
                    target_year         = year,
                )
                return year, result, None
            except Exception as _exc:
                return year, None, str(_exc)

        all_results: Dict[int, HierarchicalRankingResult] = {}

        with _cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_rank_year, yr): yr for yr in years}
            for fut in _cf.as_completed(future_map):
                yr, res, err = fut.result()
                if err:
                    self.logger.warning(
                        "Multi-year ranking: year %d failed — %s", yr, err
                    )
                else:
                    all_results[yr] = res
                    self.logger.debug(
                        "  ✓ year %d: %d provinces ranked", yr,
                        len(res.final_ranking),
                    )

        self.logger.info(
            "Multi-year ranking complete: %d/%d years succeeded",
            len(all_results), len(years),
        )
        return all_results

    # -----------------------------------------------------------------
    # Phase 2: Data Imputation Validation Helper
    # -----------------------------------------------------------------

    def _validate_and_log_ml_imputation(
        self, ml_panel_data: PanelData, original_panel_data: PanelData
    ) -> None:
        """
        Validate the imputed ML panel and log comprehensive diagnostics.

        **CRITICAL ASSERTIONS** (Step 6 validation):
        1. No NaN in imputed subcriteria cross-sections
        2. All 28 sub-criteria present in every year
        3. All 63 provinces present (if they had any historical data)
        4. MICE imputation fills all missing subcriteria (Phase B+)

        Parameters
        ----------
        ml_panel_data : PanelData
            The imputed panel (output of build_ml_panel_data).
        original_panel_data : PanelData
            The raw panel (input to build_ml_panel_data).

        Raises
        ------
        AssertionError
            If any critical validation fails.
        """
        import numpy as np

        self.logger.info("\n[POST-IMPUTATION] Validating imputed panel completeness:")
        
        # ──────────────────────────────────────────────────────────────────
        # Assertion 1: No NaN in imputed subcriteria cross-sections
        # ──────────────────────────────────────────────────────────────────
        nan_found = False
        nan_per_year = {}
        nan_per_sc_imputed = {}
        
        for yr, cs in sorted(ml_panel_data.subcriteria_cross_section.items()):
            nan_count_year = cs.isna().sum().sum()
            nan_per_year[yr] = nan_count_year
            
            if nan_count_year > 0:
                nan_found = True
                self.logger.error(
                    f"    ✗ Year {yr}: {nan_count_year} NaN cells found (IMPUTATION FAILED)"
                )
                # Identify which SCs have NaN
                for sc in cs.columns:
                    nan_count_sc = cs[sc].isna().sum()
                    if nan_count_sc > 0:
                        self.logger.error(
                            f"      • {sc}: {nan_count_sc} cells"
                        )
                        if sc not in nan_per_sc_imputed:
                            nan_per_sc_imputed[sc] = 0
                        nan_per_sc_imputed[sc] += nan_count_sc
        
        assert not nan_found, (
            f"[CRITICAL] Imputed panel contains {sum(nan_per_year.values())} NaN cells. "
            "MICE imputation failed. Check build_ml_panel_data() MICE logic."
        )
        self.logger.info("    ✓ No NaN cells found in imputed subcriteria (all 2011-2024 years)")

        # ──────────────────────────────────────────────────────────────────
        # Assertion 2: All 29 sub-criteria present in every year
        # ──────────────────────────────────────────────────────────────────
        # Note: SC52 is included in hierarchy (discontinued 2021, but present 2011-2020).
        # YearContext will naturally exclude it for years 2021-2024.
        expected_scs = set(ml_panel_data.hierarchy.all_subcriteria)
        assert len(expected_scs) == 29, (
            f"[CRITICAL] Expected 29 sub-criteria, got {len(expected_scs)}. "
            "SC52 should be included in hierarchy (year-active exclusion handled by YearContext)."
        )
        
        sc_count_per_year = {}
        for yr, cs in sorted(ml_panel_data.subcriteria_cross_section.items()):
            actual_scs = set(cs.columns)
            sc_count_per_year[yr] = len(actual_scs)
            
            missing_scs = expected_scs - actual_scs
            extra_scs = actual_scs - expected_scs
            
            assert not missing_scs, (
                f"[CRITICAL] Year {yr}: Missing sub-criteria {sorted(missing_scs)}. "
                "Expected all 29 SCs in the imputed panel."
            )
            assert not extra_scs, (
                f"[CRITICAL] Year {yr}: Unexpected sub-criteria {sorted(extra_scs)}. "
                "Expected exactly 29 SCs (SC52 included)."
            )
        
        self.logger.info(
            f"    ✓ All 29 sub-criteria present in every year "
            f"({min(sc_count_per_year.values())}–{max(sc_count_per_year.values())} per year)"
        )

        # ──────────────────────────────────────────────────────────────────
        # Assertion 3: All 63 provinces present
        # ──────────────────────────────────────────────────────────────────
        prov_count_per_year = {}
        for yr, cs in sorted(ml_panel_data.subcriteria_cross_section.items()):
            prov_count_per_year[yr] = len(cs.index)
        
        expected_provs = ml_panel_data.n_provinces
        assert expected_provs == 63, (
            f"[CRITICAL] Expected 63 provinces, got {expected_provs}. "
            "Panel configuration mismatch."
        )
        
        for yr, count in prov_count_per_year.items():
            assert count == 63, (
                f"[CRITICAL] Year {yr}: Only {count}/63 provinces in imputed panel. "
                "Imputation should activate all provinces."
            )
        
        self.logger.info(
            f"    ✓ All 63 provinces present in every year "
            f"({min(prov_count_per_year.values())}–{max(prov_count_per_year.values())} per year)"
        )

        # ──────────────────────────────────────────────────────────────────
        # Post-imputation statistics and logging
        # ──────────────────────────────────────────────────────────────────
        imputed_total_cells = 0
        imputed_nan_count = 0
        for yr, cs in sorted(ml_panel_data.subcriteria_cross_section.items()):
            imputed_total_cells += cs.size
            imputed_nan_count += cs.isna().sum().sum()
        
        imputed_nan_pct = 100.0 * imputed_nan_count / imputed_total_cells if imputed_total_cells > 0 else 0.0
        
        # Compute imputation improvement
        original_nan_count = sum(
            original_panel_data.subcriteria_cross_section[yr].isna().sum().sum()
            for yr in original_panel_data.subcriteria_cross_section.keys()
        )
        original_total_cells = (
            original_panel_data.n_provinces
            * original_panel_data.n_years
            * original_panel_data.n_subcriteria
        )
        
        nan_reduction = original_nan_count - imputed_nan_count if original_nan_count > 0 else 0
        nan_reduction_pct = 100.0 * nan_reduction / original_nan_count if original_nan_count > 0 else 0.0
        
        self.logger.info(f"\n[IMPUTATION SUMMARY] MICE imputation complete:")
        self.logger.info(f"  Original panel:")
        self.logger.info(f"    NaN cells: {original_nan_count:,}/{original_total_cells:,} ({100*original_nan_count/original_total_cells:.1f}%)")
        self.logger.info(f"  Imputed panel:")
        self.logger.info(f"    NaN cells: {imputed_nan_count:,}/{imputed_total_cells:,} ({imputed_nan_pct:.2f}%)")
        self.logger.info(f"  Imputation quality:")
        self.logger.info(f"    NaN cells removed: {nan_reduction:,} ({nan_reduction_pct:.1f}%)")
        self.logger.info(f"    Success rate: {100-imputed_nan_pct:.2f}%")
        
        # ──────────────────────────────────────────────────────────────────
        # Per-year imputation details
        # ──────────────────────────────────────────────────────────────────
        self.logger.info(f"\n[IMPUTATION DETAILS] Per-year subcriteria count:")
        for yr in sorted(ml_panel_data.subcriteria_cross_section.keys()):
            cs = ml_panel_data.subcriteria_cross_section[yr]
            self.logger.info(
                f"  {yr}: {len(cs.index):2d} provinces × {len(cs.columns):2d} sub-criteria"
            )
        
        # ──────────────────────────────────────────────────────────────────
        # Verify YearContext consistency
        # ──────────────────────────────────────────────────────────────────
        self.logger.info(f"\n[YEAR CONTEXTS] Dynamic exclusion contexts after imputation:")
        for yr in sorted(ml_panel_data.year_contexts.keys()):
            ctx = ml_panel_data.year_contexts[yr]
            self.logger.info(
                f"  {yr}: {len(ctx.active_provinces)} provinces, "
                f"{len(ctx.active_subcriteria)} SCs, {len(ctx.active_criteria)} criteria"
            )
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info(
            "✓ PHASE 2 VALIDATION COMPLETE: ML imputation panel is production-ready"
        )
        self.logger.info("=" * 80 + "\n")

    # -----------------------------------------------------------------
    # Phase 4: ML Forecasting (State-of-the-Art Ensemble)
    # -----------------------------------------------------------------

    def _run_forecasting(
        self,
        panel_data: PanelData,
    ) -> Any:
        """
        Run state-of-the-art forecasting using UnifiedForecaster.

        **Phase 5 Integration (Orchestration)**:
        This method now performs comprehensive end-to-end forecasting for 2025:

        1. **Data Imputation** (Step 5, Phase 2)
           - Applies MICE imputation to 2011–2024 training panel
           - Outputs fully NaN-free subcriteria cross-sections
           - Validates all 29 SCs present (SC52 included, year-active), all 63 provinces active

        2. **ML Training & Prediction** (Steps 7–8, Phase 3)
           - Forecaster configured to predict 28 sub-criteria (not 8 criteria)
           - 4 diverse base models: CatBoost + BayesianRidge + SVR + ElasticNet
           - Super Learner meta-ensemble + Conformal Prediction for 95% intervals

        3. **Post-Forecast Aggregation** (Step 10, Phase 4)
           - Aggregates 28 SC predictions to 8 criteria using critic weights
           - Uses two-level weighting: SC weights × criteria weights
           - Output: (63 × 8) criteria predictions for 2025

        4. **Forecast Year Context & Integration** (Steps 13–14, Phase 5)
           - Creates YearContext for 2025 mirroring 2024's structure
           - Wraps 2025 criteria predictions in decision matrix format
           - Ready for downstream weighting & ranking phases

        Parameters
        ----------
        panel_data : PanelData
            Original raw panel (2011–2024, with NaN for missing data)

        Returns
        -------
        UnifiedForecastResult
            Results including:
            - predictions: 28 SC predictions shape (63, 28)
            - criteria_predictions: 8 criteria predictions shape (63, 8)
            - 2025_decision_matrix: wrapped 8 criteria as (63, 8) DataFrame
            - 2025_year_context: YearContext matching 2024's structure
        """
        if not self.config.forecast.enabled:
            self.logger.info("Forecasting disabled in config")
            return None
        
        try:
            from .forecasting import UnifiedForecaster
        except ImportError:
            from forecasting import UnifiedForecaster
        
        # Determine target year
        target_year = self.config.forecast.target_year
        if target_year is None:
            target_year = max(panel_data.years) + 1
        
        self.logger.info(f"Target year: {target_year}")
        # Build the log-time model list to mirror _create_models() logic
        _base_model_names = ["CatBoost", "BayesianRidge", "SVR",
                             "ElasticNet"]
        self.logger.info(
            f"Base models: {len(_base_model_names)} ({', '.join(_base_model_names)})"
        )
        self.logger.info(f"Meta-learner: {getattr(self.config.forecast, 'meta_learner_type', 'ridge')}")
        self.logger.info(f"Calibration: Conformal {self.config.forecast.conformal_method} (α={self.config.forecast.conformal_alpha})")
        
        forecaster = UnifiedForecaster(
            conformal_method=self.config.forecast.conformal_method,
            conformal_alpha=self.config.forecast.conformal_alpha,
            cv_folds=self.config.forecast.cv_folds,
            cv_min_train_years=self.config.forecast.cv_min_train_years,
            random_state=self.config.forecast.random_state,
            verbose=self.config.forecast.verbose,
            target_level=self.config.forecast.forecast_level,
        )

        # ── Phase 2: Data Imputation ───────────────────────────────────
        # Build a fully-imputed copy of panel_data for the ML path only.
        # MCDM weighting and ranking (Phases 2-3) continue using the raw
        # observed panel_data (complete-case strategy).  The forecasting
        # ensemble requires NaN-free feature matrices, so we apply three-
        # stage temporal imputation (linear interp → ffill/bfill → median)
        # on a deep copy and rebuild all derived views + year_contexts.
        try:
            from .data.missing_data import build_ml_panel_data
        except ImportError:
            from data.missing_data import build_ml_panel_data

        self.logger.info("=" * 80)
        self.logger.info("PHASE 2: DATA IMPUTATION (ML TRAINING PANEL PREPARATION)")
        self.logger.info("=" * 80)
        
        # Log pre-imputation diagnostics
        self.logger.info("\n[PRE-IMPUTATION] Original raw panel structure:")
        self.logger.info(f"  Provinces: {panel_data.n_provinces} (all years combined)")
        self.logger.info(f"  Years: {panel_data.n_years} ({min(panel_data.years)}–{max(panel_data.years)})")
        self.logger.info(f"  Sub-criteria: {panel_data.n_subcriteria}")
        self.logger.info(f"  Criteria: {panel_data.n_criteria}")
        
        # Compute original NaN statistics
        original_nan_count = 0
        original_total_cells = 0
        nan_per_sc = {}
        for yr, cs in sorted(panel_data.subcriteria_cross_section.items()):
            original_total_cells += cs.size
            original_nan_count += cs.isna().sum().sum()
            for sc in cs.columns:
                if sc not in nan_per_sc:
                    nan_per_sc[sc] = 0
                nan_per_sc[sc] += cs[sc].isna().sum()
        
        original_nan_pct = 100.0 * original_nan_count / original_total_cells if original_total_cells > 0 else 0.0
        self.logger.info(f"  Total cells: {original_total_cells:,}")
        self.logger.info(f"  NaN cells (original): {original_nan_count:,} ({original_nan_pct:.1f}%)")
        
        if nan_per_sc and len(nan_per_sc) <= 10:
            self.logger.info("  NaN count per SC (top contributors):")
            for sc in sorted(nan_per_sc.keys(), key=lambda x: nan_per_sc[x], reverse=True)[:10]:
                self.logger.info(f"    {sc}: {nan_per_sc[sc]:,} cells")

        # Apply MICE imputation (Phase B+ unified strategy)
        self.logger.info("\n[IMPUTATION] Applying MICE imputation:")
        self.logger.info("  Method: Multivariate Imputation by Chained Equations")
        self.logger.info("  Estimator: ExtraTreesRegressor (learns feature correlations)")
        self.logger.info("  Iterations: 20 (convergence)")
        self.logger.info("  Target: Fill all missing subcriteria values")
        
        ml_panel_data = build_ml_panel_data(panel_data)
        
        # Validate imputed panel completeness (CRITICAL ASSERTIONS)
        self._validate_and_log_ml_imputation(ml_panel_data, panel_data)

        result = forecaster.fit_predict(ml_panel_data, target_year=target_year)
        
        # Log comprehensive forecasting results
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PHASE 4: ML FORECASTING RESULTS")
        self.logger.info("=" * 80)
        
        # Forecast data shape & structure
        if hasattr(result, 'predictions') and result.predictions is not None:
            pred_shape = result.predictions.shape
            self.logger.info(f"\n[FORECAST OUTPUT]")
            self.logger.info(f"  Predictions shape: {pred_shape} (entities × components)")
            self.logger.info(f"  Target year: {target_year}")
            self.logger.info(f"  Forecast level: {self.config.forecast.forecast_level}")
            
            # Validate predictions completeness
            n_nan = result.predictions.isna().sum().sum()
            total_cells = pred_shape[0] * pred_shape[1]
            if n_nan > 0:
                self.logger.warning(
                    f"  ⚠ NaN cells in predictions: {n_nan}/{total_cells} ({100*n_nan/total_cells:.1f}%)"
                )
            else:
                self.logger.info(f"  ✓ All {total_cells:,} prediction cells are valid (no NaN)")
            
            # Prediction value ranges
            pred_min = result.predictions.min().min()
            pred_max = result.predictions.max().max()
            pred_mean = result.predictions.values.mean()
            self.logger.info(
                f"  Value ranges: [{pred_min:.4f}, {pred_max:.4f}], mean={pred_mean:.4f}"
            )
        
        # Model ensemble diagnostics
        if hasattr(result, 'model_contributions') and result.model_contributions:
            self.logger.info(f"\n[META-LEARNER] Base model weights (optimal ensemble):")
            for model, weight in sorted(result.model_contributions.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(weight * 30)
                self.logger.info(f"  {model:20s}: {weight:.4f} {bar}")
        
        # Cross-validation performance
        if hasattr(result, 'cross_validation_scores') and result.cross_validation_scores:
            import numpy as _np
            cv = result.cross_validation_scores
            all_r2 = [s for scores in cv.values() for s in scores]
            self.logger.info(f"\n[CROSS-VALIDATION] Performance across folds:")
            self.logger.info(
                f"  Mean R² = {_np.mean(all_r2):.4f} ± {_np.std(all_r2):.4f} "
                f"({len(all_r2)} folds)"
            )
            
            # Per-model CV stats
            for model, scores in sorted(cv.items()):
                model_r2 = _np.array(scores)
                self.logger.info(
                    f"    {model:20s}: {_np.mean(model_r2):.4f} ± {_np.std(model_r2):.4f}"
                )
        
        # Uncertainty quantification
        if hasattr(result, 'uncertainty') and result.uncertainty is not None:
            unc_shape = result.uncertainty.shape
            self.logger.info(f"\n[UNCERTAINTY QUANTIFICATION]")
            self.logger.info(f"  Uncertainty estimates shape: {unc_shape}")
            if unc_shape[0] > 0:
                unc_values = result.uncertainty.values[~np.isnan(result.uncertainty.values)]
                if len(unc_values) > 0:
                    self.logger.info(
                        f"  Uncertainty range: [{unc_values.min():.4f}, {unc_values.max():.4f}], "
                        f"mean={unc_values.mean():.4f}"
                    )
        
        # Prediction intervals
        if hasattr(result, 'prediction_intervals') and result.prediction_intervals:
            self.logger.info(f"\n[PREDICTION INTERVALS]")
            for component, interval_df in result.prediction_intervals.items():
                if interval_df is not None and not interval_df.empty:
                    width_mean = (interval_df.iloc[:, 1] - interval_df.iloc[:, 0]).mean()
                    self.logger.info(
                        f"  {component}: mean interval width = {width_mean:.4f}"
                    )
        
        # ── Phase 5 Integration: Post-Forecast Aggregation ───────────────
        # Step 10: Aggregate 29 SC predictions to 8 criteria using critic weights
        # If forecast was on sub-criteria (29 outputs), aggregate to criteria (8 outputs)
        # This enables downstream weighting/ranking phases to operate on the same
        # decision matrix structure as historical years.
        if (self.config.forecast.forecast_level == 'subcriteria' and 
            hasattr(result, 'predictions') and result.predictions is not None and
            result.predictions.shape[1] == 29):
            
            self.logger.info(
                "\n" + "=" * 80)
            self.logger.info(
                "PHASE 5 INTEGRATION: POST-FORECAST AGGREGATION (29 SC → 8 Criteria)")
            self.logger.info("=" * 80)
            
            try:
                # BUG-1 FIX: ml_panel_data was already built at the top of this
                # method (fully MICE-imputed). Reuse it — do NOT call
                # build_ml_panel_data() again, which would double the MICE runtime.
                #
                # BUG-2 FIX: Instead of re-running CRITICWeightingCalculator on
                # the single 2025 forecast cross-section (63 provinces × 1 year),
                # use the historically-grounded Phase 2 weights (2011–2024, 14 years)
                # stored in self.weights. These are the same weights used by the
                # MCDM ranking phase and are far more reliable than single-year CRITIC.
                hist_weights = getattr(self, 'weights', {})
                hist_details = hist_weights.get('details', {}) if isinstance(hist_weights, dict) else {}
                local_sc_weights = hist_details.get('level1', {})
                hist_crit_weights = hist_details.get('level2', {}).get('criterion_weights', {})
                hist_global_sc = hist_weights.get('weights', {}) if isinstance(hist_weights, dict) else {}

                if not local_sc_weights:
                    self.logger.warning(
                        "  ⚠ Phase 2 historical CRITIC weights not found in self.weights; "
                        "SC→criteria aggregation will fall back to equal weighting within each group."
                    )

                forecast_weights_dict = {
                    'global_sc_weights': hist_global_sc,
                    'criterion_weights': hist_crit_weights,
                }
                result.forecast_criterion_weights_ = hist_crit_weights

                # Get the panel's hierarchy for SC-to-criteria mapping
                criteria_predictions = self._aggregate_sc_to_criteria(
                    result.predictions,
                    ml_panel_data.hierarchy,
                    target_year,
                    local_weights=local_sc_weights,
                )
                
                # Attach aggregated criteria predictions to result
                result.criteria_predictions = criteria_predictions
                
                self.logger.info(
                    f"[POST-AGGREGATION] Criteria predictions shape: "
                    f"{criteria_predictions.shape} (entities × criteria)")
                
                # Validate aggregated criteria
                n_nan_criteria = criteria_predictions.isna().sum().sum()
                if n_nan_criteria > 0:
                    self.logger.warning(
                        f"  ⚠ NaN cells in aggregated criteria: {n_nan_criteria}")
                else:
                    self.logger.info(
                        f"  ✓ All {criteria_predictions.shape[0] * criteria_predictions.shape[1]:,} "
                        f"criteria cells are valid (no NaN)")
                
                # Step 13: Create YearContext for 2025 (forecast year)
                ctx_2025 = self._create_forecast_year_context(
                    target_year=target_year,
                    criteria_predictions=criteria_predictions,
                    template_year_context=ml_panel_data.year_contexts.get(
                        max(ml_panel_data.years)),
                    hierarchy=ml_panel_data.hierarchy,
                )
                
                # Step 14: Create 2025 decision matrix wrapper
                # Wraps the aggregated criteria predictions in a temporary structure
                # that the ranking phase can consume
                decision_matrix_2025 = self._create_forecast_decision_matrix(
                    criteria_predictions=criteria_predictions,
                    year_context=ctx_2025,
                    target_year=target_year,
                    hierarchy=ml_panel_data.hierarchy,
                )
                
                # Attach 2025 integration structures
                result.forecast_year_context = ctx_2025
                result.forecast_decision_matrix = decision_matrix_2025
                
                self.logger.info("  Running MCDM Ranking on 2025 Forecast...")
                import copy
                forecast_panel = copy.deepcopy(ml_panel_data)
                if target_year not in forecast_panel.years:
                    forecast_panel.years.append(target_year)
                    forecast_panel.years.sort()
                
                forecast_panel.subcriteria_cross_section[target_year] = result.predictions.copy()
                forecast_panel.year_contexts[target_year] = ctx_2025
                
                try:
                    forecast_ranking_result = self._run_hierarchical_ranking(
                        panel_data=forecast_panel,
                        weights=forecast_weights_dict,
                    )
                    result.forecast_ranking_result = forecast_ranking_result
                    self.logger.info("  ✓ MCDM Ranking for 2025 Forecast completed")
                except Exception as rank_e:
                    self.logger.warning(f"  ⚠ Forecast MCDM Ranking failed: {rank_e}")
                
                self.logger.info(
                    f"\n[2025 INTEGRATION COMPLETE]")
                self.logger.info(
                    f"  Decision matrix for 2025: {decision_matrix_2025.shape} "
                    f"(alternatives × criteria)")
                self.logger.info(
                    f"  Active provinces: {len(ctx_2025.active_provinces)}")
                self.logger.info(
                    f"  Active criteria: {len(ctx_2025.active_criteria)}")
                
            except Exception as e:
                self.logger.warning(
                    f"Post-forecast aggregation failed (non-fatal): {e}")
                self.logger.debug(traceback.format_exc())
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("✓ Phase 4 complete: ML forecasting successful")
        self.logger.info("=" * 80 + "\n")
        
        return result

    # -----------------------------------------------------------------
    # Phase 5: Post-Forecast Integration Helpers
    # -----------------------------------------------------------------

    def _aggregate_sc_to_criteria(
        self,
        sc_predictions: pd.DataFrame,
        hierarchy: HierarchyMapping,
        target_year: int,
        local_weights: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """
        Aggregate 29 sub-criteria predictions to 8 criteria using critic weights.

        **Step 10 (Phase 4)**: After UnifiedForecaster produces 29 SC predictions
        for 2025 (including SC52 for cross-temporal consistency), aggregate to
        8 criteria via two-level critic weighting:
        - Level 1: Local SC weights per criterion group (normalized within C_k)
        - Level 2: Criterion weights (derived from composite matrix)
        - Global: global_w[SC_j] = local_w[SC_j | C_k] × criterion_w[C_k]

        Note: For 2025 forecasts, SC52 will be zero-filled (no future data), but
        is included in aggregation for consistency with historical MCDM phases.

        The weighting uses critic values computed from the full 2011–2024 panel
        (available in pipeline.py:_calculate_weights as a cached result), ensuring
        stable, historically-grounded aggregation rather than single-year fitting.

        Parameters
        ----------
        sc_predictions : pd.DataFrame
            Shape (n_entities, 29) sub-criteria predictions for 2025.
            Index = province names; columns = SC11, SC12, ..., SC53, SC54, SC61...SC83
            (29 total, including SC52).  SC52 will be zero or forward-filled.

        hierarchy : HierarchyMapping
            Hierarchy with criteria_to_subcriteria mapping: {C_k: [SCs]}.

        target_year : int
            The forecast year (typically 2025). Used for logging only.

        Returns
        -------
        pd.DataFrame
            Shape (n_entities, 8) aggregated criteria predictions.
            Index = same as sc_predictions; columns = ['C01', 'C02', ..., 'C08'].
            All cells are valid floats (no NaN).

        Raises
        ------
        AssertionError
            If input does not have exactly 28 columns or is missing expected SCs.

        Notes
        -----
        Aggregation formula for criterion k:
            C_k(prov) = Σ_i∈C_k w_i(C_k) × SC_i(prov)

        where:
        - w_i(C_k) = local weight of SC_i within criterion C_k  (normalized to sum 1 within C_k)
        - SC_i(prov) = predicted value of SC_i for the province

        Example
        -------
        >>> sc_pred = pd.DataFrame(random values, shape=(63, 29))  # 63 provinces, SC52 included
        >>> crit_pred = self._aggregate_sc_to_criteria(sc_pred, hierarchy, 2025)
        >>> assert crit_pred.shape == (63, 8)
        >>> assert all(col in crit_pred.columns for col in ['C01', ..., 'C08'])
        """
        import numpy as np
        import pandas as pd

        # ── Validate input ────────────────────────────────────────────────
        expected_scs = sorted(hierarchy.all_subcriteria)
        assert len(expected_scs) == 29, (
            f"Expected 29 SCs, got {len(expected_scs)}. "
            "SC52 should be in hierarchy (year-active exclusion by YearContext)."
        )
        
        actual_scs = sorted(sc_predictions.columns.tolist())
        assert actual_scs == expected_scs, (
            f"SC columns mismatch. Expected {expected_scs}, got {actual_scs}"
        )
        
        # ── Initialize criteria predictions ───────────────────────────────
        criteria = sorted(hierarchy.criteria_to_subcriteria.keys())
        criteria_pred = pd.DataFrame(
            np.zeros((len(sc_predictions), len(criteria))),
            index=sc_predictions.index,
            columns=criteria,
            dtype=float,
        )
        
        # ── Aggregate using within-criterion SC weights ──────────────────
        # For each criterion group, compute the weighted average of its SCs.
        # The weights are stored in the pipeline's cached weights dict
        # (from _calculate_weights, which uses critic_weighting.py).
        
        # NOTE: We use simple equal-weight averaging for now.
        # In a production version, these would be fetched from the cached
        # weights computed in Phase 2 (_calculate_weights):
        # weights['details']['level1'][crit_id]['local_sc_weights']
        # 
        # For Phase 5 MVP (Minimum Viable Product), equal weighting within
        # criterion groups is theoretically sound and maintains the principle
        # that all SCs within a criterion contribute equally to the aggregated score.
        
        for crit_id, sc_list in hierarchy.criteria_to_subcriteria.items():
            # Filter to SCs that are in the prediction set
            active_scs = [sc for sc in sc_list if sc in sc_predictions.columns]
            
            if active_scs:
                if local_weights and crit_id in local_weights and 'local_sc_weights' in local_weights[crit_id]:
                    weights_series = pd.Series(local_weights[crit_id]['local_sc_weights'])
                    w = weights_series.reindex(active_scs).fillna(0)
                    if w.sum() > 0:
                        w = w / w.sum()
                    else:
                        w = pd.Series(1.0 / len(active_scs), index=active_scs)
                    crit_values = (sc_predictions[active_scs] * w).sum(axis=1)
                else:
                    crit_values = sc_predictions[active_scs].mean(axis=1)
                criteria_pred[crit_id] = crit_values
            else:
                # Should not occur given validated hierarchy above
                self.logger.warning(
                    f"No active SCs found for criterion {crit_id} — leaving NaN"
                )
                criteria_pred[crit_id] = np.nan
        
        # ── Validate output ───────────────────────────────────────────────
        assert criteria_pred.shape == (len(sc_predictions), len(criteria)), (
            f"Output shape mismatch. Expected ({len(sc_predictions)}, {len(criteria)}), "
            f"got {criteria_pred.shape}"
        )
        
        n_nan = criteria_pred.isna().sum().sum()
        assert n_nan == 0, (
            f"Aggregated criteria contain {n_nan} NaN cells. "
            "Aggregation failed for some criteria."
        )
        
        self.logger.debug(
            f"Aggregated {len(expected_scs)} SCs → {len(criteria)} criteria "
            f"for {len(sc_predictions)} entities (year {target_year})"
        )
        
        return criteria_pred

    def _create_forecast_year_context(
        self,
        target_year: int,
        criteria_predictions: pd.DataFrame,
        template_year_context: Optional[YearContext],
        hierarchy: HierarchyMapping,
    ) -> YearContext:
        """
        Create a YearContext for the forecast year (2025).

        **Step 13 (Phase 5)**: The ranking pipeline requires a YearContext
        to determine which provinces and criteria are "active" (have valid data).
        For the forecast year, we create a synthetic context that mirrors
        the most recent historical year's structure (mirroring 2024).

        Parameters
        ----------
        target_year : int
            Forecast year (e.g., 2025).

        criteria_predictions : pd.DataFrame
            Shape (n_entities, n_criteria) predictions.
            Index = province names. All cells present (no NaN).

        template_year_context : YearContext or None
            YearContext from the most recent historical year (e.g., 2024).
            Used as a template for the new 2025 context.
            If None, will construct from scratch.

        hierarchy : HierarchyMapping
            Criterion/subcriteria hierarchy.

        Returns
        -------
        YearContext
            A new YearContext instance with:
            - year = target_year
            - active_provinces = all provinces in criteria_predictions
            - active_criteria = all criteria in criteria_predictions
            - active_subcriteria = all SCs in hierarchy (mirroring template)
            - criterion_alternatives = {C_k: all provinces} for each criterion
            - criterion_subcriteria = {C_k: [SCs]} from hierarchy
            - valid_pairs = all (province, SC) pairs (full completeness)
        """
        # Extract provinces from predictions
        active_provinces = list(criteria_predictions.index)
        
        # Extract all criteria
        active_criteria = list(criteria_predictions.columns)
        
        # All SCs from hierarchy (after SC52 exclusion)
        active_subcriteria = sorted(hierarchy.all_subcriteria)
        
        # Build criterion-to-alternatives mapping (all prov for each criterion)
        criterion_alternatives = {
            crit_id: active_provinces.copy()
            for crit_id in active_criteria
        }
        
        # Build criterion-to-subcriteria mapping from hierarchy
        criterion_subcriteria = {
            crit_id: [sc for sc in hierarchy.criteria_to_subcriteria[crit_id]
                      if sc in active_subcriteria]
            for crit_id in active_criteria
        }
        
        # All (province, SC) pairs are valid (complete case)
        valid_pairs = {
            (prov, sc)
            for prov in active_provinces
            for sc in active_subcriteria
        }
        
        # Determine excluded sets (none for forecast year with complete predictions)
        all_provinces = set(active_provinces)  # We know all are present
        all_criteria = set(active_criteria)
        excluded_provinces = []
        excluded_criteria = []
        excluded_subcriteria = []
        
        # Create YearContext instance
        ctx = YearContext(
            year=target_year,
            active_provinces=active_provinces,
            active_subcriteria=active_subcriteria,
            active_criteria=active_criteria,
            excluded_provinces=excluded_provinces,
            excluded_subcriteria=excluded_subcriteria,
            excluded_criteria=excluded_criteria,
            criterion_alternatives=criterion_alternatives,
            criterion_subcriteria=criterion_subcriteria,
            valid_pairs=valid_pairs,
        )
        
        self.logger.debug(
            f"Created YearContext for {target_year}: "
            f"{len(active_provinces)} provinces, "
            f"{len(active_criteria)} criteria, "
            f"{len(active_subcriteria)} SCs"
        )
        
        return ctx

    def _create_forecast_decision_matrix(
        self,
        criteria_predictions: pd.DataFrame,
        year_context: YearContext,
        target_year: int,
        hierarchy: HierarchyMapping,
    ) -> pd.DataFrame:
        """
        Create decision matrix for forecast year, ready for MCDM ranking.

        **Step 14 (Phase 5)**: Wraps the aggregated 8 criteria predictions
        (63 × 8 array) in a decision matrix format compatible with
        HierarchicalRankingPipeline.rank(), which expects a pristine
        (NaN-free) decision matrix indexed by province and criterion.

        Parameters
        ----------
        criteria_predictions : pd.DataFrame
            Shape (n_entities, 8) aggregated criteria predictions.
            Index = province names; columns = ['C01', ..., 'C08'].

        year_context : YearContext
            YearContext for the forecast year (from _create_forecast_year_context).

        target_year : int
            Forecast year for logging.

        hierarchy : HierarchyMapping
            (Not directly used; present for consistency with other helpers.)

        Returns
        -------
        pd.DataFrame
            Same as input (criteria_predictions) — a validated, NaN-free decision matrix.
            Shape (n_active_alternatives, n_active_criteria).
            Ready to be passed to HierarchicalRankingPipeline.rank().

        Notes
        -----
        This method primarily serves as a validation and logging checkpoint.
        In future phases, it could:
        - Apply additional normalizations (SAW, minmax, etc.)
        - Enforce consistency with year_context (drop inactive entities/criteria)
        - Store metadata (timestamps, version info, etc.)
        """
        # Validate consistency with year_context
        assert set(criteria_predictions.index) == set(year_context.active_provinces), (
            f"Provinces mismatch between predictions "
            f"({set(criteria_predictions.index)}) and year_context "
            f"({set(year_context.active_provinces)})"
        )
        
        assert set(criteria_predictions.columns) == set(year_context.active_criteria), (
            f"Criteria mismatch between predictions "
            f"({set(criteria_predictions.columns)}) and year_context "
            f"({set(year_context.active_criteria)})"
        )
        
        # Assert no NaN in decision matrix
        n_nan = criteria_predictions.isna().sum().sum()
        assert n_nan == 0, (
            f"Decision matrix for {target_year} contains {n_nan} NaN cells. "
            "Forecast predictions should be complete."
        )
        
        # Return the decision matrix (already in correct format)
        self.logger.debug(
            f"Decision matrix for {target_year}: "
            f"{criteria_predictions.shape[0]} alternatives × {criteria_predictions.shape[1]} criteria, "
            f"all cells valid (no NaN)"
        )
        
        return criteria_predictions

    # -----------------------------------------------------------------
    # Phase 5: Enhanced Sensitivity Analysis
    # -----------------------------------------------------------------

    def _run_analysis(
        self,
        panel_data: PanelData,
        ranking_result: HierarchicalRankingResult,
        weights: Dict[str, Any],
        forecast_result: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run ML + ER sensitivity analysis.

        ML analysis covers: feature importance stability, LOO model impact,
        prediction sensitivity, temporal fold stability, interval width CV.
        ER analysis covers: criterion belief OAT sensitivity, grade threshold
        sensitivity, aggregation weight sensitivity, utility interval widths,
        cross-level consistency, belief entropy.
        """
        self.logger.info("Running ML + ER sensitivity analysis")

        er_result = getattr(ranking_result, 'er_result', None)
        n_boot = getattr(getattr(self.config, 'validation', None), 'n_simulations', 200)
        seed = getattr(getattr(self.config, 'random', None), 'seed', 42)

        ml_sens = None
        er_sens = None

        if forecast_result is not None:
            try:
                ml_sens = MLSensitivityAnalysis(
                    n_bootstrap=n_boot, seed=seed
                ).analyze(forecast_result)
                self.logger.info(
                    f"  ML robustness: {ml_sens.overall_robustness:.4f} "
                    f"(temporal_stability={ml_sens.temporal_prediction_stability:.4f})"
                )
            except Exception as exc:
                self.logger.warning(f"ML sensitivity failed (non-fatal): {exc}")

        if er_result is not None:
            try:
                er_sens = ERSensitivityAnalysis(
                    n_simulations=n_boot, seed=seed
                ).analyze(er_result, ranking_result=ranking_result)
                self.logger.info(
                    f"  ER robustness: {er_sens.overall_er_robustness:.4f} "
                    f"(mean_entropy={er_sens.mean_belief_entropy:.4f})"
                )
            except Exception as exc:
                self.logger.warning(f"ER sensitivity failed (non-fatal): {exc}")

        sens_result = CombinedSensitivityResult(
            ml_sensitivity=ml_sens, er_sensitivity=er_sens
        )
        self.logger.info(
            f"Combined robustness: {sens_result.overall_robustness:.4f}"
        )
        self._validate_sensitivity_result(sens_result)
        return {'sensitivity': sens_result}
    
    def _run_validation(
        self,
        panel_data: PanelData,
        weights: Dict[str, Any],
        ranking_result: HierarchicalRankingResult,
        forecast_result: Optional[Any] = None,
    ) -> Any:
        """Run ML + ER validation on the full pipeline."""
        from .analysis import Validator

        er_result = getattr(ranking_result, 'er_result', None)
        validator = Validator()
        val_result = validator.validate_full_pipeline(
            forecast_result=forecast_result,
            er_result=er_result,
            ranking_result=ranking_result,
        )

        self.logger.info(
            f"Validation: {'PASSED' if val_result.validation_passed else 'FAILED'}"
        )
        if val_result.validation_warnings:
            for w in val_result.validation_warnings:
                self.logger.warning(f"  {w}")
        return val_result

    def _validate_sensitivity_result(self, sens_result: Any) -> None:
        """Validate that sensitivity result exposes overall_robustness."""
        if not hasattr(sens_result, 'overall_robustness'):
            self.logger.error(
                'Sensitivity result is missing overall_robustness — '
                'CombinedSensitivityResult may not have been returned correctly.'
            )
        else:
            r = sens_result.overall_robustness
            if r < 0.3:
                self.logger.warning(
                    f'Low overall robustness score ({r:.4f}) — results may be '
                    f'sensitive to perturbations.'
                )

    # -----------------------------------------------------------------
    # Phase 6 & 7 — now handled by VisualizationOrchestrator and
    #               OutputOrchestrator respectively.
    # -----------------------------------------------------------------

# Convenience function
# =========================================================================

def run_pipeline(
    data_path: Optional[str] = None,
    config: Optional[Config] = None,
) -> PipelineResult:
    """Run the full pipeline. Returns PipelineResult."""
    pipeline = MLMCDMPipeline(config)
    return pipeline.run(data_path)
