# -*- coding: utf-8 -*-
"""
Visualization Orchestration Layer.

This package provides the `VisualizationOrchestrator`, which manages the 
instantiation and execution of all phase-specific plotters. It acts as a 
facade, delegating specific plotting tasks to specialized classes for 
weighting, ranking, and forecasting.

Key Features
------------
- **Centralized Control**: Manages unified output directories and DPI 
  settings across all visualization modules.
- **Fail-Safe Execution**: Provides a `generate_all` entry point with 
  per-figure error isolation, ensuring one plotting failure doesn't 
  block the entire pipeline.
- **Delegated Interface**: Exposes individual plotting methods from 
  sub-plotters for granular result exploration.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

from .base import BasePlotter, apply_style, HAS_MATPLOTLIB
from .ranking_plots import RankingPlotter
from .weighting_plots import WeightingPlotter
from .mcdm_plots import MCDMPlotter
from .mcdm_ranking_figures import MCDMRankingPlotter
from .sensitivity_plots import SensitivityPlotter
from .forecast_plots import ForecastPlotter
from .summary_plots import SummaryPlotter


class VisualizationOrchestrator:
    """
    Central coordinator for all analytical visualizations.

    Manages a suite of specialized plotters and provides a unified 
    `generate_all` interface for the main execution pipeline.

    Attributes
    ----------
    output_dir : str
        The root directory for all saved figures.
    ranking : RankingPlotter
        Plotter for stage-specific ranking diagnostics.
    mcdm_ranking : MCDMRankingPlotter
        Plotter for comparative MCDM method distributions.
    weighting : WeightingPlotter
        Plotter for criteria and sub-criteria weights.
    mcdm : MCDMPlotter
        Plotter for method agreement and stability.
    forecast : ForecastPlotter
        Plotter for ML model performance and intervals.
    """

    def __init__(
        self,
        output_dir: str = 'output/result/figures',
        dpi: int = 300,
        ranking_top_n: int = 20,
    ):
        """
        Initialize the orchestrator and its sub-plotters.

        Parameters
        ----------
        output_dir : str, default='output/result/figures'
            Root path for figure storage. Sub-directories will be created 
            automatically (e.g., 'ranking/', 'weighting/').
        dpi : int, default=300
            Resolution for all generated plots.
        ranking_top_n : int, default=20
            The number of alternatives to display in various top-N charts.
        """
        self.output_dir = output_dir

        # How many provinces to show in top-N charts (from VisualizationConfig)
        self._ranking_top_n: int = max(1, int(ranking_top_n))

        # Each plotter writes into its own phase subfolder
        self.ranking = RankingPlotter(f'{output_dir}/ranking', dpi)
        self.mcdm_ranking = MCDMRankingPlotter(f'{output_dir}/ranking', dpi)
        self.weighting = WeightingPlotter(f'{output_dir}/weighting', dpi)
        self.mcdm = MCDMPlotter(f'{output_dir}/ranking', dpi)  # fig06, fig08e, fig08f → ranking
        self.forecast = ForecastPlotter(f'{output_dir}/forecasting', dpi)
        self._plotters = [
            self.ranking, self.mcdm_ranking, self.weighting, self.mcdm,
            self.forecast,
        ]

    # ------------------------------------------------------------------
    # Delegated convenience methods
    # ------------------------------------------------------------------

    # Ranking
    def plot_final_ranking_summary(self, *a, **kw):
        return self.ranking.plot_final_ranking_summary(*a, **kw)

    def plot_multiyear_slopegraph(self, *a, **kw):
        return self.ranking.plot_multiyear_slopegraph(*a, **kw)

    def plot_belief_heatmap(self, *a, **kw):
        return self.ranking.plot_belief_heatmap(*a, **kw)

    def plot_rank_uncertainty_scatter(self, *a, **kw):
        return self.ranking.plot_rank_uncertainty_scatter(*a, **kw)

    def plot_mc_rank_uncertainty(self, *a, **kw):
        return self.ranking.plot_mc_rank_uncertainty(*a, **kw)

    # Weighting
    def plot_weights_comparison(self, *a, **kw):
        return self.weighting.plot_weights_comparison(*a, **kw)

    def plot_mc_weight_uncertainty(self, *a, **kw):
        return self.weighting.plot_mc_weight_uncertainty(*a, **kw)

    def plot_criterion_weights_comparison(self, *a, **kw):
        return self.weighting.plot_criterion_weights_comparison(*a, **kw)

    def plot_weight_deviation(self, *a, **kw):
        return self.weighting.plot_weight_deviation(*a, **kw)

    def plot_weight_radar(self, *a, **kw):
        return self.weighting.plot_weight_radar(*a, **kw)

    def plot_weight_heatmap(self, *a, **kw):
        return self.weighting.plot_weight_heatmap(*a, **kw)

    def plot_weight_radar_grouped(self, *a, **kw):
        return self.weighting.plot_weight_radar_grouped(*a, **kw)

    def plot_weight_hierarchical_rose(self, *a, **kw):
        return self.weighting.plot_weight_hierarchical_rose(*a, **kw)

    # MCDM
    def plot_method_agreement_matrix(self, *a, **kw):
        return self.mcdm.plot_method_agreement_matrix(*a, **kw)

    def plot_rank_parallel_coordinates(self, *a, **kw):
        return self.mcdm.plot_rank_parallel_coordinates(*a, **kw)

    def plot_mcdm_composite_scatter(self, *a, **kw):
        return self.mcdm.plot_mcdm_composite_scatter(*a, **kw)

    def plot_method_stability_comparison(self, *a, **kw):
        return self.mcdm.plot_method_stability_comparison(*a, **kw)

    def plot_method_disc_power_comparison(self, *a, **kw):
        return self.mcdm.plot_method_disc_power_comparison(*a, **kw)

    # MCDM Ranking (new ranking-specific visualizations)
    def plot_all_method_distributions(self, *a, **kw):
        """All 6 MCDM method score distribution charts."""
        return self.mcdm_ranking.plot_all_method_distributions(*a, **kw)

    def plot_method_agreement_matrix_ranking(self, *a, **kw):
        """6×6 method agreement matrix (Spearman correlation)."""
        return self.mcdm_ranking.plot_method_agreement_matrix(*a, **kw)

    def plot_kendall_w_analysis(self, *a, **kw):
        """Kendall's W concordance analysis."""
        return self.mcdm_ranking.plot_kendall_w_analysis(*a, **kw)

    def plot_rank_profiles(self, *a, **kw):
        """Rank profiles for top-N alternatives."""
        return self.mcdm_ranking.plot_rank_profiles(*a, **kw)

    def plot_method_boxplots(self, *a, **kw):
        """Method score distribution boxplots."""
        return self.mcdm_ranking.plot_method_boxplots(*a, **kw)

    def plot_method_scatter_comparison(self, *a, **kw):
        """Pairwise method scatter plot matrix."""
        return self.mcdm_ranking.plot_method_scatter_comparison(*a, **kw)

    def plot_perturbation_sensitivity(self, *a, **kw):
        """Perturbation sensitivity analysis."""
        return self.mcdm_ranking.plot_perturbation_sensitivity(*a, **kw)

    # Sensitivity
    def plot_sensitivity_tornado(self, *a, **kw):
        return self.sensitivity.plot_sensitivity_tornado(*a, **kw)

    def plot_sensitivity_analysis(self, *a, **kw):
        return self.sensitivity.plot_sensitivity_analysis(*a, **kw)

    def plot_subcriteria_sensitivity(self, *a, **kw):
        return self.sensitivity.plot_subcriteria_sensitivity(*a, **kw)

    def plot_top_n_stability(self, *a, **kw):
        return self.sensitivity.plot_top_n_stability(*a, **kw)

    def plot_temporal_stability(self, *a, **kw):
        return self.sensitivity.plot_temporal_stability(*a, **kw)

    def plot_rank_volatility(self, *a, **kw):
        return self.sensitivity.plot_rank_volatility(*a, **kw)



    def plot_tornado_butterfly(self, *a, **kw):
        return self.sensitivity.plot_tornado_butterfly(*a, **kw)

    def plot_subcriteria_dotstrip(self, *a, **kw):
        return self.sensitivity.plot_subcriteria_dotstrip(*a, **kw)

    def plot_stability_line_ci(self, *a, **kw):
        return self.sensitivity.plot_stability_line_ci(*a, **kw)

    def plot_rank_stability_scatter(self, *a, **kw):
        return self.sensitivity.plot_rank_stability_scatter(*a, **kw)

    def plot_rank_change_violin(self, *a, **kw):
        return self.sensitivity.plot_rank_change_violin(*a, **kw)

    # Forecast
    def plot_forecast_scatter(self, *a, **kw):
        return self.forecast.plot_forecast_scatter(*a, **kw)

    def plot_forecast_residuals(self, *a, **kw):
        return self.forecast.plot_forecast_residuals(*a, **kw)

    def plot_feature_importance(self, *a, **kw):
        return self.forecast.plot_feature_importance(*a, **kw)

    def plot_feature_importance_single(self, *a, **kw):
        return self.forecast.plot_feature_importance_single(*a, **kw)

    def plot_model_weights_donut(self, *a, **kw):
        return self.forecast.plot_model_weights_donut(*a, **kw)

    def plot_model_performance(self, *a, **kw):
        return self.forecast.plot_model_performance(*a, **kw)

    def plot_cv_boxplots(self, *a, **kw):
        return self.forecast.plot_cv_boxplots(*a, **kw)

    def plot_prediction_intervals(self, *a, **kw):
        return self.forecast.plot_prediction_intervals(*a, **kw)

    def plot_rank_change_bubble(self, *a, **kw):
        return self.forecast.plot_rank_change_bubble(*a, **kw)

    def plot_ensemble_architecture(self, *a, **kw):
        return self.forecast.plot_ensemble_architecture(*a, **kw)

    def plot_holdout_comparison(self, *a, **kw):
        return self.forecast.plot_holdout_comparison(*a, **kw)

    def plot_model_contribution_dots(self, *a, **kw):
        return self.forecast.plot_model_contribution_dots(*a, **kw)

    def plot_conformal_coverage(self, *a, **kw):
        return self.forecast.plot_conformal_coverage(*a, **kw)

    def plot_province_forecast_comparison(self, *a, **kw):
        return self.forecast.plot_province_forecast_comparison(*a, **kw)

    def plot_score_trajectory(self, *a, **kw):
        return self.forecast.plot_score_trajectory(*a, **kw)

    # Summary
    def plot_executive_dashboard(self, *a, **kw):
        return self.summary.plot_executive_dashboard(*a, **kw)

    # ------------------------------------------------------------------
    # Aggregated bookkeeping
    # ------------------------------------------------------------------

    def get_generated_figures(self) -> List[str]:
        """
        Retrieve a list of all figures generated across all plotters.

        Returns
        -------
        List[str]
            Absolute paths to all saved PNG files.
        """
        out: List[str] = []
        for p in self._plotters:
            out.extend(p.get_generated_figures())
        return out

    # ------------------------------------------------------------------
    # generate_all – single entry point
    # ------------------------------------------------------------------

    def generate_all(
        self,
        panel_data: Any,
        weights: Dict[str, Any],
        ranking_result: Any,
        analysis_results: Dict[str, Any],
        forecast_result: Any = None,
        multi_year_results: Optional[Dict[int, Any]] = None,
        weight_all_years: Optional[Dict[int, Any]] = None,
    ) -> int:
        """
        Execute the full visualization suite.

        Iteratively calls plotting methods for weighting, ranking, 
        stability, and forecasting results. Figures are skipped 
        gracefully if their required data is missing or if an 
        plotting-specific error occurs.

        Parameters
        ----------
        panel_data : PanelData
            Data containing temporal context for multi-year charts.
        weights : Dict[str, Any]
            Calculated weights from the CRITIC engine.
        ranking_result : RankingResult
            MCDM aggregation outcomes.
        analysis_results : Dict[str, Any]
            Sensitivity and validation metrics.
        forecast_result : UnifiedForecastResult, optional
            ML prediction intervals and performance stats.
        multi_year_results : Dict[int, Any], optional
            Results across all years for longitudinal charts.
        weight_all_years : Dict[int, Any], optional
            Weight variations over time.

        Returns
        -------
        int
            The total number of figures successfully saved to disk.
        """
        count = 0

        def _inc(path):
            nonlocal count
            if path:
                count += 1

        def _safe(fn, *args, **kwargs):
            """Call a plot function, catching any exception so one failure
            never aborts the remaining figures."""
            nonlocal count
            try:
                path = fn(*args, **kwargs)
                if path:
                    count += 1
            except Exception as _exc:
                _logger.warning(
                    'Figure skipped [%s]: %s',
                    getattr(fn, '__name__', repr(fn)), _exc,
                )

        # Use the active province list from the result's index so that
        # dynamically-excluded provinces never appear in figures.
        provinces = (
            list(ranking_result.final_scores.index)
            if hasattr(ranking_result.final_scores, 'index')
            else list(panel_data.provinces)
        )
        scores = np.asarray(
            ranking_result.final_scores.values
            if hasattr(ranking_result.final_scores, 'values')
            else ranking_result.final_scores,
        )
        ranks = np.asarray(
            ranking_result.final_ranking.values
            if hasattr(ranking_result.final_ranking, 'values')
            else ranking_result.final_ranking,
        )
        subcriteria = weights['subcriteria']

        # ── Ranking ───────────────────────────────────────────────
        # fig01c — multi-year slope graph (only when multi-year data exists)
        if multi_year_results:
            _safe(self.ranking.plot_multiyear_slopegraph,
                multi_year_results,
                top_n=getattr(self, '_ranking_top_n', 20),
            )

        # fig01d and fig01e skipped (ER removed)

        # fig02b — MC rank-uncertainty error-bar chart
        mc_stats = weights.get('mc_province_stats', {})
        if mc_stats:
            _safe(self.ranking.plot_mc_rank_uncertainty, mc_stats)

        # ── Weights ───────────────────────────────────────────────
        sc_arr = weights['sc_array']
        details = weights.get('details', {})
        l1 = details.get('level1', {})
        crit_groups_map = weights.get('criteria_groups', {})

        # Build w_dict with CRITIC weights
        w_dict: Dict[str, np.ndarray] = {'CRITIC': sc_arr}

        # fig03b — (MC weight uncertainty removed — deterministic pipeline)
        # fig03d — (Diverging deviation removed — no Hybrid baseline)
        # (fig03, fig03c, fig04a, fig04b removed — redundant per spec)

        # fig04 — Weight radar: 14-panel all-years grid, or single-year fallback
        _safe(self.weighting.plot_weight_radar, w_dict, subcriteria,
              weight_all_years=weight_all_years)

        # fig04c — Hierarchical rose: 14-panel all-years grid, or single-year fallback
        _safe(self.weighting.plot_weight_hierarchical_rose, w_dict, subcriteria,
              weight_all_years=weight_all_years)

        # fig05 — Weight heatmap: years × sub-criteria (14 years), or single-year fallback
        _safe(self.weighting.plot_weight_heatmap, w_dict, subcriteria,
              weight_all_years=weight_all_years)

        # ── MCDM agreement ───────────────────────────────────────
        all_method_ranks: Dict[str, np.ndarray] = {}
        for crit_id, method_ranks in ranking_result.criterion_method_ranks.items():
            for method, rank_series in method_ranks.items():
                col = f'{crit_id}_{method}'
                all_method_ranks[col] = (
                    rank_series.values
                    if hasattr(rank_series, 'values')
                    else np.asarray(rank_series)
                )

        # fig06 — clustered Spearman agreement heatmap
        if all_method_ranks:
            _safe(self.mcdm.plot_method_agreement_matrix, all_method_ranks)

        # fig08e — Method stability comparison (cross-criteria Spearman ρ)
        _safe(self.mcdm.plot_method_stability_comparison, ranking_result)

        # fig08f — Method discriminatory power comparison (score IQR)
        _safe(self.mcdm.plot_method_disc_power_comparison, ranking_result)

        # ── MCDM Ranking (New professional ranking visualizations) ──────
        if multi_year_results:
            # fig09* — Individual method score distributions (6 charts)
            _safe(self.mcdm_ranking.plot_all_method_distributions, multi_year_results)

            # fig10 — Method agreement matrix (Spearman 6×6 heatmap)
            _safe(self.mcdm_ranking.plot_method_agreement_matrix, multi_year_results)

            # fig11 — Kendall's W concordance analysis (4-panel interpretation)
            _safe(self.mcdm_ranking.plot_kendall_w_analysis, multi_year_results)

            # fig12 — Rank profiles (parallel coordinates for top-N)
            _safe(self.mcdm_ranking.plot_rank_profiles, multi_year_results)

            # fig13 — Method boxplots (score distribution comparison)
            _safe(self.mcdm_ranking.plot_method_boxplots, multi_year_results)

            # fig14 — Method scatter matrix (pairwise 6×6 comparisons)
            _safe(self.mcdm_ranking.plot_method_scatter_comparison, multi_year_results)

            # fig15 — Perturbation sensitivity & robustness analysis
            _safe(self.mcdm_ranking.plot_perturbation_sensitivity, multi_year_results)

        # ── Sensitivity ──────────────────────────────────────────
        sens = analysis_results.get('sensitivity')
        if sens is not None:
            crit_sens = getattr(sens, 'criteria_sensitivity', {})
            sc_sens   = getattr(sens, 'subcriteria_sensitivity', {})
            rank_stab = getattr(sens, 'rank_stability', {})
            top_n_stab = getattr(sens, 'top_n_stability', {})
            pert_a    = getattr(sens, 'perturbation_analysis', {})

            # fig09 — classic criteria tornado
            if crit_sens:
                _safe(self.sensitivity.plot_sensitivity_tornado, crit_sens)

            # fig09b — butterfly tornado (two-sided diverging)
            if crit_sens:
                _safe(self.sensitivity.plot_tornado_butterfly,
                    crit_sens, perturbation_analysis=pert_a,
                )

            # fig09c — subcriteria dot/strip grouped by criterion
            if sc_sens:
                _safe(self.sensitivity.plot_subcriteria_dotstrip, sc_sens)

            # fig10 — subcriteria bar (top-20; backward-compat)
            if sc_sens:
                _safe(self.sensitivity.plot_subcriteria_sensitivity, sc_sens)

            # fig11 — top-N stability
            if top_n_stab:
                _safe(self.sensitivity.plot_top_n_stability, top_n_stab)

            # fig12 — temporal stability
            temp_stab = getattr(sens, 'temporal_stability', {})
            if temp_stab:
                _safe(self.sensitivity.plot_temporal_stability, temp_stab)

            # fig13 — rank volatility bar (all provinces)
            if rank_stab:
                _safe(self.sensitivity.plot_rank_volatility, rank_stab)

            # fig13b — stability sorted line + CI bands
            if rank_stab:
                _safe(self.sensitivity.plot_stability_line_ci, rank_stab)

            # fig14 — rank vs stability scatter (quadrant analysis)
            try:
                final_rk = ranking_result.final_ranking
                if rank_stab and final_rk:
                    _inc(self.sensitivity.plot_rank_stability_scatter(
                        final_rk, rank_stab,
                    ))
            except Exception as _exc:
                _logger.debug('rank-stability scatter skipped: %s', _exc)

            # fig14b — violin of rank-change distributions
            if pert_a:
                _safe(self.sensitivity.plot_rank_change_violin,
                    pert_a, provinces=provinces,
                )

        # ── Forecast ──────────────────────────────────────────────
        if forecast_result is not None:
            if hasattr(forecast_result, 'forecast_metadata') and forecast_result.forecast_metadata:
                self.forecast.set_metadata(forecast_result.forecast_metadata)
                
            # Extract all data up-front; each figure call is independent via
            # _safe() — one failure never aborts the remaining figures.
            _fr = forecast_result  # convenience alias
            _ti = getattr(_fr, 'training_info', {}) or {}
            _actual    = _ti.get('y_test')
            _predicted = _ti.get('y_pred')
            _ent       = _ti.get('test_entities')
            _contribs  = getattr(_fr, 'model_contributions', {}) or {}
            _cv_scores = getattr(_fr, 'cross_validation_scores', {}) or {}
            _intervals = getattr(_fr, 'prediction_intervals', {}) or {}
            _pred_df   = getattr(_fr, 'predictions', None)
            _pred_year = getattr(_fr, 'target_year', max(panel_data.years) + 1)

            # Per-model holdout predictions (populated when holdout year exists)
            _per_model_ho  = _ti.get('per_model_holdout_predictions')
            # Phase D: new data keys
            _per_model_oof = _ti.get('per_model_oof_predictions')
            _per_model_fi  = _ti.get('per_model_feature_importance')
            _cv_fold_years = _ti.get('cv_fold_val_years')
            # Feature names for fig18b (from aggregated importance DataFrame)
            _feat_names: Optional[List[str]] = None
            if hasattr(_fr, 'feature_importance'):
                _fi_df = _fr.feature_importance
                if hasattr(_fi_df, 'index'):
                    _feat_names = list(_fi_df.index)

            # fig16b — ensemble architecture flowchart (data-driven model list)
            _safe(self.forecast.plot_ensemble_architecture,
                  model_names=list(_contribs.keys()) if _contribs else None,
                  model_contributions=_contribs or None)

            # fig16 / fig17 / fig22b / fig16c — require actual vs predicted data
            if _actual is not None and _predicted is not None:
                _a = np.asarray(_actual)
                _p = np.asarray(_predicted)
                # fig16 — actual vs predicted scatter
                _safe(self.forecast.plot_forecast_scatter,
                      _a, _p, entity_names=_ent)
                # fig17 — residual diagnostics (4-panel)
                _safe(self.forecast.plot_forecast_residuals, _a, _p)
                # fig22b — conformal coverage calibration curve
                _safe(self.forecast.plot_conformal_coverage, _a, _p)
                # fig16c — holdout comparison (per-model lines + ensemble)
                _safe(self.forecast.plot_holdout_comparison,
                      _a, _p,
                      per_model_predictions=_per_model_ho,
                      entity_names=_ent,
                      model_contributions=_contribs or None)

            # fig18 — feature importance lollipop
            if hasattr(_fr, 'feature_importance'):
                _imp = _fr.feature_importance
                if hasattr(_imp, 'columns'):
                    _imp_dict = (
                        _imp['Importance'].to_dict()
                        if 'Importance' in _imp.columns
                        else _imp.iloc[:, 0].to_dict()
                    )
                else:
                    _imp_dict = _imp
                _safe(self.forecast.plot_feature_importance, _imp_dict)

            # fig19 — model weights donut
            if _contribs:
                _safe(self.forecast.plot_model_weights_donut, _contribs)

            # fig19b — model contribution bubble (weight vs CV R²)
            if _contribs:
                _safe(self.forecast.plot_model_contribution_dots,
                      _contribs,
                      cross_validation_scores=_cv_scores or None,
                      model_performance=getattr(_fr, 'model_performance', None))

            # fig20 — per-model performance comparison bars
            if getattr(_fr, 'model_performance', None):
                _safe(self.forecast.plot_model_performance,
                      _fr.model_performance)

            # fig21 — CV box plots (per-fold R² distribution)
            if _cv_scores:
                _safe(self.forecast.plot_cv_boxplots, _cv_scores)

            # fig22 — conformal prediction intervals (top-N provinces)
            if _intervals and _pred_df is not None:
                _lower = _intervals.get('lower')
                _upper = _intervals.get('upper')
                if _lower is not None and _upper is not None:
                    _safe(self.forecast.plot_prediction_intervals,
                          _pred_df, _lower, _upper)

            # fig23 / fig23b / fig23c — require predictions DataFrame
            if _pred_df is not None and len(_pred_df.columns) > 0:
                _pc = _pred_df.columns[0]
                _ps = _pred_df[_pc].values
                # fig23 — rank change bubble (historical vs forecast)
                _safe(self.forecast.plot_rank_change_bubble,
                      provinces, scores, _ps, prediction_year=_pred_year)
                # fig23b — actual vs forecast grouped bar (top-N provinces)
                _safe(self.forecast.plot_province_forecast_comparison,
                      provinces, scores, _pred_df,
                      intervals=_intervals or None,
                      prediction_year=_pred_year)
                # fig23c — score trajectory: historical line + forecast CI
                _safe(self.forecast.plot_score_trajectory,
                      provinces, scores, _pred_df,
                      intervals=_intervals or None,
                      panel_years=list(panel_data.years),
                      target_year=_pred_year)

            # ── Phase D: 9 new analytical figures ────────────────────── #

            # fig17b — residual distributions per model + ensemble
            if (
                _actual is not None and _predicted is not None
                and _per_model_oof
            ):
                _safe(self.forecast.plot_residual_distributions,
                      np.asarray(_actual), np.asarray(_predicted),
                      per_model_predictions=_per_model_oof)

            # fig18b — per-model feature importance heatmap
            if _per_model_fi:
                _safe(self.forecast.plot_per_model_importance_heatmap,
                      _per_model_fi, feature_names=_feat_names)

            # fig20b — multi-metric radar (or R² bar fallback)
            _mp = getattr(_fr, 'model_performance', None) or {}
            if _mp or _per_model_oof:
                _safe(self.forecast.plot_model_metric_radar,
                      _mp,
                      per_model_predictions=_per_model_oof or None,
                      y_test=(np.asarray(_actual)
                               if _actual is not None else None))

            # fig20c — bootstrap metric confidence intervals
            if _actual is not None:
                _safe(self.forecast.plot_bootstrap_metric_ci,
                      np.asarray(_actual),
                      per_model_predictions=_per_model_oof or None,
                      y_pred_ensemble=(np.asarray(_predicted)
                                       if _predicted is not None else None))

            # fig22c — interval calibration scatter
            if (
                _actual is not None and _predicted is not None
                and _intervals
            ):
                _lower_ci = _intervals.get('lower')
                _upper_ci = _intervals.get('upper')
                if _lower_ci is not None and _upper_ci is not None:
                    _safe(self.forecast.plot_interval_calibration_scatter,
                          np.asarray(_actual), np.asarray(_predicted),
                          _lower_ci, _upper_ci,
                          entity_names=_ent)

            # fig23d — entity error analysis (sorted horizontal bars)
            if _actual is not None and _predicted is not None:
                _safe(self.forecast.plot_entity_error_analysis,
                      np.asarray(_actual), np.asarray(_predicted),
                      entity_names=_ent)

            # fig23e — temporal training curve (walk-forward CV R² vs year)
            if _cv_scores:
                _safe(self.forecast.plot_temporal_training_curve,
                      _cv_scores,
                      cv_fold_val_years=_cv_fold_years)

            # fig24a — pairwise OOF prediction correlation heatmap
            if _per_model_oof and len(_per_model_oof) > 1:
                _safe(self.forecast.plot_prediction_correlation_heatmap,
                      _per_model_oof)

            # fig24b — pairwise OOF prediction scatter matrix
            if _per_model_oof and len(_per_model_oof) > 1:
                _safe(self.forecast.plot_prediction_scatter_matrix,
                      _per_model_oof)

        return count


# Backward-compatible factory
def create_visualizer(output_dir: str = 'output/result/figures') -> VisualizationOrchestrator:
    """
    Factory function for VisualizationOrchestrator.

    Parameters
    ----------
    output_dir : str, default='output/result/figures'
        Root path for storage.

    Returns
    -------
    VisualizationOrchestrator
        A configured instance of the orchestrator.
    """
    return VisualizationOrchestrator(output_dir=output_dir)


__all__ = [
    'VisualizationOrchestrator',
    'create_visualizer',
    'BasePlotter',
    'RankingPlotter',
    'WeightingPlotter',
    'MCDMPlotter',
    'SensitivityPlotter',
    'ForecastPlotter',
    'SummaryPlotter',
]
