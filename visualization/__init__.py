# -*- coding: utf-8 -*-
"""
Visualization Package
=====================

Modular, publication-quality figure generation split into six
phase-specific plotter classes coordinated by a single
``VisualizationOrchestrator``.

Quick start::

    from visualization import VisualizationOrchestrator
    viz = VisualizationOrchestrator('result/figures')
    count = viz.generate_all(panel_data, weights, ranking_result, ...)
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
from .sensitivity_plots import SensitivityPlotter
from .forecast_plots import ForecastPlotter
from .summary_plots import SummaryPlotter


class VisualizationOrchestrator:
    """
    Coordinate all phase-specific plotters and expose a single
    ``generate_all()`` entry point that replaces the old monolithic
    ``PanelVisualizer.generate_all()`` and ``pipeline._generate_all_visualizations()``.

    Also exposes every individual ``plot_*`` method via delegation so
    that callers do not need to know which sub-plotter owns which figure.
    """

    def __init__(
        self,
        output_dir: str = 'result/figures',
        dpi: int = 300,
        ranking_top_n: int = 20,
    ):
        self.output_dir = output_dir

        # How many provinces to show in top-N charts (from VisualizationConfig)
        self._ranking_top_n: int = max(1, int(ranking_top_n))

        # Each plotter writes into its own phase subfolder
        self.ranking = RankingPlotter(f'{output_dir}/ranking', dpi)
        self.weighting = WeightingPlotter(f'{output_dir}/weighting', dpi)
        self.mcdm = MCDMPlotter(f'{output_dir}/mcdm', dpi)
        self.sensitivity = SensitivityPlotter(f'{output_dir}/sensitivity', dpi)
        self.forecast = ForecastPlotter(f'{output_dir}/forecasting', dpi)
        self.summary = SummaryPlotter(f'{output_dir}/summary', dpi)
        self._plotters = [
            self.ranking, self.weighting, self.mcdm,
            self.sensitivity, self.forecast, self.summary,
        ]

    # ------------------------------------------------------------------
    # Delegated convenience methods
    # ------------------------------------------------------------------

    # Ranking
    def plot_final_ranking(self, *a, **kw):
        return self.ranking.plot_final_ranking(*a, **kw)

    def plot_final_ranking_summary(self, *a, **kw):
        return self.ranking.plot_final_ranking_summary(*a, **kw)

    def plot_score_distribution(self, *a, **kw):
        return self.ranking.plot_score_distribution(*a, **kw)

    def plot_tier_ranking(self, *a, **kw):
        return self.ranking.plot_tier_ranking(*a, **kw)

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

    def plot_method_agreement_per_criterion(self, *a, **kw):
        return self.mcdm.plot_method_agreement_per_criterion(*a, **kw)

    def plot_rank_parallel_coordinates(self, *a, **kw):
        return self.mcdm.plot_rank_parallel_coordinates(*a, **kw)

    def plot_criterion_parallel_grid(self, *a, **kw):
        return self.mcdm.plot_criterion_parallel_grid(*a, **kw)

    def plot_criterion_scores(self, *a, **kw):
        return self.mcdm.plot_criterion_scores(*a, **kw)

    def plot_mcdm_composite_scatter(self, *a, **kw):
        return self.mcdm.plot_mcdm_composite_scatter(*a, **kw)

    def plot_criterion_er_utility_heatmap(self, *a, **kw):
        return self.mcdm.plot_criterion_er_utility_heatmap(*a, **kw)

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

    def plot_er_uncertainty(self, *a, **kw):
        return self.sensitivity.plot_er_uncertainty(*a, **kw)

    def plot_robustness_summary(self, *a, **kw):
        return self.sensitivity.plot_robustness_summary(*a, **kw)

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
        """All figure paths across every sub-plotter."""
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
    ) -> int:
        """Generate every applicable figure. Returns count produced."""
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
        # fig01 — lollipop gradient ranking
        _safe(self.ranking.plot_final_ranking, provinces, scores, ranks)

        # fig01b — tier-band lollipop
        _safe(self.ranking.plot_tier_ranking, provinces, scores, ranks)

        # fig01c — multi-year slope graph (only when multi-year data exists)
        if multi_year_results:
            _safe(self.ranking.plot_multiyear_slopegraph,
                multi_year_results,
                top_n=getattr(self, '_ranking_top_n', 20),
            )

        # fig01d — ER belief distribution heatmap
        _safe(self.ranking.plot_belief_heatmap, ranking_result, provinces)

        # fig01e — rank vs uncertainty scatter
        _safe(self.ranking.plot_rank_uncertainty_scatter, ranking_result, provinces)

        # fig02 — score distribution histogram + KDE
        _safe(self.ranking.plot_score_distribution, scores)

        # fig02b — MC rank-uncertainty error-bar chart
        mc_stats = weights.get('mc_province_stats', {})
        if mc_stats:
            _safe(self.ranking.plot_mc_rank_uncertainty, mc_stats)

        # ── Weights ───────────────────────────────────────────────
        sc_arr = weights['sc_array']
        details = weights.get('details', {})
        l1 = details.get('level1', {})
        crit_groups_map = weights.get('criteria_groups', {})

        # Retrieve entropy / CRITIC / hybrid SC weight arrays
        entropy_sc_d  = weights.get('entropy_sc_weights', {})
        critic_sc_d   = weights.get('critic_sc_weights', {})
        hybrid_sc_d   = dict(zip(subcriteria, sc_arr))
        entropy_sc_arr = np.array([float(entropy_sc_d.get(sc, 0.0)) for sc in subcriteria])
        critic_sc_arr  = np.array([float(critic_sc_d.get(sc, 0.0))  for sc in subcriteria])

        # Build w_dict for comparison (only include methods with data)
        w_dict: Dict[str, np.ndarray] = {'Hybrid': sc_arr}
        if entropy_sc_arr.sum() > 1e-9:
            w_dict['Entropy'] = entropy_sc_arr
        if critic_sc_arr.sum() > 1e-9:
            w_dict['CRITIC'] = critic_sc_arr

        # MC CI arrays from level2 diagnostics
        l2mc = details.get('level2', {}).get('mc_diagnostics', {})
        mc_means_full = np.array([float(l2mc.get('mean_weights', {}).get(sc, sc_arr[i]))
                                   for i, sc in enumerate(subcriteria)])
        mc_stds_full  = np.array([float(l2mc.get('std_weights',  {}).get(sc, 0.0))
                                   for i, sc in enumerate(subcriteria)])
        ci_lo = np.array([float(l2mc.get('ci_lower_2_5',  {}).get(sc, sc_arr[i]))
                          for i, sc in enumerate(subcriteria)])
        ci_hi = np.array([float(l2mc.get('ci_upper_97_5', {}).get(sc, sc_arr[i]))
                          for i, sc in enumerate(subcriteria)])

        # fig03 — Three-method grouped bar with CI
        _safe(self.weighting.plot_weights_comparison,
            w_dict, subcriteria,
            ci_lower=ci_lo if ci_lo.any() else None,
            ci_upper=ci_hi if ci_hi.any() else None,
        )

        # fig03b — MC weight uncertainty error-bar chart
        if mc_stds_full.any():
            _safe(self.weighting.plot_mc_weight_uncertainty,
                subcriteria,
                mc_means=mc_means_full,
                mc_stds=mc_stds_full,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                criteria_groups=crit_groups_map if crit_groups_map else None,
            )

        # fig03c — Criterion-level three-method horizontal bar + donut
        entropy_crit_d = weights.get('entropy_criterion_weights', {})
        critic_crit_d  = weights.get('critic_criterion_weights', {})
        hybrid_crit_d  = weights.get('criterion_weights', {})
        if entropy_crit_d or critic_crit_d:
            _safe(self.weighting.plot_criterion_weights_comparison,
                entropy_crit=entropy_crit_d,
                critic_crit=critic_crit_d,
                hybrid_crit=hybrid_crit_d,
            )

        # fig03d — Diverging deviation from hybrid baseline
        if entropy_sc_arr.sum() > 1e-9 or critic_sc_arr.sum() > 1e-9:
            _safe(self.weighting.plot_weight_deviation,
                entropy_sc=entropy_sc_d,
                critic_sc=critic_sc_d,
                hybrid_sc=hybrid_sc_d,
                subcriteria=subcriteria,
            )

        # fig04 — Radar (all 3 methods, all 29 sub-criteria)
        _safe(self.weighting.plot_weight_radar, w_dict, subcriteria)

        # fig04a + fig04b — Criteria-level radar + 8 group radars
        #   (locally-normalised weights per group)
        _safe(self.weighting.plot_weight_radar_grouped, w_dict, subcriteria)

        # fig04c — Hierarchical rose / coxcomb chart  [creative]
        _safe(self.weighting.plot_weight_hierarchical_rose, w_dict, subcriteria)

        # fig05 — Heatmap (sequential SC order, no dendrogram)
        _safe(self.weighting.plot_weight_heatmap, w_dict, subcriteria)

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

        # fig06b — per-criterion avg Spearman bar
        _safe(self.mcdm.plot_method_agreement_per_criterion, ranking_result)

        # fig07 — 2×4 grid of per-criterion parallel-coord panels
        _safe(self.mcdm.plot_criterion_parallel_grid, ranking_result, provinces)

        # fig08 — per-criterion method score panels (one file per criterion)
        for crit_id, method_scores in ranking_result.criterion_method_scores.items():
            _safe(self.mcdm.plot_criterion_scores,
                method_scores, crit_id, top_n=5,
                save_name=f'fig08_{crit_id}_scores.png',
            )

        # fig08b — MCDM composite vs ER final score scatter
        _safe(self.mcdm.plot_mcdm_composite_scatter, ranking_result, provinces)

        # fig08c — Province × Criterion ER utility heatmap
        _safe(self.mcdm.plot_criterion_er_utility_heatmap, ranking_result, provinces)


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

            # fig25 — robustness summary infographic
            if hasattr(sens, 'overall_robustness'):
                _safe(self.sensitivity.plot_robustness_summary,
                    sens.overall_robustness,
                    getattr(sens, 'confidence_level', 0.95),
                    crit_sens,
                    top_n_stab,
                )

        # ER uncertainty (fig15)
        try:
            unc = ranking_result.er_result.uncertainty
            _inc(self.sensitivity.plot_er_uncertainty(unc, provinces))
        except Exception as _exc:
            _logger.debug('ER uncertainty plot skipped: %s', _exc)

        # ── Forecast ──────────────────────────────────────────────
        if forecast_result is not None:
            try:
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

                # Holdout per-model predictions
                _per_model_ho = _ti.get('per_model_holdout_predictions')

                # fig16 — actual vs predicted scatter
                if _actual is not None and _predicted is not None:
                    _a = np.asarray(_actual)
                    _p = np.asarray(_predicted)
                    _inc(self.forecast.plot_forecast_scatter(
                        _a, _p, entity_names=_ent,
                    ))
                    # fig17 — residual diagnostics (4-panel)
                    _inc(self.forecast.plot_forecast_residuals(_a, _p))
                    # fig22b — conformal coverage calibration curve
                    _inc(self.forecast.plot_conformal_coverage(_a, _p))
                    # fig16c — comprehensive holdout comparison
                    _inc(self.forecast.plot_holdout_comparison(
                        _a, _p,
                        per_model_predictions=_per_model_ho,
                        entity_names=_ent,
                        model_contributions=_contribs or None,
                    ))

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
                    _inc(self.forecast.plot_feature_importance(_imp_dict))

                # fig19 — model weights donut
                if _contribs:
                    _inc(self.forecast.plot_model_weights_donut(_contribs))

                # fig19b — model contribution bubble (weight vs CV R²)
                if _contribs:
                    _inc(self.forecast.plot_model_contribution_dots(
                        _contribs,
                        cross_validation_scores=_cv_scores or None,
                        model_performance=getattr(_fr, 'model_performance', None),
                    ))

                # fig20 — model performance comparison
                if getattr(_fr, 'model_performance', None):
                    _inc(self.forecast.plot_model_performance(
                        _fr.model_performance,
                    ))

                # fig21 — CV box plots
                if _cv_scores:
                    _inc(self.forecast.plot_cv_boxplots(_cv_scores))

                # fig22 — conformal prediction intervals (top-N)
                if _intervals and _pred_df is not None:
                    _lower = _intervals.get('lower')
                    _upper = _intervals.get('upper')
                    if _lower is not None and _upper is not None:
                        _inc(self.forecast.plot_prediction_intervals(
                            _pred_df, _lower, _upper,
                        ))

                # fig23 — rank change bubble
                if _pred_df is not None and len(_pred_df.columns) > 0:
                    _pc = _pred_df.columns[0]
                    _ps = _pred_df[_pc].values
                    _inc(self.forecast.plot_rank_change_bubble(
                        provinces, scores, _ps, prediction_year=_pred_year,
                    ))
                    # fig23b — actual vs forecast grouped bar (top-N provinces)
                    _inc(self.forecast.plot_province_forecast_comparison(
                        provinces, scores, _pred_df,
                        intervals=_intervals or None,
                        prediction_year=_pred_year,
                    ))
                    # fig23c — score trajectory: historical line + forecast CI
                    _inc(self.forecast.plot_score_trajectory(
                        provinces, scores, _pred_df,
                        intervals=_intervals or None,
                        panel_years=list(panel_data.years),
                        target_year=_pred_year,
                    ))
            except Exception as _exc:
                _logger.warning('Forecast visualization failed: %s', _exc)

        return count


# Backward-compatible factory
def create_visualizer(output_dir: str = 'result/figures') -> VisualizationOrchestrator:
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
