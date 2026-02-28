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

import numpy as np
from typing import Any, Dict, List, Optional

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

    def __init__(self, output_dir: str = 'result/figures', dpi: int = 300):
        self.output_dir = output_dir
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

    # Weighting
    def plot_weights_comparison(self, *a, **kw):
        return self.weighting.plot_weights_comparison(*a, **kw)

    def plot_weight_radar(self, *a, **kw):
        return self.weighting.plot_weight_radar(*a, **kw)

    def plot_weight_heatmap(self, *a, **kw):
        return self.weighting.plot_weight_heatmap(*a, **kw)

    # MCDM
    def plot_method_agreement_matrix(self, *a, **kw):
        return self.mcdm.plot_method_agreement_matrix(*a, **kw)

    def plot_rank_parallel_coordinates(self, *a, **kw):
        return self.mcdm.plot_rank_parallel_coordinates(*a, **kw)

    def plot_criterion_scores(self, *a, **kw):
        return self.mcdm.plot_criterion_scores(*a, **kw)

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
    ) -> int:
        """Generate every applicable figure. Returns count produced."""
        count = 0

        def _inc(path):
            nonlocal count
            if path:
                count += 1

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
        _inc(self.ranking.plot_final_ranking(provinces, scores, ranks))
        _inc(self.ranking.plot_score_distribution(scores))

        # ── Weights ───────────────────────────────────────────────
        sc_arr = weights['sc_array']
        # Build per-criterion mean-weight view from Level 1 diagnostics
        w_dict: Dict[str, np.ndarray] = {'Hybrid Weighting': sc_arr}
        l1 = weights.get('details', {}).get('level1', {})
        crit_groups = {
            crit_id: data.get('local_sc_weights', {})
            for crit_id, data in l1.items()
        }
        if crit_groups:
            # Build a local-weight array for an "Entropy+CRITIC blend" reference
            local_arr = np.array([
                next(
                    (crit_groups[cid].get(sc, 0.0) for cid, lw in crit_groups.items() if sc in lw),
                    0.0,
                )
                for sc in subcriteria
            ])
            if local_arr.sum() > 0:
                local_arr = local_arr / local_arr.sum()
            w_dict['Level-1 Local'] = local_arr
        _inc(self.weighting.plot_weights_comparison(w_dict, subcriteria))
        _inc(self.weighting.plot_weight_radar(w_dict, subcriteria))
        _inc(self.weighting.plot_weight_heatmap(w_dict, subcriteria))

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
        if all_method_ranks:
            _inc(self.mcdm.plot_method_agreement_matrix(all_method_ranks))
            _inc(self.mcdm.plot_rank_parallel_coordinates(
                all_method_ranks, provinces,
            ))

        for crit_id, method_scores in ranking_result.criterion_method_scores.items():
            _inc(self.mcdm.plot_criterion_scores(
                method_scores, crit_id, top_n=20,
                save_name=f'fig08_{crit_id}_scores.png',
            ))

        # ── Sensitivity ──────────────────────────────────────────
        sens = analysis_results.get('sensitivity')
        if sens is not None:
            if hasattr(sens, 'criteria_sensitivity') and sens.criteria_sensitivity:
                _inc(self.sensitivity.plot_sensitivity_tornado(
                    sens.criteria_sensitivity,
                ))
            if hasattr(sens, 'subcriteria_sensitivity') and sens.subcriteria_sensitivity:
                _inc(self.sensitivity.plot_subcriteria_sensitivity(
                    sens.subcriteria_sensitivity,
                ))
            if hasattr(sens, 'top_n_stability') and sens.top_n_stability:
                _inc(self.sensitivity.plot_top_n_stability(
                    sens.top_n_stability,
                ))
            if hasattr(sens, 'temporal_stability') and sens.temporal_stability:
                _inc(self.sensitivity.plot_temporal_stability(
                    sens.temporal_stability,
                ))
            if hasattr(sens, 'rank_stability') and sens.rank_stability:
                _inc(self.sensitivity.plot_rank_volatility(
                    sens.rank_stability,
                ))
            if hasattr(sens, 'overall_robustness'):
                _inc(self.sensitivity.plot_robustness_summary(
                    sens.overall_robustness,
                    getattr(sens, 'confidence_level', 0.95),
                    getattr(sens, 'criteria_sensitivity', {}),
                    getattr(sens, 'top_n_stability', {}),
                ))

        # ER uncertainty
        try:
            unc = ranking_result.er_result.uncertainty
            _inc(self.sensitivity.plot_er_uncertainty(unc, provinces))
        except Exception:
            pass

        # ── Forecast ──────────────────────────────────────────────
        if forecast_result is not None:
            try:
                if hasattr(forecast_result, 'training_info'):
                    ti = forecast_result.training_info
                    actual = ti.get('y_test')
                    predicted = ti.get('y_pred')
                    if actual is not None and predicted is not None:
                        ent = ti.get('test_entities')
                        _inc(self.forecast.plot_forecast_scatter(
                            np.asarray(actual), np.asarray(predicted),
                            entity_names=ent,
                        ))
                        _inc(self.forecast.plot_forecast_residuals(
                            np.asarray(actual), np.asarray(predicted),
                        ))

                if hasattr(forecast_result, 'feature_importance'):
                    imp = forecast_result.feature_importance
                    if hasattr(imp, 'to_dict'):
                        imp_dict = (
                            imp['Importance'].to_dict()
                            if 'Importance' in imp.columns
                            else imp.iloc[:, 0].to_dict()
                        )
                    else:
                        imp_dict = imp
                    _inc(self.forecast.plot_feature_importance(imp_dict))

                if (hasattr(forecast_result, 'model_contributions')
                        and forecast_result.model_contributions):
                    _inc(self.forecast.plot_model_weights_donut(
                        forecast_result.model_contributions,
                    ))

                if (hasattr(forecast_result, 'model_performance')
                        and forecast_result.model_performance):
                    _inc(self.forecast.plot_model_performance(
                        forecast_result.model_performance,
                    ))

                if (hasattr(forecast_result, 'cross_validation_scores')
                        and forecast_result.cross_validation_scores):
                    _inc(self.forecast.plot_cv_boxplots(
                        forecast_result.cross_validation_scores,
                    ))

                if (hasattr(forecast_result, 'prediction_intervals')
                        and forecast_result.prediction_intervals):
                    preds = forecast_result.predictions
                    intervals = forecast_result.prediction_intervals
                    lower = intervals.get('lower')
                    upper = intervals.get('upper')
                    if lower is not None and upper is not None:
                        _inc(self.forecast.plot_prediction_intervals(
                            preds, lower, upper,
                        ))

                if (hasattr(forecast_result, 'predictions')
                        and forecast_result.predictions is not None):
                    pred_df = forecast_result.predictions
                    if len(pred_df.columns) > 0:
                        pred_col = pred_df.columns[0]
                        pred_scores = pred_df[pred_col].values
                        _inc(self.forecast.plot_rank_change_bubble(
                            provinces, scores, pred_scores,
                            prediction_year=getattr(
                                forecast_result, 'target_year',
                                max(panel_data.years) + 1,
                            ),
                        ))
            except Exception:
                pass

        # ── Executive dashboard ───────────────────────────────────
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
                    f'Confidence Level   : '
                    f'{getattr(sens, "confidence_level", 0.95):.0%}\n'
                )
            _inc(self.summary.plot_executive_dashboard({
                'kpis': kpis,
                'top_10': top10,
                'fused_weights': weights['sc_array'],
                'subcriteria_names': subcriteria,
                'robustness_text': rob_text,
            }))
        except Exception:
            pass

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
