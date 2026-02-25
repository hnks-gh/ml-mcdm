# -*- coding: utf-8 -*-
"""
CSV & JSON Data Writer for ML-MCDM Pipeline
============================================

All structured numerical output (weights, rankings, scores, forecasts,
sensitivity analysis) is persisted through this single writer class.
Every file lands in ``result/results/``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

_logger = logging.getLogger(__name__)


class CsvWriter:
    """Write CSV / JSON result files into ``<base_dir>/results/``."""

    def __init__(self, base_output_dir: str = 'result'):
        self.results_dir = Path(base_output_dir) / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._saved_files: List[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record(self, path: Path) -> str:
        s = str(path)
        self._saved_files.append(s)
        return s

    def _save_csv(self, df: pd.DataFrame, name: str,
                  float_fmt: str = '%.6f', **kwargs) -> str:
        path = self.results_dir / name
        df.to_csv(path, float_format=float_fmt, **kwargs)
        return self._record(path)

    def _save_json(self, obj: Any, name: str) -> str:
        path = self.results_dir / name
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, default=str, ensure_ascii=False)
        return self._record(path)

    def get_saved_files(self) -> List[str]:
        return list(self._saved_files)

    # ==================================================================
    #  1. WEIGHTS
    # ==================================================================

    def save_weights(self, weights: Dict[str, Any],
                     subcriteria: List[str]) -> str:
        """Save hybrid weighting results to weights_analysis.csv."""
        global_sc_w  = weights.get('global_sc_weights', {})
        criterion_w  = weights.get('criterion_weights', {})
        sc_arr       = np.asarray(weights.get('sc_array', [global_sc_w.get(sc, 0.0) for sc in subcriteria]))
        details      = weights.get('details', {})
        l1           = details.get('level1', {})

        # Build per-SC lookup for Level 1 diagnostics
        sc_to_crit: Dict[str, str] = {}
        for crit_id, sc_list in {
            cid: data.get('local_sc_weights', {}).keys()
            for cid, data in l1.items()
        }.items():
            for sc in sc_list:
                sc_to_crit[sc] = crit_id

        rows = []
        for i, sc in enumerate(subcriteria):
            crit_id = sc_to_crit.get(sc, '')
            l1_entry = l1.get(crit_id, {})
            local_w  = l1_entry.get('local_sc_weights', {})
            mc       = l1_entry.get('mc_diagnostics', {})
            row = {
                'Subcriteria':    sc,
                'Criterion':      crit_id,
                'Global_Weight':  float(global_sc_w.get(sc, sc_arr[i] if i < len(sc_arr) else 0.0)),
                'Criterion_Weight': float(criterion_w.get(crit_id, 0.0)),
                'Local_SC_Weight': float(local_w.get(sc, 0.0)),
                'MC_Mean':        float(mc.get('mean_weights', {}).get(sc, 0.0)),
                'MC_Std':         float(mc.get('std_weights', {}).get(sc, 0.0)),
                'MC_CV':          float(mc.get('cv_weights', {}).get(sc, 0.0)),
                'CI_Lower_2_5':   float(mc.get('ci_lower_2_5', {}).get(sc, 0.0)),
                'CI_Upper_97_5':  float(mc.get('ci_upper_97_5', {}).get(sc, 0.0)),
            }
            rows.append(row)

        df = pd.DataFrame(rows).set_index('Subcriteria')
        df['Rank_Global'] = df['Global_Weight'].rank(ascending=False, method='min').astype(int)

        # Append criterion-level summary rows at the bottom
        crit_rows = []
        for crit_id, v in sorted(criterion_w.items()):
            l2_diag = details.get('level2', {}).get('mc_diagnostics', {})
            crit_rows.append({
                'Subcriteria': f'[CRITERION] {crit_id}',
                'Criterion': crit_id,
                'Global_Weight': float(v),
                'Criterion_Weight': float(v),
                'Local_SC_Weight': float(v),
                'MC_Mean': float(l2_diag.get('mean_weights', {}).get(crit_id, v)),
                'MC_Std':  float(l2_diag.get('std_weights', {}).get(crit_id, 0.0)),
                'MC_CV':   float(l2_diag.get('cv_weights', {}).get(crit_id, 0.0)),
                'CI_Lower_2_5':  float(l2_diag.get('ci_lower_2_5', {}).get(crit_id, 0.0)),
                'CI_Upper_97_5': float(l2_diag.get('ci_upper_97_5', {}).get(crit_id, 0.0)),
                'Rank_Global': 0,
            })

        if crit_rows:
            crit_df = pd.DataFrame(crit_rows).set_index('Subcriteria')
            df = pd.concat([df, crit_df])

        return self._save_csv(df, 'weights_analysis.csv')

    # ==================================================================
    #  2. RANKINGS
    # ==================================================================

    def save_rankings(self, ranking_result: Any,
                      provinces: List[str]) -> str:
        scores = ranking_result.final_scores
        ranks = ranking_result.final_ranking

        # Use the active province list from the result's index so that
        # dynamically-excluded provinces are never written to output.
        active_provinces = (
            list(scores.index)
            if hasattr(scores, 'index')
            else provinces
        )

        df = pd.DataFrame({
            'Province': active_provinces,
            'ER_Score': np.asarray(scores.values if hasattr(scores, 'values') else scores),
            'ER_Rank': np.asarray(ranks.values if hasattr(ranks, 'values') else ranks, dtype=int),
        })

        n = len(df)
        df['Percentile'] = ((n - df['ER_Rank'] + 1) / n * 100).round(1)

        mean_s = df['ER_Score'].mean()
        std_s = df['ER_Score'].std()
        df['Z_Score'] = ((df['ER_Score'] - mean_s) / std_s).round(4) if std_s > 0 else 0

        df['Tier'] = pd.cut(
            df['ER_Rank'],
            bins=[0, n * 0.1, n * 0.25, n * 0.5, n * 0.75, n + 1],
            labels=['Elite (Top 10%)', 'High (10-25%)', 'Upper-Mid (25-50%)',
                    'Lower-Mid (50-75%)', 'Low (75-100%)'],
        )

        try:
            unc = ranking_result.er_result.uncertainty
            if 'belief_entropy' in unc.columns:
                df['Belief_Entropy'] = unc['belief_entropy'].values
            if 'utility_interval_width' in unc.columns:
                df['Utility_Interval_Width'] = unc['utility_interval_width'].values
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        df['Kendall_W'] = ranking_result.kendall_w

        df = df.sort_values('ER_Rank').reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = 'Position'
        return self._save_csv(df, 'final_rankings.csv', float_fmt='%.4f')

    # ==================================================================
    #  3. MCDM SCORES PER CRITERION
    # ==================================================================

    def save_mcdm_scores_by_criterion(
        self, ranking_result: Any, provinces: List[str],
    ) -> Dict[str, str]:
        saved = {}
        for crit_id, method_scores in ranking_result.criterion_method_scores.items():
            # Resolve per-criterion active province list from the first series' index
            _first = next(iter(method_scores.values()), None)
            crit_provinces = (
                list(_first.index)
                if _first is not None and hasattr(_first, 'index')
                else provinces
            )
            score_df = pd.DataFrame(index=crit_provinces)
            rank_df = pd.DataFrame(index=crit_provinces)

            for method, series in method_scores.items():
                vals = series.values if hasattr(series, 'values') else np.asarray(series)
                score_df[f'{method}_Score'] = vals
                rank_df[f'{method}_Rank'] = pd.Series(vals, index=crit_provinces).rank(
                    ascending=False, method='min').astype(int)

            score_cols = list(score_df.columns)
            score_df['Mean_Score'] = score_df[score_cols].mean(axis=1)
            score_df['StdDev_Score'] = score_df[score_cols].std(axis=1)
            score_df['CV_Score'] = np.where(
                score_df['Mean_Score'] > 0,
                score_df['StdDev_Score'] / score_df['Mean_Score'], 0)

            combined = pd.concat([score_df, rank_df], axis=1)
            combined.index.name = 'Province'

            rank_cols = list(rank_df.columns)
            combined['Mean_Rank'] = rank_df[rank_cols].mean(axis=1).round(2)
            combined['Consensus_Rank'] = combined['Mean_Rank'].rank(method='min').astype(int)

            fname = f'mcdm_scores_{crit_id}.csv'
            path = self._save_csv(combined, fname, float_fmt='%.4f')
            saved[crit_id] = str(path)
        return saved

    # ==================================================================
    #  4. RANK COMPARISON (all methods x all criteria)
    # ==================================================================

    def save_rank_comparison(self, ranking_result: Any,
                             provinces: List[str]) -> str:
        # The final ranking's province list is the authoritative active set.
        active_provinces = (
            list(ranking_result.final_ranking.index)
            if hasattr(ranking_result.final_ranking, 'index')
            else provinces
        )

        all_ranks = {}
        for crit_id, method_ranks in ranking_result.criterion_method_ranks.items():
            for method, ranks_series in method_ranks.items():
                col = f'{crit_id}_{method}'
                if hasattr(ranks_series, 'reindex'):
                    # Reindex to the global active province list so that
                    # provinces absent from this criterion's active subset
                    # are represented as NaN rather than sizing mismatches.
                    vals = ranks_series.reindex(active_provinces).values
                else:
                    vals = np.asarray(ranks_series)
                all_ranks[col] = vals

        df = pd.DataFrame(all_ranks, index=active_provinces)
        df.index.name = 'Province'

        df['Mean_Rank'] = df.mean(axis=1).round(2)
        df['Median_Rank'] = df.median(axis=1).round(2)
        df['StdDev_Rank'] = df.std(axis=1).round(2)
        summary = ['Mean_Rank', 'Median_Rank', 'StdDev_Rank']
        base = df.drop(columns=summary, errors='ignore')
        df['Best_Rank'] = base.min(axis=1)
        df['Worst_Rank'] = base.max(axis=1)
        df['Rank_Range'] = df['Worst_Rank'] - df['Best_Rank']

        return self._save_csv(df, 'mcdm_rank_comparison.csv', float_fmt='%.0f')

    # ==================================================================
    #  5. FORECAST RESULTS
    # ==================================================================

    def save_forecast_results(self, forecast_result: Any) -> Dict[str, str]:
        saved = {}

        # Predictions
        try:
            preds = forecast_result.predictions
            if preds is not None and not preds.empty:
                saved['predictions'] = self._save_csv(preds, 'forecast_predictions.csv', float_fmt='%.4f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # Prediction intervals
        try:
            intervals = forecast_result.prediction_intervals
            if intervals:
                lower = intervals.get('lower')
                upper = intervals.get('upper')
                if lower is not None and upper is not None:
                    combined = pd.DataFrame(index=lower.index)
                    for col in lower.columns:
                        combined[f'{col}_lower'] = lower[col]
                        combined[f'{col}_upper'] = upper[col]
                        if col in forecast_result.predictions.columns:
                            combined[f'{col}_point'] = forecast_result.predictions[col]
                            combined[f'{col}_width'] = upper[col] - lower[col]
                    combined.index.name = 'Entity'
                    saved['prediction_intervals'] = self._save_csv(
                        combined, 'prediction_intervals.csv', float_fmt='%.4f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # Model contributions
        try:
            contrib = forecast_result.model_contributions
            if contrib:
                df = pd.DataFrame([
                    {'Model': k, 'Weight': v, 'Rank': 0, 'Weight_Pct': f'{v*100:.1f}%'}
                    for k, v in sorted(contrib.items(), key=lambda x: x[1], reverse=True)
                ])
                df['Rank'] = range(1, len(df) + 1)
                df = df.set_index('Rank')
                saved['model_contributions'] = self._save_csv(
                    df, 'model_contributions.csv', float_fmt='%.6f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # Model performance
        try:
            perf = forecast_result.model_performance
            if perf:
                rows = [{'Model': m, **metrics} for m, metrics in perf.items()]
                df = pd.DataFrame(rows).set_index('Model')
                saved['model_performance'] = self._save_csv(
                    df, 'model_performance.csv', float_fmt='%.4f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # Feature importance
        try:
            imp = forecast_result.feature_importance
            if imp is not None and not imp.empty:
                imp_copy = imp.copy()
                if 'Importance' not in imp_copy.columns and len(imp_copy.columns) > 0:
                    imp_copy['Importance'] = imp_copy.iloc[:, 0]
                imp_copy = imp_copy.sort_values('Importance', ascending=False)
                imp_copy['Cumulative'] = imp_copy['Importance'].cumsum()
                imp_copy['Rank'] = range(1, len(imp_copy) + 1)
                saved['feature_importance'] = self._save_csv(
                    imp_copy, 'feature_importance.csv', float_fmt='%.6f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # Cross-validation scores
        try:
            cv = forecast_result.cross_validation_scores
            if cv:
                rows = {}
                for model, scores in cv.items():
                    row = {f'Fold_{i+1}': s for i, s in enumerate(scores)}
                    row.update(Mean=np.mean(scores), StdDev=np.std(scores),
                               Min=np.min(scores), Max=np.max(scores))
                    rows[model] = row
                df = pd.DataFrame(rows).T
                df.index.name = 'Model'
                saved['cross_validation'] = self._save_csv(
                    df, 'cross_validation_scores.csv', float_fmt='%.4f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # Holdout performance
        try:
            holdout = forecast_result.holdout_performance
            if holdout:
                df = pd.DataFrame([holdout])
                df.index = ['Holdout']
                df.index.name = 'Set'
                saved['holdout'] = self._save_csv(df, 'holdout_performance.csv', float_fmt='%.4f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # Forecast summary JSON
        try:
            summary = {
                'n_models': len(forecast_result.model_contributions) if forecast_result.model_contributions else 0,
                'n_predictions': len(forecast_result.predictions) if forecast_result.predictions is not None else 0,
                'model_weights': forecast_result.model_contributions,
                'holdout_performance': forecast_result.holdout_performance,
                'training_info': {
                    k: v for k, v in (forecast_result.training_info or {}).items()
                    if k not in ('y_test', 'y_pred', 'test_entities', 'X_test')
                },
                'data_summary': forecast_result.data_summary,
            }
            saved['summary'] = self._save_json(summary, 'forecast_summary.json')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        return saved

    # ==================================================================
    #  6. SENSITIVITY / ANALYSIS
    # ==================================================================

    def save_analysis_results(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        saved = {}
        sens = analysis_results.get('sensitivity')
        if sens is None:
            return saved

        # Criteria sensitivity
        try:
            if hasattr(sens, 'criteria_sensitivity') and sens.criteria_sensitivity:
                df = pd.DataFrame([
                    {'Criterion': k, 'Sensitivity': v}
                    for k, v in sorted(sens.criteria_sensitivity.items(),
                                       key=lambda x: x[1], reverse=True)
                ]).set_index('Criterion')
                df['Rank'] = range(1, len(df) + 1)
                df['Interpretation'] = df['Sensitivity'].apply(
                    lambda x: 'High' if x > 0.1 else ('Medium' if x > 0.05 else 'Low'))
                saved['criteria'] = self._save_csv(df, 'sensitivity_criteria.csv', float_fmt='%.6f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # Subcriteria sensitivity
        try:
            if hasattr(sens, 'subcriteria_sensitivity') and sens.subcriteria_sensitivity:
                df = pd.DataFrame([
                    {'Subcriteria': k, 'Sensitivity': v}
                    for k, v in sorted(sens.subcriteria_sensitivity.items(),
                                       key=lambda x: x[1], reverse=True)
                ]).set_index('Subcriteria')
                df['Rank'] = range(1, len(df) + 1)
                saved['subcriteria'] = self._save_csv(df, 'sensitivity_subcriteria.csv', float_fmt='%.6f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # Rank stability
        try:
            if hasattr(sens, 'rank_stability') and sens.rank_stability:
                df = pd.DataFrame([
                    {'Province': k, 'Stability': v}
                    for k, v in sorted(sens.rank_stability.items(),
                                       key=lambda x: x[1], reverse=True)
                ]).set_index('Province')
                df['Rank_by_Stability'] = range(1, len(df) + 1)
                df['Classification'] = df['Stability'].apply(
                    lambda x: 'Very Stable' if x > 0.9 else (
                        'Stable' if x > 0.7 else ('Moderate' if x > 0.5 else 'Volatile')))
                saved['rank_stability'] = self._save_csv(
                    df, 'sensitivity_rank_stability.csv', float_fmt='%.4f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # Top-N stability
        try:
            if hasattr(sens, 'top_n_stability') and sens.top_n_stability:
                df = pd.DataFrame([
                    {'Top_N': f'Top-{n}', 'N': n, 'Stability': v}
                    for n, v in sorted(sens.top_n_stability.items())
                ]).set_index('Top_N')
                saved['top_n'] = self._save_csv(df, 'sensitivity_top_n_stability.csv', float_fmt='%.4f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # Temporal stability
        try:
            if hasattr(sens, 'temporal_stability') and sens.temporal_stability:
                df = pd.DataFrame([
                    {'Year_Pair': k, 'Rank_Correlation': v}
                    for k, v in sorted(sens.temporal_stability.items())
                ]).set_index('Year_Pair')
                df['Interpretation'] = df['Rank_Correlation'].apply(
                    lambda x: 'Strong' if x > 0.8 else ('Moderate' if x > 0.5 else 'Weak'))
                saved['temporal'] = self._save_csv(df, 'sensitivity_temporal.csv', float_fmt='%.4f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # IFS sensitivity
        try:
            ifs_data = {
                'Parameter': ['Membership (μ)', 'Non-membership (ν)'],
                'Sensitivity': [
                    getattr(sens, 'ifs_membership_sensitivity', 0),
                    getattr(sens, 'ifs_nonmembership_sensitivity', 0),
                ],
            }
            df = pd.DataFrame(ifs_data).set_index('Parameter')
            df['Interpretation'] = df['Sensitivity'].apply(
                lambda x: 'High' if x > 0.1 else ('Medium' if x > 0.05 else 'Low'))
            saved['ifs'] = self._save_csv(df, 'sensitivity_ifs.csv', float_fmt='%.6f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # Summary JSON
        try:
            robustness = {
                'overall_robustness': getattr(sens, 'overall_robustness', 0),
                'confidence_level': getattr(sens, 'confidence_level', 0.95),
                'n_criteria_analyzed': len(getattr(sens, 'criteria_sensitivity', {})),
                'n_subcriteria_analyzed': len(getattr(sens, 'subcriteria_sensitivity', {})),
                'n_provinces_stability': len(getattr(sens, 'rank_stability', {})),
                'ifs_membership_sensitivity': getattr(sens, 'ifs_membership_sensitivity', 0),
                'ifs_nonmembership_sensitivity': getattr(sens, 'ifs_nonmembership_sensitivity', 0),
                'top_n_stability': getattr(sens, 'top_n_stability', {}),
            }
            saved['summary'] = self._save_json(robustness, 'sensitivity_summary.json')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        return saved

    # ==================================================================
    #  7. ER UNCERTAINTY
    # ==================================================================

    def save_er_uncertainty(self, ranking_result: Any,
                            provinces: List[str]) -> Optional[str]:
        try:
            unc = ranking_result.er_result.uncertainty.copy()
            unc.index.name = 'Province'
            return self._save_csv(unc, 'prediction_uncertainty_er.csv', float_fmt='%.6f')
        except Exception as _exc:
            _logger.debug('er_uncertainty skipped: %s', _exc)
            return None

    # ==================================================================
    #  8. DATA SUMMARY
    # ==================================================================

    def save_data_summary(self, panel_data: Any) -> str:
        latest_year = max(panel_data.years)
        df = panel_data.subcriteria_cross_section[latest_year]
        summary = df.describe().T
        summary['Skewness'] = df.skew()
        summary['Kurtosis'] = df.kurtosis()
        summary['CV'] = np.where(summary['mean'] != 0,
                                  summary['std'] / summary['mean'], 0)
        summary['IQR'] = summary['75%'] - summary['25%']
        summary.index.name = 'Subcriteria'
        return self._save_csv(summary, 'data_summary_statistics.csv')

    # ==================================================================
    #  9. CRITERION WEIGHTS (was bypassed in old pipeline.py)
    # ==================================================================

    def save_criterion_weights(self, criterion_weights: Dict[str, float]) -> str:
        df = pd.DataFrame([criterion_weights])
        return self._save_csv(df, 'criterion_weights.csv', index=False)

    # ==================================================================
    # 10. EXECUTION SUMMARY (was bypassed in old pipeline.py)
    # ==================================================================

    def save_execution_summary(
        self,
        panel_data: Any = None,
        ranking_result: Any = None,
        execution_time: float = 0.0,
    ) -> str:
        """Build and persist an execution summary as JSON.

        Parameters
        ----------
        panel_data:
            PanelData object returned by the data loader.
        ranking_result:
            HierarchicalRankingResult from the ranking pipeline.
        execution_time:
            Wall-clock seconds for the full pipeline run.
        """
        summary: Dict[str, Any] = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'execution_time_seconds': round(float(execution_time), 3),
        }

        # --- Panel metadata ---
        if panel_data is not None:
            try:
                summary['panel'] = {
                    'n_provinces': int(panel_data.n_provinces),
                    'n_years': int(panel_data.n_years),
                    'years': list(panel_data.years),
                    'n_criteria': int(panel_data.n_criteria),
                    'n_subcriteria': int(panel_data.n_subcriteria),
                    'provinces': list(panel_data.provinces),
                }
            except Exception as _exc:
                _logger.debug('panel metadata skipped: %s', _exc)
                summary['panel'] = None

        # --- Final rankings ---
        if ranking_result is not None:
            try:
                final_ranking = ranking_result.final_ranking
                final_scores = ranking_result.final_scores
                # Convert Series to {province: rank/score} dicts
                ranking_dict = {
                    str(k): int(v) for k, v in final_ranking.items()
                }
                scores_dict = {
                    str(k): round(float(v), 6) for k, v in final_scores.items()
                }
                # Top-5 provinces by rank (ascending = best)
                top_provinces = (
                    final_ranking.sort_values().head(5).index.tolist()
                )
                summary['ranking'] = {
                    'final_ranking': ranking_dict,
                    'final_scores': scores_dict,
                    'top_5_provinces': [str(p) for p in top_provinces],
                    'n_ranked': len(ranking_dict),
                }
            except Exception as _exc:
                _logger.debug('ranking summary skipped: %s', _exc)
                summary['ranking'] = None

        return self._save_json(summary, 'execution_summary.json')

    # ==================================================================
    # 11. CONFIG SNAPSHOT (was bypassed in old pipeline.py)
    # ==================================================================

    def save_config_snapshot(self, config: Any) -> Optional[str]:
        try:
            data = config.to_dict() if hasattr(config, 'to_dict') else {}
            return self._save_json(data, 'config_snapshot.json')
        except Exception as _exc:
            _logger.debug('config_snapshot skipped: %s', _exc)
            return None


__all__ = ['CsvWriter']
