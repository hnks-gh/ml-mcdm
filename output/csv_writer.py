# -*- coding: utf-8 -*-
"""
CSV & JSON Data Writer for ML-MCDM Pipeline
============================================

All structured numerical output (weights, rankings, scores, forecasts,
sensitivity analysis) is persisted through this single writer class.
Files are organised under ``output/result/csv/<phase>/``:

  - weighting/   — weights_analysis.csv,
                   critic_weights_YYYY.csv (one per year),
                   critic_sc_weights_all_years.csv  (SC x Year global weights),
                   sc_local_weights_all_years.csv   (SC x Year local weights),
                   critic_criterion_weights_all_years.csv (Criterion x Year)
  - mcdm/        — final_rankings.csv, prediction_uncertainty_er.csv,
                   mcdm_scores_*.csv, mcdm_rank_comparison.csv
  - ranking/     — mcdm_scores_YYYY.csv (per-year per-method per-criterion)
  - forecasting/ — forecast_*, model_*, feature_importance, cv scores
  - sensitivity/ — sensitivity_*.csv, sensitivity_summary.json
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

_logger = logging.getLogger(__name__)


class CsvWriter:
    """Write CSV / JSON result files into ``<base_dir>/csv/<phase>/``."""

    # Canonical phase names
    PHASES = ('weighting', 'mcdm', 'ranking', 'forecasting', 'sensitivity')

    def __init__(self, base_output_dir: str = 'output/result'):
        from . import _sanitize_output_dir
        base = _sanitize_output_dir(base_output_dir)
        self.csv_dir = base / 'csv'
        # Create one subdirectory per phase
        for phase in self.PHASES:
            (self.csv_dir / phase).mkdir(parents=True, exist_ok=True)
            setattr(self, f'{phase}_dir', self.csv_dir / phase)
        # Backward-compat alias
        self.results_dir = self.csv_dir
        self._saved_files: List[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record(self, path: Path) -> str:
        s = str(path)
        self._saved_files.append(s)
        return s

    def _save_csv(self, df: pd.DataFrame, name: str,
                  directory: Optional[Path] = None,
                  float_fmt: str = '%.6g', **kwargs) -> str:
        d = directory if directory is not None else self.csv_dir
        path = d / name
        df.to_csv(path, float_format=float_fmt, **kwargs)
        return self._record(path)

    def _save_json(self, obj: Any, name: str,
                   directory: Optional[Path] = None) -> str:
        d = directory if directory is not None else self.csv_dir
        path = d / name
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
        """Save CRITIC weighting results to weights_analysis.csv."""
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

        return self._save_csv(df, 'weights_analysis.csv',
                                directory=self.weighting_dir)

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
        return self._save_csv(df, 'final_rankings.csv', float_fmt='%.4f',
                                directory=self.mcdm_dir)

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
            path = self._save_csv(combined, fname, float_fmt='%.4f',
                                   directory=self.mcdm_dir)
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

        return self._save_csv(df, 'mcdm_rank_comparison.csv', float_fmt='%.0f',
                                directory=self.mcdm_dir)

    # ==================================================================
    #  5. FORECAST RESULTS
    # ==================================================================

    def save_forecast_results(self, forecast_result: Any) -> Dict[str, str]:
        saved = {}
        _dir = self.forecasting_dir

        # Predictions
        try:
            preds = forecast_result.predictions
            if preds is not None and not preds.empty:
                saved['predictions'] = self._save_csv(
                    preds, 'forecast_predictions.csv', directory=_dir, float_fmt='%.4f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # Prediction intervals
        try:
            intervals = forecast_result.prediction_intervals
            if intervals:
                lower = intervals.get('lower')
                upper = intervals.get('upper')
                if lower is not None and upper is not None:
                    cols_dict = {}
                    for col in lower.columns:
                        cols_dict[f'{col}_lower'] = lower[col]
                        cols_dict[f'{col}_upper'] = upper[col]
                        if col in forecast_result.predictions.columns:
                            cols_dict[f'{col}_point'] = forecast_result.predictions[col]
                            cols_dict[f'{col}_width'] = upper[col] - lower[col]
                    combined = pd.concat(cols_dict, axis=1)
                    combined.index.name = 'Entity'
                    saved['prediction_intervals'] = self._save_csv(
                        combined, 'prediction_intervals.csv',
                        directory=_dir, float_fmt='%.4f')
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
                    df, 'model_contributions.csv', directory=_dir, float_fmt='%.6f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # Model performance
        try:
            perf = forecast_result.model_performance
            if perf:
                rows = [{'Model': m, **metrics} for m, metrics in perf.items()]
                df = pd.DataFrame(rows).set_index('Model')
                saved['model_performance'] = self._save_csv(
                    df, 'model_performance.csv', directory=_dir, float_fmt='%.4f')
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
                    imp_copy, 'feature_importance.csv', directory=_dir, float_fmt='%.6f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # Cross-validation scores
        try:
            cv = forecast_result.cross_validation_scores
            if cv:
                rows = {}
                for model, scores in cv.items():
                    row = {f'Fold_{i+1}': s for i, s in enumerate(scores)}
                    row.update(Mean=np.nanmean(scores), StdDev=np.nanstd(scores),
                               Min=np.nanmin(scores), Max=np.nanmax(scores))
                    rows[model] = row
                df = pd.DataFrame(rows).T
                df.index.name = 'Model'
                saved['cross_validation'] = self._save_csv(
                    df, 'cross_validation_scores.csv', directory=_dir, float_fmt='%.4f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # Holdout performance
        try:
            holdout = forecast_result.holdout_performance
            if holdout:
                df = pd.DataFrame([holdout])
                df.index = ['Holdout']
                df.index.name = 'Set'
                saved['holdout'] = self._save_csv(
                    df, 'holdout_performance.csv', directory=_dir, float_fmt='%.4f')
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
            saved['summary'] = self._save_json(summary, 'forecast_summary.json',
                                               directory=_dir)
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        return saved

    # ==================================================================
    #  6. SENSITIVITY / ANALYSIS
    # ==================================================================

    def save_analysis_results(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        saved = {}
        _dir = self.sensitivity_dir
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
                saved['criteria'] = self._save_csv(
                    df, 'sensitivity_criteria.csv', directory=_dir, float_fmt='%.6f')
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
                saved['subcriteria'] = self._save_csv(
                    df, 'sensitivity_subcriteria.csv', directory=_dir, float_fmt='%.6f')
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
                    df, 'sensitivity_rank_stability.csv', directory=_dir, float_fmt='%.4f')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # Top-N stability
        try:
            if hasattr(sens, 'top_n_stability') and sens.top_n_stability:
                df = pd.DataFrame([
                    {'Top_N': f'Top-{n}', 'N': n, 'Stability': v}
                    for n, v in sorted(sens.top_n_stability.items())
                ]).set_index('Top_N')
                saved['top_n'] = self._save_csv(
                    df, 'sensitivity_top_n_stability.csv', directory=_dir, float_fmt='%.4f')
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
                saved['temporal'] = self._save_csv(
                    df, 'sensitivity_temporal.csv', directory=_dir, float_fmt='%.4f')
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
                'top_n_stability': getattr(sens, 'top_n_stability', {}),
            }
            saved['summary'] = self._save_json(robustness, 'sensitivity_summary.json',
                                               directory=_dir)
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
            return self._save_csv(unc, 'prediction_uncertainty_er.csv',
                                    directory=self.mcdm_dir, float_fmt='%.6f')
        except Exception as _exc:
            _logger.debug('er_uncertainty skipped: %s', _exc)
            return None

    # ==================================================================
    #  8. CRITERION WEIGHTS  [REMOVED — redundant with per-year all-years
    #     matrices; panel-level snapshot caused value discrepancy vs the
    #     correct per-year critic_weights_YYYY.csv files]
    # ==================================================================

    # ==================================================================
    #  9. EXECUTION SUMMARY  [REMOVED — summary/ phase removed]
    # ==================================================================

    # ==================================================================
    # 10. CONFIG SNAPSHOT  [REMOVED — summary/ phase removed]
    # ==================================================================

    # ==================================================================
    # 11. METHOD WEIGHTS (panel-level critic_weights.csv)  [REMOVED —
    #     panel-level snapshot redundant and inconsistent with per-year
    #     critic_weights_YYYY.csv files]
    # ==================================================================

    # ==================================================================
    # 13. MC PROVINCE STATS — Monte-Carlo rank uncertainty per province
    # ==================================================================

    def save_mc_province_stats(self, weights: Dict[str, Any]) -> Optional[str]:
        """
        Write MC rank-uncertainty statistics for each province derived from
        the Level-2 Monte-Carlo ensemble.

        File produced
        -------------
        weighting/mc_province_rankings.csv
        """
        try:
            mps = weights.get('mc_province_stats', {})
            if not mps:
                return None

            mean_r   = mps.get('province_mean_rank', {})
            std_r    = mps.get('province_std_rank', {})
            prob_t1  = mps.get('province_prob_top1', {})
            prob_tk  = mps.get('province_prob_topK', {})

            all_provinces = sorted(set(mean_r) | set(std_r))
            rows = []
            for prov in all_provinces:
                mr = float(mean_r.get(prov, 0.0))
                sr = float(std_r.get(prov, 0.0))
                rows.append({
                    'Province':        prov,
                    'MC_Mean_Rank':    mr,
                    'MC_Std_Rank':     sr,
                    'MC_CV_Rank':      sr / mr if mr > 0 else 0.0,
                    'MC_Prob_Top1':    float(prob_t1.get(prov, 0.0)),
                    'MC_Prob_TopK':    float(prob_tk.get(prov, 0.0)),
                })

            df = pd.DataFrame(rows).set_index('Province')
            df = df.sort_values('MC_Mean_Rank')
            df['MC_Nominal_Rank'] = range(1, len(df) + 1)

            # Win-rate matrix (optional)
            win_mat = mps.get('rank_win_matrix')
            if win_mat is not None and hasattr(win_mat, 'to_csv'):
                win_path = Path(self.weighting_dir) / 'mc_rank_win_matrix.csv'
                win_mat.to_csv(win_path, float_format='%.4f')
                self._record(str(win_path))

            return self._save_csv(
                df, 'mc_province_rankings.csv',
                directory=self.weighting_dir, float_fmt='%.4f')
        except Exception as _exc:
            _logger.debug('mc_province_stats skipped: %s', _exc)
            return None

    # ==================================================================
    # 14. ALL-YEARS RANKINGS
    # ==================================================================

    def save_rankings_all_years(
        self,
        multi_year_results: Dict[int, Any],
        provinces: List[str],
    ) -> Dict[str, str]:
        """
        Write Province × Year matrices for ER scores and integer ranks,
        plus a long-format criterion-level ER utility table.

        Files produced
        --------------
        ranking/rankings_all_years.csv    — score matrix
        ranking/ranks_all_years.csv       — rank matrix
        ranking/criterion_er_scores_all_years.csv — long format
        """
        saved: Dict[str, str] = {}
        if not multi_year_results:
            return saved

        years = sorted(multi_year_results.keys())

        # ── Score and rank Province × Year matrices ─────────────────────
        try:
            score_data: Dict[int, pd.Series] = {}
            rank_data:  Dict[int, pd.Series] = {}
            for yr in years:
                yr_res = multi_year_results[yr]
                fs = yr_res.final_scores if hasattr(yr_res, 'final_scores') else None
                fr = yr_res.final_ranking if hasattr(yr_res, 'final_ranking') else None
                if fs is not None:
                    score_data[yr] = fs
                if fr is not None:
                    rank_data[yr] = fr

            if score_data:
                score_df = pd.DataFrame(score_data)
                score_df.index.name = 'Province'
                score_df.columns.name = 'Year'
                score_df['Mean_Score']     = score_df.mean(axis=1)
                score_df['Trend']          = score_df[years].apply(
                    lambda row: float(np.polyfit(range(len(years)), row.values, 1)[0]),
                    axis=1)
                saved['scores'] = self._save_csv(
                    score_df, 'rankings_all_years.csv',
                    directory=self.mcdm_dir, float_fmt='%.4f')

            if rank_data:
                rank_df = pd.DataFrame(rank_data)
                rank_df.index.name = 'Province'
                rank_df.columns.name = 'Year'
                rank_df['Mean_Rank'] = rank_df[years].mean(axis=1).round(2)
                rank_df['Rank_Volatility'] = rank_df[years].std(axis=1).round(2)
                best_yr = rank_df[years].idxmin(axis=1)
                rank_df['Best_Year'] = best_yr
                saved['ranks'] = self._save_csv(
                    rank_df, 'ranks_all_years.csv',
                    directory=self.mcdm_dir, float_fmt='%.2f')
        except Exception as _exc:
            _logger.debug('all-years score/rank matrix skipped: %s', _exc)

        # ── Long-format criterion-level ER utilities ─────────────────────
        try:
            long_rows = []
            for yr in years:
                yr_res = multi_year_results[yr]
                er_res = getattr(yr_res, 'er_result', None)
                if er_res is None:
                    continue
                crit_beliefs = getattr(er_res, 'criterion_beliefs', {})
                for prov, crit_dict in crit_beliefs.items():
                    for crit_id, bd in crit_dict.items():
                        long_rows.append({
                            'Year':      yr,
                            'Province':  prov,
                            'Criterion': crit_id,
                            'ER_Utility': float(bd.average_utility()),
                            'Belief_Entropy': float(bd.belief_entropy()),
                        })
            if long_rows:
                long_df = pd.DataFrame(long_rows)
                saved['criterion_er'] = self._save_csv(
                    long_df, 'criterion_er_scores_all_years.csv',
                    directory=self.mcdm_dir, float_fmt='%.4f')
        except Exception as _exc:
            _logger.debug('criterion ER long-format skipped: %s', _exc)

        return saved

    # ==================================================================
    # 15. BELIEF DISTRIBUTIONS
    # ==================================================================

    def save_belief_distributions(
        self, ranking_result: Any, provinces: List[str]
    ) -> Optional[str]:
        """
        Write per-province, per-criterion belief distributions from Stage 1
        of the two-stage ER.  Rows are (Province, Criterion) pairs; columns
        are the grade belief degrees plus unassigned probability mass and
        Shannon entropy of the belief function.

        File produced
        -------------
        ranking/belief_distributions.csv
        """
        try:
            er_res = getattr(ranking_result, 'er_result', None)
            if er_res is None:
                return None
            crit_beliefs = getattr(er_res, 'criterion_beliefs', {})
            if not crit_beliefs:
                return None

            # Infer grade labels from first entry
            first_prov  = next(iter(crit_beliefs))
            first_crit  = next(iter(crit_beliefs[first_prov]))
            first_bd    = crit_beliefs[first_prov][first_crit]
            n_grades    = len(first_bd.degrees)
            grade_cols  = [f'Grade_{g+1}' for g in range(n_grades)]

            rows = []
            for prov in provinces:
                if prov not in crit_beliefs:
                    continue
                for crit_id, bd in sorted(crit_beliefs[prov].items()):
                    row: Dict[str, Any] = {'Province': prov, 'Criterion': crit_id}
                    for g_idx, g_col in enumerate(grade_cols):
                        row[g_col] = float(bd.degrees[g_idx]) if g_idx < len(bd.degrees) else 0.0
                    row['Unassigned']     = float(getattr(bd, 'unassigned', 0.0))
                    row['Belief_Entropy'] = float(bd.belief_entropy())
                    row['Avg_Utility']    = float(bd.average_utility())
                    rows.append(row)

            if not rows:
                return None

            df = pd.DataFrame(rows)
            df = df.sort_values(['Province', 'Criterion']).reset_index(drop=True)
            df.index.name = 'RowID'
            return self._save_csv(
                df, 'belief_distributions.csv',
                directory=self.mcdm_dir, float_fmt='%.6f')
        except Exception as _exc:
            _logger.debug('belief distributions skipped: %s', _exc)
            return None

    # ==================================================================
    # 16. MCDM COMPOSITE COMPARISON (all methods + ER, one row per province)
    # ==================================================================

    def save_mcdm_composite_comparison(
        self, ranking_result: Any, provinces: List[str]
    ) -> Optional[str]:
        """
        Write a single wide table with composite (averaged across criteria)
        scores and ranks for every MCDM method plus the final ER score, and
        Spearman rank correlations vs. ER.

        File produced
        -------------
        mcdm/mcdm_composite_scores.csv
        """
        try:
            active_provinces = (
                list(ranking_result.final_ranking.index)
                if hasattr(ranking_result.final_ranking, 'index')
                else provinces
            )

            # Aggregate per-criterion method scores → composite per method
            method_composites: Dict[str, pd.Series] = {}
            for crit_id, method_scores in ranking_result.criterion_method_scores.items():
                for method, series in method_scores.items():
                    s = series.reindex(active_provinces) if hasattr(series, 'reindex') else pd.Series(
                        np.asarray(series), index=active_provinces)
                    if method not in method_composites:
                        method_composites[method] = s
                    else:
                        method_composites[method] = method_composites[method].add(s, fill_value=0.0)

            n_criteria = max(
                len(ranking_result.criterion_method_scores), 1)
            for method in list(method_composites.keys()):
                method_composites[method] = method_composites[method] / n_criteria

            df = pd.DataFrame(method_composites, index=active_provinces)
            df.index.name = 'Province'

            # Rank columns
            score_cols = list(df.columns)
            for col in score_cols:
                df[f'{col}_Rank'] = df[col].rank(ascending=False, method='min').astype(int)

            # ER final score + rank
            er_scores = ranking_result.final_scores.reindex(active_provinces)
            df['ER_Score'] = er_scores.values
            df['ER_Rank']  = ranking_result.final_ranking.reindex(active_provinces).values

            # Summary stats across method ranks
            rank_cols = [f'{c}_Rank' for c in score_cols]
            df['Mean_Method_Score'] = df[score_cols].mean(axis=1)
            df['Rank_Range']        = df[rank_cols].max(axis=1) - df[rank_cols].min(axis=1)

            # Spearman correlation of each method vs. ER rank
            er_rank = df['ER_Rank']
            from scipy.stats import spearmanr
            spearman_row: Dict[str, float] = {}
            for col in score_cols:
                r_col = f'{col}_Rank'
                rho, _ = spearmanr(df[r_col], er_rank)
                spearman_row[col] = round(float(rho), 4)
            df['Spearman_vs_ER'] = df[score_cols].apply(
                lambda row: float(np.mean([spearman_row.get(c, 0) for c in score_cols])), axis=1)

            df = df.sort_values('ER_Rank').reset_index()
            df.index = df.index + 1
            df.index.name = 'Position'

            return self._save_csv(
                df, 'mcdm_composite_scores.csv',
                directory=self.mcdm_dir, float_fmt='%.4f')
        except Exception as _exc:
            _logger.debug('mcdm composite comparison skipped: %s', _exc)
            return None

    # ==================================================================
    # 17. INDIVIDUAL MODEL PREDICTIONS
    # ==================================================================

    def save_individual_model_predictions(
        self, forecast_result: Any
    ) -> Optional[str]:
        """
        Write a wide table with each base model's out-of-sample predictions
        alongside the Super Learner ensemble point forecast and the conformal
        prediction interval bounds.

        File produced
        -------------
        forecasting/individual_model_predictions.csv
        """
        try:
            tinfo = forecast_result.training_info or {}

            # Base model predictions stored under training_info['model_predictions']
            model_preds = tinfo.get('model_predictions', {})
            if not model_preds and not hasattr(forecast_result, 'predictions'):
                return None

            ensemble_preds = forecast_result.predictions
            if ensemble_preds is None or ensemble_preds.empty:
                return None

            index = ensemble_preds.index

            df = pd.DataFrame(index=index)
            df.index.name = 'Entity'

            # Individual base model columns
            for model_name, pred_arr in model_preds.items():
                try:
                    arr = np.asarray(pred_arr).flatten()
                    if len(arr) == len(index):
                        df[f'{model_name}_pred'] = arr
                except Exception as _exc:
                    _logger.debug('model column %s skipped: %s', model_name, _exc)

            # Ensemble / Super Learner column
            if not ensemble_preds.empty and ensemble_preds.shape[1] >= 1:
                df['SuperLearner_pred'] = ensemble_preds.iloc[:, 0].values

            # Conformal prediction intervals
            try:
                intervals = forecast_result.prediction_intervals
                if intervals:
                    lower = intervals.get('lower')
                    upper = intervals.get('upper')
                    if lower is not None and not lower.empty:
                        df['CI_Lower'] = lower.iloc[:, 0].values
                    if upper is not None and not upper.empty:
                        df['CI_Upper'] = upper.iloc[:, 0].values
                    if 'CI_Lower' in df.columns and 'CI_Upper' in df.columns:
                        df['CI_Width'] = df['CI_Upper'] - df['CI_Lower']
            except Exception as _exc:
                _logger.debug('conformal interval columns skipped: %s', _exc)

            # Model weights for reference
            weights = forecast_result.model_contributions or {}
            weight_row = {k: round(v, 6) for k, v in weights.items()}
            df.attrs['model_weights'] = str(weight_row)

            return self._save_csv(
                df, 'individual_model_predictions.csv',
                directory=self.forecasting_dir, float_fmt='%.4f')
        except Exception as _exc:
            _logger.debug('individual model predictions skipped: %s', _exc)
            return None

    # ==================================================================
    # 18. PERTURBATION DETAIL — sensitivity × perturbation level matrix
    # ==================================================================

    def save_perturbation_detail(
        self, analysis_results: Dict[str, Any]
    ) -> Optional[str]:
        """
        Persist Monte-Carlo perturbation analysis results.

        Handles **two** data shapes produced by ``SensitivityAnalysis``:

        1. *Summary dict* (the default):  keys ``mean_rank``,
           ``std_rank``, ``rank_range`` — each a 1-D array indexed by
           alternative.  Written as a tidy table with one row per
           alternative.
        2. *Criterion × perturbation-level* matrix (legacy /
           custom):  ``dict[criterion, dict[level, corr]]`` or
           ``pd.DataFrame``.

        File produced
        -------------
        sensitivity/perturbation_detail.csv
        """
        try:
            sens = analysis_results.get('sensitivity')
            if sens is None:
                return None

            # Try common attribute names
            pert = (
                getattr(sens, 'perturbation_analysis', None)
                or getattr(sens, 'weight_perturbation', None)
                or getattr(sens, 'perturbation_detail', None)
            )
            if pert is None:
                return None

            # ── Shape 1: summary dict from SensitivityAnalysis ──────────
            if (
                isinstance(pert, dict)
                and 'mean_rank' in pert
                and 'std_rank' in pert
            ):
                import numpy as _np
                mean_rk = _np.asarray(pert['mean_rank'])
                std_rk  = _np.asarray(pert['std_rank'])
                rk_range = _np.asarray(pert.get('rank_range', std_rk * 0))
                n = len(mean_rk)

                df = pd.DataFrame({
                    'Mean_Rank':  mean_rk,
                    'Std_Rank':   std_rk,
                    'Rank_Range': rk_range,
                })
                df.index.name = 'Alternative'
                df['Stability_Score'] = 1.0 - (
                    std_rk / (std_rk.max() + 1e-12)
                )
                return self._save_csv(
                    df, 'perturbation_detail.csv',
                    directory=self.sensitivity_dir, float_fmt='%.4f')

            # ── Shape 2: DataFrame or nested criterion→level dict ───────
            if isinstance(pert, pd.DataFrame):
                df = pert.copy()
            elif isinstance(pert, dict):
                df = pd.DataFrame(pert).T
            else:
                return None

            df.index.name = 'Criterion'

            # Numeric columns only for summary stats
            num_cols = df.select_dtypes(include='number').columns
            if len(num_cols):
                df['Mean_Correlation']  = df[num_cols].mean(axis=1)
                df['Min_Correlation']   = df[num_cols].min(axis=1)
                df['Sensitivity_Score'] = 1 - df['Mean_Correlation']
                df['Sensitivity_Rank']  = df['Sensitivity_Score'].rank(
                    ascending=False, method='min').astype(int)

            return self._save_csv(
                df, 'perturbation_detail.csv',
                directory=self.sensitivity_dir, float_fmt='%.4f')
        except Exception as _exc:
            _logger.debug('perturbation detail skipped: %s', _exc)
            return None

    # ==================================================================
    # 19. PER-YEAR CRITIC WEIGHTS
    # ==================================================================

    def save_weights_all_years(
        self, weight_all_years: Dict[int, Any]
    ) -> Dict[str, str]:
        """
        Persist per-year CRITIC weight tables computed from each year's
        cross-section independently.

        Files produced
        --------------
        weighting/critic_weights_YYYY.csv          — one per year
        weighting/critic_sc_weights_all_years.csv  — SC × Year matrix (global weights)
        weighting/critic_criterion_weights_all_years.csv — Criterion × Year matrix
        """
        saved: Dict[str, str] = {}
        if not weight_all_years:
            return saved

        years = sorted(weight_all_years.keys())

        # ── Per-year individual files ────────────────────────────────────
        for yr in years:
            try:
                yw = weight_all_years[yr]
                sc_list     = yw.get('subcriteria', [])
                global_sc   = yw.get('global_sc_weights', {})
                crit_w      = yw.get('criterion_weights', {})
                details     = yw.get('details', {})
                level1      = details.get('level1', {})
                groups      = yw.get('criteria_groups', {})

                # Build SC → criterion lookup
                sc_to_crit: Dict[str, str] = {}
                for cid, sc_cols in groups.items():
                    for sc in sc_cols:
                        sc_to_crit[sc] = cid

                rows = []
                for sc in sc_list:
                    cid   = sc_to_crit.get(sc, '')
                    l1c   = level1.get(cid, {}).get('local_sc_weights', {})
                    rows.append({
                        'Subcriteria':          sc,
                        'Criterion':            cid,
                        'Critic_Local_Weight':  float(l1c.get(sc, 0.0)),
                        'Critic_Crit_Weight':   float(crit_w.get(cid, 0.0)),
                        'Critic_Global_Weight': float(global_sc.get(sc, 0.0)),
                    })

                if not rows:
                    continue

                df = pd.DataFrame(rows).set_index('Subcriteria')
                df['Rank'] = df['Critic_Global_Weight'].rank(
                    ascending=False, method='min').astype(int)

                fname = f'critic_weights_{yr}.csv'
                saved[f'year_{yr}'] = self._save_csv(
                    df, fname, directory=self.weighting_dir, float_fmt='%.6f')
            except Exception as _exc:
                _logger.debug('per-year weights year %d skipped: %s', yr, _exc)

        # ── SC × Year global-weight matrix ───────────────────────────────
        try:
            sc_year_data: Dict[int, Dict[str, float]] = {}
            all_scs: List[str] = []
            for yr in years:
                gw = weight_all_years[yr].get('global_sc_weights', {})
                sc_year_data[yr] = gw
                for sc in gw:
                    if sc not in all_scs:
                        all_scs.append(sc)

            sc_df = pd.DataFrame(
                {yr: [sc_year_data[yr].get(sc, float('nan')) for sc in all_scs]
                 for yr in years},
                index=all_scs,
            )
            sc_df.index.name  = 'Subcriteria'
            sc_df.columns.name = 'Year'
            sc_df['Mean_Weight'] = sc_df[years].mean(axis=1)
            sc_df['StdDev']      = sc_df[years].std(axis=1)
            saved['sc_matrix'] = self._save_csv(
                sc_df, 'critic_sc_weights_all_years.csv',
                directory=self.weighting_dir, float_fmt='%.6f')
        except Exception as _exc:
            _logger.debug('SC weight matrix skipped: %s', _exc)

        # ── Criterion × Year weight matrix ───────────────────────────────
        try:
            crit_year_data: Dict[int, Dict[str, float]] = {}
            all_crits: List[str] = []
            for yr in years:
                cw = weight_all_years[yr].get('criterion_weights', {})
                crit_year_data[yr] = cw
                for c in sorted(cw):
                    if c not in all_crits:
                        all_crits.append(c)
            all_crits.sort()

            crit_df = pd.DataFrame(
                {yr: [crit_year_data[yr].get(c, float('nan')) for c in all_crits]
                 for yr in years},
                index=all_crits,
            )
            crit_df.index.name  = 'Criterion'
            crit_df.columns.name = 'Year'
            crit_df['Mean_Weight'] = crit_df[years].mean(axis=1)
            crit_df['StdDev']      = crit_df[years].std(axis=1)
            saved['crit_matrix'] = self._save_csv(
                crit_df, 'critic_criterion_weights_all_years.csv',
                directory=self.weighting_dir, float_fmt='%.6f')
        except Exception as _exc:
            _logger.debug('criterion weight matrix skipped: %s', _exc)

        # ── SC × Year LOCAL-weight matrix ─────────────────────────────────
        # Complements critic_sc_weights_all_years.csv (global SC weights).
        # Each cell is the CRITIC within-group local weight for that SC in
        # that year (sums to 1 within each criterion group, not globally).
        try:
            sc_local_year_data: Dict[int, Dict[str, float]] = {}
            all_scs_local: List[str] = []
            for yr in years:
                yw = weight_all_years[yr]
                details_yr = yw.get('details', {})
                level1_yr  = details_yr.get('level1', {})
                groups_yr  = yw.get('criteria_groups', {})
                sc_local: Dict[str, float] = {}
                for cid, sc_cols in groups_yr.items():
                    l1c = level1_yr.get(cid, {}).get('local_sc_weights', {})
                    for sc in sc_cols:
                        sc_local[sc] = float(l1c.get(sc, float('nan')))
                sc_local_year_data[yr] = sc_local
                for sc in sc_local:
                    if sc not in all_scs_local:
                        all_scs_local.append(sc)

            local_df = pd.DataFrame(
                {yr: [sc_local_year_data[yr].get(sc, float('nan'))
                      for sc in all_scs_local]
                 for yr in years},
                index=all_scs_local,
            )
            local_df.index.name  = 'Subcriteria'
            local_df.columns.name = 'Year'
            local_df['Mean_Weight'] = local_df[years].mean(axis=1)
            local_df['StdDev']      = local_df[years].std(axis=1)
            saved['sc_local_matrix'] = self._save_csv(
                local_df, 'sc_local_weights_all_years.csv',
                directory=self.weighting_dir, float_fmt='%.6f')
        except Exception as _exc:
            _logger.debug('SC local weight matrix skipped: %s', _exc)

        return saved

    # ==================================================================
    # 20. PER-YEAR PER-METHOD MCDM SCORES (14 CSVs)
    # ==================================================================

    def save_method_scores_all_years(
        self, multi_year_results: Dict[int, Any]
    ) -> Dict[str, str]:
        """
        Persist per-year MCDM method scores and final ER rankings for all
        provinces, one CSV per year.

        Each file is long-format: one row per (Province, Criterion) pair
        with columns for each of the 6 MCDM method scores plus the
        Stage-1 ER utility and the final ER score/rank for that province.

        Files produced
        --------------
        ranking/mcdm_scores_YYYY.csv   — one file per year (14 total)
        """
        saved: Dict[str, str] = {}
        if not multi_year_results:
            return saved

        for yr, yr_res in sorted(multi_year_results.items()):
            try:
                method_scores = getattr(yr_res, 'criterion_method_scores', {})
                if not method_scores:
                    continue

                final_scores  = getattr(yr_res, 'final_scores', None)
                final_ranking = getattr(yr_res, 'final_ranking', None)

                # Discover all methods present this year
                methods_seen: List[str] = []
                for crit_scores in method_scores.values():
                    for m in crit_scores:
                        if m not in methods_seen:
                            methods_seen.append(m)

                # Stage-1 ER utilities (per province per criterion)
                er_res        = getattr(yr_res, 'er_result', None)
                crit_beliefs  = getattr(er_res, 'criterion_beliefs', {}) if er_res else {}

                rows = []
                for crit_id in sorted(method_scores):
                    crit_data = method_scores[crit_id]
                    # Identify provinces for this criterion from first method series
                    first_series = next(iter(crit_data.values()), None)
                    if first_series is None:
                        continue
                    provinces_crit = (
                        list(first_series.index)
                        if hasattr(first_series, 'index')
                        else []
                    )

                    for prov in provinces_crit:
                        row: Dict[str, Any] = {
                            'Year':      yr,
                            'Province':  prov,
                            'Criterion': crit_id,
                        }

                        # Per-method score
                        for method in methods_seen:
                            series = crit_data.get(method)
                            if series is not None:
                                val = (
                                    float(series.loc[prov])
                                    if hasattr(series, 'loc') and prov in series.index
                                    else float('nan')
                                )
                            else:
                                val = float('nan')
                            row[f'{method}_Score'] = val

                        # Stage-1 ER utility for this (province, criterion)
                        bd = crit_beliefs.get(prov, {}).get(crit_id)
                        row['ER_Stage1_Utility'] = (
                            float(bd.average_utility()) if bd is not None else float('nan'))

                        # Final ER score and rank (province-level, repeated per crit row)
                        if final_scores is not None and hasattr(final_scores, 'get'):
                            row['Final_ER_Score'] = float(
                                final_scores.get(prov, float('nan')))
                        elif final_scores is not None and hasattr(final_scores, 'loc'):
                            row['Final_ER_Score'] = (
                                float(final_scores.loc[prov])
                                if prov in final_scores.index
                                else float('nan'))
                        else:
                            row['Final_ER_Score'] = float('nan')

                        if final_ranking is not None and hasattr(final_ranking, 'loc'):
                            row['ER_Rank'] = (
                                int(final_ranking.loc[prov])
                                if prov in final_ranking.index
                                else -1)
                        else:
                            row['ER_Rank'] = -1

                        rows.append(row)

                if not rows:
                    continue

                df = pd.DataFrame(rows)
                df = df.sort_values(['ER_Rank', 'Province', 'Criterion']).reset_index(drop=True)
                df.index.name = 'RowID'

                fname = f'mcdm_scores_{yr}.csv'
                saved[f'year_{yr}'] = self._save_csv(
                    df, fname, directory=self.ranking_dir, float_fmt='%.4f')

            except Exception as _exc:
                _logger.debug('per-year method scores year %d skipped: %s', yr, _exc)

        return saved


__all__ = ['CsvWriter']
