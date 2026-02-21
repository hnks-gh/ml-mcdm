# -*- coding: utf-8 -*-
"""
Production-Ready Output Manager for ML-MCDM Pipeline
=====================================================

Comprehensive persistence layer that exports:
  - Complete CSV result sets (weights, rankings, scores, forecasts, sensitivity)
  - Publication-quality LaTeX-ready report (Markdown + tables)
  - Structured JSON metadata
  - Summary statistics for every pipeline phase

All numeric outputs use consistent formatting (6 decimal places for weights,
4 for scores/ranks) and include headers, indices, and metadata rows.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import warnings
import textwrap

warnings.filterwarnings('ignore')


class OutputManager:
    """
    Centralised output persistence for the ML-MCDM pipeline.

    Responsible for writing every CSV, JSON, and report file.
    All public methods return the path(s) of files written.
    """

    def __init__(self, base_output_dir: str = 'outputs'):
        self.base_dir = Path(base_output_dir)
        self.results_dir = self.base_dir / 'results'
        self.reports_dir = self.base_dir / 'reports'
        self.logs_dir = self.base_dir / 'logs'

        for d in (self.results_dir, self.reports_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

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

    def save_weights(self, weights_dict: Dict[str, np.ndarray],
                     subcriteria: List[str]) -> str:
        """
        Save comprehensive weight analysis CSV.

        Columns: Subcriteria, Entropy, CRITIC, MEREC, StdDev, Fused,
                 Rank_Entropy, Rank_CRITIC, Rank_MEREC, Rank_StdDev, Rank_Fused,
                 CV_across_methods, Mean, Median, Min, Max, Range
        """
        df = pd.DataFrame({'Subcriteria': subcriteria})

        method_names = ['entropy', 'critic', 'merec', 'std_dev', 'fused']
        display_names = ['Entropy', 'CRITIC', 'MEREC', 'StdDev', 'Fused']

        for mname, dname in zip(method_names, display_names):
            w = np.asarray(weights_dict[mname])
            df[dname] = w

        # Ranks per method (1 = highest weight)
        for mname, dname in zip(method_names, display_names):
            w = np.asarray(weights_dict[mname])
            df[f'Rank_{dname}'] = pd.Series(w).rank(ascending=False, method='min').astype(int).values

        # Cross-method statistics
        weight_matrix = np.column_stack([
            np.asarray(weights_dict[m]) for m in method_names[:4]
        ])  # exclude fused from cross-method stats
        df['Mean_4Methods'] = weight_matrix.mean(axis=1)
        df['Median_4Methods'] = np.median(weight_matrix, axis=1)
        df['StdDev_4Methods'] = weight_matrix.std(axis=1)
        df['CV_4Methods'] = np.where(
            df['Mean_4Methods'] > 0,
            df['StdDev_4Methods'] / df['Mean_4Methods'],
            0
        )
        df['Min_4Methods'] = weight_matrix.min(axis=1)
        df['Max_4Methods'] = weight_matrix.max(axis=1)
        df['Range_4Methods'] = df['Max_4Methods'] - df['Min_4Methods']

        # Agreement: how close is fused to mean of individual methods?
        df['Fused_vs_Mean_Diff'] = np.abs(
            np.asarray(weights_dict['fused']) - df['Mean_4Methods'].values
        )

        df = df.set_index('Subcriteria')
        return self._save_csv(df, 'weights_analysis.csv')

    # ==================================================================
    #  2. RANKINGS
    # ==================================================================

    def save_rankings(self, ranking_result: Any,
                      provinces: List[str]) -> str:
        """
        Save final ER rankings with all detail columns.
        """
        scores = ranking_result.final_scores
        ranks = ranking_result.final_ranking

        df = pd.DataFrame({
            'Province': provinces,
            'ER_Score': np.asarray(scores.values if hasattr(scores, 'values') else scores),
            'ER_Rank': np.asarray(ranks.values if hasattr(ranks, 'values') else ranks, dtype=int),
        })

        # Percentile rank
        n = len(df)
        df['Percentile'] = ((n - df['ER_Rank'] + 1) / n * 100).round(1)

        # Score z-score
        mean_s = df['ER_Score'].mean()
        std_s = df['ER_Score'].std()
        df['Z_Score'] = ((df['ER_Score'] - mean_s) / std_s).round(4) if std_s > 0 else 0

        # Tier classification
        df['Tier'] = pd.cut(
            df['ER_Rank'],
            bins=[0, n * 0.1, n * 0.25, n * 0.5, n * 0.75, n + 1],
            labels=['Elite (Top 10%)', 'High (10-25%)', 'Upper-Mid (25-50%)',
                    'Lower-Mid (50-75%)', 'Low (75-100%)'],
        )

        # Uncertainty columns if available
        try:
            unc = ranking_result.er_result.uncertainty
            if 'belief_entropy' in unc.columns:
                df['Belief_Entropy'] = unc['belief_entropy'].values
            if 'utility_interval_width' in unc.columns:
                df['Utility_Interval_Width'] = unc['utility_interval_width'].values
        except Exception:
            pass

        # Kendall's W as metadata
        df['Kendall_W'] = ranking_result.kendall_w

        df = df.sort_values('ER_Rank').reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = 'Position'
        return self._save_csv(df, 'final_rankings.csv', float_fmt='%.4f')

    # ==================================================================
    #  3. MCDM SCORES PER CRITERION
    # ==================================================================

    def save_mcdm_scores_by_criterion(
        self,
        ranking_result: Any,
        provinces: List[str],
    ) -> Dict[str, str]:
        """
        For each criterion group: Province × Method score matrix + ranks.
        Returns dict of {criterion: filepath}.
        """
        saved = {}
        for crit_id, method_scores in ranking_result.criterion_method_scores.items():
            # Build score DataFrame
            score_df = pd.DataFrame(index=provinces)
            rank_df = pd.DataFrame(index=provinces)

            for method, series in method_scores.items():
                vals = series.values if hasattr(series, 'values') else np.asarray(series)
                score_df[f'{method}_Score'] = vals
                rank_df[f'{method}_Rank'] = pd.Series(vals, index=provinces).rank(
                    ascending=False, method='min'
                ).astype(int)

            # Summary columns
            score_cols = [c for c in score_df.columns]
            score_df['Mean_Score'] = score_df[score_cols].mean(axis=1)
            score_df['StdDev_Score'] = score_df[score_cols].std(axis=1)
            score_df['CV_Score'] = np.where(
                score_df['Mean_Score'] > 0,
                score_df['StdDev_Score'] / score_df['Mean_Score'],
                0
            )

            # Merge scores and ranks
            combined = pd.concat([score_df, rank_df], axis=1)
            combined.index.name = 'Province'

            # Mean rank + final rank for this criterion
            rank_cols = [c for c in rank_df.columns]
            combined['Mean_Rank'] = rank_df[rank_cols].mean(axis=1).round(2)
            combined['Consensus_Rank'] = combined['Mean_Rank'].rank(method='min').astype(int)

            fname = f'mcdm_scores_{crit_id}.csv'
            path = self._save_csv(combined, fname, float_fmt='%.4f')
            saved[crit_id] = str(path)

        return saved

    # ==================================================================
    #  4. RANK COMPARISON (all methods × all criteria)
    # ==================================================================

    def save_rank_comparison(self, ranking_result: Any,
                             provinces: List[str]) -> str:
        """Full rank matrix: Province × (Criterion_Method)."""
        all_ranks = {}
        for crit_id, method_ranks in ranking_result.criterion_method_ranks.items():
            for method, ranks_series in method_ranks.items():
                col = f"{crit_id}_{method}"
                vals = ranks_series.values if hasattr(ranks_series, 'values') else np.asarray(ranks_series)
                all_ranks[col] = vals

        df = pd.DataFrame(all_ranks, index=provinces)
        df.index.name = 'Province'

        # Summary columns
        df['Mean_Rank'] = df.mean(axis=1).round(2)
        df['Median_Rank'] = df.median(axis=1).round(2)
        df['StdDev_Rank'] = df.std(axis=1).round(2)
        df['Best_Rank'] = df.drop(columns=['Mean_Rank', 'Median_Rank', 'StdDev_Rank'], errors='ignore').min(axis=1).astype(int)
        df['Worst_Rank'] = df.drop(columns=['Mean_Rank', 'Median_Rank', 'StdDev_Rank', 'Best_Rank'], errors='ignore').max(axis=1).astype(int)
        df['Rank_Range'] = df['Worst_Rank'] - df['Best_Rank']

        return self._save_csv(df, 'mcdm_rank_comparison.csv', float_fmt='%.0f')

    # ==================================================================
    #  5. FORECAST RESULTS
    # ==================================================================

    def save_forecast_results(self, forecast_result: Any) -> Dict[str, str]:
        """
        Save all forecast outputs:
          - predictions.csv
          - prediction_intervals.csv
          - model_contributions.csv
          - model_performance.csv
          - feature_importance.csv
          - cross_validation_scores.csv
          - forecast_summary.json
        """
        saved = {}

        # Predictions
        try:
            preds = forecast_result.predictions
            if preds is not None and not preds.empty:
                path = self._save_csv(preds, 'forecast_predictions.csv', float_fmt='%.4f')
                saved['predictions'] = path
        except Exception:
            pass

        # Prediction intervals
        try:
            intervals = forecast_result.prediction_intervals
            if intervals:
                lower = intervals.get('lower')
                upper = intervals.get('upper')
                if lower is not None and upper is not None:
                    # Combine into one DataFrame
                    combined = pd.DataFrame(index=lower.index)
                    for col in lower.columns:
                        combined[f'{col}_lower'] = lower[col]
                        combined[f'{col}_upper'] = upper[col]
                        if col in forecast_result.predictions.columns:
                            combined[f'{col}_point'] = forecast_result.predictions[col]
                            combined[f'{col}_width'] = upper[col] - lower[col]
                    combined.index.name = 'Entity'
                    path = self._save_csv(combined, 'prediction_intervals.csv', float_fmt='%.4f')
                    saved['prediction_intervals'] = path
        except Exception:
            pass

        # Model contributions
        try:
            contrib = forecast_result.model_contributions
            if contrib:
                df = pd.DataFrame([
                    {'Model': k, 'Weight': v,
                     'Rank': 0,
                     'Weight_Pct': f'{v*100:.1f}%'}
                    for k, v in sorted(contrib.items(), key=lambda x: x[1], reverse=True)
                ])
                df['Rank'] = range(1, len(df) + 1)
                df = df.set_index('Rank')
                path = self._save_csv(df, 'model_contributions.csv', float_fmt='%.6f')
                saved['model_contributions'] = path
        except Exception:
            pass

        # Model performance
        try:
            perf = forecast_result.model_performance
            if perf:
                rows = []
                for model, metrics in perf.items():
                    row = {'Model': model}
                    row.update(metrics)
                    rows.append(row)
                df = pd.DataFrame(rows).set_index('Model')
                path = self._save_csv(df, 'model_performance.csv', float_fmt='%.4f')
                saved['model_performance'] = path
        except Exception:
            pass

        # Feature importance
        try:
            imp = forecast_result.feature_importance
            if imp is not None and not imp.empty:
                # Ensure it has an Importance column or create one
                if 'Importance' not in imp.columns and len(imp.columns) > 0:
                    imp_copy = imp.copy()
                    imp_copy['Importance'] = imp_copy.iloc[:, 0]
                else:
                    imp_copy = imp.copy()
                imp_copy = imp_copy.sort_values('Importance', ascending=False)
                imp_copy['Cumulative'] = imp_copy['Importance'].cumsum()
                imp_copy['Rank'] = range(1, len(imp_copy) + 1)
                path = self._save_csv(imp_copy, 'feature_importance.csv', float_fmt='%.6f')
                saved['feature_importance'] = path
        except Exception:
            pass

        # Cross-validation scores
        try:
            cv = forecast_result.cross_validation_scores
            if cv:
                max_folds = max(len(v) for v in cv.values())
                rows = {}
                for model, scores in cv.items():
                    row = {f'Fold_{i+1}': s for i, s in enumerate(scores)}
                    row['Mean'] = np.mean(scores)
                    row['StdDev'] = np.std(scores)
                    row['Min'] = np.min(scores)
                    row['Max'] = np.max(scores)
                    rows[model] = row
                df = pd.DataFrame(rows).T
                df.index.name = 'Model'
                path = self._save_csv(df, 'cross_validation_scores.csv', float_fmt='%.4f')
                saved['cross_validation'] = path
        except Exception:
            pass

        # Holdout performance
        try:
            holdout = forecast_result.holdout_performance
            if holdout:
                df = pd.DataFrame([holdout])
                df.index = ['Holdout']
                df.index.name = 'Set'
                path = self._save_csv(df, 'holdout_performance.csv', float_fmt='%.4f')
                saved['holdout'] = path
        except Exception:
            pass

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
            path = self._save_json(summary, 'forecast_summary.json')
            saved['summary'] = path
        except Exception:
            pass

        return saved

    # ==================================================================
    #  6. ANALYSIS / SENSITIVITY
    # ==================================================================

    def save_analysis_results(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save all sensitivity analysis outputs.
        """
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
                    lambda x: 'High' if x > 0.1 else ('Medium' if x > 0.05 else 'Low')
                )
                path = self._save_csv(df, 'sensitivity_criteria.csv', float_fmt='%.6f')
                saved['criteria'] = path
        except Exception:
            pass

        # Subcriteria sensitivity
        try:
            if hasattr(sens, 'subcriteria_sensitivity') and sens.subcriteria_sensitivity:
                df = pd.DataFrame([
                    {'Subcriteria': k, 'Sensitivity': v}
                    for k, v in sorted(sens.subcriteria_sensitivity.items(),
                                       key=lambda x: x[1], reverse=True)
                ]).set_index('Subcriteria')
                df['Rank'] = range(1, len(df) + 1)
                path = self._save_csv(df, 'sensitivity_subcriteria.csv', float_fmt='%.6f')
                saved['subcriteria'] = path
        except Exception:
            pass

        # Rank stability per province
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
                        'Stable' if x > 0.7 else (
                            'Moderate' if x > 0.5 else 'Volatile'
                        )
                    )
                )
                path = self._save_csv(df, 'sensitivity_rank_stability.csv', float_fmt='%.4f')
                saved['rank_stability'] = path
        except Exception:
            pass

        # Top-N stability
        try:
            if hasattr(sens, 'top_n_stability') and sens.top_n_stability:
                df = pd.DataFrame([
                    {'Top_N': f'Top-{n}', 'N': n, 'Stability': v}
                    for n, v in sorted(sens.top_n_stability.items())
                ]).set_index('Top_N')
                path = self._save_csv(df, 'sensitivity_top_n_stability.csv', float_fmt='%.4f')
                saved['top_n'] = path
        except Exception:
            pass

        # Temporal stability
        try:
            if hasattr(sens, 'temporal_stability') and sens.temporal_stability:
                df = pd.DataFrame([
                    {'Year_Pair': k, 'Rank_Correlation': v}
                    for k, v in sorted(sens.temporal_stability.items())
                ]).set_index('Year_Pair')
                df['Interpretation'] = df['Rank_Correlation'].apply(
                    lambda x: 'Strong' if x > 0.8 else ('Moderate' if x > 0.5 else 'Weak')
                )
                path = self._save_csv(df, 'sensitivity_temporal.csv', float_fmt='%.4f')
                saved['temporal'] = path
        except Exception:
            pass

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
                lambda x: 'High' if x > 0.1 else ('Medium' if x > 0.05 else 'Low')
            )
            path = self._save_csv(df, 'sensitivity_ifs.csv', float_fmt='%.6f')
            saved['ifs'] = path
        except Exception:
            pass

        # Overall robustness summary JSON
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
            path = self._save_json(robustness, 'sensitivity_summary.json')
            saved['summary'] = path
        except Exception:
            pass

        return saved

    # ==================================================================
    #  7. ER UNCERTAINTY
    # ==================================================================

    def save_er_uncertainty(self, ranking_result: Any,
                            provinces: List[str]) -> Optional[str]:
        """Save ER belief entropy and utility interval width per province."""
        try:
            unc = ranking_result.er_result.uncertainty.copy()
            unc.index.name = 'Province'
            return self._save_csv(unc, 'prediction_uncertainty_er.csv', float_fmt='%.6f')
        except Exception:
            return None

    # ==================================================================
    #  8. DATA SUMMARY STATISTICS
    # ==================================================================

    def save_data_summary(self, panel_data: Any) -> str:
        """Descriptive statistics for the latest-year cross section."""
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
    #  9. COMPREHENSIVE REPORT
    # ==================================================================

    def build_comprehensive_report(
        self,
        panel_data: Any,
        weights: Dict[str, Any],
        ranking_result: Any,
        forecast_result: Optional[Any],
        analysis_results: Dict[str, Any],
        execution_time: float,
        figure_paths: List[str] = None,
    ) -> str:
        """
        Build a comprehensive, publication-ready analysis report in formal
        academic style.

        Returns the report text AND saves it to reports/report.txt.
        """
        from datetime import datetime

        W = 98  # line width
        lines: List[str] = []

        # ── formatting helpers ───────────────────────────────────
        def _sec(num: str, title: str, level: int = 1):
            """Emit a numbered section heading."""
            lines.append('')
            if level == 1:
                lines.append('=' * W)
                lines.append(f'  {num}  {title.upper()}')
                lines.append('=' * W)
            elif level == 2:
                lines.append(f'  {num}  {title}')
                lines.append('  ' + '-' * (len(num) + len(title) + 2))
            else:
                lines.append(f'    {num}  {title}')

        def _p(*parts):
            """Emit a paragraph (each part is one line, auto-indented)."""
            for p in parts:
                for wrapped in textwrap.wrap(p, width=W - 6):
                    lines.append('    ' + wrapped)

        def _kv(label: str, value: str, indent: int = 6):
            """Key-value pair with right-aligned label."""
            pad = ' ' * indent
            lines.append(f'{pad}{label:<36s}: {value}')

        def _tbl(headers: List[str], rows: List[List[str]],
                 col_w: Optional[List[int]] = None):
            """Emit a plain fixed-width table."""
            if col_w is None:
                col_w = [max(14, len(h) + 2) for h in headers]
            hdr = '    ' + '  '.join(
                f'{h:^{w}}' for h, w in zip(headers, col_w))
            lines.append(hdr)
            lines.append('    ' + '  '.join('-' * w for w in col_w))
            for row in rows:
                cells = []
                for c, w in zip(row, col_w):
                    try:
                        float(c.replace('+', '').replace('%', ''))
                        cells.append(f'{c:>{w}}')
                    except (ValueError, AttributeError):
                        cells.append(f'{c:<{w}}')
                lines.append('    ' + '  '.join(cells))

        # ── derived arrays ──────────────────────────────────────
        scores_arr = np.asarray(
            ranking_result.final_scores.values
            if hasattr(ranking_result.final_scores, 'values')
            else ranking_result.final_scores)
        ranks_arr = np.asarray(
            ranking_result.final_ranking.values
            if hasattr(ranking_result.final_ranking, 'values')
            else ranking_result.final_ranking, dtype=int)
        order = np.argsort(ranks_arr)
        n_prov = len(panel_data.provinces)
        n_years = len(panel_data.years)
        fused = np.asarray(weights['fused'])
        subcriteria = weights['subcriteria']
        sens = analysis_results.get('sensitivity')

        # ================================================================
        # FRONT MATTER
        # ================================================================
        lines.append('=' * W)
        lines.append('')
        lines.append('  MULTI-CRITERIA DECISION ANALYSIS OF VIETNAMESE PROVINCIAL')
        lines.append('  COMPETITIVENESS: AN INTEGRATED IFS-EVIDENTIAL REASONING')
        lines.append('  APPROACH WITH MACHINE-LEARNING FORECASTING')
        lines.append('')
        lines.append('  Technical Analysis Report')
        lines.append('')
        lines.append('=' * W)
        lines.append('')
        _kv('Date of Generation', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 2)
        _kv('Computational Runtime', f'{execution_time:.2f} s', 2)
        _kv('Framework Version', 'ML-MCDM v4.0', 2)
        lines.append('')

        # TABLE OF CONTENTS
        lines.append('  TABLE OF CONTENTS')
        lines.append('  ' + '-' * 20)
        toc = [
            '1.   Executive Summary',
            '2.   Data Description and Descriptive Statistics',
            '3.   Objective Weight Derivation',
            '4.   Hierarchical Evidential Reasoning Ranking',
            '5.   Criterion-Level MCDM Evaluation',
            '6.   Inter-Method Agreement and Concordance Analysis',
            '7.   Sensitivity and Robustness Analysis',
            '8.   Machine-Learning Forecasting',
            '9.   Validity Assessment',
            '10.  Methodological Notes and References',
            '      Appendix A  --  Output File Inventory',
        ]
        for t in toc:
            lines.append(f'    {t}')

        # ================================================================
        # 1. EXECUTIVE SUMMARY
        # ================================================================
        _sec('1.', 'Executive Summary')
        lines.append('')
        _p(
            f'This report documents a comprehensive multi-criteria decision-making '
            f'(MCDM) evaluation of {n_prov} Vietnamese provinces over the period '
            f'{min(panel_data.years)}\u2013{max(panel_data.years)} ({n_years} years). '
            f'The analytical framework integrates {panel_data.n_subcriteria} subcriteria '
            f'organised into {panel_data.n_criteria} criteria groups, evaluated through '
            f'{len(ranking_result.methods_used)} MCDM methods comprising both classical '
            f'and Intuitionistic Fuzzy Set (IFS) variants. Final provincial rankings '
            f'are obtained via a two-stage Evidential Reasoning (ER) aggregation '
            f'procedure that combines belief structures from all constituent methods '
            f'while explicitly quantifying residual uncertainty.'
        )
        lines.append('')
        _p(
            'Subcriteria weights are derived through four independent objective '
            'weighting techniques -- Entropy, CRITIC, MEREC, and Standard Deviation '
            '-- subsequently fused using a reliability-weighted Bayesian bootstrap '
            'scheme. A machine-learning forecasting module, based on a Super Learner '
            'meta-ensemble with Conformal Prediction intervals, projects provincial '
            'scores one year ahead, providing a forward-looking dimension to the '
            'assessment.'
        )
        lines.append('')

        # Key findings
        lines.append('  Principal Findings')
        lines.append('  ' + '-' * 20)
        lines.append('')

        _tbl(
            ['Rank', 'Province', 'ER Score'],
            [[str(i + 1),
              panel_data.provinces[order[i]],
              f'{scores_arr[order[i]]:.4f}']
             for i in range(min(5, n_prov))],
            [6, 24, 12],
        )
        lines.append('')
        lines.append('    Table 1(a).  Highest-ranked provinces.')
        lines.append('')

        _tbl(
            ['Rank', 'Province', 'ER Score'],
            [[str(n_prov - i),
              panel_data.provinces[order[-(i + 1)]],
              f'{scores_arr[order[-(i + 1)]]:.4f}']
             for i in range(min(5, n_prov))],
            [6, 24, 12],
        )
        lines.append('')
        lines.append('    Table 1(b).  Lowest-ranked provinces.')
        lines.append('')

        _kv("Kendall's W (concordance)", f'{ranking_result.kendall_w:.4f}')
        if sens and hasattr(sens, 'overall_robustness'):
            _kv('Overall Robustness Index', f'{sens.overall_robustness:.4f}')
            _kv('Confidence Level',
                f'{getattr(sens, "confidence_level", 0.95):.0%}')
        lines.append('')

        # ================================================================
        # 2. DATA DESCRIPTION
        # ================================================================
        _sec('2.', 'Data Description and Descriptive Statistics')
        lines.append('')
        _p(
            f'The dataset comprises a balanced panel of {n_prov} provinces observed '
            f'annually from {min(panel_data.years)} to {max(panel_data.years)}, '
            f'yielding {n_prov * n_years:,} province-year observations. Each '
            f'observation is characterised by {panel_data.n_subcriteria} subcriteria '
            f'nested within {panel_data.n_criteria} higher-order criteria groups. '
            f'Subcriteria capture both benefit-type and cost-type indicators, '
            f'as specified in the project codebook.'
        )
        lines.append('')

        _kv('Number of Provinces (N)', str(n_prov))
        _kv('Temporal Span', f'{min(panel_data.years)}\u2013{max(panel_data.years)}')
        _kv('Number of Annual Periods (T)', str(n_years))
        _kv('Number of Criteria', str(panel_data.n_criteria))
        _kv('Number of Subcriteria', str(panel_data.n_subcriteria))
        _kv('Total Panel Observations (N x T)', f'{n_prov * n_years:,}')
        lines.append('')

        # Descriptive stats for latest year
        try:
            latest_year = max(panel_data.years)
            desc = panel_data.subcriteria_cross_section[latest_year].describe()
            lines.append(f'    Table 2.  Descriptive statistics for the cross-section of year {latest_year}.')
            lines.append('')
            rows_desc = []
            for sc in subcriteria:
                if sc in desc.columns:
                    mu = desc[sc]['mean']
                    sd = desc[sc]['std']
                    mn = desc[sc]['min']
                    mx = desc[sc]['max']
                    cv = sd / mu if mu != 0 else 0.0
                    rows_desc.append([sc, f'{mu:.4f}', f'{sd:.4f}',
                                      f'{mn:.4f}', f'{mx:.4f}', f'{cv:.4f}'])
            _tbl(['Subcriteria', 'Mean', 'Std Dev', 'Min', 'Max', 'CV'],
                  rows_desc, [14, 10, 10, 10, 10, 10])
            lines.append('')
        except Exception:
            pass

        # ================================================================
        # 3. WEIGHT ANALYSIS
        # ================================================================
        _sec('3.', 'Objective Weight Derivation')
        lines.append('')
        _p(
            'Subcriteria weights are determined through four mathematically '
            'independent objective weighting procedures, each exploiting a '
            'distinct property of the decision matrix. The individual weight '
            'vectors are subsequently integrated into a single fused vector '
            'using a reliability-weighted Bayesian bootstrap fusion scheme that '
            'accounts for inter-method agreement at the subcriteria level.'
        )
        lines.append('')
        _p(
            'The Entropy method (Shannon, 1948) assigns higher weight to '
            'subcriteria exhibiting greater informational diversity. CRITIC '
            '(Diakoulaki et al., 1995) simultaneously considers contrast '
            'intensity and inter-criteria correlation. MEREC (Keshavarz-Ghorabaee '
            'et al., 2021) quantifies subcriteria importance via removal effects, '
            'while Standard Deviation weighting reflects dispersion in raw values.'
        )
        lines.append('')

        rows_w = []
        for i, sc in enumerate(subcriteria):
            rows_w.append([
                sc,
                f'{weights["entropy"][i]:.4f}',
                f'{weights["critic"][i]:.4f}',
                f'{weights["merec"][i]:.4f}',
                f'{weights["std_dev"][i]:.4f}',
                f'{weights["fused"][i]:.4f}',
            ])
        lines.append('    Table 3.  Subcriteria weights by method and fused weights.')
        lines.append('')
        _tbl(['Subcriteria', 'Entropy', 'CRITIC', 'MEREC', 'Std Dev', 'Fused'],
             rows_w, [14, 10, 10, 10, 10, 10])
        lines.append('')

        lines.append('    Fused Weight Summary Statistics')
        lines.append('    ' + '-' * 32)
        _kv('Sum', f'{fused.sum():.6f}')
        _kv('Maximum', f'{fused.max():.6f}  ({subcriteria[np.argmax(fused)]})')
        _kv('Minimum', f'{fused.min():.6f}  ({subcriteria[np.argmin(fused)]})')
        _kv('Range', f'{fused.max() - fused.min():.6f}')
        _kv('Shannon Entropy of Fused Vector',
            f'{-np.sum(fused * np.log(fused + 1e-12)):.4f}')
        lines.append('')

        # Cross-method correlation
        method_weights = np.column_stack([
            weights['entropy'], weights['critic'],
            weights['merec'], weights['std_dev']
        ])
        try:
            corr_matrix = np.corrcoef(method_weights.T)
            method_labels = ['Entropy', 'CRITIC', 'MEREC', 'Std Dev']
            lines.append('    Table 4.  Pearson correlation matrix of weight vectors.')
            lines.append('')
            corr_rows = []
            for i, ml in enumerate(method_labels):
                corr_rows.append(
                    [ml] + [f'{corr_matrix[i, j]:.3f}' for j in range(4)])
            _tbl([''] + method_labels, corr_rows,
                 [10, 10, 10, 10, 10])
            lines.append('')
        except Exception:
            pass

        # ================================================================
        # 4. HIERARCHICAL ER RANKING
        # ================================================================
        _sec('4.', 'Hierarchical Evidential Reasoning Ranking')
        lines.append('')
        _p(
            'The Evidential Reasoning approach (Yang & Xu, 2002) is employed as a '
            'two-stage belief-structure aggregation mechanism. In Stage 1, the scores '
            f'produced by {len(ranking_result.methods_used)} MCDM methods for each '
            'criterion group are transformed into basic probability assignments and '
            'combined using the ER analytical algorithm, yielding criterion-level '
            'belief degrees. In Stage 2, these criterion-level beliefs are further '
            'aggregated using the derived criterion weights to obtain the final '
            'composite score for each province. Uncertainty is explicitly preserved '
            'through residual belief mass allocated to an unassigned frame.'
        )
        lines.append('')

        _kv('Aggregation Framework', 'Evidential Reasoning (Yang & Xu, 2002)')
        _kv('Fuzzy Extension', 'Intuitionistic Fuzzy Sets (Atanassov, 1986)')
        _kv('Number of MCDM Methods', str(len(ranking_result.methods_used)))
        _kv("Kendall's W", f'{ranking_result.kendall_w:.4f}')
        _kv('Target Year', str(ranking_result.target_year))
        lines.append('')

        # Full ranking table
        mean_s = scores_arr.mean()
        std_s = scores_arr.std() if scores_arr.std() > 0 else 1.0
        rank_rows = []
        for idx in order:
            r = ranks_arr[idx]
            s = scores_arr[idx]
            z = (s - mean_s) / std_s
            pct = (n_prov - r + 1) / n_prov * 100
            q = 'Q1' if pct >= 75 else ('Q2' if pct >= 50 else ('Q3' if pct >= 25 else 'Q4'))
            rank_rows.append([
                str(r), panel_data.provinces[idx],
                f'{s:.4f}', f'{z:+.3f}', q,
            ])
        lines.append('    Table 5.  Complete provincial ranking by ER composite score.')
        lines.append('')
        _tbl(['Rank', 'Province', 'ER Score', 'Z-Score', 'Quartile'],
             rank_rows, [6, 24, 10, 10, 10])
        lines.append('')

        # Score distribution
        lines.append('    Distributional Properties of ER Scores')
        lines.append('    ' + '-' * 40)
        _kv('Mean', f'{scores_arr.mean():.4f}')
        _kv('Median', f'{np.median(scores_arr):.4f}')
        _kv('Standard Deviation', f'{scores_arr.std():.4f}')
        _kv('Skewness', f'{pd.Series(scores_arr).skew():.4f}')
        _kv('Excess Kurtosis', f'{pd.Series(scores_arr).kurtosis():.4f}')
        iqr = np.percentile(scores_arr, 75) - np.percentile(scores_arr, 25)
        _kv('Inter-Quartile Range (IQR)', f'{iqr:.4f}')
        lines.append('')

        # ER Uncertainty
        try:
            unc = ranking_result.er_result.uncertainty
            lines.append('    Evidential Reasoning Uncertainty Diagnostics')
            lines.append('    ' + '-' * 46)
            _kv('Mean Belief Entropy',
                f'{unc["belief_entropy"].mean():.4f} (SD = {unc["belief_entropy"].std():.4f})')
            _kv('Mean Utility Interval Width',
                f'{unc["utility_interval_width"].mean():.4f} '
                f'(SD = {unc["utility_interval_width"].std():.4f})')
            lines.append('')
        except Exception:
            pass

        # ================================================================
        # 5. PER-CRITERION ANALYSIS
        # ================================================================
        _sec('5.', 'Criterion-Level MCDM Evaluation')
        lines.append('')
        _p(
            f'Each of the {panel_data.n_criteria} criteria groups is independently '
            f'evaluated by the full battery of {len(ranking_result.methods_used)} '
            f'MCDM methods. Normalised method scores within each criterion are '
            f'subsequently used as evidence in the Stage 1 ER aggregation. The '
            f'criterion-level weights applied at Stage 2 are reported below, followed '
            f'by a summary of the three highest-performing provinces per criterion.'
        )
        lines.append('')

        # Criterion weights
        crit_w = ranking_result.criterion_weights_used
        crit_rows = [[c, f'{w:.6f}']
                     for c, w in sorted(crit_w.items())]
        lines.append('    Table 6.  Criterion weights applied in Stage 2 ER aggregation.')
        lines.append('')
        _tbl(['Criterion', 'Weight'], crit_rows, [14, 14])
        lines.append('')

        # Top 3 per criterion
        for crit_id in sorted(ranking_result.criterion_method_scores.keys()):
            method_scores = ranking_result.criterion_method_scores[crit_id]
            all_scores_crit = []
            for method, series in method_scores.items():
                vals = series.values if hasattr(series, 'values') else np.asarray(series)
                all_scores_crit.append(vals)
            avg_scores = np.mean(all_scores_crit, axis=0)
            top3_idx = np.argsort(avg_scores)[-3:][::-1]
            lines.append(f'    {crit_id}:  ', )
            for j, idx in enumerate(top3_idx):
                lines.append(
                    f'      ({j+1}) {panel_data.provinces[idx]:<20}  '
                    f'mean score = {avg_scores[idx]:.4f}')
        lines.append('')

        # ================================================================
        # 6. RANK AGREEMENT
        # ================================================================
        _sec('6.', 'Inter-Method Agreement and Concordance Analysis')
        lines.append('')
        _p(
            "Kendall's coefficient of concordance (W) provides a non-parametric "
            "measure of agreement among the rank orderings produced by the "
            f"constituent MCDM methods. The observed value W = "
            f"{ranking_result.kendall_w:.4f} indicates "
            + ('strong' if ranking_result.kendall_w >= 0.7
               else ('moderate' if ranking_result.kendall_w >= 0.5
                     else ('fair' if ranking_result.kendall_w >= 0.3
                           else 'weak')))
            + " concordance, suggesting that the integrated methods yield "
            + ('highly consistent' if ranking_result.kendall_w >= 0.7
               else ('broadly consistent' if ranking_result.kendall_w >= 0.5
                     else 'partially divergent'))
            + ' rankings.'
        )
        lines.append('')

        # Frequency in top 5
        try:
            top5_sets = []
            for crit_id, method_ranks in ranking_result.criterion_method_ranks.items():
                for method, rank_series in method_ranks.items():
                    vals = (rank_series.values if hasattr(rank_series, 'values')
                            else np.asarray(rank_series))
                    top5_sets.append(set(np.argsort(vals)[:5]))
            from collections import Counter
            all_top5 = Counter()
            for s in top5_sets:
                for idx in s:
                    all_top5[idx] += 1
            total_combos = len(top5_sets)

            t5_rows = [
                [panel_data.provinces[idx], str(count),
                 f'{count / total_combos:.1%}']
                for idx, count in all_top5.most_common(10)
            ]
            lines.append('    Table 7.  Provinces most frequently ranked in the top 5')
            lines.append('              across all criterion-method combinations.')
            lines.append('')
            _tbl(['Province', 'Count', 'Frequency'],
                 t5_rows, [24, 10, 12])
            lines.append('')
        except Exception:
            pass

        # ================================================================
        # 7. SENSITIVITY ANALYSIS
        # ================================================================
        _sec('7.', 'Sensitivity and Robustness Analysis')
        lines.append('')
        if sens:
            _p(
                'A comprehensive sensitivity analysis is conducted to assess the '
                'stability of the ranking with respect to perturbations in weight '
                'values, IFS membership parameters, and temporal variation. '
                'Robustness is measured as a composite index aggregating multiple '
                'stability dimensions.'
            )
            lines.append('')
            _kv('Overall Robustness Index', f'{sens.overall_robustness:.4f}')
            _kv('Confidence Level',
                f'{getattr(sens, "confidence_level", 0.95):.0%}')
            lines.append('')

            # 7.1 Criteria sensitivity
            if hasattr(sens, 'criteria_sensitivity') and sens.criteria_sensitivity:
                _sec('7.1', 'Criteria Weight Sensitivity', 2)
                lines.append('')
                _p(
                    'Each criterion weight is perturbed individually while '
                    'renormalising the remaining weights. The sensitivity index '
                    'records the mean absolute rank displacement resulting from '
                    'each perturbation. Higher values indicate greater influence '
                    'on the final ranking.'
                )
                lines.append('')
                cs_rows = []
                for k, v in sorted(sens.criteria_sensitivity.items(),
                                   key=lambda x: x[1], reverse=True):
                    interp = ('High' if v > 0.1
                              else ('Medium' if v > 0.05 else 'Low'))
                    cs_rows.append([k, f'{v:.4f}', interp])
                lines.append('    Table 8.  Criteria weight sensitivity indices.')
                lines.append('')
                _tbl(['Criterion', 'Sensitivity', 'Classification'],
                     cs_rows, [14, 12, 16])
                lines.append('')

            # 7.2 Subcriteria sensitivity
            if hasattr(sens, 'subcriteria_sensitivity') and sens.subcriteria_sensitivity:
                _sec('7.2', 'Subcriteria Weight Sensitivity (Top 15)', 2)
                lines.append('')
                ss_rows = [
                    [k, f'{v:.4f}']
                    for k, v in sorted(sens.subcriteria_sensitivity.items(),
                                       key=lambda x: x[1], reverse=True)[:15]
                ]
                lines.append('    Table 9.  Most influential subcriteria by sensitivity index.')
                lines.append('')
                _tbl(['Subcriteria', 'Sensitivity'], ss_rows, [14, 12])
                lines.append('')

            # 7.3 Top-N stability
            if hasattr(sens, 'top_n_stability') and sens.top_n_stability:
                _sec('7.3', 'Top-N Ranking Stability', 2)
                lines.append('')
                _p(
                    'The Top-N stability metric quantifies the proportion of '
                    'provinces that remain within the top N positions across '
                    'all weight perturbation scenarios. A value close to 1.0 '
                    'indicates that the top-N set is invariant to reasonable '
                    'weight changes.'
                )
                lines.append('')
                tn_rows = [
                    [f'Top-{n}', f'{stab:.4f}', f'{stab:.1%}']
                    for n, stab in sorted(sens.top_n_stability.items())
                ]
                lines.append('    Table 10.  Top-N set stability under weight perturbation.')
                lines.append('')
                _tbl(['Tier', 'Index', 'Percentage'], tn_rows, [10, 10, 12])
                lines.append('')

            # 7.4 Temporal stability
            if hasattr(sens, 'temporal_stability') and sens.temporal_stability:
                _sec('7.4', 'Temporal Rank Stability', 2)
                lines.append('')
                _p(
                    "Spearman rank correlations between consecutive years' "
                    "rankings indicate the degree of temporal persistence in "
                    "provincial performance."
                )
                lines.append('')
                ts_rows = [
                    [pair, f'{corr:.4f}',
                     'Strong' if corr > 0.8 else ('Moderate' if corr > 0.5 else 'Weak')]
                    for pair, corr in sorted(sens.temporal_stability.items())
                ]
                lines.append('    Table 11.  Year-to-year rank correlation coefficients.')
                lines.append('')
                _tbl(['Year Pair', 'Spearman rho', 'Strength'],
                     ts_rows, [16, 14, 12])
                lines.append('')

            # 7.5 IFS sensitivity
            _sec('7.5', 'IFS Membership Parameter Sensitivity', 2)
            lines.append('')
            _p(
                'Perturbations to the IFS membership (mu) and non-membership (nu) '
                'functions assess the sensitivity of the ranking to the fuzzy '
                'parameterisation. Low sensitivity values indicate that the IFS '
                'extension is robustly calibrated.'
            )
            lines.append('')
            _kv('Membership (mu) sensitivity',
                f'{getattr(sens, "ifs_membership_sensitivity", 0):.4f}')
            _kv('Non-membership (nu) sensitivity',
                f'{getattr(sens, "ifs_nonmembership_sensitivity", 0):.4f}')
            lines.append('')

            # 7.6 Province rank stability
            if hasattr(sens, 'rank_stability') and sens.rank_stability:
                _sec('7.6', 'Provincial Rank Stability', 2)
                lines.append('')
                _p(
                    'The rank stability coefficient for each province measures '
                    'the invariance of its ranking position across all '
                    'perturbation scenarios. Values approaching 1.0 indicate '
                    'a highly stable position, while lower values suggest '
                    'sensitivity to methodological assumptions.'
                )
                lines.append('')
                sorted_stab = sorted(sens.rank_stability.items(),
                                     key=lambda x: x[1])

                vol_rows = [[p, f'{s:.4f}'] for p, s in sorted_stab[:10]]
                stb_rows = [[p, f'{s:.4f}'] for p, s in sorted_stab[-10:]]

                lines.append('    Table 12(a).  Ten most volatile provinces.')
                lines.append('')
                _tbl(['Province', 'Stability'], vol_rows, [24, 12])
                lines.append('')

                lines.append('    Table 12(b).  Ten most stable provinces.')
                lines.append('')
                _tbl(['Province', 'Stability'], stb_rows, [24, 12])
                lines.append('')
        else:
            _p('Sensitivity analysis was not executed in this run.')
            lines.append('')

        # ================================================================
        # 8. ML FORECASTING
        # ================================================================
        _sec('8.', 'Machine-Learning Forecasting')
        lines.append('')
        if forecast_result is not None:
            _p(
                'A Super Learner meta-ensemble (van der Laan et al., 2007) is '
                'employed to forecast provincial scores one period ahead. The '
                'Super Learner constructs an optimally weighted combination of '
                'heterogeneous base learners via cross-validated risk '
                'minimisation. Prediction intervals are obtained through '
                'Conformal Prediction (Vovk et al., 2005), which provides '
                'distribution-free, finite-sample coverage guarantees.'
            )
            lines.append('')

            # 8.1 Model contributions
            if (hasattr(forecast_result, 'model_contributions')
                    and forecast_result.model_contributions):
                _sec('8.1', 'Super Learner Model Contributions', 2)
                lines.append('')
                mc_rows = [
                    [model, f'{w:.4f}', f'{w * 100:.1f}%']
                    for model, w in sorted(
                        forecast_result.model_contributions.items(),
                        key=lambda x: x[1], reverse=True)
                ]
                lines.append('    Table 13.  Base-learner weights in the Super Learner ensemble.')
                lines.append('')
                _tbl(['Model', 'Weight', 'Contribution (%)'],
                     mc_rows, [28, 10, 16])
                lines.append('')

            # 8.2 Model performance
            if (hasattr(forecast_result, 'model_performance')
                    and forecast_result.model_performance):
                _sec('8.2', 'Individual Model Performance', 2)
                lines.append('')
                perf = forecast_result.model_performance
                all_metrics = sorted(
                    {m for metrics in perf.values() for m in metrics})
                mp_rows = [
                    [model] + [f'{perf[model].get(m, 0):.4f}'
                               for m in all_metrics]
                    for model in sorted(perf.keys())
                ]
                lines.append('    Table 14.  Out-of-sample performance metrics by base learner.')
                lines.append('')
                _tbl(['Model'] + [m.upper() for m in all_metrics],
                     mp_rows, [28] + [10] * len(all_metrics))
                lines.append('')

            # 8.3 Cross-validation
            if (hasattr(forecast_result, 'cross_validation_scores')
                    and forecast_result.cross_validation_scores):
                _sec('8.3', 'Cross-Validation Results', 2)
                lines.append('')
                cv_rows = []
                for model, sc in sorted(
                        forecast_result.cross_validation_scores.items()):
                    a = np.asarray(sc)
                    cv_rows.append([
                        model, f'{a.mean():.4f}', f'{a.std():.4f}',
                        f'{a.min():.4f}', f'{a.max():.4f}',
                    ])
                lines.append('    Table 15.  K-fold cross-validation summary (R-squared).')
                lines.append('')
                _tbl(['Model', 'Mean', 'Std Dev', 'Min', 'Max'],
                     cv_rows, [28, 10, 10, 10, 10])
                lines.append('')

            # 8.4 Holdout
            if (hasattr(forecast_result, 'holdout_performance')
                    and forecast_result.holdout_performance):
                _sec('8.4', 'Holdout Validation Set Performance', 2)
                lines.append('')
                for metric, val in forecast_result.holdout_performance.items():
                    _kv(metric, f'{val:.4f}')
                lines.append('')

            # 8.5 Feature importance
            if (hasattr(forecast_result, 'feature_importance')
                    and forecast_result.feature_importance is not None):
                imp = forecast_result.feature_importance
                if not imp.empty:
                    _sec('8.5', 'Feature Importance (Top 20)', 2)
                    lines.append('')
                    _p(
                        'Feature importance scores are derived from the '
                        'meta-ensemble and reflect the marginal predictive '
                        'contribution of each temporal feature.'
                    )
                    lines.append('')
                    if 'Importance' in imp.columns:
                        imp_sorted = imp.sort_values(
                            'Importance', ascending=False).head(20)
                        cumsum = 0.0
                        fi_rows = []
                        for rank, (feat, row) in enumerate(
                                imp_sorted.iterrows(), 1):
                            cumsum += row['Importance']
                            fi_rows.append([
                                str(rank), str(feat),
                                f'{row["Importance"]:.4f}',
                                f'{cumsum:.4f}',
                            ])
                        lines.append('    Table 16.  Feature importance ranking.')
                        lines.append('')
                        _tbl(['Rank', 'Feature', 'Importance', 'Cumulative'],
                             fi_rows, [6, 32, 12, 12])
                        lines.append('')

            # 8.6 Prediction intervals
            if (hasattr(forecast_result, 'prediction_intervals')
                    and forecast_result.prediction_intervals):
                intervals = forecast_result.prediction_intervals
                lower = intervals.get('lower')
                upper = intervals.get('upper')
                if lower is not None and upper is not None:
                    _sec('8.6', 'Conformal Prediction Interval Diagnostics', 2)
                    lines.append('')
                    widths_arr = (upper.values - lower.values).flatten()
                    _kv('Nominal Coverage Target', '95%')
                    _kv('Mean Interval Width', f'{widths_arr.mean():.4f}')
                    _kv('Median Interval Width', f'{np.median(widths_arr):.4f}')
                    _kv('Minimum Width', f'{widths_arr.min():.4f}')
                    _kv('Maximum Width', f'{widths_arr.max():.4f}')
                    lines.append('')
        else:
            _p(
                'The machine-learning forecasting module was not executed '
                'in the current pipeline run.'
            )
            lines.append('')

        # ================================================================
        # 9. VALIDITY ASSESSMENT
        # ================================================================
        _sec('9.', 'Validity Assessment')
        lines.append('')
        _p(
            'A set of diagnostic checks is performed to validate the internal '
            'consistency and correctness of the analysis outputs. Each criterion '
            'is assessed against an expected condition; passing indicates conformity '
            'with methodological requirements.'
        )
        lines.append('')

        checks = [
            ('Weight normalisation (sum = 1)',
             abs(fused.sum() - 1.0) < 0.01),
            ('Rank completeness (all provinces ranked)',
             len(set(ranks_arr)) == n_prov),
            ('Score range validity (0 <= s <= 1)',
             scores_arr.min() >= 0 and scores_arr.max() <= 1),
            ("Kendall's W >= 0.5 (moderate concordance)",
             ranking_result.kendall_w >= 0.5),
        ]
        if sens and hasattr(sens, 'overall_robustness'):
            checks.append((
                'Robustness index >= 0.7',
                sens.overall_robustness >= 0.7))

        vc_rows = [
            [desc, 'PASS' if passed else 'FAIL']
            for desc, passed in checks
        ]
        lines.append('    Table 17.  Internal validity diagnostics.')
        lines.append('')
        _tbl(['Criterion', 'Result'], vc_rows, [50, 8])
        lines.append('')

        # ================================================================
        # 10. METHODOLOGICAL NOTES
        # ================================================================
        _sec('10.', 'Methodological Notes and References')
        lines.append('')
        _p(
            'This section provides a concise methodological summary and key '
            'literature references for each analytical component.'
        )
        lines.append('')

        _sec('10.1', 'Objective Weighting Methods', 2)
        lines.append('')
        notes = [
            ('Entropy', 'Shannon (1948).  Information-theoretic derivation exploiting '
             'probability-distribution diversity across alternatives.'),
            ('CRITIC', 'Diakoulaki, Mavrotas & Papayannakis (1995).  Weights based on '
             'the product of contrast intensity (standard deviation) and '
             'conflicting information (inter-criteria correlation).'),
            ('MEREC', 'Keshavarz-Ghorabaee, Amiri, Zavadskas, Turskis & '
             'Antucheviciene (2021).  Method based on the Effect of Removal of a '
             'Criterion; quantifies each criterion\'s contribution via logarithmic '
             'performance removal ratios.'),
            ('Standard Deviation', 'Dispersion-based weighting assigning higher weight '
             'to criteria exhibiting greater variability across alternatives.'),
            ('Fusion', 'Reliability-weighted Bayesian bootstrap combination that '
             'accounts for inter-method agreement at the subcriteria level.'),
        ]
        for name, desc in notes:
            lines.append(f'    {name}')
            for w in textwrap.wrap(desc, width=W - 10):
                lines.append(f'        {w}')
            lines.append('')

        _sec('10.2', 'MCDM Methods', 2)
        lines.append('')
        _p(
            'Twelve methods are applied: six classical (TOPSIS, VIKOR, PROMETHEE, '
            'COPRAS, EDAS, SAW) and six Intuitionistic Fuzzy Set extensions '
            '(IFS-TOPSIS, IFS-VIKOR, IFS-PROMETHEE, IFS-COPRAS, IFS-EDAS, IFS-SAW). '
            'All methods are executed on the same normalised decision matrix and '
            'weight vector, ensuring comparability of rank orderings.'
        )
        lines.append('')

        _sec('10.3', 'Evidential Reasoning Aggregation', 2)
        lines.append('')
        _p(
            'The ER framework (Yang & Xu, 2002) transforms method outputs into '
            'basic probability assignments over a predefined evaluation grade '
            'set. A recursive analytical algorithm combines beliefs while '
            'preserving residual uncertainty. Stage 1 aggregates methods within '
            'each criterion; Stage 2 aggregates criteria into the final ranking.'
        )
        lines.append('')

        _sec('10.4', 'Intuitionistic Fuzzy Sets', 2)
        lines.append('')
        _p(
            'IFS (Atanassov, 1986) extends classical fuzzy sets with a '
            'non-membership function, introducing a hesitancy margin '
            'h = 1 - mu - nu that captures epistemic uncertainty. Membership '
            'is derived from normalised scores; non-membership from temporal '
            'variance; hesitancy absorbs residual ambiguity.'
        )
        lines.append('')

        _sec('10.5', 'Machine-Learning Forecasting', 2)
        lines.append('')
        _p(
            'The Super Learner (van der Laan, Polley & Hubbard, 2007) constructs '
            'an optimal convex combination of heterogeneous base learners '
            '(Gradient Boosting, Bayesian Ridge, Panel VAR, Quantile Random Forest, '
            'Hierarchical Bayes, Neural Additive Models) via cross-validated risk '
            'minimisation. Conformal Prediction (Vovk, Gammerman & Shafer, 2005) '
            'provides distribution-free prediction intervals with finite-sample '
            'coverage guarantees.'
        )
        lines.append('')

        # ================================================================
        # APPENDIX A: OUTPUT FILES
        # ================================================================
        _sec('A.', 'Output File Inventory', 1)
        lines.append('')
        csv_files = sorted(
            f for f in self._saved_files if f.endswith('.csv'))
        json_files = sorted(
            f for f in self._saved_files if f.endswith('.json'))

        if csv_files:
            lines.append('    CSV Result Files')
            lines.append('    ' + '-' * 20)
            for f in csv_files:
                lines.append(f'      {Path(f).name}')
            lines.append('')
        if json_files:
            lines.append('    JSON Metadata Files')
            lines.append('    ' + '-' * 22)
            for f in json_files:
                lines.append(f'      {Path(f).name}')
            lines.append('')
        if figure_paths:
            lines.append(f'    Figures ({len(figure_paths)} files)')
            lines.append('    ' + '-' * 20)
            for fp in sorted(figure_paths):
                lines.append(f'      {Path(fp).name}')
            lines.append('')

        # ── end ─────────────────────────────────────────────────
        lines.append('')
        lines.append('=' * W)
        lines.append('  END OF REPORT')
        lines.append('=' * W)

        report_text = '\n'.join(lines)

        report_path = self.reports_dir / 'report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        self._record(report_path)

        return report_text


# =========================================================================
# Factory
# =========================================================================

def create_output_manager(base_output_dir: str = 'outputs') -> OutputManager:
    """Create and return an OutputManager instance."""
    return OutputManager(base_output_dir=base_output_dir)
