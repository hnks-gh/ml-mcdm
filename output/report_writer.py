# -*- coding: utf-8 -*-
"""
Publication-Quality Markdown + LaTeX Report Writer
===================================================

Generates ``result/reports/report.md`` — a comprehensive analysis
report using proper Markdown headings, pipe tables, and LaTeX math
blocks for equations.  Renderable with any Markdown viewer or
convertible to PDF via Pandoc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

_logger = logging.getLogger(__name__)


class ReportWriter:
    """Build and save a publication-quality Markdown report."""

    def __init__(self, base_output_dir: str = 'result'):
        self.reports_dir = Path(base_output_dir) / 'reports'
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.reports_dir / 'report.md'

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build_report(
        self,
        panel_data: Any,
        weights: Dict[str, Any],
        ranking_result: Any,
        forecast_result: Optional[Any],
        analysis_results: Dict[str, Any],
        execution_time: float,
        figure_paths: Optional[List[str]] = None,
        saved_files: Optional[List[str]] = None,
    ) -> str:
        """Build the full report and write it to disk.

        Returns the report text.
        """
        L: List[str] = []  # accumulator

        # Derived arrays
        scores_arr = np.asarray(
            ranking_result.final_scores.values
            if hasattr(ranking_result.final_scores, 'values')
            else ranking_result.final_scores)
        ranks_arr = np.asarray(
            ranking_result.final_ranking.values
            if hasattr(ranking_result.final_ranking, 'values')
            else ranking_result.final_ranking, dtype=int)
        order = np.argsort(ranks_arr)
        # Use the ranked province list from the result itself so that
        # dynamically-excluded provinces are not present in output tables.
        _active_provs: List[str] = (
            list(ranking_result.final_scores.index)
            if hasattr(ranking_result.final_scores, 'index')
            else list(panel_data.provinces)
        )
        n_prov = len(_active_provs)
        n_years = len(panel_data.years)
        fused = np.asarray(weights['sc_array'])
        subcriteria = weights['subcriteria']
        sens = analysis_results.get('sensitivity')

        # ── Front Matter ─────────────────────────────────────────
        L.append('---')
        L.append('title: "Multi-Criteria Decision Analysis of Vietnamese Provincial Competitiveness"')
        L.append('subtitle: "A Traditional MCDM + Evidential Reasoning Approach with Machine-Learning Forecasting"')
        L.append(f'date: "{datetime.now().strftime("%Y-%m-%d")}"')
        L.append('---')
        L.append('')

        # ── Table of Contents ────────────────────────────────────
        L.append('## Table of Contents')
        L.append('')
        toc = [
            ('1', 'Executive Summary', 'executive-summary'),
            ('2', 'Data Description and Descriptive Statistics', 'data-description-and-descriptive-statistics'),
            ('3', 'Objective Weight Derivation', 'objective-weight-derivation'),
            ('4', 'Hierarchical Evidential Reasoning Ranking', 'hierarchical-evidential-reasoning-ranking'),
            ('5', 'Criterion-Level MCDM Evaluation', 'criterion-level-mcdm-evaluation'),
            ('6', 'Inter-Method Agreement and Concordance Analysis', 'inter-method-agreement-and-concordance-analysis'),
            ('7', 'Sensitivity and Robustness Analysis', 'sensitivity-and-robustness-analysis'),
            ('8', 'Machine-Learning Forecasting', 'machine-learning-forecasting'),
            ('9', 'Validity Assessment', 'validity-assessment'),
            ('10', 'Methodological Notes and References', 'methodological-notes-and-references'),
            ('A', 'Output File Inventory', 'output-file-inventory'),
        ]
        for num, title, anchor in toc:
            L.append(f'- [{num}. {title}](#{anchor})')
        L.append('')

        # ── Metadata ─────────────────────────────────────────────
        L.append(f'> **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  ')
        L.append(f'> **Runtime:** {execution_time:.2f} s  ')
        L.append(f'> **Framework:** ML-MCDM v4.0')
        L.append('')

        # ============================================================
        # 1. Executive Summary
        # ============================================================
        L.append('# 1. Executive Summary')
        L.append('')
        L.append(
            f'This report documents a comprehensive multi-criteria decision-making '
            f'(MCDM) evaluation of **{n_prov}** Vietnamese provinces over the period '
            f'**{min(panel_data.years)}–{max(panel_data.years)}** ({n_years} years). '
            f'The analytical framework integrates {panel_data.n_subcriteria} subcriteria '
            f'organised into {panel_data.n_criteria} criteria groups, evaluated through '
            f'{len(ranking_result.methods_used)} classical MCDM methods.'
        )
        L.append('')
        L.append(
            'Final provincial rankings are obtained via a two-stage Evidential Reasoning (ER) '
            'aggregation procedure that combines belief structures from all constituent methods '
            'while explicitly quantifying residual uncertainty.'
        )
        L.append('')

        L.append('> **Key Finding:** Top-ranked and bottom-ranked provinces:')
        L.append('')

        # Top 5
        L.append('**Table 1(a). Highest-ranked provinces.**')
        L.append('')
        L.append('| Rank | Province | ER Score |')
        L.append('| ---: | :--- | ---: |')
        for i in range(min(5, n_prov)):
            idx = order[i]
            L.append(f'| {i+1} | {_active_provs[idx]} | {scores_arr[idx]:.4f} |')
        L.append('')

        # Bottom 5
        L.append('**Table 1(b). Lowest-ranked provinces.**')
        L.append('')
        L.append('| Rank | Province | ER Score |')
        L.append('| ---: | :--- | ---: |')
        for i in range(min(5, n_prov)):
            idx = order[-(i + 1)]
            L.append(f'| {n_prov - i} | {_active_provs[idx]} | {scores_arr[idx]:.4f} |')
        L.append('')

        L.append(f"- **Kendall's $W$ (concordance):** {ranking_result.kendall_w:.4f}")
        if sens and hasattr(sens, 'overall_robustness'):
            L.append(f'- **Overall Robustness Index:** {sens.overall_robustness:.4f}')
            L.append(f'- **Confidence Level:** {getattr(sens, "confidence_level", 0.95):.0%}')
        L.append('')

        # ============================================================
        # 2. Data Description
        # ============================================================
        L.append('# 2. Data Description and Descriptive Statistics')
        L.append('')
        L.append(
            f'The dataset comprises a balanced panel of {n_prov} provinces observed '
            f'annually from {min(panel_data.years)} to {max(panel_data.years)}, '
            f'yielding {n_prov * n_years:,} province-year observations.'
        )
        L.append('')
        L.append(f'| Parameter | Value |')
        L.append(f'| :--- | ---: |')
        L.append(f'| Provinces ($N$) | {n_prov} |')
        L.append(f'| Temporal span | {min(panel_data.years)}–{max(panel_data.years)} |')
        L.append(f'| Annual periods ($T$) | {n_years} |')
        L.append(f'| Criteria | {panel_data.n_criteria} |')
        L.append(f'| Subcriteria | {panel_data.n_subcriteria} |')
        L.append(f'| Total observations ($N \\times T$) | {n_prov * n_years:,} |')
        L.append('')

        # Descriptive stats
        try:
            latest_year = max(panel_data.years)
            desc = panel_data.subcriteria_cross_section[latest_year].describe()
            L.append(f'**Table 2. Descriptive statistics ({latest_year} cross-section).**')
            L.append('')
            L.append('| Subcriteria | Mean | Std Dev | Min | Max | CV |')
            L.append('| :--- | ---: | ---: | ---: | ---: | ---: |')
            for sc in subcriteria:
                if sc in desc.columns:
                    mu = desc[sc]['mean']
                    sd = desc[sc]['std']
                    mn = desc[sc]['min']
                    mx = desc[sc]['max']
                    cv = sd / mu if mu != 0 else 0.0
                    L.append(f'| {sc} | {mu:.4f} | {sd:.4f} | {mn:.4f} | {mx:.4f} | {cv:.4f} |')
            L.append('')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # ============================================================
        # 3. Weight Derivation
        # ============================================================
        L.append('# 3. Objective Weight Derivation')
        L.append('')
        L.append(
            'Subcriteria weights are derived through a two-level hierarchical Monte Carlo '
            'ensemble that blends Shannon Entropy and CRITIC via a Beta-distributed mixing '
            'coefficient.  Level 1 produces local SC weights within each criterion group; '
            'Level 2 determines criterion-level weights from a composite matrix.  '
            'Global SC weights are the product of local and criterion weights, re-normalised '
            'to the simplex.'
        )
        L.append('')
        L.append('The global weight of subcriteria $j$ in criterion group $C_k$ is:')
        L.append('')
        L.append(
            r'$$w_j = \frac{u_j^{(k)} \cdot v_k}{'
            r'\sum_{k^{\prime}} \sum_{j^{\prime} \in C_{k^{\prime}}} u_{j^{\prime}}^{(k^{\prime})} v_{k^{\prime}}}$$'
        )
        L.append('')
        L.append('where $u_j^{(k)}$ is the Level-1 local SC weight and $v_k$ is the Level-2 criterion weight.')
        L.append('')

        # Hybrid weighting table — global SC weights with MC diagnostics
        details = weights.get('details', {})
        l1_diag = details.get('level1', {})
        crit_w  = weights.get('criterion_weights', {})
        L.append('**Table 3. Subcriteria global weights (Hybrid MC Ensemble).**')
        L.append('')
        L.append('| Subcriteria | Criterion | Criterion Weight | Local Weight | Global Weight | MC Std | MC CV |')
        L.append('| :--- | :--- | ---: | ---: | ---: | ---: | ---: |')
        sc_to_crit = {}
        for cid, cdata in l1_diag.items():
            for sc in cdata.get('local_sc_weights', {}):
                sc_to_crit[sc] = cid
        for sc in subcriteria:
            cid = sc_to_crit.get(sc, '')
            local_w   = l1_diag.get(cid, {}).get('local_sc_weights', {}).get(sc, 0.0)
            mc_diag   = l1_diag.get(cid, {}).get('mc_diagnostics', {})
            mc_std    = mc_diag.get('std_weights', {}).get(sc, 0.0)
            mc_cv     = mc_diag.get('cv_weights',  {}).get(sc, 0.0)
            gw        = weights['global_sc_weights'].get(sc, 0.0)
            v_k       = crit_w.get(cid, 0.0)
            L.append(
                f'| {sc} | {cid} | {v_k:.4f} | {local_w:.4f} | {gw:.4f} | {mc_std:.4f} | {mc_cv:.4f} |'
            )
        L.append('')

        L.append(f'- **Sum of global weights:** {fused.sum():.6f}')
        L.append(f'- **Max weight:** {fused.max():.6f} ({subcriteria[np.argmax(fused)]})')
        L.append(f'- **Min weight:** {fused.min():.6f} ({subcriteria[np.argmin(fused)]})')
        L.append(f'- **Shannon entropy $H(\\mathbf{{{{w}}}})$:** {-np.sum(fused * np.log(fused + 1e-12)):.4f}')
        L.append('')

        # Level 2 MC diagnostics summary
        l2_diag = details.get('level2', {}).get('mc_diagnostics', {})
        if l2_diag:
            L.append('**Table 4. Level-2 criterion weights (MC diagnostics).**')
            L.append('')
            L.append('| Criterion | Weight | MC Mean | MC Std | 95% CI Lower | 95% CI Upper |')
            L.append('| :--- | ---: | ---: | ---: | ---: | ---: |')
            for cid, v_k in sorted(crit_w.items()):
                mc_mean = l2_diag.get('mean_weights',   {}).get(cid, v_k)
                mc_std  = l2_diag.get('std_weights',    {}).get(cid, 0.0)
                ci_lo   = l2_diag.get('ci_lower_2_5',  {}).get(cid, 0.0)
                ci_hi   = l2_diag.get('ci_upper_97_5', {}).get(cid, 0.0)
                L.append(f'| {cid} | {v_k:.4f} | {mc_mean:.4f} | {mc_std:.4f} | {ci_lo:.4f} | {ci_hi:.4f} |')
            L.append('')
            tau = l2_diag.get('avg_kendall_tau', 0)
            kw  = l2_diag.get('kendall_w', 0)
            L.append(f'- **Level-2 Avg Kendall τ:** {tau:.4f}')
            L.append(f"- **Level-2 Kendall's W:** {kw:.4f}")
            L.append('')

        # ============================================================
        # 4. ER Ranking
        # ============================================================
        L.append('# 4. Hierarchical Evidential Reasoning Ranking')
        L.append('')
        L.append(
            'The ER approach (Yang & Xu, 2002) aggregates MCDM scores into '
            'belief structures.  The recursive ER algorithm for combining '
            'two evidence bodies is:'
        )
        L.append('')
        L.append('$$m_{1 \\oplus 2}(H_n) = \\frac{m_1(H_n) m_2(\\Theta) + m_2(H_n) m_1(\\Theta) + m_1(H_n) m_2(H_n)}{1 - K}$$')
        L.append('')
        L.append('where $K = \\sum_{H_i \\cap H_j = \\varnothing} m_1(H_i) m_2(H_j)$ is the conflict factor.')
        L.append('')

        L.append(f'- **Aggregation:** Evidential Reasoning (Yang & Xu, 2002)')
        L.append(f'- **MCDM Methods:** {len(ranking_result.methods_used)}')
        L.append(f"- **Kendall's $W$:** {ranking_result.kendall_w:.4f}")
        L.append(f'- **Target Year:** {ranking_result.target_year}')
        L.append('')

        # Full ranking table
        mean_s = scores_arr.mean()
        std_s = scores_arr.std() if scores_arr.std() > 0 else 1.0
        L.append('**Table 5. Complete provincial ranking by ER composite score.**')
        L.append('')
        L.append('| Rank | Province | ER Score | $z$-Score | Quartile |')
        L.append('| ---: | :--- | ---: | ---: | :---: |')
        for idx in order:
            r = ranks_arr[idx]
            s = scores_arr[idx]
            z = (s - mean_s) / std_s
            pct = (n_prov - r + 1) / n_prov * 100
            q = 'Q1' if pct >= 75 else ('Q2' if pct >= 50 else ('Q3' if pct >= 25 else 'Q4'))
            L.append(f'| {r} | {_active_provs[idx]} | {s:.4f} | {z:+.3f} | {q} |')
        L.append('')

        # Score distribution
        L.append('### Distributional Properties')
        L.append('')
        iqr = np.percentile(scores_arr, 75) - np.percentile(scores_arr, 25)
        L.append(f'| Statistic | Value |')
        L.append(f'| :--- | ---: |')
        L.append(f'| Mean | {scores_arr.mean():.4f} |')
        L.append(f'| Median | {np.median(scores_arr):.4f} |')
        L.append(f'| Std Dev | {scores_arr.std():.4f} |')
        L.append(f'| Skewness | {pd.Series(scores_arr).skew():.4f} |')
        L.append(f'| Excess Kurtosis | {pd.Series(scores_arr).kurtosis():.4f} |')
        L.append(f'| IQR | {iqr:.4f} |')
        L.append('')

        # ER Uncertainty
        try:
            unc = ranking_result.er_result.uncertainty
            L.append('### Evidential Reasoning Uncertainty')
            L.append('')
            L.append(f'- **Mean Belief Entropy:** {unc["belief_entropy"].mean():.4f} '
                     f'(SD = {unc["belief_entropy"].std():.4f})')
            L.append(f'- **Mean Utility Interval Width:** '
                     f'{unc["utility_interval_width"].mean():.4f} '
                     f'(SD = {unc["utility_interval_width"].std():.4f})')
            L.append('')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # ============================================================
        # 5. Criterion-Level
        # ============================================================
        L.append('# 5. Criterion-Level MCDM Evaluation')
        L.append('')
        L.append(
            f'Each of the {panel_data.n_criteria} criteria groups is independently '
            f'evaluated by {len(ranking_result.methods_used)} MCDM methods.'
        )
        L.append('')

        # Criterion weights
        crit_w = ranking_result.criterion_weights_used
        L.append('**Table 6. Criterion weights (Stage 2 ER).**')
        L.append('')
        L.append('| Criterion | Weight |')
        L.append('| :--- | ---: |')
        for c, w in sorted(crit_w.items()):
            L.append(f'| {c} | {w:.6f} |')
        L.append('')

        # Top 3 per criterion
        for crit_id in sorted(ranking_result.criterion_method_scores.keys()):
            method_scores = ranking_result.criterion_method_scores[crit_id]
            all_sc = [
                (s.values if hasattr(s, 'values') else np.asarray(s))
                for s in method_scores.values()
            ]
            avg = np.mean(all_sc, axis=0)
            top3 = np.argsort(avg)[-3:][::-1]
            # Resolve province names from the criterion's own series index
            _crit_provs: List[str] = (
                list(list(method_scores.values())[0].index)
                if method_scores and hasattr(list(method_scores.values())[0], 'index')
                else _active_provs
            )
            L.append(f'**{crit_id}** — top 3: '
                     + ', '.join(f'{_crit_provs[i]} ({avg[i]:.4f})' for i in top3))
        L.append('')

        # ============================================================
        # 6. Concordance
        # ============================================================
        L.append('# 6. Inter-Method Agreement and Concordance Analysis')
        L.append('')
        strength = ('strong' if ranking_result.kendall_w >= 0.7
                    else 'moderate' if ranking_result.kendall_w >= 0.5
                    else 'fair' if ranking_result.kendall_w >= 0.3
                    else 'weak')
        L.append(
            f"Kendall's coefficient of concordance $W = {ranking_result.kendall_w:.4f}$ "
            f'indicates **{strength}** agreement among the {len(ranking_result.methods_used)} methods.'
        )
        L.append('')

        # Top-5 frequency
        try:
            from collections import Counter
            top5_sets = []
            for crit_id, method_ranks in ranking_result.criterion_method_ranks.items():
                for method, rank_series in method_ranks.items():
                    vals = (rank_series.values if hasattr(rank_series, 'values')
                            else np.asarray(rank_series))
                    top5_sets.append(set(np.argsort(vals)[:5]))
            all_top5 = Counter()
            for s in top5_sets:
                for idx in s:
                    all_top5[idx] += 1
            total = len(top5_sets)

            L.append('**Table 7. Provinces most frequently ranked in the top 5.**')
            L.append('')
            L.append('| Province | Count | Frequency |')
            L.append('| :--- | ---: | ---: |')
            for idx, count in all_top5.most_common(10):
                L.append(f'| {_active_provs[idx]} | {count} | {count/total:.1%} |')
            L.append('')
        except Exception as _exc:
            _logger.debug('section skipped: %s', _exc)

        # ============================================================
        # 7. Sensitivity
        # ============================================================
        L.append('# 7. Sensitivity and Robustness Analysis')
        L.append('')
        if sens:
            L.append(f'- **Overall Robustness Index:** {sens.overall_robustness:.4f}')
            L.append(f'- **Confidence Level:** {getattr(sens, "confidence_level", 0.95):.0%}')
            L.append('')

            # 7.1 Criteria
            if hasattr(sens, 'criteria_sensitivity') and sens.criteria_sensitivity:
                L.append('## 7.1 Criteria Weight Sensitivity')
                L.append('')
                L.append('**Table 8. Criteria weight sensitivity indices.**')
                L.append('')
                L.append('| Criterion | Sensitivity | Classification |')
                L.append('| :--- | ---: | :---: |')
                for k, v in sorted(sens.criteria_sensitivity.items(),
                                   key=lambda x: x[1], reverse=True):
                    cls_ = 'High' if v > 0.1 else ('Medium' if v > 0.05 else 'Low')
                    L.append(f'| {k} | {v:.4f} | {cls_} |')
                L.append('')

            # 7.2 Subcriteria
            if hasattr(sens, 'subcriteria_sensitivity') and sens.subcriteria_sensitivity:
                L.append('## 7.2 Subcriteria Weight Sensitivity (Top 15)')
                L.append('')
                L.append('**Table 9. Most influential subcriteria.**')
                L.append('')
                L.append('| Subcriteria | Sensitivity |')
                L.append('| :--- | ---: |')
                items = sorted(sens.subcriteria_sensitivity.items(),
                               key=lambda x: x[1], reverse=True)[:15]
                for k, v in items:
                    L.append(f'| {k} | {v:.4f} |')
                L.append('')

            # 7.3 Top-N
            if hasattr(sens, 'top_n_stability') and sens.top_n_stability:
                L.append('## 7.3 Top-N Ranking Stability')
                L.append('')
                L.append('**Table 10. Top-N set stability under weight perturbation.**')
                L.append('')
                L.append('| Tier | Index | Percentage |')
                L.append('| :--- | ---: | ---: |')
                for n, stab in sorted(sens.top_n_stability.items()):
                    L.append(f'| Top-{n} | {stab:.4f} | {stab:.1%} |')
                L.append('')

            # 7.4 Temporal
            if hasattr(sens, 'temporal_stability') and sens.temporal_stability:
                L.append('## 7.4 Temporal Rank Stability')
                L.append('')
                L.append('**Table 11. Year-to-year rank correlation.**')
                L.append('')
                L.append('| Year Pair | Spearman $\\rho$ | Strength |')
                L.append('| :--- | ---: | :---: |')
                for pair, corr in sorted(sens.temporal_stability.items()):
                    s_ = 'Strong' if corr > 0.8 else ('Moderate' if corr > 0.5 else 'Weak')
                    L.append(f'| {pair} | {corr:.4f} | {s_} |')
                L.append('')

            # 7.5 Province stability
            if hasattr(sens, 'rank_stability') and sens.rank_stability:
                L.append('## 7.5 Provincial Rank Stability')
                L.append('')
                sorted_stab = sorted(sens.rank_stability.items(), key=lambda x: x[1])

                L.append('**Table 12(a). Ten most volatile provinces.**')
                L.append('')
                L.append('| Province | Stability |')
                L.append('| :--- | ---: |')
                for p, s in sorted_stab[:10]:
                    L.append(f'| {p} | {s:.4f} |')
                L.append('')

                L.append('**Table 12(b). Ten most stable provinces.**')
                L.append('')
                L.append('| Province | Stability |')
                L.append('| :--- | ---: |')
                for p, s in sorted_stab[-10:]:
                    L.append(f'| {p} | {s:.4f} |')
                L.append('')
        else:
            L.append('*Sensitivity analysis was not executed in this run.*')
            L.append('')

        # ============================================================
        # 8. ML Forecasting
        # ============================================================
        L.append('# 8. Machine-Learning Forecasting')
        L.append('')
        if forecast_result is not None:
            L.append(
                'A Super Learner meta-ensemble (van der Laan et al., 2007) forecasts '
                'provincial scores one period ahead.  Prediction intervals are obtained '
                'through Conformal Prediction (Vovk et al., 2005).'
            )
            L.append('')

            # 8.1 Model contributions
            if hasattr(forecast_result, 'model_contributions') and forecast_result.model_contributions:
                L.append('## 8.1 Super Learner Model Contributions')
                L.append('')
                L.append('**Table 13. Base-learner weights.**')
                L.append('')
                L.append('| Model | Weight | Contribution |')
                L.append('| :--- | ---: | ---: |')
                for m, w in sorted(forecast_result.model_contributions.items(),
                                   key=lambda x: x[1], reverse=True):
                    L.append(f'| {m} | {w:.4f} | {w*100:.1f}% |')
                L.append('')

            # 8.2 Model performance
            if hasattr(forecast_result, 'model_performance') and forecast_result.model_performance:
                L.append('## 8.2 Individual Model Performance')
                L.append('')
                perf = forecast_result.model_performance
                all_metrics = sorted({m for d in perf.values() for m in d})
                L.append('**Table 14. Out-of-sample performance metrics.**')
                L.append('')
                hdr = '| Model | ' + ' | '.join(m.upper() for m in all_metrics) + ' |'
                sep = '| :--- |' + ' ---: |' * len(all_metrics)
                L.append(hdr)
                L.append(sep)
                for model in sorted(perf.keys()):
                    row = ' | '.join(f'{perf[model].get(m, 0):.4f}' for m in all_metrics)
                    L.append(f'| {model} | {row} |')
                L.append('')

            # 8.3 CV
            if hasattr(forecast_result, 'cross_validation_scores') and forecast_result.cross_validation_scores:
                L.append('## 8.3 Cross-Validation Results')
                L.append('')
                L.append('**Table 15. K-fold CV summary ($R^2$).**')
                L.append('')
                L.append('| Model | Mean | Std Dev | Min | Max |')
                L.append('| :--- | ---: | ---: | ---: | ---: |')
                for model, sc in sorted(forecast_result.cross_validation_scores.items()):
                    a = np.asarray(sc)
                    L.append(f'| {model} | {a.mean():.4f} | {a.std():.4f} '
                             f'| {a.min():.4f} | {a.max():.4f} |')
                L.append('')

            # 8.4 Holdout
            if hasattr(forecast_result, 'holdout_performance') and forecast_result.holdout_performance:
                L.append('## 8.4 Holdout Validation')
                L.append('')
                for metric, val in forecast_result.holdout_performance.items():
                    L.append(f'- **{metric}:** {val:.4f}')
                L.append('')

            # 8.5 Feature importance
            if (hasattr(forecast_result, 'feature_importance')
                    and forecast_result.feature_importance is not None):
                imp = forecast_result.feature_importance
                if not imp.empty and 'Importance' in imp.columns:
                    L.append('## 8.5 Feature Importance (Top 20)')
                    L.append('')
                    L.append('**Table 16. Feature importance ranking.**')
                    L.append('')
                    L.append('| Rank | Feature | Importance | Cumulative |')
                    L.append('| ---: | :--- | ---: | ---: |')
                    imp_s = imp.sort_values('Importance', ascending=False).head(20)
                    cum = 0.0
                    for rank, (feat, row) in enumerate(imp_s.iterrows(), 1):
                        cum += row['Importance']
                        L.append(f'| {rank} | {feat} | {row["Importance"]:.4f} | {cum:.4f} |')
                    L.append('')

            # 8.6 Prediction intervals
            if (hasattr(forecast_result, 'prediction_intervals')
                    and forecast_result.prediction_intervals):
                intervals = forecast_result.prediction_intervals
                lower = intervals.get('lower')
                upper = intervals.get('upper')
                if lower is not None and upper is not None:
                    L.append('## 8.6 Conformal Prediction Interval Diagnostics')
                    L.append('')
                    widths = (upper.values - lower.values).flatten()
                    L.append(f'- **Nominal Coverage:** 95%')
                    L.append(f'- **Mean Width:** {widths.mean():.4f}')
                    L.append(f'- **Median Width:** {np.median(widths):.4f}')
                    L.append(f'- **Range:** [{widths.min():.4f}, {widths.max():.4f}]')
                    L.append('')
        else:
            L.append('*Forecasting module was not executed.*')
            L.append('')

        # ============================================================
        # 9. Validity
        # ============================================================
        L.append('# 9. Validity Assessment')
        L.append('')
        checks = [
            ('Weight normalisation ($\\sum w = 1$)', abs(fused.sum() - 1.0) < 0.01),
            ('Rank completeness (all provinces ranked)', len(set(ranks_arr)) == n_prov),
            ('Score range ($0 \\le s \\le 1$)',
             scores_arr.min() >= 0 and scores_arr.max() <= 1),
            ("Kendall's $W \\ge 0.5$", ranking_result.kendall_w >= 0.5),
        ]
        if sens and hasattr(sens, 'overall_robustness'):
            checks.append(('Robustness $\\ge 0.7$', sens.overall_robustness >= 0.7))

        L.append('**Table 17. Internal validity diagnostics.**')
        L.append('')
        L.append('| Criterion | Result |')
        L.append('| :--- | :---: |')
        for desc, passed in checks:
            L.append(f'| {desc} | {"PASS" if passed else "FAIL"} |')
        L.append('')

        # ============================================================
        # 10. Methodological Notes
        # ============================================================
        L.append('# 10. Methodological Notes and References')
        L.append('')
        L.append('## 10.1 Objective Weighting Methods')
        L.append('')
        notes = [
            ('**Entropy**', 'Shannon (1948). Information-theoretic derivation exploiting '
             'probability-distribution diversity across alternatives.'),
            ('**CRITIC**', 'Diakoulaki, Mavrotas & Papayannakis (1995). Weights based on '
             'contrast intensity and conflicting inter-criteria correlation.'),
            ('**MEREC**', 'Keshavarz-Ghorabaee et al. (2021). Method based on the Effect '
             'of Removal of a Criterion via logarithmic removal ratios.'),
            ('**Standard Deviation**', 'Dispersion-based weighting assigning higher weight '
             'to criteria exhibiting greater variability.'),
            ('**Fusion**', 'Reliability-weighted Bayesian bootstrap combination accounting '
             'for inter-method agreement at the subcriteria level.'),
        ]
        for name, desc in notes:
            L.append(f'- {name} — {desc}')
        L.append('')

        L.append('## 10.2 MCDM Methods')
        L.append('')
        L.append(
            'Six classical methods: TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, SAW. '
            'All share the same normalized decision matrix and weight vector.'
        )
        L.append('')

        L.append('## 10.3 Evidential Reasoning')
        L.append('')
        L.append(
            'The ER framework (Yang & Xu, 2002) transforms method outputs into basic '
            'probability assignments over an evaluation grade set. Stage 1 aggregates '
            'methods within each criterion; Stage 2 aggregates criteria into the final score.'
        )
        L.append('')

        L.append('## 10.4 Machine-Learning Forecasting')
        L.append('')
        L.append(
            'Super Learner (van der Laan, Polley & Hubbard, 2007) constructs an optimal '
            'convex combination of heterogeneous base learners. Conformal Prediction '
            '(Vovk, Gammerman & Shafer, 2005) provides distribution-free intervals.'
        )
        L.append('')

        # ============================================================
        # Appendix A
        # ============================================================
        L.append('# Appendix A — Output File Inventory')
        L.append('')
        all_files = sorted(saved_files or [])
        csv_files = [f for f in all_files if f.endswith('.csv')]
        json_files = [f for f in all_files if f.endswith('.json')]

        if csv_files:
            L.append('### CSV Result Files')
            L.append('')
            for f in csv_files:
                L.append(f'- `{Path(f).name}`')
            L.append('')
        if json_files:
            L.append('### JSON Metadata Files')
            L.append('')
            for f in json_files:
                L.append(f'- `{Path(f).name}`')
            L.append('')
        if figure_paths:
            L.append(f'### Figures ({len(figure_paths)} files)')
            L.append('')
            for fp in sorted(figure_paths):
                L.append(f'- `{Path(fp).name}`')
            L.append('')

        L.append('---')
        L.append('*End of report.*')

        # ── Write ────────────────────────────────────────────────
        report_text = '\n'.join(L)
        with open(self._path, 'w', encoding='utf-8') as fh:
            fh.write(report_text)
        return report_text

    @property
    def path(self) -> str:
        return str(self._path)


__all__ = ['ReportWriter']
