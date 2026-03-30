"""
Centralized Output Orchestration.

This module provides the `OutputOrchestrator` class, which serves as the 
central hub for coordinating all data persistence and reporting tasks. 
It delegates specific writing responsibilities to `CsvWriter` (for 
structured data) and `ReportWriter` (for human-readable summaries).

Key Features
------------
- **Unified Entry Point**: Provides a single `save_all` method to persist 
  all pipeline artefacts synchronously.
- **Fail-Safe Operation**: Implements robust error handling to ensure 
  that failure in one output component (e.g., PDF generation) does not 
  block others.
- **Structured Storage**: Manages the directory hierarchy for weights, 
  rankings, forecasts, and analysis results.

Notes
-----
The orchestrator maintains internal state of all files saved during its 
lifecycle, accessible via `get_saved_files()`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .csv_writer import CsvWriter
from .report_writer import ReportWriter

logger = logging.getLogger('ml_mcdm')


class OutputOrchestrator:
    """
    Coordinator for multi-format pipeline output.

    Attributes
    ----------
    base_dir : str
        The root directory for all generated output files.
    csv : CsvWriter
        Writer instance for CSV-formatted datasets.
    report : ReportWriter
        Writer instance for Markdown and PDF reports.
    """

    def __init__(self, base_output_dir: str = 'output/result'):
        """
        Initialize the output orchestrator.

        Parameters
        ----------
        base_output_dir : str, default='output/result'
            The directory where all results will be stored. Validated 
            upon initialization.
        """
        from . import _sanitize_output_dir
        _sanitize_output_dir(base_output_dir)  # validate early
        self.base_dir = base_output_dir
        self.csv = CsvWriter(base_output_dir)
        self.report = ReportWriter(base_output_dir)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def save_all(
        self,
        panel_data: Any,
        weights: Dict[str, Any],
        ranking_result: Any,
        forecast_result: Optional[Any],
        analysis_results: Dict[str, Any],
        execution_time: float,
        figure_paths: Optional[List[str]] = None,
        config: Optional[Any] = None,
        multi_year_results: Optional[Dict[int, Any]] = None,
        weight_all_years: Optional[Dict[int, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Persist all pipeline artefacts and return a summary.

        Parameters
        ----------
        panel_data : PanelData
            The input panel data object used for labelling.
        weights : Dict[str, Any]
            Dictionary containing criteria weights and subcriteria metadata.
        ranking_result : RankingResult
            The results from the MCDM ranking engine.
        forecast_result : UnifiedForecastResult, optional
            The results from the ML forecasting engine.
        analysis_results : Dict[str, Any]
            Dictionary containing sensitivity and validation results.
        execution_time : float
            Total pipeline execution time in seconds.
        figure_paths : List[str], optional
            Paths to generated visualization plots.
        config : Config, optional
            The pipeline configuration object.
        multi_year_results : Dict[int, Any], optional
            MCDM scores mapped by year.
        weight_all_years : Dict[int, Any], optional
            CRITIC weights mapped by year.

        Returns
        -------
        Dict[str, Any]
            Summary dictionary containing 'saved_files', 'report_path', 
            and 'total' count.
        """
        subcriteria = weights['subcriteria']

        # 1. Weights (hybrid MC ensemble)
        self.csv.save_weights(weights, subcriteria)
        logger.info('Saved: weights_analysis.csv')

        # 2. Final Rankings (ER composite ranking across all criteria)
        if ranking_result is not None and ranking_result.final_ranking is not None:
            try:
                path_rank = self.csv.save_rankings(ranking_result, panel_data.provinces)
                logger.info(f'Saved: {Path(path_rank).name}')
            except Exception as _exc:
                logger.warning(f'save_rankings (final composite) failed: {_exc}')

        # 3. MCDM scores per criterion (long format, all years)
        if multi_year_results:
            saved_scores = self.csv.save_mcdm_scores_by_criterion(multi_year_results)
            logger.info(f'Saved: mcdm_scores_*.csv ({len(saved_scores)} files)')
        else:
            logger.info('Skipped: mcdm_scores_*.csv (no multi_year_results)')

        # 4. Rank comparison matrix — skipped (mcdm_rank_comparison.csv removed per spec)

        # 5. ER uncertainty — skipped (prediction_uncertainty_er.csv removed per spec)

        # 6. Forecasting results
        if forecast_result is not None:
            saved_fc = self.csv.save_forecast_results(forecast_result)
            for key, path in saved_fc.items():
                logger.info(f'Saved: {Path(path).name}')

        # 7. Sensitivity analysis
        if analysis_results.get('sensitivity'):
            saved_an = self.csv.save_analysis_results(analysis_results)
            for key, path in saved_an.items():
                logger.info(f'Saved: {Path(path).name}')

        # ── All-years outputs ─────────────────────────────────────────────

        # 8. All-years score / rank matrices — skipped (rankings_all_years.csv,
        #    ranks_all_years.csv, criterion_er_scores_all_years.csv removed per spec)

        # 9. Belief distributions (Stage-1 ER)
        try:
            path_bd = self.csv.save_belief_distributions(
                ranking_result, panel_data.provinces)
            if path_bd:
                logger.info(f'Saved: {Path(path_bd).name}')
        except Exception as _exc:
            logger.warning(f'save_belief_distributions failed: {_exc}')

        # 10. MCDM composite scores (all methods + ER, all years)
        if multi_year_results:
            try:
                path_mc = self.csv.save_mcdm_composite_scores_all_years(multi_year_results)
                if path_mc:
                    logger.info(f'Saved: {Path(path_mc).name}')
            except Exception as _exc:
                logger.warning(f'save_mcdm_composite_scores_all_years failed: {_exc}')

        # 11. Individual base-model predictions
        if forecast_result is not None:
            try:
                path_imp = self.csv.save_individual_model_predictions(forecast_result)
                if path_imp:
                    logger.info(f'Saved: {Path(path_imp).name}')
            except Exception as _exc:
                logger.warning(f'save_individual_model_predictions failed: {_exc}')

        # 12. Perturbation detail matrix
        if analysis_results.get('sensitivity'):
            try:
                path_pd = self.csv.save_perturbation_detail(analysis_results)
                if path_pd:
                    logger.info(f'Saved: {Path(path_pd).name}')
            except Exception as _exc:
                logger.warning(f'save_perturbation_detail failed: {_exc}')

        # 13. Per-year CRITIC weights (14 individual files + 3 summary matrices)
        if weight_all_years:
            try:
                saved_yw = self.csv.save_weights_all_years(weight_all_years)
                for key in saved_yw:
                    logger.info(f'Saved: {Path(saved_yw[key]).name}')
            except Exception as _exc:
                logger.warning(f'save_weights_all_years failed: {_exc}')

        # 14. Temporal stability analysis (window-based, per-year metrics)
        if hasattr(weights, 'temporal_stability') and weights.temporal_stability is not None:
            try:
                path_ts = self.csv.save_temporal_stability(weights.temporal_stability)
                if path_ts:
                    logger.info(f'Saved: {Path(path_ts).name}')
            except Exception as _exc:
                logger.warning(f'save_temporal_stability failed: {_exc}')

        # 15. Sensitivity analysis (three-tier perturbation results)
        if hasattr(weights, 'sensitivity_analysis') and weights.sensitivity_analysis is not None:
            try:
                path_sa = self.csv.save_sensitivity_analysis(weights.sensitivity_analysis)
                if path_sa:
                    logger.info(f'Saved: {Path(path_sa).name}')
            except Exception as _exc:
                logger.warning(f'save_sensitivity_analysis failed: {_exc}')

        # 16. Per-year per-method MCDM scores (14 long-format CSVs)
        if multi_year_results:
            logger.info(f'[DEBUG] Attempting to save ranking CSVs for {len(multi_year_results)} years')
            try:
                saved_ms = self.csv.save_method_scores_all_years(multi_year_results)
                if saved_ms:
                    logger.info(f'[DEBUG] save_method_scores_all_years returned {len(saved_ms)} files')
                    for key in saved_ms:
                        logger.info(f'Saved: {Path(saved_ms[key]).name}')
                else:
                    logger.warning('[DEBUG] save_method_scores_all_years returned empty dict (no ranking data)')
            except Exception as _exc:
                logger.error(f'[DEBUG] save_method_scores_all_years raised exception: {_exc}', exc_info=True)
                logger.warning(f'save_method_scores_all_years failed: {_exc}')
        else:
            logger.warning('[DEBUG] Skipped ranking CSVs: multi_year_results is empty or None')

        # 19. Markdown report
        try:
            self.report.build_report(
                panel_data=panel_data,
                weights=weights,
                ranking_result=ranking_result,
                forecast_result=forecast_result,
                analysis_results=analysis_results,
                execution_time=execution_time,
                figure_paths=figure_paths,
                saved_files=self.csv.get_saved_files(),
            )
            logger.info('Saved: report.md (comprehensive analysis report)')
        except Exception as exc:
            logger.warning(f'Report generation failed: {exc}')

        total = len(self.csv.get_saved_files()) + 1  # +1 for report
        logger.info(f'Total output files: {total}')
        logger.info(f'All results saved to {self.base_dir}')

        return {
            'saved_files': self.csv.get_saved_files(),
            'report_path': self.report.path,
            'total': total,
        }

    def get_saved_files(self) -> List[str]:
        """All files written so far including report."""
        files = self.csv.get_saved_files()
        if Path(self.report.path).exists():
            files.append(self.report.path)
        return files


__all__ = ['OutputOrchestrator']
