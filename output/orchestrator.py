# -*- coding: utf-8 -*-
"""
Output Orchestrator
===================

Central hub coordinating all output writers.  Replaces the former
``_save_all_results()`` method in ``pipeline.py`` by delegating to
``CsvWriter`` and ``ReportWriter``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .csv_writer import CsvWriter
from .report_writer import ReportWriter

logger = logging.getLogger('ml_mcdm')


class OutputOrchestrator:
    """Coordinate saving all results in one call."""

    def __init__(self, base_output_dir: str = 'result'):
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
    ) -> Dict[str, Any]:
        """
        Persist every artefact and return a summary dict.

        This replaces ``MLMCDMPipeline._save_all_results()``.
        """
        subcriteria = weights['subcriteria']

        # 1. Weights (hybrid MC ensemble)
        self.csv.save_weights(weights, subcriteria)
        logger.info('Saved: weights_analysis.csv')

        # 2. Rankings
        self.csv.save_rankings(ranking_result, panel_data.provinces)
        logger.info('Saved: final_rankings.csv')

        # 3. MCDM scores per criterion
        saved_scores = self.csv.save_mcdm_scores_by_criterion(
            ranking_result, panel_data.provinces,
        )
        logger.info(f'Saved: mcdm_scores_*.csv ({len(saved_scores)} files)')

        # 4. Rank comparison matrix
        self.csv.save_rank_comparison(ranking_result, panel_data.provinces)
        logger.info('Saved: mcdm_rank_comparison.csv')

        # 5. Criterion weights
        self.csv.save_criterion_weights(ranking_result.criterion_weights_used)
        logger.info('Saved: criterion_weights.csv')

        # 6. ER uncertainty
        self.csv.save_er_uncertainty(ranking_result, panel_data.provinces)
        logger.info('Saved: prediction_uncertainty_er.csv')

        # 7. Data summary
        self.csv.save_data_summary(panel_data)
        logger.info('Saved: data_summary_statistics.csv')

        # 8. Forecasting results
        if forecast_result is not None:
            saved_fc = self.csv.save_forecast_results(forecast_result)
            for key, path in saved_fc.items():
                logger.info(f'Saved: {Path(path).name}')

        # 9. Sensitivity analysis
        if analysis_results.get('sensitivity'):
            saved_an = self.csv.save_analysis_results(analysis_results)
            for key, path in saved_an.items():
                logger.info(f'Saved: {Path(path).name}')

        # 10. Execution summary (JSON)
        self.csv.save_execution_summary(
            panel_data=panel_data,
            ranking_result=ranking_result,
            execution_time=execution_time,
        )
        logger.info('Saved: execution_summary.json')

        # 11. Config snapshot
        if config is not None:
            self.csv.save_config_snapshot(config)
            logger.info('Saved: config_snapshot.json')

        # ── New B-series outputs ──────────────────────────────────────────

        # 12. Per-method weight tables (Entropy / CRITIC / Hybrid)
        try:
            saved_mw = self.csv.save_method_weights(weights)
            for key in saved_mw:
                logger.info(f'Saved: {Path(saved_mw[key]).name}')
        except Exception as _exc:
            logger.warning(f'save_method_weights failed: {_exc}')

        # 13. MC province rank-uncertainty statistics
        try:
            path_mps = self.csv.save_mc_province_stats(weights)
            if path_mps:
                logger.info(f'Saved: {Path(path_mps).name}')
        except Exception as _exc:
            logger.warning(f'save_mc_province_stats failed: {_exc}')

        # 14. All-years score / rank matrices + criterion ER long-format
        if multi_year_results:
            try:
                saved_ay = self.csv.save_rankings_all_years(
                    multi_year_results, panel_data.provinces)
                for key in saved_ay:
                    logger.info(f'Saved: {Path(saved_ay[key]).name}')
            except Exception as _exc:
                logger.warning(f'save_rankings_all_years failed: {_exc}')

        # 15. Belief distributions (Stage-1 ER)
        try:
            path_bd = self.csv.save_belief_distributions(
                ranking_result, panel_data.provinces)
            if path_bd:
                logger.info(f'Saved: {Path(path_bd).name}')
        except Exception as _exc:
            logger.warning(f'save_belief_distributions failed: {_exc}')

        # 16. MCDM composite comparison (all methods + ER)
        try:
            path_mc = self.csv.save_mcdm_composite_comparison(
                ranking_result, panel_data.provinces)
            if path_mc:
                logger.info(f'Saved: {Path(path_mc).name}')
        except Exception as _exc:
            logger.warning(f'save_mcdm_composite_comparison failed: {_exc}')

        # 17. Individual base-model predictions
        if forecast_result is not None:
            try:
                path_imp = self.csv.save_individual_model_predictions(forecast_result)
                if path_imp:
                    logger.info(f'Saved: {Path(path_imp).name}')
            except Exception as _exc:
                logger.warning(f'save_individual_model_predictions failed: {_exc}')

        # 18. Perturbation detail matrix
        if analysis_results.get('sensitivity'):
            try:
                path_pd = self.csv.save_perturbation_detail(analysis_results)
                if path_pd:
                    logger.info(f'Saved: {Path(path_pd).name}')
            except Exception as _exc:
                logger.warning(f'save_perturbation_detail failed: {_exc}')

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
