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

    def __init__(self, base_output_dir: str = 'outputs'):
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
    ) -> Dict[str, Any]:
        """
        Persist every artefact and return a summary dict.

        This replaces ``MLMCDMPipeline._save_all_results()``.
        """
        subcriteria = weights['subcriteria']

        # 1. Weights
        self.csv.save_weights(
            {k: weights[k] for k in ('entropy', 'critic', 'merec', 'std_dev', 'fused')},
            subcriteria,
        )
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

        # 12. Markdown report
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
