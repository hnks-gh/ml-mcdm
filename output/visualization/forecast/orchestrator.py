# -*- coding: utf-8 -*-
"""Forecast Visualization Orchestrator

Coordinates the end-to-end visualization pipeline:
1. Adapts UnifiedForecastResult → ForecastVizPayload
2. Validates payload against figure requirements
3. Executes figures in dependency order via manifest
4. Returns structured result metadata
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

from output.visualization.forecast.contracts import ForecastVizPayload
from output.visualization.forecast.adapters import UnifiedResultAdapter
from output.visualization.forecast.io_manifest import (
    get_manifest, FigureCategory, FigureSpec,
)
from output.visualization.forecast.charts import (
    AccuracyCharts,
    EnsembleCharts,
    UncertaintyCharts,
    InterpretabilityCharts,
    ImpactCharts,
    DiversityCharts,
    TemporalCharts,
)

logger = logging.getLogger(__name__)


class ForecastVisualizationOrchestrator:
    """Orchestrate end-to-end forecast visualization generation.
    
    Responsibilities:
    - Transform UnifiedForecastResult → ForecastVizPayload
    - Load manifest and determine executable figures
    - Delegate to appropriate chart modules
    - Handle errors with graceful fallback
    - Return structured execution report
    """
    
    def __init__(self, output_dir: str = 'output/visualization/forecast/result'):
        """Initialize orchestrator with output directory.
        
        Args:
            output_dir: Directory for all generated figures.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load manifest
        self.manifest = get_manifest()
        
        # Initialize chart modules
        self.charts = {
            'accuracy': AccuracyCharts(output_dir=str(self.output_dir)),
            'ensemble': EnsembleCharts(output_dir=str(self.output_dir)),
            'uncertainty': UncertaintyCharts(output_dir=str(self.output_dir)),
            'interpretability': InterpretabilityCharts(output_dir=str(self.output_dir)),
            'impact': ImpactCharts(output_dir=str(self.output_dir)),
            'diversity': DiversityCharts(output_dir=str(self.output_dir)),
            'temporal': TemporalCharts(output_dir=str(self.output_dir)),
        }
        
        logger.info(f'Orchestrator initialized with output_dir={self.output_dir}')
    
    def generate_from_unified_result(
        self,
        forecast_result: Any,
        panel_data: Optional[Any] = None,
        ranking_result: Optional[Any] = None,
        essential_only: bool = False,
        advanced_only: bool = False,
    ) -> 'ExecutionReport':
        """Generate all applicable figures from UnifiedForecastResult.
        
        Args:
            forecast_result: UnifiedForecastResult from forecasting pipeline.
            panel_data: Optional YearPanel for temporal context.
            ranking_result: Optional ranking result for current scores.
            essential_only: Generate only essential figure suite.
            advanced_only: Generate only advanced/optional figures.
            
        Returns:
            ExecutionReport with status per figure.
        """
        logger.info('Starting forecast visualization generation from UnifiedForecastResult...')
        
        # Step 1: Adapt data
        try:
            payload = UnifiedResultAdapter.adapt(
                forecast_result=forecast_result,
                panel_data=panel_data,
                ranking_result=ranking_result,
            )
            logger.info(f'Payload adapted: {payload}')
        except Exception as e:
            logger.error(f'Failed to adapt UnifiedForecastResult: {e}', exc_info=True)
            return ExecutionReport(
                success=False,
                error=f'Adaptation failed: {e}',
                figures={},
            )
        
        # Step 2: Generate figures
        return self.generate_from_payload(
            payload=payload,
            essential_only=essential_only,
            advanced_only=advanced_only,
        )
    
    def generate_from_payload(
        self,
        payload: ForecastVizPayload,
        essential_only: bool = False,
        advanced_only: bool = False,
    ) -> 'ExecutionReport':
        """Generate all applicable figures from ForecastVizPayload.
        
        Args:
            payload: ForecastVizPayload with typed fields.
            essential_only: Generate only essential figure suite.
            advanced_only: Generate only advanced/optional figures.
            
        Returns:
            ExecutionReport with status per figure.
        """
        logger.info('Starting figure generation from ForecastVizPayload...')
        
        # Determine which figures to execute
        all_figures = self.manifest.get_all_figures()
        
        if essential_only:
            target_figures = self.manifest.get_essential_figures()
            logger.info('Generating essential figure suite only')
        elif advanced_only:
            target_figures = self.manifest.get_optional_figures()
            logger.info('Generating advanced/optional figures only')
        else:
            target_figures = all_figures
            logger.info('Generating complete figure suite')
        
        # Execute figures in manifest order
        report = ExecutionReport()
        
        for figure_spec in self.manifest.get_execution_order():
            # Skip if not in target set
            if figure_spec not in target_figures:
                continue
            
            # Check if figure can execute
            can_execute, reason = figure_spec.can_execute(payload)
            if not can_execute:
                report.mark_skipped(
                    figure_spec.figure_id,
                    reason=reason,
                    title=figure_spec.title,
                )
                logger.debug(f'{figure_spec.figure_id} skipped: {reason}')
                continue
            
            # Get chart module
            chart_module = self.charts.get(figure_spec.module)
            if chart_module is None:
                report.mark_failed(
                    figure_spec.figure_id,
                    error=f'Chart module not found: {figure_spec.module}',
                    title=figure_spec.title,
                )
                logger.warning(f'Chart module not found: {figure_spec.module}')
                continue
            
            # Execute figure
            try:
                logger.info(f'Generating {figure_spec.figure_id}: {figure_spec.title}')
                method = getattr(chart_module, figure_spec.method_name, None)
                if method is None:
                    raise AttributeError(f'Method not found: {figure_spec.method_name}')
                
                output_path = method(
                    payload,
                    save_name=figure_spec.save_name,
                )
                
                if output_path:
                    report.mark_success(
                        figure_spec.figure_id,
                        output_path=output_path,
                        title=figure_spec.title,
                    )
                    logger.info(f'{figure_spec.figure_id} generated: {output_path}')
                else:
                    report.mark_skipped(
                        figure_spec.figure_id,
                        reason='METHOD_RETURNED_NONE',
                        title=figure_spec.title,
                    )
                    logger.debug(f'{figure_spec.figure_id} returned None (matplotlib unavailable?)')
            
            except Exception as e:
                report.mark_failed(
                    figure_spec.figure_id,
                    error=str(e),
                    title=figure_spec.title,
                )
                logger.warning(f'{figure_spec.figure_id} failed: {e}', exc_info=True)
        
        # Summarize
        logger.info(f'Figure generation complete: '
                   f'{report.num_success} success, '
                   f'{report.num_failed} failed, '
                   f'{report.num_skipped} skipped')
        
        return report


class FigureResult:
    """Result metadata for a single figure."""
    
    def __init__(
        self,
        figure_id: str,
        title: str,
        status: str,  # 'success', 'failed', 'skipped'
        output_path: Optional[str] = None,
        error: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        self.figure_id = figure_id
        self.title = title
        self.status = status
        self.output_path = output_path
        self.error = error
        self.reason = reason
    
    def is_success(self) -> bool:
        return self.status == 'success'
    
    def is_failed(self) -> bool:
        return self.status == 'failed'
    
    def is_skipped(self) -> bool:
        return self.status == 'skipped'


class ExecutionReport:
    """Report of figure generation execution."""
    
    def __init__(self):
        self.figures: Dict[str, FigureResult] = {}
        self.success = True
        self.error: Optional[str] = None
    
    def mark_success(self, figure_id: str, output_path: str, title: str):
        """Record successful figure generation."""
        self.figures[figure_id] = FigureResult(
            figure_id=figure_id,
            title=title,
            status='success',
            output_path=output_path,
        )
    
    def mark_failed(self, figure_id: str, error: str, title: str):
        """Record failed figure generation."""
        self.figures[figure_id] = FigureResult(
            figure_id=figure_id,
            title=title,
            status='failed',
            error=error,
        )
        if self.success:
            self.success = False
    
    def mark_skipped(self, figure_id: str, reason: str, title: str):
        """Record skipped figure."""
        self.figures[figure_id] = FigureResult(
            figure_id=figure_id,
            title=title,
            status='skipped',
            reason=reason,
        )
    
    @property
    def num_success(self) -> int:
        return sum(1 for f in self.figures.values() if f.is_success())
    
    @property
    def num_failed(self) -> int:
        return sum(1 for f in self.figures.values() if f.is_failed())
    
    @property
    def num_skipped(self) -> int:
        return sum(1 for f in self.figures.values() if f.is_skipped())
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            '',
            '=' * 80,
            'FORECAST VISUALIZATION EXECUTION REPORT',
            '=' * 80,
            '',
            f'Summary: {self.num_success} success, {self.num_failed} failed, {self.num_skipped} skipped',
            '',
        ]
        
        if self.error:
            lines.extend([
                'FATAL ERROR:',
                f'  {self.error}',
                '',
            ])
        
        # Group by status
        successful = [f for f in self.figures.values() if f.is_success()]
        if successful:
            lines.extend([
                'GENERATED:',
            ])
            for result in sorted(successful, key=lambda x: x.figure_id):
                lines.append(f'  [{result.figure_id}] {result.title}')
                lines.append(f'      -> {result.output_path}')
            lines.append('')
        
        failed = [f for f in self.figures.values() if f.is_failed()]
        if failed:
            lines.extend([
                'FAILED:',
            ])
            for result in sorted(failed, key=lambda x: x.figure_id):
                lines.append(f'  [{result.figure_id}] {result.title}')
                lines.append(f'      Error: {result.error}')
            lines.append('')
        
        skipped = [f for f in self.figures.values() if f.is_skipped()]
        if skipped:
            lines.extend([
                'SKIPPED:',
            ])
            for result in sorted(skipped, key=lambda x: x.figure_id):
                lines.append(f'  [{result.figure_id}] {result.title}')
                lines.append(f'      Reason: {result.reason}')
            lines.append('')
        
        lines.extend([
            '=' * 80,
            '',
        ])
        
        return '\n'.join(lines)
    
    def __str__(self) -> str:
        return self.get_summary()


__all__ = [
    'ForecastVisualizationOrchestrator',
    'ExecutionReport',
    'FigureResult',
]
