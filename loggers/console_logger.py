# -*- coding: utf-8 -*-
"""
Professional Console Logger for ML-MCDM Pipeline
=================================================

Provides concise, colour-coded, structured output designed specifically
for real-time monitoring of pipeline execution.  All console output is
routed through this single class so that monitoring is consistent.

Design goals
------------
* One-line status per step (no wall of text)
* Phase banners with timing
* Compact metric / table display
* Final run summary replacing old ``print_results()``
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

from .context import Colors, LogContext, PhaseMetrics


# Width of the banner / separator lines
_LINE_W = 70


class ConsoleLogger:
    """Structured, professional console logger for monitoring pipeline runs."""

    def __init__(self, use_color: Optional[bool] = None):
        self._color = Colors.supports_color() if use_color is None else use_color
        self._phase_stack: List[PhaseMetrics] = []
        self._all_phases: List[PhaseMetrics] = []

    # ------------------------------------------------------------------
    # Colour helpers
    # ------------------------------------------------------------------

    def _c(self, text: str, *codes: str) -> str:
        if not self._color:
            return text
        return ''.join(codes) + text + Colors.RESET

    # ------------------------------------------------------------------
    # Low-level write
    # ------------------------------------------------------------------

    def _write(self, msg: str) -> None:
        sys.stdout.write(msg + '\n')
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Banners & separators
    # ------------------------------------------------------------------

    def banner(self, title: str, subtitle: str = '') -> None:
        """Print a prominent banner (e.g. at startup)."""
        self._write('')
        self._write(self._c('=' * _LINE_W, Colors.BOLD, Colors.BLUE))
        self._write(self._c(f'  {title}', Colors.BOLD, Colors.BRIGHT_WHITE))
        if subtitle:
            self._write(self._c(f'  {subtitle}', Colors.DIM))
        self._write(self._c('=' * _LINE_W, Colors.BOLD, Colors.BLUE))
        self._write('')

    def separator(self, char: str = '-') -> None:
        self._write(self._c(char * _LINE_W, Colors.DIM))

    # ------------------------------------------------------------------
    # Phase management (context manager)
    # ------------------------------------------------------------------

    @contextmanager
    def phase(self, name: str, number: int = None,
              total_phases: int = 7) -> Generator[_PhaseCtx, None, None]:
        """Context manager that prints phase start / end with timing.

        Example::

            with console.phase('Data Loading') as p:
                data = load(...)
                p.detail(f'{len(data)} records loaded')
        """
        if number is None:
            number = len(self._all_phases) + 1
        label = f'[{number}/{total_phases}] {name}'
        metrics = PhaseMetrics(name=name, start_time=time.time(),
                               steps_total=total_phases)
        self._phase_stack.append(metrics)
        self._all_phases.append(metrics)
        LogContext.set('phase', name)

        self._write('')
        self._write(self._c(f'>> {label}', Colors.BOLD, Colors.CYAN))

        ctx = _PhaseCtx(self, metrics)
        try:
            yield ctx
        except Exception as exc:
            metrics.end_time = time.time()
            metrics.status = 'failed'
            LogContext.remove('phase')
            self._phase_stack.pop()
            self._write(self._c(
                f'   FAIL  {label}  ({metrics.elapsed:.2f}s) — {type(exc).__name__}: {exc}',
                Colors.RED, Colors.BOLD,
            ))
            raise
        else:
            metrics.end_time = time.time()
            metrics.status = 'completed'
            LogContext.remove('phase')
            self._phase_stack.pop()
            self._write(self._c(
                f'   OK    {label}  ({metrics.elapsed:.2f}s)',
                Colors.GREEN,
            ))

    # ------------------------------------------------------------------
    # Step / metric / table helpers (used inside phases)
    # ------------------------------------------------------------------

    def step(self, message: str) -> None:
        """Print a substep inside the current phase."""
        self._write(self._c(f'   . {message}', Colors.WHITE))

    def metric(self, label: str, value: Any, unit: str = '') -> None:
        """Print a key-value metric."""
        if isinstance(value, float):
            val_str = f'{value:.4f}'
        else:
            val_str = str(value)
        suffix = f' {unit}' if unit else ''
        self._write(self._c(f'     {label}: ', Colors.DIM) + f'{val_str}{suffix}')

    def metrics(self, data: Dict[str, Any]) -> None:
        """Print a set of metrics on a single line, comma-separated."""
        parts = []
        for k, v in data.items():
            if isinstance(v, float):
                parts.append(f'{k}={v:.4f}')
            else:
                parts.append(f'{k}={v}')
        self._write(self._c('     ', Colors.DIM) + ', '.join(parts))

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[str]],
              col_widths: Optional[Sequence[int]] = None, indent: int = 6) -> None:
        """Print a compact fixed-width table."""
        if col_widths is None:
            col_widths = [max(len(h) + 2, 12) for h in headers]
        pad = ' ' * indent
        hdr = pad + '  '.join(f'{h:^{w}}' for h, w in zip(headers, col_widths))
        self._write(self._c(hdr, Colors.BOLD))
        self._write(pad + '  '.join('-' * w for w in col_widths))
        for row in rows:
            cells = []
            for c, w in zip(row, col_widths):
                try:
                    float(str(c).replace('+', '').replace('%', ''))
                    cells.append(f'{c:>{w}}')
                except (ValueError, AttributeError):
                    cells.append(f'{c:<{w}}')
            self._write(pad + '  '.join(cells))

    # ------------------------------------------------------------------
    # Informational / warning / error
    # ------------------------------------------------------------------

    def info(self, message: str) -> None:
        self._write(self._c(f'  i {message}', Colors.GREEN))

    def success(self, message: str) -> None:
        self._write(self._c(f'  OK {message}', Colors.BRIGHT_GREEN, Colors.BOLD))

    def warning(self, message: str) -> None:
        self._write(self._c(f'  ! {message}', Colors.YELLOW))

    def error(self, message: str) -> None:
        self._write(self._c(f'  X {message}', Colors.RED, Colors.BOLD))

    # ------------------------------------------------------------------
    # Run summary (replaces old main.py print_results)
    # ------------------------------------------------------------------

    def show_run_summary(self, result: Any) -> None:
        """Print an end-of-run summary covering every pipeline phase."""
        import numpy as np

        self._write('')
        self._write(self._c('=' * _LINE_W, Colors.BOLD, Colors.BLUE))
        self._write(self._c('  RESULTS SUMMARY', Colors.BOLD, Colors.BRIGHT_WHITE))
        self._write(self._c('=' * _LINE_W, Colors.BOLD, Colors.BLUE))

        # Data overview
        pd_ = result.panel_data
        self._write(self._c('\n  DATA', Colors.BOLD))
        self.metric('Provinces', len(pd_.provinces))
        self.metric('Years', f'{min(pd_.years)}-{max(pd_.years)} ({len(pd_.years)} yr)')
        self.metric('Subcriteria', pd_.n_subcriteria)
        self.metric('Criteria', pd_.n_criteria)

        # Top 10 rankings
        self._write(self._c('\n  TOP 10 RANKINGS (Evidential Reasoning)', Colors.BOLD))
        ranking_df = result.get_final_ranking_df()
        rows = []
        for _, row in ranking_df.head(10).iterrows():
            rows.append([str(int(row['ER_Rank'])), row['Province'],
                         f'{row["ER_Score"]:.4f}'])
        self.table(['Rank', 'Province', 'ER Score'], rows, [6, 25, 10])

        # Concordance
        self._write(self._c('\n  CONCORDANCE', Colors.BOLD))
        w = result.ranking_result.kendall_w
        interp = ('Strong agreement' if w > 0.7
                  else 'Moderate agreement' if w > 0.5
                  else 'Weak agreement')
        self.metric("Kendall's W", w)
        self.metric('Interpretation', interp)

        # Sensitivity
        if result.sensitivity_result:
            self._write(self._c('\n  SENSITIVITY', Colors.BOLD))
            self.metric('Robustness', result.sensitivity_result.overall_robustness)

        # Forecast
        if result.forecast_result:
            self._write(self._c('\n  ML FORECASTING', Colors.BOLD))
            cv_scores = result.forecast_result.cross_validation_scores
            all_cv = [s for scores in cv_scores.values() for s in scores]
            self.metric('Mean CV R²', np.mean(all_cv))
            self.metric('Std  CV R²', np.std(all_cv))

        # Runtime
        self._write(self._c(f'\n  RUNTIME : {result.execution_time:.2f}s', Colors.BOLD))
        self._write(self._c('=' * _LINE_W, Colors.BOLD, Colors.BLUE))

    # ------------------------------------------------------------------
    # Completion banner
    # ------------------------------------------------------------------

    def show_completion(self, output_dir: str = 'outputs') -> None:
        """Print the final 'analysis complete' box."""
        self._write('')
        self._write(self._c('=' * _LINE_W, Colors.BOLD, Colors.GREEN))
        self._write(self._c('  ANALYSIS COMPLETE', Colors.BOLD, Colors.BRIGHT_GREEN))
        self._write(self._c('=' * _LINE_W, Colors.BOLD, Colors.GREEN))
        self._write(f'  All outputs saved to {output_dir}/:')
        self._write('    figures/  — high-resolution charts (300 DPI)')
        self._write('    results/  — numerical data (CSV / JSON)')
        self._write('    reports/  — comprehensive Markdown report')
        self._write(self._c('=' * _LINE_W, Colors.BOLD, Colors.GREEN))
        self._write('')


# ------------------------------------------------------------------
# Phase context helper returned by ConsoleLogger.phase()
# ------------------------------------------------------------------

class _PhaseCtx:
    """Lightweight proxy for logging detail inside a phase block."""

    def __init__(self, logger: ConsoleLogger, metrics: PhaseMetrics):
        self._logger = logger
        self.metrics = metrics

    # Delegate common helpers
    def detail(self, message: str) -> None:
        self._logger.step(message)

    def metric(self, label: str, value: Any, unit: str = '') -> None:
        self._logger.metric(label, value, unit)

    def metrics(self, data: Dict[str, Any]) -> None:
        self._logger.metrics(data)

    def warning(self, message: str) -> None:
        self._logger.warning(message)


__all__ = ['ConsoleLogger']
