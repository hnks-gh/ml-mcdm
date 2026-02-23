# -*- coding: utf-8 -*-
"""
ML-MCDM Logging Package
========================

Two-channel logging system:
  * **ConsoleLogger** — concise, colour-coded monitoring output
  * **DebugLogger** — exhaustive structured JSON for post-hoc inspection

Usage::

    from ml_mcdm.loggers import setup_logging
    console, debug = setup_logging('outputs')
"""

from .context import Colors, LogContext, PhaseMetrics
from .console_logger import ConsoleLogger
from .debug_logger import DebugLogger
from .decorators import log_execution, log_exceptions, log_context, timed_operation

from typing import Tuple


def setup_logging(
    output_dir: str = 'outputs',
) -> Tuple[ConsoleLogger, DebugLogger]:
    """Create and return both loggers.

    Parameters
    ----------
    output_dir : str
        Root output directory; debug JSON goes to ``<output_dir>/logs/``.

    Returns
    -------
    tuple[ConsoleLogger, DebugLogger]
    """
    console = ConsoleLogger()
    debug = DebugLogger(output_dir=f'{output_dir}/logs')
    return console, debug


# Backward-compat shims so existing ``from .logger import ...`` still work
# when migration is in progress.
def setup_logger(name: str = 'ml_mcdm', **kwargs):
    """Legacy shim — returns a stdlib Logger wired to the debug backend."""
    import logging
    return logging.getLogger(name)


def get_logger(name: str = 'ml_mcdm'):
    import logging
    return logging.getLogger(name)


def get_module_logger(module_name: str):
    import logging
    return logging.getLogger(f'ml_mcdm.{module_name}')


class ProgressLogger:
    """Legacy shim wrapping :class:`ConsoleLogger.phase`."""

    def __init__(self, logger, operation, **kwargs):
        self._logger = logger
        self._operation = operation

    def __enter__(self):
        import logging
        if isinstance(self._logger, logging.Logger):
            self._logger.info(f'>> Starting: {self._operation}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import logging
        if isinstance(self._logger, logging.Logger):
            if exc_type:
                self._logger.error(f'FAIL {self._operation}: {exc_val}')
            else:
                self._logger.info(f'[OK] Completed: {self._operation}')
        return False

    def update(self, n=1, message=''):
        pass

    def log_step(self, step_name, status='done'):
        pass


__all__ = [
    # Primary API
    'setup_logging',
    'ConsoleLogger',
    'DebugLogger',

    # Context & metrics
    'Colors',
    'LogContext',
    'PhaseMetrics',

    # Decorators & context managers
    'log_execution',
    'log_exceptions',
    'log_context',
    'timed_operation',

    # Backward-compat shims
    'setup_logger',
    'get_logger',
    'get_module_logger',
    'ProgressLogger',
]
