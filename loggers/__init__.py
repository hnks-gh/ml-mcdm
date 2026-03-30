# -*- coding: utf-8 -*-
"""
Logging and Monitoring Infrastructure.

This package provides a dual-channel logging system:
- **ConsoleLogger**: Concise, color-coded output for real-time monitoring.
- **DebugLogger**: Exhaustive, structured JSON logging for post-hoc inspection.

It also includes decorators and context managers for performance profiling 
and exception tracking across the ML-MCDM pipeline.
"""

from .context import Colors, LogContext, PhaseMetrics
from .console_logger import ConsoleLogger
from .debug_logger import DebugLogger
from .decorators import log_execution, log_exceptions, log_context, timed_operation

from typing import Tuple


def setup_logging(
    output_dir: str = 'outputs',
) -> Tuple[ConsoleLogger, DebugLogger]:
    """
    Initialize the dual-channel logging system.

    Parameters
    ----------
    output_dir : str, default='outputs'
        The root directory for outputs. Debug logs will be saved to 
        ``<output_dir>/logs/``.

    Returns
    -------
    Tuple[ConsoleLogger, DebugLogger]
        A pair of (console_logger, debug_logger) instances.
    """
    console = ConsoleLogger()
    debug = DebugLogger(output_dir=f'{output_dir}/logs')
    return console, debug


# Backward-compat shims so existing ``from .logger import ...`` still work
# when migration is in progress.
def setup_logger(name: str = 'ml_mcdm', **kwargs):
    """
    Legacy shim for stdlib logging initialization.

    Parameters
    ----------
    name : str, default='ml_mcdm'
        The logger name.
    **kwargs
        Additional arguments passed to legacy callers.

    Returns
    -------
    logging.Logger
        A standard library logger instance.
    """
    import logging
    return logging.getLogger(name)


def get_logger(name: str = 'ml_mcdm'):
    """
    Retrieve a standard library logger by name.

    Parameters
    ----------
    name : str, default='ml_mcdm'
        The logger name.

    Returns
    -------
    logging.Logger
        The requested logger instance.
    """
    import logging
    return logging.getLogger(name)


def get_module_logger(module_name: str):
    """
    Retrieve a logger for a specific ML-MCDM module.

    Parameters
    ----------
    module_name : str
        The short name of the module (e.g., 'ranking').

    Returns
    -------
    logging.Logger
        A logger scoped to 'ml_mcdm.<module_name>'.
    """
    import logging
    return logging.getLogger(f'ml_mcdm.{module_name}')


class ProgressLogger:
    """
    Legacy shim wrapping ConsoleLogger phase transitions.

    Provides a context manager interface for tracking operation progress.
    """

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
