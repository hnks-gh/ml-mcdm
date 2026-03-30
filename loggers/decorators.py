# -*- coding: utf-8 -*-
"""
Logging Decorators and Performance Monitoring Context Managers.

This module provides reusable utilities to automate execution logging, 
exception tracking, and performance profiling across the ML-MCDM pipeline. 
It supports dynamic thread-local context injection and standardized timing 
outputs for both console and debug channels.
"""

from __future__ import annotations

import time
import logging
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, Optional

from .context import LogContext


# =============================================================================
# Decorators
# =============================================================================

def log_execution(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    show_args: bool = False,
    show_result: bool = False,
) -> Callable:
    """
    Decorator that logs function entry, exit, and execution time.

    Parameters
    ----------
    logger : logging.Logger, optional
        The logger instance to use. Defaults to the 'ml_mcdm' root logger.
    level : int, default=logging.DEBUG
        The logging level for entry/exit messages.
    show_args : bool, default=False
        If True, includes truncated function arguments in the entry log.
    show_result : bool, default=False
        If True, includes truncated return values in the exit log.

    Returns
    -------
    Callable
        The wrapped function with execution logging.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger('ml_mcdm')

            func_name = func.__qualname__

            if show_args:
                args_str = ', '.join(
                    [repr(a)[:50] for a in args]
                    + [f'{k}={repr(v)[:50]}' for k, v in kwargs.items()]
                )
                logger.log(level, f'Calling {func_name}({args_str})')
            else:
                logger.log(level, f'Calling {func_name}')

            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                if show_result:
                    result_str = repr(result)[:100]
                    logger.log(level, f'{func_name} returned {result_str} ({elapsed:.3f}s)')
                else:
                    logger.log(level, f'{func_name} completed ({elapsed:.3f}s)')
                return result
            except Exception as exc:
                elapsed = time.time() - start
                logger.error(f'{func_name} failed after {elapsed:.3f}s: {exc}')
                raise

        return wrapper
    return decorator


def log_exceptions(
    logger: Optional[logging.Logger] = None,
    level: int = logging.ERROR,
    reraise: bool = True,
) -> Callable:
    """
    Decorator that captures and logs unhandled exceptions.

    Ensures that unexpected failures are recorded in both the standard 
    and structured debug logs with full traceback information.

    Parameters
    ----------
    logger : logging.Logger, optional
        The logger instance to use.
    level : int, default=logging.ERROR
        The logging level for the exception message.
    reraise : bool, default=True
        If True, re-raises the exception after logging.

    Returns
    -------
    Callable
        The wrapped function with automated exception logging.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger('ml_mcdm')
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                logger.log(level, f'Exception in {func.__qualname__}: {exc}',
                           exc_info=True)
                if reraise:
                    raise
        return wrapper
    return decorator


# =============================================================================
# Context Managers
# =============================================================================

@contextmanager
def log_context(**kwargs: Any) -> Generator[None, None, None]:
    """
    Temporarily inject metadata into the thread-local logging context.

    Parameters
    ----------
    **kwargs
        Arbitrary key/value pairs to add to the logger context 
        (e.g., phase='DataLoading', province='Hanoi').
    """
    for key, value in kwargs.items():
        LogContext.set(key, value)
    try:
        yield
    finally:
        for key in kwargs:
            LogContext.remove(key)


@contextmanager
def timed_operation(
    logger: logging.Logger,
    operation: str,
    level: int = logging.INFO,
) -> Generator[None, None, None]:
    """
    Context manager that benchmarks a specific operation.

    Parameters
    ----------
    logger : logging.Logger
        The logger instance to use.
    operation : str
        A descriptive name for the timed task.
    level : int, default=logging.INFO
        The logging level for both start and finish messages.
    """
    start = time.time()
    logger.log(level, f'Starting: {operation}')
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.log(level, f'Finished: {operation} ({elapsed:.3f}s)')


__all__ = [
    'log_execution',
    'log_exceptions',
    'log_context',
    'timed_operation',
]
