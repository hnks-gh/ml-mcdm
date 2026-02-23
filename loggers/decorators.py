# -*- coding: utf-8 -*-
"""
Logging Decorators and Context Managers
=======================================

Reusable decorators (``log_execution``, ``log_exceptions``) and context
managers (``log_context``, ``timed_operation``) that emit to both the
console logger and the debug logger.
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
    """Decorator that logs function entry, exit, and timing."""

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
    """Decorator that captures and logs unhandled exceptions."""

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
    """Temporarily inject key/value pairs into the thread-local log context."""
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
    """Context manager that logs start / finish with elapsed time."""
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
