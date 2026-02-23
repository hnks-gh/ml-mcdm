# -*- coding: utf-8 -*-
"""
Structured Debug Logger for ML-MCDM Pipeline
=============================================

Records **every** detail of a pipeline run into a single structured
JSON array file (``outputs/logs/debug_<timestamp>.json``).  Designed
for post-hoc inspection or automated quality-assurance tooling.

Each entry carries: timestamp, level, module, function, line, phase,
message, optional structured *data* payload, and optional duration_ms.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .context import Colors, LogContext


class DebugLogger:
    """Accumulates structured log entries and flushes to a JSON array file."""

    def __init__(self, output_dir: str = 'outputs/logs'):
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._path = self._dir / f'debug_{ts}.json'
        self._entries: List[Dict[str, Any]] = []

        # Also set up a stdlib DEBUG-level handler for submodules
        # that still use ``logging.getLogger('ml_mcdm')``
        self._stdlib_logger = logging.getLogger('ml_mcdm')
        self._stdlib_logger.setLevel(logging.DEBUG)
        self._stdlib_logger.propagate = False
        # Intercept handler
        self._handler = _InterceptHandler(self)
        self._stdlib_logger.addHandler(self._handler)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def debug(self, message: str, *, data: Any = None,
              module: str = '', function: str = '', line: int = 0) -> None:
        self._add('DEBUG', message, data=data, module=module,
                  function=function, line=line)

    def info(self, message: str, *, data: Any = None,
             module: str = '', function: str = '', line: int = 0) -> None:
        self._add('INFO', message, data=data, module=module,
                  function=function, line=line)

    def warning(self, message: str, *, data: Any = None,
                module: str = '', function: str = '', line: int = 0) -> None:
        self._add('WARNING', message, data=data, module=module,
                  function=function, line=line)

    def error(self, message: str, *, data: Any = None,
              module: str = '', function: str = '', line: int = 0) -> None:
        self._add('ERROR', message, data=data, module=module,
                  function=function, line=line)

    def exception(self, message: str, exc: Optional[BaseException] = None,
                  *, module: str = '', function: str = '', line: int = 0) -> None:
        tb = traceback.format_exc() if exc is None else traceback.format_exception(
            type(exc), exc, exc.__traceback__)
        self._add('ERROR', message, data={'traceback': tb},
                  module=module, function=function, line=line)

    def log_data(self, label: str, payload: Any, *,
                 module: str = '', function: str = '') -> None:
        """Store an arbitrary structured data payload (arrays, dicts, …)."""
        self._add('DATA', label, data=payload, module=module, function=function)

    # ------------------------------------------------------------------
    # Flush / close
    # ------------------------------------------------------------------

    def flush(self) -> str:
        """Write accumulated entries to disk and return the file path."""
        with open(self._path, 'w', encoding='utf-8') as fh:
            json.dump(self._entries, fh, indent=2, default=_json_default,
                      ensure_ascii=False)
        return str(self._path)

    def close(self) -> str:
        """Flush and detach the stdlib intercept handler."""
        path = self.flush()
        self._stdlib_logger.removeHandler(self._handler)
        return path

    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _add(self, level: str, message: str, *,
             data: Any = None, module: str = '', function: str = '',
             line: int = 0, duration_ms: Optional[float] = None) -> None:
        ctx = LogContext.get()
        entry: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'module': module,
            'function': function,
            'line': line,
            'phase': ctx.get('phase', ''),
            'message': Colors.strip(str(message)),
        }
        if data is not None:
            entry['data'] = data
        if duration_ms is not None:
            entry['duration_ms'] = round(duration_ms, 3)
        self._entries.append(entry)


# ------------------------------------------------------------------
# Stdlib-compatible intercept handler
# ------------------------------------------------------------------

class _InterceptHandler(logging.Handler):
    """Bridges stdlib ``logging`` records into :class:`DebugLogger`."""

    def __init__(self, debug_logger: DebugLogger):
        super().__init__(level=logging.DEBUG)
        self._dl = debug_logger

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._dl._add(
                level=record.levelname,
                message=record.getMessage(),
                module=record.module,
                function=record.funcName,
                line=record.lineno,
            )
        except Exception:
            self.handleError(record)


# ------------------------------------------------------------------
# JSON serialisation helper
# ------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Fallback serialiser for numpy / pandas objects."""
    import numpy as np
    import pandas as pd
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='list')
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return str(obj)


__all__ = ['DebugLogger']
