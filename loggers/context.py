# -*- coding: utf-8 -*-
"""
Shared Context, Metrics, and Color Utilities for ML-MCDM Logging
================================================================

Thread-safe context tracking, phase-timing metrics, and ANSI color
helpers used by both the console and debug loggers.
"""

import os
import re
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# =============================================================================
# ANSI Colour Helpers
# =============================================================================

class Colors:
    """ANSI escape sequences for terminal styling."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Foreground
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"

    ANSI_PATTERN = re.compile(r'\033\[[0-9;]*m')

    @classmethod
    def strip(cls, text: str) -> str:
        """Remove all ANSI codes from *text*."""
        return cls.ANSI_PATTERN.sub('', text)

    @classmethod
    def supports_color(cls) -> bool:
        """Return ``True`` if the hosting terminal supports ANSI colours."""
        if os.getenv("NO_COLOR"):
            return False
        if os.getenv("FORCE_COLOR"):
            return True
        if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            return False
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                return True
            except Exception:
                return os.getenv("TERM") == "xterm"
        return True


# =============================================================================
# Thread-Local Log Context
# =============================================================================

class LogContext:
    """Thread-local key/value store for contextual log annotations."""

    _local = threading.local()

    @classmethod
    def get(cls) -> Dict[str, Any]:
        if not hasattr(cls._local, 'context'):
            cls._local.context = {}
        return cls._local.context

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        cls.get()[key] = value

    @classmethod
    def remove(cls, key: str) -> None:
        cls.get().pop(key, None)

    @classmethod
    def clear(cls) -> None:
        cls._local.context = {}


# =============================================================================
# Phase Metrics
# =============================================================================

@dataclass
class PhaseMetrics:
    """Timing and step-count information for a single pipeline phase."""

    name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"
    steps_total: int = 0
    steps_completed: int = 0
    sub_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def progress(self) -> float:
        if self.steps_total == 0:
            return 0.0
        return (self.steps_completed / self.steps_total) * 100


__all__ = [
    'Colors',
    'LogContext',
    'PhaseMetrics',
]
