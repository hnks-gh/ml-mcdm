# -*- coding: utf-8 -*-
"""
Professional logging system for ML-MCDM Pipeline.

Features:
- Colored console output with level-based styling
- Clean file logging (no ANSI codes)
- Log rotation with configurable size limits
- Structured JSON logging for machine parsing
- Hierarchical module logging
- Context tracking (pipeline phases, correlation IDs)
- Performance metrics and timing
- Progress tracking with visual feedback
"""

import logging
import logging.handlers
import json
import sys
import os
import re
import time
import threading
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
from enum import Enum


# =============================================================================
# Constants
# =============================================================================

LOG_NAME = "ml_mcdm"
DEFAULT_LOG_FORMAT = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
CONSOLE_DATE_FORMAT = "%H:%M:%S"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# =============================================================================
# ANSI Color Definitions
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    # Reset
    RESET = "\033[0m"
    
    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    
    # ANSI code pattern for stripping
    ANSI_PATTERN = re.compile(r'\033\[[0-9;]*m')
    
    @classmethod
    def strip(cls, text: str) -> str:
        """Remove all ANSI codes from text."""
        return cls.ANSI_PATTERN.sub('', text)
    
    @classmethod
    def supports_color(cls) -> bool:
        """Check if terminal supports colors."""
        if os.getenv("NO_COLOR"):
            return False
        if os.getenv("FORCE_COLOR"):
            return True
        if not hasattr(sys.stdout, "isatty"):
            return False
        if not sys.stdout.isatty():
            return False
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                # Enable ANSI escape sequences on Windows 10+
                kernel32.SetConsoleMode(
                    kernel32.GetStdHandle(-11), 7
                )
                return True
            except Exception:
                return os.getenv("TERM") == "xterm"
        return True


# =============================================================================
# Custom Formatters
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Formatter with ANSI color codes for console output.
    
    Provides level-based coloring and formatting for improved readability.
    """
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BRIGHT_RED + Colors.BOLD,
    }
    
    LEVEL_ICONS = {
        logging.DEBUG: "🔍",
        logging.INFO: "ℹ️ ",
        logging.WARNING: "⚠️ ",
        logging.ERROR: "❌",
        logging.CRITICAL: "🔥",
    }
    
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_icons: bool = False,
        use_colors: bool = True
    ):
        super().__init__(fmt, datefmt)
        self.use_icons = use_icons
        self.use_colors = use_colors and Colors.supports_color()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        # Create a copy to avoid modifying the original
        record = logging.makeLogRecord(record.__dict__)
        
        if self.use_colors:
            color = self.LEVEL_COLORS.get(record.levelno, "")
            record.levelname = f"{color}{Colors.BOLD}{record.levelname:8}{Colors.RESET}"
            record.msg = f"{color}{record.msg}{Colors.RESET}"
        
        if self.use_icons:
            icon = self.LEVEL_ICONS.get(record.levelno, "")
            record.msg = f"{icon} {record.msg}"
        
        return super().format(record)


class CleanFormatter(logging.Formatter):
    """
    Clean formatter for file output without ANSI codes.
    
    Produces clean, parseable log entries suitable for file storage.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record without colors."""
        # Create a copy to avoid modifying the original
        record = logging.makeLogRecord(record.__dict__)
        
        # Strip any ANSI codes from the message
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = Colors.strip(record.msg)
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Produces machine-readable JSON log entries.
    """
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
        self._default_keys = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'pathname', 'process', 'processName', 'relativeCreated',
            'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
            'taskName', 'message'
        }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        # Strip ANSI codes from message
        message = Colors.strip(record.getMessage())
        
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Add extra fields
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in self._default_keys:
                    try:
                        json.dumps(value)  # Check if serializable
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


# =============================================================================
# Custom Handlers
# =============================================================================

class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Thread-safe rotating file handler with error recovery.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record with thread safety."""
        with self._lock:
            try:
                super().emit(record)
            except Exception:
                self.handleError(record)


# =============================================================================
# Context Management
# =============================================================================

@dataclass
class LogContext:
    """
    Thread-local context for logging.
    
    Tracks pipeline phases, correlation IDs, and other contextual information.
    """
    _local = threading.local()
    
    @classmethod
    def get(cls) -> Dict[str, Any]:
        """Get current context."""
        if not hasattr(cls._local, 'context'):
            cls._local.context = {}
        return cls._local.context
    
    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set a context value."""
        ctx = cls.get()
        ctx[key] = value
    
    @classmethod
    def remove(cls, key: str) -> None:
        """Remove a context value."""
        ctx = cls.get()
        ctx.pop(key, None)
    
    @classmethod
    def clear(cls) -> None:
        """Clear all context."""
        cls._local.context = {}


class ContextFilter(logging.Filter):
    """
    Filter that adds context information to log records.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to record."""
        ctx = LogContext.get()
        for key, value in ctx.items():
            setattr(record, key, value)
        return True


# =============================================================================
# Progress Tracking
# =============================================================================

@dataclass
class PhaseMetrics:
    """Metrics for a pipeline phase."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"
    steps_total: int = 0
    steps_completed: int = 0
    sub_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def progress(self) -> float:
        """Get progress percentage."""
        if self.steps_total == 0:
            return 0.0
        return (self.steps_completed / self.steps_total) * 100


class ProgressLogger:
    """
    Context manager for logging progress of pipeline phases.
    
    Provides timing, step tracking, and visual progress indicators.
    
    Example:
        with ProgressLogger(logger, "Data Processing", total_steps=100) as progress:
            for i, item in enumerate(items):
                process(item)
                progress.update(message=f"Processed {item}")
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        total_steps: int = 0,
        show_progress_bar: bool = False,
        log_every: int = 1
    ):
        """
        Initialize progress logger.
        
        Parameters
        ----------
        logger : logging.Logger
            Logger instance to use
        operation : str
            Name of the operation being tracked
        total_steps : int
            Total number of steps (0 for unknown)
        show_progress_bar : bool
            Whether to show a visual progress bar
        log_every : int
            Log progress every N steps
        """
        self.logger = logger
        self.operation = operation
        self.total_steps = total_steps
        self.show_progress_bar = show_progress_bar
        self.log_every = log_every
        
        self.metrics = PhaseMetrics(name=operation, start_time=0.0, steps_total=total_steps)
        self._step_count = 0
    
    def __enter__(self) -> 'ProgressLogger':
        """Start the operation."""
        self.metrics.start_time = time.time()
        self.metrics.status = "running"
        
        LogContext.set("phase", self.operation)
        
        self.logger.info(f">> Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Complete the operation."""
        self.metrics.end_time = time.time()
        elapsed = self.metrics.elapsed
        
        LogContext.remove("phase")
        
        if exc_type is None:
            self.metrics.status = "completed"
            self.logger.info(
                f"[OK] Completed: {self.operation} "
                f"({elapsed:.2f}s{self._get_rate_info()})"
            )
        else:
            self.metrics.status = "failed"
            self.logger.error(
                f"✗ Failed: {self.operation} ({elapsed:.2f}s) - {exc_type.__name__}: {exc_val}"
            )
        
        return False
    
    def _get_rate_info(self) -> str:
        """Get rate information string."""
        if self._step_count > 0 and self.metrics.elapsed > 0:
            rate = self._step_count / self.metrics.elapsed
            return f", {rate:.1f} items/s"
        return ""
    
    def update(self, n: int = 1, message: str = "") -> None:
        """
        Update progress.
        
        Parameters
        ----------
        n : int
            Number of steps completed
        message : str
            Optional status message
        """
        self._step_count += n
        self.metrics.steps_completed += n
        
        if self._step_count % self.log_every == 0:
            if self.total_steps > 0:
                pct = (self._step_count / self.total_steps) * 100
                progress_msg = f"Progress: {self._step_count}/{self.total_steps} ({pct:.1f}%)"
                if message:
                    progress_msg += f" - {message}"
                self.logger.debug(progress_msg)
            elif message:
                self.logger.debug(f"Step {self._step_count}: {message}")
    
    def set_metric(self, name: str, value: Any) -> None:
        """
        Set a custom metric.
        
        Parameters
        ----------
        name : str
            Metric name
        value : Any
            Metric value
        """
        self.metrics.sub_metrics[name] = value
    
    def log_step(self, step_name: str, status: str = "done") -> None:
        """
        Log a named step completion.
        
        Parameters
        ----------
        step_name : str
            Name of the step
        status : str
            Status indicator
        """
        icons = {"done": "[OK]", "skip": "[SKIP]", "warn": "[WARN]", "fail": "[FAIL]"}
        icon = icons.get(status, "•")
        self.logger.info(f"  {icon} {step_name}")


# =============================================================================
# Logger Factory
# =============================================================================

class LoggerFactory:
    """
    Factory for creating and managing loggers.
    
    Provides centralized logger configuration and management.
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    _configured: bool = False
    _root_logger: Optional[logging.Logger] = None
    
    @classmethod
    def setup(
        cls,
        name: str = LOG_NAME,
        level: Union[int, str, LogLevel] = logging.INFO,
        log_file: Optional[Path] = None,
        json_file: Optional[Path] = None,
        console: bool = True,
        use_colors: bool = True,
        use_icons: bool = False,
        max_bytes: int = MAX_LOG_SIZE,
        backup_count: int = BACKUP_COUNT,
        **kwargs
    ) -> logging.Logger:
        """
        Setup and configure the root logger.
        
        Parameters
        ----------
        name : str
            Logger name
        level : int, str, or LogLevel
            Logging level
        log_file : Path, optional
            Path for plain text log file
        json_file : Path, optional
            Path for JSON log file
        console : bool
            Enable console output
        use_colors : bool
            Enable colored console output
        use_icons : bool
            Enable emoji icons in console
        max_bytes : int
            Maximum log file size before rotation
        backup_count : int
            Number of backup files to keep
        
        Returns
        -------
        logging.Logger
            Configured logger instance
        """
        # Convert level if needed
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        elif isinstance(level, LogLevel):
            level = level.value
        
        # Get or create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers.clear()
        logger.propagate = False
        
        # Add context filter
        logger.addFilter(ContextFilter())
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            # Use console_level if provided, otherwise use level
            actual_console_level = kwargs.get('console_level', level)
            console_handler.setLevel(actual_console_level)
            
            if use_colors:
                console_fmt = ColoredFormatter(
                    fmt='%(asctime)s │ %(levelname)s │ %(message)s',
                    datefmt=CONSOLE_DATE_FORMAT,
                    use_colors=use_colors,
                    use_icons=use_icons
                )
            else:
                # Simple text format without colors
                console_fmt = CleanFormatter(
                    fmt='%(asctime)s | %(message)s',
                    datefmt=CONSOLE_DATE_FORMAT
                )
            console_handler.setFormatter(console_fmt)
            logger.addHandler(console_handler)
        
        # File handler (debug log - logs everything at DEBUG level)
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = SafeRotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            # Always log at DEBUG level to capture all details
            file_handler.setLevel(logging.DEBUG)
            file_fmt = CleanFormatter(
                fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt=DEFAULT_DATE_FORMAT
            )
            file_handler.setFormatter(file_fmt)
            logger.addHandler(file_handler)
        
        # JSON file handler
        if json_file:
            json_file = Path(json_file)
            json_file.parent.mkdir(parents=True, exist_ok=True)
            
            json_handler = SafeRotatingFileHandler(
                json_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            json_handler.setLevel(level)
            json_handler.setFormatter(JSONFormatter())
            logger.addHandler(json_handler)
        
        cls._root_logger = logger
        cls._loggers[name] = logger
        cls._configured = True
        
        return logger
    
    @classmethod
    def get_logger(cls, name: str = LOG_NAME) -> logging.Logger:
        """
        Get a logger instance.
        
        Parameters
        ----------
        name : str
            Logger name (can be hierarchical, e.g., 'ml_mcdm.mcdm.topsis')
        
        Returns
        -------
        logging.Logger
            Logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Create hierarchical logger
        if cls._configured and cls._root_logger:
            # If name starts with root logger name, create child logger
            root_name = cls._root_logger.name
            if name.startswith(root_name):
                logger = logging.getLogger(name)
            else:
                # Create as child of root
                full_name = f"{root_name}.{name}"
                logger = logging.getLogger(full_name)
        else:
            # Not configured yet, setup default
            logger = cls.setup(name)
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def get_module_logger(cls, module_name: str) -> logging.Logger:
        """
        Get a logger for a specific module.
        
        Parameters
        ----------
        module_name : str
            Module name (e.g., 'mcdm.topsis')
        
        Returns
        -------
        logging.Logger
            Module-specific logger
        """
        root_name = cls._root_logger.name if cls._root_logger else LOG_NAME
        full_name = f"{root_name}.{module_name}"
        return cls.get_logger(full_name)


# =============================================================================
# Convenience Functions
# =============================================================================

def setup_logger(
    name: str = LOG_NAME,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
    debug_file: Optional[Path] = None
) -> logging.Logger:
    """
    Setup and configure logger (convenience function).
    
    Parameters
    ----------
    name : str
        Logger name
    level : int
        Logging level for console output
    log_file : Path, optional
        Path for log file (deprecated, use debug_file instead)
    console : bool
        Enable console output (simple text, no colors)
    debug_file : Path, optional
        Path for debug log file (logs everything at DEBUG level)
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    
    Notes
    -----
    - Console output: INFO level, simple text format (no colors)
    - Debug file: DEBUG level, detailed logging with timestamps
    """
    # Use debug_file if provided, otherwise fall back to log_file for backwards compatibility
    actual_debug_file = debug_file if debug_file else log_file
    
    return LoggerFactory.setup(
        name=name,
        level=logging.DEBUG,  # Set root logger to DEBUG to capture all
        log_file=actual_debug_file,
        console=console,
        use_colors=False,  # Simple text output
        use_icons=False,
        console_level=level  # INFO level for console
    )


def get_logger(name: str = LOG_NAME) -> logging.Logger:
    """
    Get existing logger or create default one.
    
    Parameters
    ----------
    name : str
        Logger name
    
    Returns
    -------
    logging.Logger
        Logger instance
    """
    return LoggerFactory.get_logger(name)


def get_module_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Parameters
    ----------
    module_name : str
        Module name
    
    Returns
    -------
    logging.Logger
        Module logger
    """
    return LoggerFactory.get_module_logger(module_name)


# =============================================================================
# Decorators
# =============================================================================

def log_execution(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    show_args: bool = False,
    show_result: bool = False
) -> Callable:
    """
    Decorator to log function execution.
    
    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use (defaults to module logger)
    level : int
        Log level
    show_args : bool
        Whether to log function arguments
    show_result : bool
        Whether to log function result
    
    Returns
    -------
    Callable
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger()
            
            func_name = func.__qualname__
            
            if show_args:
                args_str = ", ".join(
                    [repr(a)[:50] for a in args] + 
                    [f"{k}={repr(v)[:50]}" for k, v in kwargs.items()]
                )
                logger.log(level, f"Calling {func_name}({args_str})")
            else:
                logger.log(level, f"Calling {func_name}")
            
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                
                if show_result:
                    result_str = repr(result)[:100]
                    logger.log(level, f"{func_name} returned {result_str} ({elapsed:.3f}s)")
                else:
                    logger.log(level, f"{func_name} completed ({elapsed:.3f}s)")
                
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"{func_name} failed after {elapsed:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


def log_exceptions(
    logger: Optional[logging.Logger] = None,
    level: int = logging.ERROR,
    reraise: bool = True
) -> Callable:
    """
    Decorator to log exceptions.
    
    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use
    level : int
        Log level for exceptions
    reraise : bool
        Whether to reraise the exception
    
    Returns
    -------
    Callable
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(
                    level,
                    f"Exception in {func.__qualname__}: {e}",
                    exc_info=True
                )
                if reraise:
                    raise
        
        return wrapper
    return decorator


# =============================================================================
# Context Managers
# =============================================================================

@contextmanager
def log_context(**kwargs):
    """
    Context manager for adding temporary context to logs.
    
    Example:
        with log_context(user_id=123, operation="process"):
            logger.info("Processing data")  # Will include user_id and operation
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
    level: int = logging.INFO
):
    """
    Context manager for timing operations.
    
    Example:
        with timed_operation(logger, "data loading"):
            load_data()
    """
    start = time.time()
    logger.log(level, f"Starting: {operation}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.log(level, f"Finished: {operation} ({elapsed:.3f}s)")


# =============================================================================
# Pipeline-Specific Utilities
# =============================================================================

class PipelineLogger:
    """
    Specialized logger for ML-MCDM pipeline operations.
    
    Provides structured logging for pipeline phases, methods, and results.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize pipeline logger.
        
        Parameters
        ----------
        logger : logging.Logger
            Base logger instance
        """
        self.logger = logger
        self._phase_stack: List[str] = []
    
    def banner(self, title: str, char: str = "═", width: int = 60) -> None:
        """Log a banner message."""
        self.logger.info(char * width)
        self.logger.info(f"{title.center(width)}")
        self.logger.info(char * width)
    
    def section(self, title: str) -> None:
        """Log a section header."""
        self.logger.info(f"\n{'─' * 40}")
        self.logger.info(f"  {title}")
        self.logger.info(f"{'─' * 40}")
    
    def metric(self, name: str, value: Any, unit: str = "") -> None:
        """Log a metric value."""
        if isinstance(value, float):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)
        
        if unit:
            self.logger.info(f"  • {name}: {value_str} {unit}")
        else:
            self.logger.info(f"  • {name}: {value_str}")
    
    def metrics(self, metrics_dict: Dict[str, Any]) -> None:
        """Log multiple metrics."""
        for name, value in metrics_dict.items():
            self.metric(name, value)
    
    def ranking(self, rankings: List[tuple], title: str = "Rankings", top_n: int = 5) -> None:
        """Log a ranking list."""
        self.logger.info(f"\n  {title} (Top {top_n}):")
        for i, (entity, score) in enumerate(rankings[:top_n], 1):
            self.logger.info(f"    {i}. {entity}: {score:.4f}")
    
    def method_result(self, method: str, result_type: str, value: Any) -> None:
        """Log an MCDM method result."""
        self.logger.info(f"  {method}: {result_type} = {value:.4f}")
    
    def step(self, message: str, status: str = "info") -> None:
        """Log a step with status indicator."""
        icons = {
            "info": "*",
            "done": "[OK]",
            "skip": "[SKIP]", 
            "warn": "[WARN]",
            "error": "[X]"
        }
        icon = icons.get(status, "•")
        self.logger.info(f"  {icon} {message}")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main setup functions
    'setup_logger',
    'get_logger',
    'get_module_logger',
    
    # Classes
    'LoggerFactory',
    'ProgressLogger',
    'PipelineLogger',
    'LogContext',
    'LogLevel',
    'Colors',
    
    # Formatters
    'ColoredFormatter',
    'CleanFormatter', 
    'JSONFormatter',
    
    # Decorators
    'log_execution',
    'log_exceptions',
    
    # Context managers
    'log_context',
    'timed_operation',
    
    # Constants
    'LOG_NAME',
]
