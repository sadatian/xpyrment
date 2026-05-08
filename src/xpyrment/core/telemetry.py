"""Centralized Telemetry, Structured JSON Logging, and Execution Profiling (Block 56).

This module provides logging formatters, custom structured logging routines, and context manager
profilers to measure execution stages, high-resolution times, and peak memory allocations.
"""

from contextlib import contextmanager
import json
import logging
import os
import sys
import time
import tracemalloc
from typing import Any, Dict, Generator, Optional


class JSONFormatter(logging.Formatter):
    """Custom logging formatter that structures log records into standard, compliance-ready JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record to a serialized JSON string.

        Args:
            record (logging.LogRecord): The log record to process.

        Returns:
            str: JSON string containing standard log fields and any structured extra payload.
        """
        log_payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
        }

        # Seamlessly merge any dictionary fields passed as "extra_fields"
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_payload.update(record.extra_fields)

        return json.dumps(log_payload)


def configure_telemetry(level: int = logging.INFO, stream: Any = sys.stdout) -> logging.Logger:
    """Configures and registers the centralized JSON telemetry logger handlers.

    Args:
        level (int): Log filter level threshold (e.g. logging.INFO). Defaults to logging.INFO.
        stream (Any): Output stream target. Defaults to sys.stdout.

    Returns:
        logging.Logger: Configured telemetry Logger instance.
    """
    logger = logging.getLogger("xpyrment.telemetry")
    logger.setLevel(level)
    logger.propagate = False

    # Remove duplicates
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream)
    formatter = JSONFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logger() -> logging.Logger:
    """Returns the centralized JSON telemetry logger instance.

    Ensures the logger is fully configured with default handlers if unconfigured.

    Returns:
        logging.Logger: Active telemetry Logger.
    """
    logger = logging.getLogger("xpyrment.telemetry")
    if not logger.handlers:
        configure_telemetry()
    return logger


class ExecutionProfiler:
    """Context manager and decorator tracking processing stages, execution duration, and peak memory usage.

    In production platforms, fine-grained telemetry profile statistics are crucial to detect performance bottlenecks,
    resource hot spots, and algorithmic memory leaks (especially within bootstrap, MCMC, or massive-scale matrix solvers).
    """

    def __init__(self, stage_name: str, logger: Optional[logging.Logger] = None):
        """Initializes the ExecutionProfiler.

        Args:
            stage_name (str): Identifier label of the current processing stage (e.g., "bootstrap_resampling").
            logger (Optional[logging.Logger]): Custom target logger instance. Defaults to None.
        """
        self.stage_name = stage_name
        self.logger = logger or get_logger()
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.peak_memory_bytes: int = 0

    def __enter__(self) -> "ExecutionProfiler":
        """Enters the context boundary, initiating tracemalloc memory tracing and epoch timers.

        Returns:
            ExecutionProfiler: Active instance.
        """
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        tracemalloc.clear_traces()

        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exits the context boundary, stops tracking, records peaks, and logs structured JSON telemetry metrics.

        Args:
            exc_type (Any): Exception type raised within the context.
            exc_val (Any): Exception value.
            exc_tb (Any): Traceback object.

        Returns:
            bool: False to propagate any exceptions raised in the block.
        """
        self.end_time = time.perf_counter()
        elapsed_seconds = self.end_time - self.start_time

        # Fetch peak memory from tracemalloc
        _, peak = tracemalloc.get_traced_memory()
        self.peak_memory_bytes = peak

        status = "SUCCESS" if exc_type is None else "FAILED"
        profile_metrics = {
            "stage": self.stage_name,
            "duration_seconds": elapsed_seconds,
            "peak_memory_kb": self.peak_memory_bytes / 1024.0,
            "status": status,
        }

        log_msg = f"Profile completed for stage '{self.stage_name}' with status {status}"

        if exc_type is not None:
            profile_metrics["error_type"] = exc_type.__name__
            profile_metrics["error_message"] = str(exc_val)
            self.logger.error(log_msg, extra={"extra_fields": profile_metrics})
        else:
            self.logger.info(log_msg, extra={"extra_fields": profile_metrics})

        return False  # Propagate standard exceptions

    def __call__(self, func: Any) -> Any:
        """Allows class to function seamlessly as an execution profiler decorator for standard python functions.

        Args:
            func (Any): Callable target function to wrap.

        Returns:
            Any: Decorated callable function.
        """
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
