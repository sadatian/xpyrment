"""Core engine abstractions, state management, exception classes, and shared types.

This package provides the foundational structural mechanisms for the `xpyrment` package:
- `Experiment`: The central orchestration state container that governs execution.
- `ExperimentState`: The rigid phase-gating mechanism (CREATED -> PLANNED -> DESIGNED -> RUNNING -> ANALYZED -> REPORTED).
- `ExperimentRegistry`: Cryptographic hashing and pre-registration validator to prevent post-hoc changes.
- Custom Exceptions: Robust, informative error feedback to protect experimental integrity (`PhaseOrderError`, `SRMError`, `AliasError`).
- Strict Typing schemas: Standardized TypedDict representation (`MetricResult`) of calculation outputs.
"""

from xpyrment.core.exceptions import PhaseOrderError, SRMError, AliasError
from xpyrment.core.experiment import Experiment
from xpyrment.core.registry import ExperimentRegistry
from xpyrment.core.state import ExperimentState
from xpyrment.core.types import MetricType, MetricResult
from xpyrment.core.serialization import make_serializable, serialize_to_json
from xpyrment.core.telemetry import configure_telemetry, get_logger, ExecutionProfiler
from xpyrment.core.cache import (
    StatisticalCache,
    statistical_cache,
    cached_statistical,
    cached_t_cdf,
    cached_t_ppf,
    cached_norm_cdf,
    cached_norm_ppf,
    cached_chi2_cdf,
    cached_chi2_sf,
    cached_chi2_ppf,
)

__all__ = [
    "Experiment",
    "ExperimentState",
    "ExperimentRegistry",
    "PhaseOrderError",
    "SRMError",
    "AliasError",
    "MetricType",
    "MetricResult",
    "make_serializable",
    "serialize_to_json",
    "configure_telemetry",
    "get_logger",
    "ExecutionProfiler",
    "StatisticalCache",
    "statistical_cache",
    "cached_statistical",
    "cached_t_cdf",
    "cached_t_ppf",
    "cached_norm_cdf",
    "cached_norm_ppf",
    "cached_chi2_cdf",
    "cached_chi2_sf",
    "cached_chi2_ppf",
]
