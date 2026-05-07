"""Metrics package for taxonomy, guardrails, and transformations.

This package houses the core definitions and statistical calculation routines for metrics
evaluated during an experiment:
- `BaseMetric`: The abstract base class establishing standard evaluation contracts.
- `MeanMetric`: For continuous measurements, supporting pre-period CUPED adjustments.
- `ProportionMetric`: For binary binomial event rates.
- `RatioMetric`: For compound aggregate metrics (e.g., Click-Through Rate) evaluated via Delta Method variance.
- `GuardrailMetric`: Special monitoring wrapper to prevent platform/business regressions.
- `log_transform`: Normalization for extremely skewed continuous metrics.
- `delta_normalization`: Taylor expansion adjustment for advanced aggregate metrics.
"""

from xpyrment.metrics.guardrails import GuardrailMetric
from xpyrment.metrics.taxonomy import BaseMetric, MeanMetric, ProportionMetric, RatioMetric
from xpyrment.metrics.transformations import log_transform, delta_normalization

__all__ = [
    "BaseMetric",
    "MeanMetric",
    "ProportionMetric",
    "RatioMetric",
    "GuardrailMetric",
    "log_transform",
    "delta_normalization",
]
