"""Guardrail metrics to protect core platform health and business stability.

This module defines `GuardrailMetric`, which wraps regular statistical metrics (such as latency,
error rates, or severe business indicators like unsubscribe rate) with critical threshold
boundaries. During the course of an experiment, guardrails are checked to detect whether a
treatment arm has caused severe platform degradation, triggering automated termination or alert signals.
"""

from typing import Dict, Any
from xpyrment.metrics.taxonomy import BaseMetric


class GuardrailMetric:
    """Defines a guardrail metric with specific breach thresholds.

    Guardrail metrics are designed to prevent treatment arms from causing catastrophic regressions
    on critical secondary metrics. Unlike primary metrics (where we search for significant positive
    changes), guardrail metrics are evaluated to ensure that they do not deteriorate beyond a
    pre-specified tolerance boundary, irrespective of statistical significance.

    Attributes:
        metric (BaseMetric): The underlying metric to monitor (e.g., MeanMetric, RatioMetric).
        max_allowed_change (float): The maximum tolerated relative change (positive or negative)
            expressed as a fraction (e.g., `0.01` represents a 1% threshold).

    Examples:
        ??? example "Example"

            ```python
            >>> from xpyrment.metrics.taxonomy import MeanMetric
            >>> from xpyrment.metrics.guardrails import GuardrailMetric
            >>> latency_metric = MeanMetric("Page Latency", value_col="load_time")
            >>> guardrail = GuardrailMetric(latency_metric, max_allowed_change=0.02) # 2% max increase
            >>> calc_result = {"metric_name": "Page Latency", "relative_lift": 0.035} # 3.5% lift (regression)
            >>> guardrail.check_breach(calc_result)
            True
            ```
    """

    def __init__(self, metric: BaseMetric, max_allowed_change: float = 0.01):
        """Initializes a GuardrailMetric wrapper.

        Args:
            metric (BaseMetric): The concrete metric instance being monitored.
            max_allowed_change (float): The threshold for the maximum absolute relative lift allowed before
                triggering a breach. Defaults to 0.01 (1%).
        """
        self.metric = metric
        self.max_allowed_change = max_allowed_change

    def check_breach(self, calculation_result: Dict[str, Any]) -> bool:
        r"""Determines if the calculated lift breaches the guardrail thresholds.

        Mathematical Representation:
            Let $L$ be the relative lift calculated for the wrapped metric:
            $$L = \frac{\bar{Y}_T - \bar{Y}_C}{\bar{Y}_C}$$
            A breach is detected if the magnitude of the relative lift exceeds the maximum allowed change:
            $$\text{Breach} = |L| > \text{max\_allowed\_change}$$

        Args:
            calculation_result (Dict[str, Any]): Output dictionary produced by calling
                `metric.calculate()` on experimental data.

        Returns:
            bool: True if the relative lift is larger in magnitude than `max_allowed_change` (breached),
                False otherwise.
        """
        lift = calculation_result.get("relative_lift", 0.0)
        # Breach occurs if metric deteriorates beyond max_allowed_change
        return abs(lift) > self.max_allowed_change
