from typing import Dict, Any
from xpyrment.metrics.taxonomy import BaseMetric


class GuardrailMetric:
    """Defines a guardrail metric with specific breach thresholds."""

    def __init__(self, metric: BaseMetric, max_allowed_change: float = 0.01):
        self.metric = metric
        self.max_allowed_change = max_allowed_change

    def check_breach(self, calculation_result: Dict[str, Any]) -> bool:
        """Determines if the calculated lift breaches the guardrail thresholds.

        Args:
            calculation_result (dict): Output from metric.calculate().

        Returns:
            bool: True if there is a breach, False otherwise.
        """
        lift = calculation_result.get("relative_lift", 0.0)
        # Breach occurs if metric deteriorates beyond max_allowed_change
        return abs(lift) > self.max_allowed_change
