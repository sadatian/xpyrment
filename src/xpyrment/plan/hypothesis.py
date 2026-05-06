from typing import Literal
from xpyrment.metrics.taxonomy import BaseMetric


class HypothesisSpec:
    """Specifies the hypothesis under test and binds it to a primary metric."""

    def __init__(
        self,
        primary_metric: BaseMetric,
        direction: Literal["two-sided", "greater", "less"] = "two-sided",
        description: str = "",
    ):
        self.primary_metric = primary_metric
        self.direction = direction
        self.description = description
