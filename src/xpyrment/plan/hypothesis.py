"""Hypothesis specification layer binding metrics to targeted statistical directions.

This module provides the `HypothesisSpec` container, which formalizes the core scientific or
business claim under test, linking it directly to an evaluation metric and establishing the
mathematical directionality (one-sided vs. two-sided) of the downstream inference testing.
"""

from typing import Literal
from xpyrment.metrics.taxonomy import BaseMetric


class HypothesisSpec:
    r"""Specifies the hypothesis under test and binds it to a primary metric.

    This class serves as the formal pre-registered scientific statement of the experiment's goal,
    preventing retrospective hypothesis adjustments after seeing results (HARKing). It defines
    the directional expectation of the treatment effect.

    ### Mathematical Specifications

    Let $\theta_C$ and $\theta_T$ represent the true population parameter (e.g., mean, rate, or ratio)
    of the control and treatment groups, respectively, for the bound `primary_metric`.

    1. **`"two-sided"`** (Default):
       Tests for any difference between arms, representing the standard industrial default.
       $$
       H_0: \theta_T = \theta_C \quad \text{vs.} \quad H_1: \theta_T \neq \theta_C
       $$
    2. **`"greater"`** (One-sided upper-tailed):
       Tests whether treatment is strictly better than control.
       $$
       H_0: \theta_T \le \theta_C \quad \text{vs.} \quad H_1: \theta_T > \theta_C
       $$
    3. **`"less"`** (One-sided lower-tailed):
       Tests whether treatment is strictly worse than control (typically used for testing negative
       side effects or latency increases).
       $$
       H_0: \theta_T \ge \theta_C \quad \text{vs.} \quad H_1: \theta_T < \theta_C
       $$
    Attributes:
        primary_metric (BaseMetric): The registered metric used as the primary outcome variable
            for evaluating this hypothesis.
        direction (Literal["two-sided", "greater", "less"]): The directional mathematical sign of the
            statistical test. Defaults to "two-sided".
        description (str): Text describing the business or scientific hypothesis in natural language.

    Examples:
        ??? example "Example"

            ```python
            >>> from xpyrment.metrics.taxonomy import ProportionMetric
            >>> from xpyrment.plan.hypothesis import HypothesisSpec
            >>> conv_metric = ProportionMetric("Conversion Rate", value_col="converted")
            >>> spec = HypothesisSpec(
            ...     primary_metric=conv_metric,
            ...     direction="greater",
            ...     description="Redesigned checkout button increases conversion rates."
            ... )
            >>> spec.direction
            'greater'
            ```
    """

    def __init__(
        self,
        primary_metric: BaseMetric,
        direction: Literal["two-sided", "greater", "less"] = "two-sided",
        description: str = "",
    ):
        """Initializes a new HypothesisSpec.

        Args:
            primary_metric (BaseMetric): The evaluation metric being tested.
            direction (Literal["two-sided", "greater", "less"]): The statistical test directionality.
                Options are `"two-sided"`, `"greater"`, or `"less"`. Defaults to `"two-sided"`.
            description (str): A descriptive summary of the business hypothesis. Defaults to "".
        """
        self.primary_metric = primary_metric
        self.direction = direction
        self.description = description
