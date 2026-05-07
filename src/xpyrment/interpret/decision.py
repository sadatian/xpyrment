"""Automated product decision logic and feature launch recommendations.

This module provides recommendation engines, such as `generate_launch_recommendation`, which combine
statistical significance with business economics (deployment costs, margins) to guide deployment decisions.
"""


def generate_launch_recommendation(p_value: float, relative_lift: float, cost_threshold: float = 0.0) -> str:
    """Generates automated ship, no-ship, or inconclusive launch recommendations based on statistical and economic bounds.

    Translates statistical estimates and uncertainty intervals into actionable product decisions.
    Importantly, a statistically significant positive effect is not always sufficient to justify a product launch.
    Every feature introduces operational overhead, maintenance costs, and potential technical debt.
    Therefore, the decision engine incorporates an economic cost threshold ($C$) to evaluate profitability.

    Mathematical Decision Boundaries:
        Let $\\hat{\\theta}$ be the estimated relative treatment effect (relative_lift), let $[\\theta_{\\text{lower}}, \\ \\theta_{\\text{upper}}]$
        be the $1 - \\alpha$ confidence interval, and let $C$ be the minimum acceptable relative improvement (cost_threshold)
        required to offset the feature's operational costs.

        The recommendation engine maps these boundaries to four distinct decision states:
        1. **SHIP**:
           The treatment effect is statistically significant ($p < \\alpha$), and the estimated lift exceeds the cost threshold:
           $$\\hat{\\theta} > C \\quad \\text{and} \\quad p < \\alpha$$
           (For a highly conservative strategy, we can assert that the worst-case benefit exceeds costs: $\\theta_{\\text{lower}} \\ge C$).
        2. **NO-SHIP (Uneconomic)**:
           The treatment effect is statistically significant ($p < \\alpha$), but the benefit is too small to justify the
           operational overhead:
           $$\\hat{\\theta} \\le C \\quad \\text{and} \\quad p < \\alpha$$
        3. **INCONCLUSIVE (Underpowered)**:
           There is insufficient statistical evidence to reject the null hypothesis of no effect:
           $$p \\ge \\alpha$$
           This occurs when the sample size was too small to resolve the treatment effect, or if the true treatment effect
           is actually zero.

    Args:
        p_value (float): The calculated p-value of the primary metric test.
        relative_lift (float): The estimated relative difference between treatment and control ($\\bar{Y}_T - \\bar{Y}_C) / \\bar{Y}_C$.
        cost_threshold (float): The minimum relative benefit required to warrant deployment ($C$). Defaults to 0.0.

    Returns:
        str: A structured recommendation string outlining the statistical and economic rationale.
    """
    if p_value < 0.05:
        if relative_lift > cost_threshold:
            return "SHIP: Lifts are statistically significant and exceed deployment costs."
        else:
            return "NO-SHIP: Statistically significant but falls below economic margins."
    return "INCONCLUSIVE: No statistical evidence to assert a positive lift."

