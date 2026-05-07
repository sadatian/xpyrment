"""Practical vs. statistical significance evaluations and MVE boundaries.

This module provides functions to verify whether statistically significant results possess
sufficient physical or economic magnitude to be considered of practical business value.
"""


def check_practical_significance(relative_lift: float, min_valuable_effect: float) -> bool:
    """Verifies if the measured lift satisfies the minimal valuable business effect (MVE).

    Online experimentation platforms often have massive sample sizes, which makes them highly powered.
    As a result, extremely microscopic differences (e.g., a $0.05\\%$ lift in page load times) can yield highly
    significant p-values ($p < 0.01$). However, such a minor improvement may be practically or economically
    irrelevant, failing to justify the ongoing maintenance overhead of the new code.
    This function evaluates whether the observed effect size meets or exceeds a pre-defined practical threshold.

    Statistical Significance vs. Practical Significance:
        Let $\\hat{\\theta}$ be the estimated treatment effect, let $[\\theta_{\\text{lower}}, \\ \\theta_{\\text{upper}}]$
        be its confidence interval, and let $\\delta_{\\text{MVE}}$ be the Minimum Valuable Effect (MVE).

        Three scenarios can occur when evaluating significance:
        1. **Statistically Significant and Practically Significant**:
           The null hypothesis is rejected ($p < \\alpha$), and the estimated lift exceeds the MVE:
           $$\\hat{\\theta} \\ge \\delta_{\\text{MVE}} \\quad \\text{and} \\quad p < \\alpha$$
           (Ideally, to be highly confident, we require the entire confidence interval to exceed the threshold: $\\theta_{\\text{lower}} \\ge \\delta_{\\text{MVE}}$).
        2. **Statistically Significant but NOT Practically Significant**:
           The null hypothesis is rejected ($p < \\alpha$), but the magnitude is trivial:
           $$\\hat{\\theta} < \\delta_{\\text{MVE}} \\quad \\text{and} \\quad p < \\alpha$$
           In this case, the feature should generally be rejected despite its "significant" p-value.
        3. **Practically Significant but NOT Statistically Significant**:
           The estimated point estimate is large ($\\hat{\\theta} \\ge \\delta_{\\text{MVE}}$), but we fail to reject the null hypothesis ($p \\ge \\alpha$).
           This indicates an underpowered experiment; the sample size was too small to confirm whether the apparent large effect is real or due to noise.

    Args:
        relative_lift (float): The estimated relative difference between treatment and control (point estimate).
        min_valuable_effect (float): The minimum relative lift required to be practically valuable ($\\delta_{\\text{MVE}}$).

    Returns:
        bool: True if the estimated relative lift meets or exceeds the minimum valuable effect threshold.
    """
    return relative_lift >= min_valuable_effect
