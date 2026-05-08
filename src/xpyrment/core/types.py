"""Core type definitions, TypeDicts, and Literals for the xpyrment library.

This module houses all static typing definitions, data schemas, and literal constraints shared
across the xpyrment library. It provides strict interfaces for statistical outputs to ensure
perfect alignment between metrics, validation engines, statistical routers, and report generation.
"""

from typing import Literal, TypedDict

MetricType = Literal["mean", "proportion", "ratio", "revenue"]
"""Literal representing the supported category of metrics.

Supported Types:
    - `"mean"`: A continuous or discrete numeric metric where statistics are calculated on a per-unit basis
      (e.g., average sessions per user, average page views).
    - `"proportion"`: A binary rate metric representing yes/no outcomes on a per-unit basis, equivalent to a Bernoulli
      trial (e.g., conversion rate, click-through-rate where the unit of analysis is the user).
    - `"ratio"`: An aggregated metric computed as the sum of a numerator divided by the sum of a denominator
      across all units (e.g., global Click-Through-Rate = total clicks / total impressions). Requires Delta Method
      for proper variance approximation.
    - `"revenue"`: A highly skewed continuous monetary metric (e.g., revenue per user, average order value). Often
      subject to log-transformations or specialized variance reduction.
"""


class MetricResult(TypedDict):
    r"""The canonical data schema representing the output of a statistical metric analysis.

    This TypedDict establishes a contract for all inference engines (frequentist, Bayesian,
    and sequential) and reporting utilities, ensuring that every calculated metric contains
    both descriptive statistics and rigorous statistical validation metrics.

    Attributes:
        metric_name (str): The unique identifier assigned to the analyzed metric.
        metric_type (str): The standardized type string (e.g., "Mean", "Proportion", "Ratio", "Revenue").
        control_mean (float): The sample mean ($\bar{Y}_C$) or proportion ($p_C$) calculated for the control group.
        treatment_mean (float): The sample mean ($\bar{Y}_T$) or proportion ($p_T$) calculated for the treatment group.
        control_var (float): The sample variance ($s^2_C$) calculated for the control group.
            For ratios, this represents the Delta-method approximated variance.
        treatment_var (float): The sample variance ($s^2_T$) calculated for the treatment group.
            For ratios, this represents the Delta-method approximated variance.
        control_n (int): The total count of unique units in the control group ($N_C$).
        treatment_n (int): The total count of unique units in the treatment group ($N_T$).
        absolute_difference (float): The point estimate of the absolute treatment effect:
            $$
            \Delta = \bar{Y}_T - \bar{Y}_C
            $$
        relative_lift (float): The percentage increase or decrease of the treatment mean relative to the control mean:
            $$
            \text{Lift} = \frac{\bar{Y}_T - \bar{Y}_C}{\bar{Y}_C}
            $$
        cuped_applied (bool): True if Controlled-comparison Using Pre-Existing Data (CUPED) was applied
            to adjust the variance of this metric. False otherwise.
        variance_reduction (float): The percentage reduction in variance achieved by CUPED, bounded in $[0, 1)$:
            $$
            \text{Reduction} = 1 - \frac{\text{Var}(Y_{\text{CUPED}})}{\text{Var}(Y_{\text{original}})}
            $$
        p_value (float): The statistical p-value associated with the hypothesis test. For frequentist, this represents the
            probability of observing a test statistic at least as extreme as the one computed, under the null hypothesis ($H_0$).
        ci_lower (float): The lower bound of the absolute confidence/credible interval at the $(1 - \alpha)$ confidence level.
        ci_upper (float): The upper bound of the absolute confidence/credible interval at the $(1 - \alpha)$ confidence level.
        rel_ci_lower (float): The lower bound of the relative confidence/credible interval, scaled relative to the control mean:
            $$
            \text{Rel CI Lower} = \frac{\text{CI Lower}}{\bar{Y}_C}
            $$
        rel_ci_upper (float): The upper bound of the relative confidence/credible interval, scaled relative to the control mean:
            $$
            \text{Rel CI Upper} = \frac{\text{CI Upper}}{\bar{Y}_C}
            $$
        power (float): The statistical power ($1 - \beta$) achieved by the sample size, denoting the probability of
            correctly rejecting the null hypothesis when the true treatment effect equals the observed difference.
    """

    metric_name: str
    metric_type: str
    control_mean: float
    treatment_mean: float
    control_var: float
    treatment_var: float
    control_n: int
    treatment_n: int
    absolute_difference: float
    relative_lift: float
    cuped_applied: bool
    variance_reduction: float
    p_value: float
    ci_lower: float
    ci_upper: float
    rel_ci_lower: float
    rel_ci_upper: float
    power: float
