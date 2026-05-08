r"""Power analysis and sample-size planning calculators.

This module provides industry-standard power calculators for experimental design, helping
experimenters determine the minimum required sample size per variant to detect a target Minimum
Detectable Effect (MDE) with specified Type I and Type II error thresholds ($\alpha, \beta$).
It also handles variance reduction credit (CUPED sample-size deflation) and estimated experiment
runtimes.

??? mathbox "Mathematical Specifications"

    The required sample size per variant $n$ for a two-sample t-test is given by:
    $$
    n = \frac{2 \sigma^2 \left(Z_{1 - \alpha/2} + Z_{1 - \beta}\right)^2}{\delta^2}
    $$
    where:
    - $\sigma^2$: Population variance. For binary proportions ($p$), $\sigma^2 = p(1 - p)$.
    - $Z_{1 - \alpha/2}$: Standard normal critical value for a two-sided test at significance level $\alpha$.
    - $Z_{1 - \beta}$: Standard normal quantile corresponding to the desired statistical power ($1 - \beta$).
    - $\delta$: The target absolute Minimum Detectable Effect (MDE).

    If pre-period baseline covariates are available, the CUPED variance-adjusted sample size is:
    $$
    n_{\text{CUPED}} = n \left(1 - \rho^2\right)
    $$
    where $\rho$ is the correlation between pre-period and experiment-period values.
"""

from typing import Dict, Any, Optional
import numpy as np
from scipy import stats


class ExperimentDesignResult:
    """Class to hold, format, and present experiment design and statistical power analysis results.

    Provides high-fidelity text-based representations and structured summaries of design outputs
    to help experimenters evaluate sizing requirements, potential CUPED savings, and run runtimes.

    Attributes:
        details (Dict[str, Any]): Dictionary containing raw parameter values and sizing outputs
            from the power analysis engine.
    """

    def __init__(self, details: Dict[str, Any]):
        """Initializes an ExperimentDesignResult wrapper.

        Args:
            details (Dict[str, Any]): Dictionary of design details from `design_experiment`.
        """
        self.details = details

    def summary(self) -> Dict[str, list]:
        """Compiles a structured summary of the experiment design parameters.

        Formats raw numbers into clear, readable text elements (e.g., currency, percentages,
        and human-readable sample sizes with digit grouping).

        Returns:
            Dict[str, list]: A dictionary mapping parameter names to formatted value strings.
        """
        summary = {
            "Parameter": [
                "Metric Type",
                "Baseline Value",
                "Target MDE (Absolute)",
                "Target MDE (Relative)",
                "Significance Level (Alpha)",
                "Statistical Power (1-Beta)",
                "Sample Size Per Variant",
                "Total Sample Size Required",
            ],
            "Value": [
                self.details["metric_type"].capitalize(),
                f"{self.details['baseline_value']:.4f}",
                f"{self.details['mde_absolute']:.4f}",
                f"{self.details['mde_relative']:.2%}",
                f"{self.details['alpha']:.2%}",
                f"{self.details['power']:.2%}",
                f"{int(np.ceil(self.details['sample_size_per_variant'])):,}",
                f"{int(np.ceil(self.details['total_sample_size'])):,}",
            ],
        }

        if self.details.get("pre_post_correlation"):
            corr = self.details["pre_post_correlation"]
            reduced_size = self.details["cuped_sample_size_per_variant"]
            summary["Parameter"].extend([
                "Pre-Post Correlation",
                "CUPED Sample Size Per Variant",
                "CUPED Total Sample Size",
                "CUPED Sample Size Savings",
            ])
            summary["Value"].extend([
                f"{corr:.2f}",
                f"{int(np.ceil(reduced_size)):,}",
                f"{int(np.ceil(reduced_size * 2)):,}",
                f"{self.details['cuped_savings']:.1%}",
            ])

        if self.details.get("daily_traffic"):
            summary["Parameter"].extend([
                "Daily Traffic",
                "Estimated Duration (Standard)",
            ])
            summary["Value"].extend([
                f"{int(self.details['daily_traffic']):,}/day",
                f"{self.details['duration_days_standard']:.1f} days",
            ])
            if self.details.get("pre_post_correlation"):
                summary["Parameter"].append("Estimated Duration (CUPED)")
                summary["Value"].append(f"{self.details['duration_days_cuped']:.1f} days")

        return summary

    def __repr__(self) -> str:
        """Generates an aesthetic text block summary of the experiment design parameters."""
        s = "=========================================\n"
        s += "       Experiment Design Summary        \n"
        s += "=========================================\n"
        summary_dict = self.summary()
        for param, val in zip(summary_dict["Parameter"], summary_dict["Value"]):
            s += f"{param:<30}: {val}\n"
        s += "=========================================\n"
        return s


def design_experiment(
    metric_type: str,
    baseline_value: float,
    standard_deviation: Optional[float] = None,
    mde: float = 0.05,
    mde_type: str = "relative",
    alpha: float = 0.05,
    power: float = 0.80,
    pre_post_correlation: Optional[float] = None,
    daily_traffic: Optional[int] = None,
) -> ExperimentDesignResult:
    r"""Computes the required sample size and duration for an experiment based on design constraints.

    This function performs rigorous a priori power analysis to determine required sample sizes.
    It supports continuous means, proportions, and ratio metrics, integrates pre-post correlation
    for CUPED calculation, and maps sizes to daily traffic to compute duration.

    Args:
        metric_type (str): The statistical distribution type. Options are `'mean'`,
            `'proportion'`, or `'ratio'`.
        baseline_value (float): The current historical control group value (mean or rate).
        standard_deviation (Optional[float]): The historical standard deviation of the metric.
            Required for `'mean'` and `'ratio'` metric types. Ignored for `'proportion'`.
        mde (float): The target Minimum Detectable Effect. Expressed as a fraction of baseline for
            `"relative"` (e.g., `0.05` is 5%) or directly as a raw difference for `"absolute"`.
            Defaults to 0.05.
        mde_type (str): Dictates how `mde` is interpreted. Options are `'relative'` or `'absolute'`.
            Defaults to `'relative'`.
        alpha (float): The probability of a Type I error (significance level, e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
        power (float): The desired statistical power ($1 - \beta$, e.g., 0.80 to capture true effects 80% of the time).
            Defaults to 0.80.
        pre_post_correlation (Optional[float]): The correlation coefficient ($\rho$) between baseline pre-period
            values and active experiment-period values. If provided, calculates CUPED-deflated sizing.
            Defaults to None.
        daily_traffic (Optional[int]): Expected daily volume of unique units entering the experiment. If provided,
            calculates duration. Defaults to None.

    Returns:
        ExperimentDesignResult: A wrapper object containing formatted parameters and sample sizing calculations.

    Raises:
        ValueError: If statistical inputs are out of logical bounds (e.g., negative traffic, proportion baseline
            not in $(0, 1)$, or correlation not in $[-1, 1]$).
        ValueError: If standard deviation is missing for mean/ratio metrics.

    Examples:
        ??? example "Example"

            ```python
            >>> # Planning a conversion rate proportion test (10% baseline, relative MDE of 5%)
            >>> result = design_experiment(
            ...     metric_type="proportion",
            ...     baseline_value=0.10,
            ...     mde=0.05,
            ...     mde_type="relative"
            ... )
            >>> int(result.details["sample_size_per_variant"])
            141258
            ```
    """
    metric_type = metric_type.lower()
    mde_type = mde_type.lower()

    if metric_type not in ["mean", "proportion", "ratio"]:
        raise ValueError("metric_type must be one of: 'mean', 'proportion', 'ratio'.")
    if mde_type not in ["relative", "absolute"]:
        raise ValueError("mde_type must be 'relative' or 'absolute'.")

    if mde_type == "relative":
        mde_absolute = baseline_value * mde
        mde_relative = mde
    else:
        mde_absolute = mde
        mde_relative = mde / baseline_value if baseline_value != 0 else 0.0

    if metric_type == "proportion":
        if baseline_value <= 0 or baseline_value >= 1:
            raise ValueError("For proportions, baseline_value must be strictly between 0 and 1.")
        variance = baseline_value * (1 - baseline_value)
    else:
        if standard_deviation is None:
            raise ValueError(f"standard_deviation is required for metric type '{metric_type}'.")
        variance = standard_deviation**2

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    factor = 2 * (z_alpha + z_beta) ** 2
    sample_size = factor * variance / (mde_absolute**2)

    details = {
        "metric_type": metric_type,
        "baseline_value": baseline_value,
        "standard_deviation": standard_deviation if metric_type != "proportion" else np.sqrt(variance),
        "mde_absolute": mde_absolute,
        "mde_relative": mde_relative,
        "alpha": alpha,
        "power": power,
        "sample_size_per_variant": sample_size,
        "total_sample_size": sample_size * 2,
    }

    if pre_post_correlation is not None:
        if not (-1.0 <= pre_post_correlation <= 1.0):
            raise ValueError("pre_post_correlation must be between -1.0 and 1.0.")

        vr_factor = 1.0 - (pre_post_correlation**2)
        cuped_sample_size = sample_size * vr_factor

        details["pre_post_correlation"] = pre_post_correlation
        details["cuped_sample_size_per_variant"] = cuped_sample_size
        details["cuped_savings"] = 1.0 - vr_factor

    if daily_traffic is not None:
        if daily_traffic <= 0:
            raise ValueError("daily_traffic must be positive.")
        details["daily_traffic"] = daily_traffic
        details["duration_days_standard"] = (sample_size * 2) / daily_traffic
        if pre_post_correlation is not None:
            details["duration_days_cuped"] = (cuped_sample_size * 2) / daily_traffic

    return ExperimentDesignResult(details)


def generate_power_curve_data(
    metric_type: str,
    baseline_value: float,
    standard_deviation: Optional[float] = None,
    alpha: float = 0.05,
    power: float = 0.80,
    mde_range: Optional[np.ndarray] = None,
    pre_post_correlation: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Generates sample size coordinates across a range of relative MDE values.

    This function calculates required sizing across a coordinate spectrum of possible MDEs,
    allowing downstream reporting tools to plot an interactive or static "power curve"
    graph (sample size vs. effect size).

    Args:
        metric_type (str): Metric type ('mean', 'proportion', 'ratio').
        baseline_value (float): Historical control average value.
        standard_deviation (Optional[float]): Historical metric standard deviation. Required for continuous.
        alpha (float): Significance level. Defaults to 0.05.
        power (float): Desired statistical power. Defaults to 0.80.
        mde_range (Optional[np.ndarray]): Array of relative MDE points to evaluate. If not provided,
            evaluates 50 linear coordinates in $[0.01, 0.15]$. Defaults to None.
        pre_post_correlation (Optional[float]): Pre-post correlation coefficient for CUPED-adjusted curve.
            Defaults to None.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping coordinate names to numpy arrays of results.
            Contains keys `'mde_relative'` and `'sample_size_per_variant'`, and optionally
            `'cuped_sample_size_per_variant'`.
    """
    if mde_range is None:
        mde_range = np.linspace(0.01, 0.15, 50)

    sample_sizes = []
    cuped_sample_sizes = []

    for mde_val in mde_range:
        res = design_experiment(
            metric_type=metric_type,
            baseline_value=baseline_value,
            standard_deviation=standard_deviation,
            mde=mde_val,
            mde_type="relative",
            alpha=alpha,
            power=power,
            pre_post_correlation=pre_post_correlation,
        )
        sample_sizes.append(res.details["sample_size_per_variant"])
        if pre_post_correlation is not None:
            cuped_sample_sizes.append(res.details["cuped_sample_size_per_variant"])

    ret = {
        "mde_relative": mde_range,
        "sample_size_per_variant": np.array(sample_sizes),
    }

    if pre_post_correlation is not None:
        ret["cuped_sample_size_per_variant"] = np.array(cuped_sample_sizes)

    return ret