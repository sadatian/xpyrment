from typing import Dict, Any, Optional
import numpy as np
from scipy import stats


class ExperimentDesignResult:
    """Class to hold and format experiment design and power analysis results."""

    def __init__(self, details: Dict[str, Any]):
        self.details = details

    def summary(self) -> Dict[str, list]:
        """Returns a summary of the experiment design parameters."""
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
    """Computes the required sample size for an experiment based on design constraints."""
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
    """Generates sample size coordinates across a range of MDE values to plot a power curve."""
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
