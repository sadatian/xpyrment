"""Statistical visualization and plot generation for experimental reports.

This module provides standard reporting visualizations, including horizontal forest plots
for treatment lifts and confidence intervals, and required sample size curves comparing
standard designs to variance-reduced (CUPED) designs.
"""

from typing import Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_forest(
    df_raw: pd.DataFrame,
    alpha: float = 0.05,
    title: str = "A/B Test Results - Relative Lift & 95% CIs",
    figsize: tuple = (10, 5),
) -> tuple:
    r"""Generates a horizontal forest plot visualizing relative lift and confidence intervals.

    A Forest Plot is the industrial standard for reviewing multiple metrics simultaneously. It displays
    each metric's estimated treatment lift along with its surrounding confidence bounds. This allows rapid,
    visual identification of which metrics experienced significant shifts, whether the shifts are positive
    or negative, and how much uncertainty surrounds each estimate.

    Visual Elements:
        - **Center Dots**: Represent the point estimate of the relative lift ($\\hat{\\theta}$).
        - **Horizontal Bars**: Represent the $1 - \\alpha$ confidence interval ($[\\theta_{\\text{lower}}, \\ \\theta_{\\text{upper}}]$).
        - **Vertical Reference Line**: Placed at $x = 0$ (represented as a dashed red line) to denote the Null Hypothesis
          (no effect). If a metric's horizontal bar does not cross this dashed line, the effect is statistically significant.
        - **Color Coding**: Significant shifts ($p < \\alpha$) are colored in high-contrast teal, while insignificant
          shifts are shaded in neutral slate-grey.

    Args:
        df_raw (pd.DataFrame): A DataFrame containing the statistical summary. Must include the columns:
            - `"metric_name"` (str): Name of the target metric.
            - `"relative_lift"` (float): The point estimate of relative lift.
            - `"rel_ci_lower"` (float): The lower bound of the relative confidence interval.
            - `"rel_ci_upper"` (float): The upper bound of the relative confidence interval.
            - `"p_value"` (float): The calculated p-value of the hypothesis test.
        alpha (float): Nominal significance level used to color-code significance. Defaults to 0.05.
        title (str): Title of the rendered plot. Defaults to `"A/B Test Results - Relative Lift & 95% CIs"`.
        figsize (tuple): Dimensions of the figure canvas. Defaults to `(10, 5)`.

    Returns:
        tuple: A tuple `(fig, ax)` containing:
            - `fig` (matplotlib.figure.Figure): The active matplotlib Figure canvas.
            - `ax` (matplotlib.axes.Axes): The axes container housing the rendered forest plot.
    """
    df = df_raw.copy().sort_values(by="metric_name")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    ax.axvline(0, color="#d32f2f", linestyle="--", linewidth=1.5, label="No Effect")

    y_positions = np.arange(len(df))

    sig_color = "#009688"
    nonsig_color = "#78909c"

    for idx, (_, row) in enumerate(df.iterrows()):
        lift = row["relative_lift"]
        ci_lower = row["rel_ci_lower"]
        ci_upper = row["rel_ci_upper"]
        p_val = row["p_value"]

        is_significant = p_val < alpha
        color = sig_color if is_significant else nonsig_color

        ax.plot([ci_lower, ci_upper], [idx, idx], color=color, linewidth=2.5, zorder=2)

        ax.scatter(
            lift,
            idx,
            color=color,
            s=120,
            edgecolors="black",
            linewidths=1.2,
            zorder=3,
        )

        text_label = f" {lift:+.2%} (p={p_val:.4f})"
        ax.text(
            max(ci_upper, 0) + 0.005,
            idx,
            text_label,
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold" if is_significant else "normal",
            color=color,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["metric_name"], fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Lift (%)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    import matplotlib.ticker as mtick
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    sns.despine(left=True, bottom=True)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax.grid(False, axis="y")

    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min - 0.01, x_max + 0.03)

    plt.tight_layout()
    return fig, ax


def plot_power_curve(
    power_curve_data: Dict[str, np.ndarray],
    title: str = "A/B Test Design - Required Sample Size vs. MDE",
    figsize: tuple = (10, 6),
) -> tuple:
    r"""Plots required sample size per variant across a range of Minimum Detectable Effects (MDE).

    This plotting function illustrates the fundamental trade-off in experimental planning between the
    Minimum Detectable Effect ($\\delta$, MDE) and the required sample size per variant ($N$).
    Because sample size scales quadratically with the inverse of the MDE:
    $$
    N \\propto \\frac{1}{\\delta^2}
    $$
    small increases in the precision requirements (smaller MDE) trigger massive increases in the required sample size.

    Demonstrating CUPED Sample Size Savings:
        If a pre-period covariate is registered, the plot overlays a second curve displaying the required sample size
        when applying CUPED variance reduction.
        - Let $\\rho$ be the correlation coefficient between the pre-period covariate and the post-period outcome.
        - The required sample size under CUPED ($N_{\\text{CUPED}}$) is deflated by a factor of $(1 - \\rho^2)$:
          $$
          N_{\\text{CUPED}} = N_{\\text{standard}} \\times (1 - \\rho^2)
          $$
        - The visual shaded gap between the standard curve and the CUPED curve demonstrates the direct **sample size savings**
          (and consequently, the timeline savings) gained by utilizing pre-period covariate adjustment.

    Args:
        power_curve_data (Dict[str, np.ndarray]): A dictionary containing:
            - `"mde_relative"` (np.ndarray): 1D array of hypothetical MDE percentages.
            - `"sample_size_per_variant"` (np.ndarray): Required sample size under standard Wald designs.
            - `"cuped_sample_size_per_variant"` (np.ndarray, optional): Required sample size under CUPED adjustments.
        title (str): Title of the rendered plot. Defaults to `"A/B Test Design - Required Sample Size vs. MDE"`.
        figsize (tuple): Dimensions of the figure canvas. Defaults to `(10, 6)`.

    Returns:
        tuple: A tuple `(fig, ax)` containing:
            - `fig` (matplotlib.figure.Figure): The active matplotlib Figure canvas.
            - `ax` (matplotlib.axes.Axes): The axes container housing the curves.
    """
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=figsize)

    mde_pct = power_curve_data["mde_relative"]
    standard_n = power_curve_data["sample_size_per_variant"]

    ax.plot(
        mde_pct,
        standard_n,
        color="#e53935",
        linewidth=2.5,
        marker="o",
        markersize=5,
        label="Standard A/B Design",
    )

    if "cuped_sample_size_per_variant" in power_curve_data:
        cuped_n = power_curve_data["cuped_sample_size_per_variant"]
        ax.plot(
            mde_pct,
            cuped_n,
            color="#1e88e5",
            linewidth=2.5,
            marker="s",
            markersize=5,
            label="CUPED Design (Variance Reduced)",
        )

        ax.fill_between(
            mde_pct,
            cuped_n,
            standard_n,
            color="#bbdefb",
            alpha=0.3,
            label="Sample Size Savings via CUPED",
        )

    import matplotlib.ticker as mtick
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda x, p: f"{int(x):,}"))

    ax.set_xlabel("Relative Minimum Detectable Effect (MDE)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Required Sample Size (Per Variant)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    ax.legend(fontsize=11, frameon=True, facecolor="white")
    sns.despine()

    plt.tight_layout()
    return fig, ax
