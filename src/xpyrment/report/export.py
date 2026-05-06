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
    """Generates a horizontal forest plot visualizing relative lift and confidence intervals."""
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
    """Plots required sample size per variant across a range of Minimum Detectable Effects (MDE)."""
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
