"""P-Curve analysis for governance, power checks, and p-hacking detections.

This module provides the `PCurve` class to analyze distributions of significant
p-values (p < 0.05) for system-level reporting bias and true statistical power.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class PCurve:
    """Analyzes significant p-value distributions to evaluate power and flag gaming/p-hacking.

    # TODO: Implement analytical estimation of the underlying statistical power curve based on non-central distribution fits.
    """

    def __init__(self, p_values: List[float]):
        """Initializes the PCurve analyzer.

        Args:
            p_values (List[float]): A list of all historical experiment p-values.
        """
        raw_p = np.array(p_values)
        # Select only statistically significant p-values (p < 0.05) as required by Simonsohn et al.
        self.significant_p = raw_p[(raw_p >= 0.0) & (raw_p < 0.05)]
        self.n_total = len(self.significant_p)

    def analyze(self) -> Dict[str, Any]:
        """Evaluates right-skewness (true power) and left-skewness (p-hacking / gaming).

        Returns:
            Dict[str, Any]: Counts, skewness indicators, test p-values, and warning alerts.
        """
        if self.n_total < 5:
            return {
                "n_significant": self.n_total,
                "status": "Inconclusive",
                "message": "Too few significant p-values to evaluate distribution skewness (minimum 5 required).",
                "p_right_skew": 1.0,
                "p_left_skew": 1.0,
            }

        # Count p-values in [0.0, 0.025]
        n_low = int(np.sum(self.significant_p <= 0.025))
        n_high = self.n_total - n_low

        # 1. Test for Right-Skewness (True Evidential Power)
        # H0: Significant p-values are uniformly distributed (p = 0.5 for low half).
        # H1: Distribution is right-skewed (proportion in low half > 0.5).
        p_right_skew = 1.0 - stats.binom.cdf(n_low - 1, self.n_total, 0.5)

        # 2. Test for Left-Skewness (p-hacking / selective stopping gaming)
        # H1: Distribution is left-skewed (proportion in high half > 0.5).
        p_left_skew = stats.binom.cdf(n_low, self.n_total, 0.5)

        # Determine status and alerts
        if p_left_skew < 0.05:
            status = "Reporting Bias / P-Hacking Detected"
            message = "The distribution of significant p-values is significantly left-skewed. This is a classic indicator of publication bias or selective peeking and stopping."
        elif p_right_skew < 0.05:
            status = "Strong Evidential Value"
            message = "The distribution of significant p-values is right-skewed, demonstrating high true statistical power and valid experimental effects."
        else:
            status = "Uniform / Flat (Lack of Power)"
            message = "The significant p-values are uniformly distributed, suggesting low evidential power or a high proportion of false positives."

        return {
            "n_significant": self.n_total,
            "n_low_half": n_low,
            "n_high_half": n_high,
            "ratio_low_to_high": float(n_low / n_high) if n_high > 0 else float("inf"),
            "p_right_skew": float(p_right_skew),
            "p_left_skew": float(p_left_skew),
            "status": status,
            "message": message,
        }


    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        title: str = "P-Curve Analysis",
        figsize: Tuple[int, int] = (8, 6),
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Generates a P-Curve visualization based on the stored significant p-values.

        Args:
            ax (matplotlib.axes.Axes, optional): Pre-existing axes for the plot. If None, a new figure
                and axes are created.
            title (str): Title of the rendered plot. Defaults to "P-Curve Analysis".
            figsize (Tuple[int, int]): Dimensions of the figure canvas. Defaults to (8, 6).
            **kwargs: Additional keyword arguments to pass to the underlying plot function.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The generated matplotlib Figure and Axes.
        """
        return plot_p_curve(
            p_values=self.significant_p.tolist(),
            ax=ax,
            title=title,
            figsize=figsize,
            **kwargs
        )


def plot_p_curve(
    p_values: List[float],
    ax: Optional[plt.Axes] = None,
    title: str = "P-Curve Analysis",
    figsize: Tuple[int, int] = (8, 6),
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Visualizes the density of significant p-values against a uniform null curve.

    Args:
        p_values (List[float]): A list or array of all historical experiment p-values.
        ax (matplotlib.axes.Axes, optional): Pre-existing axes for the plot. If None, a new figure
            and axes are created.
        title (str): Title of the rendered plot.
        figsize (Tuple[int, int]): Dimensions of the figure canvas.
        **kwargs: Additional keyword arguments to pass to the plot.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The generated matplotlib Figure and Axes.
    """
    raw_p = np.array(p_values)
    sig_p = raw_p[(raw_p >= 0.0) & (raw_p < 0.05)]

    if ax is None:
        fig, ax_to_use = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        ax_to_use = ax

    # Bins for p-curve: 0.01, 0.02, 0.03, 0.04, 0.05
    bins = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    bin_labels = ["0.01", "0.02", "0.03", "0.04", "0.05"]

    if len(sig_p) == 0:
        counts = [0, 0, 0, 0, 0]
        percentages = [0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        counts, _ = np.histogram(sig_p, bins=bins)
        percentages = (counts / len(sig_p)) * 100.0

    df_plot = pd.DataFrame({
        "p-value": bin_labels,
        "Percentage": percentages
    })

    # Use a temporary context to apply seaborn styling locally
    with sns.axes_style("whitegrid"):
        # Plot observed p-curve
        sns.lineplot(
            data=df_plot,
            x="p-value",
            y="Percentage",
            marker="o",
            color="#1e88e5",
            linewidth=2.5,
            markersize=8,
            label="Observed P-Curve",
            ax=ax_to_use,
            **kwargs
        )

        # Plot uniform null
        ax_to_use.axhline(
            20.0,
            color="#e53935",
            linestyle="--",
            linewidth=2,
            label="Uniform Null (No Effect)"
        )

        ax_to_use.set_ylim(0, max(100, max(percentages) + 10 if len(sig_p) > 0 else 100))
        ax_to_use.set_xlabel("p-value", fontweight="bold")
        ax_to_use.set_ylabel("Percentage of p-values (%)", fontweight="bold")
        ax_to_use.set_title(title, fontweight="bold", pad=15)
        ax_to_use.legend(frameon=True, facecolor="white")

    if ax is None:
        fig.tight_layout()

    return fig, ax_to_use
