"""Visualizations and plot generation utilities for multi-factor interactions.

This module provides standard plotting wrappers, such as `plot_interaction_heatmap`, to map
the presence and magnitude of interactions across high-dimensional experimental spaces.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_interaction_heatmap(df_interactions: pd.DataFrame) -> tuple:
    r"""Generates an interaction term heatmap using matplotlib.

    Visualizes a symmetric matrix of feature/factor interactions. Heatmaps are a highly effective diagnostic
    chart for screening complex multi-factor studies or high-dimensional covariate sets, allowing the user
    to instantly recognize clusters of strong synergy or severe interference.

    Matrix Structure:
        Let $F = \{f_1, f_2, \dots, f_m\}$ be the set of analyzed factors or covariates. The plotting engine constructs
        a symmetric $m \times m$ matrix $H$:
        - Cell $H_{i,j}$ contains the strength of the interaction between $f_i$ and $f_j$. This value can represent
          either:
          1. The absolute regression interaction coefficient ($|\beta_{\text{interaction}}|$).
          2. The model-agnostic Friedman's H-statistic ($H_{ij}$).
          3. The statistical significance transformed index ($-\log_{10}(p_{\text{value}})$).
        - Cells along the diagonal ($H_{i,i}$) are typically zeroed or set to represent the main effect of factor $f_i$.
        - The matrix is rendered using a divergent colormap (such as `RdBu` or `seismic` if mapping positive/negative coefficients)
          or a sequential colormap (such as `Viridis` or `YlOrRd` if mapping absolute H-statistics or significance).

    Args:
        df_interactions (pd.DataFrame): A rectangular or pivoted DataFrame representing the interaction strength matrix,
            with factor names as both index and column headings.

    Returns:
        tuple: A tuple `(fig, ax)` containing:
            - `fig` (matplotlib.figure.Figure): The active matplotlib Figure canvas.
            - `ax` (matplotlib.axes.Axes): The axes container housing the rendered heatmap.
    """
    # Determine the appropriate colormap based on the data values
    has_negative_values = (df_interactions.to_numpy() < 0).any()
    cmap = "RdBu_r" if has_negative_values else "YlOrRd"

    # Calculate a dynamic figure size based on the number of features
    num_features = df_interactions.shape[0]
    fig_size = max(8, min(24, num_features * 1.5))

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))

    # Render the heatmap
    sns.heatmap(
        df_interactions,
        annot=True,
        fmt=".3g",
        cmap=cmap,
        ax=ax,
        square=True,
        cbar_kws={'label': 'Interaction Strength'},
        center=0 if has_negative_values else None
    )

    # Add title and adjust layout
    ax.set_title("Factor Interaction Heatmap", pad=20, fontsize=14, fontweight="bold")
    ax.set_xlabel("Factor", fontweight="bold")
    ax.set_ylabel("Factor", fontweight="bold")
    fig.tight_layout()

    return fig, ax


def plot_interaction_effects(data: pd.DataFrame, treatment_col: str, metric_col: str, covariate_col: str) -> tuple:
    """Plots interaction effects between a treatment and a covariate on a metric.

    Generates a line plot showing the average metric value for different treatment
    groups across levels of the covariate. This helps visualize if the treatment
    effect varies depending on the covariate value (heterogeneous treatment effect).

    Example:
        ```python
        from xpyrment.interactions import plot_interaction_effects

        plot_interaction_effects(data, "treatment", "revenue", "country")
        ```

    Args:
        data (pd.DataFrame): The experimental data containing treatments, covariates, and metrics.
        treatment_col (str): The name of the column representing the treatment group.
        metric_col (str): The name of the column representing the outcome metric.
        covariate_col (str): The name of the column representing the interacting covariate.

    Returns:
        tuple: A tuple `(fig, ax)` containing the matplotlib Figure and Axes objects.
    """
    num_treatments = data[treatment_col].nunique()

    # Check if covariate is numeric or categorical
    if pd.api.types.is_numeric_dtype(data[covariate_col]) and data[covariate_col].nunique() > 10:
        # For continuous numeric covariates, bin them into quantiles or use a scatter plot with regression lines
        g = sns.lmplot(
            data=data,
            x=covariate_col,
            y=metric_col,
            hue=treatment_col,
            height=6,
            aspect=1.5,
            scatter_kws={"alpha": 0.5}
        )
        fig = g.figure
        ax = g.ax
    else:
        # For categorical or discrete covariates, use a point plot (interaction plot)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Ensure we have enough markers and linestyles for all treatment groups
        all_markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "+", "x", "X", "d", "|", "_"]
        all_linestyles = ["-", "--", "-.", ":"] * ((num_treatments // 4) + 1)

        sns.pointplot(
            data=data,
            x=covariate_col,
            y=metric_col,
            hue=treatment_col,
            dodge=True,
            markers=all_markers[:num_treatments],
            linestyles=all_linestyles[:num_treatments],
            ax=ax
        )

    ax.set_title(f"Interaction Effect of {treatment_col} and {covariate_col} on {metric_col}", pad=15)
    ax.set_xlabel(covariate_col.capitalize())
    ax.set_ylabel(metric_col.capitalize())

    fig.tight_layout()
    return fig, ax
