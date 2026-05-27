"""Visualizations and plot generation utilities for multi-factor interactions.

This module provides standard plotting wrappers, such as `plot_interaction_heatmap`, to map
the presence and magnitude of interactions across high-dimensional experimental spaces.
"""

from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def plot_interaction_heatmap(
    df_interactions: pd.DataFrame,
    annot: Optional[bool] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> tuple:
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
        annot (bool, optional): Whether to annotate the cells with numeric values. If None,
            annotations are enabled automatically only if the matrix size is small (e.g., <= 20 features).
        ax (matplotlib.axes.Axes, optional): Pre-existing axes for the plot. If None, a new figure
            and axes are created.
        **kwargs: Additional keyword arguments to pass to `seaborn.heatmap`.

    Returns:
        tuple: A tuple `(fig, ax)` containing:
            - `fig` (matplotlib.figure.Figure): The active matplotlib Figure canvas.
            - `ax` (matplotlib.axes.Axes): The axes container housing the rendered heatmap.
    """
    # Validate symmetric square matrix
    if df_interactions.shape[0] != df_interactions.shape[1]:
        raise ValueError(f"Interaction matrix must be square, got shape {df_interactions.shape}")
    if list(df_interactions.index) != list(df_interactions.columns):
        raise ValueError("Interaction matrix must have matching index and columns")

    num_features = df_interactions.shape[0]

    # Determine whether to annotate cells
    if annot is None:
        # Heuristic: only annotate for smaller matrices to keep the plot readable and fast
        annot_flag = num_features <= 20
    else:
        annot_flag = annot

    # Determine the appropriate colormap based on the numeric data values
    numeric_df = df_interactions.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise TypeError("Interaction matrix must contain numeric columns to plot heatmap.")

    has_negative_values = (numeric_df.to_numpy() < 0).any()
    cmap = kwargs.pop("cmap", "RdBu_r" if has_negative_values else "YlOrRd")

    if ax is None:
        # Calculate a dynamic figure size based on the number of features
        fig_size = max(8, min(24, num_features * 1.5))
        fig, ax_to_use = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    else:
        fig = ax.figure
        ax_to_use = ax

    # Calculate symmetric vmin/vmax for divergent colormaps
    heatmap_kwargs = {
        "annot": annot_flag,
        "fmt": ".3g",
        "cmap": cmap,
        "ax": ax_to_use,
        "square": True,
        "cbar_kws": {'label': 'Interaction Strength'}
    }

    if has_negative_values:
        max_abs = np.abs(numeric_df.to_numpy()).max()
        heatmap_kwargs["vmin"] = -max_abs
        heatmap_kwargs["vmax"] = max_abs
        heatmap_kwargs["center"] = 0
    else:
        heatmap_kwargs["vmin"] = 0

    heatmap_kwargs.update(kwargs)

    # Render the heatmap
    sns.heatmap(df_interactions, **heatmap_kwargs)

    # Add title and adjust layout
    ax_to_use.set_title("Factor Interaction Heatmap", pad=20, fontsize=14, fontweight="bold")
    ax_to_use.set_xlabel("Factor", fontweight="bold")
    ax_to_use.set_ylabel("Factor", fontweight="bold")

    if ax is None:
        fig.tight_layout()

    return fig, ax_to_use


def plot_interaction_effects(
    data: pd.DataFrame,
    treatment_col: str,
    metric_col: str,
    covariate_col: str,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> tuple:
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
        ax (matplotlib.axes.Axes, optional): Pre-existing axes for the plot. If None, a new figure
            and axes are created.
        **kwargs: Additional keyword arguments to pass to the underlying plotting functions.

    Returns:
        tuple: A tuple `(fig, ax)` containing the matplotlib Figure and Axes objects.
    """
    num_treatments = data[treatment_col].nunique()

    if ax is None:
        fig, ax_to_use = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
        ax_to_use = ax

    # Drop NaNs to prevent plotting issues
    plot_data = data.dropna(subset=[treatment_col, metric_col, covariate_col]).copy()

    # Check if covariate is numeric or categorical
    if pd.api.types.is_numeric_dtype(plot_data[covariate_col]) and plot_data[covariate_col].nunique() > 10:
        # For continuous numeric covariates, use a scatter plot with regression lines

        # Get default seaborn color palette
        palette = sns.color_palette(n_colors=num_treatments)
        treatments = sorted(plot_data[treatment_col].unique())

        for idx, treatment in enumerate(treatments):
            group_data = plot_data[plot_data[treatment_col] == treatment]

            # Combine default and provided kwargs
            regplot_kwargs = {
                "scatter_kws": {"alpha": 0.5},
                "label": str(treatment),
                "color": palette[idx]
            }
            regplot_kwargs.update(kwargs)

            sns.regplot(
                data=group_data,
                x=covariate_col,
                y=metric_col,
                ax=ax_to_use,
                **regplot_kwargs
            )

        ax_to_use.legend(title=treatment_col)
    else:
        # For categorical or discrete covariates, use a point plot (interaction plot)

        # Ensure we have enough markers and linestyles for all treatment groups
        all_markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "+", "x", "X", "d", "|", "_"]
        all_linestyles = ["-", "--", "-.", ":"] * ((num_treatments // 4) + 1)

        pointplot_kwargs = {
            "dodge": True,
            "markers": all_markers[:num_treatments],
            "linestyles": all_linestyles[:num_treatments]
        }
        pointplot_kwargs.update(kwargs)

        sns.pointplot(
            data=plot_data,
            x=covariate_col,
            y=metric_col,
            hue=treatment_col,
            ax=ax_to_use,
            **pointplot_kwargs
        )

    ax_to_use.set_title(f"Interaction Effect of {treatment_col} and {covariate_col} on {metric_col}", pad=15)
    ax_to_use.set_xlabel(covariate_col.capitalize())
    ax_to_use.set_ylabel(metric_col.capitalize())

    if ax is None:
        fig.tight_layout()

    return fig, ax_to_use
