"""Experiment analysis orchestrator, results compiler, and setup entrypoints.

This module provides the central user-facing API for launching analyses on experimental datasets.
It coordinates the execution of registered metrics, handles multiple testing corrections, manages state
transitions, and constructs the unified `AnalysisResult` data layer for plotting and reporting.
"""

from typing import Any, List, Optional
import pandas as pd

from xpyrment.core.state import ExperimentState
from xpyrment.core.experiment import Experiment
from xpyrment.analyze.corrections import apply_multiple_testing_correction


class AnalysisResult:
    """Holds results from an experiment analysis and provides summary formatting and plotting interfaces.

    This container aggregates the individual metric dictionaries calculated across control and treatment
    groups. It provides high-level APIs to compile clean summary tables and forward coordinates to the
    visualization engine.

    Attributes:
        raw_results (List[dict]): A list of metric calculation result dictionaries (keys: mean, lift, p_value, etc.).
        alpha (float): Nominal significance level (Type I error rate) used in the analysis. Defaults to 0.05.
        df_raw (pd.DataFrame): The raw, unformatted results compiled into a pandas DataFrame.
    """

    def __init__(self, raw_results: List[dict], alpha: float = 0.05):
        """Initializes an AnalysisResult.

        Args:
            raw_results (List[dict]): Raw list of metric results.
            alpha (float): Nominal significance level used.
        """
        self.raw_results = raw_results
        self.alpha = alpha
        self.df_raw = pd.DataFrame(raw_results)

    def summary(self, formatted: bool = True) -> pd.DataFrame:
        r"""Returns a summarized, human-readable DataFrame of the analysis.

        Formats raw numeric statistics (standard errors, differences, variances) into readable
        percentage lifts, relative confidence intervals, power indicators, and significance star symbols.

        Significance Star Mapping:
            - `***` : $p < 0.001$ (Highly significant)
            - `**`  : $p < 0.01$ (Significant)
            - `*`   : $p < 0.05$ (Significant)
            - No star : $p \ge 0.05$ (Not statistically significant at the nominal level $\alpha=0.05$)

        Args:
            formatted (bool): If True, returns nicely formatted strings for display (with percentage symbols,
                stars, and bracketed intervals). If False, returns the raw numeric values. Defaults to True.

        Returns:
            pd.DataFrame: A pandas DataFrame containing binned summaries of each analyzed metric.
        """
        df = self.df_raw.copy()

        if not formatted:
            return df

        summary_data = []
        for _, row in df.iterrows():
            lift_val = row["relative_lift"]
            lift_str = f"{lift_val:+.2%}" if not pd.isna(lift_val) else "N/A"

            p_val = row["p_value"]
            sig_symbol = ""
            if p_val < 0.001:
                sig_symbol = "***"
            elif p_val < 0.01:
                sig_symbol = "**"
            elif p_val < 0.05:
                sig_symbol = "*"

            p_str = f"{p_val:.4f}{sig_symbol}" if not pd.isna(p_val) else "N/A"

            lower_pct = row["rel_ci_lower"]
            upper_pct = row["rel_ci_upper"]
            ci_str = f"[{lower_pct:+.2%}, {upper_pct:+.2%}]" if not (pd.isna(lower_pct) or pd.isna(upper_pct)) else "N/A"

            power_val = row["power"]
            power_str = f"{power_val:.1%}" if not pd.isna(power_val) else "N/A"

            cuped_str = "Yes" if row["cuped_applied"] else "No"
            var_red_val = row["variance_reduction"]
            var_red_str = f"{var_red_val:.1%}" if row["cuped_applied"] and not pd.isna(var_red_val) else "-"

            summary_data.append(
                {
                    "Metric": row["metric_name"],
                    "Type": row["metric_type"],
                    "Control Mean": f"{row['control_mean']:.4f}",
                    "Treatment Mean": f"{row['treatment_mean']:.4f}",
                    "Relative Lift": lift_str,
                    "95% CI (Rel)": ci_str,
                    "p-value": p_str,
                    "Post-hoc Power": power_str,
                    "CUPED": cuped_str,
                    "Var Reduction": var_red_str,
                }
            )

        return pd.DataFrame(summary_data)

    def plot(self, **kwargs: Any) -> Any:
        """Generates and returns a forest plot of the relative metric lifts and confidence intervals.

        Forwards coordinates to the visualization module.

        Args:
            **kwargs: Plot customization arguments forwarded to `plot_forest` (e.g., figure size, colors).

        Returns:
            matplotlib.axes.Axes or plotly.graph_objects.Figure: The generated relative lift forest plot.
        """
        # Re-routed to the reporting/export layer dynamically
        from xpyrment.report.export import plot_forest
        return plot_forest(self.df_raw, alpha=self.alpha, **kwargs)


def run_analysis(
    experiment: Experiment,
    control: str = "control",
    treatment: str = "treatment",
    alpha: float = 0.05,
    multi_test_correction: Optional[str] = None,
) -> AnalysisResult:
    """Executes the statistical analysis across all registered metrics in an Experiment container.

    Iterates over each registered metric in the experiment, calculates means, relative lifts, p-values,
    confidence intervals, and power. If requested, applies multiple testing corrections across the p-values,
    updates the experiment state to `ANALYZED`, and returns a structured `AnalysisResult`.

    Mathematical Logic Flow:
        1. Validates that the experiment is currently in `ExperimentState.COLLECTED` or a compatible state.
        2. Asserts that the dataset contains the designated `control` and `treatment` variant arms.
        3. For each registered metric in `experiment.metrics`:
           - Runs `metric.calculate()`, computing group statistics, delta method variances, and test outcomes.
        4. If `multi_test_correction` is specified, extracts all p-values and applies adjustments
           (e.g., Benjamini-Hochberg FDR) before writing adjusted values back to results.
        5. Performs the programmatic transition:
           `experiment.transition_to(ExperimentState.ANALYZED)`
        6. Wraps and returns results in an `AnalysisResult` instance.

    Args:
        experiment (Experiment): The initialized, pre-registered experiment setup container.
        control (str): The label of the control variant in the treatment column. Defaults to `"control"`.
        treatment (str): The label of the treatment variant in the treatment column. Defaults to `"treatment"`.
        alpha (float): Significance level (Type I error probability) for confidence intervals. Defaults to 0.05.
        multi_test_correction (str, optional): Multiple testing correction algorithm to apply across the
            registered metrics. Options: `"bonferroni"`, `"holm"`, `"fdr_bh"`. Defaults to None.

    Returns:
        AnalysisResult: A rich, summarized results container.

    Raises:
        ValueError: If no metrics have been registered, or if control/treatment labels are missing from
            the active dataset.
        PhaseOrderError: If the experiment is in an invalid state for running analysis.
    """
    if not experiment.metrics:
        raise ValueError("No metrics have been added to the experiment.")

    unique_variants = experiment.data[experiment.treatment_col].unique()
    if control not in unique_variants:
        raise ValueError(f"Control label '{control}' not found.")
    if treatment not in unique_variants:
        raise ValueError(f"Treatment label '{treatment}' not found.")

    results = []
    for metric in experiment.metrics:
        res = metric.calculate(
            experiment.data,
            treatment_col=experiment.treatment_col,
            control=control,
            treatment=treatment,
            alpha=alpha,
        )
        results.append(res)

    # Apply corrections if requested
    if multi_test_correction and len(results) > 1:
        p_vals = [res["p_value"] for res in results]
        adjusted_p = apply_multiple_testing_correction(p_vals, alpha=alpha, method=multi_test_correction)
        for i, val in enumerate(adjusted_p):
            results[i]["p_value"] = val

    experiment.transition_to(ExperimentState.ANALYZED)
    return AnalysisResult(results, alpha=alpha)


def setup(
    data: pd.DataFrame,
    treatment_col: str,
    id_col: Optional[str] = None,
) -> Experiment:
    """Initializes the experimental setup container, serving as the library's primary entrypoint.

    Sets up the `Experiment` object with the target dataset, identifying variant and unit columns,
    and locks the state machine to `ExperimentState.DESIGNED`.

    Args:
        data (pd.DataFrame): The main experiment dataset containing exposure logs and outcomes.
        treatment_col (str): Column name containing variant strings (e.g., `"variant"`).
        id_col (str, optional): Column name containing unique unit identifiers (e.g., `"user_id"`).

    Returns:
        Experiment: A state-gated `Experiment` orchestrator instance, ready for metric registration and planning.
    """
    print("==========================================")
    print("      Initializing xpyrment Setup         ")
    print("==========================================")
    print(f"Total rows in dataset:  {len(data)}")
    print(f"Treatment column:      {treatment_col}")
    if id_col:
        print(f"ID column:             {id_col}")

    exp = Experiment(data, treatment_col, id_col)
    return exp
