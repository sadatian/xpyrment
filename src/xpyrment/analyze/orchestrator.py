"""Experiment analysis orchestrator, results compiler, and setup entrypoints.

This module provides the central user-facing API for launching analyses on experimental datasets.
It coordinates the execution of registered metrics, handles multiple testing corrections, manages state
transitions, and constructs the unified `AnalysisResult` data layer for plotting and reporting.
"""

import warnings
from typing import Any, List, Optional
import numpy as np
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
        balance_checker (Optional[Any]): Fitted balance checker object if covariates were present.
    """

    def __init__(self, raw_results: List[dict], alpha: float = 0.05, balance_checker: Optional[Any] = None):
        """Initializes an AnalysisResult.

        Args:
            raw_results (List[dict]): Raw list of metric results.
            alpha (float): Nominal significance level used.
            balance_checker (Optional[Any]): Fitted CovariateBalanceChecker if covariates were specified.
        """
        self.raw_results = raw_results
        self.alpha = alpha
        self.df_raw = pd.DataFrame(raw_results)
        self.balance_checker = balance_checker

    def love_plot(self) -> str:
        """Returns the ASCII Love Plot visualization for baseline covariate balance.

        Returns:
            str: An ASCII text-based representation or message.
        """
        if self.balance_checker is None:
            return "No covariate balance diagnostics were compiled (no covariates provided)."
        return self.balance_checker.generate_love_plot()

    def to_dict(self) -> dict:
        """Converts the complete analysis results and metadata to a robust, serializable dictionary.

        Includes significance thresholds (alpha), raw metric outcomes, and covariate balance diagnostics if present.

        Returns:
            dict: A nested dictionary with native Python types, guaranteed to be JSON serializable.
        """
        from xpyrment.core.serialization import make_serializable
        state = {
            "alpha": self.alpha,
            "metrics": self.raw_results,
            "covariate_balance": self.balance_checker.diagnostics_ if (self.balance_checker is not None) else None,
        }
        return make_serializable(state)

    def to_json(self, indent: Optional[int] = None) -> str:
        """Converts the analysis results into a standardized, portable JSON string.

        Args:
            indent (Optional[int]): If provided, formats the JSON string with this indentation level.

        Returns:
            str: Standardized JSON representation of the analysis results.
        """
        from xpyrment.core.serialization import serialize_to_json
        return serialize_to_json(self.to_dict(), indent=indent)

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

        summary_df = pd.DataFrame()
        summary_df["Metric"] = df["metric_name"]
        summary_df["Type"] = df["metric_type"]

        summary_df["Control Mean"] = df["control_mean"].map("{:.4f}".format)
        summary_df["Treatment Mean"] = df["treatment_mean"].map("{:.4f}".format)

        lift_mask = df["relative_lift"].notna()
        summary_df["Relative Lift"] = np.where(
            lift_mask,
            df["relative_lift"].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "N/A"),
            "N/A"
        )

        ci_mask = df["rel_ci_lower"].notna() & df["rel_ci_upper"].notna()
        lower_str = df["rel_ci_lower"].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "")
        upper_str = df["rel_ci_upper"].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "")
        summary_df["95% CI (Rel)"] = np.where(
            ci_mask,
            "[" + lower_str + ", " + upper_str + "]",
            "N/A"
        )

        p_mask = df["p_value"].notna()
        p_vals = df["p_value"]
        p_str = p_vals.map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

        conditions = [
            p_vals < 0.001,
            p_vals < 0.01,
            p_vals < 0.05
        ]
        choices = ["***", "**", "*"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            syms = np.select(conditions, choices, default="")

        summary_df["p-value"] = np.where(
            p_mask,
            p_str + syms,
            "N/A"
        )

        pow_mask = df["power"].notna()
        summary_df["Post-hoc Power"] = np.where(
            pow_mask,
            df["power"].map(lambda x: f"{x:.1%}" if pd.notna(x) else ""),
            "N/A"
        )

        summary_df["CUPED"] = np.where(df["cuped_applied"], "Yes", "No")

        var_mask = df["cuped_applied"] & df["variance_reduction"].notna()
        summary_df["Var Reduction"] = np.where(
            var_mask,
            df["variance_reduction"].map(lambda x: f"{x:.1%}" if pd.notna(x) else ""),
            "-"
        )

        # Automatically raise an alert / print warning if covariate imbalance is detected
        if self.balance_checker is not None and self.balance_checker.diagnostics_:
            imbalanced = []
            for name, stats in self.balance_checker.diagnostics_.items():
                if abs(stats["smd"]) > 0.1:
                    imbalanced.append(f"'{name}' (SMD={stats['smd']:+.4f})")
            if imbalanced:
                warnings.warn(
                    f"COVARIATE IMBALANCE DETECTED: The following baseline covariates have standardized mean "
                    f"differences (SMD) exceeding the standard 0.1 threshold: {', '.join(imbalanced)}. "
                    f"Consider running CUPED adjustments, matching, or checking your randomization procedure."
                )

        return summary_df

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
    covariates: Optional[List[str]] = None,
) -> AnalysisResult:
    """Executes the statistical analysis across all registered metrics in an Experiment container.

    Iterates over each registered metric in the experiment, calculates means, relative lifts, p-values,
    confidence intervals, and power. If requested, applies multiple testing corrections across the p-values,
    updates the experiment state to `ANALYZED`, and returns a structured `AnalysisResult`.

    Args:
        experiment (Experiment): The initialized, pre-registered experiment setup container.
        control (str): The label of the control variant in the treatment column. Defaults to `"control"`.
        treatment (str): The label of the treatment variant in the treatment column. Defaults to `"treatment"`.
        alpha (float): Significance level (Type I error probability) for confidence intervals. Defaults to 0.05.
        multi_test_correction (str, optional): Multiple testing correction algorithm to apply across the
            registered metrics. Options: `"bonferroni"`, `"holm"`, `"fdr_bh"`, `"fdr_by"`, `"hochberg"`. Defaults to None.
        covariates (List[str], optional): List of covariates to check balance and adjust.

    Returns:
        AnalysisResult: A rich, summarized results container.

    Raises:
        ValueError: If no metrics have been registered, or if control/treatment labels are missing from
            the active dataset.
        PhaseOrderError: If the experiment is in an invalid state for running analysis.
    """
    unique_variants = experiment.data[experiment.treatment_col].unique()
    if control not in unique_variants:
        raise ValueError(f"Control label '{control}' not found.")
    if treatment not in unique_variants:
        raise ValueError(f"Treatment label '{treatment}' not found.")

    # 1. Topological sorted evaluation of MetricRegistry DAG if present
    if getattr(experiment, "metric_registry", None) is not None:
        registry = experiment.metric_registry
        raw_inputs = {}
        for node_name, node_info in registry.nodes.items():
            if node_info["type"] == "raw" and node_name in experiment.data.columns:
                raw_inputs[node_name] = experiment.data[node_name].to_numpy()
        
        evaluated_cache = registry.evaluate(raw_inputs)
        for key, val in evaluated_cache.items():
            if key not in experiment.data.columns or registry.nodes.get(key, {}).get("type") == "derived":
                if len(val) == len(experiment.data):
                    experiment.data[key] = val

        # Auto-populate metrics from DAG if metrics list is currently empty
        if not experiment.metrics:
            from xpyrment.metrics.taxonomy import MeanMetric
            for name in evaluated_cache.keys():
                experiment.metrics.append(MeanMetric(name, value_col=name))

    if not experiment.metrics:
        raise ValueError("No metrics have been added to the experiment.")

    # Resolve global and method-specific covariates
    covs_to_check = covariates if covariates is not None else getattr(experiment, "covariates", [])

    # 2. Automated Covariate-adjusted CUPED routing
    if covs_to_check:
        for metric in experiment.metrics:
            from xpyrment.metrics.taxonomy import MeanMetric
            if isinstance(metric, MeanMetric) and not getattr(metric, "pre_period_col", None):
                possible_candidates = [
                    f"pre_{metric.value_col}",
                    f"{metric.value_col}_pre",
                    f"{metric.value_col}_baseline",
                ]
                for cand in possible_candidates:
                    if cand in covs_to_check and cand in experiment.data.columns:
                        metric.pre_period_col = cand
                        break

    # 3. Covariate imbalance checking in a single call
    balance_checker = None
    if covs_to_check:
        valid_covs = [c for c in covs_to_check if c in experiment.data.columns]
        if valid_covs:
            from xpyrment.quasi.balance import CovariateBalanceChecker
            sub_df = experiment.data[experiment.data[experiment.treatment_col].isin([control, treatment])].dropna(subset=valid_covs)
            if len(sub_df) > 0:
                import numpy as np
                X = sub_df[valid_covs].to_numpy()
                T = (sub_df[experiment.treatment_col] == treatment).astype(int).to_numpy()
                balance_checker = CovariateBalanceChecker(covariate_names=valid_covs)
                balance_checker.fit(X, T)

    # 4. Statistical Evaluation of each Metric
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

    # Apply multiple testing corrections if requested
    if multi_test_correction and len(results) > 1:
        p_vals = [res["p_value"] for res in results]
        adjusted_p = apply_multiple_testing_correction(p_vals, alpha=alpha, method=multi_test_correction)
        for i, val in enumerate(adjusted_p):
            results[i]["p_value"] = val

    experiment.transition_to(ExperimentState.ANALYZED)
    return AnalysisResult(results, alpha=alpha, balance_checker=balance_checker)


def setup(
    data: pd.DataFrame,
    treatment_col: str,
    id_col: Optional[str] = None,
    covariates: Optional[List[str]] = None,
) -> Experiment:
    """Initializes the experimental setup container, serving as the library's primary entrypoint.

    Sets up the `Experiment` object with the target dataset, identifying variant and unit columns,
    and locks the state machine to `ExperimentState.DESIGNED`.

    Args:
        data (pd.DataFrame): The main experiment dataset containing exposure logs and outcomes.
        treatment_col (str): Column name containing variant strings (e.g., `"variant"`).
        id_col (str, optional): Column name containing unique unit identifiers (e.g., `"user_id"`).
        covariates (List[str], optional): Optional list of baseline covariates.

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

    exp = Experiment(data, treatment_col, id_col, covariates=covariates)
    return exp
