from typing import List, Optional
import pandas as pd

from xpyrment.core.state import ExperimentState
from xpyrment.core.experiment import Experiment
from xpyrment.analyze.corrections import apply_multiple_testing_correction


class AnalysisResult:
    """Holds results from an experiment analysis."""

    def __init__(self, raw_results: List[dict], alpha: float = 0.05):
        self.raw_results = raw_results
        self.alpha = alpha
        self.df_raw = pd.DataFrame(raw_results)

    def summary(self, formatted: bool = True) -> pd.DataFrame:
        """Returns a summarized DataFrame of the analysis."""
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

    def plot(self, **kwargs):
        """Generates a forest plot of results."""
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
    """Executes the statistical analysis across all metrics in an Experiment.

    Args:
        experiment (Experiment): The setup container.
        control (str): Control variant name.
        treatment (str): Treatment variant name.
        alpha (float): Confidence level.
        multi_test_correction (str, optional): Correction method.

    Returns:
        AnalysisResult: Polished summary and plots.
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
    """Initializes the experimental setup container."""
    print("==========================================")
    print("      Initializing xpyrment Setup         ")
    print("==========================================")
    print(f"Total rows in dataset:  {len(data)}")
    print(f"Treatment column:      {treatment_col}")
    if id_col:
        print(f"ID column:             {id_col}")

    exp = Experiment(data, treatment_col, id_col)
    return exp
