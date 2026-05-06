import pandas as pd
import pytest

from xpyrment.analyze.orchestrator import run_analysis, setup
from xpyrment.core.exceptions import PhaseOrderError
from xpyrment.core.state import ExperimentState
from xpyrment.metrics.taxonomy import MeanMetric, ProportionMetric
from xpyrment.simulation import generate_ab_data


def test_end_to_end_setup_and_analysis():
    """Validates the complete PyCaret-style setup, run_analysis, and summary workflow."""
    df = generate_ab_data(n_samples=500, random_seed=42)

    # 1. Initialize experiment setup
    exp = setup(df, treatment_col="variant", id_col="user_id")
    assert exp.state == ExperimentState.CREATED

    # 2. Add metrics
    metric_rev = MeanMetric("Revenue", value_col="revenue", pre_period_col="pre_revenue")
    metric_conv = ProportionMetric("Conversion", value_col="converted")

    exp.add_metrics([metric_rev, metric_conv])
    assert len(exp.metrics) == 2

    # 3. Running metrics before setup should work, but adding metrics after running analysis should fail
    results = exp.run_analysis(control="control", treatment="treatment")
    assert exp.state == ExperimentState.ANALYZED

    # Ensure adding metrics after analysis fails (phase-gated rule)
    with pytest.raises(PhaseOrderError):
        exp.add_metrics(metric_rev)

    # 4. Check results summary
    summary_df = results.summary(formatted=True)
    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 2
    assert "Metric" in summary_df.columns
    assert "Relative Lift" in summary_df.columns
    assert "p-value" in summary_df.columns


def test_multiple_comparison_corrections():
    """Asserts that multiple comparison p-value adjustments are properly calculated."""
    df = generate_ab_data(n_samples=400, random_seed=7)

    exp = setup(df, treatment_col="variant", id_col="user_id")
    exp.add_metrics([
        MeanMetric("Revenue", value_col="revenue"),
        ProportionMetric("Conversion", value_col="converted")
    ])

    # Run without adjustments
    res_raw = run_analysis(exp, control="control", treatment="treatment", multi_test_correction=None)
    raw_p_values = res_raw.df_raw["p_value"].tolist()

    # Re-run with Bonferroni adjustment
    # Note: re-running from ANALYZED state is allowed, but we reset state for testing
    exp.state = ExperimentState.CREATED
    res_adj = run_analysis(exp, control="control", treatment="treatment", multi_test_correction="bonferroni")
    adj_p_values = res_adj.df_raw["p_value"].tolist()

    # With multiple tests, adjusted p-values must be greater than or equal to unadjusted ones
    for raw, adj in zip(raw_p_values, adj_p_values):
        assert adj >= raw
