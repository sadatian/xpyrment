import pandas as pd
import pytest
import numpy as np

from xpyrment.analyze.orchestrator import setup
from xpyrment.analyze.registry import MetricRegistry
from xpyrment.simulation import generate_ab_data


def test_fluent_and_covariates():
    """Verifies the fluent API, add_covariates, automatic CUPED routing, and Love Plot generation."""
    # Generate mock experiment dataset
    df = generate_ab_data(n_samples=200, random_seed=42)
    # Add a custom pre_period covariate to ensure it maps
    df["pre_revenue"] = df["revenue"] * 0.9 + np.random.default_rng(42).normal(0, 1, len(df))

    # Fluent setup with covariates
    exp = setup(df, treatment_col="variant", id_col="user_id")
    exp.add_covariates(["pre_revenue", "converted"])

    # Fluent register_metric
    exp.register_metric("revenue", metric_type="mean")  # Will automatically route "pre_revenue" as pre_period_col
    exp.register_metric("converted", metric_type="proportion")

    # Run analysis
    res = exp.run_analysis(control="control", treatment="treatment")

    # Verify results
    assert len(res.df_raw) == 2
    assert res.balance_checker is not None

    # Check automated CUPED routing was applied because "pre_revenue" matched "revenue"
    rev_row = res.df_raw[res.df_raw["metric_name"] == "revenue"].iloc[0]
    assert rev_row["cuped_applied"]

    # Assert Love Plot generation works
    love = res.love_plot()
    assert "COVARIATE BALANCE LOVE PLOT" in love
    assert "pre_revenue" in love


def test_covariate_imbalance_warning():
    """Asserts that a warning is automatically raised when covariate imbalance is detected (SMD > 0.1)."""
    # Create artificial imbalance
    rng = np.random.default_rng(101)
    df = pd.DataFrame({
        "variant": ["control"] * 50 + ["treatment"] * 50,
        "revenue": rng.normal(10, 2, 100),
        "pre_revenue": list(rng.normal(5, 1, 50)) + list(rng.normal(7, 1, 50)),  # strong imbalance
        "user_id": range(100)
    })

    exp = setup(df, treatment_col="variant", id_col="user_id", covariates=["pre_revenue"])
    exp.register_metric("revenue")

    res = exp.run_analysis(control="control", treatment="treatment")
    assert res.balance_checker is not None
    
    # Run and verify warning triggers upon calling res.summary()
    with pytest.warns(UserWarning, match="COVARIATE IMBALANCE DETECTED"):
        res.summary()
    
    # Confirm that pre_revenue has SMD > 0.1
    smd = res.balance_checker.diagnostics_["pre_revenue"]["smd"]
    assert abs(smd) > 0.1


def test_topological_metric_dag_evaluation():
    """Validates that a MetricRegistry DAG is topologically evaluated and analyzed automatically inside run_analysis."""
    df = pd.DataFrame({
        "variant": ["control"] * 50 + ["treatment"] * 50,
        "clicks": np.random.default_rng(1).poisson(5, 100),
        "impressions": np.random.default_rng(1).poisson(100, 100),
        "user_id": range(100)
    })

    # Create MetricRegistry and set up a derived ratio metric
    registry = MetricRegistry()
    registry.add_raw("clicks")
    registry.add_raw("impressions")
    # Derived metric click_through_rate = clicks / impressions
    registry.add_derived("click_through_rate", ["clicks", "impressions"], lambda c, i: np.where(i > 0, c / i, 0.0))

    exp = setup(df, treatment_col="variant", id_col="user_id")
    exp.metric_registry = registry

    # Run analysis without manually adding metrics (will auto-populate from registry)
    res = exp.run_analysis(control="control", treatment="treatment")

    # Should contain 3 metrics evaluated
    assert len(res.df_raw) == 3
    names = res.df_raw["metric_name"].tolist()
    assert "clicks" in names
    assert "impressions" in names
    assert "click_through_rate" in names
