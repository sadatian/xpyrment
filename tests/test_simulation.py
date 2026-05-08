import numpy as np
import pandas as pd
from xpyrment.simulation import generate_ab_data, ExperimentSimulator


def test_generate_ab_data():
    """Tests shape and boundary values of synthetic A/B datasets generated for xpyrment."""
    n_samples = 1000
    df = generate_ab_data(n_samples=n_samples, treatment_fraction=0.5, random_seed=42)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_samples

    expected_cols = {
        "user_id",
        "variant",
        "pre_revenue",
        "revenue",
        "converted",
        "pre_impressions",
        "pre_clicks",
        "impressions",
        "clicks",
    }
    assert expected_cols.issubset(set(df.columns))

    # Assert variant assignments are correct
    assert set(df["variant"].unique()) == {"control", "treatment"}

    # Assert non-negativity
    assert (df["pre_revenue"] >= 0).all()
    assert (df["revenue"] >= 0).all()
    assert ((df["converted"] == 0) | (df["converted"] == 1)).all()


def test_experiment_simulator_panel():
    sim = ExperimentSimulator(random_seed=100)
    df = sim.generate_synthetic_panel(
        n_samples=200,
        baseline_mean=12.0,
        treatment_effect=2.0,
        non_compliance_rate=0.1,
        spillover_effect=0.5,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200
    assert "covariate_age" in df.columns
    assert "covariate_tenure" in df.columns
    assert "assigned_treatment" in df.columns
    assert "actual_treatment" in df.columns
    assert "outcome" in df.columns


def test_experiment_simulator_monte_carlo():
    sim = ExperimentSimulator(random_seed=42)
    # 5 runs of size 100 to ensure fast execution
    res = sim.run_monte_carlo(
        n_simulations=5,
        n_samples=100,
        treatment_effect=1.5,
    )

    assert res["num_simulations_run"] == 5
    assert "empirical_mean_estimate" in res
    assert "empirical_bias" in res
    assert "empirical_mean_squared_error" in res
    assert "empirical_rejection_rate" in res
    assert 0.0 <= res["empirical_rejection_rate"] <= 1.0

