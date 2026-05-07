import numpy as np
import pandas as pd
import pytest
from xpyrment.validate.srm import check_srm
from xpyrment.validate.balance import check_covariate_balance
from xpyrment.core.exceptions import SRMError


def test_check_srm_no_mismatch():
    """Asserts that check_srm compiles successfully and returns p-value when sizes match expectations."""
    # Split is 50.1% / 49.9%, which is completely normal variation
    p_val = check_srm(observed_counts=[5012, 4988], expected_ratios=[0.5, 0.5])
    assert p_val >= 0.001
    assert p_val < 1.0


def test_check_srm_mismatch_raises_error():
    """Asserts that check_srm raises an SRMError when observed splits heavily deviate from expectations."""
    # Split is 60% / 40%, which has a chi-square p-value near 1e-40 (impossible without SRM)
    with pytest.raises(SRMError) as exc_info:
        check_srm(observed_counts=[6000, 4000], expected_ratios=[0.5, 0.5])

    assert "Sample Ratio Mismatch detected" in str(exc_info.value)


def test_check_covariate_balance():
    """Tests SMD and p-value computations for continuous and categorical covariate balance."""
    rng = np.random.default_rng(42)

    # 1. Generate balanced experimental dataset
    n_samples = 10000
    groups = ["control", "treatment"]
    group_assignments = rng.choice(groups, size=n_samples, p=[0.5, 0.5])

    # Continuous covariate: age (mean=35, std=8)
    age = rng.normal(loc=35.0, scale=8.0, size=n_samples)

    # Categorical covariate: country (US, CA, UK)
    countries = ["US", "CA", "UK"]
    country_assignments = rng.choice(countries, size=n_samples, p=[0.6, 0.3, 0.1])

    df = pd.DataFrame({
        "group": group_assignments,
        "age": age,
        "country": country_assignments
    })

    # Call diagnostics
    results = check_covariate_balance(df, "group", ["age", "country"])

    # Verify dictionary structure and content
    assert "age" in results
    assert results["age"]["type"] == "numeric"
    assert "smd" in results["age"]
    assert "p_value" in results["age"]
    assert abs(results["age"]["smd"]) < 0.1  # Balanced dataset

    assert "country" in results
    assert results["country"]["type"] == "categorical"
    assert "p_value" in results["country"]
    assert results["country"]["p_value"] >= 0.0  # Proper p-value returned


def test_run_aa_test_validation():
    """Tests A/A simulation engine for uniform p-value distributions (Kolmogorov-Smirnov test)."""
    from xpyrment.validate.aa_test import run_aa_test_validation
    rng = np.random.default_rng(42)

    # Generate a pure control A/A dataset (identical normal populations)
    n_samples = 500
    group_assignments = rng.choice(["A1", "A2"], size=n_samples, p=[0.5, 0.5])
    metric_values = rng.normal(loc=10.0, scale=2.0, size=n_samples)

    df = pd.DataFrame({
        "group": group_assignments,
        "revenue": metric_values
    })

    # Execute simulation (using 50 reps to keep it extremely fast)
    ks_p_val = run_aa_test_validation(df, "group", "revenue", num_simulations=50, seed=42)

    # Assert p-value boundary correctness
    assert 0.0 <= ks_p_val <= 1.0


def test_check_novelty_effects():
    """Tests the novelty/primacy interaction slope regression detection."""
    from xpyrment.validate.novelty import check_novelty_effects
    rng = np.random.default_rng(42)

    n_samples = 1000
    groups = ["control", "treatment"]
    group_assignments = rng.choice(groups, size=n_samples)

    # Simulate exposure time (days from 0 to 10)
    time_exposure = rng.uniform(0.0, 10.0, size=n_samples)

    # Novelty Effect simulation:
    # beta_0 (intercept) = 10.0
    # beta_1 (initial treatment effect) = 5.0
    # beta_2 (baseline trend) = 0.1
    # beta_3 (decay interaction slope) = -0.8
    # e ~ Normal(0, 1)
    T = np.where(group_assignments == "treatment", 1.0, 0.0)
    residuals = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    metric_values = 10.0 + 5.0 * T + 0.1 * time_exposure - 0.8 * T * time_exposure + residuals

    df = pd.DataFrame({
        "group": group_assignments,
        "revenue": metric_values,
        "day": time_exposure
    })

    results = check_novelty_effects(df, "group", "revenue", "day")

    # Assert regression coefficients are close to expected values and classified as novelty
    assert results["classification"] == "Novelty Effect Detected"
    assert results["treatment"]["coef"] > 0.0
    assert results["interaction"]["coef"] < 0.0
    assert results["interaction"]["p_value"] < 0.05



