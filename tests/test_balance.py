"""Unit tests for Covariate Balance Checking & Love Plots (Block 42)."""

import numpy as np
import pytest
from xpyrment.quasi.balance import CovariateBalanceChecker


def test_covariate_balance_checker():
    rng = np.random.default_rng(42)
    N = 1000
    P = 3

    covs = rng.normal(size=(N, P))
    # Artificially shift covariate_1 on treatment
    treatment = rng.binomial(1, 0.5, size=N)
    covs[treatment == 1, 1] += 0.8  # Strong shift

    checker = CovariateBalanceChecker(covariate_names=["baseline_age", "historical_clicks", "user_tenure"])
    checker.fit(covs, treatment)

    diagnostics = checker.diagnostics_
    assert "baseline_age" in diagnostics
    assert "historical_clicks" in diagnostics
    assert "user_tenure" in diagnostics

    # historical_clicks (index 1) should show a large standardized mean difference (SMD)
    assert abs(diagnostics["historical_clicks"]["smd"]) > 0.5
    # baseline_age (index 0) should show an SMD near 0 (within standard sampling error)
    assert abs(diagnostics["baseline_age"]["smd"]) < 0.4

    # Ensure ASCII Love Plot compiles and contains key words
    love_plot = checker.generate_love_plot()
    assert isinstance(love_plot, str)
    assert "LOVE PLOT" in love_plot
    assert "historical_clicks" in love_plot


def test_balance_ks_and_mahalanobis():
    """Validates Kolmogorov-Smirnov distance and Mahalanobis joint balance diagnostics."""
    from xpyrment.validate.balance import check_covariate_balance
    import pandas as pd
    
    # Create unbalanced data
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "treatment": [0] * n + [1] * n,
        "cov1": np.concatenate([np.random.normal(0, 1, n), np.random.normal(0.5, 1.2, n)]),
        "cov2": np.concatenate([np.random.normal(0, 1, n), np.random.normal(0.2, 1.0, n)]),
        "cat1": ["A"] * n + ["B"] * n
    })
    
    results = check_covariate_balance(df, "treatment", ["cov1", "cov2", "cat1"])
    
    assert "cov1" in results
    assert "ks_statistic" in results["cov1"]
    assert "ks_p_value" in results["cov1"]
    assert "_multivariate" in results
    assert "mahalanobis_distance" in results["_multivariate"]
    assert results["_multivariate"]["n_covariates"] == 2

def test_check_covariate_balance_errors():
    """Validates that check_covariate_balance handles missing columns and missing groups correctly."""
    from xpyrment.validate.balance import check_covariate_balance
    import pandas as pd
    import pytest

    # Test for < 2 distinct groups
    df_missing_groups = pd.DataFrame({
        "treatment": [1, 1, 1],
        "cov1": [1.0, 2.0, 3.0]
    })
    with pytest.raises(ValueError, match="Balance check requires at least 2 distinct groups"):
        check_covariate_balance(df_missing_groups, "treatment", ["cov1"])

    # Test for missing covariate
    df_missing_cov = pd.DataFrame({
        "treatment": [0, 0, 1, 1],
        "cov1": [1.0, 2.0, 3.0, 4.0]
    })
    with pytest.raises(KeyError, match="Covariate column 'cov_missing' not found in DataFrame"):
        check_covariate_balance(df_missing_cov, "treatment", ["cov_missing"])
