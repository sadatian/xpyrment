"""Unit tests for Meta-Regression with Knapp-Hartung Standard Errors (Block 38)."""

import numpy as np
import pytest
from xpyrment.analyze.meta_regression import MetaRegressor


def test_meta_regression_knapp_hartung():
    rng = np.random.default_rng(42)
    J = 15  # Number of studies/cohorts

    # Generate covariates (e.g. baseline conversion rate or study sample sizes)
    X = np.hstack([np.ones((J, 1)), rng.uniform(0.1, 0.9, size=(J, 1))])

    # True coefficients: baseline effect = 1.0, covariate slope = 2.5
    beta_true = np.array([1.0, 2.5])

    # Within-study variances
    v = rng.uniform(0.01, 0.05, size=J)

    # Between-study variance tau^2 = 0.02
    tau_sq = 0.02
    study_effects = np.zeros(J)
    for j in range(J):
        study_mean = np.dot(X[j], beta_true) + rng.normal(scale=np.sqrt(tau_sq))
        study_effects[j] = study_mean + rng.normal(scale=np.sqrt(v[j]))

    regressor = MetaRegressor(l2_penalty=1e-5)
    regressor.fit(study_effects, v, X)

    results = regressor.results
    assert "between_study_variance_tau_sq" in results
    assert "coefficients" in results
    assert "knapp_hartung_standard_errors" in results

    # Reconstruct true coefficients approximately
    assert regressor.beta_[0] == pytest.approx(1.0, abs=0.5)
    assert regressor.beta_[1] == pytest.approx(2.5, abs=0.8)
    assert regressor.tau_sq_ > 0.0
    assert len(regressor.se_) == 2


def test_meta_regression_hac():
    """Asserts that Newey-West HAC calculations operate correctly with positive outcomes, lags, and Bartlett weights."""
    rng = np.random.default_rng(42)
    J = 20  # Number of cohorts

    # Generate sequential covariates showing some correlation
    X = np.hstack([np.ones((J, 1)), np.linspace(0.1, 0.9, J).reshape(-1, 1)])
    beta_true = np.array([2.0, -1.0])

    # Within-cohort variances
    v = rng.uniform(0.02, 0.08, size=J)
    
    # Introduce serial autocorrelation in outcomes
    residuals = np.zeros(J)
    for j in range(1, J):
        residuals[j] = 0.5 * residuals[j - 1] + rng.normal(scale=0.1)
    
    y = np.dot(X, beta_true) + residuals

    # 1. Classic fit (for comparison)
    reg_classic = MetaRegressor().fit(y, v, X, cov_type="classic")
    se_classic = reg_classic.results["knapp_hartung_standard_errors"]

    # 2. HAC fit with automatic lag order
    reg_hac = MetaRegressor().fit(y, v, X, cov_type="hac")
    results_hac = reg_hac.results
    
    assert "hac_standard_errors" in results_hac
    assert "standard_errors" in results_hac
    assert results_hac["cov_type"] == "hac"

    se_hac = results_hac["hac_standard_errors"]
    assert np.all(se_hac > 0.0)
    assert len(se_hac) == 2

    # 3. HAC fit with fixed custom lag (l=2)
    reg_hac_l2 = MetaRegressor().fit(y, v, X, cov_type="hac", hac_lag=2)
    se_hac_l2 = reg_hac_l2.se_
    assert np.all(se_hac_l2 > 0.0)


def test_meta_regression_edge_cases():
    """Validates meta-regression robust handling of negative variances, empty input arrays, and invalid cov_types."""
    reg = MetaRegressor()
    X = np.ones((5, 1))
    y = np.array([1.0, 2.0, 1.5, 2.5, 3.0])
    v = np.array([0.1, 0.2, 0.15, -0.2, 0.1])  # Has a negative variance value

    with pytest.raises(ValueError, match="within-study variances must be positive"):
        reg.fit(y, v, X)

    # Empty inputs
    with pytest.raises(ValueError, match="empty dataset"):
        reg.fit(np.array([]), np.array([]), np.array([]))

    # Invalid cov_type
    with pytest.raises(ValueError, match="Unknown cov_type"):
        reg.fit(y, np.abs(v), X, cov_type="invalid_type")

