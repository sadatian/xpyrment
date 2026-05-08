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
