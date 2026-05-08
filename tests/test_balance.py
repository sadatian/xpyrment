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
