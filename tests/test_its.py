"""Unit tests for Interrupted Time Series (ITS) with HAC Standard Errors (Block 35)."""

import numpy as np
import pytest
from xpyrment.analyze.its import InterruptedTimeSeries


def test_interrupted_time_series():
    rng = np.random.default_rng(42)
    T = 80
    treatment_index = 40

    # Model: baseline = 10.0, trend = 0.05, level_shift = 3.0, slope_shift = -0.02
    t = np.arange(T, dtype=float)
    D = (t >= treatment_index).astype(float)
    P = np.maximum(0.0, t - treatment_index)

    # Adding correlated noise (AR(1) error term to test HAC standard errors)
    errors = np.zeros(T)
    errors[0] = rng.normal()
    for s in range(1, T):
        errors[s] = 0.5 * errors[s-1] + rng.normal(scale=0.1)

    outcomes = 10.0 + 0.05 * t + 3.0 * D - 0.02 * P + errors

    its = InterruptedTimeSeries(treatment_index=treatment_index)
    its.fit(outcomes)

    results = its.results
    assert "intercept" in results
    assert "time_trend" in results
    assert "level_shift" in results
    assert "post_slope_shift" in results

    # Recovery of key coefficients
    assert results["intercept"]["coefficient"] == pytest.approx(10.0, abs=0.5)
    assert results["level_shift"]["coefficient"] == pytest.approx(3.0, abs=0.5)
    # HAC standard errors should be positive
    assert results["level_shift"]["hac_standard_error"] > 0.0
