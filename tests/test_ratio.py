"""Unit tests for Ratio Metric Delta Method Delta-Variance Estimation (Block 45)."""

import numpy as np
import pytest
from xpyrment.analyze.ratio import RatioMetricDeltaMethod


def test_ratio_metric_delta_method():
    rng = np.random.default_rng(42)
    N = 500

    # Let numerator Y be clicks, denominator X be views
    # Control group: ~10% CTR
    den_ctrl = rng.poisson(lam=10.0, size=N).astype(float)
    num_ctrl = rng.binomial(den_ctrl.astype(int), p=0.10).astype(float)

    # Treatment group: ~12% CTR
    den_trt = rng.poisson(lam=10.0, size=N).astype(float)
    num_trt = rng.binomial(den_trt.astype(int), p=0.12).astype(float)

    estimator = RatioMetricDeltaMethod()
    estimator.fit(num_ctrl, den_ctrl, num_trt, den_trt)

    res = estimator.results

    assert res["control_ratio"] == pytest.approx(0.10, abs=0.02)
    assert res["treatment_ratio"] == pytest.approx(0.12, abs=0.02)
    assert res["treatment_effect"] == pytest.approx(0.02, abs=0.01)

    # Standard errors should be extremely small given N=500 and base variance
    assert 0.0 < res["pooled_standard_error"] < 0.01
    assert res["z_statistic"] > 2.0
    assert res["p_value"] < 0.05
