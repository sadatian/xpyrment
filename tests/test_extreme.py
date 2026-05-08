"""Unit tests for Extreme Value Theory (EVT) (Block 32)."""

import numpy as np
import pytest
from xpyrment.analyze.extreme import ExtremeValueTailEstimator


def test_extreme_value_tail_estimator():
    rng = np.random.default_rng(42)
    # Generate exponential data (which is a special case of GPD with shape xi = 0)
    data = rng.exponential(scale=10.0, size=1000)

    estimator = ExtremeValueTailEstimator(percentile=0.90)
    estimator.fit(data)

    metrics = estimator.metrics
    assert "percentile_threshold" in metrics
    assert "gpd_scale_sigma" in metrics
    assert "gpd_shape_xi" in metrics
    assert "expected_shortfall" in metrics

    # For an exponential tail, the threshold should be near -10 * ln(0.10) ~ 23
    assert estimator.threshold_ == pytest.approx(23.0, abs=3.0)
    # Shape parameter xi should be close to 0
    assert estimator.shape_ == pytest.approx(0.0, abs=0.25)
    # Expected shortfall should be strictly greater than threshold
    assert estimator.expected_shortfall() > estimator.threshold_
