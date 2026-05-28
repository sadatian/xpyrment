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


def test_extreme_value_tail_estimator_few_exceedances():
    """Asserts that fewer than 5 exceedances triggers the safe default parameters fallback."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    estimator = ExtremeValueTailEstimator(percentile=0.95)
    estimator.fit(data)
    assert estimator.scale_ == 1.0
    assert estimator.shape_ == 0.1


def test_extreme_value_tail_estimator_constant_exceedances():
    """Asserts zero variance exceedances are handled safely without divide-by-zero crashes."""
    # Data where all values above 90th percentile are identical
    data = np.concatenate([np.ones(90) * 1.0, np.ones(10) * 100.0])
    estimator = ExtremeValueTailEstimator(percentile=0.90)
    estimator.fit(data)
    
    # Assert fit succeeded and scale is extremely large due to var_ex fallback
    assert estimator.scale_ > 1e10
    assert estimator.shape_ < -1e10
    
    # Expected shortfall should resolve cleanly without division by zero
    assert estimator.expected_shortfall() > 0


def test_extreme_value_tail_estimator_boundary_clipping():
    """Asserts that computed shape parameter is always mathematically < 0.5 under MOM, and safeguard functions."""
    # Generate highly heavy-tailed outcomes to force shape close to 0.5
    data = np.array([1.0] * 100 + [100.0, 10000.0, 1000000.0, 100000000.0, 10000000000.0])
    estimator = ExtremeValueTailEstimator(percentile=0.95)
    estimator.fit(data)
    # Assert MOM shape is indeed less than 0.5
    assert estimator.shape_ < 0.5
    
    # Verify the helper method clips to 0.49 when manual values are set
    estimator.shape_ = 0.6
    estimator._clip_shape()
    assert estimator.shape_ == 0.49


def test_extreme_value_tail_estimator_infinite_shortfall():
    """Asserts that shape parameter >= 1.0 results in infinite expected shortfall (infinite tail mean)."""
    estimator = ExtremeValueTailEstimator()
    estimator.scale_ = 2.0
    estimator.shape_ = 1.2
    estimator.threshold_ = 10.0
    assert estimator.expected_shortfall() == float("inf")
