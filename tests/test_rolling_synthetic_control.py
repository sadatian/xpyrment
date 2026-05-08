"""Unit tests for Rolling Synthetic Controls under Structural Breaks (Block 40)."""

import numpy as np
import pytest
from xpyrment.quasi.rolling_synthetic_control import RollingSyntheticControl


def test_rolling_synthetic_control():
    rng = np.random.default_rng(42)
    T = 40
    D = 5

    # Generate 5 donor units
    donors = rng.normal(size=(T, D))

    # True weights: 0.3 on donor 0, 0.7 on donor 1, others are 0
    weights_true = np.array([0.3, 0.7, 0.0, 0.0, 0.0])
    treated = np.dot(donors, weights_true) + rng.normal(scale=0.01, size=T)

    rsc = RollingSyntheticControl(lambda_l1=0.0, lambda_l2=1e-5)
    rsc.fit(treated, donors)

    assert len(rsc.weights_) == D
    # Simplex constraints: weights must sum to 1
    assert np.sum(rsc.weights_) == pytest.approx(1.0, abs=1e-5)
    # All weights must be non-negative
    for w in rsc.weights_:
        assert w >= -1e-8

    # Recover true weights approximately
    assert rsc.weights_[0] == pytest.approx(0.3, abs=0.1)
    assert rsc.weights_[1] == pytest.approx(0.7, abs=0.1)

    # Predict timeline
    pred = rsc.predict(donors)
    assert pred.shape == (T,)
    # Tracking error should be minimal
    assert np.mean((treated - pred) ** 2) < 0.05
