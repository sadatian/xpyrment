"""Unit tests for Differential Privacy (DP) Noise Addition (Block 33)."""

import numpy as np
import pytest
from xpyrment.network.privacy import DifferentialPrivacyEngine


def test_dp_sensitivity():
    engine = DifferentialPrivacyEngine(bounds=(0.0, 10.0))
    # Sensitivity of mean with N = 100 should be 10.0 / 100 = 0.1
    assert engine.get_mean_sensitivity(100) == pytest.approx(0.1)


def test_laplace_and_gaussian_mechanisms():
    engine = DifferentialPrivacyEngine(bounds=(0.0, 1.0))
    rng = np.random.default_rng(42)

    mean_val = 0.5
    N = 1000

    # Test Laplace mechanism (pure epsilon-DP)
    mean_lap = engine.add_laplace_noise(mean_val, N, epsilon=1.0, rng=rng)
    assert isinstance(mean_lap, float)
    assert abs(mean_lap - mean_val) < 0.1  # Noise should be small for large N

    # Test Gaussian mechanism (approximate epsilon, delta-DP)
    mean_gauss = engine.add_gaussian_noise(mean_val, N, epsilon=1.0, delta=1e-5, rng=rng)
    assert isinstance(mean_gauss, float)
    assert abs(mean_gauss - mean_val) < 0.1

    # Check bounds safety errors
    with pytest.raises(ValueError):
        engine.add_laplace_noise(mean_val, N, epsilon=-0.5)

    with pytest.raises(ValueError):
        engine.add_gaussian_noise(mean_val, N, epsilon=1.0, delta=1.5)
