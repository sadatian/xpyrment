"""Unit tests for Optimal Transport & Quantile Distributional Effects (Block 34)."""

import numpy as np
import pytest
from xpyrment.quasi.optimal_transport import QuantileOptimalTransport


def test_quantile_treatment_effects():
    # Setup control as N(0, 1), treatment as N(1.5, 1) (pure location shift of 1.5)
    rng = np.random.default_rng(42)
    control = rng.normal(loc=0.0, scale=1.0, size=500)
    treatment = rng.normal(loc=1.5, scale=1.0, size=500)

    ot = QuantileOptimalTransport(quantiles=np.array([0.25, 0.50, 0.75]))
    qte = ot.compute_qte(control, treatment)

    # Every quantile shift should recover location shift of 1.5 approximately
    assert qte[0.25] == pytest.approx(1.5, abs=0.25)
    assert qte[0.50] == pytest.approx(1.5, abs=0.25)
    assert qte[0.75] == pytest.approx(1.5, abs=0.25)


def test_wasserstein_distance():
    rng = np.random.default_rng(42)
    control = rng.normal(loc=0.0, scale=1.0, size=200)
    # Location shift of 2.0
    treatment = rng.normal(loc=2.0, scale=1.0, size=200)

    ot = QuantileOptimalTransport()
    w1 = ot.compute_wasserstein_distance(control, treatment, p=1)

    # 1D Wasserstein-1 distance with shift of 2.0 should be very close to 2.0
    assert w1 == pytest.approx(2.0, abs=0.2)
