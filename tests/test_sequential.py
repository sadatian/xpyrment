"""Unit tests for Group Sequential Lan-DeMets Alpha Spending (Block 36)."""

import numpy as np
import pytest
from xpyrment.analyze.sequential import GroupSequentialMonitor


def test_sequential_spending_obf():
    # O'Brien-Fleming spending should be extremely conservative at early looks
    monitor = GroupSequentialMonitor(alpha=0.05, spending_type="obrien_fleming")

    alpha_spent_1 = monitor.alpha_spent(0.2)
    alpha_spent_2 = monitor.alpha_spent(0.5)
    alpha_spent_3 = monitor.alpha_spent(1.0)

    assert alpha_spent_1 < alpha_spent_2 < alpha_spent_3
    assert alpha_spent_3 == pytest.approx(0.05)

    # At t=0.2, alpha spent should be extremely near zero
    assert alpha_spent_1 < 1e-3


def test_sequential_boundaries_pocock():
    # Pocock type spending is more aggressive early on
    monitor = GroupSequentialMonitor(alpha=0.05, spending_type="pocock")
    
    looks = [0.25, 0.50, 0.75, 1.00]
    boundaries = monitor.compute_boundaries(looks)

    assert len(boundaries["critical_z_boundaries"]) == 4
    # The incremental alphas must sum to total alpha (0.05)
    assert sum(boundaries["incremental_alpha_spent"]) == pytest.approx(0.05)
    
    # Critical z boundaries should be positive reals
    for z in boundaries["critical_z_boundaries"]:
        assert z > 1.5
