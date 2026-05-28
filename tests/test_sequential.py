"""Unit tests for Group Sequential Lan-DeMets Alpha Spending (Block 36)."""

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


def test_sequential_spending_type_validation():
    """Asserts that invalid spending_type parameter raises a ValueError."""
    with pytest.raises(ValueError, match="spending_type must be either"):
        GroupSequentialMonitor(spending_type="invalid_spending")


def test_sequential_alpha_spent_boundaries():
    """Asserts that information fractions outside (0, 1) return exact boundary alpha budgets."""
    monitor = GroupSequentialMonitor(alpha=0.05, spending_type="obrien_fleming")
    assert monitor.alpha_spent(-0.5) == 0.0
    assert monitor.alpha_spent(0.0) == 0.0
    assert monitor.alpha_spent(1.5) == 0.05
    assert monitor.alpha_spent(1.0) == 0.05


def test_sequential_compute_boundaries_validation():
    """Asserts that invalid information fractions in compute_boundaries raise ValueError."""
    monitor = GroupSequentialMonitor(alpha=0.05)
    with pytest.raises(ValueError, match="must be strictly in the range"):
        monitor.compute_boundaries([0.5, 1.2])
        
    with pytest.raises(ValueError, match="must be strictly in the range"):
        monitor.compute_boundaries([-0.1, 0.5])


def test_sequential_incremental_floor():
    """Asserts that extremely close consecutive looks trigger the spent alpha floor safeguard."""
    monitor = GroupSequentialMonitor(alpha=0.05, spending_type="pocock")
    # Two extremely close looks to force tiny difference in cumulative spent alphas
    looks = [0.5, 0.5000000001]
    res = monitor.compute_boundaries(looks)
    
    assert res["incremental_alpha_spent"][1] == 1e-9
    assert res["critical_z_boundaries"][1] > 5.0  # Extremely high Z boundary for tiny alpha

