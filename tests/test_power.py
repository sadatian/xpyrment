"""Unit tests for Analytical Power Analysis & Sample Size Estimator (Block 44)."""

import numpy as np
import pytest
from xpyrment.design.power import AnalyticalPowerCalculator


def test_analytical_power_calculator():
    calc = AnalyticalPowerCalculator(alpha=0.05, power=0.80)

    # Standard two-sample size calculation
    # For variance = 1.0, mde = 0.25: required sample size per group is ~251
    n_req = calc.compute_sample_size(mde=0.25, variance=1.0)
    assert n_req == pytest.approx(251, abs=3)

    # Recover MDE approximately
    mde_calc = calc.compute_mde(sample_size=n_req, variance=1.0)
    assert mde_calc == pytest.approx(0.25, abs=0.01)

    # Recover Power approximately
    pwr = calc.compute_power(sample_size=n_req, mde=0.25, variance=1.0)
    assert pwr == pytest.approx(0.80, abs=0.02)


def test_clustered_design_power_adjustment():
    calc = AnalyticalPowerCalculator(alpha=0.05, power=0.80)
    
    base_n = 200
    # Clustered design adjustment with avg_cluster_size = 10, icc = 0.05
    # VIF = 1 + (10 - 1) * 0.05 = 1 + 9 * 0.05 = 1.45
    # adjusted_n = 200 * 1.45 = 290
    adj_n = calc.adjust_for_clusters(base_n, avg_cluster_size=10, icc=0.05)
    assert adj_n == 290
