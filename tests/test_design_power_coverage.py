"""Unit tests for statistical power analysis calculations (Block 29)."""

import numpy as np
import pytest
from xpyrment.design.power import AnalyticalPowerCalculator

def test_analytical_power_calculator():
    calc = AnalyticalPowerCalculator()
    size_per_group = calc.compute_sample_size(
        mde=0.005,
        variance=0.09
    )
    assert size_per_group > 0
    assert size_per_group == pytest.approx(56525, rel=0.05)

def test_analytical_power_md_effect():
    calc = AnalyticalPowerCalculator()
    mde = calc.compute_mde(
        sample_size=56525,
        variance=0.09
    )
    assert mde == pytest.approx(0.005, rel=0.05)

def test_power_calculator_invalid_inputs():
    calc = AnalyticalPowerCalculator()
    with pytest.raises(ValueError, match="strictly positive"):
        calc.compute_sample_size(-1.0, 0.09)
    with pytest.raises(ValueError, match="strictly positive"):
        calc.compute_sample_size(0.05, -0.09)
    with pytest.raises(ValueError, match="strictly positive"):
        calc.compute_mde(-100, 0.09)

def test_compute_power():
    calc = AnalyticalPowerCalculator()
    power = calc.compute_power(sample_size=56525, mde=0.005, variance=0.09)
    assert power == pytest.approx(0.80, rel=0.05)

    with pytest.raises(ValueError, match="strictly positive"):
        calc.compute_power(-10, 0.005, 0.09)

def test_adjust_for_clusters():
    calc = AnalyticalPowerCalculator()
    assert calc.adjust_for_clusters(100, 1, 0.1) == 100
    assert calc.adjust_for_clusters(100, 10, 0.1) == 190

    with pytest.raises(ValueError, match="in the range"):
        calc.adjust_for_clusters(100, 10, -0.1)
