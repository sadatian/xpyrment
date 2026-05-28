"""Unit tests for Winsorization & Outlier Stabilization (Block 43)."""

import numpy as np
import pytest
from xpyrment.analyze.outliers import WinsorizationEngine


def test_winsorization_engine():
    # Symmetric 5% capping at each end
    engine = WinsorizationEngine(bounds=(0.05, 0.95))
    
    # 1 to 100
    data = np.arange(1, 101, dtype=float)
    transformed = engine.fit_transform(data)

    assert engine.lower_val_ == pytest.approx(5.95)
    assert engine.upper_val_ == pytest.approx(95.05)

    # Values below 5.95 should be capped to 5.95
    assert transformed[0] == pytest.approx(5.95)
    assert transformed[4] == pytest.approx(5.95)
    # Values above 95.05 should be capped to 95.05
    assert transformed[-1] == pytest.approx(95.05)
    assert transformed[-5] == pytest.approx(95.05)

    # Interior values should remain unchanged
    assert transformed[50] == 51.0


def test_winsorization_bounds_validation():
    """Asserts that invalid Winsorization bounds raise a ValueError."""
    with pytest.raises(ValueError, match="satisfy 0.0 <= lower < upper <= 1.0"):
        WinsorizationEngine(bounds=(0.95, 0.05))

    with pytest.raises(ValueError, match="satisfy 0.0 <= lower < upper <= 1.0"):
        WinsorizationEngine(bounds=(-0.1, 0.9))

    with pytest.raises(ValueError, match="satisfy 0.0 <= lower < upper <= 1.0"):
        WinsorizationEngine(bounds=(0.1, 1.5))


def test_winsorization_empty_input():
    """Asserts that empty inputs are handled safely without errors."""
    engine = WinsorizationEngine()
    empty_arr = np.array([])
    
    # Fit empty array
    engine.fit(empty_arr)
    assert engine.lower_val_ == 0.0
    assert engine.upper_val_ == 0.0
    
    # Transform empty array
    res = engine.transform(empty_arr)
    assert len(res) == 0


def test_winsorization_asymmetric():
    """Asserts asymmetric Winsorization bounds function correctly."""
    # Only cap upper 10%
    engine = WinsorizationEngine(bounds=(0.0, 0.90))
    data = np.arange(1, 11, dtype=float)
    transformed = engine.fit_transform(data)

    # Lower bound is 0th percentile (1.0), upper bound is 90th percentile (9.1)
    assert engine.lower_val_ == 1.0
    assert engine.upper_val_ == pytest.approx(9.1)

    # Values at or below 1.0 should remain unchanged
    assert transformed[0] == 1.0
    # Values above 9.1 should be capped
    assert transformed[-1] == pytest.approx(9.1)

