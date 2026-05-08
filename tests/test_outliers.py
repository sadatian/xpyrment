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
