import pytest
import numpy as np
from xpyrment.analyze.outliers import WinsorizationEngine

def test_winsorization_engine_empty_fit_transform():
    engine = WinsorizationEngine()
    res = engine.fit_transform(np.array([]))
    assert len(res) == 0

def test_winsorization_engine_invalid_bounds():
    with pytest.raises(ValueError, match="must satisfy"):
        WinsorizationEngine(bounds=(0.9, 0.1))

def test_winsorization_engine_transform_empty():
    engine = WinsorizationEngine()
    engine.lower_val_ = 0.0
    engine.upper_val_ = 1.0
    res = engine.transform(np.array([]))
    assert len(res) == 0
