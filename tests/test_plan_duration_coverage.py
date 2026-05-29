import pytest
from xpyrment.plan.duration import estimate_duration_days

def test_estimate_duration_days():
    assert estimate_duration_days(1000, 100) == 10.0

    with pytest.raises(ValueError, match="required_sample_size must be greater"):
        estimate_duration_days(0, 100)

    with pytest.raises(ValueError, match="daily_traffic must be greater"):
        estimate_duration_days(100, 0)
