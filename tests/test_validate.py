import pytest
from xpyrment.validate.srm import check_srm
from xpyrment.core.exceptions import SRMError


def test_check_srm_no_mismatch():
    """Asserts that check_srm compiles successfully and returns p-value when sizes match expectations."""
    # Split is 50.1% / 49.9%, which is completely normal variation
    p_val = check_srm(observed_counts=[5012, 4988], expected_ratios=[0.5, 0.5])
    assert p_val >= 0.001
    assert p_val < 1.0


def test_check_srm_mismatch_raises_error():
    """Asserts that check_srm raises an SRMError when observed splits heavily deviate from expectations."""
    # Split is 60% / 40%, which has a chi-square p-value near 1e-40 (impossible without SRM)
    with pytest.raises(SRMError) as exc_info:
        check_srm(observed_counts=[6000, 4000], expected_ratios=[0.5, 0.5])

    assert "Sample Ratio Mismatch detected" in str(exc_info.value)
