import numpy as np
import pytest

from xpyrment.analyze.inference.bootstrap import run_bootstrap_ci, run_block_bootstrap_ci


def test_bootstrap_basic_percentile_and_bca():
    """Asserts that both percentile and BCa bootstrap methods return valid, logical boundaries."""
    rng = np.random.default_rng(42)
    # Strongly right-skewed data
    data = rng.exponential(scale=10.0, size=500)

    # 1. Percentile CI
    lower_p, upper_p = run_bootstrap_ci(data, num_resamples=1000, confidence_level=0.95, method="percentile", random_seed=101)
    assert lower_p < upper_p
    # Sample mean is ~10, so CI of mean should surround 10
    assert 8.0 < lower_p < 11.0
    assert 10.0 < upper_p < 13.0

    # 2. BCa CI (corrects for skewness)
    lower_b, upper_b = run_bootstrap_ci(data, num_resamples=1000, confidence_level=0.95, method="bca", random_seed=101)
    assert lower_b < upper_b
    assert 8.0 < lower_b < 11.0
    assert 10.0 < upper_b < 13.0


def test_bootstrap_reproducibility():
    """Asserts that supplying a random seed results in exactly reproducible confidence intervals."""
    rng = np.random.default_rng(100)
    data = rng.normal(loc=5.0, scale=1.0, size=200)

    # Run 1
    low1, up1 = run_bootstrap_ci(data, num_resamples=500, confidence_level=0.95, method="bca", random_seed=42)
    # Run 2
    low2, up2 = run_bootstrap_ci(data, num_resamples=500, confidence_level=0.95, method="bca", random_seed=42)

    assert low1 == low2
    assert up1 == up2


def test_bootstrap_zero_variance_fallback():
    """Validates that constant arrays are handled safely, returning point estimates without raising ZeroDivisionError."""
    # Constant array (zero variance)
    constant_data = np.array([42.0] * 100)

    lower, upper = run_bootstrap_ci(constant_data, num_resamples=1000, method="bca")
    assert lower == 42.0
    assert upper == 42.0

    lower_p, upper_p = run_bootstrap_ci(constant_data, num_resamples=1000, method="percentile")
    assert lower_p == 42.0
    assert upper_p == 42.0


def test_bootstrap_degenerate_resamples():
    """Validates that sparse arrays that produce degenerate zero-variance replicates fall back gracefully."""
    # Array with just one non-zero value, but we set random_seed such that
    # the outlier is never sampled in the bootstrap resamples (or very rarely,
    # causing degenerate replicates).
    # To force degenerate, let's make an array of 50 zeros and 1 tiny non-zero value.
    data = np.zeros(50)
    data[0] = 1e-15
    
    # The input variance is > 0
    assert np.var(data) > 0.0

    lower, upper = run_bootstrap_ci(data, num_resamples=50, method="bca", random_seed=42)
    # The bounds should just collapse to the point estimate safely without BCa math errors
    assert lower == float(np.mean(data))
    assert upper == float(np.mean(data))


def test_bootstrap_large_chunked_execution():
    """Verifies that large arrays/resamples trigger chunked vectorized processing and run successfully."""
    # Setup data size such that num_resamples * size exceeds 10_000_000 elements threshold
    data = np.random.default_rng(42).normal(loc=10.0, scale=2.0, size=15000)

    # Total elements = 1000 * 15000 = 15,000,000 > 10,000,000 threshold
    lower, upper = run_bootstrap_ci(data, num_resamples=1000, confidence_level=0.90, method="percentile", random_seed=7)
    assert lower < upper
    assert 9.0 < lower < 10.5
    assert 9.5 < upper < 11.0


def test_bootstrap_empty_validation():
    """Asserts that running on empty array triggers proper ValueError."""
    empty_data = np.array([])
    with pytest.raises(ValueError, match="Cannot run bootstrap on empty data group"):
        run_bootstrap_ci(empty_data)


def test_bootstrap_invalid_method():
    """Asserts that an unknown method parameter raises ValueError."""
    data = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="Unknown bootstrap method"):
        run_bootstrap_ci(data, method="invalid_bootstrap_method")


def test_block_bootstrap_basic():
    """Asserts that both moving and circular block bootstraps produce valid, logical intervals."""
    rng = np.random.default_rng(42)
    # Simulate a highly autocorrelated AR(1) process
    data = np.zeros(200)
    for i in range(1, 200):
        data[i] = 0.7 * data[i - 1] + rng.normal()

    # MBB with percentile
    lower_m, upper_m = run_block_bootstrap_ci(
        data, block_size=10, num_resamples=1000, bootstrap_method="moving", ci_method="percentile", random_seed=42
    )
    assert lower_m < upper_m
    assert -1.5 < lower_m < 0.5
    assert -0.5 < upper_m < 1.5

    # CBB with BCa
    lower_c, upper_c = run_block_bootstrap_ci(
        data, block_size=10, num_resamples=1000, bootstrap_method="circular", ci_method="bca", random_seed=42
    )
    assert lower_c < upper_c
    assert -1.5 < lower_c < 0.5
    assert -0.5 < upper_c < 1.5


def test_block_bootstrap_consistency_with_iid():
    """Asserts that block bootstrap with block_size=1 yields identical statistical bounds to i.i.d. bootstrap."""
    rng = np.random.default_rng(100)
    data = rng.normal(loc=10.0, scale=2.0, size=50)

    # I.I.D percentile
    low_iid, up_iid = run_bootstrap_ci(
        data, num_resamples=500, confidence_level=0.95, method="percentile", random_seed=99
    )

    # Block moving with block_size=1 percentile
    low_block, up_block = run_block_bootstrap_ci(
        data, block_size=1, num_resamples=500, confidence_level=0.95, bootstrap_method="moving", ci_method="percentile", random_seed=99
    )

    assert low_iid == pytest.approx(low_block)
    assert up_iid == pytest.approx(up_block)


def test_block_bootstrap_reproducibility():
    """Asserts that seed replication holds exactly for block bootstrap."""
    data = np.arange(100, dtype=float)

    # Run 1
    low1, up1 = run_block_bootstrap_ci(
        data, block_size=5, num_resamples=100, bootstrap_method="circular", ci_method="bca", random_seed=123
    )
    # Run 2
    low2, up2 = run_block_bootstrap_ci(
        data, block_size=5, num_resamples=100, bootstrap_method="circular", ci_method="bca", random_seed=123
    )

    assert low1 == low2
    assert up1 == up2


def test_block_bootstrap_edge_cases():
    """Validates block bootstrap parameter boundaries and error triggers."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Empty array
    with pytest.raises(ValueError, match="empty data group"):
        run_block_bootstrap_ci(np.array([]), block_size=2)

    # Block size zero or negative
    with pytest.raises(ValueError, match="Block size must be a positive integer"):
        run_block_bootstrap_ci(data, block_size=0)
    with pytest.raises(ValueError, match="Block size must be a positive integer"):
        run_block_bootstrap_ci(data, block_size=-3)

    # Block size larger than n
    with pytest.raises(ValueError, match="cannot exceed data length"):
        run_block_bootstrap_ci(data, block_size=6)

    # Constant array (zero variance)
    const_data = np.array([7.0] * 20)
    low, up = run_block_bootstrap_ci(const_data, block_size=4, num_resamples=100, bootstrap_method="moving", ci_method="bca")
    assert low == 7.0
    assert up == 7.0


def test_block_bootstrap_large_chunked():
    """Asserts chunked vectorized block bootstrap works perfectly for large datasets."""
    # Size that triggers chunking: total elements = 500 * 25000 = 12,500,000 > 10,000,000 threshold
    data = np.arange(25000, dtype=float)
    low, up = run_block_bootstrap_ci(
        data, block_size=100, num_resamples=500, bootstrap_method="moving", ci_method="percentile", random_seed=42
    )
    assert low < up

