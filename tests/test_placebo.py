"""Unit tests for Difference-in-Differences Parallel Trends Placebo Tests (Block 46)."""

import numpy as np
import pytest
from xpyrment.quasi.diff_in_diff import ParallelTrendsPlaceboTest


def test_parallel_trends_placebo_test():
    rng = np.random.default_rng(42)
    N = 100

    # Parallel trends hold
    # Y = 10 + 2 * time + 0.5 * treatment + noise
    treatment = rng.binomial(1, 0.5, size=N)
    time = rng.choice([1.0, 2.0, 3.0, 4.0], size=N)
    y = 10.0 + 2.0 * time + 0.5 * treatment + rng.normal(0, 0.1, size=N)

    # Placebo start is inside the pre-treatment window (time < 4.0)
    tester = ParallelTrendsPlaceboTest(significance_level=0.05)
    res = tester.fit_placebo_test(y, treatment, time, treatment_start_time=4.0)

    assert "placebo_coefficient" in res
    assert "placebo_p_value" in res
    # Because trends are parallel, placebo interaction coefficient should be insignificant (p > 0.05)
    assert res["trends_parallel"] is True


def test_placebo_test_violations():
    rng = np.random.default_rng(42)
    N = 200

    # Violating parallel trends in the pre-period
    # For control, Y = 10 + 1 * time
    # For treatment, Y = 10 + 4 * time (differential pre-period trends!)
    treatment = rng.binomial(1, 0.5, size=N)
    time = rng.choice([1.0, 2.0, 3.0, 4.0], size=N)
    y = 10.0 + (1.0 + 3.0 * treatment) * time + rng.normal(0, 0.1, size=N)

    tester = ParallelTrendsPlaceboTest(significance_level=0.05)
    res = tester.fit_placebo_test(y, treatment, time, treatment_start_time=4.0)

    # Placebo interaction should be highly significant, rejecting parallel trends (trends_parallel = False)
    assert res["trends_parallel"] is False
