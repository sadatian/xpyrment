"""Unit tests for Welch's t-test and Mann-Whitney U tests (analyze/inference/frequentist.py)."""

import pytest
import numpy as np
from scipy import stats
from xpyrment.analyze.inference.frequentist import run_welch_t_test, run_mann_whitney_u


def test_run_welch_t_test_ideal():
    """Asserts Welch's t-test outputs are mathematically correct and match scipy solutions."""
    rng = np.random.default_rng(42)
    group_a = rng.normal(loc=10.0, scale=2.0, size=50)
    group_b = rng.normal(loc=12.0, scale=4.0, size=30)
    
    res = run_welch_t_test(group_a, group_b)
    
    # Calculate using scipy.stats.ttest_ind with equal_var=False
    scipy_res = stats.ttest_ind(group_b, group_a, equal_var=False)
    
    assert res["t_statistic"] == pytest.approx(scipy_res.statistic)
    assert res["p_value"] == pytest.approx(scipy_res.pvalue)
    assert res["df"] == pytest.approx(scipy_res.df)
    assert res["difference"] == pytest.approx(np.mean(group_b) - np.mean(group_a))


def test_run_welch_t_test_insufficient_samples():
    """Asserts that having fewer than 2 valid samples returns fallback defaults."""
    # Group A too small
    res1 = run_welch_t_test(np.array([1.0]), np.array([1.0, 2.0, 3.0]))
    assert res1["t_statistic"] == 0.0
    assert res1["p_value"] == 1.0
    assert res1["difference"] == 0.0

    # Group B too small (contains NaNs)
    res2 = run_welch_t_test(np.array([1.0, 2.0]), np.array([np.nan, 3.0]))
    assert res2["t_statistic"] == 0.0
    assert res2["p_value"] == 1.0


def test_run_welch_t_test_zero_variance():
    """Asserts zero variance groups are handled safely without divide-by-zero crashes."""
    group_a = np.array([10.0, 10.0, 10.0])
    group_b = np.array([10.0, 10.0, 10.0])
    
    res = run_welch_t_test(group_a, group_b)
    assert res["t_statistic"] == 0.0
    assert res["p_value"] == 1.0
    assert res["difference"] == 0.0


def test_run_mann_whitney_u_ideal():
    """Asserts Mann-Whitney U rank-sum test evaluates correct statistics and p-values."""
    rng = np.random.default_rng(42)
    group_a = rng.uniform(0.0, 5.0, 40)
    group_b = rng.uniform(2.0, 7.0, 40)  # Upward rank shift
    
    res = run_mann_whitney_u(group_a, group_b)
    scipy_res = stats.mannwhitneyu(group_b, group_a, alternative="two-sided")
    
    assert res["u_statistic"] == pytest.approx(scipy_res.statistic)
    assert res["p_value"] == pytest.approx(scipy_res.pvalue)


def test_run_mann_whitney_u_empty():
    """Asserts empty or purely NaN arrays return clean default statistics."""
    res1 = run_mann_whitney_u(np.array([]), np.array([1.0, 2.0]))
    assert res1["u_statistic"] == 0.0
    assert res1["p_value"] == 1.0
    
    res2 = run_mann_whitney_u(np.array([np.nan, np.nan]), np.array([1.0]))
    assert res2["u_statistic"] == 0.0
    assert res2["p_value"] == 1.0
