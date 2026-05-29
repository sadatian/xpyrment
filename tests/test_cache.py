"""Comprehensive unit tests for the Thread-Safe LRU StatisticalCache and stats wrappers (Block 66)."""

import threading
from typing import Any
import time
import numpy as np
import pytest
from scipy import stats

from xpyrment.core.cache import (
    StatisticalCache,
    cached_statistical,
    cached_t_cdf,
    cached_t_ppf,
    cached_norm_cdf,
    cached_norm_ppf,
    cached_chi2_cdf,
    cached_chi2_sf,
    cached_chi2_ppf,
    welch_satterthwaite_df,
)


def test_cache_basic_operations() -> None:
    """Verifies standard statistical cache set, get, clear, hits, and misses."""
    cache = StatisticalCache(maxsize=10)
    assert cache.hits == 0
    assert cache.misses == 0

    # Cache miss
    assert cache.get("key1") is None
    assert cache.hits == 0
    assert cache.misses == 1

    # Cache set & hit
    cache.set("key1", 42.0)
    assert cache.get("key1") == 42.0
    assert cache.hits == 1
    assert cache.misses == 1

    # Clear cache
    cache.clear()
    assert cache.get("key1") is None
    assert cache.hits == 0
    assert cache.misses == 1  # Reset to 0 hits/misses, then 1 miss from get


def test_cache_lru_eviction() -> None:
    """Verifies that size-bounded eviction conforms strictly to Least Recently Used logic."""
    cache = StatisticalCache(maxsize=3)

    # Fill cache
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)

    # Exceed capacity: oldest "a" should be popped
    cache.set("d", 4)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3
    assert cache.get("d") == 4

    # Make "b" the most recently used (MRU)
    assert cache.get("b") == 2
    
    # Exceed capacity: "c" should be popped (since "b" was refreshed)
    cache.set("e", 5)
    assert cache.get("c") is None
    assert cache.get("b") == 2
    assert cache.get("d") == 4
    assert cache.get("e") == 5


def test_cache_thread_safety() -> None:
    """Ensures concurrency safety across dozens of threads reading and writing simultaneously."""
    cache = StatisticalCache(maxsize=50)
    num_threads = 20
    iterations = 200
    errors = []

    def worker(worker_id: int) -> None:
        try:
            for i in range(iterations):
                key = f"key_{worker_id}_{i % 10}"
                # Rapid get and set
                cache.get(key)
                cache.set(key, i * 2)
                cache.get(key)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrency exceptions occurred: {errors}"


def test_cached_decorator_hashable_and_unhashable() -> None:
    """Verifies that cached decorator handles generic inputs, including non-hashable fallbacks."""
    cache = StatisticalCache(maxsize=5)
    call_count = 0

    @cached_statistical(cache)
    def dummy_func(x: Any, y: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if isinstance(x, np.ndarray):
            return x * 2
        return x + y

    # First call: cache miss
    res1 = dummy_func(5, 10)
    assert res1 == 15
    assert call_count == 1

    # Second call: cache hit
    res2 = dummy_func(5, 10)
    assert res2 == 15
    assert call_count == 1  # Not incremented

    # Third call with different parameters: cache miss
    res3 = dummy_func(5, 20)
    assert res3 == 25
    assert call_count == 2

    # Call with non-hashable inputs (numpy array): should bypass cache cleanly
    arr = np.array([1, 2, 3])
    res_arr1 = dummy_func(arr, None)
    assert np.array_equal(res_arr1, np.array([2, 4, 6]))
    assert call_count == 3

    # Consecutive call with same non-hashable input: runs again (bypassed)
    res_arr2 = dummy_func(arr, None)
    assert np.array_equal(res_arr2, np.array([2, 4, 6]))
    assert call_count == 4


def test_cached_stats_wrappers_precision() -> None:
    """Verifies cached distribution cdf/ppf wrappers match raw Scipy outputs exactly."""
    # Test values
    df_val = 14.5
    x_val = 1.96
    q_val = 0.975

    # 1. t-distribution
    assert abs(cached_t_cdf(x_val, df_val) - stats.t.cdf(x_val, df=df_val)) < 1e-12
    assert abs(cached_t_ppf(q_val, df_val) - stats.t.ppf(q_val, df=df_val)) < 1e-12

    # 2. norm-distribution
    assert abs(cached_norm_cdf(x_val) - stats.norm.cdf(x_val)) < 1e-12
    assert abs(cached_norm_ppf(q_val) - stats.norm.ppf(q_val)) < 1e-12

    # 3. chi2-distribution
    assert abs(cached_chi2_cdf(x_val, df_val) - stats.chi2.cdf(x_val, df=df_val)) < 1e-12
    assert abs(cached_chi2_sf(x_val, df_val) - stats.chi2.sf(x_val, df=df_val)) < 1e-12
    assert abs(cached_chi2_ppf(q_val, df_val) - stats.chi2.ppf(q_val, df=df_val)) < 1e-12

    # 4. Welch-Satterthwaite degrees of freedom
    var_a, n_a, var_b, n_b = 2.5, 30, 4.2, 45
    num = (var_a / n_a + var_b / n_b) ** 2
    den = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
    expected_df = num / den
    assert abs(welch_satterthwaite_df(var_a, n_a, var_b, n_b) - expected_df) < 1e-12
