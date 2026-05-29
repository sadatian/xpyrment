"""Thread-safe, size-bounded LRU cache for statistical calculations (Block 66).

This module provides the `StatisticalCache` class, caching decorators, and centralized
wrappers for scipy distribution lookups to avoid high-frequency mathematical recalculations.
"""

import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, Hashable, Optional, Tuple


class StatisticalCache:
    """A thread-safe, size-bounded LRU cache for high-frequency statistical operations."""

    def __init__(self, maxsize: int = 1024) -> None:
        """Initializes the statistical cache layer.

        Args:
            maxsize (int): Maximum number of entries before LRU eviction triggers.
        """
        self.maxsize = maxsize
        self.cache: OrderedDict[Hashable, Any] = OrderedDict()
        self.lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: Hashable) -> Optional[Any]:
        """Retrieves an item from the cache, moving it to the end (MRU).

        Args:
            key (Hashable): Unique lookup key.

        Returns:
            Optional[Any]: The cached value if hit, else None.
        """
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self._hits += 1
                return self.cache[key]
            self._misses += 1
            return None

    def set(self, key: Hashable, value: Any) -> None:
        """Sets a value in the cache, evicting the oldest item if maxsize is exceeded.

        Args:
            key (Hashable): Unique lookup key.
            value (Any): Value to store.
        """
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)  # Evict oldest (first inserted in OrderedDict)

    def clear(self) -> None:
        """Clears all cached entries and resets statistics."""
        with self.lock:
            self.cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def hits(self) -> int:
        """Total number of cache hits."""
        with self.lock:
            return self._hits

    @property
    def misses(self) -> int:
        """Total number of cache misses."""
        with self.lock:
            return self._misses


def cached_statistical(cache_instance: StatisticalCache) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """A decorator to cache statistical function outputs based on hashable parameters.

    If inputs are unhashable (e.g. numpy arrays), it gracefully bypasses caching.

    Args:
        cache_instance (StatisticalCache): The cache store instance to use.

    Returns:
        Callable: The decorated function wrapper.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build key using function name and arguments
            key: Tuple[Any, ...] = (func.__name__,) + args
            if kwargs:
                key += tuple(sorted(kwargs.items()))

            # Verify key hashability
            try:
                hash(key)
                can_cache = True
            except TypeError:
                can_cache = False

            if not can_cache:
                return func(*args, **kwargs)

            val = cache_instance.get(key)
            if val is not None:
                return val

            res = func(*args, **kwargs)
            cache_instance.set(key, res)
            return res
        return wrapper
    return decorator


# Instantiate standard package-wide shared statistical cache
statistical_cache = StatisticalCache(maxsize=2048)


@cached_statistical(statistical_cache)
def cached_t_cdf(x: float, df: float) -> float:
    """Cached t-distribution cumulative distribution function (CDF)."""
    from scipy import stats
    return float(stats.t.cdf(x, df=df))


@cached_statistical(statistical_cache)
def cached_t_ppf(q: float, df: float) -> float:
    """Cached t-distribution percent point function (PPF/Inverse CDF)."""
    from scipy import stats
    return float(stats.t.ppf(q, df=df))


@cached_statistical(statistical_cache)
def cached_norm_cdf(x: float) -> float:
    """Cached standard normal cumulative distribution function (CDF)."""
    from scipy import stats
    return float(stats.norm.cdf(x))


@cached_statistical(statistical_cache)
def cached_norm_ppf(q: float) -> float:
    """Cached standard normal percent point function (PPF/Inverse CDF)."""
    from scipy import stats
    return float(stats.norm.ppf(q))


@cached_statistical(statistical_cache)
def cached_chi2_cdf(x: float, df: float) -> float:
    """Cached Chi-squared cumulative distribution function (CDF)."""
    from scipy import stats
    return float(stats.chi2.cdf(x, df=df))


@cached_statistical(statistical_cache)
def cached_chi2_sf(x: float, df: float) -> float:
    """Cached Chi-squared survival function (SF / 1 - CDF)."""
    from scipy import stats
    return float(stats.chi2.sf(x, df=df))


@cached_statistical(statistical_cache)
def cached_chi2_ppf(q: float, df: float) -> float:
    """Cached Chi-squared percent point function (PPF/Inverse CDF)."""
    from scipy import stats
    return float(stats.chi2.ppf(q, df=df))


@cached_statistical(statistical_cache)
def welch_satterthwaite_df(var_a: float, n_a: int, var_b: float, n_b: int) -> float:
    """Computes Satterthwaite approximation for Welch t-test degrees of freedom.

    Args:
        var_a (float): Sample variance of group A.
        n_a (int): Sample size of group A.
        var_b (float): Sample variance of group B.
        n_b (int): Sample size of group B.

    Returns:
        float: Calculated degrees of freedom.
    """
    num = (var_a / n_a + var_b / n_b) ** 2
    den = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
    return float(num / den) if den > 0 else float(n_a + n_b - 2)
