import numpy as np


def run_welch_t_test(group_a: np.ndarray, group_b: np.ndarray) -> dict:
    """Performs Welch's t-test for difference of means with unequal variances."""
    # This is currently implemented inline within taxonomy.py
    return {}


def run_mann_whitney_u(group_a: np.ndarray, group_b: np.ndarray) -> dict:
    """Performs nonparametric Mann-Whitney U test for ordinal or non-normal continuous data."""
    # TODO: Implement full scipy Mann-Whitney U integration
    return {}
