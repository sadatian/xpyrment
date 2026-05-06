import numpy as np


def compute_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Computes standard standardized effect size using Cohen's d formula."""
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    n_a, n_b = len(group_a), len(group_b)

    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std > 0:
        return (mean_b - mean_a) / pooled_std
    return 0.0
