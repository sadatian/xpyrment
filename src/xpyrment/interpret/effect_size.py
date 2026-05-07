"""Standardized effect size computation, focusing on scale-free difference metrics.

This module provides functions to calculate standardized difference statistics, such as Cohen's d,
to evaluate the practical (rather than just statistical) magnitude of experimental impacts.
"""

import numpy as np


def compute_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Computes standard standardized effect size using Cohen's d formula.

    Cohen's d (Cohen, 1988) is a standardized, scale-free effect size measure representing the difference
    between two group means in terms of standard deviation units. While p-values measure the statistical
    evidence against a null hypothesis (and are heavily dependent on sample size), Cohen's d measures the
    *practical magnitude* of the treatment effect, making it comparable across entirely different metrics or experiments.

    Mathematical Formulation:
        Let $N_A$, $N_B$ be sample sizes, let $\\bar{X}_A$, $\\bar{X}_B$ be sample means, and let $s_A^2$, $s_B^2$ be the
        unbiased sample variances of the two experimental groups (Control A and Treatment B respectively).

        The pooled sample standard deviation $s_{\\text{pooled}}$ is defined as:
        $$s_{\\text{pooled}} = \\sqrt{\\frac{(N_A - 1)s_A^2 + (N_B - 1)s_B^2}{N_A + N_B - 2}}$$
        
        The Cohen's d statistic is computed as:
        $$d = \\frac{\\bar{X}_B - \\bar{X}_A}{s_{\\text{pooled}}}$$

    Standard Classification Heuristics:
        - $|d| < 0.2$: Negligible effect size.
        - $0.2 \\le |d| < 0.5$: Small effect size (e.g., most successful digital A/B tests).
        - $0.5 \\le |d| < 0.8$: Medium effect size.
        - $|d| \\ge 0.8$: Large effect size (indicates highly impactful, structural interventions).

    Args:
        group_a (np.ndarray): 1D array of outcomes for Control (Group A).
        group_b (np.ndarray): 1D array of outcomes for Treatment (Group B).

    Returns:
        float: The calculated Cohen's d statistic.
    """
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    n_a, n_b = len(group_a), len(group_b)

    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std > 0:
        return (mean_b - mean_a) / pooled_std
    return 0.0

