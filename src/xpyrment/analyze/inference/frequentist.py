"""Frequentist parametric and non-parametric statistical tests.

This module provides standard frequentist testing engines, implementing Welch's t-test for unequal
variances (with Satterthwaite degrees of freedom) and the non-parametric Mann-Whitney U rank-sum test.
"""

import numpy as np


def run_welch_t_test(group_a: np.ndarray, group_b: np.ndarray) -> dict:
    """Performs Welch's t-test for difference of means with unequal variances.

    Welch's t-test is a two-sample location test used to test the hypothesis that two populations have equal means
    ($H_0: \\mu_A = \\mu_B$). Unlike Student's t-test, Welch's t-test does not assume equal variances,
    making it the standard default for digital and scientific A/B testing.

    Args:
        group_a (np.ndarray): Array of numeric outcomes for control (Group A).
        group_b (np.ndarray): Array of numeric outcomes for treatment (Group B).

    Returns:
        dict: A dictionary containing:
            - `"t_statistic"` (float): The calculated Welch's t-value.
            - `"p_value"` (float): The two-sided p-value.
            - `"df"` (float): The approximated Satterthwaite degrees of freedom.
            - `"difference"` (float): Absolute difference between means ($\\bar{X}_B - \\bar{X}_A$).
    """
    from scipy import stats

    val_a = group_a[~np.isnan(group_a)]
    val_b = group_b[~np.isnan(group_b)]

    n_a = len(val_a)
    n_b = len(val_b)

    if n_a < 2 or n_b < 2:
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "df": float(n_a + n_b - 2),
            "difference": 0.0
        }

    mean_a = np.mean(val_a)
    mean_b = np.mean(val_b)
    var_a = np.var(val_a, ddof=1)
    var_b = np.var(val_b, ddof=1)

    se_diff = np.sqrt(var_a / n_a + var_b / n_b)
    diff = mean_b - mean_a

    if se_diff > 0.0:
        t_stat = diff / se_diff
        num = (var_a / n_a + var_b / n_b) ** 2
        den = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
        df = num / den if den > 0 else (n_a + n_b - 2)
        p_val = 2 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df))
    else:
        t_stat = 0.0
        p_val = 1.0
        df = float(n_a + n_b - 2)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "df": float(df),
        "difference": float(diff)
    }


def run_mann_whitney_u(group_a: np.ndarray, group_b: np.ndarray) -> dict:
    """Performs nonparametric Mann-Whitney U test for ordinal or non-normal continuous data.

    The Mann-Whitney U test evaluates the null hypothesis that the probability that a randomly drawn
    observation from Group B is larger than a randomly drawn observation from Group A is equal to 0.5.
    This test is non-parametric; it does not assume normality, making it extremely robust against extreme outliers.

    Args:
        group_a (np.ndarray): Array of numeric outcomes for control (Group A).
        group_b (np.ndarray): Array of numeric outcomes for treatment (Group B).

    Returns:
        dict: A dictionary containing:
            - `"u_statistic"` (float): The calculated Mann-Whitney U-value.
            - `"p_value"` (float): The two-sided asymptotic p-value.
    """
    from scipy import stats

    val_a = group_a[~np.isnan(group_a)]
    val_b = group_b[~np.isnan(group_b)]

    if len(val_a) == 0 or len(val_b) == 0:
        return {
            "u_statistic": 0.0,
            "p_value": 1.0
        }

    res = stats.mannwhitneyu(val_b, val_a, alternative="two-sided")

    return {
        "u_statistic": float(res.statistic),
        "p_value": float(res.pvalue)
    }

    # TODO: Add Brunner-Munzel test as a robust alternative to Mann-Whitney U when variances are highly unequal.
    # TODO: Implement Fisher's Exact test and G-test of independence for high-precision categorical conversions.
