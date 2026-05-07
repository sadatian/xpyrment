"""Frequentist parametric and non-parametric statistical tests.

This module provides standard frequentist testing engines, implementing Welch's t-test for unequal
variances (with Satterthwaite degrees of freedom) and the non-parametric Mann-Whitney U rank-sum test.
"""

import numpy as np


def run_welch_t_test(group_a: np.ndarray, group_b: np.ndarray) -> dict:
    """Performs Welch's t-test for difference of means with unequal variances.

    Welch's t-test is a two-sample location test used to test the hypothesis that two populations have equal means
    ($H_0: \\mu_A = \\mu_B$). Unlike Student's t-test, Welch's t-test does not assume equal variances
    ($\\sigma_A^2 \\neq \\sigma_B^2$) or equal sample sizes ($N_A \\neq N_B$), making it the standard default
    for digital and scientific A/B testing.

    Mathematical Representation:
        Let $\\bar{X}_A$, $\\bar{X}_B$ be sample means, let $s_A^2$, $s_B^2$ be sample variances, and let $N_A$, $N_B$
        be the sample sizes of Group A (control) and Group B (treatment) respectively.
        
        The Welch's t-statistic is computed as:
        $$t = \\frac{\\bar{X}_B - \\bar{X}_A}{\\sqrt{\\frac{s_A^2}{N_A} + \\frac{s_B^2}{N_B}}}$$
        
        The degrees of freedom $\\nu$ are approximated using the **Welch-Satterthwaite Equation**:
        $$\\nu \\approx \\frac{\\left( \\frac{s_A^2}{N_A} + \\frac{s_B^2}{N_B} \\right)^2}{\\frac{\\left( \\frac{s_A^2}{N_A} \\right)^2}{N_A - 1} + \\frac{\\left( \\frac{s_B^2}{N_B} \\right)^2}{N_B - 1}}$$
        
        Under $H_0$, the t-statistic asymptotically follows a Student's t-distribution with $\\nu$ degrees of freedom.
        The two-sided p-value is:
        $$p = 2 \\times P(T_{\\nu} \\ge |t|)$$
        where $T_{\\nu}$ represents the Student's t-distribution random variable.

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
    # This is currently implemented inline within taxonomy.py
    return {}


def run_mann_whitney_u(group_a: np.ndarray, group_b: np.ndarray) -> dict:
    """Performs nonparametric Mann-Whitney U test for ordinal or non-normal continuous data.

    The Mann-Whitney U test (also known as the Wilcoxon rank-sum test) evaluates the null hypothesis
    that the probability that a randomly drawn observation from Group B is larger than a randomly drawn
    observation from Group A is equal to 0.5 ($H_0: P(Y_B > Y_A) = 0.5$).
    This test is non-parametric; it does not assume normality, making it extremely robust against extreme outliers
    and highly skewed distribution shapes typical of digital engagement data (e.g., number of messages sent).

    Mathematical Representation:
        1. Combine all $N = N_A + N_B$ observations from both groups and rank them in ascending order from 1 to $N$.
           (In case of ties, assign the average of the ranks).
        2. Sum the assigned ranks for Group A: $R_A$.
        3. Compute the U-statistics for each group:
           $$U_A = N_A N_B + \\frac{N_A(N_A + 1)}{2} - R_A$$
           $$U_B = N_A N_B - U_A$$
           $$U = \\min(U_A, \\ U_B)$$
        4. For large samples ($N_A, N_B > 20$), the distribution of $U$ asymptotically approaches normality:
           $$Z = \\frac{U - m_U}{\\sigma_U}$$
           where the mean $m_U$ and standard deviation $\\sigma_U$ are:
           $$m_U = \\frac{N_A N_B}{2} \\quad \\text{and} \\quad \\sigma_U = \\sqrt{\\frac{N_A N_B (N_A + N_B + 1)}{12}}$$
           (with adjustments applied to $\\sigma_U$ if there are tied ranks).

    Args:
        group_a (np.ndarray): Array of numeric outcomes for control (Group A).
        group_b (np.ndarray): Array of numeric outcomes for treatment (Group B).

    Returns:
        dict: A dictionary containing:
            - `"u_statistic"` (float): The calculated Mann-Whitney U-value.
            - `"p_value"` (float): The two-sided asymptotic p-value.
    """
    # TODO: Implement full scipy Mann-Whitney U integration
    return {}
