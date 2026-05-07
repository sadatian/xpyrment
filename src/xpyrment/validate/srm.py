"""Sample Ratio Mismatch (SRM) validation using Pearson Chi-Square Goodness-of-Fit tests.

This module provides the diagnostic engine for detecting Sample Ratio Mismatches (SRMs). SRMs are
critical indicator signals of experiment integrity breaches (e.g., assignment skew, tracking failures,
or redirection bugs).
"""

from typing import List
from scipy import stats
from xpyrment.core.exceptions import SRMError


def check_srm(observed_counts: List[int], expected_ratios: List[float]) -> float:
    r"""Calculates the Chi-square p-value to check for Sample Ratio Mismatch (SRM).

    Sample Ratio Mismatch (SRM) is one of the most critical diagnostic flags in web and system experimentation.
    It indicates that the observed sample allocation counts deviate from the planned/designed allocation ratios.
    This method performs a Pearson Chi-square goodness-of-fit test to determine whether the observed counts
    are statistically compatible with the expected ratios.

    Mathematical Formulation:
        Let $k$ be the number of variants, let $O_i$ be the observed count of units in variant $i$ ($i \in \{1, \dots, k\}$),
        and let $r_i$ be the planned allocation ratio for variant $i$.
        The total observed sample size is:
        $$N = \sum_{i=1}^{k} O_i$$
        The expected sample count $E_i$ for variant $i$ is calculated as:
        $$E_i = N \times \frac{r_i}{\sum_{j=1}^{k} r_j}$$
        The Pearson Chi-square test statistic is computed as:
        $$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}$$
        Under the null hypothesis $H_0$ (there is no SRM, and the assignment mechanism is unbiased):
        $$\chi^2 \sim \chi^2_{k-1}$$
        where $k-1$ is the degrees of freedom of the distribution. The p-value is calculated as:
        $$p = 1 - F_{\chi^2_{k-1}}(\chi^2_{\text{calc}})$$
        where $F$ is the cumulative distribution function of the Chi-square distribution.

    Interpretation Threshold:
        - If $p < 0.001$ ($0.1\%$ significance): The null hypothesis of perfect assignment is rejected. An SRM is
          highly likely, signaling a telemetry or system bug that invalidates downstream causal inferences.
        - Common causes of SRM: browser-specific treatment crashes, asymmetric page-redirection delays, bot filters
          interacting with treatment flags, or mid-experiment changes in allocation rates.

    Args:
        observed_counts (List[int]): The actual recorded sample sizes allocated to each variant (e.g., `[50122, 49878]`).
        expected_ratios (List[float]): The target allocation proportions or relative weights (e.g., `[0.5, 0.5]`).

    Returns:
        float: The calculated p-value of the goodness-of-fit test.

    Raises:
        SRMError: If the computed p-value is strictly less than 0.001, indicating a severe, non-random mismatch.

    Example:
        >>> # Perfectly fine allocation (p ~ 0.81)
        >>> check_srm([5012, 4988], [0.5, 0.5])
        0.8096180371302821
        >>> # Severe mismatch (triggers SRMError)
        >>> try:
        ...     check_srm([4500, 5500], [0.5, 0.5])
        ... except SRMError as e:
        ...     print("Error detected!")
        Error detected!
    """
    total_observed = sum(observed_counts)
    sum_ratios = sum(expected_ratios)

    expected_counts = [ratio * total_observed / sum_ratios for ratio in expected_ratios]

    # Perform chi-square goodness-of-fit test
    _, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

    if p_value < 0.001:
        raise SRMError(
            f"Sample Ratio Mismatch detected (p={p_value:.4e}). "
            f"Observed counts: {observed_counts}, Expected ratios: {expected_ratios}"
        )

    return p_value

