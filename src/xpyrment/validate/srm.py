from typing import List
from scipy import stats
from xpyrment.core.exceptions import SRMError


def check_srm(observed_counts: List[int], expected_ratios: List[float]) -> float:
    """Calculates the Chi-square p-value to check for Sample Ratio Mismatch (SRM).

    Args:
        observed_counts: Number of units allocated to each variant (e.g. [5012, 4988]).
        expected_ratios: Expected allocation ratios (e.g. [0.5, 0.5]).

    Returns:
        float: p-value of the goodness-of-fit test. If p < 0.001, SRM is highly likely.
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
