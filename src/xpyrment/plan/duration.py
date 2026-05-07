"""Duration estimation utilities based on statistical requirements and traffic pipelines.

This module provides tools to estimate the temporal duration of an experiment run, mapping the
conceptually abstract "required sample size" derived from statistical power analysis into physical
calendar time (days) using observed traffic rates and ramp-up schedules.
"""


def estimate_duration_days(required_sample_size: int, daily_traffic: int) -> float:
    """Estimates the required experiment run duration in days.

    Translates the calculated target sample size ($N_{\text{required}}$) into the estimated calendar days
    needed to accumulate that sample volume based on active daily traffic ($T_{\text{daily}}$).

    Mathematical Model:
        The duration in days ($D$) is computed as:
        $$D = \frac{N_{\text{required}}}{T_{\text{daily}}}$$
        where $N_{\text{required}}$ represents the combined total sample size across all active arms
        (control + treatment arms) or the single-arm requirement multiplied by the number of arms.

    Args:
        required_sample_size (int): The total sample size needed across all arms combined
            (e.g., control $n$ + treatment $n$). Must be greater than zero.
        daily_traffic (int): The expected number of unique qualifying experimental units (e.g., users,
            sessions, or pageviews) entering the experiment pipeline per day. Must be greater than zero.

    Returns:
        float: Estimated run duration in decimal calendar days.

    Raises:
        ValueError: If `required_sample_size` or `daily_traffic` is less than or equal to zero.

    Examples:
        ??? example "Example"

            ```python
            >>> estimate_duration_days(required_sample_size=50000, daily_traffic=5000)
            10.0
            ```
    """
    if required_sample_size <= 0:
        raise ValueError("required_sample_size must be greater than zero.")
    if daily_traffic <= 0:
        raise ValueError("daily_traffic must be greater than zero.")
    return required_sample_size / daily_traffic
