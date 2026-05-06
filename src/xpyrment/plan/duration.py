def estimate_duration_days(required_sample_size: int, daily_traffic: int) -> float:
    """Estimates the experiment duration in days based on required size and traffic.

    Args:
        required_sample_size (int): Total samples needed.
        daily_traffic (int): Daily active users/units.

    Returns:
        float: Estimated duration in days.
    """
    if daily_traffic <= 0:
        raise ValueError("daily_traffic must be greater than zero.")
    return required_sample_size / daily_traffic
