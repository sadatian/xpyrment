def check_practical_significance(relative_lift: float, min_valuable_effect: float) -> bool:
    """Verifies if the measured lift satisfies the minimal valuable business effect (MVE)."""
    return relative_lift >= min_valuable_effect
