class StoppingRules:
    """Implements early-stopping rules like mixture sequential probability ratio tests (mSPRT)."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def check_msprt_stop(self, lambda_value: float) -> bool:
        """Determines if the mSPRT likelihood ratio exceeds the safety stopping bounds."""
        # Boundaries typically equal 1 / alpha
        boundary = 1.0 / self.alpha
        return lambda_value > boundary
