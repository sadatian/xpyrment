class SequentialInference:
    """Computes sequential monitoring bounds and always-valid confidence intervals."""

    def calculate_always_valid_ci(self, sample_size: int, alpha: float) -> tuple:
        """Computes continuous monitoring boundaries to prevent alpha inflation from peeking."""
        # TODO: Implement sequential boundary functions (mSPRT or Pocock)
        return (-1.0, 1.0)
