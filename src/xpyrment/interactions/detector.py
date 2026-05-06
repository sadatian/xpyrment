class InteractionDetector:
    """Dispatches multi-factor, covariate-treatment, and non-linear interactions."""

    def __init__(self, experiment):
        self.experiment = experiment

    def detect_all(self) -> dict:
        """Runs ANOVA and regression checks to identify interaction terms."""
        # TODO: Implement dispatcher
        return {}
