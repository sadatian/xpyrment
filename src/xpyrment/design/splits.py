from typing import Dict, List


class TrafficSplitter:
    """Manages traffic fractions, holdout configurations, and ramp-up schedules."""

    def __init__(self, allocations: Dict[str, float], holdout_percentage: float = 0.0):
        self.allocations = allocations
        self.holdout_percentage = holdout_percentage
        total_alloc = sum(allocations.values()) + holdout_percentage
        if abs(total_alloc - 1.0) > 1e-5:
            raise ValueError("Total allocations including holdout must equal 1.0.")

    def get_ramp_schedule(self) -> List[float]:
        """Returns the ramp schedule configuration."""
        # TODO: Implement full ramp-up schedule generator
        return [0.01, 0.10, 0.50, 1.0]
