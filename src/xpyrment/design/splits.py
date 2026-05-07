"""Traffic split coordinators, holdout groups, and exposure ramp schedules.

This module provides the `TrafficSplitter` class, which manages how traffic is divided between
multiple active variants, controls global experimental holdout structures, and configures step-wise
treatment exposure ramp schedules.
"""

from typing import Dict, List


class TrafficSplitter:
    """Manages traffic split fractions, global holdout configurations, and progressive exposure ramp schedules.

    A `TrafficSplitter` divides incoming traffic into discrete experimental buckets. It supports fractional
    allocations, standard control and treatment arms, and long-term holdout groups.

    Business and Operational Context:
        1. **Fractional Allocations**: Allows unequal variants (e.g., 90% control, 10% treatment) to limit
           exposure during early launch states.
        2. **Long-Term Holdouts**: A portion of users can be entirely withheld from all experiments within a product area
           (the "holdout group") to measure the long-term cumulative effects and potential interaction effects of
           independent product changes over months.
        3. **Exposure Ramp Schedules**: Step-wise exposure schedules (e.g., exposing 1% of users, then 10%, then 50%,
           then 100%) act as standard release gates. This mitigates operational risk by validating telemetry
           and checking for guardrail breaches before exposing the entire user base.

    Attributes:
        allocations (Dict[str, float]): A dictionary mapping variant names to their allocation weight
            fractions (values bounded in $[0, 1]$).
        holdout_percentage (float): Percentage of global traffic completely excluded from selection
            and routed to a static holdout arm. Bounded in $[0, 1]$.

    Example:
        >>> allocations = {"control": 0.45, "treatment": 0.45}
        >>> splitter = TrafficSplitter(allocations=allocations, holdout_percentage=0.10)
        >>> splitter.holdout_percentage
        0.1
    """

    def __init__(self, allocations: Dict[str, float], holdout_percentage: float = 0.0):
        """Initializes a new TrafficSplitter container.

        Validates that the sum of variant allocations and holdout percentages totals exactly 1.0.

        Args:
            allocations (Dict[str, float]): Mapping of variant labels to active traffic split weights.
            holdout_percentage (float): Fractional traffic diverted into a holdout group. Defaults to 0.0.

        Raises:
            ValueError: If any individual allocation weight is negative.
            ValueError: If the total allocation weight including the holdout percentage does not sum to 1.0.
        """
        self.allocations = allocations
        self.holdout_percentage = holdout_percentage

        # Validate bounds
        if holdout_percentage < 0.0 or holdout_percentage > 1.0:
            raise ValueError("holdout_percentage must be between 0.0 and 1.0.")
        for variant, weight in allocations.items():
            if weight < 0.0 or weight > 1.0:
                raise ValueError(f"Allocation for variant '{variant}' must be between 0.0 and 1.0.")

        total_alloc = sum(allocations.values()) + holdout_percentage
        if abs(total_alloc - 1.0) > 1e-5:
            raise ValueError("Total allocations including holdout must equal 1.0.")

    def get_ramp_schedule(self) -> List[float]:
        r"""Generates progressive exposure ramp-up schedule coordinates.

        Ramping exposure represents a crucial risk-management strategy. This method returns a list of active exposure
        fractions defining sequential release gating stages.

        Mathematical Representation:
            Let $R$ be the ramp-up schedule array:
            $$R = [r_1, r_2, \dots, r_m]$$
            where $r_j \in [0, 1]$ represents the fraction of total traffic exposed to the experiment during step $j$,
            such that $r_j \le r_{j+1}$.

        Returns:
            List[float]: Ordered list of target traffic fractions representing progressive exposure stages.
        """
        # TODO: Implement full ramp-up schedule generator
        return [0.01, 0.10, 0.50, 1.0]

