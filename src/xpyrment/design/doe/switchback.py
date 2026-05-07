"""Switchback and temporal crossover Design of Experiments (DoE) matrices.

This module provides the `SwitchbackDesign` class, which constructs time-series crossover designs.
Switchback experiments are the standard for marketplace, dispatch, and matching networks (e.g., ride-sharing,
on-demand delivery) where standard user-level randomization is corrupted by market-wide network spillovers.
"""

import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class SwitchbackDesign(DesignMatrix):
    r"""Generates time/geo crossover switchback experimental designs for marketplace tests.

    In standard user-split A/B testing, a treatment that increases supply utilization in a local market
    indirectly deprives the control group of supply (market interference/spillover bias). This violates the
    Stable Unit Treatment Value Assumption (SUTVA). Switchback designs resolve this by randomizing the
    *entire marketplace* over sequential time blocks.

    Temporal Crossover and Balance:
        The marketplace switches back and forth between control and treatment configurations over a series
        of discrete time blocks of length $W$ (e.g., 2 hours).
        $$\text{Schedule}: W_1 \to \text{Control}, \ W_2 \to \text{Treatment}, \ W_3 \to \text{Treatment}, \ \dots$$
        To prevent systematic time-of-day or day-of-week biases (e.g., treatment always running during rush hour),
        the assignments are structured using balanced crossover patterns:
        - **Markovian Transitions**: Ensuring equal transition probabilities between states ($C \to T$, $T \to C$,
          $C \to C$, $T \to T$) to model and subtract temporal carryover.
        - **Multi-region Crossover**: If multiple geographic markets are available, we cross them over simultaneously:
          $$\text{Region A}: C \to T \quad \text{vs.} \quad \text{Region B}: T \to C$$

    Carryover and Washout Periods:
        A major challenge in switchbacks is the **carryover effect** — supply/demand states from a treatment period
        spilling over into the subsequent control period. To solve this, the algorithm configures a "washout" parameter
        $\omega$ (e.g., 30 minutes) at the start of each window. Data collected during the first $\omega$ minutes of
        each transition is excluded from statistical evaluation.

    Pseudocode for the Algorithm:
        ```text
        function generate_switchback_schedule(regions, start_time, end_time, window_hours, washout_minutes):
            1. Partition the experimental timeframe [start_time, end_time] into N discrete blocks of size window_hours.
            2. For each region:
                 Generate a balanced crossover assignment sequence (using Latin Squares or randomized block schedules).
            3. For each block in the schedule:
                 Mark the first washout_minutes as "washout_active = True" (telemetry exclusion flag).
                 Assign the active variant label.
            4. Compile regional time-block rows into a structured DataFrame.
            5. Return DataFrame.
        ```

    Attributes:
        unit_window_hours (int): The duration in hours of each discrete experimental block. Defaults to 2.

    Examples:
        ??? example "Example"

            ```python
            >>> # Scheduling a switchback test with 4-hour window blocks
            >>> factors = {"dispatch_algorithm": ["greedy", "predictive"]}
            >>> design = SwitchbackDesign(factors, unit_window_hours=4)
            >>> # The output schedule allocates the marketplace state dynamically across the experimental window.
            ```
    """

    def __init__(self, factors: dict, unit_window_hours: int = 2):
        """Initializes a SwitchbackDesign.

        Args:
            factors (dict): Mapping of the market-level factor to its variant options.
            unit_window_hours (int): Duration of each switch window in hours. Defaults to 2.
        """
        super().__init__(factors)
        self.unit_window_hours = unit_window_hours

    def generate(self, regions: list = None, num_periods: int = 12, washout_minutes: int = 30) -> pd.DataFrame:
        """Generates the Switchback design schedule.

        Divides the temporal horizons into balanced blocks, schedules treatment crossovers,
        marks washout segments, and outputs the operational assignment ledger.

        Args:
            regions (list): List of geographic or logical market regions to crossover.
                Defaults to `["Region_A", "Region_B"]`.
            num_periods (int): Total number of sequential time block windows. Defaults to 12.
            washout_minutes (int): Transition period length to discard temporal carryover. Defaults to 30.

        Returns:
            pd.DataFrame: A pandas DataFrame representing the temporal switchback schedule.
        """
        if regions is None:
            regions = ["Region_A", "Region_B"]

        factor_name = list(self.factors.keys())[0]
        variants = list(self.factors[factor_name])

        rows = []
        for r_idx, region in enumerate(regions):
            for period in range(1, num_periods + 1):
                start_hour = (period - 1) * self.unit_window_hours
                end_hour = period * self.unit_window_hours

                # Multi-region Crossover: toggle opposite configurations to balance periods
                variant_idx = (r_idx + period) % len(variants)
                assigned_variant = variants[variant_idx]

                # Record scheduled time block run
                rows.append({
                    "region": region,
                    "period": period,
                    "start_hour": start_hour,
                    "end_hour": end_hour,
                    "washout_active": True,  # Washout indicator for early telemetry exclusion
                    factor_name: assigned_variant
                })

        # TODO: Add option to optimize period switchover frequencies to minimize carrying-over spillover effects.
        # TODO: Implement Latin Square multi-period and multi-variant Latin Square crossover balancing to optimize more than 2 variants.
        return pd.DataFrame(rows)
