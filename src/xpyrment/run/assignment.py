"""Live experiment assignment tracking and first-touch deduplication.

This module provides the `AssignmentLogger` class, which handles the operational logging of unit exposures to
variants, enforcing first-touch attribution to ensure correct temporal ordering for causal inference.
"""

import pandas as pd


class AssignmentLogger:
    r"""Manages tracking live assignment exposures and deduplicating multiple logs.

    In live online systems, users can trigger assignment events repeatedly (e.g., refreshing a page or navigating
    back to a screen). For rigorous statistical evaluation, we must identify and lock the exact moment of
    *initial exposure* for each unit.

    ??? mathbox "First-Touch Attribution and Causal Ordering"

        To establish a valid causal relationship, any metric event $Y$ must occur *after* the initial exposure
        to the treatment $T$:
        $$
        t_{\text{metric}} \ge t_{\text{initial\_exposure}}
        $$
        If we attribute a user's metric events to their assignment using a later exposure timestamp, we violate this
        temporal sequence, potentially including pre-treatment behavior in our post-treatment metric calculations,
        introducing severe selection bias.
        
        The `AssignmentLogger` records all exposure events and deduplicates them by sorting chronologically and retaining
        only the earliest occurrence per unit.

    Pseudocode for first-touch deduplication:
        ```text
        function get_deduplicated_exposures(exposures_list):
            1. Convert exposures_list to DataFrame.
            2. Sort DataFrame by "timestamp" in ascending order.
            3. Drop duplicate rows where "unit_id" is identical, keeping the first occurrence.
            4. Return the cleaned DataFrame.
        ```
    """

    def __init__(self):
        """Initializes an empty AssignmentLogger."""
        self._exposures = []

    def log_assignment(self, unit_id: str, variant: str, timestamp: str):
        """Logs an assignment exposure event.

        Args:
            unit_id (str): Unique identifier of the experimental unit (e.g. user_id, cookie_id).
            variant (str): The assigned variant label (e.g. "control", "treatment_a").
            timestamp (str): ISO-8601 or epoch timestamp when the exposure occurred.
        """
        self._exposures.append({
            "unit_id": unit_id,
            "variant": variant,
            "timestamp": timestamp
        })

    def get_deduplicated_exposures(self) -> pd.DataFrame:
        """Deduplicates multiple logs of the same unit_id, keeping only the earliest exposure.

        This enforces the first-touch exposure model, establishing a robust temporal baseline for
        subsequent metric windowing and causal analysis.

        Returns:
            pd.DataFrame: A pandas DataFrame containing unique `unit_id` rows, with their original
                earliest `variant` and `timestamp`.
        """
        df = pd.DataFrame(self._exposures)
        if df.empty:
            return df
        return df.sort_values("timestamp").drop_duplicates(subset=["unit_id"], keep="first")
