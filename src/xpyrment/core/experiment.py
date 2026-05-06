from typing import List, Optional, Union
import pandas as pd

from xpyrment.core.exceptions import PhaseOrderError
from xpyrment.core.state import ExperimentState
from xpyrment.metrics.taxonomy import BaseMetric


class Experiment:
    """The central orchestration class for setting up and running experiments.

    Enforces strict state transitions across the experimental lifecycle.
    """

    def __init__(self, data: pd.DataFrame, treatment_col: str, id_col: Optional[str] = None):
        self.data = data.copy()
        self.treatment_col = treatment_col
        self.id_col = id_col
        self.metrics: List[BaseMetric] = []
        self.state = ExperimentState.CREATED

        if treatment_col not in self.data.columns:
            raise ValueError(f"Treatment column '{treatment_col}' not found in DataFrame.")
        if id_col and id_col not in self.data.columns:
            raise ValueError(f"ID column '{id_col}' not found in DataFrame.")

    def transition_to(self, target_state: ExperimentState):
        """Enforces state transition logic to guarantee the phase-gated execution flow."""
        current_val = list(ExperimentState).index(self.state)
        target_val = list(ExperimentState).index(target_state)

        # Allow transitioning forward, or re-running analysis
        if target_val < current_val and not (
            self.state == ExperimentState.ANALYZED and target_state == ExperimentState.ANALYZED
        ):
            raise PhaseOrderError(
                f"Cannot transition backwards from {self.state} to {target_state}."
            )

        self.state = target_state

    def add_metrics(self, metrics: Union[BaseMetric, List[BaseMetric]]) -> "Experiment":
        """Adds metrics to the experiment. Allowed in CREATED and PLANNED phases."""
        if self.state not in [ExperimentState.CREATED, ExperimentState.PLANNED]:
            raise PhaseOrderError(
                f"Cannot add metrics while in state {self.state}. Must be in CREATED or PLANNED."
            )

        if isinstance(metrics, list):
            self.metrics.extend(metrics)
        else:
            self.metrics.append(metrics)

        return self
