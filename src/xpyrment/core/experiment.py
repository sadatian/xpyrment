"""Central orchestrator for experiment setup, configuration, and phase management.

This module houses the `Experiment` class, which acts as the core controller and lifecycle
coordinator for both industrial A/B testing and classical Design of Experiments. It encapsulates
the underlying pandas DataFrame, tracks added metrics, and enforces phase-gating rules using
the `ExperimentState` state machine.
"""

from typing import List, Optional, Union
import pandas as pd

from xpyrment.core.exceptions import PhaseOrderError
from xpyrment.core.state import ExperimentState
from xpyrment.metrics.taxonomy import BaseMetric


class Experiment:
    """The central orchestration class for setting up, configuring, and executing experiments.

    The `Experiment` class binds the experimental dataset, defines treatment structures, maps
    the metric taxonomy, and strictly enforces state transitions across the execution lifecycle.
    Through the state-machine rules, it ensures that all calculations are performed sequentially
    and reproducibly, eliminating retrospective tampering or incorrect state usage.

    Attributes:
        data (pd.DataFrame): A copy of the input DataFrame containing assignments and telemetry.
        treatment_col (str): The column in `data` identifying the treatment arm assignments.
        id_col (Optional[str]): The column in `data` representing unique unit IDs.
        metrics (List[BaseMetric]): List of metrics registered for statistical calculation.
        state (ExperimentState): The current lifecycle phase of the experiment.

    State Gating Mechanism:
        Execution functions across downstream submodules verify that the experiment is in the
        appropriate state before proceeding. For example, running power analysis transitions the
        state from `CREATED` to `PLANNED`. Running randomization moves from `PLANNED` to `DESIGNED`.
        Analyzing results requires a transition to `ANALYZED`.

    Examples:
        ??? example "Example"

            ```python
            >>> import pandas as pd
            >>> from xpyrment import Experiment
            >>> from xpyrment.metrics.taxonomy import MeanMetric
            >>> df = pd.DataFrame({"user_id": [1, 2, 3], "group": ["control", "treatment", "control"], "revenue": [10.5, 12.0, 9.5]})
            >>> exp = Experiment(df, treatment_col="group", id_col="user_id")
            >>> exp.state
            <ExperimentState.CREATED: 'CREATED'>
            >>> metric = MeanMetric("Revenue Metric", value_col="revenue")
            >>> exp.add_metrics(metric)
            >>> exp.state
            <ExperimentState.PLANNED: 'PLANNED'>
            ```
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        id_col: Optional[str] = None,
        covariates: Optional[List[str]] = None,
    ):
        """Initializes a new Experiment orchestration container.

        Copies the input DataFrame to guarantee immutability of the source dataset during internal
        state transitions and potential data transformations (e.g., CUPED alignment or log scaling).

        Args:
            data (pd.DataFrame): The source DataFrame containing unit-level data.
            treatment_col (str): Name of the column designating experimental groups/arms.
            id_col (Optional[str]): Name of the column containing unique identifiers for each experimental unit.
                Required for certain operations like sequential analysis and user assignments.
            covariates (Optional[List[str]]): List of baseline covariates for balance checking or adjustments.

        Raises:
            ValueError: If `treatment_col` or `id_col` is not found in the input DataFrame columns.
        """
        self.data = data.copy()
        self.treatment_col = treatment_col
        self.id_col = id_col
        self.metrics: List[BaseMetric] = []
        self.covariates: List[str] = covariates or []
        self.metric_registry: Optional[Any] = None
        self.state = ExperimentState.CREATED

        if treatment_col not in self.data.columns:
            raise ValueError(f"Treatment column '{treatment_col}' not found in DataFrame.")
        if id_col and id_col not in self.data.columns:
            raise ValueError(f"ID column '{id_col}' not found in DataFrame.")

    def transition_to(self, target_state: ExperimentState) -> None:
        r"""Enforces transition logic to guarantee the phase-gated execution flow.

        Uses the ordinal indices of `ExperimentState` members to verify that the transition is
        monotonically increasing (forward-only).

        Mathematical/Logical Representation:
            Let $S$ be the ordered tuple of states:
            $$
            S = (\text{CREATED}, \text{PLANNED}, \text{DESIGNED}, \text{RUNNING}, \text{ANALYZED}, \text{REPORTED})
            $$
            A state transition from state $s_1$ to state $s_2$ is valid if and only if:
            $$
            \text{Index}(s_2) \ge \text{Index}(s_1)
            $$
            with a special exemption permitting $s_1 = \text{ANALYZED} \rightarrow s_2 = \text{ANALYZED}$ to support
            re-running statistical engines on the locked design data.

        Args:
            target_state (ExperimentState): The state the experiment is attempting to transition into.

        Raises:
            PhaseOrderError: If a backwards state transition is attempted, or if transition is otherwise unauthorized.
        """
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
        """Adds statistical metrics to the experiment configuration.

        Successfully registering a metric moves the experiment from `CREATED` to `PLANNED` state, representing
        that the evaluation criteria have been defined prior to running designs, validations, or analyses.

        Args:
            metrics (Union[BaseMetric, List[BaseMetric]]): A single metric object or a list of metrics
                (inheriting from `BaseMetric`) to bind to the experiment lifecycle.

        Returns:
            Experiment: The experiment instance itself (for fluent API chaining).

        Raises:
            PhaseOrderError: If the experiment has already progressed past the `PLANNED` phase. This restriction
                prevents retrospectively adding metrics to match statistical noise (post-hoc metrics selection/p-hacking).
        """
        if self.state not in [ExperimentState.CREATED, ExperimentState.PLANNED]:
            raise PhaseOrderError(
                f"Cannot add metrics while in state {self.state}. Must be in CREATED or PLANNED."
            )

        if isinstance(metrics, list):
            self.metrics.extend(metrics)
        else:
            self.metrics.append(metrics)

        # Transition the experiment from CREATED to PLANNED if metrics are added
        if self.state == ExperimentState.CREATED:
            self.transition_to(ExperimentState.PLANNED)

        return self

    def add_covariates(self, names: Union[str, List[str]]) -> "Experiment":
        """Adds baseline covariates to the experiment configuration.

        Args:
            names (Union[str, List[str]]): A single covariate column name or a list of names.

        Returns:
            Experiment: The experiment instance itself (for fluent API chaining).
        """
        if isinstance(names, list):
            for name in names:
                if name not in self.covariates:
                    self.covariates.append(name)
        else:
            if names not in self.covariates:
                self.covariates.append(names)
        return self

    def register_metric(
        self,
        name: str,
        metric_type: str = "mean",
        value_col: Optional[str] = None,
        covariate: Optional[str] = None,
        numerator_col: Optional[str] = None,
        denominator_col: Optional[str] = None,
        pre_numerator_col: Optional[str] = None,
        pre_denominator_col: Optional[str] = None,
    ) -> "Experiment":
        """Conveniently registers a metric and appends it to the configuration.

        Args:
            name (str): Unique descriptive name of the metric.
            metric_type (str): Type of metric. Options: "mean", "proportion", "ratio". Defaults to "mean".
            value_col (str, optional): Column name containing experiment period values. Defaults to the metric name.
            covariate (str, optional): Pre-period covariate column name for CUPED. Defaults to None.
            numerator_col (str, optional): Column containing numerator values for RatioMetric.
            denominator_col (str, optional): Column containing denominator values for RatioMetric.
            pre_numerator_col (str, optional): Pre-period numerator column for RatioMetric CUPED.
            pre_denominator_col (str, optional): Pre-period denominator column for RatioMetric CUPED.

        Returns:
            Experiment: The experiment instance itself (for fluent API chaining).
        """
        # Resolve imports lazily to prevent circular imports
        from xpyrment.metrics.taxonomy import MeanMetric, ProportionMetric, RatioMetric

        m_type = metric_type.lower()
        if m_type == "mean":
            col = value_col if value_col is not None else name
            metric = MeanMetric(name, value_col=col, pre_period_col=covariate)
        elif m_type == "proportion":
            col = value_col if value_col is not None else name
            metric = ProportionMetric(name, value_col=col, pre_period_col=covariate)
        elif m_type == "ratio":
            if numerator_col is None or denominator_col is None:
                raise ValueError("Both 'numerator_col' and 'denominator_col' must be specified for ratio metrics.")
            metric = RatioMetric(
                name,
                numerator_col=numerator_col,
                denominator_col=denominator_col,
                pre_numerator_col=pre_numerator_col,
                pre_denominator_col=pre_denominator_col,
            )
        else:
            raise ValueError(f"Unknown metric_type: '{metric_type}'. Expected 'mean', 'proportion', or 'ratio'.")

        return self.add_metrics(metric)
