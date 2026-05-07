"""State machine and phase-gating representation for experiment lifecycles.

This module provides the core enumeration of states that an experiment must transition through,
ensuring strict process discipline, reproducibility, and mathematical safety by preventing
post-hoc modification of hypotheses or designs after data ingestion or during analysis.
"""

from enum import Enum


class ExperimentState(Enum):
    """Enforces the phase-gated state of the experimental lifecycle.

    This enum acts as the source of truth for the state machine within `Experiment`. State transitions
    are restricted to a forward-only unidirectional progression, preventing common experimental
    malpractices such as p-hacking, retrospective hypothesis creation, or design tampering.

    State Diagram & Authorized Transitions:
        ```mermaid
        stateDiagram-v2
            [*] --> CREATED : Initialization
            CREATED --> PLANNED : add_metrics()
            PLANNED --> DESIGNED : design() / do_doe()
            DESIGNED --> RUNNING : start_experiment()
            RUNNING --> ANALYZED : run_analysis()
            ANALYZED --> ANALYZED : re-run (with frozen data)
            ANALYZED --> REPORTED : compile_report()
            REPORTED --> [*]
        ```

    States Description:
        - **CREATED**: The experiment has been instantiated with raw data and a designated treatment
          column. No configuration or planning has occurred yet.
        - **PLANNED**: Hypotheses have been bound, primary and secondary metrics have been assigned, and
          power calculation (required sample size and Minimum Detectable Effect) has been performed.
        - **DESIGNED**: Randomization scheme, traffic split fractions, ramp-up schedules, or classical
          factorial designs (DoE matrices) have been generated and locked.
        - **RUNNING**: The experiment is actively ingesting live assignment and telemetry data. Sequential
          monitoring (e.g., mSPRT) or early-stopping boundaries are actively checked.
        - **ANALYZED**: Ingested data is locked, and statistical inference engines (frequentist, Bayesian,
          or sequential) have run. CUPED variance reduction and multi-comparison corrections are finalized.
        - **REPORTED**: The full lifecycle audit trail, key metrics, and decision recommendations have been
          serialized into an immutable Experiment Card or exported (JSON/PDF).

    Exceptions & Gating:
        - Any attempt to transition backwards in the sequence (e.g., from `RUNNING` to `PLANNED` to add a new
          metric) will raise a `PhaseOrderError`.
        - Re-running analysis in the `ANALYZED` state to adjust statistical parameters (e.g., alpha, multiple
          comparison correction method) is authorized without violating state order rules, provided the
          underlying experimental design remains frozen.
    """

    CREATED = "CREATED"
    PLANNED = "PLANNED"
    DESIGNED = "DESIGNED"
    RUNNING = "RUNNING"
    ANALYZED = "ANALYZED"
    REPORTED = "REPORTED"

