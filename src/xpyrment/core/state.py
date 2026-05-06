from enum import Enum


class ExperimentState(Enum):
    """Enforces the phase-gated state of the experimental lifecycle."""

    CREATED = "CREATED"
    PLANNED = "PLANNED"
    DESIGNED = "DESIGNED"
    RUNNING = "RUNNING"
    ANALYZED = "ANALYZED"
    REPORTED = "REPORTED"
