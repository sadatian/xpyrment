import pandas as pd
import pytest

from xpyrment.core.exceptions import PhaseOrderError
from xpyrment.core.state import ExperimentState
from xpyrment.core.experiment import Experiment
from xpyrment.core.registry import ExperimentRegistry
from xpyrment.metrics.taxonomy import MeanMetric


def test_experiment_state_transitions():
    """Validates monotonic state transitions and phase gating constraints."""
    df = pd.DataFrame({"user_id": [1, 2, 3], "group": ["control", "treatment", "control"], "revenue": [10.0, 15.0, 11.0]})
    exp = Experiment(df, treatment_col="group", id_col="user_id")

    # Initial state must be CREATED
    assert exp.state == ExperimentState.CREATED

    # Adding a metric transitions state automatically to PLANNED
    metric = MeanMetric("Revenue Metric", value_col="revenue")
    exp.add_metrics(metric)
    assert exp.state == ExperimentState.PLANNED
    assert len(exp.metrics) == 1

    # Transitioning forward is allowed
    exp.transition_to(ExperimentState.DESIGNED)
    assert exp.state == ExperimentState.DESIGNED

    exp.transition_to(ExperimentState.RUNNING)
    assert exp.state == ExperimentState.RUNNING

    exp.transition_to(ExperimentState.ANALYZED)
    assert exp.state == ExperimentState.ANALYZED

    # Special exemption: ANALYZED -> ANALYZED is allowed
    exp.transition_to(ExperimentState.ANALYZED)
    assert exp.state == ExperimentState.ANALYZED

    exp.transition_to(ExperimentState.REPORTED)
    assert exp.state == ExperimentState.REPORTED

    # Monotonic forward-only check: cannot transition backwards
    with pytest.raises(PhaseOrderError):
        exp.transition_to(ExperimentState.CREATED)

    with pytest.raises(PhaseOrderError):
        exp.transition_to(ExperimentState.PLANNED)


def test_metric_addition_restrictions():
    """Asserts that adding metrics past the PLANNED state raises an error."""
    df = pd.DataFrame({"user_id": [1, 2, 3], "group": ["control", "treatment", "control"], "revenue": [10.0, 15.0, 11.0]})
    exp = Experiment(df, treatment_col="group", id_col="user_id")

    metric1 = MeanMetric("Revenue1", value_col="revenue")
    metric2 = MeanMetric("Revenue2", value_col="revenue")

    # Adding in CREATED is fine
    exp.add_metrics(metric1)
    assert exp.state == ExperimentState.PLANNED

    # Adding in PLANNED is fine
    exp.add_metrics(metric2)
    assert len(exp.metrics) == 2

    # Moving to DESIGNED
    exp.transition_to(ExperimentState.DESIGNED)

    # Adding metrics in DESIGNED or later must fail (prevents post-hoc metric selecting)
    with pytest.raises(PhaseOrderError):
        exp.add_metrics(MeanMetric("PostHoc", value_col="revenue"))


def test_experiment_registry_hashing():
    """Tests SHA-256 pre-registration specification verification."""
    registry = ExperimentRegistry()
    spec = {
        "primary_metric": "conversion_rate",
        "alpha": 0.05,
        "target_n": 10000,
        "metrics": ["conversion_rate", "revenue_per_user"]
    }

    # Registering returns a valid 64-character SHA-256 hex digest
    spec_hash = registry.register_spec("EXP-101", spec)
    assert isinstance(spec_hash, str)
    assert len(spec_hash) == 64

    # Verification with identical dict is successful
    assert registry.verify_spec("EXP-101", spec) is True

    # Mutated config must fail verification
    mutated_spec = spec.copy()
    mutated_spec["alpha"] = 0.01
    assert registry.verify_spec("EXP-101", mutated_spec) is False

    # Verifying unregistered ID must return False
    assert registry.verify_spec("EXP-999", spec) is False
