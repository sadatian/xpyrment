import pandas as pd
import pytest

from xpyrment.core.exceptions import PhaseOrderError
from xpyrment.core.state import ExperimentState
from xpyrment.core.experiment import Experiment
from xpyrment.core.registry import ExperimentRegistry
from xpyrment.metrics.taxonomy import MeanMetric
from xpyrment.plan.hypothesis import HypothesisSpec
from xpyrment.plan.preregistration import PreregistrationCard


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


def test_hypothesis_spec():
    """Tests hypothesis specification properties and directionality."""
    m = MeanMetric("CTR", value_col="clicks")
    h = HypothesisSpec(m, description="Increase CTR", direction="greater")
    assert h.direction == "greater"
    assert h.primary_metric.name == "CTR"


def test_preregistration_card():
    """Tests preregistration card verification logic."""
    card = PreregistrationCard("exp1", {"mde": 0.05, "power": 0.8})
    assert card.verify({"mde": 0.05, "power": 0.8}) is True
    assert card.verify({"mde": 0.05, "power": 0.9}) is False


def test_experiment_initialization_failures():
    """Verifies that Experiment raises ValueError for missing columns."""
    df = pd.DataFrame({"user_id": [1, 2], "group": ["control", "treatment"]})

    with pytest.raises(ValueError, match="Treatment column 'missing_col' not found"):
        Experiment(df, treatment_col="missing_col")

    with pytest.raises(ValueError, match="ID column 'missing_id' not found"):
        Experiment(df, treatment_col="group", id_col="missing_id")


def test_experiment_covariates():
    """Verifies add_covariates for both single name and lists, ensuring duplicate protection."""
    df = pd.DataFrame({"user_id": [1, 2], "group": ["control", "treatment"], "age": [30, 40], "income": [50000, 60000]})
    exp = Experiment(df, treatment_col="group", id_col="user_id")

    # 1. Add single covariate
    exp.add_covariates("age")
    assert exp.covariates == ["age"]

    # 2. Add duplicate single covariate (should be ignored)
    exp.add_covariates("age")
    assert exp.covariates == ["age"]

    # 3. Add list of covariates with duplicates
    exp.add_covariates(["age", "income", "income"])
    assert exp.covariates == ["age", "income"]


def test_experiment_metric_registrations():
    """Tests that register_metric correctly initializes and appends Mean, Proportion, and Ratio metrics."""
    df = pd.DataFrame({
        "user_id": [1, 2],
        "group": ["control", "treatment"],
        "rev": [10.0, 15.0],
        "pre_rev": [8.0, 12.0],
        "clicks": [1, 0],
        "num": [5, 6],
        "den": [10, 12],
        "pre_num": [4, 5],
        "pre_den": [10, 10]
    })
    exp = Experiment(df, treatment_col="group", id_col="user_id")

    # 1. Mean metric registration
    exp.register_metric("revenue", metric_type="mean", value_col="rev", covariate="pre_rev")
    assert len(exp.metrics) == 1
    assert exp.metrics[0].name == "revenue"
    assert exp.metrics[0].value_col == "rev"
    assert exp.metrics[0].pre_period_col == "pre_rev"

    # 2. Mean metric fallback to name
    exp.register_metric("rev", metric_type="mean")
    assert len(exp.metrics) == 2
    assert exp.metrics[1].value_col == "rev"

    # 3. Proportion metric registration
    exp.register_metric("ctr", metric_type="proportion", value_col="clicks")
    assert len(exp.metrics) == 3
    assert exp.metrics[2].name == "ctr"
    assert exp.metrics[2].value_col == "clicks"

    # 4. Ratio metric registration
    exp.register_metric(
        "rpc",
        metric_type="ratio",
        numerator_col="num",
        denominator_col="den",
        pre_numerator_col="pre_num",
        pre_denominator_col="pre_den"
    )
    assert len(exp.metrics) == 4
    assert exp.metrics[3].name == "rpc"
    assert exp.metrics[3].numerator_col == "num"
    assert exp.metrics[3].denominator_col == "den"
    assert exp.metrics[3].pre_numerator_col == "pre_num"
    assert exp.metrics[3].pre_denominator_col == "pre_den"

    # 5. Ratio metric failures (missing numerator/denominator)
    with pytest.raises(ValueError, match="Both 'numerator_col' and 'denominator_col' must be specified"):
        exp.register_metric("fail_ratio", metric_type="ratio", numerator_col="num")

    # 6. Unknown metric type failure
    with pytest.raises(ValueError, match="Unknown metric_type: 'invalid'"):
        exp.register_metric("fail_type", metric_type="invalid")

