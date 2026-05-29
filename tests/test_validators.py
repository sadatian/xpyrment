import pytest
import numpy as np
import pandas as pd
from xpyrment.core.exceptions import BoundaryValidationError
from xpyrment.core.validators import (
    assert_probability,
    assert_positive_int,
    assert_non_empty,
    assert_finite,
)
from xpyrment.validate.srm import check_srm
from xpyrment.core.experiment import Experiment
from xpyrment.plan.power import design_experiment


def test_boundary_validation_error_subclass():
    """Asserts that BoundaryValidationError inherits from ValueError for backward compatibility."""
    assert issubclass(BoundaryValidationError, ValueError)
    try:
        raise BoundaryValidationError("test error")
    except ValueError as e:
        assert str(e) == "test error"


def test_assert_probability():
    """Verifies that assert_probability validates range [0.0, 1.0] and handles bad types."""
    # Valid probabilities
    assert_probability(0.0, "p")
    assert_probability(0.5, "p")
    assert_probability(1.0, "p")
    assert_probability("0.25", "p")

    # Invalid probabilities
    with pytest.raises(BoundaryValidationError, match="must be in range"):
        assert_probability(-0.01, "p")
    with pytest.raises(BoundaryValidationError, match="must be in range"):
        assert_probability(1.01, "p")

    # None and bad types
    with pytest.raises(BoundaryValidationError, match="cannot be None"):
        assert_probability(None, "p")
    with pytest.raises(BoundaryValidationError, match="must be a float"):
        assert_probability("not a float", "p")


def test_assert_positive_int():
    """Verifies assert_positive_int validates positive integers and handles bad types/floats/booleans."""
    # Valid positive integers
    assert_positive_int(1, "n")
    assert_positive_int(100, "n")
    assert_positive_int(np.int64(50), "n")

    # Zero or negative
    with pytest.raises(BoundaryValidationError, match="must be positive"):
        assert_positive_int(0, "n")
    with pytest.raises(BoundaryValidationError, match="must be positive"):
        assert_positive_int(-5, "n")

    # Bad types
    with pytest.raises(BoundaryValidationError, match="cannot be None"):
        assert_positive_int(None, "n")
    with pytest.raises(BoundaryValidationError, match="must be an integer"):
        assert_positive_int(1.5, "n")
    with pytest.raises(BoundaryValidationError, match="must be an integer"):
        assert_positive_int(True, "n")  # Booleans are subclasses of int in Python but should fail
    with pytest.raises(BoundaryValidationError, match="must be an integer"):
        assert_positive_int("10", "n")


def test_assert_non_empty():
    """Verifies assert_non_empty detects empty arrays, dataframes, collections, and strings."""
    # Valid non-empty structures
    assert_non_empty([1], "list")
    assert_non_empty((1, 2), "tuple")
    assert_non_empty("hello", "str")
    assert_non_empty(np.array([1, 2, 3]), "numpy")
    assert_non_empty(pd.Series([1.0]), "series")
    assert_non_empty(pd.DataFrame({"a": [1]}), "dataframe")

    # Empty structures
    with pytest.raises(BoundaryValidationError, match="cannot be empty"):
        assert_non_empty([], "list")
    with pytest.raises(BoundaryValidationError, match="cannot be empty"):
        assert_non_empty((), "tuple")
    with pytest.raises(BoundaryValidationError, match="cannot be empty"):
        assert_non_empty("", "str")
    with pytest.raises(BoundaryValidationError, match="cannot be empty"):
        assert_non_empty(np.array([]), "numpy")
    with pytest.raises(BoundaryValidationError, match="cannot be empty"):
        assert_non_empty(pd.Series([], dtype=float), "series")
    with pytest.raises(BoundaryValidationError, match="cannot be empty"):
        assert_non_empty(pd.DataFrame(), "dataframe")

    # None
    with pytest.raises(BoundaryValidationError, match="cannot be None"):
        assert_non_empty(None, "structure")


def test_assert_finite():
    """Verifies assert_finite detects NaN or Inf in numpy arrays, pandas, and lists."""
    # Valid finite inputs
    assert_finite([1, 2, 3.5], "list")
    assert_finite(np.array([1.0, 2.0, 3.0]), "numpy")
    assert_finite(pd.Series([1.0, -2.5]), "series")
    assert_finite(pd.DataFrame({"a": [1.0, 2.0]}), "dataframe")

    # NaN presence
    with pytest.raises(BoundaryValidationError, match="contains NaN values"):
        assert_finite([1.0, np.nan, 3.0], "list")
    with pytest.raises(BoundaryValidationError, match="contains NaN values"):
        assert_finite(np.array([1.0, np.nan, 3.0]), "numpy")
    with pytest.raises(BoundaryValidationError, match="contains NaN values"):
        assert_finite(pd.Series([1.0, np.nan, 3.0]), "series")
    with pytest.raises(BoundaryValidationError, match="contains NaN values"):
        assert_finite(pd.DataFrame({"a": [1.0, np.nan]}), "dataframe")

    # Inf presence
    with pytest.raises(BoundaryValidationError, match="contains infinite"):
        assert_finite([1.0, np.inf, 3.0], "list")
    with pytest.raises(BoundaryValidationError, match="contains infinite"):
        assert_finite(np.array([1.0, -np.inf, 3.0]), "numpy")
    with pytest.raises(BoundaryValidationError, match="contains infinite"):
        assert_finite(pd.Series([1.0, np.inf, 3.0]), "series")
    with pytest.raises(BoundaryValidationError, match="contains infinite"):
        assert_finite(pd.DataFrame({"a": [1.0, np.inf]}), "dataframe")

    # None
    with pytest.raises(BoundaryValidationError, match="cannot be None"):
        assert_finite(None, "arr")


def test_check_srm_raises_boundary_validation_error():
    """Asserts that check_srm raises BoundaryValidationError on invalid inputs."""
    with pytest.raises(BoundaryValidationError, match="Length of observed_counts and expected_ratios must be equal"):
        check_srm(observed_counts=[100, 100], expected_ratios=[0.5])

    with pytest.raises(BoundaryValidationError, match="must be non-negative"):
        check_srm(observed_counts=[100, -5], expected_ratios=[0.5, 0.5])

    with pytest.raises(BoundaryValidationError, match="must be strictly positive"):
        check_srm(observed_counts=[100, 100], expected_ratios=[0.5, 0.0])


def test_experiment_init_raises_boundary_validation_error():
    """Asserts that Experiment constructor raises BoundaryValidationError on out-of-bounds configurations."""
    df = pd.DataFrame({"user_id": [1, 2], "group": ["control", "treatment"]})

    with pytest.raises(BoundaryValidationError, match="cannot be empty"):
        Experiment(pd.DataFrame(), "group", "user_id")

    with pytest.raises(BoundaryValidationError, match="Treatment column .* not found"):
        Experiment(df, "invalid_column", "user_id")

    with pytest.raises(BoundaryValidationError, match="ID column .* not found"):
        Experiment(df, "group", "invalid_id")


def test_design_experiment_raises_boundary_validation_error():
    """Asserts that design_experiment raises BoundaryValidationError on invalid inputs."""
    # Invalid alpha/power
    with pytest.raises(BoundaryValidationError, match="Parameter 'alpha' must be in range"):
        design_experiment(metric_type="mean", baseline_value=10.0, standard_deviation=2.0, alpha=-0.05)

    with pytest.raises(BoundaryValidationError, match="Parameter 'power' must be in range"):
        design_experiment(metric_type="mean", baseline_value=10.0, standard_deviation=2.0, power=1.5)

    # Invalid proportion baseline
    with pytest.raises(BoundaryValidationError, match="For proportions, baseline_value must be strictly between 0 and 1"):
        design_experiment(metric_type="proportion", baseline_value=1.5)

    # Invalid daily traffic
    with pytest.raises(BoundaryValidationError, match="must be positive"):
        design_experiment(metric_type="proportion", baseline_value=0.1, daily_traffic=0)
