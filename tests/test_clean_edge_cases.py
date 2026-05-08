import numpy as np
import pytest

from xpyrment.validate.clean import clean_array, validate_estimation_inputs, verify_collinearity
from xpyrment.quasi.diff_in_diff import DifferenceInDifferences
from xpyrment.quasi.instrumental_variables import InstrumentalVariables2SLS


def test_clean_array_basic():
    """Validates that clean_array processes standard arrays and raises correct errors on extreme inputs."""
    # Standard array
    arr = np.array([1.5, 2.5, 3.5])
    cleaned = clean_array(arr)
    assert np.array_equal(cleaned, arr)

    # Empty array
    with pytest.raises(ValueError, match="Input 'test_empty' is empty"):
        clean_array(np.array([]), name="test_empty")

    # None array
    with pytest.raises(ValueError, match="cannot be None"):
        clean_array(None)

    # NaNs
    nan_arr = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="contains NaN values"):
        clean_array(nan_arr, allow_nan=False)

    # Inf
    inf_arr = np.array([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="contains infinite"):
        clean_array(inf_arr, allow_inf=False)

    # Zero Variance
    const_arr = np.array([5.0, 5.0, 5.0])
    with pytest.raises(ValueError, match="has zero variance"):
        clean_array(const_arr, allow_zero_var=False, name="test_const")


def test_validate_estimation_inputs_mismatches():
    """Asserts that validate_estimation_inputs checks length matching and sample counts."""
    y = np.array([1, 2, 3])
    x = np.array([1, 2])  # Mismatch

    with pytest.raises(ValueError, match="Length mismatch"):
        validate_estimation_inputs(y, x)

    # Insufficient samples
    y_short = np.array([1.0])
    with pytest.raises(ValueError, match="Insufficient samples"):
        validate_estimation_inputs(y_short, min_samples=2)


def test_verify_collinearity_singular():
    """Asserts that verify_collinearity catches perfect collinearity and rank-deficiency."""
    # Singular design matrix (column 2 is exactly column 1 multiplied by 2)
    X_singular = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ])

    with pytest.raises(ValueError, match="Design matrix is singular or has perfect multicollinearity"):
        verify_collinearity(X_singular)

    # Underdetermined system
    X_under = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    with pytest.raises(ValueError, match="Underdetermined system"):
        verify_collinearity(X_under)


def test_did_estimator_edge_cases():
    """Validates that DiD model fitting is robust and throws clean errors on extreme collinearity or NaNs."""
    did = DifferenceInDifferences()

    y = np.array([10.0, 12.0, 11.0, np.nan])  # Contains NaN
    treatment = np.array([0, 1, 0, 1])
    post = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="contains NaN values"):
        did.fit(y, treatment, post)

    # Collinear covariates / perfect collinearity
    y_ok = np.array([10.0, 12.0, 11.0, 15.0])
    # T and post are exactly identical (perfect collinearity)
    treatment_coll = np.array([0, 1, 0, 1])
    post_coll = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError, match="Design matrix is singular"):
        did.fit(y_ok, treatment_coll, post_coll)


def test_iv_estimator_edge_cases():
    """Validates that InstrumentalVariables2SLS is robust against constant treatments and instruments."""
    iv = InstrumentalVariables2SLS()

    outcome = np.array([5.0, 6.0, 7.0])
    treatment_const = np.array([1.0, 1.0, 1.0])  # Zero variance
    instrument = np.array([0, 1, 0])

    with pytest.raises(ValueError, match="treatment_received' has zero variance"):
        iv.fit(outcome, treatment_const, instrument)

    instrument_const = np.array([0.0, 0.0, 0.0])  # Zero variance
    treatment = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="instrument' has zero variance"):
        iv.fit(outcome, treatment, instrument_const)
