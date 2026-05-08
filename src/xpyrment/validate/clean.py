"""Unified input cleaning, missing value filtering, and collinearity check safeguards (Block 57).

Secures estimators and statistical models against edge-case input anomalies (NaNs, infinite values,
zero variance columns, insufficient samples, and singular/collinear matrices).
"""

from typing import Any, List, Tuple, Union
import numpy as np


def clean_array(
    arr: Any,
    allow_nan: bool = False,
    allow_inf: bool = False,
    allow_zero_var: bool = True,
    name: str = "array"
) -> np.ndarray:
    """Standardizes, validates, and cleans an input array.

    Args:
        arr (Any): Input structure (array-like, list, pandas Series).
        allow_nan (bool): If False, raises ValueError if NaN values are detected. Defaults to False.
        allow_inf (bool): If False, raises ValueError if infinite values are detected. Defaults to False.
        allow_zero_var (bool): If False, raises ValueError if variance is zero. Defaults to True.
        name (str): Logical name of the variable to format error messages. Defaults to "array".

    Returns:
        np.ndarray: Cleaned, converted 1D or 2D numpy array.

    Raises:
        ValueError: If array is empty, contains NaNs/infs, or has zero variance (when disallowed).
    """
    if arr is None:
        raise ValueError(f"Input '{name}' cannot be None.")

    # Convert to NumPy array safely
    try:
        np_arr = np.asarray(arr, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Input '{name}' must be numeric or convertible to numeric array: {str(e)}")

    if np_arr.size == 0:
        raise ValueError(f"Input '{name}' is empty.")

    # Check for NaNs
    if not allow_nan and np.any(np.isnan(np_arr)):
        raise ValueError(f"Input '{name}' contains NaN values. Please filter missing data before estimation.")

    # Check for infinite values
    if not allow_inf and np.any(np.isinf(np_arr)):
        raise ValueError(f"Input '{name}' contains infinite (inf or -inf) values.")

    # Check for zero variance
    if not allow_zero_var and np_arr.size > 1:
        if np.all(np_arr == np_arr[0]) or np.var(np_arr) < 1e-12:
            raise ValueError(f"Input '{name}' has zero variance (constant values detected).")

    return np_arr


def validate_estimation_inputs(
    y: Any,
    *args: Any,
    min_samples: int = 2,
    allow_zero_var_y: bool = False
) -> Tuple[np.ndarray, ...]:
    """Cleans, matches, and validates multiple arrays simultaneously for estimator safety.

    Args:
        y (Any): Dependent target outcome array.
        *args (Any): Variable length list of feature / predictor arrays.
        min_samples (int): Minimum required observations. Defaults to 2.
        allow_zero_var_y (bool): If False, ensures target variable y has positive variance. Defaults to False.

    Returns:
        Tuple[np.ndarray, ...]: Converted, validated numpy arrays with matched lengths.

    Raises:
        ValueError: If lengths do not match or values are invalid.
    """
    cleaned_y = clean_array(y, allow_zero_var=allow_zero_var_y, name="dependent variable (Y)")
    n_samples = len(cleaned_y)

    if n_samples < min_samples:
        raise ValueError(f"Insufficient samples for estimation. Got {n_samples}, need at least {min_samples}.")

    cleaned_args = []
    for idx, arg in enumerate(args):
        cleaned_arg = clean_array(arg, name=f"argument_{idx+1}")
        if len(cleaned_arg) != n_samples:
            raise ValueError(
                f"Length mismatch: 'dependent variable (Y)' has {n_samples} rows, "
                f"but 'argument_{idx+1}' has {len(cleaned_arg)} rows."
            )
        cleaned_args.append(cleaned_arg)

    return (cleaned_y, *cleaned_args)


def verify_collinearity(X: np.ndarray, tolerance: float = 1e-12) -> None:
    """Verifies that the design matrix is full rank and has no extreme collinearity.

    Args:
        X (np.ndarray): Design/Predictor matrix of shape (n_samples, p_features).
        tolerance (float): Singular value threshold below which values are treated as zero. Defaults to 1e-12.

    Raises:
        ValueError: If the matrix is singular, collinear, or rank-deficient.
    """
    if X.size == 0:
        raise ValueError("Design matrix is empty.")

    n_samples, p_features = X.shape

    if n_samples < p_features:
        raise ValueError(
            f"Underdetermined system: Sample count ({n_samples}) is less than feature count ({p_features}). "
            "Model parameters cannot be uniquely identified."
        )

    # Compute Rank using SVD
    rank = np.linalg.matrix_rank(X, tol=tolerance)
    if rank < p_features:
        raise ValueError(
            f"Design matrix is singular or has perfect multicollinearity. "
            f"Expected rank {p_features}, got {rank}."
        )
