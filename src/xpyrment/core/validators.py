"""Unified validation rules, dynamic boundaries, and range assertions (Block 67).

This module implements runtime validator checks to guard core and statistical functions
against mathematically invalid, out-of-bounds, empty, or non-finite inputs.
"""

from typing import Any, Union, Optional
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from xpyrment.core.exceptions import BoundaryValidationError


def assert_probability(val: Any, name: str) -> None:
    """Assures that a value is in the valid probability range [0.0, 1.0].

    Args:
        val (Any): Input value to validate.
        name (str): Parameter name for custom error formatting.

    Raises:
        BoundaryValidationError: If the value is not a float or outside [0.0, 1.0].
    """
    if val is None:
        raise BoundaryValidationError(f"Parameter '{name}' cannot be None.")
    try:
        f_val = float(val)
    except (ValueError, TypeError):
        raise BoundaryValidationError(f"Parameter '{name}' must be a float, got type {type(val).__name__}.")

    if not (0.0 <= f_val <= 1.0):
        raise BoundaryValidationError(f"Parameter '{name}' must be in range [0.0, 1.0], got {f_val}.")


def assert_positive_int(val: Any, name: str) -> None:
    """Assures that a value is an integer greater than or equal to 1.

    Args:
        val (Any): Input value to validate.
        name (str): Parameter name for custom error formatting.

    Raises:
        BoundaryValidationError: If the value is not an integer or is less than 1.
    """
    if val is None:
        raise BoundaryValidationError(f"Parameter '{name}' cannot be None.")

    # Exclude booleans since isinstance(True, int) is True
    if not isinstance(val, (int, np.integer)) or isinstance(val, bool):
        raise BoundaryValidationError(f"Parameter '{name}' must be an integer, got type {type(val).__name__}.")

    if val < 1:
        raise BoundaryValidationError(f"{name} must be positive, got {val}.")


def assert_non_empty(arr: Any, name: str) -> None:
    """Validates that an array-like, DataFrame, Series, collection or string is not empty.

    Args:
        arr (Any): Input array, collection, or DataFrame.
        name (str): Parameter name for custom error formatting.

    Raises:
        BoundaryValidationError: If the input has zero elements or is empty.
    """
    if arr is None:
        raise BoundaryValidationError(f"Parameter '{name}' cannot be None.")

    if isinstance(arr, (pd.DataFrame, pd.Series, np.ndarray)):
        if arr.size == 0:
            raise BoundaryValidationError(f"Parameter '{name}' cannot be empty.")
    elif isinstance(arr, (list, tuple, dict, set, str)):
        if len(arr) == 0:
            raise BoundaryValidationError(f"Parameter '{name}' cannot be empty.")
    else:
        try:
            if len(arr) == 0:
                raise BoundaryValidationError(f"Parameter '{name}' cannot be empty.")
        except TypeError:
            raise BoundaryValidationError(
                f"Parameter '{name}' has no length/dimension and cannot be audited for non-emptiness."
            )


def assert_finite(arr: Any, name: str) -> None:
    """Audits arrays, Series, DataFrames, lists, or tuples to guarantee zero occurrence of NaN or Inf.

    Args:
        arr (Any): Input array-like structures.
        name (str): Parameter name for custom error formatting.

    Raises:
        BoundaryValidationError: If any element is NaN, Inf, or -Inf.
    """
    if arr is None:
        raise BoundaryValidationError(f"Parameter '{name}' cannot be None.")

    try:
        if isinstance(arr, (pd.DataFrame, pd.Series)):
            if isinstance(arr, pd.DataFrame):
                if arr.isnull().any().any():
                    raise BoundaryValidationError(f"Parameter '{name}' contains NaN values.")
                # Filter to only numeric columns for infinity checks
                numeric_df = arr.select_dtypes(include=np.number)
                if np.isinf(numeric_df).any().any():
                    raise BoundaryValidationError(f"Parameter '{name}' contains infinite (inf or -inf) values.")
            else:
                if arr.isnull().any():
                    raise BoundaryValidationError(f"Parameter '{name}' contains NaN values.")
                if pd.api.types.is_numeric_dtype(arr):
                    if np.isinf(arr).any():
                        raise BoundaryValidationError(f"Parameter '{name}' contains infinite (inf or -inf) values.")
        else:
            np_arr = np.asarray(arr)
            if np.issubdtype(np_arr.dtype, np.number):
                if np.any(np.isnan(np_arr)):
                    raise BoundaryValidationError(f"Parameter '{name}' contains NaN values.")
                if np.any(np.isinf(np_arr)):
                    raise BoundaryValidationError(f"Parameter '{name}' contains infinite (inf or -inf) values.")
    except Exception as e:
        if isinstance(e, BoundaryValidationError):
            raise
        raise BoundaryValidationError(f"Parameter '{name}' verification failed: {str(e)}.")


def validate_secure_path(
    target: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
    allow_temp: bool = True,
) -> Path:
    """Validates that a path does not escape designated secure directories (Workspace / Temp).

    Args:
        target (Union[str, Path]): The target file or directory path to validate.
        base_dir (Optional[Union[str, Path]]): Whitelisted base directory path. Defaults to
            active project workspace root ("c:/Users/Dan/projects/xpyrment").
        allow_temp (bool): If True, whitelists the system temporary directory. Defaults to True.

    Returns:
        Path: Resolved absolute path.

    Raises:
        PermissionError: If the resolved path escapes the whitelisted boundaries.
    """
    if target is None:
        raise BoundaryValidationError("Target path cannot be None.")

    target_path = Path(target).resolve()
    
    # Workspace baseline - defaults to active workspace root: "c:/Users/Dan/projects/xpyrment"
    workspace_dir = Path(base_dir or "c:/Users/Dan/projects/xpyrment").resolve()
    
    # System temporary baseline
    temp_dir = Path(tempfile.gettempdir()).resolve()
    
    # Check if target resides in workspace or temp directory
    in_workspace = target_path.is_relative_to(workspace_dir)
    in_temp = allow_temp and target_path.is_relative_to(temp_dir)
    
    if not (in_workspace or in_temp):
        raise PermissionError(f"Security Block: Resolved path '{target_path}' escapes secure boundaries.")
    
    return target_path
