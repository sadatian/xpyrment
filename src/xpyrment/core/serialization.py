"""Robust serialization utilities to guarantee native Python and JSON compatibility (Block 52).

Translates all scientific computing structures (NumPy floats, integers, arrays,
and booleans) into plain Python representations recursively, avoiding JSON serialization crashes.
"""

import json
from typing import Any, Optional


def make_serializable(obj: Any) -> Any:
    """Recursively converts numpy and non-serializable objects to native, standard JSON-compliant types.

    Args:
        obj (Any): The nested object or value to convert.

    Returns:
        Any: Standard Python dictionary, list, float, int, bool, or string.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [make_serializable(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist())
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return str(val)  # String standard for portable audit trails
        return val
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif obj is None:
        return None
    elif hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            return str(obj)
    else:
        try:
            # Handle statsmodels or other custom objects
            if type(obj).__name__ == "ContrastResults" or type(obj).__name__ == "RegressionResultsWrapper":
                return str(obj)
            return str(obj)
        except Exception:
            return None


def serialize_to_json(obj: Any, indent: Optional[int] = None) -> str:
    """Converts a nested object recursively to serializable format and dumps it as a JSON string.

    Args:
        obj (Any): The object to convert and serialize.
        indent (Optional[int]): Indentation level for pretty-printing.

    Returns:
        str: Validated JSON string.
    """
    return json.dumps(make_serializable(obj), indent=indent)
