import numpy as np
import pandas as pd


def log_transform(df: pd.DataFrame, col: str) -> pd.Series:
    """Transforms continuous metrics using log transformation for skewed distributions.

    y_transformed = log(y + 1)
    """
    return np.log1p(df[col])


def delta_normalization(df: pd.DataFrame, col: str) -> pd.Series:
    """Normalizes metrics using the delta expansion (Stub/Scaffolding)."""
    # TODO: Implement full delta normalization
    return df[col]
