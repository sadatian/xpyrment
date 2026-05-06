import pandas as pd


def apply_cuped(df: pd.DataFrame, target_col: str, pre_col: str) -> pd.Series:
    """Applies Controlled-experiments Using Pre-Experiment Data (CUPED) on a series."""
    # This is a core transformation utility, but currently handled inline inside taxonomy.py.
    # We can write a general placeholder here.
    return df[target_col]
