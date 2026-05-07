"""Variance reduction algorithms, focusing on continuous and ratio-level CUPED.

This module provides high-performance variance reduction utilities. Reducing metric variance is
mathematically equivalent to increasing the signal-to-noise ratio, enabling massive gains in statistical
power and substantial reductions in required sample size (and therefore runtime).
"""

import pandas as pd


def apply_cuped(df: pd.DataFrame, target_col: str, pre_col: str) -> pd.Series:
    """Applies Controlled-experiments Using Pre-Experiment Data (CUPED) on a series.

    CUPED (Deng et al., 2013) is the standard variance reduction method in modern online experimentation. It uses
    pre-experiment covariate data to remove pre-existing user-level variation, leaving a highly concentrated treatment signal.

    Args:
        df (pd.DataFrame): The dataset containing both target and pre-period columns.
        target_col (str): Column name representing the post-experiment metric of interest ($Y$).
        pre_col (str): Column name representing the pre-period covariate ($X$).

    Returns:
        pd.Series: A pandas Series containing the CUPED-adjusted values.
    """
    clean_df = df[[target_col, pre_col]].dropna()
    if len(clean_df) < 2:
        return df[target_col]

    cov_yx = clean_df[target_col].cov(clean_df[pre_col])
    var_x = clean_df[pre_col].var()

    if var_x > 0.0:
        theta = cov_yx / var_x
    else:
        theta = 0.0

    mean_x = clean_df[pre_col].mean()

    # Apply CUPED adjustment to all rows, maintaining the original Series shape and index
    adjusted = df[target_col] - theta * (df[pre_col] - mean_x)
    return adjusted

    # TODO: Implement multi-covariate CUPAC (Controlled-experiments Using Pre-Experiment Data and Machine Learning) to support non-linear ML-based predictions as covariates.
    # TODO: Add automatic pre-period alignment diagnostics to confirm the covariate is truly independent of treatment assignments.
