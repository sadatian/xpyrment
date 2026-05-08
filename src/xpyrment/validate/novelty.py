"""Novelty and primacy effect diagnostics using temporal interaction models.

This module provides detection systems for time-varying treatment effects, helping experimenters
distinguish stable, long-term changes from temporary user behavior shifts triggered by feature
novelty or initial friction (primacy).
"""

import pandas as pd


def check_novelty_effects(df: pd.DataFrame, treatment_col: str, metric_col: str, time_col: str) -> dict:
    r"""Detects novelty or primacy effects by tracking treatment effect size evolution over time.

    In online user testing, two common behavioral biases can distort short-term results:
    - **Novelty Effect**: Users are initially drawn to a redesigned feature, leading to a temporary
      surge in engagement that decays back to baseline.
    - **Primacy (or Learning) Effect**: Users are initially slowed down, causing a temporary dip
      in conversion that recovers once they adapt to the change.

    ### Mathematical Representation and Regression Detection

    We fit an ordinary least squares (OLS) regression model with an interaction term
    between treatment $T_i \in \{0, 1\}$ and elapsed time $t_i$:
    $$
    Y_i = \beta_0 + \beta_1 T_i + \beta_2 t_i + \beta_3 (T_i \times t_i) + \varepsilon_i
    $$
    Args:
        df (pd.DataFrame): The experimental dataset.
        treatment_col (str): Column name identifying experimental groups/arms.
        metric_col (str): Column containing the evaluated metric (continuous or rates).
        time_col (str): Column name representing the timestamp or elapsed date index.

    Returns:
        dict: A dictionary containing estimated interaction coefficients, standard errors, p-values,
            and behavioral bias classifications.
    """
    import numpy as np
    from scipy import stats

    clean_df = df[[treatment_col, metric_col, time_col]].dropna().copy()
    n = len(clean_df)
    if n < 5:
        raise ValueError("At least 5 samples are required to fit the OLS novelty interaction model.")

    # Convert treatment_col to binary indicator (0 = control, 1 = treatment)
    groups = sorted(clean_df[treatment_col].unique())
    if len(groups) < 2:
        raise ValueError("At least 2 groups are required in treatment_col.")
    
    clean_df["T"] = clean_df[treatment_col].map({groups[0]: 0.0, groups[1]: 1.0}).fillna(0.0)

    # Convert time_col to numerical index
    if pd.api.types.is_datetime64_any_dtype(clean_df[time_col]):
        t_min = clean_df[time_col].min()
        clean_df["t"] = (clean_df[time_col] - t_min).dt.total_seconds() / (24 * 3600)
    else:
        clean_df["t"] = clean_df[time_col].astype(float)

    # Compute interaction column
    clean_df["T_x_t"] = clean_df["T"] * clean_df["t"]

    # Build OLS design matrix X and response y
    X = np.column_stack([
        np.ones(n),
        clean_df["T"].values,
        clean_df["t"].values,
        clean_df["T_x_t"].values
    ])
    y = clean_df[metric_col].values

    # Fit OLS: beta = (X.T X)^-1 X.T y
    XTX = np.dot(X.T, X)
    try:
        inv_XTX = np.linalg.inv(XTX)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Design matrix is singular. Ensure there is variance in treatment, time, and their interaction."
        )

    beta = np.dot(inv_XTX, np.dot(X.T, y))

    # Calculate residuals, residual variance, and standard errors of coefficients
    residuals = y - np.dot(X, beta)
    rss = np.sum(residuals**2)
    p_params = 4
    
    sigma_sq = rss / (n - p_params)
    cov_beta = sigma_sq * inv_XTX
    se = np.sqrt(np.diag(cov_beta))

    # Compute t-statistics and two-tailed p-values
    t_stats = beta / se
    p_vals = 2 * (1.0 - stats.t.cdf(np.abs(t_stats), df=n - p_params))

    beta_0, beta_1, beta_2, beta_3 = beta
    se_0, se_1, se_2, se_3 = se
    p_0, p_1, p_2, p_3 = p_vals

    classification = "Stable Treatment Effect"
    if p_3 < 0.05:
        if beta_1 > 0.0 and beta_3 < 0.0:
            classification = "Novelty Effect Detected"
        elif beta_1 < 0.0 and beta_3 > 0.0:
            classification = "Primacy Effect Detected"

    return {
        "intercept": {"coef": float(beta_0), "std_err": float(se_0), "p_value": float(p_0)},
        "treatment": {"coef": float(beta_1), "std_err": float(se_1), "p_value": float(p_1)},
        "time": {"coef": float(beta_2), "std_err": float(se_2), "p_value": float(p_2)},
        "interaction": {"coef": float(beta_3), "std_err": float(se_3), "p_value": float(p_3)},
        "classification": classification
    }

    # TODO: Implement non-linear decay estimation (such as exponential decay curves) using non-linear least squares optimization.
    # TODO: Add support for weighted least squares (WLS) where observation weights scale with daily binned traffic volumes.
