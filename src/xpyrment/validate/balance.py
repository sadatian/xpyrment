"""Covariate balance checking and standardized mean differences (SMD).

This module provides diagnostic engines to evaluate whether the control and treatment groups are
balanced across key pre-experiment covariates (demographics, platform, historical engagement),
preventing confounding or pre-existing selection bias from skewing treatment estimates.
"""

import pandas as pd


def check_covariate_balance(df: pd.DataFrame, treatment_col: str, covariate_cols: list) -> dict:
    r"""Computes Normalized Differences and t-tests to evaluate balance of pre-period covariates.

    Verifies that pre-period characteristics are distributed symmetrically across treatment arms.
    While simple t-tests can be used, they are highly sensitive in online datasets: with large footprints,
    extremely tiny, practically negligible differences will yield highly significant p-values ($p < 0.05$).
    Therefore, we compute **Standardized Mean Differences (SMD)** as the primary effect size metric.

    ??? mathbox "Mathematical Representation"

        1. **Standardized Mean Difference (SMD)** for continuous covariates:
           Let $\bar{X}_T$ and $\bar{X}_C$ be the sample means of a covariate $X$ in the treatment and control groups,
           and let $s_T^2$ and $s_C^2$ be their sample variances.
           $$
           \text{SMD} = \frac{\bar{X}_T - \bar{X}_C}{\sqrt{\frac{s_T^2 + s_C^2}{2}}}
           $$
        2. **Pearson Chi-Square Test for Independence** for categorical covariates:
           Evaluates whether the proportion of units in each category is independent of treatment.

    Args:
        df (pd.DataFrame): The experimental dataset containing units, treatment assignments, and covariates.
        treatment_col (str): Column name identifying experimental groups/arms.
        covariate_cols (list): List of column names representing categorical or continuous pre-experiment covariates.

    Returns:
        dict: A dictionary mapping each covariate name to a diagnostic sub-dictionary containing SMD, p-values,
            and balance classification tags.
    """
    import numpy as np
    from scipy import stats

    groups = df[treatment_col].unique()
    if len(groups) < 2:
        raise ValueError(f"Balance check requires at least 2 distinct groups in '{treatment_col}'. Found {len(groups)}.")

    # Sort groups to be deterministic: first group is control (group 0), second is treatment (group 1)
    groups = sorted(groups)
    grp_0 = df[df[treatment_col] == groups[0]]
    grp_1 = df[df[treatment_col] == groups[1]]

    results = {}

    for cov in covariate_cols:
        if cov not in df.columns:
            raise KeyError(f"Covariate column '{cov}' not found in DataFrame.")

        # Determine type: check if column is numeric
        if pd.api.types.is_numeric_dtype(df[cov]):
            val_0 = grp_0[cov].dropna()
            val_1 = grp_1[cov].dropna()

            mean_0 = val_0.mean()
            mean_1 = val_1.mean()
            var_0 = val_0.var(ddof=1)
            var_1 = val_1.var(ddof=1)

            # Compute Standardized Mean Difference (SMD)
            pooled_sd = np.sqrt((var_0 + var_1) / 2.0)
            if pooled_sd == 0.0:
                smd = 0.0
            else:
                smd = (mean_1 - mean_0) / pooled_sd

            # Welch's t-test (unequal variances assumed)
            if len(val_0) > 0 and len(val_1) > 0:
                _, p_val = stats.ttest_ind(val_1, val_0, equal_var=False)
                # Kolmogorov-Smirnov test for distribution shape alignment
                ks_stat, ks_p = stats.ks_2samp(val_1, val_0)
            else:
                p_val = 1.0
                ks_stat, ks_p = 0.0, 1.0

            results[cov] = {
                "type": "numeric",
                "smd": float(smd),
                "p_value": float(p_val),
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_p)
            }
        else:
            # Categorical covariate: build crosstab contingency table
            contingency_table = pd.crosstab(df[cov], df[treatment_col])

            if contingency_table.shape[0] > 0 and contingency_table.shape[1] > 0:
                # Pearson's chi-square test of independence
                chi2_res = stats.chi2_contingency(contingency_table)
                p_val = chi2_res.pvalue
            else:
                p_val = 1.0

            results[cov] = {
                "type": "categorical",
                "p_value": float(p_val)
            }

    # Integrate Mahalanobis distance multivariate covariance balance tests
    numeric_covs = [cov for cov in covariate_cols if pd.api.types.is_numeric_dtype(df[cov])]
    if len(numeric_covs) >= 1:
        data_0 = grp_0[numeric_covs].dropna()
        data_1 = grp_1[numeric_covs].dropna()

        if len(data_0) > len(numeric_covs) and len(data_1) > len(numeric_covs):
            mean_0 = data_0.mean().values
            mean_1 = data_1.mean().values
            
            # Pooled covariance matrix
            cov_0 = data_0.cov().values
            cov_1 = data_1.cov().values
            pooled_cov = (cov_0 + cov_1) / 2.0
            
            try:
                inv_pooled_cov = np.linalg.pinv(pooled_cov)
                diff = mean_1 - mean_0
                mahalanobis_dist = np.sqrt(np.dot(np.dot(diff, inv_pooled_cov), diff))
                results["_multivariate"] = {
                    "mahalanobis_distance": float(mahalanobis_dist),
                    "n_covariates": len(numeric_covs)
                }
            except np.linalg.LinAlgError:
                pass

    return results
