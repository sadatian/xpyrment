r"""A/A test simulations and false-positive rate validation.

This module provides validation systems for checking the empirical Type I error rate ($\alpha$)
of the experimental pipeline by performing statistical A/A test evaluations on historical or control data.
"""

import pandas as pd


def run_aa_test_validation(
    df: pd.DataFrame,
    treatment_col: str,
    metric_col: str,
    num_simulations: int = 100,
    seed: int = 42
) -> dict:
    r"""Runs an A/A test validation check, asserting that identical splits exhibit no treatment effect.

    An A/A test compares two groups that receive the exact same experience. The objective is to validate
    the statistical pipeline and confirm that the empirical false positive rate matches theoretical expectations.

    Args:
        df (pd.DataFrame): The historical control dataset.
        treatment_col (str): Column name representing the mock or actual assignments.
        metric_col (str): Column name containing the numeric values under test.
        num_simulations (int): Number of permutation splits to simulate. Defaults to 100.
        seed (int): Seed for random generator to guarantee reproducibility. Defaults to 42.

    Returns:
        dict: A dictionary containing:
            - ks_pvalue: The Kolmogorov-Smirnov test p-value indicating goodness-of-fit to a Uniform(0, 1) distribution.
            - empirical_alpha_05: The raw empirical rejection rate at alpha=0.05.
            - fdr_alpha_05: The rejection rate after applying Benjamini-Hochberg False Discovery Rate control.
    """
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(seed)

    p_values = []

    # Clean the metric array to avoid NaNs interfering
    clean_df = df[[treatment_col, metric_col]].dropna()
    if len(clean_df) < 4:
        return {"ks_pvalue": 1.0, "empirical_alpha_05": 0.0, "fdr_alpha_05": 0.0}

    treatment_vals = clean_df[treatment_col].values
    metric_vals = clean_df[metric_col].values

    unique_vals = np.unique(treatment_vals)
    if len(unique_vals) < 2:
        raise ValueError(f"A/A test requires at least 2 distinct groups in '{treatment_col}'. Found {len(unique_vals)}.")

    n1 = int(np.sum(treatment_vals == unique_vals[0]))
    n2 = len(treatment_vals) - n1
    N = len(treatment_vals)

    chunk_size = 5000
    for i in range(0, num_simulations, chunk_size):
        current_chunk = min(chunk_size, num_simulations - i)
        
        # Generate random permutations using argsort of random uniform
        rand_idx = rng.random((current_chunk, N)).argsort(axis=1)
        
        idx1 = rand_idx[:, :n1]
        idx2 = rand_idx[:, n1:]
        
        A1 = metric_vals[idx1] 
        A2 = metric_vals[idx2] 
        
        mean1 = A1.mean(axis=1)
        mean2 = A2.mean(axis=1)
        
        var1 = A1.var(axis=1, ddof=1)
        var2 = A2.var(axis=1, ddof=1)
        
        vn1 = var1 / n1
        vn2 = var2 / n2
        
        with np.errstate(divide='ignore', invalid='ignore'):
            t_stat = (mean2 - mean1) / np.sqrt(vn1 + vn2)
            df_stat = (vn1 + vn2)**2 / ( (vn1**2)/(n1-1) + (vn2**2)/(n2-1) )
            
            p_vals = 2 * stats.t.sf(np.abs(t_stat), df_stat)
            
        p_vals = np.nan_to_num(p_vals, nan=1.0)
        p_values.extend(p_vals)

    # Perform Kolmogorov-Smirnov goodness-of-fit test against a continuous Uniform(0, 1) CDF
    ks_res = stats.kstest(p_values, "uniform")
    
    # False Discovery Rate (FDR) control using Benjamini-Hochberg
    p_values = np.array(p_values)
    empirical_alpha = np.mean(p_values < 0.05)
    fdr_pvals = stats.false_discovery_control(p_values)
    fdr_alpha = np.mean(fdr_pvals < 0.05)

    return {
        "ks_pvalue": float(ks_res.pvalue),
        "empirical_alpha_05": float(empirical_alpha),
        "fdr_alpha_05": float(fdr_alpha)
    }
