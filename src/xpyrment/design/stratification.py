"""Stratified and clustered randomization engines for balanced covariate distributions.

This module provides stratified randomization algorithms. Stratification ensures that critical
pre-experiment covariates (e.g., country, customer segment, historical purchase volume) are balanced
proportionally across all treatment arms, reducing pre-experiment bias and enhancing statistical power.
"""

import pandas as pd


import numpy as np
import pandas as pd
from typing import List, Optional


def stratified_randomization(
    df: pd.DataFrame,
    strata_cols: list,
    variants: Optional[List[str]] = None,
    treatment_col: str = "variant",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    r"""Performs stratified randomization to ensure balance on continuous/categorical covariates.

    In simple randomization, small sample sizes or highly variable covariates can lead to accidental
    imbalance across treatment arms, introducing selection bias. Stratified randomization resolves this
    by dividing the population into mutually exclusive, homogeneous subgroups (strata) based on the
    provided covariates, and then executing independent randomization within each individual stratum.

    Mathematical and Algorithmic Background:
        Let $D$ be the dataset of size $N$. Let $C = \{c_1, c_2, \dots, c_p\}$ be the set of stratification
        columns.
        1. **Strata Construction**:
           We partition $D$ into $K$ disjoint subsets (strata), $\{D_1, D_2, \dots, D_K\}$, such that within each
           subset $D_j$, all units share identical values for all stratification columns $C$:
           $$
           D = \bigcup_{j=1}^{K} D_j \quad \text{where} \quad D_a \cap D_b = \emptyset \ \ \forall \ a \neq b
           $$
        2. **Intra-Stratum Randomization**:
           For each stratum $D_j$, units are randomly permuted and assigned to treatment arms. This guarantees that
           if treatment arm proportions are $\{w_1, w_2, \dots, w_k\}$, then within every stratum $D_j$, the assignment
           counts follow:
           $$
           n_{j, \text{arm } i} \approx w_i \times |D_j|
           $$
           This reduces the variance of the treatment effect estimator by removing the variance contribution of the
           stratification covariates.

    Args:
        df (pd.DataFrame): The input DataFrame containing experimental units and their covariate values.
        strata_cols (list): List of column names in `df` representing categorical or binned continuous
            covariates to use as stratification factors.
        variants (Optional[List[str]]): Ordered list of variant labels. Defaults to `["control", "treatment"]`.
        treatment_col (str): Column name where the assigned variant label will be written. Defaults to `"variant"`.
        random_state (Optional[int]): Integer seed to initialize local random state generator for reproducibility.

    Returns:
        pd.DataFrame: A new DataFrame with treatment assignments balanced across the specified strata.
    """
    if variants is None:
        variants = ["control", "treatment"]

    if not variants:
        raise ValueError("variants list cannot be empty.")

    # Instantiate isolated local generator
    rng = np.random.default_rng(random_state)
    assigned_df = df.copy()
    assigned_df[treatment_col] = None

    # Group by the specified strata columns
    for _, group in assigned_df.groupby(strata_cols):
        n_group = len(group)
        if n_group == 0:
            continue
        
        # Generate balanced sequence of variants of length n_group
        group_assignments = [variants[i % len(variants)] for i in range(n_group)]
        group_assignments = np.array(group_assignments)
        
        # Shuffle assignments using local generator to enforce seeding rules
        rng.shuffle(group_assignments)
        
        # Assign shuffled variant labels back to corresponding rows
        assigned_df.loc[group.index, treatment_col] = group_assignments

    return assigned_df
