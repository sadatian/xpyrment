"""Interactive regression modeling and Likelihood Ratio Testing (LRT).

This module provides functions to test covariate-treatment interactions by fitting nested linear regression
models and performing Likelihood Ratio Tests.
"""

import pandas as pd


def check_treatment_covariate_interaction(df: pd.DataFrame, treatment_col: str, covariate_col: str, target_col: str) -> float:
    r"""Computes a Likelihood Ratio Test (LRT) to check if a covariate significantly interacts with the treatment split.

    Evaluates whether the treatment effect varies across different values of a pre-period covariate.
    To determine if the interaction term is statistically necessary (rather than just overfitting the sample),
    we fit nested regression models and perform a classical Likelihood Ratio Test.

    Mathematical Formulation of Nested Models:
        We define two models representing competing hypotheses:
        1. **Restricted Null Model ($M_{\\text{null}}$)** (additive, assuming no interaction):
           $$
           Y_i = \\beta_0 + \\beta_1 T_i + \\beta_2 C_i + \\varepsilon_i
           $$
        2. **Unrestricted Alternative Model ($M_{\\text{alt}}$)** (interactive, assuming interaction):
           $$
           Y_i = \\beta_0 + \\beta_1 T_i + \\beta_2 C_i + \\beta_3 (T_i \\times C_i) + \\varepsilon_i
           $$
        where:
        - $Y_i$: The target outcome metric ($target\\_col$) for unit $i$.
        - $T_i$: The treatment group indicator ($treatment\\_col$, e.g., $0$ or $1$).
        - $C_i$: The pre-period covariate ($covariate\\_col$, e.g., device type or baseline revenue).
        - $T_i \\times C_i$: The interaction/product term.

    The Likelihood Ratio Test (LRT):
        Let $\\ln L(M_{\\text{null}})$ and $\\ln L(M_{\\text{alt}})$ be the maximized log-likelihood values of the nested models.
        The test statistic $D$ is computed as:
        $$
        D = 2 \\left( \\ln L(M_{\\text{alt}}) - \\ln L(M_{\\text{null}}) \\right)
        $$
        Under the null hypothesis $H_0: \\beta_3 = 0$ (no interaction), the test statistic $D$ asymptotically follows a
        Chi-square distribution with degrees of freedom equal to the difference in the number of parameters:
        $$
        D \\sim \\chi^2_{df_{\\text{alt}} - df_{\\text{null}}} = \\chi^2_1
        $$
        (since we added exactly one interaction parameter, $\\beta_3$).
        
        The resulting p-value is calculated as:
        $$
        p = 1 - F_{\\chi^2_1}(D)
        $$
        where $F$ is the cumulative distribution function of the Chi-square distribution with 1 degree of freedom.
        If $p < 0.05$, the alternative model is selected, confirming that the treatment effect varies across levels of the covariate.

    Args:
        df (pd.DataFrame): The experimental dataset.
        treatment_col (str): Column containing treatment assignments.
        covariate_col (str): Column containing the target pre-period covariate under evaluation.
        target_col (str): Column containing the outcome response variable ($Y$).

    Returns:
        float: The calculated p-value of the Likelihood Ratio Test. A value $< 0.05$ indicates a significant interaction.
    """
    import statsmodels.formula.api as smf
    from scipy.stats import chi2

    # Model 1: Additive (Restricted)
    formula_null = f"{target_col} ~ {treatment_col} + {covariate_col}"
    model_null = smf.ols(formula_null, data=df).fit()

    # Model 2: Interactive (Unrestricted)
    formula_alt = f"{target_col} ~ {treatment_col} * {covariate_col}"
    model_alt = smf.ols(formula_alt, data=df).fit()

    # Likelihood Ratio Test
    # For OLS, LRT = n * ln(RSS_null / RSS_alt)
    # But statsmodels OLS results have .llf (log-likelihood)
    llf_null = model_null.llf
    llf_alt = model_alt.llf

    lr_stat = 2 * (llf_alt - llf_null)
    p_value = chi2.sf(lr_stat, df=1)

    return float(p_value)
