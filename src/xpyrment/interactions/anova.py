"""Factorial Analysis of Variance (ANOVA) and statistical interaction testing.

This module provides functions to partition experimental variation across multiple independent factors
and their interactions, enabling comprehensive Analysis of Variance for classical design of experiments (DoE).
"""

import pandas as pd


def run_factorial_anova(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    r"""Computes factorial ANOVA tables with interaction terms for DoE factors.

    Factorial ANOVA decomposes the total variability of an experimental outcome into portions attributable
    to main factor effects, multi-factor interaction effects, and random error. This is crucial for verifying
    which process factors have a statistically significant impact on the response variable, and whether factors
    behave synergetically or antagonistically when combined.

    Mathematical Formulation:
        For a two-factor experimental design (Factor $A$ with $I$ levels, Factor $B$ with $J$ levels, and $K$ replicates per cell),
        the response $Y_{ijk}$ is modeled as:
        $$
        Y_{ijk} = \\mu + \\alpha_i + \\beta_j + (\\alpha\\beta)_{ij} + \\varepsilon_{ijk}
        $$
        where:
        - $\\mu$: The grand mean of the response.
        - $\\alpha_i$: The main effect of Factor $A$ at level $i$ (subject to $\\sum_{i=1}^I \\alpha_i = 0$).
        - $\\beta_j$: The main effect of Factor $B$ at level $j$ (subject to $\\sum_{j=1}^J \\beta_j = 0$).
        - $(\\alpha\\beta)_{ij}$ smokes: The interaction effect between Factor $A$ at level $i$ and Factor $B$ at level $j$
          (subject to $\\sum_{i=1}^I (\\alpha\\beta)_{ij} = \\sum_{j=1}^J (\\alpha\\beta)_{ij} = 0$).
        - $\\varepsilon_{ijk}$: Independent and identically distributed normal error terms, $\\varepsilon_{ijk} \\sim \\mathcal{N}(0, \\sigma^2)$.

    Decomposition of Sum of Squares (SS):
        The total sum of squares ($SS_{\\text{Total}}$) measures total sample variation:
        $$
        SS_{\\text{Total}} = SS_A + SS_B + SS_{AB} + SS_{\\text{Error}}
        $$
        where:
        - $SS_A = J K \\sum_{i=1}^I (\\bar{Y}_{i\\cdot\\cdot} - \\bar{Y}_{\\cdot\\cdot\\cdot})^2$ (Main effect $A$)
        - $SS_B = I K \\sum_{j=1}^J (\\bar{Y}_{\\cdot j\\cdot} - \\bar{Y}_{\\cdot\\cdot\\cdot})^2$ (Main effect $B$)
        - $SS_{AB} = K \\sum_{i=1}^I \\sum_{j=1}^J (\\bar{Y}_{ij\\cdot} - \\bar{Y}_{i\\cdot\\cdot} - \\bar{Y}_{\\cdot j\\cdot} + \\bar{Y}_{\\cdot\\cdot\\cdot})^2$ (Interaction effect $AB$)
        - $SS_{\\text{Error}} = \\sum_{i=1}^I \\sum_{j=1}^J \\sum_{k=1}^K (Y_{ijk} - \\bar{Y}_{ij\\cdot})^2$ (Residual variation)

    F-Test Ratios:
        Significance of each effect is evaluated by comparing the Mean Square ($MS = SS / df$) against the residual Mean Square ($MS_{\\text{Error}}$):
        - For Factor $A$:
          $$
          F_A = \\frac{MS_A}{MS_{\\text{Error}}} = \\frac{SS_A / (I-1)}{SS_{\\text{Error}} / [IJ(K-1)]} \\sim F_{I-1, \\ IJ(K-1)}
          $$
        - For Interaction $AB$:
          $$
          F_{AB} = \\frac{MS_{AB}}{MS_{\\text{Error}}} = \\frac{SS_{AB} / [(I-1)(J-1)]}{SS_{\\text{Error}} / [IJ(K-1)]} \\sim F_{(I-1)(J-1), \\ IJ(K-1)}
          $$
        A significant $F_{AB}$ ($p < 0.05$) proves that the effect of Factor $A$ depends on the level of Factor $B$. This indicates that
        interpreting main effects alone is statistically misleading; the interaction must be evaluated.

    Args:
        df (pd.DataFrame): The experimental dataset.
        formula (str): R-style regression formula (e.g., `"revenue ~ factor_a * factor_b"`).

    Returns:
        pd.DataFrame: A standard ANOVA table detailing Sum of Squares, degrees of freedom ($df$), F-statistics,
            and p-values for each term.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Fit ordinary least squares model using R-style formula
    model = smf.ols(formula, data=df).fit()
    
    # Compute Type II ANOVA table
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    return anova_table
