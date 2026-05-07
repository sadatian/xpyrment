"""Novelty and primacy effect diagnostics using temporal interaction models.

This module provides detection systems for time-varying treatment effects, helping experimenters
distinguish stable, long-term changes from temporary user behavior shifts triggered by feature
novelty or initial friction (primacy).
"""

import pandas as pd


def check_novelty_effects(df: pd.DataFrame, treatment_col: str, metric_col: str, time_col: str) -> dict:
    r"""Detects novelty or primacy effects by tracking treatment effect size evolution over time.

    In online user testing, two common behavioral biases can distort short-term results:
    - **Novelty Effect**: Users are initially drawn to a redesigned feature out of curiosity, leading to a temporary
      surge in engagement that slowly decays back to baseline as the feature becomes familiar.
    - **Primacy (or Learning) Effect**: Users are initially slowed down, confused, or frustrated by a new interface,
      causing a temporary dip in conversion that eventually recovers once they adapt to the change.

    Mathematical Representation and Regression Detection:
        We can model time-varying treatment effects by fitting a linear regression model with an interaction term
        between the treatment assignment variable $T_i \in \{0, 1\}$ and the time of exposure $t_i$:
        $$Y_i = \beta_0 + \beta_1 T_i + \beta_2 t_i + \beta_3 (T_i \times t_i) + \varepsilon_i$$
        where:
        - $Y_i$: The metric value for unit $i$.
        - $\beta_1$: The initial treatment effect at $t = 0$.
        - $\beta_2$: The baseline temporal trend in the control group.
        - $\beta_3$: The interaction coefficient. This is the parameter of interest; it measures the rate of change
          in the treatment effect per unit of time.

        Interpretation of the Interaction Coefficient ($\beta_3$):
        1. **Null Hypothesis ($H_0: \beta_3 = 0$)**: The treatment effect is stable and constant over the experimental period.
        2. **Novelty Effect ($\beta_1 > 0$ and $\beta_3 < 0$)**: The treatment has a positive initial effect that decays over time.
        3. **Primacy Effect ($\beta_1 < 0$ and $\beta_3 > 0$)**: The treatment has a negative initial effect that trends upward
           as users learn the interface.
        
        The statistical significance of the interaction is evaluated using the t-statistic of $\beta_3$. If the p-value
        associated with $\beta_3$ is less than 0.05, we reject the null hypothesis of a constant treatment effect.

    Pseudocode for the Detection Engine:
        ```text
        function check_novelty_effects(DataFrame df, String treatment_col, String metric_col, String time_col):
            1. Convert time_col to numerical index representing elapsed time (e.g., days since start).
            2. Fit ordinary least squares (OLS) regression:
               metric_col ~ treatment_col * time_col
            3. Extract coefficients beta_1 (treatment), beta_3 (treatment:time_col) and their p-values.
            4. If p_value(beta_3) < 0.05:
                 If beta_1 > 0 and beta_3 < 0:
                   Classify as "Novelty Effect Detected" (positive but decaying).
                 Else if beta_1 < 0 and beta_3 > 0:
                   Classify as "Primacy Effect Detected" (negative but recovering).
            5. Return detailed summary of OLS estimates and classification flags.
        ```

    Args:
        df (pd.DataFrame): The experimental dataset.
        treatment_col (str): Column name identifying experimental groups/arms.
        metric_col (str): Column containing the evaluated metric (continuous or rates).
        time_col (str): Column name representing the timestamp or elapsed date index.

    Returns:
        dict: A dictionary containing estimated interaction coefficients, standard errors, p-values,
            and behavioral bias classifications.
    """
    # TODO: Implement time-series slope/interaction checks
    return {}

