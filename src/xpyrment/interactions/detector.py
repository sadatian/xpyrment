"""Interaction dispatcher and multi-model statistical screening.

This module provides the `InteractionDetector` class, which automates the discovery of multi-factor,
covariate-treatment, and non-linear response surface interactions using linear models and ANOVA.
"""


class InteractionDetector:
    """Dispatches multi-factor, covariate-treatment, and non-linear interactions.

    In complex online and physical experiments, interventions rarely operate in a vacuum. The treatment effect of
    one change may depend heavily on the status of other features (multi-factor interaction) or the characteristics
    of the experimental unit (covariate-treatment interaction, or Heterogeneous Treatment Effect).
    The `InteractionDetector` screens for these interactions automatically.

    Mathematical Interaction Categories:
        1. **Multi-Factor Synergy or Interference** (Factor-Factor Interaction):
           Evaluates whether combining Treatment 1 ($T_1$) and Treatment 2 ($T_2$) yields a response that differs
           from the sum of their individual effects:
           $$Y = \\beta_0 + \\beta_1 T_1 + \\beta_2 T_2 + \\beta_3 (T_1 \\times T_2) + \\varepsilon$$
           - If $\\beta_3 > 0$, the factors are synergistic.
           - If $\\beta_3 < 0$, the factors interfere with each other (redundancy or collision).

        2. **Covariate-Treatment Interaction** (Heterogeneous Treatment Effects - HTE):
           Evaluates whether the treatment effect varies systematically across pre-experiment characteristics $C$
           (e.g., country, browser, mobile vs. desktop, or historical spending):
           $$Y = \\beta_0 + \\beta_1 T + \\beta_2 C + \\beta_3 (T \\times C) + \\varepsilon$$
           A significant $\\beta_3$ ($p < 0.05$) indicates that the treatment effect varies across subpopulations.

        3. **Response Surface Curvature** (Non-linear Interaction):
           In continuous Design of Experiments (DoE), evaluates quadratic curvatures and continuous interaction slopes:
           $$Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\beta_3 X_1^2 + \\beta_4 X_2^2 + \\beta_5 (X_1 \\times X_2) + \\varepsilon$$
           where $\\beta_5$ captures the continuous twisting of the response landscape, and $\\beta_3, \\beta_4$ capture curvature.

    Algorithmic Screening Workflow:
        ```text
        function detect_all(experiment):
            Initialize interaction_results = {}
            For each pair of factors (A, B) in design:
                Run OLS: Y ~ A * B
                Extract interaction p-value.
                If p-value < 0.05:
                    Add to interaction_results["factor_factor"]
                    
            For each covariate C in experiment.covariates:
                Run OLS: Y ~ Treatment * C
                Extract interaction p-value.
                If p-value < 0.05:
                    Add to interaction_results["heterogeneous_treatment_effects"]
                    
            Return interaction_results
        ```

    Attributes:
        experiment (Experiment): The completed or active experiment container.
    """

    def __init__(self, experiment):
        """Initializes an InteractionDetector.

        Args:
            experiment (Experiment): The experiment container containing datasets, metrics, and factor definitions.
        """
        self.experiment = experiment

    def detect_all(self) -> dict:
        """Runs ANOVA and regression checks to identify interaction terms across factors and covariates.

        Fits multi-variable linear regression models with product terms and flags statistically
        significant interactions.

        Returns:
            dict: A dictionary grouping detected interactions, their estimated coefficients, standard errors,
                and p-values.
        """
        # TODO: Implement dispatcher
        return {}

