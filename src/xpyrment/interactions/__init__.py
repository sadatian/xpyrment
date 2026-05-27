"""Multi-factor experimental design and covariate-treatment interaction analysis.

This package provides tools to discover, model, and visualize interactions within experimental studies.
Understanding interactions is critical to determine whether multiple treatments conflict or work synergistically,
and whether a treatment effect is heterogeneous across user characteristics.

Submodules:
- `detector`: Orchestrates and dispatches multi-model interaction checks across covariates and factors.
- `anova`: Performs Factorial Analysis of Variance (ANOVA) and partitions sum of squares.
- `regression`: Fits interactive regression models and conducts Likelihood Ratio Tests (LRT).
- `hstat`: Quantifies model-agnostic interaction strengths using Friedman's H-statistic.
- `shap`: Calculates second-order game-theoretic Shapley interaction values (TreeSHAP).
- `plots`: Renders symmetric heatmaps and diagnostic visualization charts.
"""

from xpyrment.interactions.detector import InteractionDetector
from xpyrment.interactions.anova import run_factorial_anova
from xpyrment.interactions.regression import check_treatment_covariate_interaction
from xpyrment.interactions.shap import calculate_shap_interactions
from xpyrment.interactions.hstat import compute_friedman_h_statistic
from xpyrment.interactions.plots import plot_interaction_heatmap, plot_interaction_effects

__all__ = [
    "InteractionDetector",
    "run_factorial_anova",
    "check_treatment_covariate_interaction",
    "calculate_shap_interactions",
    "compute_friedman_h_statistic",
    "plot_interaction_heatmap",
    "plot_interaction_effects",
]
