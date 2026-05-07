"""Meta-analysis estimators for combining historical experiment results.

This module provides the `MetaAnalysis` class supporting both Fixed-Effects and
Random-Effects (DerSimonian-Laird) analytical pooling models.
"""

from typing import Dict, Any, List
import numpy as np
from scipy import stats


class MetaAnalysis:
    """Combines point estimates and variances from multiple independent experiments.

    # TODO: Support alternative random-effects variance estimators (such as Hedges-Olkin or Sidik-Jonkman) to compare against DerSimonian-Laird.
    # TODO: Implement study-level meta-regression adjustments supporting auxiliary study covariates (e.g., historical run duration).
    # TODO: Support Trim-and-Fill algorithms to estimate and adjust pooled estimates for funnel plot asymmetry / publication bias.
    """

    def __init__(self, estimates: List[float], variances: List[float]):
        """Initializes the MetaAnalysis estimator.

        Args:
            estimates (List[float]): Point estimates (e.g., treatment effects) from each study.
            variances (List[float]): Variances (squared standard errors) from each study.
        """
        if len(estimates) != len(variances):
            raise ValueError("The estimates and variances lists must have the same length.")
        if len(estimates) < 2:
            raise ValueError("Meta-analysis requires at least two experiments.")
        if any(v <= 0.0 for v in variances):
            raise ValueError("All study variances must be strictly positive.")

        self.estimates = np.array(estimates)
        self.variances = np.array(variances)
        self.k = len(estimates)

    def fit_fixed_effects(self) -> Dict[str, Any]:
        """Fits a Fixed-Effects inverse-variance weighted meta-analysis.

        Returns:
            Dict[str, Any]: Pooled effect, standard error, z-statistic, p-value, and confidence interval.
        """
        # Weights are inverse of variance
        w = 1.0 / self.variances
        sum_w = np.sum(w)

        pooled_effect = np.sum(w * self.estimates) / sum_w
        pooled_var = 1.0 / sum_w
        pooled_se = np.sqrt(pooled_var)

        z_stat = pooled_effect / pooled_se
        p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat)))

        ci_lower = pooled_effect - 1.96 * pooled_se
        ci_upper = pooled_effect + 1.96 * pooled_se

        return {
            "pooled_effect": float(pooled_effect),
            "standard_error": float(pooled_se),
            "z_statistic": float(z_stat),
            "p_value": float(p_val),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }

    def fit_random_effects(self) -> Dict[str, Any]:
        """Fits a Random-Effects (DerSimonian-Laird) meta-analysis model.

        Returns:
            Dict[str, Any]: Pooled effect, standard error, p-value, confidence interval, Q, and tau^2.
        """
        # Step 1: Compute fixed-effects parameters
        fe = self.fit_fixed_effects()
        pooled_fe = fe["pooled_effect"]

        # Fixed weights
        w = 1.0 / self.variances

        # Cochran's Q heterogeneity statistic
        q = float(np.sum(w * ((self.estimates - pooled_fe) ** 2)))

        # Cochran's Q degrees of freedom: k - 1
        df = self.k - 1

        # Calculate between-study variance (tau^2) using DerSimonian-Laird estimator
        sum_w = np.sum(w)
        sum_w_sq = np.sum(w ** 2)

        numerator = q - df
        denominator = sum_w - (sum_w_sq / sum_w)

        tau_sq = max(0.0, numerator / denominator)

        # Random-effects weights
        w_random = 1.0 / (self.variances + tau_sq)
        sum_w_random = np.sum(w_random)

        pooled_effect = np.sum(w_random * self.estimates) / sum_w_random
        pooled_var = 1.0 / sum_w_random
        pooled_se = np.sqrt(pooled_var)

        z_stat = pooled_effect / pooled_se
        p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat)))

        ci_lower = pooled_effect - 1.96 * pooled_se
        ci_upper = pooled_effect + 1.96 * pooled_se

        return {
            "pooled_effect": float(pooled_effect),
            "standard_error": float(pooled_se),
            "z_statistic": float(z_stat),
            "p_value": float(p_val),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "cochrans_q": q,
            "tau_squared": tau_sq,
        }
