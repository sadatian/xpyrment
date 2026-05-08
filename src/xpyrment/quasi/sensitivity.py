"""Rosenbaum Bounds & Omission Bias Sensitivity Analysis (Block 29).

Provides robust sensitivity analysis tools (Rosenbaum's Gamma Bounds and Cinelli & Hazlett's
partial R^2 bounds) to test the resilience of causal estimates against omitted confounders.
"""

from typing import Dict, List, Union
import numpy as np
from scipy.stats import binom


class CausalSensitivityAnalyzer:
    """Performs sensitivity analyses to quantify resilience against unobserved confounding.

    Supports:
    1. Rosenbaum's Gamma Bounds for matched pairwise observations (Sign-Test bounds).
    2. Cinelli & Hazlett (2020) Partial R^2 benchmarking and Robustness Value (RV) estimation.

    # TODO: Add a contour-plot generation utility that maps adjusted effect boundaries over a 2D grid of r2_d_u and r2_y_u.
    # TODO: Support Wilcoxon signed-rank sum statistic bounds as a distribution-free alternative to sign-test Rosenbaum bounds.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def rosenbaum_bounds(differences: np.ndarray, gammas: List[float]) -> Dict[float, Dict[str, float]]:
        """Computes Rosenbaum's Gamma sign-test p-value bounds for unobserved confounding odds.

        Args:
            differences (np.ndarray): Matched pairwise treatment-control outcome differences.
            gammas (List[float]): Unobserved assignment odds multipliers to evaluate (must be >= 1.0).

        Returns:
            Dict: Dictionary mapping each Gamma level to upper and lower bounds of the sign-test p-value.
        """
        diffs = differences[differences != 0.0]
        n = len(diffs)
        if n == 0:
            raise ValueError("No non-zero pairwise differences available for Rosenbaum bounds.")

        # S+ is the number of positive treatment-control differences
        s_plus = int(np.sum(diffs > 0.0))

        bounds = {}
        for gamma in gammas:
            if gamma < 1.0:
                raise ValueError("Rosenbaum odds factor Gamma must be >= 1.0.")

            # Under H0 with confounding Gamma, probability of positive difference is bounded:
            p_max = gamma / (1.0 + gamma)
            p_min = 1.0 / (1.0 + gamma)

            # Upper bound p-value (under-estimating treatment effect if s_plus > n/2)
            if s_plus >= n / 2:
                # Probability of getting at least s_plus successes
                p_val_upper = float(1.0 - binom.cdf(s_plus - 1, n, p_max))
                p_val_lower = float(binom.cdf(s_plus, n, p_min))
            else:
                p_val_upper = float(binom.cdf(s_plus, n, p_max))
                p_val_lower = float(1.0 - binom.cdf(s_plus - 1, n, p_min))

            bounds[gamma] = {
                "upper_p_value": p_val_upper,
                "lower_p_value": p_val_lower,
                "s_plus": float(s_plus),
                "n_matched": float(n),
            }

        return bounds

    @staticmethod
    def cinelli_hazlett_sensitivity(
        treatment_effect: float,
        standard_error: float,
        df: int,
        r2_d_u: float = 0.05,
        r2_y_u: float = 0.05,
    ) -> Dict[str, Union[float, str]]:
        """Computes Cinelli & Hazlett (2020) partial R^2 bias bounds and Robustness Value (RV).

        Args:
            treatment_effect (float): Estimated treatment coefficient (tau_hat).
            standard_error (float): Standard error of the treatment coefficient estimate.
            df (int): Degrees of freedom of the regression model.
            r2_d_u (float): Partial R^2 of treatment with unobserved confounder U.
            r2_y_u (float): Partial R^2 of outcome with unobserved confounder U.

        Returns:
            Dict: Sensitivity metrics including bias bounds, adjusted treatment effect, and Robustness Value.
        """
        if df <= 0:
            raise ValueError("Degrees of freedom must be strictly positive.")
        if not (0.0 <= r2_d_u < 1.0) or not (0.0 <= r2_y_u < 1.0):
            raise ValueError("Partial R^2 parameters must be in the range [0.0, 1.0).")

        t_stat = treatment_effect / standard_error

        # 1. Compute the Robustness Value (RV) to reduce estimate to exactly zero
        # RV represents the equal strength of confounding (R^2_D~U = R^2_Y~U) needed to nullify effect
        f_sq = (t_stat ** 2) / df
        
        # Robustness Value formula:
        rv = 0.5 * (np.sqrt(f_sq * (f_sq + 4.0)) - f_sq)

        # 2. Compute the bias bound for the specified r2_d_u and r2_y_u
        # bias = SE * sqrt(df * r2_d_u * r2_y_u / (1 - r2_d_u))
        multiplier = np.sqrt(df * r2_d_u * r2_y_u / (1.0 - r2_d_u))
        bias = standard_error * multiplier

        # Adjusted treatment effect
        sign = np.sign(treatment_effect)
        adjusted_effect = treatment_effect - sign * bias

        return {
            "original_effect": treatment_effect,
            "standard_error": standard_error,
            "t_statistic": t_stat,
            "degrees_of_freedom": float(df),
            "robustness_value": float(rv),
            "bias_bound": float(bias),
            "adjusted_effect": float(adjusted_effect),
            "is_significant_after_bias": "Yes" if (np.abs(adjusted_effect) / standard_error > 1.96) else "No",
        }
