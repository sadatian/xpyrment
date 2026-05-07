"""Difference-in-Differences (DiD) quasi-experimental estimators.

This module provides the `DifferenceInDifferences` class to measure causal treatment
effects when clean randomized control groups are unavailable, including trend validations.
"""

from typing import Dict, Any
import numpy as np
from scipy import stats


def fit_ols(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Fits an ordinary least squares regression model analytically.

    Args:
        X (np.ndarray): Predictors matrix of shape (n_samples, p_features).
        y (np.ndarray): Target outcomes.

    Returns:
        Dict[str, Any]: Coefficients, standard errors, t-statistics, and p-values.
    """
    n_samples, p_features = X.shape
    X_bias = np.hstack([np.ones((n_samples, 1)), X])
    p_params = X_bias.shape[1]

    # beta = (X^T X)^-1 X^T y
    XTX = np.dot(X_bias.T, X_bias)
    beta = np.linalg.solve(XTX, np.dot(X_bias.T, y))

    # Compute residuals and standard errors
    residuals = y - np.dot(X_bias, beta)
    rss = np.sum(residuals ** 2)
    df = n_samples - p_params

    sigma_sq = rss / df
    cov_matrix = sigma_sq * np.linalg.inv(XTX)
    standard_errors = np.sqrt(np.diag(cov_matrix))

    t_stats = beta / standard_errors
    p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df))

    return {
        "beta": beta,
        "standard_errors": standard_errors,
        "t_stats": t_stats,
        "p_values": p_values,
        "df": df,
    }


class DifferenceInDifferences:
    """Estimates the causal impact of a treatment using Difference-in-Differences (DiD).

    # TODO: Support incorporating external covariate matrices into OLS adjustments.
    # TODO: Implement cluster-robust standard errors to handle correlated errors across repeat-measure cohort panels.
    """

    def __init__(self):
        """Initializes the DiD estimator."""
        self.treatment_effect = 0.0
        self.p_value = 1.0
        self.standard_error = 0.0
        self.summary_results = {}

    def fit(self, y: np.ndarray, treatment: np.ndarray, post: np.ndarray):
        """Fits the DiD OLS interaction model: Y = b0 + b1*T + b2*P + d*(T*P).

        Args:
            y (np.ndarray): Outcome values.
            treatment (np.ndarray): Binary treatment indicator (1 if treated, 0 if control).
            post (np.ndarray): Binary time indicator (1 if post-treatment, 0 if pre-treatment).
        """
        # Stack treatment, post, and interaction term (treatment * post)
        X = np.column_stack([treatment, post, treatment * post])
        results = fit_ols(X, y)

        # The interaction coefficient is the treatment effect (index 3 of beta including bias)
        self.treatment_effect = float(results["beta"][3])
        self.standard_error = float(results["standard_errors"][3])
        self.p_value = float(results["p_values"][3])
        self.summary_results = results
        return self

    def check_parallel_trends(self, y_pre: np.ndarray, treatment_pre: np.ndarray, period_pre: np.ndarray) -> bool:
        """Verifies the parallel trends assumption using historical pre-period timelines.

        Fits a pseudo-interaction on pre-treatment periods. Returns True if the pre-period
        trend interaction is statistically insignificant (p-value > 0.05).

        Args:
            y_pre (np.ndarray): Pre-period historical outcome values.
            treatment_pre (np.ndarray): Pre-period treatment indicator.
            period_pre (np.ndarray): Numeric pre-period index indicator.

        Returns:
            bool: True if trends are parallel, False otherwise.
        """
        X = np.column_stack([treatment_pre, period_pre, treatment_pre * period_pre])
        results = fit_ols(X, y_pre)

        p_val_interaction = float(results["p_values"][3])
        return p_val_interaction > 0.05
