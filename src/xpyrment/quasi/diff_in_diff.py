"""Difference-in-Differences (DiD) quasi-experimental estimators.

This module provides the `DifferenceInDifferences` class to measure causal treatment
effects when clean randomized control groups are unavailable, including trend validations.
"""

from typing import Dict, Any, Optional
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
    from xpyrment.validate.clean import validate_estimation_inputs, verify_collinearity, clean_array

    # Clean dependent outcome variable y
    y_clean, = validate_estimation_inputs(y)
    
    # Convert and clean predictors X
    X_clean = np.asarray(X, dtype=np.float64)
    if X_clean.ndim == 1:
        X_clean = X_clean.reshape(-1, 1)
    
    for col in range(X_clean.shape[1]):
        X_clean[:, col] = clean_array(X_clean[:, col], name=f"predictor_col_{col}")

    n_samples, p_features = X_clean.shape
    X_bias = np.hstack([np.ones((n_samples, 1)), X_clean])
    p_params = X_bias.shape[1]

    # Validate collinearity of combined bias matrix
    verify_collinearity(X_bias)

    # beta = (X^T X)^-1 X^T y
    XTX = np.dot(X_bias.T, X_bias)
    beta = np.linalg.solve(XTX, np.dot(X_bias.T, y_clean))

    # Compute residuals and standard errors
    residuals = y_clean - np.dot(X_bias, beta)
    rss = np.sum(residuals ** 2)
    df = n_samples - p_params

    # Robust fallback for perfect fit / zero-residuals or saturated systems
    if df < 0:
        raise ValueError(
            f"Underdetermined system: Insufficient samples for OLS. Got {n_samples} samples and {p_params} parameters."
        )

    sigma_sq = rss / df if (df > 0 and rss > 1e-12) else 0.0
    cov_matrix = sigma_sq * np.linalg.inv(XTX) if (sigma_sq > 0 and df > 0) else np.zeros((p_params, p_params))
    standard_errors = np.sqrt(np.diag(cov_matrix)) if df > 0 else np.array([np.inf] * p_params)

    # Guard against zero standard error division or zero degrees of freedom
    t_stats = np.where((standard_errors > 1e-12) & (standard_errors < np.inf), beta / standard_errors, 0.0)
    p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df)) if df > 0 else np.ones(p_params)

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

    def to_dict(self) -> dict:
        """Converts the estimator results to a JSON-serializable dictionary.

        Returns:
            dict: Standard Python dictionary containing treatment effect, SE, p-value, and summary results.
        """
        from xpyrment.core.serialization import make_serializable
        state = {
            "treatment_effect": self.treatment_effect,
            "standard_error": self.standard_error,
            "p_value": self.p_value,
            "summary_results": self.summary_results,
        }
        return make_serializable(state)

    def to_json(self, indent: Optional[int] = None) -> str:
        """Converts the estimator results to a standardized JSON string.

        Args:
            indent (Optional[int]): Indentation level.

        Returns:
            str: JSON string.
        """
        from xpyrment.core.serialization import serialize_to_json
        return serialize_to_json(self.to_dict(), indent=indent)


class ParallelTrendsPlaceboTest:
    """Performs rigorous multi-period pre-treatment placebo testing to validate Parallel Trends.

    TODO: Support panel clustered standard errors to account for correlation over repeated unit observations.
    TODO: Add synthetic difference-in-differences (SDID) unit weights and time weights inside the placebo trends check.
    """

    def __init__(self, significance_level: float = 0.05) -> None:
        """Initializes the placebo trend tester.

        Args:
            significance_level (float): The Type I error threshold for placebo rejection. Defaults to 0.05.
        """
        self.alpha = significance_level

    def fit_placebo_test(self, y: np.ndarray, treatment: np.ndarray, time: np.ndarray, treatment_start_time: float) -> Dict[str, Any]:
        """Runs a placebo regression restricting the timeline to pre-treatment periods.

        Simulates a "placebo treatment" starting at the midpoint of the pre-treatment phase.
        If the interaction coefficient of this placebo treatment is statistically significant,
        then the Parallel Trends assumption is likely violated.
        """
        # Restrict to pre-treatment data
        pre_idx = (time < treatment_start_time)
        
        y_pre = y[pre_idx]
        t_pre = treatment[pre_idx]
        time_pre = time[pre_idx]

        if len(y_pre) == 0:
            raise ValueError("No pre-treatment periods found under the specified treatment_start_time.")

        unique_times = np.unique(time_pre)
        if len(unique_times) < 2:
            raise ValueError("Placebo tests require at least 2 distinct pre-treatment periods.")

        # मिडपॉइंट placebo start time
        placebo_start = float(np.median(unique_times))
        
        # Binary indicator for placebo post-period
        placebo_post = (time_pre >= placebo_start).astype(int)

        # Fit placebo regression: Y = b0 + b1*T + b2*PlaceboPost + d*(T * PlaceboPost)
        X = np.column_stack([t_pre, placebo_post, t_pre * placebo_post])
        results = fit_ols(X, y_pre)

        # The placebo coefficient is at index 3 of beta
        placebo_coef = float(results["beta"][3])
        placebo_se = float(results["standard_errors"][3])
        placebo_p = float(results["p_values"][3])

        # If placebo is significant, parallel trends is rejected (violated)
        trends_parallel = placebo_p > self.alpha

        self.results_ = {
            "placebo_coefficient": placebo_coef,
            "placebo_standard_error": placebo_se,
            "placebo_p_value": placebo_p,
            "trends_parallel": trends_parallel,
            "placebo_start_time": placebo_start,
        }

        return self.results_

    def to_dict(self) -> dict:
        """Converts the placebo test results to a JSON-serializable dictionary.

        Returns:
            dict: Standard Python dictionary.
        """
        from xpyrment.core.serialization import make_serializable
        state = {
            "alpha": self.alpha,
            "results": getattr(self, "results_", None)
        }
        return make_serializable(state)

    def to_json(self, indent: Optional[int] = None) -> str:
        """Converts the placebo test results to a standardized JSON string.

        Args:
            indent (Optional[int]): Indentation level.

        Returns:
            str: JSON string.
        """
        from xpyrment.core.serialization import serialize_to_json
        return serialize_to_json(self.to_dict(), indent=indent)

