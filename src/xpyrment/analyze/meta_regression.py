"""Meta-Regression with Knapp-Hartung Standard Errors (Block 38).

Aggregates study-level treatment effect estimates across various trials and cohorts,
using DerSimonian-Laird between-study variance and Knapp-Hartung standard error corrections.
"""

from typing import Dict, Union
import numpy as np


class MetaRegressor:
    """Solves random-effects meta-regression models with Knapp-Hartung standard errors.

    # TODO: Add Restricted Maximum Likelihood (REML) iteration solver as an alternative to the closed-form DerSimonian-Laird estimator.
    # TODO: Support empirical Bayes shrinkage estimations of individual study-level random effects (u_j).
    """

    def __init__(self, l2_penalty: float = 1e-5) -> None:
        """Initializes the meta-regressor.

        Args:
            l2_penalty (float): Ridge penalty for numerical stability. Defaults to 1e-5.
        """
        self.l2_penalty = l2_penalty
        self.beta_: np.ndarray = np.array([])
        self.se_: np.ndarray = np.array([])
        self.p_values_: np.ndarray = np.array([])
        self.tau_sq_: float = 0.0  # Between-study variance parameter

    def fit(self, outcomes: np.ndarray, variances: np.ndarray, covariates: np.ndarray) -> "MetaRegressor":
        """Fits the random-effects meta-regression model.

        Args:
            outcomes (np.ndarray): Study-level estimated treatment effects of shape (J,).
            variances (np.ndarray): Study-level within-study variances (SE^2) of shape (J,).
            covariates (np.ndarray): Covariate matrix of shape (J, P).
        """
        J = len(outcomes)
        P = covariates.shape[1] if len(covariates.shape) > 1 else 1
        
        y = outcomes.astype(float)
        v = variances.astype(float)
        X = covariates.reshape(J, P).astype(float)

        # 1. Estimate initial fixed-effects coefficients to get residuals
        W_fe = np.diag(1.0 / v)
        XTWX_fe = np.dot(np.dot(X.T, W_fe), X) + self.l2_penalty * np.eye(P)
        beta_fe = np.linalg.solve(XTWX_fe, np.dot(np.dot(X.T, W_fe), y))

        # Heterogeneity statistic Q
        residuals_fe = y - np.dot(X, beta_fe)
        Q = np.sum((residuals_fe ** 2) / v)

        # 2. Compute between-study variance tau^2 using DerSimonian-Laird (DL) formula
        # Trace parameter: trace( W - W * X * (X^T * W * X)^-1 * X^T * W )
        inv_XTWX_fe = np.linalg.pinv(XTWX_fe)
        term1 = np.sum(1.0 / v)
        
        W_X = X / v.reshape(-1, 1)
        term2 = np.trace(np.dot(np.dot(W_X.T, W_X), inv_XTWX_fe))
        
        denom = term1 - term2
        if denom <= 0.0:
            denom = 1e-9

        self.tau_sq_ = max(0.0, (Q - (J - P)) / denom)

        # 3. Compute final random-effects coefficients using weights w_j = 1 / (v_j + tau^2)
        w_re = 1.0 / (v + self.tau_sq_)
        W_re = np.diag(w_re)
        
        XTWX_re = np.dot(np.dot(X.T, W_re), X) + self.l2_penalty * np.eye(P)
        self.beta_ = np.linalg.solve(XTWX_re, np.dot(np.dot(X.T, W_re), y))

        # 4. Compute Knapp-Hartung (KH) standard error adjustments
        # Standard covariance matrix
        inv_XTWX_re = np.linalg.pinv(XTWX_re)
        
        # Knapp-Hartung scale factor q
        residuals_re = y - np.dot(X, self.beta_)
        df = J - P
        
        if df > 0:
            q = np.sum(w_re * (residuals_re ** 2)) / df
        else:
            q = 1.0

        # Enforce positive scale factor
        if q < 1.0:
            q = 1.0

        cov_kh = q * inv_XTWX_re
        self.se_ = np.sqrt(np.maximum(0.0, np.diag(cov_kh)))

        # 5. Two-tailed p-values using Student's t-distribution with (J - P) df
        from scipy.stats import t as t_dist
        self.p_values_ = np.zeros(P)
        for i in range(P):
            se = self.se_[i]
            if se > 0 and df > 0:
                t_stat = self.beta_[i] / se
                self.p_values_[i] = float(2 * (1 - t_dist.cdf(abs(t_stat), df)))
            else:
                self.p_values_[i] = 1.0

        return self

    @property
    def results(self) -> Dict[str, Union[float, np.ndarray]]:
        """Returns fitted parameters, tau^2, and Knapp-Hartung coefficient stats."""
        return {
            "between_study_variance_tau_sq": self.tau_sq_,
            "coefficients": self.beta_,
            "knapp_hartung_standard_errors": self.se_,
            "p_values": self.p_values_,
        }
