"""Non-Gaussian Copula-Based Multi-Metric Inference (Block 24).

Jointly models multiple correlated non-Gaussian experimental metrics (e.g. Conversion and Revenue)
using empirical copulas to compute joint covariance structures and run robust joint Wald hypothesis tests.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2


class CopulaMultiMetricInference:
    """Joint inference engine for correlated non-Gaussian metrics using empirical Gaussian Copulas.

    Estimates univariate empirical cumulative distribution functions (CDFs) to project
    marginal observations onto uniform space [0, 1], maps to standard normal space,
    estimates the latent correlation structure, and performs a joint Wald test.

    # TODO: Support parametric copula families (such as Clayton or Gumbel) to capture asymmetric tail dependencies.
    # TODO: Add multivariate p-value corrections (e.g. step-down procedures) to control Family-Wise Error Rate (FWER) under copula.
    """

    def __init__(self, l2_penalty: float = 1e-6) -> None:
        """Initializes the copula inference engine.

        Args:
            l2_penalty (float): Regularization for covariance matrix inversion. Defaults to 1e-6.
        """
        self.l2_penalty = l2_penalty
        self.copula_correlation_: np.ndarray = None
        self.marginal_cdfs_: List[Callable[[np.ndarray], np.ndarray]] = []

    def _compute_empirical_ranks(self, x: np.ndarray) -> np.ndarray:
        """Computes empirical ranks scaled to (0, 1) to avoid infinite normal scores."""
        N = len(x)
        ranks = np.argsort(np.argsort(x)) + 1
        return ranks / (N + 1)

    def fit_copula(self, df: pd.DataFrame, metric_cols: List[str]) -> "CopulaMultiMetricInference":
        """Fits empirical marginals and constructs the latent copula correlation matrix.

        Args:
            df (pd.DataFrame): DataFrame containing experimental metric columns.
            metric_cols (List[str]): List of metric column names to include in the joint copula.

        Returns:
            CopulaMultiMetricInference: Fitted engine.
        """
        D = len(metric_cols)
        N = len(df)
        
        Z = np.zeros((N, D))

        for j, col in enumerate(metric_cols):
            val = df[col].to_numpy().astype(float)
            # Map values to uniform empirical ranks
            u = self._compute_empirical_ranks(val)
            # Map uniform ranks to standard normal scores
            Z[:, j] = norm.ppf(u)

        # Compute latent Pearson correlation matrix of standard normal scores
        self.copula_correlation_ = np.corrcoef(Z, rowvar=False)
        if D == 1:
            self.copula_correlation_ = np.array([[1.0]])

        return self

    def test_joint_shift(
        self, df: pd.DataFrame, treatment_col: str, metric_cols: List[str]
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Performs a joint Wald test of treatment effects across all correlated metrics.

        H0: delta_1 = delta_2 = ... = delta_D = 0
        
        Args:
            df (pd.DataFrame): Experimental dataset.
            treatment_col (str): Column indicating binary treatment assignment (0 = control, 1 = treated).
            metric_cols (List[str]): Correlated metric column names.

        Returns:
            Dict: Dictionary containing joint Wald statistic, joint p-value, individual lifts, and covariance.
        """
        D = len(metric_cols)
        if self.copula_correlation_ is None or self.copula_correlation_.shape[0] != D:
            self.fit_copula(df, metric_cols)

        # Split into control and treatment groups
        control_df = df[df[treatment_col] == 0]
        treated_df = df[df[treatment_col] == 1]

        N_c = len(control_df)
        N_t = len(treated_df)

        if N_c < 2 or N_t < 2:
            raise ValueError("Control and treated groups must contain at least 2 observations.")

        # 1. Compute individual treatment effects (mean differences) and standard errors
        delta = np.zeros(D)
        se = np.zeros(D)

        for j, col in enumerate(metric_cols):
            y_c = control_df[col].to_numpy().astype(float)
            y_t = treated_df[col].to_numpy().astype(float)

            delta[j] = np.mean(y_t) - np.mean(y_c)
            var_c = np.var(y_c, ddof=1)
            var_t = np.var(y_t, ddof=1)
            se[j] = np.sqrt(var_c / N_c + var_t / N_t)

        # 2. Reconstruct the joint covariance matrix of the treatment effect estimates
        # Cov(delta_j, delta_l) = R_jl * se_j * se_l
        Sigma_delta = np.zeros((D, D))
        for j in range(D):
            for l in range(D):
                Sigma_delta[j, l] = self.copula_correlation_[j, l] * se[j] * se[l]

        # Regularize to prevent singular matrix errors
        Sigma_delta_reg = Sigma_delta + self.l2_penalty * np.eye(D)

        # 3. Compute joint Wald statistic: W = delta^T * Sigma^-1 * delta
        try:
            inv_sigma = np.linalg.inv(Sigma_delta_reg)
            wald_stat = float(np.dot(delta, np.dot(inv_sigma, delta)))
        except np.linalg.LinAlgError:
            # Fallback using pseudoinverse if inversion fails
            inv_sigma = np.linalg.pinv(Sigma_delta_reg)
            wald_stat = float(np.dot(delta, np.dot(inv_sigma, delta)))

        # 4. Compute joint p-value under asymptotic Chi-squared distribution
        p_val = float(1.0 - chi2.cdf(wald_stat, df=D))

        return {
            "wald_statistic": wald_stat,
            "p_value": p_val,
            "deltas": delta,
            "standard_errors": se,
            "copula_correlation": self.copula_correlation_,
            "covariance_matrix": Sigma_delta,
        }
