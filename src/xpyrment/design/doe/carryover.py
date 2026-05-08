"""Intertemporal Carryover Decompositions in Switchback & Crossover Designs (Block 26).

Models and estimates direct treatment effects (DTE) alongside carryover/spillover decay
rates from preceding treatment periods.
"""

from typing import Dict, Union
import numpy as np


class CarryoverDecomposition:
    """Estimates Direct Treatment Effects (DTE) and decay curves of intertemporal carryover effects.

    Fits the model:
        Y_t = beta_0 + beta_direct * T_t + beta_carryover * T_{t-1} * exp(-lambda * dt_t) + epsilon_t
    
    Using an optimized coordinate grid search over the exponential decay parameter lambda.

    # TODO: Extend the carryover decomposition to handle multi-stage lag structures (e.g., T_{t-2} and T_{t-3}) with distinct decay vectors.
    # TODO: Add a profile likelihood fallback solver to compute joint asymptotic confidence intervals for both lambda and the beta parameters.
    """

    def __init__(self, l2_penalty: float = 1e-5) -> None:
        """Initializes the carryover decomposer.

        Args:
            l2_penalty (float): L2 regularization parameter for OLS stability. Defaults to 1e-5.
        """
        self.l2_penalty = l2_penalty
        self.beta_direct_: float = 0.0
        self.beta_carryover_: float = 0.0
        self.beta_baseline_: float = 0.0
        self.lambda_: float = 0.0
        self.standard_errors_: Dict[str, float] = {}
        self.p_values_: Dict[str, float] = {}

    def fit(self, outcomes: np.ndarray, treatments: np.ndarray, times: np.ndarray) -> "CarryoverDecomposition":
        """Fits the carryover decomposition model.

        Args:
            outcomes (np.ndarray): Target outcome series of shape (N,). Must be sorted chronologically.
            treatments (np.ndarray): Binary treatment series of shape (N,) (0 = control, 1 = treated).
            times (np.ndarray): Chronological timestamps of shape (N,).

        Returns:
            CarryoverDecomposition: Fitted decomposer.
        """
        N = len(outcomes)
        if N < 3:
            raise ValueError("Carryover estimation requires at least 3 historical observations.")

        Y = outcomes[1:].astype(float)
        T_curr = treatments[1:].astype(float)
        T_prev = treatments[:-1].astype(float)
        
        # Calculate time elapsed since last transition dt_t
        dt = np.diff(times).astype(float)

        # Grid search over lambda range to minimize MSE
        best_mse = float("inf")
        best_lambda = 0.0
        best_beta = np.zeros(3)
        best_cov = np.zeros((3, 3))

        # Search lambda in log-space from exp(-5) to exp(2)
        lambdas = np.logspace(-2, 1, 100)

        for lmb in lambdas:
            # Construct the carryover feature: T_{t-1} * exp(-lambda * dt_t)
            carryover_feat = T_prev * np.exp(-lmb * dt)

            # Design matrix: [intercept, current_treatment, carryover_feature]
            X = np.vstack([np.ones(N - 1), T_curr, carryover_feat]).T
            
            # Solve OLS: beta = (X^T X + L2 * I)^-1 X^T Y
            XTX = np.dot(X.T, X)
            XTX_reg = XTX + self.l2_penalty * np.eye(3)
            XTX_reg[0, 0] = XTX[0, 0]  # Don't regularize intercept

            try:
                beta = np.linalg.solve(XTX_reg, np.dot(X.T, Y))
            except np.linalg.LinAlgError:
                continue

            residuals = Y - np.dot(X, beta)
            mse = np.mean(residuals ** 2)

            if mse < best_mse:
                best_mse = mse
                best_lambda = lmb
                best_beta = beta
                
                # Covariance matrix of beta: var_error * (X^T X)^-1
                df = (N - 1) - 3
                if df > 0:
                    var_err = np.sum(residuals ** 2) / df
                    best_cov = var_err * np.linalg.inv(XTX_reg)
                else:
                    best_cov = np.zeros((3, 3))

        self.beta_baseline_ = float(best_beta[0])
        self.beta_direct_ = float(best_beta[1])
        self.beta_carryover_ = float(best_beta[2])
        self.lambda_ = float(best_lambda)

        # Extract standard errors & compute two-tailed p-values
        from scipy.stats import t
        df = (N - 1) - 3
        
        feature_names = ["baseline", "direct", "carryover"]
        for idx, name in enumerate(feature_names):
            se = float(np.sqrt(max(0.0, best_cov[idx, idx])))
            self.standard_errors_[name] = se
            
            if se > 0 and df > 0:
                t_stat = best_beta[idx] / se
                p_val = float(2 * (1 - t.cdf(abs(t_stat), df)))
                self.p_values_[name] = p_val
            else:
                self.p_values_[name] = 1.0

        return self

    @property
    def summary(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """Returns a summary dictionary of the fitted carryover effects."""
        return {
            "decay_constant_lambda": self.lambda_,
            "coefficients": {
                "baseline": self.beta_baseline_,
                "direct_treatment_effect": self.beta_direct_,
                "carryover_effect": self.beta_carryover_,
            },
            "standard_errors": self.standard_errors_,
            "p_values": self.p_values_,
        }
