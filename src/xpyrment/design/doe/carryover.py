"""Intertemporal Carryover Decompositions in Switchback & Crossover Designs (Block 26).

Models and estimates direct treatment effects (DTE) alongside carryover/spillover decay
rates from preceding treatment periods.
"""

from typing import Dict, Union
import numpy as np


class CarryoverDecomposition:
    """Estimates Direct Treatment Effects (DTE) and decay curves of intertemporal carryover effects.

    Fits the model:
        Y_t = beta_0 + beta_direct * T_t + sum_{k=1}^K beta_carryover_k * T_{t-k} * exp(-lambda_k * dt_t) + epsilon_t
    
    Using an optimized coordinate grid search over the exponential decay parameters lambda.
    """

    def __init__(self, l2_penalty: float = 1e-5, max_lags: int = 1) -> None:
        """Initializes the carryover decomposer.

        Args:
            l2_penalty (float): L2 regularization parameter for OLS stability. Defaults to 1e-5.
            max_lags (int): Maximum number of historical lags to consider. Defaults to 1.
        """
        self.l2_penalty = l2_penalty
        self.max_lags = max_lags
        self.beta_direct_: float = 0.0
        self.beta_carryovers_: np.ndarray = np.array([])
        self.beta_baseline_: float = 0.0
        self.lambdas_: np.ndarray = np.array([])
        self.standard_errors_: Dict[str, float] = {}
        self.p_values_: Dict[str, float] = {}
        self.ll_profile_: Dict[float, float] = {}

    def fit(self, outcomes: np.ndarray, treatments: np.ndarray, times: np.ndarray) -> "CarryoverDecomposition":
        """Fits the carryover decomposition model with multi-stage lags.

        Args:
            outcomes (np.ndarray): Target outcome series of shape (N,). Must be sorted chronologically.
            treatments (np.ndarray): Binary treatment series of shape (N,) (0 = control, 1 = treated).
            times (np.ndarray): Chronological timestamps of shape (N,).

        Returns:
            CarryoverDecomposition: Fitted decomposer.
        """
        N = len(outcomes)
        if N <= self.max_lags + 1:
            raise ValueError(f"Carryover estimation with {self.max_lags} lags requires at least {self.max_lags + 2} observations.")

        Y = outcomes[self.max_lags:].astype(float)
        T_curr = treatments[self.max_lags:].astype(float)
        dt = np.diff(times[self.max_lags-1:]).astype(float) # dt for the current period

        # For simplicity, we assume a single shared decay constant lambda across all lags
        # or we could search for each. Here we implement a shared lambda for the profile likelihood fallback.
        
        best_mse = float("inf")
        best_lambda = 0.0
        best_beta = np.zeros(2 + self.max_lags)
        best_cov = np.zeros((2 + self.max_lags, 2 + self.max_lags))

        # Search lambda in log-space
        lambdas = np.logspace(-2, 1, 100)

        for lmb in lambdas:
            # Construct carryover features for each lag k
            carryover_feats = []
            for k in range(1, self.max_lags + 1):
                T_prev_k = treatments[self.max_lags-k : -k].astype(float)
                # Decay factor scales with lag distance k
                feat = T_prev_k * np.exp(-lmb * k * dt)
                carryover_feats.append(feat)

            # Design matrix: [intercept, current_treatment, carryover_1, ..., carryover_K]
            X = np.vstack([np.ones(len(Y)), T_curr] + carryover_feats).T
            
            # Solve OLS
            XTX = np.dot(X.T, X)
            XTX_reg = XTX + self.l2_penalty * np.eye(X.shape[1])
            XTX_reg[0, 0] = XTX[0, 0]

            try:
                beta = np.linalg.solve(XTX_reg, np.dot(X.T, Y))
            except np.linalg.LinAlgError:
                continue

            residuals = Y - np.dot(X, beta)
            mse = np.mean(residuals ** 2)
            
            # Store log-likelihood for profile likelihood (assuming Gaussian noise)
            ll = -0.5 * len(Y) * (np.log(2 * np.pi * mse) + 1)
            self.ll_profile_[lmb] = ll

            if mse < best_mse:
                best_mse = mse
                best_lambda = lmb
                best_beta = beta
                
                df_resid = len(Y) - X.shape[1]
                if df_resid > 0:
                    var_err = np.sum(residuals ** 2) / df_resid
                    best_cov = var_err * np.linalg.inv(XTX_reg)

        self.beta_baseline_ = float(best_beta[0])
        self.beta_direct_ = float(best_beta[1])
        self.beta_carryovers_ = best_beta[2:].astype(float)
        self.lambdas_ = np.full(self.max_lags, best_lambda) # Shared lambda for now

        # Compute p-values and SEs
        from scipy.stats import t, chi2
        df_resid = len(Y) - (2 + self.max_lags)
        
        self.standard_errors_["baseline"] = float(np.sqrt(max(0.0, best_cov[0, 0])))
        self.standard_errors_["direct"] = float(np.sqrt(max(0.0, best_cov[1, 1])))
        
        for k in range(self.max_lags):
            self.standard_errors_[f"carryover_lag_{k+1}"] = float(np.sqrt(max(0.0, best_cov[2+k, 2+k])))

        for name, se in self.standard_errors_.items():
            if se > 0 and df_resid > 0:
                coeff = self.beta_baseline_ if name == "baseline" else (self.beta_direct_ if name == "direct" else self.beta_carryovers_[int(name.split("_")[-1])-1])
                t_stat = coeff / se
                self.p_values_[name] = float(2 * (1 - t.cdf(abs(t_stat), df_resid)))
            else:
                self.p_values_[name] = 1.0

        # Profile Likelihood CI for lambda
        if self.ll_profile_:
            max_ll = max(self.ll_profile_.values())
            # 95% CI boundary for chi2(1)
            cutoff = max_ll - chi2.ppf(0.95, 1) / 2.0
            ci_lambdas = [l for l, ll in self.ll_profile_.items() if ll >= cutoff]
            if ci_lambdas:
                self.lambda_ci_ = (min(ci_lambdas), max(ci_lambdas))
            else:
                self.lambda_ci_ = (best_lambda, best_lambda)

        return self

    @property
    def summary(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """Returns a summary dictionary of the fitted carryover effects."""
        res = {
            "decay_constant_lambda": float(self.lambdas_[0]) if len(self.lambdas_) > 0 else 0.0,
            "lambda_95_ci": getattr(self, "lambda_ci_", (0.0, 0.0)),
            "coefficients": {
                "baseline": self.beta_baseline_,
                "direct_treatment_effect": self.beta_direct_,
            },
            "standard_errors": self.standard_errors_,
            "p_values": self.p_values_,
        }
        for k, val in enumerate(self.beta_carryovers_):
            res["coefficients"][f"carryover_effect_lag_{k+1}"] = float(val)
        return res
