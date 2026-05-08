"""Interrupted Time Series (ITS) with HAC Standard Errors (Block 35).

Segments and regresses system wide policies under single chronological panels,
computing Newey-West HAC standard errors to correct for temporal autocorrelations.
"""

from typing import Dict, Union
import numpy as np


class InterruptedTimeSeries:
    """Models segmented regression over system-wide updates with Newey-West HAC standard errors.

    # TODO: Support autoregressive integrated moving average (ARIMA) error components integrated with the segmented OLS model.
    # TODO: Add dynamic lag order selection using AIC/BIC information criteria to optimize the Newey-West HAC spectral bandwidth.
    """

    def __init__(self, treatment_index: int) -> None:
        """Initializes the ITS model.

        Args:
            treatment_index (int): Step/index where the policy intervention was activated.
        """
        self.treatment_index = treatment_index
        self.beta_: np.ndarray = np.array([])
        self.standard_errors_: np.ndarray = np.array([])
        self.t_statistics_: np.ndarray = np.array([])
        self.p_values_: np.ndarray = np.array([])

    def fit(self, outcomes: np.ndarray) -> "InterruptedTimeSeries":
        """Fits the interrupted time series model.

        Args:
            outcomes (np.ndarray): Chronological series of outcomes.
        """
        T = len(outcomes)
        if T <= 4:
            raise ValueError("Interrupted time series requires more than 4 data points.")

        # Construct segmented regressors
        t_arr = np.arange(T, dtype=float)
        D_arr = (t_arr >= self.treatment_index).astype(float)
        P_arr = np.maximum(0.0, t_arr - self.treatment_index)

        # Design matrix X: [intercept, time, level_shift, slope_shift]
        X = np.vstack([np.ones(T), t_arr, D_arr, P_arr]).T
        Y = outcomes.astype(float)

        # Solve OLS: beta = (X^T X)^-1 X^T Y
        XTX = np.dot(X.T, X)
        try:
            self.beta_ = np.linalg.solve(XTX, np.dot(X.T, Y))
        except np.linalg.LinAlgError:
            self.beta_ = np.linalg.pinv(XTX).dot(np.dot(X.T, Y))

        # Compute residuals
        residuals = Y - np.dot(X, self.beta_)

        # --- Compute Newey-West HAC Covariance Matrix ---
        # Automatically choose lag threshold L = floor(4 * (T / 100)^(2/9))
        L = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
        if L >= T:
            L = T - 1
        if L < 1:
            L = 1

        # Gamma_0 (standard White heteroskedasticity matrix)
        Gamma_0 = np.zeros((4, 4))
        for t in range(T):
            x_t = X[t].reshape(-1, 1)
            Gamma_0 += (residuals[t] ** 2) * np.dot(x_t, x_t.T)

        # Adding weighted auto-covariance terms
        Gamma_lag_sum = np.zeros((4, 4))
        for l in range(1, L + 1):
            # Bartlett weight: 1 - l / (L + 1)
            w_l = 1.0 - (l / (L + 1.0))
            Gamma_l = np.zeros((4, 4))
            for t in range(l, T):
                x_t = X[t].reshape(-1, 1)
                x_t_lag = X[t - l].reshape(-1, 1)
                Gamma_l += residuals[t] * residuals[t - l] * np.dot(x_t, x_t_lag.T)
            Gamma_lag_sum += w_l * (Gamma_l + Gamma_l.T)

        # Omega_HAC pooled spectral matrix
        Omega_HAC = Gamma_0 + Gamma_lag_sum

        # Covariance matrix: (X^T X)^-1 * Omega_HAC * (X^T X)^-1
        XTX_inv = np.linalg.pinv(XTX)
        cov_matrix = np.dot(np.dot(XTX_inv, Omega_HAC), XTX_inv)

        # Standard errors & statistics
        df = T - 4
        from scipy.stats import t as t_dist
        
        self.standard_errors_ = np.sqrt(np.maximum(0.0, np.diag(cov_matrix)))
        self.t_statistics_ = np.zeros(4)
        self.p_values_ = np.zeros(4)

        for i in range(4):
            se = self.standard_errors_[i]
            if se > 0:
                t_stat = self.beta_[i] / se
                self.t_statistics_[i] = t_stat
                self.p_values_[i] = float(2 * (1 - t_dist.cdf(abs(t_stat), df))) if df > 0 else 1.0
            else:
                self.t_statistics_[i] = 0.0
                self.p_values_[i] = 1.0

        return self

    @property
    def results(self) -> Dict[str, Dict[str, float]]:
        """Returns coefficient summaries, standard errors, and p-values."""
        names = ["intercept", "time_trend", "level_shift", "post_slope_shift"]
        summary = {}
        for idx, name in enumerate(names):
            summary[name] = {
                "coefficient": float(self.beta_[idx]),
                "hac_standard_error": float(self.standard_errors_[idx]),
                "t_statistic": float(self.t_statistics_[idx]),
                "p_value": float(self.p_values_[idx]),
            }
        return summary
