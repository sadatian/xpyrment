"""Instrumental Variables (IV) with 2-Stage Least Squares (2SLS) (Block 37).

Estimates Complier Average Causal Effects (CACE) when non-compliance violates
standard randomization assumptions.
"""

from typing import Dict, Optional
import numpy as np


class InstrumentalVariables2SLS:
    """Estimates Complier Average Causal Effect (CACE) using two-stage least squares (2SLS).

    # TODO: Implement robust Huber-White sandwich standard error estimators to handle heteroskedasticity in the second-stage residuals.
    # TODO: Support multi-valued discrete and continuous treatment indicators under general control-function approaches.
    """

    def __init__(self, l2_penalty: float = 1e-5) -> None:
        """Initializes the 2SLS solver.

        Args:
            l2_penalty (float): Small ridge penalty for stage-specific OLS stability. Defaults to 1e-5.
        """
        self.l2_penalty = l2_penalty
        self.cace_: float = 0.0
        self.se_cace_: float = 0.0
        self.p_value_: float = 1.0
        self.stage1_f_stat_: float = 0.0

    def fit(self, outcome: np.ndarray, treatment_received: np.ndarray, instrument: np.ndarray) -> "InstrumentVariables2SLS":
        """Fits the 2SLS model:

            Stage 1: D_hat = Z * gamma
            Stage 2: Y = D_hat * beta + epsilon
        """
        from xpyrment.validate.clean import validate_estimation_inputs, verify_collinearity, clean_array
        
        # Clean and match all three vectors safely
        Y, D, Z = validate_estimation_inputs(outcome, treatment_received, instrument, min_samples=3)
        N = len(Y)

        # Enforce that treatment received and instrument are not constants (variance > 0)
        _ = clean_array(Z, allow_zero_var=False, name="instrument")
        _ = clean_array(D, allow_zero_var=False, name="treatment_received")

        # --- Stage 1: Regress Treatment Received on Instrument Assignment ---
        X1 = np.vstack([np.ones(N), Z]).T
        verify_collinearity(X1)
        # Solve OLS: gamma = (X1^T X1)^-1 X1^T D
        XTX1 = np.dot(X1.T, X1) + self.l2_penalty * np.eye(2)
        XTX1[0, 0] = np.dot(X1[:, 0], X1[:, 0])  # No penalty on intercept
        
        gamma = np.linalg.solve(XTX1, np.dot(X1.T, D))
        D_hat = np.dot(X1, gamma)

        # Compute Stage 1 F-statistic to diagnose weak instruments
        residuals1 = D - D_hat
        df_num1 = 1
        df_den1 = N - 2
        var_err1 = np.sum(residuals1 ** 2) / df_den1 if df_den1 > 0 else 1e-9
        
        # Mean square model / Mean square error
        ms_model1 = np.sum((D_hat - np.mean(D)) ** 2) / df_num1
        self.stage1_f_stat_ = float(ms_model1 / max(1e-9, var_err1))

        # --- Stage 2: Regress Outcome on Predicted Treatment ---
        X2 = np.vstack([np.ones(N), D_hat]).T
        XTX2 = np.dot(X2.T, X2) + self.l2_penalty * np.eye(2)
        XTX2[0, 0] = np.dot(X2[:, 0], X2[:, 0])

        beta = np.linalg.solve(XTX2, np.dot(X2.T, Y))
        self.cace_ = float(beta[1])

        # --- Standard Error and Covariance Matrix Computation ---
        # Note: SE must be computed using ACTUAL treatment D in the residual vector!
        actual_design = np.vstack([np.ones(N), D]).T
        residuals2 = Y - np.dot(actual_design, beta)
        
        df_den2 = N - 2
        var_err2 = np.sum(residuals2 ** 2) / df_den2 if df_den2 > 0 else 1e-9
        
        # cov_beta = var_err * (X2_hat^T X2_hat)^-1
        cov_beta = var_err2 * np.linalg.inv(XTX2)
        self.se_cace_ = float(np.sqrt(max(0.0, cov_beta[1, 1])))

        # Compute p-value
        from scipy.stats import t
        if self.se_cace_ > 0 and df_den2 > 0:
            t_stat = self.cace_ / self.se_cace_
            self.p_value_ = float(2 * (1 - t.cdf(abs(t_stat), df_den2)))
        else:
            self.p_value_ = 1.0

        return self

    @property
    def summary(self) -> Dict[str, float]:
        """Returns 2SLS estimation results."""
        return {
            "complier_average_causal_effect": self.cace_,
            "standard_error": self.se_cace_,
            "p_value": self.p_value_,
            "stage1_weak_instrument_f_statistic": self.stage1_f_stat_,
        }

    def to_dict(self) -> dict:
        """Converts the estimator results to a JSON-serializable dictionary.

        Returns:
            dict: Standard Python dictionary containing complaint average causal effect, SE, p-value, etc.
        """
        from xpyrment.core.serialization import make_serializable
        state = {
            "l2_penalty": self.l2_penalty,
            "complier_average_causal_effect": self.cace_,
            "standard_error": self.se_cace_,
            "p_value": self.p_value_,
            "stage1_weak_instrument_f_statistic": self.stage1_f_stat_,
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
