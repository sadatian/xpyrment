"""Automated Covariate Balance Checking & Love Plots (Block 42).

Provides algebraic assessment of sample randomized balances across multi-dimensional baseline
covariates, exporting standardized SMD scores and printable Love Plot visualizations.
"""

from typing import Dict, List
import numpy as np


class CovariateBalanceChecker:
    """Computes SMDs, variance ratios, and outputs formatted balance diagnostics.

    TODO: Add Support for Mahalanobis-based joint multi-dimensional covariate imbalance distance checks.
    TODO: Implement empirical CDF distance checks (Kolmogorov-Smirnov) to assess higher-order covariate distribution balance.
    """

    def __init__(self, covariate_names: List[str] = None) -> None:
        """Initializes the balance checker.

        Args:
            covariate_names (List[str], optional): Custom list of labels for each covariate column.
        """
        self.cov_names = covariate_names
        self.diagnostics_: Dict[str, Dict[str, float]] = {}

    def fit(self, covariates: np.ndarray, treatment: np.ndarray) -> "CovariateBalanceChecker":
        """Calculates standardized mean differences (SMD) and variance ratios (VR).

        Args:
            covariates (np.ndarray): Covariate matrix of shape (N, P).
            treatment (np.ndarray): Binary treatment assignment vector of shape (N,) with values 0 or 1.
        """
        X = covariates.astype(float)
        T = treatment.astype(int)

        N, P = X.shape
        if len(T) != N:
            raise ValueError("Row dimension of covariates must match length of treatment vector.")

        # Resolve covariate names
        if self.cov_names is None:
            self.cov_names = [f"covariate_{i}" for i in range(P)]
        elif len(self.cov_names) != P:
            raise ValueError("Length of covariate_names must match the number of columns in covariates.")

        ctrl_idx = (T == 0)
        trt_idx = (T == 1)

        X_ctrl = X[ctrl_idx]
        X_trt = X[trt_idx]

        self.diagnostics_ = {}
        for idx, name in enumerate(self.cov_names):
            vals_ctrl = X_ctrl[:, idx]
            vals_trt = X_trt[:, idx]

            mean_ctrl = float(np.mean(vals_ctrl)) if len(vals_ctrl) > 0 else 0.0
            mean_trt = float(np.mean(vals_trt)) if len(vals_trt) > 0 else 0.0

            var_ctrl = float(np.var(vals_ctrl, ddof=1)) if len(vals_ctrl) > 1 else 0.0
            var_trt = float(np.var(vals_trt, ddof=1)) if len(vals_trt) > 1 else 0.0

            # Standardized Mean Difference (SMD)
            # SMD = (Mean_T - Mean_C) / sqrt((Var_T + Var_C) / 2)
            denom_smd = np.sqrt((var_ctrl + var_trt) / 2.0)
            smd = (mean_trt - mean_ctrl) / denom_smd if denom_smd > 0.0 else 0.0

            # Variance Ratio (VR) = Var_T / Var_C
            vr = var_trt / var_ctrl if var_ctrl > 0.0 else 1.0

            self.diagnostics_[name] = {
                "mean_control": mean_ctrl,
                "mean_treatment": mean_trt,
                "variance_control": var_ctrl,
                "variance_treatment": var_trt,
                "smd": float(smd),
                "variance_ratio": float(vr),
            }

        return self

    def generate_love_plot(self) -> str:
        """Generates a text-based ASCII 'Love Plot' representing Standardized Mean Differences."""
        if not self.diagnostics_:
            raise ValueError("Model must be fitted before generating Love Plots.")

        # Header
        lines = [
            "=================== COVARIATE BALANCE LOVE PLOT ===================",
            f"{'Covariate Name':<25} | {'SMD Balance':<31} | {'Value':<6}",
            "-------------------------------------------------------------------",
        ]

        # Draw an ASCII chart centered at 0.0 running from -0.5 to 0.5
        # Range of 21 columns representing intervals of 0.05
        for name, stats in self.diagnostics_.items():
            smd = stats["smd"]
            
            # Clamp SMD for visualization
            smd_clamped = max(-0.5, min(0.5, smd))
            # Calculate position on [-0.5, 0.5] mapped to index 0 to 20
            col_idx = int(round((smd_clamped + 0.5) * 20))
            
            chart_chars = list("----------|----------")
            # Index 10 is center '|' (representing 0.0)
            chart_chars[col_idx] = "X"

            chart_str = "".join(chart_chars)
            lines.append(f"{name:<25} | [{chart_str}] | {smd:+.4f}")

        lines.append("================---------------------------------================")
        lines.append("Legend: [Left: Treatment < Control] | [Center '|': Balanced] | [Right: Treatment > Control]")
        return "\n".join(lines)
