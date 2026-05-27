"""Analytical Power Analysis & Sample Size Estimator (Block 44).

Estimates statistical power, required sample sizes, and Minimum Detectable Effects (MDE)
for standard as well as cluster randomized experimental designs.
"""

import numpy as np
from scipy.stats import norm


class AnalyticalPowerCalculator:
    """Computes sample sizes, statistical power, and Minimum Detectable Effects (MDE).

    TODO: Extend power calculations to unequal allocation ratios with multiple treatment arms using Dunnett's adjustment.
    TODO: Add exact simulation-based power curves utilizing empirical bootstrap baseline variance matrices.
    """

    def __init__(self, alpha: float = 0.05, power: float = 0.80) -> None:
        """Initializes the power calculator.

        Args:
            alpha (float): Targeted significance/Type I error rate. Defaults to 0.05.
            power (float): Targeted power/1 - Type II error rate. Defaults to 0.80.
        """
        self.alpha = alpha
        self.power = power

    def compute_sample_size(self, mde: float, variance: float, ratio: float = 1.0) -> int:
        """Computes sample size per group for a two-sample t-test.

         ratio = n_treatment / n_control
        """
        if mde <= 0.0 or variance <= 0.0:
            raise ValueError("MDE and variance must be strictly positive values.")

        z_alpha = norm.ppf(1.0 - self.alpha / 2.0)
        z_beta = norm.ppf(self.power)

        # Standard formulation: n_control = (1 + 1/ratio) * ( (z_alpha + z_beta) * sigma / mde )^2
        n_ctrl = (1.0 + 1.0 / ratio) * ((z_alpha + z_beta) ** 2) * variance / (mde ** 2)
        return int(np.ceil(n_ctrl))

    def compute_mde(self, sample_size: int, variance: float, ratio: float = 1.0) -> float:
        """Computes Minimum Detectable Effect (MDE) under configured constraints."""
        if sample_size <= 0 or variance <= 0.0:
            raise ValueError("Sample size and variance must be strictly positive values.")

        z_alpha = norm.ppf(1.0 - self.alpha / 2.0)
        z_beta = norm.ppf(self.power)

        # mde = (z_alpha + z_beta) * sqrt( variance * (1 + 1/ratio) / n_control )
        factor = variance * (1.0 + 1.0 / ratio) / sample_size
        return float((z_alpha + z_beta) * np.sqrt(factor))

    def compute_power(self, sample_size: int, mde: float, variance: float, ratio: float = 1.0) -> float:
        """Computes statistical power given sample size and target treatment effect size."""
        if sample_size <= 0 or mde <= 0.0 or variance <= 0.0:
            raise ValueError("All arguments must be strictly positive to estimate power.")

        z_alpha = norm.ppf(1.0 - self.alpha / 2.0)
        
        # Power = Phi( mde / sqrt(variance * (1 + 1/ratio) / n_control) - z_alpha )
        se = np.sqrt(variance * (1.0 + 1.0 / ratio) / sample_size)
        z_val = (mde / se) - z_alpha
        return float(norm.cdf(z_val))

    def adjust_for_clusters(self, base_n: int, avg_cluster_size: int, icc: float) -> int:
        """Adjusts sample size requirements for clustered designs using the VIF model.

            VIF = 1 + (M - 1) * ICC
        """
        if avg_cluster_size <= 1:
            return base_n
        if not (0.0 <= icc <= 1.0):
            raise ValueError("Intra-cluster correlation (ICC) must be in the range [0, 1].")

        vif = 1.0 + (avg_cluster_size - 1) * icc
        return int(np.ceil(base_n * vif))
