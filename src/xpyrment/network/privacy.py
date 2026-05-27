"""Differential Privacy (DP) Noise Addition for Secure Analytics (Block 33).

Applies Laplace and Gaussian privacy mechanisms to protect experiment metrics and
summary statistics before external sharing or pooling.
"""

from typing import Tuple
import numpy as np


class DifferentialPrivacyEngine:
    """Calibrates and adds mathematical noise to statistics to satisfy DP bounds.

    # TODO: Implement Renyi Differential Privacy (RDP) accounting to support tight composition over multi-pass queries.
    # TODO: Support private covariance matrix noise injection based on the Wishart mechanism or advanced output perturbation.
    """

    def __init__(self, bounds: Tuple[float, float]) -> None:
        """Initializes the privacy engine with the global support bounds of the metric.

        Args:
            bounds (Tuple[float, float]): The physical bounds [min_val, max_val] of the metric.
        """
        self.min_val, self.max_val = bounds
        self.range_ = self.max_val - self.min_val

    def get_mean_sensitivity(self, n_samples: int) -> float:
        """Computes the global sensitivity of the sample mean:

            Delta = (max - min) / N
        """
        if n_samples <= 0:
            raise ValueError("Sample size must be strictly positive to compute sensitivity.")
        return self.range_ / n_samples

    def add_laplace_noise(self, mean_stat: float, n_samples: int, epsilon: float, rng: np.random.Generator = None) -> float:
        """Applies the Laplace Mechanism to satisfy pure epsilon-Differential Privacy.

        Noise scale b = Sensitivity / epsilon
        """
        if epsilon <= 0.0:
            raise ValueError("Privacy budget epsilon must be strictly positive.")
            
        _rng = rng or np.random.default_rng()
        sensitivity = self.get_mean_sensitivity(n_samples)
        scale = sensitivity / epsilon

        noise = _rng.laplace(0.0, scale)
        return float(mean_stat + noise)

    def add_gaussian_noise(
        self, mean_stat: float, n_samples: int, epsilon: float, delta: float, rng: np.random.Generator = None
    ) -> float:
        """Applies the Gaussian Mechanism to satisfy approximate (epsilon, delta)-DP.

        Standard deviation sigma = Sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
        """
        if epsilon <= 0.0 or delta <= 0.0 or delta >= 1.0:
            raise ValueError("Gaussian DP budgets must satisfy epsilon > 0 and 0 < delta < 1.")

        _rng = rng or np.random.default_rng()
        sensitivity = self.get_mean_sensitivity(n_samples)
        
        sigma = (sensitivity * np.sqrt(2.0 * np.log(1.25 / delta))) / epsilon
        noise = _rng.normal(0.0, sigma)
        return float(mean_stat + noise)
