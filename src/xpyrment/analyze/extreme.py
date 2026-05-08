"""Extreme Value Theory (EVT) for Heavy-Tailed Conversions (Block 32).

Fits Generalized Pareto Distributions (GPD) over exceedance thresholds to analyze
extreme conversion lift and tail hazard ratios.
"""

from typing import Dict, Tuple, Union
import numpy as np


class ExtremeValueTailEstimator:
    """Estimates tail indices and expected extreme lift using Generalized Pareto Distributions.

    # TODO: Add a profile likelihood fallback optimizer to support shape parameter estimation when xi is outside the MOM boundary [0, 0.5].
    # TODO: Implement automated threshold choice heuristics (e.g., using Hill plots or Gertensgarbe's sequential tests).
    """

    def __init__(self, percentile: float = 0.95) -> None:
        """Initializes the extreme value estimator.

        Args:
            percentile (float): Percentile threshold to define the tail cutoff. Defaults to 0.95.
        """
        self.percentile = percentile
        self.threshold_: float = 0.0
        self.scale_: float = 0.0  # sigma parameter
        self.shape_: float = 0.0  # xi tail index parameter

    def fit(self, outcomes: np.ndarray) -> "ExtremeValueTailEstimator":
        """Fits the GPD parameters on outcomes exceeding the pre-specified percentile.

        Uses the Method of Moments (MOM) for GPD estimation (valid for shape xi < 0.5):
            sigma_MOM = 1/2 * mean_exceed * (1 + mean_exceed^2 / var_exceed)
            xi_MOM = 1/2 * (1 - mean_exceed^2 / var_exceed)
        """
        y = outcomes.astype(float)
        self.threshold_ = float(np.percentile(y, self.percentile * 100))

        exceedances = y[y > self.threshold_] - self.threshold_
        if len(exceedances) < 5:
            # Fallback if too few tail samples
            self.scale_ = 1.0
            self.shape_ = 0.1
            return self

        mean_ex = np.mean(exceedances)
        var_ex = np.var(exceedances, ddof=1)
        if var_ex <= 1e-9:
            var_ex = 1e-9

        # Method of Moments formulas
        ratio = (mean_ex ** 2) / var_ex
        self.scale_ = float(0.5 * mean_ex * (1.0 + ratio))
        self.shape_ = float(0.5 * (1.0 - ratio))

        # Enforce physical constraints for heavy tails
        if self.scale_ <= 1e-5:
            self.scale_ = 1e-5
        if self.shape_ >= 0.5:
            self.shape_ = 0.49  # Stabilize boundary

        return self

    def expected_shortfall(self) -> float:
        """Calculates Expected Shortfall (conditional tail expectation) for fitted GPD.

        ES_u = u + (sigma + xi * (VaR - u)) / (1 - xi)
        At the threshold itself (VaR = u), ES is simply:
            ES = u + sigma / (1 - xi)
        """
        if self.shape_ >= 1.0:
            return float("inf")  # Tail variance is infinite
        return self.threshold_ + self.scale_ / (1.0 - self.shape_)

    @property
    def metrics(self) -> Dict[str, float]:
        """Returns GPD parameters and tail indices."""
        return {
            "percentile_threshold": self.threshold_,
            "gpd_scale_sigma": self.scale_,
            "gpd_shape_xi": self.shape_,
            "expected_shortfall": self.expected_shortfall(),
        }
