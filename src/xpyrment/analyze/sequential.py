"""Group Sequential Lan-DeMets Alpha Spending Functions (Block 36).

Computes critical boundaries for running interim inspections using alpha spending
functions, preserving exact multi-look Type I error rates.
"""

from typing import Dict, List, Union
import numpy as np
from scipy.stats import norm


class GroupSequentialMonitor:
    """Computes O'Brien-Fleming and Pocock sequential boundaries via Lan-DeMets spending functions.

    # TODO: Implement exact multivariate normal integration (e.g. using Genz-Bretz algorithms) to solve multi-look joint covariance critical bounds.
    # TODO: Support binding and non-binding futility boundaries using beta-spending formulations to support early stopping for futility.
    """

    def __init__(self, alpha: float = 0.05, spending_type: str = "obrien_fleming") -> None:
        """Initializes the group sequential monitor.

        Args:
            alpha (float): Total Type I error budget. Defaults to 0.05.
            spending_type (str): 'obrien_fleming' or 'pocock' type spending.
        """
        self.alpha = alpha
        if spending_type not in ("obrien_fleming", "pocock"):
            raise ValueError("spending_type must be either 'obrien_fleming' or 'pocock'.")
        self.spending_type = spending_type

    def alpha_spent(self, t: float) -> float:
        """Computes cumulative alpha spent at information fraction t in [0, 1]."""
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return self.alpha

        if self.spending_type == "obrien_fleming":
            # Lan-DeMets (1983) O'Brien-Fleming approximation:
            # alpha(t) = 2 * (1 - Phi(Z_{alpha/2} / sqrt(t)))
            z = norm.ppf(1.0 - self.alpha / 2.0)
            return float(2.0 * (1.0 - norm.cdf(z / np.sqrt(t))))
        else:
            # Lan-DeMets (1983) Pocock approximation:
            # alpha(t) = alpha * ln(1 + (e - 1) * t)
            return float(self.alpha * np.log(1.0 + (np.e - 1.0) * t))

    def compute_boundaries(self, information_fractions: List[float]) -> Dict[str, Union[List[float], np.ndarray]]:
        """Computes sequential critical boundary values Z_crit and p-value boundaries for each look.

        Using sequential spent alpha increments, the critical z-score at look k satisfies:
            P(Reject at look k | No previous rejections) = alpha(t_k) - alpha(t_{k-1})
        
        Using a standard sequential spending boundary approximation:
            Z_crit(k) = Phi^-1(1 - (alpha(t_k) - alpha(t_{k-1})) / 2)
        """
        t_list = sorted(information_fractions)
        K = len(t_list)
        
        cumulative_alpha = []
        incremental_alpha = []
        z_boundaries = []
        p_boundaries = []

        last_spent = 0.0
        for t in t_list:
            if not (0.0 < t <= 1.0):
                raise ValueError("Information fractions must be strictly in the range (0, 1].")

            spent = self.alpha_spent(t)
            inc = spent - last_spent
            
            # Enforce non-negativity and small floor values
            if inc < 1e-9:
                inc = 1e-9

            cumulative_alpha.append(spent)
            incremental_alpha.append(inc)

            # Two-sided critical boundary approximation
            z_crit = float(norm.ppf(1.0 - inc / 2.0))
            p_crit = float(2.0 * (1.0 - norm.cdf(z_crit)))

            z_boundaries.append(z_crit)
            p_boundaries.append(p_crit)

            last_spent = spent

        return {
            "information_fractions": t_list,
            "cumulative_alpha_spent": cumulative_alpha,
            "incremental_alpha_spent": incremental_alpha,
            "critical_z_boundaries": z_boundaries,
            "critical_p_boundaries": p_boundaries,
        }
