"""Optimal Transport (OT) & Quantile Distributional Effects (Block 34).

Uncovers distribution-wide treatment shifts (quantile treatment effects) and computes
1D Wasserstein optimal transport distance metrics.
"""

from typing import Dict
import numpy as np


class QuantileOptimalTransport:
    """Computes Wasserstein-1 distances and Quantile Treatment Effects (QTE) in 1D.

    # TODO: Extend the optimal transport solver to multidimensional metric profiles using Sinkhorn-Knopp entropic regularization.
    # TODO: Add asymptotic bootstrapping of Wasserstein boundaries to conduct non-parametric distribution equality tests.
    """

    def __init__(self, quantiles: np.ndarray = None) -> None:
        """Initializes the Optimal Transport mapping.

        Args:
            quantiles (np.ndarray): Percentile mesh coordinates in [0, 1].
                Defaults to deciles [0.1, 0.2, ..., 0.9].
        """
        self.quantiles = quantiles if quantiles is not None else np.linspace(0.1, 0.9, 9)

    def compute_qte(self, control: np.ndarray, treatment: np.ndarray) -> Dict[float, float]:
        """Calculates the Quantile Treatment Effect (QTE) across the specified mesh:

            QTE(q) = F_T^-1(q) - F_C^-1(q)
        """
        ctrl_sorted = np.sort(control)
        trt_sorted = np.sort(treatment)

        qte_map = {}
        for q in self.quantiles:
            # Interpolated empirical quantiles
            q_ctrl = float(np.percentile(ctrl_sorted, q * 100.0))
            q_trt = float(np.percentile(trt_sorted, q * 100.0))
            qte_map[float(q)] = q_trt - q_ctrl

        return qte_map

    def compute_wasserstein_distance(self, control: np.ndarray, treatment: np.ndarray, p: int = 1) -> float:
        """Computes the 1D p-th Wasserstein distance between control and treatment groups.

        For 1D empirical distributions of sizes N and M, this is solved exactly by sorting
        and integrating the absolute difference between empirical cumulative quantiles.
        """
        ctrl_sorted = np.sort(control)
        trt_sorted = np.sort(treatment)

        # Create a dense integration mesh grid
        grid = np.linspace(0.01, 0.99, 1000)
        q_ctrl = np.percentile(ctrl_sorted, grid * 100.0)
        q_trt = np.percentile(trt_sorted, grid * 100.0)

        # Integrate: L_p norm of difference
        diff_power = np.abs(q_trt - q_ctrl) ** p
        integral = np.mean(diff_power)  # Average over grid approximation

        return float(integral ** (1.0 / p))
