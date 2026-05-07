"""Neighborhood Exposure Models for network spillover effect estimation.

This module provides the `NeighborhoodExposure` class to classify network nodes into
distinct exposure states and compute Direct and Indirect treatment effects.
"""

from typing import Dict, Any
import numpy as np


class NeighborhoodExposure:
    """Estimates direct treatment and indirect network spillover effects.

    # TODO: Support fractional neighborhood exposure thresholds (e.g., classifying nodes as exposed only when >20% of their neighbors are treated).
    # TODO: Implement bootstrap or Horvitz-Thompson variance solvers to provide standard errors and p-values for DTE and ISE.
    # TODO: Support multi-hop network exposures (e.g., 2-hop exposures where a node is influenced by friends-of-friends) to capture deeper peer cascades.
    """

    def __init__(self, adjacency: np.ndarray):
        """Initializes the NeighborhoodExposure estimator.

        Args:
            adjacency (np.ndarray): Symmetric adjacency matrix representing connections.
        """
        self.adjacency = adjacency

    def classify_exposure(self, treatment: np.ndarray) -> np.ndarray:
        """Classifies each node's network exposure state based on treatment assignments of neighbors.

        Exposure States:
            0: Pure Control (T=0, 0 treated neighbors)
            1: Control Spillover / Leakage (T=0, >0 treated neighbors)
            2: Isolated Treatment (T=1, 0 treated neighbors)
            3: Treated Exposure (T=1, >0 treated neighbors)

        Args:
            treatment (np.ndarray): Node-level binary treatment assignment vector.

        Returns:
            np.ndarray: Vector of integer exposure states.
        """
        n_nodes = self.adjacency.shape[0]
        states = np.zeros(n_nodes, dtype=int)

        for i in range(n_nodes):
            # Find indices of neighbors connected to node i
            neighbors = np.where(self.adjacency[i] > 0)[0]
            n_treated_neighbors = np.sum(treatment[neighbors])

            if treatment[i] == 0:
                if n_treated_neighbors == 0:
                    states[i] = 0  # Pure Control
                else:
                    states[i] = 1  # Control Spillover
            else:
                if n_treated_neighbors == 0:
                    states[i] = 2  # Isolated Treatment
                else:
                    states[i] = 3  # Treated Exposure

        return states

    def estimate_effects(self, y: np.ndarray, treatment: np.ndarray) -> Dict[str, Any]:
        """Classifies exposure states and estimates Direct and Indirect spillover treatment effects.

        Args:
            y (np.ndarray): Numeric outcome vector.
            treatment (np.ndarray): Binary treatment assignment vector.

        Returns:
            Dict[str, Any]: Mapping of state statistics and estimated DTE and ISE effects.
        """
        states = self.classify_exposure(treatment)

        # Calculate state average outcomes
        mean_0 = np.mean(y[states == 0]) if np.any(states == 0) else np.nan
        mean_1 = np.mean(y[states == 1]) if np.any(states == 1) else np.nan
        mean_2 = np.mean(y[states == 2]) if np.any(states == 2) else np.nan
        mean_3 = np.mean(y[states == 3]) if np.any(states == 3) else np.nan

        # Direct Treatment Effect (DTE): Treated Exposure vs. Pure Control
        dte = mean_3 - mean_0 if not np.isnan(mean_3) and not np.isnan(mean_0) else np.nan
        # Indirect Spillover Effect (ISE): Control Spillover vs. Pure Control (leakage measure)
        ise = mean_1 - mean_0 if not np.isnan(mean_1) and not np.isnan(mean_0) else np.nan

        return {
            "exposure_states": states,
            "mean_pure_control": mean_0,
            "mean_control_spillover": mean_1,
            "mean_isolated_treatment": mean_2,
            "mean_treated_exposure": mean_3,
            "direct_treatment_effect": dte,
            "indirect_spillover_effect": ise,
        }
