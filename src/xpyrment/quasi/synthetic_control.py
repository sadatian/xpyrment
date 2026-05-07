"""Synthetic Controls for quasi-experimental unit synthesis.

This module provides the `SyntheticControl` class to synthesize virtual control units
from donor pools when randomized splits are impossible.
"""

import numpy as np
from scipy.optimize import minimize


class SyntheticControl:
    """Synthesizes a virtual control unit from a weighted combination of unexposed donor pools."""

    def __init__(self):
        """Initializes the SyntheticControl estimator."""
        self.weights = None
        self.synthetic_outcome = None
        self.treatment_effect = None

    def fit(self, y_treated_pre: np.ndarray, y_donor_pre: np.ndarray):
        """Optimizes donor weights minimizing pre-period mean squared error.

        Weights are constrained to sum to 1.0 (no scaling bias) and lie in [0, 1] (no extrapolation).

        Args:
            y_treated_pre (np.ndarray): Outcome history of the treated unit (shape: T_pre,).
            y_donor_pre (np.ndarray): Outcome history of donor pool units (shape: T_pre, n_donors).
        """
        n_donors = y_donor_pre.shape[1]

        # Loss function: sum of squared prediction differences
        def loss_func(w):
            pred = np.dot(y_donor_pre, w)
            return np.sum((y_treated_pre - pred) ** 2)

        # Equality constraint: weights sum to 1.0
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        # Convex bounds: each weight must reside within [0.0, 1.0]
        bounds = [(0.0, 1.0) for _ in range(n_donors)]
        # Uniform initial weights guess
        w0 = np.ones(n_donors) / n_donors

        res = minimize(loss_func, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        self.weights = res.x
        return self

    def estimate_effect(self, y_treated_post: np.ndarray, y_donor_post: np.ndarray) -> np.ndarray:
        """Estimates the treatment effect path across post-treatment periods.

        Args:
            y_treated_post (np.ndarray): Post-period outcome values of the treated unit (shape: T_post,).
            y_donor_post (np.ndarray): Post-period outcome values of donor pool units (shape: T_post, n_donors).

        Returns:
            np.ndarray: Vector of treatment effects over time (treated_post - synthetic_post).
        """
        self.synthetic_outcome = np.dot(y_donor_post, self.weights)
        self.treatment_effect = y_treated_post - self.synthetic_outcome
        return self.treatment_effect
