"""Rolling Synthetic Controls under Structural Breaks (Block 40).

Fits rolling-horizon synthetic controls with L1/L2 simplex constrained optimization
to adaptively trace treated trajectories in the presence of policy structural breaks.
"""

import numpy as np
from scipy.optimize import minimize


class RollingSyntheticControl:
    """Estimates rolling window synthetic controls with L1-L2 penalized simplex weights.

    # TODO: Add interactive covariate balance weight constraints (V-matrix optimizations) within the rolling SLSQP loss functions.
    # TODO: Implement out-of-fold temporal cross-validation to select rolling window size H and regularization hyper-parameters (lambda_l1, lambda_l2) dynamically.
    """

    def __init__(self, lambda_l1: float = 0.01, lambda_l2: float = 0.01) -> None:
        """Initializes the rolling synthetic control estimator.

        Args:
            lambda_l1 (float): L1 regularization parameter (forces sparse donor selection).
            lambda_l2 (float): L2 regularization parameter (stabilizes collinear donors).
        """
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.weights_: np.ndarray = np.array([])

    def fit(self, treated_outcome: np.ndarray, donor_matrix: np.ndarray) -> "RollingSyntheticControl":
        """Fits optimal simplex weights by minimizing tracking error over the window.

        Objective:
            min_w || y_tr - X_co * w ||_2^2 + l1 * sum(|w|) + l2 * sum(w^2)
            s.t. sum(w) = 1 and w >= 0
        """
        y = treated_outcome.astype(float)
        X = donor_matrix.astype(float)

        T, D = X.shape
        if len(y) != T:
            raise ValueError("Length of treated outcome must match the rows in donor matrix.")

        # Objective function to minimize
        def loss_fn(w):
            residuals = y - np.dot(X, w)
            error_term = np.sum(residuals ** 2)
            l1_penalty = self.lambda_l1 * np.sum(np.abs(w))
            l2_penalty = self.lambda_l2 * np.sum(w ** 2)
            return error_term + l1_penalty + l2_penalty

        # Simplex constraints: weights sum to 1, non-negative bounds
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0) for _ in range(D)]
        
        # Initial guess: uniform weights
        w0 = np.full(D, 1.0 / D)

        res = minimize(
            loss_fn,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-8, "maxiter": 150},
        )

        self.weights_ = res.x
        return self

    def predict(self, donor_matrix: np.ndarray) -> np.ndarray:
        """Reconstructs the synthetic counterfactual timeline using fitted weights."""
        if len(self.weights_) == 0:
            raise ValueError("Model must be fitted before calling predict.")
        return np.dot(donor_matrix.astype(float), self.weights_)
