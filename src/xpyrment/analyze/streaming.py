"""Low-latency streaming OLS/Ridge regression via recursive least squares (RLS) and Woodbury updates.

Allows fitting online multivariable OLS/Ridge models over streaming event sequences
without recomputing covariance inverses from scratch.

# TODO: Support dynamic forgetting factors (exponential decay weighting) to allow the streaming model to track non-stationary regimes in high-frequency event streams.
"""

import numpy as np


class StreamingOLS:
    """Streaming Ordinary Least Squares and Ridge Regression.

    Maintains the running coefficients (beta) and the inverse of the covariance matrix
    using the Woodbury matrix identity (Sherman-Morrison formula). Updates take
    O(P^2) time complexity per sample rather than O(P^3) offline inversion time.
    """

    def __init__(self, n_features: int, l2_penalty: float = 1.0, fit_intercept: bool = True) -> None:
        """Initializes the streaming OLS regression model.

        Args:
            n_features (int): Number of independent features (excluding bias).
            l2_penalty (float): Ridge L2 regularization multiplier (lambda). Defaults to 1.0.
            fit_intercept (bool): If True, automatically appends a bias term. Defaults to True.
        """
        self.n_features = n_features
        self.l2_penalty = l2_penalty
        self.fit_intercept = fit_intercept

        # Total model parameters (including bias if fit_intercept is True)
        self.p_dims = n_features + 1 if fit_intercept else n_features

        # Initialize the coefficient vector to zero
        self.beta = np.zeros(self.p_dims)

        # Initialize the inverse covariance matrix: P = (1 / l2_penalty) * I
        # Using a small positive regularization constant prevents singular matrix issues on init
        self.P = (1.0 / max(l2_penalty, 1e-9)) * np.eye(self.p_dims)
        if fit_intercept:
            # Set the intercept's diagonal element to a very large prior variance (1e9)
            # which corresponds to a prior precision (L2 penalty) of 1e-9 (effectively 0.0)
            self.P[0, 0] = 1e9
        self.n_samples = 0

    def _prepare_vector(self, x: np.ndarray) -> np.ndarray:
        """Appends a bias term if fit_intercept is True and flattens to a 1D vector."""
        x = x.ravel()
        if x.shape[0] != self.n_features:
            raise ValueError(
                f"Input feature vector has dimension {x.shape[0]}, expected {self.n_features}."
            )
        if self.fit_intercept:
            return np.hstack([1.0, x])
        return x

    def update(self, x: np.ndarray, y: float) -> None:
        """Performs a single-sample recursive least squares update.

        Updates the running coefficient vector (beta) and covariance inverse (P) in O(P^2) time.

        Args:
            x (np.ndarray): 1D array of features of shape (n_features,).
            y (float): Numeric target value.
        """
        x_vec = self._prepare_vector(x)
        self.n_samples += 1

        # Compute intermediate gain components: P * x
        Px = np.dot(self.P, x_vec)

        # Denominator scalar: 1 + x^T * P * x
        denom = 1.0 + np.dot(x_vec, Px)

        # Update gain vector g
        gain = Px / denom

        # Update the coefficients: beta_new = beta_old + gain * (y - x^T * beta_old)
        error = y - np.dot(x_vec, self.beta)
        self.beta += gain * error

        # Update covariance inverse via Sherman-Morrison: P = P - gain * (x^T * P)
        self.P -= np.outer(gain, Px)

    def update_batch(self, X: np.ndarray, y: np.ndarray) -> None:
        """Updates the running model with a batch of observations.

        Loops through each row to perform recursive rank-1 updates.

        Args:
            X (np.ndarray): Feature matrix of shape (M, n_features).
            y (np.ndarray): Target outcomes of shape (M,).
        """
        X = np.atleast_2d(X)
        y = y.ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("Input features and targets must have matching sample dimensions.")

        for i in range(X.shape[0]):
            self.update(X[i], float(y[i]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target value using the active coefficient weights.

        Args:
            X (np.ndarray): Feature matrix of shape (M, n_features).

        Returns:
            np.ndarray: Vector of predictions of shape (M,).
        """
        X = np.atleast_2d(X)
        if self.fit_intercept:
            X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
            return np.dot(X_bias, self.beta)
        return np.dot(X, self.beta)

    @property
    def coefficients(self) -> np.ndarray:
        """Returns the current estimated regression coefficients."""
        return self.beta
