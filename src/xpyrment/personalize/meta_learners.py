"""Meta-learners for estimating Heterogeneous Treatment Effects (HTE).

This module provides standard meta-learning structures, including S-Learner, T-Learner,
and X-Learner, using analytical multi-variable Ridge regression as base models.
"""

from typing import Optional
import numpy as np


class RidgeRegressor:
    """Analytical Ridge regression estimator for high-performance linear model fitting."""

    def __init__(self, alpha: float = 1.0):
        """Initializes the RidgeRegressor.

        Args:
            alpha (float): Ridge regularization L2 parameter. Defaults to 1.0.
        """
        self.alpha = alpha
        self.beta = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the regression model analytically using closed-form normal equations.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).
        """
        n_samples = X.shape[0]
        # Append bias vector of ones
        X_bias = np.hstack([np.ones((n_samples, 1)), X])
        p_params = X_bias.shape[1]

        # Formulate and solve normal equations: (X^T X + alpha * I) beta = X^T y
        XTX = np.dot(X_bias.T, X_bias)
        I = np.eye(p_params)
        I[0, 0] = 0.0  # Do not regularize the intercept term

        self.beta = np.linalg.solve(XTX + self.alpha * I, np.dot(X_bias.T, y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target values for the given features.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted vector.
        """
        n_samples = X.shape[0]
        X_bias = np.hstack([np.ones((n_samples, 1)), X])
        return np.dot(X_bias, self.beta)


class SLearner:
    """Single-model meta-learner for Conditional Average Treatment Effect (CATE) estimation."""

    def __init__(self, alpha: float = 1.0):
        """Initializes the S-Learner.

        Args:
            alpha (float): Regularization parameter for the base Ridge model. Defaults to 1.0.
        """
        self.model = RidgeRegressor(alpha=alpha)

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray):
        """Fits the single model on the joint features, treatment, and interaction terms [X, T, X * T].

        Args:
            X (np.ndarray): Covariates matrix.
            treatment (np.ndarray): Binary treatment assignment vector.
            y (np.ndarray): Numeric outcome vector.
        """
        # Formulate explicit treatment-covariate interactions: X * T
        X_inter = X * treatment.reshape(-1, 1)
        X_comb = np.hstack([X, treatment.reshape(-1, 1), X_inter])
        self.model.fit(X_comb, y)
        return self

    def estimate_effect(self, X: np.ndarray) -> np.ndarray:
        """Estimates individual-level causal treatment effects (CATE).

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Vector of estimated treatment effects.
        """
        # Counterfactual treatment: T = 1, Interaction = X * 1 = X
        X_treat = np.hstack([X, np.ones((X.shape[0], 1)), X])
        # Counterfactual control: T = 0, Interaction = X * 0 = 0
        X_control = np.hstack([X, np.zeros((X.shape[0], 1)), np.zeros_like(X)])

        return self.model.predict(X_treat) - self.model.predict(X_control)


class TLearner:
    """Two-model meta-learner for Conditional Average Treatment Effect (CATE) estimation."""

    def __init__(self, alpha: float = 1.0):
        """Initializes the T-Learner.

        Args:
            alpha (float): Regularization parameter for the two Ridge models. Defaults to 1.0.
        """
        self.model_0 = RidgeRegressor(alpha=alpha)
        self.model_1 = RidgeRegressor(alpha=alpha)

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray):
        """Fits separate estimators for control and treatment observations.

        Args:
            X (np.ndarray): Covariates matrix.
            treatment (np.ndarray): Binary treatment assignment vector.
            y (np.ndarray): Numeric outcome vector.
        """
        X_0 = X[treatment == 0]
        y_0 = y[treatment == 0]
        X_1 = X[treatment == 1]
        y_1 = y[treatment == 1]

        self.model_0.fit(X_0, y_0)
        self.model_1.fit(X_1, y_1)
        return self

    def estimate_effect(self, X: np.ndarray) -> np.ndarray:
        """Estimates individual-level causal treatment effects (CATE).

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Vector of estimated treatment effects.
        """
        return self.model_1.predict(X) - self.model_0.predict(X)


class XLearner:
    """Cross-model meta-learner for unbalanced Conditional Average Treatment Effect (CATE) estimation.

    # TODO: Support arbitrary custom base-learner regression estimators instead of strictly RidgeRegressor.
    # TODO: Implement a logistic regression or other custom propensity score model rather than a constant mean scalar.
    """

    def __init__(self, alpha: float = 1.0):
        """Initializes the X-Learner.

        Args:
            alpha (float): Regularization parameter for the internal models. Defaults to 1.0.
        """
        self.mu_0 = RidgeRegressor(alpha=alpha)
        self.mu_1 = RidgeRegressor(alpha=alpha)
        self.tau_0 = RidgeRegressor(alpha=alpha)
        self.tau_1 = RidgeRegressor(alpha=alpha)
        self.propensity_score = 0.5

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray):
        """Executes the 4-stage X-Learner optimization algorithm on unbalanced assignments.

        Args:
            X (np.ndarray): Covariates matrix.
            treatment (np.ndarray): Binary treatment assignment vector.
            y (np.ndarray): Numeric outcome vector.
        """
        X_0 = X[treatment == 0]
        y_0 = y[treatment == 0]
        X_1 = X[treatment == 1]
        y_1 = y[treatment == 1]

        # Stage 1: Fit base models on control and treatment groups
        self.mu_0.fit(X_0, y_0)
        self.mu_1.fit(X_1, y_1)

        # Stage 2: Compute imputed counterfactual treatment effect residuals
        D_1 = y_1 - self.mu_0.predict(X_1)
        D_0 = self.mu_1.predict(X_0) - y_0

        # Stage 3: Fit secondary residual estimators
        self.tau_1.fit(X_1, D_1)
        self.tau_0.fit(X_0, D_0)

        # Stage 4: Record propensity score (ratio of treated units)
        self.propensity_score = float(np.mean(treatment))
        return self

    def estimate_effect(self, X: np.ndarray) -> np.ndarray:
        """Estimates individual-level causal treatment effects (CATE).

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Propensity-weighted treatment effects vector.
        """
        p = self.propensity_score
        pred_0 = self.tau_0.predict(X)
        pred_1 = self.tau_1.predict(X)
        return p * pred_0 + (1.0 - p) * pred_1
