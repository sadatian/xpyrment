"""Structural Nested Mean Models (SNMM) and Sequential G-Estimation (Block 22).

Estimates time-varying sequential causal treatment effects across multiple stages
under time-varying confounding using Robins' backward-induction sequential g-estimation.
"""

from typing import List
import numpy as np


class StructuralNestedMeanModel:
    """Structural Nested Mean Model (SNMM) for multi-stage causal inference.

    Solves Robins' sequential g-estimation closed-form equations backwards from the
    terminal stage to estimate unbiased stage-specific and interaction causal effects.

    # TODO: Implement doubly robust sequential g-estimation incorporating stage-specific baseline outcome models.
    # TODO: Add wild bootstrap inference over sequential stages to compute joint confidence intervals for beta vectors.
    """

    def __init__(self, l2_penalty: float = 1e-6) -> None:
        """Initializes the SNMM estimator.

        Args:
            l2_penalty (float): Regularization parameter to ensure stability in matrix inversion.
                Defaults to 1e-6.
        """
        self.l2_penalty = l2_penalty
        self.stage_coefficients_: List[np.ndarray] = []

    def fit(self, X_list: List[np.ndarray], A_list: List[np.ndarray], y: np.ndarray) -> "StructuralNestedMeanModel":
        """Performs multi-stage backward-induction sequential g-estimation.

        Args:
            X_list (List[np.ndarray]): List of covariate matrices of length S (number of stages),
                where each matrix is of shape (N, P_s).
            A_list (List[np.ndarray]): List of treatment vectors of length S,
                where each vector is of shape (N,) containing binary assignments.
            y (np.ndarray): Numeric final outcome vector of shape (N,).

        Returns:
            StructuralNestedMeanModel: Fitted estimator.
        """
        N = y.shape[0]
        S = len(X_list)
        if len(A_list) != S:
            raise ValueError("X_list and A_list must have the same number of stages.")

        y_curr = y.astype(float).copy()
        self.stage_coefficients_ = [np.array([])] * S

        # Perform g-estimation backward from stage S-1 down to 0
        for s in reversed(range(S)):
            Xs = X_list[s]
            As = A_list[s].ravel()

            # 1. Fit Propensity model: E[As | Xs] using linear regression with bias
            Xs_bias = np.hstack([np.ones((N, 1)), Xs])
            XTX = np.dot(Xs_bias.T, Xs_bias)
            XTX_reg = XTX + self.l2_penalty * np.eye(Xs_bias.shape[1])
            # Don't regularize intercept
            XTX_reg[0, 0] = XTX[0, 0]
            
            beta_prop = np.linalg.solve(XTX_reg, np.dot(Xs_bias.T, As))
            A_pred = np.dot(Xs_bias, beta_prop)

            # Treatment residuals (instrument)
            A_res = As - A_pred

            # 2. Build instrument weight matrix W and design matrix Z for g-estimation
            # Structural model for stage s: psi_s(As, Xs) = As * [1, Xs] * beta_s
            Zs = np.hstack([np.ones((N, 1)), Xs])  # (N, P_s + 1)
            
            # W_i = (As_i - A_pred_i) * Zs_i
            W = Zs * A_res.reshape(-1, 1)  # (N, P_s + 1)

            # Left-hand side of equation: sum_i W_i * As_i * Zs_i^T
            LHS = np.zeros((Zs.shape[1], Zs.shape[1]))
            for i in range(N):
                LHS += np.outer(W[i], As[i] * Zs[i])

            # Right-hand side of equation: sum_i y_curr_i * W_i
            RHS = np.dot(W.T, y_curr)

            # Solve LHS * beta_s = RHS with L2 stabilization
            LHS_reg = LHS + self.l2_penalty * np.eye(Zs.shape[1])
            beta_s = np.linalg.solve(LHS_reg, RHS)
            self.stage_coefficients_[s] = beta_s

            # 3. Blip down current outcome: subtract estimated causal effect
            # y_curr_prev = y_curr - As * Zs * beta_s
            effect_s = As * np.dot(Zs, beta_s)
            y_curr -= effect_s

        return self

    def predict_blip_outcome(self, X_list: List[np.ndarray], A_list: List[np.ndarray], y: np.ndarray, stage: int) -> np.ndarray:
        """Returns the outcome variable 'blipped down' (de-causalized) up to the specified stage index.

        Args:
            X_list (List[np.ndarray]): Covariate matrices.
            A_list (List[np.ndarray]): Treatment vectors.
            y (np.ndarray): Original outcome vector.
            stage (int): Stage index (0-indexed). All treatment effects from this stage onwards are removed.

        Returns:
            np.ndarray: Blipped-down outcome vector.
        """
        y_blipped = y.astype(float).copy()
        S = len(self.stage_coefficients_)
        
        for s in range(stage, S):
            beta_s = self.stage_coefficients_[s]
            Xs = X_list[s]
            As = A_list[s].ravel()
            Zs = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
            effect_s = As * np.dot(Zs, beta_s)
            y_blipped -= effect_s
            
        return y_blipped

    @property
    def coefficients(self) -> List[np.ndarray]:
        """Returns the list of estimated causal parameter vectors for each stage."""
        return self.stage_coefficients_
