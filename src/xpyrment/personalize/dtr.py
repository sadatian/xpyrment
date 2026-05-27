"""Dynamic Treatment Regimes (DTR) & Q-Learning (Block 31).

Estimates personalized multi-stage sequential treatment regimes using backward-induction
Q-learning to optimize long-term causal outcomes.
"""

from typing import Tuple
import numpy as np


class QFactorModel:
    """Represents a single-stage Q-factor linear regression model:

        Q(H, A) = beta^T * [1, H, A, H * A]

    # TODO: Support non-linear Q-functions using basis expansion (e.g. B-splines) for continuous history states.
    # TODO: Implement doubly robust Q-learning corrections to protect against Q-model specification biases.
    """

    def __init__(self, l2_penalty: float = 1e-4) -> None:
        self.l2_penalty = l2_penalty
        self.beta_: np.ndarray = np.array([])

    def fit(self, H: np.ndarray, A: np.ndarray, Y: np.ndarray) -> "QFactorModel":
        """Fits the Q-factor parameters using ridge regression.

        Args:
            H (np.ndarray): History/covariates of shape (N, P).
            A (np.ndarray): Binary action vector of shape (N,) (0 or 1).
            Y (np.ndarray): Target outcome/payoff of shape (N,).
        """
        N, P = H.shape
        # Construct interaction feature column: H * A
        A_col = A.reshape(-1, 1)
        H_interaction = H * A_col

        # Design matrix X: [1, H, A, H_interaction]
        X = np.hstack([np.ones((N, 1)), H, A_col, H_interaction])
        K = X.shape[1]

        # Solve ridge OLS: beta = (X^T X + L2 * I)^-1 X^T Y
        XTX = np.dot(X.T, X)
        XTX_reg = XTX + self.l2_penalty * np.eye(K)
        XTX_reg[0, 0] = XTX[0, 0]  # No penalty on intercept

        self.beta_ = np.linalg.solve(XTX_reg, np.dot(X.T, Y))
        return self

    def predict(self, H: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Predicts Q-values for a given history and action."""
        N = len(H)
        A_col = A.reshape(-1, 1)
        H_interaction = H * A_col
        X = np.hstack([np.ones((N, 1)), H, A_col, H_interaction])
        return np.dot(X, self.beta_)


class DynamicTreatmentRegime:
    """Estimates sequential two-stage dynamic treatment policies via Q-learning."""

    def __init__(self, l2_penalty: float = 1e-4) -> None:
        self.l2_penalty = l2_penalty
        self.q_model_1_ = QFactorModel(l2_penalty)
        self.q_model_2_ = QFactorModel(l2_penalty)

    def fit(
        self,
        H1: np.ndarray,
        A1: np.ndarray,
        H2: np.ndarray,
        A2: np.ndarray,
        Y: np.ndarray,
    ) -> "DynamicTreatmentRegime":
        """Fits the two-stage sequential Q-learning model using backward induction.

        Args:
            H1 (np.ndarray): Stage 1 baseline history of shape (N, P1).
            A1 (np.ndarray): Stage 1 binary action of shape (N,).
            H2 (np.ndarray): Stage 2 intermediate history of shape (N, P2).
            A2 (np.ndarray): Stage 2 binary action of shape (N,).
            Y (np.ndarray): Final continuous outcome of shape (N,).
        """
        # --- Stage 2 Induction ---
        self.q_model_2_.fit(H2, A2, Y)

        # Compute optimal counterfactual target for Stage 1: Y_tilde = max_a2 Q2(H2, a2)
        N = len(Y)
        Q2_opt_0 = self.q_model_2_.predict(H2, np.zeros(N))
        Q2_opt_1 = self.q_model_2_.predict(H2, np.ones(N))
        Y_tilde = np.maximum(Q2_opt_0, Q2_opt_1)

        # --- Stage 1 Induction ---
        self.q_model_1_.fit(H1, A1, Y_tilde)
        return self

    def recommend(self, H1: np.ndarray, H2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Recommends optimal stage-specific policies based on history variables.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Stage 1 optimal actions, Stage 2 optimal actions.
        """
        N = len(H1)
        
        # Stage 1 Recommendations
        Q1_0 = self.q_model_1_.predict(H1, np.zeros(N))
        Q1_1 = self.q_model_1_.predict(H1, np.ones(N))
        rec_1 = (Q1_1 > Q1_0).astype(float)

        # Stage 2 Recommendations
        Q2_0 = self.q_model_2_.predict(H2, np.zeros(N))
        Q2_1 = self.q_model_2_.predict(H2, np.ones(N))
        rec_2 = (Q2_1 > Q2_0).astype(float)

        return rec_1, rec_2
