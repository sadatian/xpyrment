"""Off-Policy Evaluation (OPE) for Contextual Bandits (Block 23).

Estimates counterfactual expected rewards of a target policy using logs of historical
interaction sequences without deploying the policy online. Implements IPS, SN-IPS, and DR.
"""

from typing import Callable, Dict, Union
import numpy as np


class OffPolicyEvaluator:
    """Computes counterfactual expected rewards of target policy using historical bandit data.

    Implements:
    - Inverse Propensity Scoring (IPS)
    - Self-Normalized IPS (SN-IPS)
    - Doubly Robust (DR) estimation

    # TODO: Implement Marginalized Importance Sampling (MIS) for sequential multi-step decision processes.
    # TODO: Integrate asymptotic confidence interval estimators based on empirical Bernstein inequalities.
    """

    def __init__(self, target_policy: Callable[[np.ndarray], np.ndarray], l2_penalty: float = 1.0) -> None:
        """Initializes the off-policy evaluator.

        Args:
            target_policy (Callable): A function that takes context matrix X of shape (N, P)
                and returns action probability matrix of shape (N, K) where K is number of arms.
            l2_penalty (float): Regularization penalty for internal reward regression models.
                Defaults to 1.0.
        """
        self.target_policy = target_policy
        self.l2_penalty = l2_penalty

    def evaluate(
        self,
        X: np.ndarray,
        actions: np.ndarray,
        propensities: np.ndarray,
        rewards: np.ndarray,
    ) -> Dict[str, float]:
        """Runs the OPE estimators.

        Args:
            X (np.ndarray): Context matrix of shape (N, P).
            actions (np.ndarray): Actual actions selected in history of shape (N,).
            propensities (np.ndarray): Propensity of selected action in history, shape (N,).
            rewards (np.ndarray): Observed rewards, shape (N,).

        Returns:
            Dict[str, float]: Dictionary containing IPS, SN-IPS, and Doubly Robust expected rewards.
        """
        N = X.shape[0]
        actions = actions.ravel()
        propensities = propensities.ravel()
        rewards = rewards.ravel()

        unique_arms = np.unique(actions)
        K = len(unique_arms)

        # 1. Obtain action probability matrix from target policy
        # target_probs shape: (N, K)
        target_probs = self.target_policy(X)

        # Map unique arm labels to consecutive integers 0..K-1
        arm_to_idx = {arm: idx for idx, arm in enumerate(unique_arms)}
        
        # 2. Compute IPS weights: target_prob_for_selected_action / historical_propensity
        weights = np.zeros(N)
        for i in range(N):
            selected_arm = actions[i]
            if selected_arm in arm_to_idx:
                arm_idx = arm_to_idx[selected_arm]
                weights[i] = target_probs[i, arm_idx] / max(propensities[i], 1e-12)

        # Inverse Propensity Scoring (IPS) estimate
        ips_estimate = float(np.mean(weights * rewards))

        # Self-Normalized IPS (SN-IPS) estimate
        sum_weights = np.sum(weights)
        if sum_weights > 1e-12:
            sn_ips_estimate = float(np.sum(weights * rewards) / sum_weights)
        else:
            sn_ips_estimate = 0.0

        # 3. Doubly Robust (DR) Estimation
        # Train a Ridge regression model for each arm: Reward ~ m_k(X)
        models_coef = {}
        models_intercept = {}

        X_bias = np.hstack([np.ones((N, 1)), X])
        P = X_bias.shape[1]

        # Fit model per arm
        for arm in unique_arms:
            mask = (actions == arm)
            X_arm = X_bias[mask]
            y_arm = rewards[mask]

            if len(y_arm) > 2:
                # Solve Ridge: beta = (X^T X + lambda * I)^-1 X^T y
                XTX = np.dot(X_arm.T, X_arm)
                XTX_reg = XTX + self.l2_penalty * np.eye(P)
                # Don't regularize intercept
                XTX_reg[0, 0] = XTX[0, 0]
                
                beta = np.linalg.solve(XTX_reg, np.dot(X_arm.T, y_arm))
                models_intercept[arm] = beta[0]
                models_coef[arm] = beta[1:]
            else:
                # Fallback to simple mean reward if too few samples
                models_intercept[arm] = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
                models_coef[arm] = np.zeros(X.shape[1])

        # Compute mu_hat(X_i, a) for all actions and contexts
        # mu_hat shape: (N, K)
        mu_hat = np.zeros((N, K))
        for idx, arm in enumerate(unique_arms):
            coef = models_coef[arm]
            intercept = models_intercept[arm]
            mu_hat[:, idx] = np.dot(X, coef) + intercept

        # DR value = Mean_i ( sum_k target_prob(k) * mu_hat(X_i, k) + I(A_i = k) * weights_i * (rewards_i - mu_hat(X_i, k)) )
        dr_values = np.zeros(N)
        for i in range(N):
            # Term A: Expected reward of target policy under the regression model
            expected_model_reward = np.dot(target_probs[i], mu_hat[i])
            
            # Term B: IPW residual correction term
            selected_arm = actions[i]
            if selected_arm in arm_to_idx:
                arm_idx = arm_to_idx[selected_arm]
                residual = rewards[i] - mu_hat[i, arm_idx]
                correction = weights[i] * residual
            else:
                correction = 0.0

            dr_values[i] = expected_model_reward + correction

        dr_estimate = float(np.mean(dr_values))

        return {
            "ips": ips_estimate,
            "sn_ips": sn_ips_estimate,
            "dr": dr_estimate,
        }
