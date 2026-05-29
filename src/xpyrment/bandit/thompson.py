"""Thompson Sampling multi-armed bandit algorithms using conjugate models.

This module provides the `ThompsonSamplingBandit` class supporting both
Beta-Binomial (binary rewards) and Normal-Normal (continuous rewards) posteriors.
"""

from typing import List, Optional, Literal, Dict
import numpy as np


class ThompsonSamplingBandit:
    """Implements Thompson Sampling with support for Beta-Binomial and Normal-Normal models.

    # TODO: Support Normal-Inverse-Gamma conjugate priors for continuous rewards with unknown variance.
    # TODO: Implement Dirichlet-Multinomial Thompson Sampling to support categorical/multinomial feedback.
    # TODO: Implement batched/delayed reward Thompson Sampling updates using Gaussian Process models to handle settings where feedback is slow or clustered.
    """

    def __init__(
        self,
        arms: List[str],
        reward_type: Literal["binary", "continuous"] = "binary",
        prior_params: Optional[Dict[str, tuple]] = None,
    ) -> None:
        """Initializes the Thompson Sampling bandit.

        Args:
            arms (List[str]): List of variant names.
            reward_type (str): Type of reward, either "binary" or "continuous". Defaults to "binary".
            prior_params (Optional[Dict[str, tuple]]): Initial prior hyperparameters for each arm.
                - For "binary": tuple of (alpha, beta). Defaults to (1.0, 1.0) uniform priors.
                - For "continuous": tuple of (mu_0, precision_0), assuming known unit variance (sigma^2 = 1.0).
                  Defaults to (0.0, 1.0).
        """
        self.arms = arms
        self.reward_type = reward_type
        self.counts = {arm: 0 for arm in arms}
        self.total_steps = 0

        if prior_params is None:
            if reward_type == "binary":
                # Beta(alpha=1, beta=1)
                self.params = {arm: [1.0, 1.0] for arm in arms}
            else:
                # Normal(mu_0=0.0, precision_0=1.0)
                self.params = {arm: [0.0, 1.0] for arm in arms}
        else:
            self.params = {arm: list(prior_params[arm]) for arm in arms}

    def select_arm(self, rng: Optional[np.random.Generator] = None) -> str:
        """Draws posterior samples and selects the arm with the highest sample value.

        Returns:
            str: The selected arm/variant name.
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.reward_type == "binary":
            alphas = np.array([self.params[arm][0] for arm in self.arms])
            betas = np.array([self.params[arm][1] for arm in self.arms])
            samples = rng.beta(alphas, betas)
        else:
            mus = np.array([self.params[arm][0] for arm in self.arms])
            precs = np.array([self.params[arm][1] for arm in self.arms])
            sigmas = 1.0 / np.sqrt(precs)
            samples = rng.normal(mus, sigmas)

        best_idx = np.argmax(samples)
        return self.arms[best_idx]

    def update(self, arm: str, reward: float):
        """Performs a closed-form conjugate posterior parameter update.

        Args:
            arm (str): Name of the arm being updated.
            reward (float): Observed reward scalar.
        """
        if arm not in self.counts:
            raise ValueError(f"Arm '{arm}' is not registered in this bandit.")

        self.total_steps += 1
        self.counts[arm] += 1

        if self.reward_type == "binary":
            # Binary reward: reward must be 0 or 1
            if reward not in [0.0, 1.0, 0, 1]:
                raise ValueError("Binary Thompson Sampling requires rewards in {0, 1}.")
            # Beta update rules
            self.params[arm][0] += reward       # alpha_new = alpha + success
            self.params[arm][1] += (1.0 - reward) # beta_new = beta + failure
        else:
            # Continuous reward: Normal-Normal update with known variance (sigma^2 = 1.0)
            mu_old, prec_old = self.params[arm]
            
            # Update rules:
            # precision_new = precision_old + 1.0
            # mu_new = (precision_old * mu_old + reward) / precision_new
            prec_new = prec_old + 1.0
            mu_new = (prec_old * mu_old + reward) / prec_new
            
            self.params[arm][0] = mu_new
            self.params[arm][1] = prec_new
