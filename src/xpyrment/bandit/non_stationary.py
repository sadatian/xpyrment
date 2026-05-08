"""Non-Stationary Bandits: Discounted and Sliding-Window Thompson Sampling (Block 27).

Adapts multi-armed bandit variant allocations to environments with rapidly drifting
or seasonal reward baselines.
"""

from typing import Dict, List, Optional
import numpy as np


class DiscountedThompsonSamplingBandit:
    """Discounted Thompson Sampling Bandit (D-TS) for Bernoulli/Binary rewards.

    Applies a discount factor gamma in (0, 1] at each environment update step to decay the
    influence of older successes and failures, prioritizing recent interactions:
        alpha_a = max(gamma * alpha_a + Y_t, 1.0)
        beta_a = max(gamma * beta_a + (1 - Y_t), 1.0)
    
    # TODO: Implement a dynamic tracking update for the discount factor gamma using meta-learning (e.g., bandit-over-bandits).
    # TODO: Extend the reward model to support non-stationary continuous conjugate priors (Normal-Inverse-Gamma for linear payoffs).
    """

    def __init__(self, arms: List[str], gamma: float = 0.95) -> None:
        """Initializes the Discounted Thompson Sampling bandit.

        Args:
            arms (List[str]): List of unique identifier labels for each arm.
            gamma (float): Exponential decay discount factor in (0, 1]. Defaults to 0.95.
        """
        if not (0.0 < gamma <= 1.0):
            raise ValueError("Discount factor gamma must be in the range (0, 1].")
            
        self.arms = arms
        self.gamma = gamma
        
        # Initialize Beta conjugate parameters to uniform Beta(1, 1)
        self.alpha_ = {arm: 1.0 for arm in arms}
        self.beta_ = {arm: 1.0 for arm in arms}

    def select_arm(self, rng: Optional[np.random.Generator] = None) -> str:
        """Selects an arm using Thompson Sampling on the discounted posteriors."""
        _rng = rng or np.random.default_rng()
        best_arm = self.arms[0]
        best_sample = -1.0

        for arm in self.arms:
            sample = _rng.beta(self.alpha_[arm], self.beta_[arm])
            if sample > best_sample:
                best_sample = sample
                best_arm = arm

        return best_arm

    def update(self, arm: str, reward: float) -> None:
        """Updates the selected arm posterior with the received binary reward.

        Also applies the discount factor gamma to ALL arms to decay older reward memory.
        """
        if arm not in self.alpha_:
            raise ValueError(f"Arm '{arm}' is not registered in this bandit instance.")
        if reward not in (0.0, 1.0):
            raise ValueError("Discounted Beta-Binomial Thompson Sampling supports binary rewards in {0.0, 1.0}.")

        # Decay prior parameters of all arms
        for a in self.arms:
            self.alpha_[a] = max(self.gamma * self.alpha_[a], 1.0)
            self.beta_[a] = max(self.gamma * self.beta_[a], 1.0)

        # Add current reward observation to the selected arm
        if reward == 1.0:
            self.alpha_[arm] += 1.0
        else:
            self.beta_[arm] += 1.0

    @property
    def posteriors(self) -> Dict[str, List[float]]:
        """Returns the current alpha and beta parameters for each arm."""
        return {arm: [self.alpha_[arm], self.beta_[arm]] for arm in self.arms}


class SlidingWindowThompsonSamplingBandit:
    """Sliding-Window Thompson Sampling Bandit (SW-TS) for Bernoulli/Binary rewards.

    Computes Beta posteriors based solely on a rolling window of the last W interactions:
        alpha_a = 1.0 + sum_{i in window} Y_i
        beta_a = 1.0 + sum_{i in window} (1 - Y_i)
    """

    def __init__(self, arms: List[str], window_size: int = 100) -> None:
        """Initializes the Sliding-Window Thompson Sampling bandit.

        Args:
            arms (List[str]): List of arm labels.
            window_size (int): Size of the sliding history window. Defaults to 100.
        """
        if window_size < 1:
            raise ValueError("Sliding window size must be at least 1.")

        self.arms = arms
        self.window_size = window_size
        
        # Keep histories of actions and rewards as sequences of tuples: (arm, reward)
        self.history_: List[tuple] = []

    def select_arm(self, rng: Optional[np.random.Generator] = None) -> str:
        """Selects an arm using Thompson Sampling on the window-restricted posteriors."""
        _rng = rng or np.random.default_rng()
        
        # Re-compute counts restricted to current sliding window
        window_history = self.history_[-self.window_size:]
        
        alpha = {arm: 1.0 for arm in self.arms}
        beta = {arm: 1.0 for arm in self.arms}
        
        for arm, reward in window_history:
            if arm in alpha:
                if reward == 1.0:
                    alpha[arm] += 1.0
                else:
                    beta[arm] += 1.0

        best_arm = self.arms[0]
        best_sample = -1.0

        for arm in self.arms:
            sample = _rng.beta(alpha[arm], beta[arm])
            if sample > best_sample:
                best_sample = sample
                best_arm = arm

        return best_arm

    def update(self, arm: str, reward: float) -> None:
        """Appends the selected arm and binary reward to the history trace."""
        if arm not in self.arms:
            raise ValueError(f"Arm '{arm}' is not registered in this bandit instance.")
        if reward not in (0.0, 1.0):
            raise ValueError("Sliding-Window Thompson Sampling supports binary rewards in {0.0, 1.0}.")

        self.history_.append((arm, reward))
        # Optional: truncate memory buffer to prevent infinite memory leak
        if len(self.history_) > 2 * self.window_size:
            self.history_ = self.history_[-self.window_size:]
