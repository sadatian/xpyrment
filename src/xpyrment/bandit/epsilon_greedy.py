"""Epsilon-Greedy multi-armed bandit algorithms.

This module provides the `EpsilonGreedyBandit` class for adaptive traffic allocation
minimizing cumulative regret over experimental lifecycles.
"""

from typing import List, Optional
import numpy as np


class EpsilonGreedyBandit:
    """Implements standard and decaying Epsilon-Greedy multi-armed bandit algorithms.

    # TODO: Extend with contextual multi-armed bandit variants using online regression base learners.
    """

    def __init__(self, arms: List[str], epsilon: float = 0.1, decay_rate: float = 0.995, min_epsilon: float = 0.05):
        """Initializes the bandit.

        Args:
            arms (List[str]): List of variant names.
            epsilon (float): The exploration probability in [0.0, 1.0]. Defaults to 0.1.
            decay_rate (float): Multiplicative decay factor for decaying epsilon. Defaults to 0.995.
            min_epsilon (float): Minimum exploration probability floor. Defaults to 0.05.
        """
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"epsilon must be between 0.0 and 1.0, got {epsilon}")
        if not (0.0 <= min_epsilon <= 1.0):
            raise ValueError(f"min_epsilon must be between 0.0 and 1.0, got {min_epsilon}")
        if min_epsilon > epsilon:
            raise ValueError(f"min_epsilon ({min_epsilon}) cannot be greater than initial epsilon ({epsilon})")

        self.arms = arms
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.counts = {arm: 0 for arm in arms}
        self.values = {arm: 0.0 for arm in arms}
        self.total_steps = 0

    def select_arm(self, rng: Optional[np.random.Generator] = None) -> str:
        """Selects an arm using the active exploration rate (epsilon).

        Returns:
            str: The selected arm/variant name.
        """
        if rng is None:
            rng = np.random.default_rng()

        if rng.uniform(0.0, 1.0) < self.epsilon:
            # Explore: select completely random arm
            return rng.choice(self.arms)
        else:
            # Exploit: select arm with highest sample average reward
            best_val = -float("inf")
            best_arms = []
            for arm in self.arms:
                val = self.values[arm]
                if val > best_val:
                    best_val = val
                    best_arms = [arm]
                elif val == best_val:
                    best_arms.append(arm)
            return rng.choice(best_arms)

    def update(self, arm: str, reward: float):
        """Updates the selection count and sample average reward of the selected arm.

        Args:
            arm (str): Name of the arm being updated.
            reward (float): Observed feedback/reward scalar.
        """
        if arm not in self.counts:
            raise ValueError(f"Arm '{arm}' is not registered in this bandit.")

        self.total_steps += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        # Incremental running mean formula
        self.values[arm] += (reward - self.values[arm]) / n

        # Decay epsilon over steps
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
