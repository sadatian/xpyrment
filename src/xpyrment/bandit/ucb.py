"""Upper Confidence Bound (UCB1) multi-armed bandit algorithms.

This module provides the `UCB1Bandit` class for optimistic-under-uncertainty traffic allocation.
"""

import math
from typing import List, Optional
import numpy as np


class UCB1Bandit:
    """Implements the classical UCB1 (Upper Confidence Bound) multi-armed bandit algorithm."""

    def __init__(self, arms: List[str], c: float = 2.0):
        """Initializes the bandit.

        Args:
            arms (List[str]): List of variant names.
            c (float): Exploration variance scaling coefficient. Defaults to 2.0.
        """
        self.arms = arms
        self.c = c
        self.counts = {arm: 0 for arm in arms}
        self.values = {arm: 0.0 for arm in arms}
        self.total_steps = 0

    def select_arm(self, rng: Optional[np.random.Generator] = None) -> str:
        """Selects an arm based on the optimistic UCB1 index metric.

        Returns:
            str: The selected arm/variant name.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Phase 1: Play each arm at least once to initialize statistics
        unplayed = [arm for arm in self.arms if self.counts[arm] == 0]
        if unplayed:
            return rng.choice(unplayed)

        # Phase 2: Compute UCB1 index for each arm
        best_ucb = -float("inf")
        best_arms = []

        for arm in self.arms:
            mean_val = self.values[arm]
            n_i = self.counts[arm]
            
            # Optimistic uncertainty factor
            ucb_val = mean_val + math.sqrt((self.c * math.log(self.total_steps)) / n_i)

            if ucb_val > best_ucb:
                best_ucb = ucb_val
                best_arms = [arm]
            elif ucb_val == best_ucb:
                best_arms.append(arm)

        return rng.choice(best_arms)

    def update(self, arm: str, reward: float):
        """Updates the count and sample average reward of the selected arm.

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
