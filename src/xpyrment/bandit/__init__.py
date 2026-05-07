"""Multi-armed bandit and adaptive allocation algorithms.

Submodules:
- `epsilon_greedy`: Implements exploration-exploitation epsilon-greedy.
- `ucb`: Implements Upper Confidence Bound (UCB1).
- `thompson`: Implements Thompson Sampling using conjugate distributions.
"""

from xpyrment.bandit.epsilon_greedy import EpsilonGreedyBandit
from xpyrment.bandit.ucb import UCB1Bandit
from xpyrment.bandit.thompson import ThompsonSamplingBandit

__all__ = [
    "EpsilonGreedyBandit",
    "UCB1Bandit",
    "ThompsonSamplingBandit",
]
