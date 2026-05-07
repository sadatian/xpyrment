import pytest
import numpy as np
from xpyrment.bandit.epsilon_greedy import EpsilonGreedyBandit
from xpyrment.bandit.ucb import UCB1Bandit
from xpyrment.bandit.thompson import ThompsonSamplingBandit


def test_epsilon_greedy_bandit():
    """Validates the exploration-exploitation balance and exponential decay of EpsilonGreedyBandit."""
    rng = np.random.default_rng(42)
    bandit = EpsilonGreedyBandit(arms=["A", "B"], epsilon=0.5, decay_rate=0.90)

    # Update arm A with high average value and B with low average value
    for _ in range(10):
        bandit.update("A", 10.0)
        bandit.update("B", 1.0)

    assert bandit.values["A"] == 10.0
    assert bandit.values["B"] == 1.0

    # With a small epsilon (decayed over 20 steps), it should overwhelmingly exploit arm A
    assert bandit.epsilon == pytest.approx(0.5 * (0.90 ** 20))
    
    # Force epsilon to 0 to test pure exploitation
    bandit.epsilon = 0.0
    for _ in range(100):
        assert bandit.select_arm(rng) == "A"

    # Test error on invalid arm update
    with pytest.raises(ValueError):
        bandit.update("INVALID_ARM", 1.0)


def test_ucb1_bandit():
    """Validates that UCB1 initializes by sampling all arms, then picks optimally with logarithmic bounds."""
    rng = np.random.default_rng(42)
    bandit = UCB1Bandit(arms=["A", "B", "C"], c=2.0)

    # 1. Assert that initialization plays every arm once
    init_selections = [bandit.select_arm(rng) for _ in range(3)]
    assert set(init_selections) == {"A", "B", "C"}

    # Update counts to reflect initial plays
    for arm in init_selections:
        bandit.update(arm, 1.0)

    # 2. Update Arm A with consistently high rewards, B and C with low rewards
    for _ in range(100):
        arm = bandit.select_arm(rng)
        reward = 10.0 if arm == "A" else 0.5
        bandit.update(arm, reward)

    # Assert that the optimal arm A is selected significantly more than the others
    assert bandit.counts["A"] > bandit.counts["B"]
    assert bandit.counts["A"] > bandit.counts["C"]


def test_thompson_sampling_binary():
    """Validates Beta-Binomial conjugate posterior updating and selection convergence in Thompson Sampling."""
    rng = np.random.default_rng(42)
    bandit = ThompsonSamplingBandit(arms=["A", "B"], reward_type="binary")

    # Initial uniform priors: Beta(1, 1)
    assert bandit.params["A"] == [1.0, 1.0]

    # Arm A gets 10 successes, Arm B gets 10 failures
    for _ in range(10):
        bandit.update("A", 1.0)
        bandit.update("B", 0.0)

    # Posteriors should be updated to Beta(11, 1) and Beta(1, 11)
    assert bandit.params["A"] == [11.0, 1.0]
    assert bandit.params["B"] == [1.0, 11.0]

    # Select arm 100 times; Arm A (high success probability) should always be chosen
    selections = [bandit.select_arm(rng) for _ in range(100)]
    assert all(sel == "A" for sel in selections)

    # Verify binary reward boundary check
    with pytest.raises(ValueError):
        bandit.update("A", 5.5)


def test_thompson_sampling_continuous():
    """Validates Normal-Normal conjugate posterior parameter calculations for continuous reward Thompson Sampling."""
    rng = np.random.default_rng(42)
    
    # Custom prior: (mean_0 = 5.0, precision_0 = 2.0)
    prior = {"A": (5.0, 2.0), "B": (2.0, 1.0)}
    bandit = ThompsonSamplingBandit(arms=["A", "B"], reward_type="continuous", prior_params=prior)

    assert bandit.params["A"] == [5.0, 2.0]

    # Update Arm A with a reward of 8.0
    # precision_new = precision_old + 1 = 2 + 1 = 3
    # mu_new = (precision_old * mu_old + reward) / precision_new = (2 * 5 + 8) / 3 = 6.0
    bandit.update("A", 8.0)

    assert bandit.params["A"][1] == 3.0
    assert bandit.params["A"][0] == pytest.approx(6.0)
