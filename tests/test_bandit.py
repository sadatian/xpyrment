import pytest
import numpy as np
from xpyrment.bandit import (
    EpsilonGreedyBandit,
    UCB1Bandit,
    ThompsonSamplingBandit,
    BanditHyperparameterTuner,
    simulate_bandit_run,
)


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


def test_gp_regressor():
    """Validates basic functionality and predictive mean/variance of the scratch GaussianProcessRegressor."""
    from xpyrment.bandit.tuning import GaussianProcessRegressor

    gp = GaussianProcessRegressor(l=1.0, sigma_f=1.0, sigma_n=1e-3)

    # Simple 1D dataset: y = x^2
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 4.0, 9.0])

    gp.fit(X, y)

    # Predict at training points: posterior mean should match training points, variance should be small
    mu_train, std_train = gp.predict(X)
    assert mu_train[0] == pytest.approx(1.0, abs=1e-2)
    assert mu_train[1] == pytest.approx(4.0, abs=1e-2)
    assert mu_train[2] == pytest.approx(9.0, abs=1e-2)
    assert std_train[0] < 0.1

    # Predict at a test point far from training data: std dev should increase towards prior std dev (sigma_f = 1.0)
    mu_far, std_far = gp.predict(np.array([[20.0]]))
    assert mu_far[0] == pytest.approx(0.0, abs=0.1)  # reverts to prior mean
    assert std_far[0] == pytest.approx(1.0, abs=0.1)  # reverts to prior std dev


def test_bandit_hyperparameter_tuner():
    """Validates Bayesian Optimization tuning of EpsilonGreedyBandit hyperparameters on a static binary environment."""
    # Define a simple Bernoulli bandit environment
    arms = ["arm_A", "arm_B"]
    true_means = {"arm_A": 0.9, "arm_B": 0.2}  # A is significantly better

    # Evaluation function: runs the simulation 3 times with given hyperparameters and returns the average reward
    def evaluate_fn(params: dict) -> float:
        rewards = []
        for seed in [1, 2, 3]:
            # Scale parameters if needed (bounds are checked)
            reward = simulate_bandit_run(
                bandit_class=EpsilonGreedyBandit,
                bandit_args={
                    "epsilon": params["epsilon"],
                    "decay_rate": params["decay_rate"],
                },
                arms=arms,
                true_means=true_means,
                reward_type="binary",
                steps=100,
                seed=seed,
            )
            rewards.append(reward)
        return float(np.mean(rewards))

    # Parameter space: search bounds for epsilon and decay_rate
    bounds = {
        "epsilon": (0.01, 0.40),
        "decay_rate": (0.90, 0.999),
    }

    tuner = BanditHyperparameterTuner(bounds=bounds, evaluate_fn=evaluate_fn, l=0.5)

    # Run the optimization
    best_params = tuner.optimize(n_init=3, n_iter=5, seed=42)

    assert "epsilon" in best_params
    assert "decay_rate" in best_params
    assert bounds["epsilon"][0] <= best_params["epsilon"] <= bounds["epsilon"][1]
    assert bounds["decay_rate"][0] <= best_params["decay_rate"] <= bounds["decay_rate"][1]

    # Verify optimizer logged execution history
    assert len(tuner.X_history) == 8  # 3 init + 5 iter
    assert len(tuner.y_history) == 8
    assert tuner.best_score > 0.0


def test_off_policy_evaluation():
    """Validates the three-tier Off-Policy Evaluation (IPS, SN-IPS, DR) suite on bandit logs."""
    from xpyrment.bandit.ope import OffPolicyEvaluator

    rng = np.random.default_rng(42)
    n = 200

    # 1D Context
    X = rng.uniform(0.0, 1.0, size=(n, 1))
    
    # Historical actions selected at random (propensities = 0.5)
    actions = rng.binomial(1, 0.5, size=n)
    propensities = np.full(n, 0.5)

    # Historical rewards: arm 1 has higher baseline
    # Expected reward for arm 0 is X * 2.0; for arm 1 is X * 5.0
    rewards = np.zeros(n)
    for i in range(n):
        if actions[i] == 0:
            rewards[i] = X[i, 0] * 2.0 + rng.normal(scale=0.01)
        else:
            rewards[i] = X[i, 0] * 5.0 + rng.normal(scale=0.01)

    # Target policy: always play arm 1 with probability 1.0 (probs shape: (n, 2))
    def target_policy(context: np.ndarray) -> np.ndarray:
        probs = np.zeros((context.shape[0], 2))
        probs[:, 1] = 1.0  # arm 1
        return probs

    evaluator = OffPolicyEvaluator(target_policy=target_policy, l2_penalty=1e-3)
    estimates = evaluator.evaluate(X, actions, propensities, rewards)

    assert "ips" in estimates
    assert "sn_ips" in estimates
    assert "dr" in estimates

    # Expected value of arm 1 = E[5.0 * X] = 5.0 * 0.5 = 2.5
    assert estimates["ips"] == pytest.approx(2.5, abs=0.4)
    assert estimates["sn_ips"] == pytest.approx(2.5, abs=0.4)
    assert estimates["dr"] == pytest.approx(2.5, abs=0.4)


def test_multi_objective_tuning():
    """Validates hypervolume calculations, Pareto front detection, and Expected Hypervolume Improvement propose_next logic."""
    from xpyrment.bandit.multi_objective import MultiObjectiveTuner

    # Reference point is (0.0, 0.0)
    tuner = MultiObjectiveTuner(bounds=[(0.0, 5.0)], reference_point=(0.0, 0.0))

    # Add three points
    tuner.add_observation(np.array([1.0]), (1.0, 5.0))
    tuner.add_observation(np.array([2.0]), (2.0, 4.0))
    tuner.add_observation(np.array([3.0]), (1.5, 3.0))  # Dominated by (2.0, 4.0)

    # 1. Verify Pareto frontier extraction
    frontier = tuner.get_pareto_frontier()
    assert frontier.shape[0] == 2
    # The non-dominated points are exactly (1, 5) and (2, 4)
    point_set = {tuple(p) for p in frontier}
    assert (1.0, 5.0) in point_set
    assert (2.0, 4.0) in point_set

    # 2. Verify hypervolume calculation
    # Area = (1.0 - 0.0) * (5.0 - 0.0) + (2.0 - 1.0) * (4.0 - 0.0) = 5.0 + 4.0 = 9.0
    hv = tuner.compute_hypervolume(frontier)
    assert hv == pytest.approx(9.0)

    # 3. Test EHVI computation and proposal selection
    candidates = np.linspace(0.1, 4.9, 10).reshape(-1, 1)
    best_candidate = tuner.propose_next(candidates, n_samples=10)
    
    assert best_candidate.shape == (1,)
    assert 0.0 <= best_candidate[0] <= 5.0


def test_non_stationary_bandits():
    """Validates the state transitions and sampling of Discounted and Sliding-Window Thompson Sampling bandits."""
    from xpyrment.bandit.non_stationary import DiscountedThompsonSamplingBandit, SlidingWindowThompsonSamplingBandit

    rng = np.random.default_rng(42)

    # 1. Test Discounted Thompson Sampling
    d_bandit = DiscountedThompsonSamplingBandit(arms=["A", "B"], gamma=0.8)
    
    # Update arm A with some successes
    for _ in range(5):
        d_bandit.update("A", 1.0)
    # Update arm B with some failures
    for _ in range(5):
        d_bandit.update("B", 0.0)

    posteriors = d_bandit.posteriors
    assert "A" in posteriors
    assert "B" in posteriors
    # Parameters should be decayed and strictly positive
    assert posteriors["A"][0] > 1.0
    assert posteriors["B"][1] > 1.0

    # Ensure arm selection operates without crashing
    for _ in range(50):
        selected = d_bandit.select_arm(rng)
        assert selected in ["A", "B"]

    # 2. Test Sliding-Window Thompson Sampling
    sw_bandit = SlidingWindowThompsonSamplingBandit(arms=["A", "B"], window_size=5)

    # Push a series of updates
    for _ in range(10):
        sw_bandit.update("A", 1.0)
    for _ in range(10):
        sw_bandit.update("B", 0.0)

    # Within the window of size 5, only arm B should have updates (since we did B last)
    # Uniform alpha, beta (initial 1.0) + window updates
    selected_sw = [sw_bandit.select_arm(rng) for _ in range(100)]
    # Arm A has 0 plays in the last 5 steps (so Beta(1, 1))
    # Arm B has 5 plays with reward 0 in the last 5 steps (so Beta(1, 6))
    # Since Beta(1, 1) sample is statistically much larger than Beta(1, 6), Arm A should dominate selections
    assert selected_sw.count("A") > selected_sw.count("B")


def test_thompson_sampling_default_rng():
    """Verifies that select_arm initializes a default RNG if none is passed."""
    # 1. Binary reward type
    bandit_bin = ThompsonSamplingBandit(arms=["A", "B"], reward_type="binary")
    selected_bin = bandit_bin.select_arm() # No rng passed
    assert selected_bin in ["A", "B"]

    # 2. Continuous reward type
    bandit_cont = ThompsonSamplingBandit(arms=["A", "B"], reward_type="continuous")
    selected_cont = bandit_cont.select_arm() # No rng passed
    assert selected_cont in ["A", "B"]




