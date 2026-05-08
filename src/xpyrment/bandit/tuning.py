"""Auto-tuned hyperparameter optimization for adaptive bandits using Bayesian Optimization.

Optimizes multi-armed bandit hyperparameters (e.g. exploration probability epsilon,
UCB scaling c, Thompson sampling prior parameters) via Gaussian Process regression and
Expected Improvement (EI) optimization.

# TODO: Implement automatic kernel lengthscale optimization (marginal likelihood maximization) via Brent's method or gradient descent on GP log likelihood.
"""

from typing import Callable, Dict, List, Optional, Tuple, Type, Any
import numpy as np
from scipy.stats import norm


class GaussianProcessRegressor:
    """Gaussian Process Regressor with Radial Basis Function (RBF) kernel.

    Fits a Gaussian Process prior to observations and estimates the posterior
    mean and standard deviation at arbitrary input coordinates.
    """

    def __init__(self, l: float = 0.5, sigma_f: float = 1.0, sigma_n: float = 1e-4) -> None:
        """Initializes the Gaussian Process Regressor.

        Args:
            l (float): Length-scale of the RBF kernel. Defaults to 0.5.
            sigma_f (float): Signal variance scaling of the kernel. Defaults to 1.0.
            sigma_n (float): Noise standard deviation added to diagonal for regularization.
                Defaults to 1e-4.
        """
        self.l = l
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.L_: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Computes the Radial Basis Function / Squared Exponential kernel matrix.

        K_{i,j} = sigma_f^2 * exp(-||x1_i - x2_j||^2 / (2 * l^2))
        """
        # Efficient vectorized pairwise distance computation
        sq_dist = (
            np.sum(X1**2, axis=1, keepdims=True)
            + np.sum(X2**2, axis=1)
            - 2 * np.dot(X1, X2.T)
        )
        # Numerical stability clip for floating point precision issues
        sq_dist = np.clip(sq_dist, 0.0, None)
        return (self.sigma_f**2) * np.exp(-sq_dist / (2.0 * (self.l**2)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the GP regressor to the observed data.

        Precomputes either the Cholesky decomposition of the covariance matrix
        or its pseudoinverse for numerical stability.

        Args:
            X (np.ndarray): Input training points of shape (N, D).
            y (np.ndarray): Target outcomes of shape (N,).
        """
        self.X_train = np.atleast_2d(X)
        self.y_train = y.ravel()
        n = self.X_train.shape[0]

        # K(X, X) + sigma_n^2 * I
        K = self._rbf_kernel(self.X_train, self.X_train) + (self.sigma_n**2) * np.eye(n)

        try:
            self.L_ = np.linalg.cholesky(K)
            self.K_inv = None
        except np.linalg.LinAlgError:
            self.L_ = None
            self.K_inv = np.linalg.pinv(K)

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts the posterior mean and standard deviation at test coordinates.

        Args:
            X_test (np.ndarray): Evaluation coordinates of shape (M, D).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Posterior mean vector of shape (M,).
                - Posterior standard deviation vector of shape (M,).
        """
        X_test = np.atleast_2d(X_test)
        if self.X_train is None:
            m = X_test.shape[0]
            return np.zeros(m), self.sigma_f * np.ones(m)

        K_trans = self._rbf_kernel(X_test, self.X_train)
        K_test = self._rbf_kernel(X_test, X_test)

        if self.L_ is not None:
            # Solve using stable Cholesky factor
            # L alpha = y_train -> alpha = L^-T L^-1 y_train
            alpha = np.linalg.solve(self.L_, self.y_train)
            alpha = np.linalg.solve(self.L_.T, alpha)
            mu = np.dot(K_trans, alpha)

            v = np.linalg.solve(self.L_, K_trans.T)
            var = np.diag(K_test) - np.sum(v**2, axis=0)
        else:
            # Fallback to pseudoinverse
            mu = np.dot(K_trans, np.dot(self.K_inv, self.y_train))
            var = np.diag(K_test) - np.sum(
                np.dot(K_trans, self.K_inv) * K_trans, axis=1
            )

        var = np.clip(var, 1e-12, None)
        return mu, np.sqrt(var)


class BanditHyperparameterTuner:
    """Tunes hyperparameter bounds of multi-armed bandit algorithms using Bayesian Optimization.

    Applies a Gaussian Process regressor and optimizes the Expected Improvement (EI)
    acquisition function over continuous search spaces to find bandit parameters
    maximizing cumulative simulation outcomes.
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        evaluate_fn: Callable[[Dict[str, float]], float],
        l: float = 0.5,
    ) -> None:
        """Initializes the BanditHyperparameterTuner.

        Args:
            bounds (Dict[str, Tuple[float, float]]): Dictionary mapping parameter names
                to their search ranges (lower_bound, upper_bound).
                Example: {'epsilon': (0.01, 0.4), 'decay_rate': (0.9, 0.999)}
            evaluate_fn (Callable[[Dict[str, float]], float]): Evaluator callback taking a
                parameter assignment dictionary and returning a reward scalar to maximize.
            l (float): Length-scale parameter for the Gaussian Process RBF kernel.
                Defaults to 0.5.
        """
        self.bounds = bounds
        self.evaluate_fn = evaluate_fn
        self.param_names = list(bounds.keys())
        self.gp = GaussianProcessRegressor(l=l)

        self.X_history: List[np.ndarray] = []
        self.y_history: List[float] = []
        self.best_params: Dict[str, float] = {}
        self.best_score: float = -float("inf")

    def _dict_to_array(self, d: Dict[str, float]) -> np.ndarray:
        """Converts a parameter dictionary to a 1D NumPy array."""
        return np.array([d[name] for name in self.param_names])

    def _array_to_dict(self, arr: np.ndarray) -> Dict[str, float]:
        """Converts a 1D NumPy array of parameter coordinates to a dictionary."""
        return {name: float(arr[i]) for i, name in enumerate(self.param_names)}

    def expected_improvement(
        self, mu: np.ndarray, sigma: np.ndarray, best_y: float
    ) -> np.ndarray:
        """Computes the Expected Improvement (EI) acquisition function values.

        EI(x) = (mu(x) - f(x^+))*Phi(Z) + sigma(x)*phi(Z)

        Args:
            mu (np.ndarray): Expected posterior predictions.
            sigma (np.ndarray): Posterior standard deviation uncertainty.
            best_y (float): The current maximum observed evaluation score.

        Returns:
            np.ndarray: Vector of Expected Improvement values.
        """
        sigma_safe = np.clip(sigma, 1e-9, None)
        improvement = mu - best_y
        Z = improvement / sigma_safe

        phi = norm.pdf(Z)
        Phi = norm.cdf(Z)

        ei = improvement * Phi + sigma_safe * phi
        # Zero improvement expectation if variance is vanishing
        ei = np.where(sigma < 1e-9, np.maximum(0.0, improvement), ei)
        return ei

    def optimize(
        self, n_init: int = 5, n_iter: int = 15, seed: Optional[int] = None
    ) -> Dict[str, float]:
        """Executes the Bayesian Optimization sequence to tune bandit hyperparameters.

        Args:
            n_init (int): Number of initial randomized parameters to evaluate. Defaults to 5.
            n_iter (int): Number of sequential acquisition optimization iterations. Defaults to 15.
            seed (Optional[int]): Random seed for deterministic replicability.

        Returns:
            Dict[str, float]: The optimal hyperparameter configuration found.
        """
        rng = np.random.default_rng(seed)
        num_params = len(self.param_names)

        # 1. Random Initialization Pass
        for _ in range(n_init):
            sample = {}
            for name, (low, high) in self.bounds.items():
                sample[name] = float(rng.uniform(low, high))

            score = self.evaluate_fn(sample)
            self.X_history.append(self._dict_to_array(sample))
            self.y_history.append(score)

            if score > self.best_score:
                self.best_score = score
                self.best_params = sample

        # 2. Sequential Bayesian Optimization Loop
        for _ in range(n_iter):
            X_train = np.array(self.X_history)
            y_train = np.array(self.y_history)

            # Fit Gaussian Process posterior
            self.gp.fit(X_train, y_train)

            # Draw a dense collection of candidate points to maximize the acquisition function
            # Monte Carlo candidate generation is robust to non-convex EI surfaces
            n_candidates = 2000
            candidates = np.zeros((n_candidates, num_params))
            for i, name in enumerate(self.param_names):
                low, high = self.bounds[name]
                candidates[:, i] = rng.uniform(low, high, size=n_candidates)

            # Compute predictions on candidates
            mu, sigma = self.gp.predict(candidates)
            ei_scores = self.expected_improvement(mu, sigma, self.best_score)

            # Find candidate that maximizes Expected Improvement
            best_candidate_idx = np.argmax(ei_scores)
            selected_coords = candidates[best_candidate_idx]
            selected_params = self._array_to_dict(selected_coords)

            # Evaluate selected parameters
            score = self.evaluate_fn(selected_params)

            self.X_history.append(selected_coords)
            self.y_history.append(score)

            if score > self.best_score:
                self.best_score = score
                self.best_params = selected_params

        return self.best_params


def simulate_bandit_run(
    bandit_class: Type[Any],
    bandit_args: Dict[str, Any],
    arms: List[str],
    true_means: Dict[str, float],
    reward_type: Literal["binary", "continuous"] = "binary",
    steps: int = 500,
    seed: Optional[int] = None,
) -> float:
    """Simulates a multi-armed bandit run on static Bernoulli or Gaussian reward arms.

    Useful as an evaluation harness for tuning multi-armed bandits.

    Args:
        bandit_class (Type): Class of bandit to instantiate (e.g., EpsilonGreedyBandit).
        bandit_args (Dict[str, Any]): Hyperparameter assignments passed to class constructor.
        arms (List[str]): List of variant/arm identifiers.
        true_means (Dict[str, float]): True reward means of the arms.
        reward_type (str): Type of rewards, either 'binary' or 'continuous'. Defaults to 'binary'.
        steps (int): Simulation duration/horizon. Defaults to 500.
        seed (Optional[int]): Random seed.

    Returns:
        float: Cumulative rewards gathered across the simulation horizon.
    """
    rng = np.random.default_rng(seed)
    bandit = bandit_class(arms=arms, **bandit_args)
    total_reward = 0.0

    for _ in range(steps):
        selected = bandit.select_arm(rng=rng)
        mean = true_means[selected]

        if reward_type == "binary":
            reward = float(rng.binomial(1, mean))
        else:
            reward = float(rng.normal(mean, 1.0))

        bandit.update(selected, reward)
        total_reward += reward

    return total_reward
