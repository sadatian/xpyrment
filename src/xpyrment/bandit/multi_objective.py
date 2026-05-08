"""Multi-Objective Expected Hypervolume Improvement (EHVI) Bayesian Optimization (Block 25).

Provides a Multi-Objective Hyperparameter Tuner that tracks the Pareto-optimal frontier of multiple
competing experimental outcomes, leveraging Gaussian Process surrogates and Expected Hypervolume
Improvement (EHVI) with Monte Carlo integration to select optimal designs.
"""

from typing import List, Tuple
import numpy as np
from xpyrment.bandit.tuning import GaussianProcessRegressor


class MultiObjectiveTuner:
    """Multi-Objective Hyperparameter Tuner using Expected Hypervolume Improvement (EHVI).

    Tracks the Pareto-optimal frontier of two or more competing objectives and uses
    Gaussian Process models to maximize Expected Hypervolume Improvement relative to
    a worst-acceptable reference point.

    # TODO: Implement 3D+ hypervolume partitioning algorithms (e.g., Overmars and Yap) to support arbitrary objective dimensions.
    # TODO: Add hypervolume-based probability of improvement (HV-PI) as an alternative multi-objective acquisition metric.
    """

    def __init__(self, bounds: List[Tuple[float, float]], reference_point: Tuple[float, float]) -> None:
        """Initializes the multi-objective tuner.

        Args:
            bounds (List[Tuple[float, float]]): List of bounds (min, max) for each hyperparameter dimension.
            reference_point (Tuple[float, float]): Reference point representing the minimum acceptable values
                for each objective. Points that do not exceed this reference in all dimensions are disregarded.
        """
        self.bounds = bounds
        self.reference_point = np.array(reference_point)
        self.X_obs: List[np.ndarray] = []
        self.y_obs: List[np.ndarray] = []

        # Gaussian Process model for each objective (assume 2 objectives for exact 2D hypervolume)
        self.gp_models = [GaussianProcessRegressor(l=0.5, sigma_n=1e-3) for _ in range(2)]

    def add_observation(self, x: np.ndarray, y: Tuple[float, float]) -> None:
        """Adds an observed input parameter vector and its corresponding outcome objectives.

        Args:
            x (np.ndarray): Hyperparameter vector of shape (D,).
            y (Tuple[float, float]): Multi-objective outcomes.
        """
        self.X_obs.append(np.atleast_1d(x))
        self.y_obs.append(np.array(y))

    def get_pareto_frontier(self) -> np.ndarray:
        """Computes the set of non-dominated observed outcome points (Pareto frontier)."""
        if not self.y_obs:
            return np.empty((0, 2))

        points = np.vstack(self.y_obs)
        is_efficient = np.ones(points.shape[0], dtype=bool)

        for i, c in enumerate(points):
            # c is dominated if there exists another point j such that
            # all(points[j] >= c) and any(points[j] > c)
            dominated = np.any(np.all(points >= c, axis=1) & np.any(points > c, axis=1))
            is_efficient[i] = not dominated

        return points[is_efficient]

    def compute_hypervolume(self, points: np.ndarray) -> float:
        """Computes the 2D hypervolume of non-dominated points relative to reference_point.

        Uses the sweep-line interval aggregation formula:
        Area = (x_1 - r_1) * (y_1 - r_2) + sum_{i=2}^p (x_i - x_{i-1}) * (y_i - r_2)
        """
        if points.size == 0:
            return 0.0

        r1, r2 = self.reference_point

        # 1. Filter points that dominate/exceed the reference point in both objectives
        valid_mask = (points[:, 0] > r1) & (points[:, 1] > r2)
        valid_points = points[valid_mask]
        if valid_points.shape[0] == 0:
            return 0.0

        # 2. Extract Pareto-efficient subset
        is_eff = np.ones(valid_points.shape[0], dtype=bool)
        for i, c in enumerate(valid_points):
            dominated = np.any(np.all(valid_points >= c, axis=1) & np.any(valid_points > c, axis=1))
            is_eff[i] = not dominated
        pareto = valid_points[is_eff]

        # 3. Sort by first objective ascending
        idx = np.argsort(pareto[:, 0])
        sorted_pareto = pareto[idx]

        # 4. Compute hypervolume
        n_points = sorted_pareto.shape[0]
        x1, y1 = sorted_pareto[0]
        hv = (x1 - r1) * (y1 - r2)

        for i in range(1, n_points):
            xi, yi = sorted_pareto[i]
            x_prev, _ = sorted_pareto[i - 1]
            hv += (xi - x_prev) * (yi - r2)

        return float(hv)

    def compute_ehvi(self, x_test: np.ndarray, n_samples: int = 200) -> np.ndarray:
        """Computes Monte Carlo Expected Hypervolume Improvement (EHVI) at test coordinates.

        Args:
            x_test (np.ndarray): Test coordinates of shape (M, D).
            n_samples (int): Number of Monte Carlo draws. Defaults to 200.

        Returns:
            np.ndarray: Expected Hypervolume Improvement vector of shape (M,).
        """
        x_test = np.atleast_2d(x_test)
        M = x_test.shape[0]

        if not self.y_obs:
            # If no observations, return dummy value indicating high variance/exploration
            return np.ones(M)

        # 1. Fit GP models to each objective
        X_train = np.vstack(self.X_obs)
        Y_train = np.vstack(self.y_obs)

        for m in range(2):
            self.gp_models[m].fit(X_train, Y_train[:, m])

        # Get predictions for all test points
        means = []
        stds = []
        for m in range(2):
            mu, sigma = self.gp_models[m].predict(x_test)
            means.append(mu)
            stds.append(np.clip(sigma, 1e-8, None))

        # Current Pareto frontier and its hypervolume
        current_pareto = self.get_pareto_frontier()
        current_hv = self.compute_hypervolume(current_pareto)

        ehvi_values = np.zeros(M)

        rng = np.random.default_rng(42)

        # 2. Monte Carlo integration of Hypervolume Improvement
        for i in range(M):
            # Sample objectives from posterior distributions
            sampled_obj_1 = rng.normal(loc=means[0][i], scale=stds[0][i], size=n_samples)
            sampled_obj_2 = rng.normal(loc=means[1][i], scale=stds[1][i], size=n_samples)

            improvements = np.zeros(n_samples)
            for s in range(n_samples):
                # Construct virtual candidate point
                candidate = np.array([sampled_obj_1[s], sampled_obj_2[s]])
                
                # Compute hypervolume of current frontier unioned with the candidate
                combined_frontier = np.vstack([current_pareto, candidate])
                new_hv = self.compute_hypervolume(combined_frontier)
                
                improvements[s] = max(0.0, new_hv - current_hv)

            ehvi_values[i] = float(np.mean(improvements))

        return ehvi_values

    def propose_next(self, candidates: np.ndarray, n_samples: int = 150) -> np.ndarray:
        """Selects the candidate coordinate that maximizes Expected Hypervolume Improvement.

        Args:
            candidates (np.ndarray): Set of parameter choices of shape (N_cand, D).
            n_samples (int): Number of Monte Carlo draws. Defaults to 150.

        Returns:
            np.ndarray: The selected parameter choice of shape (D,).
        """
        ehvi = self.compute_ehvi(candidates, n_samples=n_samples)
        best_idx = np.argmax(ehvi)
        return candidates[best_idx]
