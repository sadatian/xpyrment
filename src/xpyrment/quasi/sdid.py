"""Synthetic Difference-in-Differences (SDID) estimator (Arkhangelsky et al., 2021).

Combines unit weights (Synthetic Controls) and time weights (Difference-in-Differences)
to compute a regularized, doubly weighted treatment effect estimator.
"""

from typing import Tuple
import numpy as np
from scipy.optimize import minimize


class SyntheticDifferenceInDifferences:
    """Synthetic Difference-in-Differences (SDID) treatment effect estimator for panel datasets.

    # TODO: Implement placebo-based inference and standard error estimation using block bootstrap.
    # The bootstrap samples are drawn at the unit level to preserve temporal correlation,
    # computing the empirical variance of the bootstrap estimates: Var(tau_sdid) = 1/(B-1) * sum (tau_b - bar{tau})^2.
    """

    def __init__(self, l2_penalty: float = 1e-4) -> None:
        """Initializes the SyntheticDifferenceInDifferences estimator.

        Args:
            l2_penalty (float): Regularization parameter for unit and time weights to handle collinearity.
                Defaults to 1e-4.
        """
        self.l2_penalty = l2_penalty
        self.unit_weights = None
        self.time_weights = None
        self.treatment_effect = None

    def fit_estimate(
        self,
        y_control: np.ndarray,
        y_treated: np.ndarray,
        t_pre: int,
    ) -> float:
        """Optimizes weights and estimates the SDID treatment effect.

        Args:
            y_control (np.ndarray): Panel outcomes of control units, shape (T, N_co).
            y_treated (np.ndarray): Panel outcomes of treated units, shape (T, N_tr).
            t_pre (int): Number of pre-treatment time periods. Treatment starts at period t_pre.

        Returns:
            float: Estimated doubly weighted treatment effect (tau_sdid).
        """
        T, N_co = y_control.shape
        _, N_tr = y_treated.shape

        T_pre = t_pre
        T_post = T - t_pre

        if T_post <= 0:
            raise ValueError("Pre-treatment periods 't_pre' must be strictly less than total periods 'T'.")

        # Partition outcomes into pre and post periods
        y_co_pre = y_control[:T_pre, :]    # (T_pre, N_co)
        y_co_post = y_control[T_pre:, :]  # (T_post, N_co)

        y_tr_pre = y_treated[:T_pre, :]    # (T_pre, N_tr)
        y_tr_post = y_treated[T_pre:, :]  # (T_post, N_tr)

        # Average outcomes
        y_tr_pre_avg = np.mean(y_tr_pre, axis=1)    # (T_pre,)
        y_co_post_avg = np.mean(y_co_post, axis=0)  # (N_co,)

        # 1. Optimize Unit Weights (omega)
        # We find a weight vector omega that matches the pre-treatment average of treated units
        def unit_loss(w):
            pred = np.dot(y_co_pre, w)  # (T_pre,)
            diff = y_tr_pre_avg - pred
            intercept = np.mean(diff)
            penalty = self.l2_penalty * np.sum(w**2)
            return np.sum((diff - intercept) ** 2) + penalty

        unit_constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        unit_bounds = [(0.0, 1.0) for _ in range(N_co)]
        w0_unit = np.ones(N_co) / N_co

        res_unit = minimize(unit_loss, w0_unit, method="SLSQP", bounds=unit_bounds, constraints=unit_constraints)
        self.unit_weights = res_unit.x

        # 2. Optimize Time Weights (lambda)
        # We find a weight vector lambda that matches the average post-treatment outcomes of control units
        def time_loss(l):
            pred = np.dot(y_co_pre.T, l)  # (N_co,)
            diff = y_co_post_avg - pred
            intercept = np.mean(diff)
            penalty = self.l2_penalty * np.sum(l**2)
            return np.sum((diff - intercept) ** 2) + penalty

        time_constraints = {"type": "eq", "fun": lambda l: np.sum(l) - 1.0}
        time_bounds = [(0.0, 1.0) for _ in range(T_pre)]
        w0_time = np.ones(T_pre) / T_pre

        res_time = minimize(time_loss, w0_time, method="SLSQP", bounds=time_bounds, constraints=time_constraints)
        self.time_weights = res_time.x

        # 3. Compute the Synthetic Difference-in-Differences Estimator (tau_sdid)
        # Post-treatment average differences (treated average vs weighted control average)
        post_tr_avg = np.mean(y_tr_post)  # Scalar: average of all treated units across all post-treatment periods
        post_co_weighted = np.mean(np.dot(y_co_post, self.unit_weights))  # Scalar: weighted control average in post-period
        delta_post = post_tr_avg - post_co_weighted

        # Pre-treatment weighted differences (weighted by lambda over time)
        pre_tr_weighted = np.dot(y_tr_pre_avg, self.time_weights)  # Scalar: time-weighted pre-treatment average of treated
        pre_co_weighted = np.dot(np.dot(y_co_pre, self.unit_weights), self.time_weights)  # Scalar: doubly weighted control average
        delta_pre = pre_tr_weighted - pre_co_weighted

        self.treatment_effect = float(delta_post - delta_pre)
        return self.treatment_effect
