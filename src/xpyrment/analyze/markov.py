"""Multi-State Markov Transition Journey Modeling (Block 30).

Jointly models discrete user journey funnels and sequential state-to-state transitions,
computing transition probability matrices, stationary distributions, and running
homogeneity tests to identify treatment impact.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import chi2


class MarkovJourneyAnalyzer:
    """Analyzes chronological user state transitions in experimentation datasets.

    Extracts transition probability matrices, stationary distributions (long-term states),
    and executes Chi-squared transition homogeneity tests.

    # TODO: Add continuous-time Markov intensity matrix (Q) estimations to model exact duration stay times within states.
    # TODO: Implement bootstrap confidence interval approximations for stationary distribution probability shifts.
    """

    def __init__(self, states: List[str]) -> None:
        """Initializes the Markov journey analyzer.

        Args:
            states (List[str]): Unique ordered labels of possible states in the funnel.
        """
        self.states = sorted(list(set(states)))
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.S = len(self.states)

    def extract_transitions(self, df: pd.DataFrame, user_col: str, state_col: str) -> List[Tuple[str, str]]:
        """Helper to extract chronological pairwise transitions from a DataFrame grouped by user.

        Args:
            df (pd.DataFrame): DataFrame containing chronological sequence of states per user.
            user_col (str): Column indicating user ID.
            state_col (str): Column indicating user state.

        Returns:
            List[Tuple[str, str]]: Chronological transitions.
        """
        transitions = []
        # Group by user and preserve chronological sorting
        for _, group in df.groupby(user_col, sort=False):
            state_list = group[state_col].tolist()
            for i in range(len(state_list) - 1):
                s_from, s_to = state_list[i], state_list[i + 1]
                if s_from in self.state_to_idx and s_to in self.state_to_idx:
                    transitions.append((s_from, s_to))
        return transitions

    def compute_transition_matrix(self, transitions: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
        """Computes transition count and probability matrices.

        Args:
            transitions (List[Tuple[str, str]]): List of state-to-state transitions.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transition count matrix (S, S) and probability matrix (S, S).
        """
        counts = np.zeros((self.S, self.S))
        for s_from, s_to in transitions:
            i = self.state_to_idx[s_from]
            j = self.state_to_idx[s_to]
            counts[i, j] += 1.0

        probs = np.zeros((self.S, self.S))
        for i in range(self.S):
            row_sum = np.sum(counts[i])
            if row_sum > 0:
                probs[i] = counts[i] / row_sum
            else:
                # Absorbing state behavior (transition to itself with probability 1.0)
                probs[i, i] = 1.0

        return counts, probs

    def compute_stationary_distribution(self, probs: np.ndarray, max_iters: int = 150, tol: float = 1e-7) -> np.ndarray:
        """Computes the stationary distribution pi of the Markov transition matrix via power iteration.

        Solves pi * P = pi subject to sum(pi) = 1.0.
        """
        # Start with a uniform distribution
        pi = np.full(self.S, 1.0 / self.S)
        
        for _ in range(max_iters):
            pi_new = np.dot(pi, probs)
            diff = np.linalg.norm(pi_new - pi, 1)
            pi = pi_new
            if diff < tol:
                break
                
        return pi

    def test_transition_homogeneity(
        self, control_transitions: List[Tuple[str, str]], treatment_transitions: List[Tuple[str, str]]
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Runs a Chi-squared homogeneity test of transition matrices between Control and Treatment.

        H0: P^(C) = P^(T) (transition probabilities are identical across groups)
        """
        c_counts, c_probs = self.compute_transition_matrix(control_transitions)
        t_counts, t_probs = self.compute_transition_matrix(treatment_transitions)

        total_chi_stat = 0.0
        total_df = 0

        # Perform test row by row (state by state)
        for i in range(self.S):
            # For state i, look at the counts of transitions to all j states
            c_row = c_counts[i]
            t_row = t_counts[i]

            sum_c = np.sum(c_row)
            sum_t = np.sum(t_row)
            sum_total = sum_c + sum_t

            if sum_c > 0 and sum_t > 0:
                # Expected probabilities under homogeneous transitions
                pooled_row = (c_row + t_row) / sum_total

                # Only test over non-zero expected bins to avoid division by zero
                valid_mask = (pooled_row > 0.0)
                n_valid_bins = np.sum(valid_mask)

                if n_valid_bins > 1:
                    exp_c = sum_c * pooled_row[valid_mask]
                    exp_t = sum_t * pooled_row[valid_mask]

                    obs_c = c_row[valid_mask]
                    obs_t = t_row[valid_mask]

                    chi_stat = np.sum(((obs_c - exp_c) ** 2) / exp_c) + np.sum(((obs_t - exp_t) ** 2) / exp_t)
                    total_chi_stat += chi_stat
                    # df = (rows - 1) * (cols - 1) = (2 - 1) * (n_valid_bins - 1)
                    total_df += (n_valid_bins - 1)

        p_value = 1.0 - chi2.cdf(total_chi_stat, df=total_df) if total_df > 0 else 1.0

        return {
            "chi2_statistic": total_chi_stat,
            "degrees_of_freedom": float(total_df),
            "p_value": float(p_value),
            "control_transition_probabilities": c_probs,
            "treatment_transition_probabilities": t_probs,
        }
