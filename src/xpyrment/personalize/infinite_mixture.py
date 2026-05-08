"""Infinite Dirichlet Process Mixture Clustering (Block 39).

Groups user response signals into latent clusters without manually specifying the cluster
size K beforehand, using a collapsed Gibbs sampler over Dirichlet Process conjugates.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import norm


class InfiniteDirichletClusterer:
    """Implements Dirichlet Process Mixture Models (DPMM) using Collapsed Gibbs Sampling.

    # TODO: Extend the collapsed Gibbs sampler to multivariate Normal-Inverse-Wishart conjugate mixtures.
    # TODO: Implement a Variational Inference (VI) coordinate ascent solver (Blei-Jordan, 2006) to accelerate clustering speed on massive scale datasets.
    """

    def __init__(self, alpha: float = 1.0, prior_mean: float = 0.0, prior_var: float = 10.0, noise_var: float = 1.0) -> None:
        """Initializes the Dirichlet Process mixture model.

        Args:
            alpha (float): DP concentration parameter. Higher alpha means more clusters. Defaults to 1.0.
            prior_mean (float): Prior mean of cluster centroids (mu_0).
            prior_var (float): Prior variance of cluster centroids (tau^2_0).
            noise_var (float): Known standard variance of observations (sigma^2).
        """
        self.alpha = alpha
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.noise_var = noise_var
        self.assignments_: List[int] = []

    def fit_predict(self, data: np.ndarray, n_iters: int = 50, rng: np.random.Generator = None) -> List[int]:
        """Runs the collapsed Gibbs sampler over data points.

        Args:
            data (np.ndarray): 1D outcome data.
            n_iters (int): Number of Gibbs iterations to run. Defaults to 50.

        Returns:
            List[int]: Cluster assignments for each data point.
        """
        _rng = rng or np.random.default_rng(42)
        N = len(data)
        y = data.astype(float)

        # Initialize all observations to a single cluster 0
        assignments = [0] * N
        cluster_counts = {0: N}
        cluster_sums = {0: float(np.sum(y))}

        for _ in range(n_iters):
            for i in range(N):
                y_i = y[i]
                c_curr = assignments[i]

                # 1. Remove point i from its current cluster
                cluster_counts[c_curr] -= 1
                cluster_sums[c_curr] -= y_i

                # Clean up empty clusters
                if cluster_counts[c_curr] == 0:
                    del cluster_counts[c_curr]
                    del cluster_sums[c_curr]

                # 2. Calculate probabilities for existing clusters and a brand new one
                existing_clusters = list(cluster_counts.keys())
                probs = []

                for k in existing_clusters:
                    # Posterior distribution of cluster k mean given current members
                    n_k = cluster_counts[k]
                    sum_k = cluster_sums[k]
                    
                    # Posterior precision / variance of centroid mu_k
                    post_prec = 1.0 / self.prior_var + n_k / self.noise_var
                    post_var = 1.0 / post_prec
                    post_mean = post_var * (self.prior_mean / self.prior_var + sum_k / self.noise_var)

                    # Marginal predictive likelihood: N(y_i | post_mean, post_var + noise_var)
                    prob_y = norm.pdf(y_i, post_mean, np.sqrt(post_var + self.noise_var))
                    probs.append(n_k * prob_y)

                # Predictive likelihood for a brand new cluster: N(y_i | prior_mean, prior_var + noise_var)
                prob_new = norm.pdf(y_i, self.prior_mean, np.sqrt(self.prior_var + self.noise_var))
                probs.append(self.alpha * prob_new)

                # Normalize probability vector
                probs_arr = np.array(probs)
                sum_probs = np.sum(probs_arr)
                if sum_probs > 0:
                    probs_arr /= sum_probs
                else:
                    probs_arr = np.ones(len(probs_arr)) / len(probs_arr)

                # 3. Sample new assignment for point i
                chosen_idx = _rng.choice(len(probs_arr), p=probs_arr)
                
                if chosen_idx < len(existing_clusters):
                    c_new = existing_clusters[chosen_idx]
                else:
                    # Create a new unique cluster label
                    all_labels = set(cluster_counts.keys())
                    c_new = 0
                    while c_new in all_labels:
                        c_new += 1

                # Re-add point i to the selected cluster
                assignments[i] = c_new
                if c_new not in cluster_counts:
                    cluster_counts[c_new] = 0
                    cluster_sums[c_new] = 0.0

                cluster_counts[c_new] += 1
                cluster_sums[c_new] += y_i

        self.assignments_ = assignments
        return assignments
