"""Graph-based cluster randomization and community partition algorithms.

This module provides the `ClusterRandomizer` class to execute cluster-level treatment
randomization over network nodes, reducing network spillover/leakage effects.
"""

from typing import List, Optional
import numpy as np


def label_propagation(adjacency: np.ndarray, max_iters: int = 30, seed: Optional[int] = None) -> np.ndarray:
    """Detects community partitions using the fast Label Propagation Algorithm (LPA).

    LPA runs in O(E) complexity, making it highly suitable for large sparse graphs.

    Args:
        adjacency (np.ndarray): Symmetric adjacency matrix of shape (n_nodes, n_nodes).
        max_iters (int): Maximum convergence iterations. Defaults to 30.
        seed (Optional[int]): Random seed for deterministic node permutation.

    Returns:
        np.ndarray: Vector of integer labels representing node communities.
    """
    n_nodes = adjacency.shape[0]
    labels = np.arange(n_nodes)

    rng = np.random.default_rng(seed)

    for _ in range(max_iters):
        order = rng.permutation(n_nodes)
        changed = False
        for node in order:
            # Find neighboring node indices
            neighbors = np.where(adjacency[node] > 0)[0]
            if len(neighbors) == 0:
                continue
            neighbor_labels = labels[neighbors]
            # Select the most frequent label in neighbor set
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            best_label = unique_labels[np.argmax(counts)]

            if labels[node] != best_label:
                labels[node] = best_label
                changed = True

        if not changed:
            break

    return labels


class ClusterRandomizer:
    """Manages graph-based community detection and cluster-level treatment randomizations."""

    def __init__(self, adjacency: np.ndarray):
        """Initializes the ClusterRandomizer.

        Args:
            adjacency (np.ndarray): Symmetric adjacency matrix representing node connections.
        """
        if not np.allclose(adjacency, adjacency.T):
            raise ValueError("The adjacency matrix must be symmetric (undirected network graph).")

        self.adjacency = adjacency
        self.n_nodes = adjacency.shape[0]
        self.labels = None

    def detect_communities(self, seed: Optional[int] = None) -> np.ndarray:
        """Runs label propagation community partition over the network adjacency matrix.

        Args:
            seed (Optional[int]): Optional random seed.

        Returns:
            np.ndarray: Labels vector.
        """
        self.labels = label_propagation(self.adjacency, seed=seed)
        return self.labels

    def assign_treatments(self, ratio: float = 0.5, seed: Optional[int] = None) -> np.ndarray:
        """Assigns entire communities/clusters to treatment (1) or control (0) to control spillover.

        Args:
            ratio (float): Approximate proportion of clusters assigned to treatment. Defaults to 0.5.
            seed (Optional[int]): Optional random seed.

        Returns:
            np.ndarray: Node-level treatment assignment vector of shape (n_nodes,).
        """
        if self.labels is None:
            self.detect_communities(seed=seed)

        unique_clusters = np.unique(self.labels)
        n_clusters = len(unique_clusters)

        rng = np.random.default_rng(seed)
        # Determine number of clusters to treat
        n_treat = int(round(n_clusters * ratio))
        treated_clusters = rng.choice(unique_clusters, size=n_treat, replace=False)

        cluster_map = {c: (1 if c in treated_clusters else 0) for c in unique_clusters}
        node_treatment = np.array([cluster_map[label] for label in self.labels])

        return node_treatment
