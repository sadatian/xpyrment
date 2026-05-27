"""Large-Scale Distributed Graph Partitioning & Entropy-Balanced LPA (Block 21).

Implements entropy-constrained Label Propagation community detection to partition
large-scale network graphs into size-balanced communities to prevent giant cluster
collapse and minimize normalized cut leakage.
"""

from typing import Dict, List, Union
import numpy as np


class EntropyBalancedGraphPartitioner:
    """Graph partitioner using Entropy-Constrained Label Propagation Algorithm (EC-LPA).

    Balances community sizes by adding a penalty proportional to current community size,
    maximizing community partition entropy while minimizing normalized cut:
    
    NCut = Sum_k (cut(C_k, C_k^c) / vol(C_k))

    # TODO: Implement asynchronous lock-free update rules (Hogwild!) to parallelize label changes on massive graphs.
    # TODO: Support Louvain modularity criteria (Q-index) as an alternative objective to balance small local clusters.
    """

    def __init__(self, adjacency_dict: Dict[Union[int, str], List[Union[int, str]]], gamma: float = 0.1) -> None:
        """Initializes the partitioner.

        Args:
            adjacency_dict (Dict): Adjacency list mapping node IDs to list of neighbor IDs.
            gamma (float): Balancing penalty factor. Larger values favor more equal community sizes.
                Defaults to 0.1.
        """
        self.adj = adjacency_dict
        self.gamma = gamma
        self.nodes = list(self.adj.keys())
        self.N = len(self.nodes)

    def fit_predict(self, max_iters: int = 30, seed: int = 42) -> Dict[Union[int, str], int]:
        """Runs the Entropy-Constrained Label Propagation Algorithm.

        Args:
            max_iters (int): Maximum LPA iterations.
            seed (int): Random seed for permutation.

        Returns:
            Dict: Dictionary mapping node keys to community label indices.
        """
        # 1. Initialize each node in its own unique community
        labels = {node: idx for idx, node in enumerate(self.nodes)}
        
        # Track community sizes
        community_sizes = {idx: 1 for idx in range(self.N)}

        rng = np.random.default_rng(seed)

        for _ in range(max_iters):
            order = rng.permutation(self.nodes)
            changed = False

            for node in order:
                neighbors = self.adj[node]
                if not neighbors:
                    continue

                # Count neighbor community votes
                votes: Dict[int, float] = {}
                for neighbor in neighbors:
                    if neighbor in labels:
                        lbl = labels[neighbor]
                        votes[lbl] = votes.get(lbl, 0.0) + 1.0

                # Apply entropy balance penalty: vote_score = count - gamma * (size_of_community / N)
                best_label = labels[node]
                best_score = -float("inf")

                for lbl, vote_count in votes.items():
                    size_lbl = community_sizes.get(lbl, 0)
                    score = vote_count - self.gamma * (size_lbl / self.N)
                    if score > best_score:
                        best_score = score
                        best_label = lbl

                if labels[node] != best_label:
                    # Update community sizes
                    old_label = labels[node]
                    community_sizes[old_label] -= 1
                    if community_sizes[old_label] == 0:
                        del community_sizes[old_label]

                    labels[node] = best_label
                    community_sizes[best_label] = community_sizes.get(best_label, 0) + 1
                    changed = True

            if not changed:
                break

        return labels

    def compute_size_entropy(self, labels: Dict[Union[int, str], int]) -> float:
        """Computes the Shannon entropy of community sizes: H = - sum (p_k * log(p_k))."""
        from collections import Counter
        counts = Counter(labels.values())
        entropy = 0.0
        for count in counts.values():
            p = count / self.N
            if p > 0:
                entropy -= p * np.log(p)
        return float(entropy)

    def compute_normalized_cut(self, labels: Dict[Union[int, str], int]) -> float:
        """Computes the Normalized Cut (NCut) value of the community partition.

        NCut(C) = Sum_k (cut(C_k, C_k^c) / vol(C_k))
        """
        unique_labels = set(labels.values())
        n_cut_total = 0.0

        for k in unique_labels:
            # Nodes in community k
            nodes_k = {node for node, lbl in labels.items() if lbl == k}
            
            cut = 0.0
            vol = 0.0
            
            for node in nodes_k:
                neighbors = self.adj[node]
                vol += len(neighbors)
                for neighbor in neighbors:
                    if labels.get(neighbor) != k:
                        cut += 1.0

            if vol > 0:
                n_cut_total += cut / vol
            else:
                n_cut_total += 0.0

        return float(n_cut_total)
