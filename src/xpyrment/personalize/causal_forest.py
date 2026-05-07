"""Causal Trees and Causal Forests for Heterogeneous Treatment Effects (HTE).

This module provides custom implementations of Causal Trees (Athey & Imbens, 2016)
and Causal Forests to partition treatment effect variances across covariates.
"""

from typing import List, Optional
import numpy as np


class CausalTreeNode:
    """Represents a split node or leaf within a CausalTree."""

    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["CausalTreeNode"] = None,
        right: Optional["CausalTreeNode"] = None,
        effect: float = 0.0,
    ):
        """Initializes a node.

        Args:
            feature (Optional[int]): The splitting feature index.
            threshold (Optional[float]): The splitting value threshold.
            left (Optional[CausalTreeNode]): Left child node.
            right (Optional[CausalTreeNode]): Right child node.
            effect (float): Calculated treatment effect at this node/leaf.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.effect = effect


class CausalTree:
    """Decision-tree based treatment effect partitioning (Athey & Imbens 2016)."""

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 5):
        """Initializes the CausalTree.

        Args:
            max_depth (int): Maximum depth of the tree. Defaults to 3.
            min_samples_leaf (int): Minimum required observations in each leaf. Defaults to 5.
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray):
        """Fits the CausalTree partitioning.

        Args:
            X (np.ndarray): Covariate feature matrix of shape (n_samples, n_features).
            treatment (np.ndarray): Binary treatment indicators.
            y (np.ndarray): Outcome target vector.
        """
        self.root = self._build_tree(X, treatment, y, depth=0)
        return self

    def _calculate_effect(self, treatment: np.ndarray, y: np.ndarray) -> float:
        """Calculates the average treatment effect difference (mean treatment - mean control)."""
        y_t = y[treatment == 1]
        y_c = y[treatment == 0]
        if len(y_t) == 0 or len(y_c) == 0:
            return 0.0
        return float(np.mean(y_t) - np.mean(y_c))

    def _build_tree(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray, depth: int) -> CausalTreeNode:
        """Recursively builds the causal decision tree splits."""
        n_samples, n_features = X.shape
        node_effect = self._calculate_effect(treatment, y)

        # Base case
        if depth >= self.max_depth or n_samples < 2 * self.min_samples_leaf:
            return CausalTreeNode(effect=node_effect)

        best_score = -float("inf")
        best_split = None

        for f in range(n_features):
            thresholds = np.unique(X[:, f])
            for t in thresholds:
                left_mask = X[:, f] <= t
                right_mask = ~left_mask

                n_l = np.sum(left_mask)
                n_r = n_samples - n_l

                # Leaf size validation
                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue

                # Ensure both arms exist in both child node splits
                t_l = treatment[left_mask]
                t_r = treatment[right_mask]
                if np.sum(t_l == 1) < 2 or np.sum(t_l == 0) < 2 or np.sum(t_r == 1) < 2 or np.sum(t_r == 0) < 2:
                    continue

                tau_l = self._calculate_effect(treatment[left_mask], y[left_mask])
                tau_r = self._calculate_effect(treatment[right_mask], y[right_mask])

                # Maximize Athey-Imbens causal partition variance: n_l * n_r * (tau_l - tau_r)^2
                score = n_l * n_r * ((tau_l - tau_r) ** 2)

                if score > best_score:
                    best_score = score
                    best_split = (f, t, left_mask, right_mask)

        if best_split is None:
            return CausalTreeNode(effect=node_effect)

        f, t, left_mask, right_mask = best_split
        left_child = self._build_tree(X[left_mask], treatment[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], treatment[right_mask], y[right_mask], depth + 1)

        return CausalTreeNode(feature=f, threshold=t, left=left_child, right=right_child, effect=node_effect)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the conditional treatment effect (CATE) for each row.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Vector of estimated effects.
        """
        return np.array([self._predict_row(self.root, row) for row in X])

    def _predict_row(self, node: CausalTreeNode, row: np.ndarray) -> float:
        """Traverses the tree to predict the outcome for a single row."""
        if node.feature is None:
            return node.effect

        if row[node.feature] <= node.threshold:
            return self._predict_row(node.left, row)
        else:
            return self._predict_row(node.right, row)


class CausalForest:
    """Ensemble of CausalTrees to estimate smooth heterogeneous treatment effects."""

    def __init__(self, n_estimators: int = 10, max_depth: int = 3, min_samples_leaf: int = 5):
        """Initializes the CausalForest.

        Args:
            n_estimators (int): Number of independent tree bootstrap samples. Defaults to 10.
            max_depth (int): Maximum depth of each tree. Defaults to 3.
            min_samples_leaf (int): Minimum leaf constraints. Defaults to 5.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees: List[CausalTree] = []

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray, rng: Optional[np.random.Generator] = None):
        """Fits an ensemble of bootstrapped causal trees.

        Args:
            X (np.ndarray): Feature covariates.
            treatment (np.ndarray): Binary treatment indices.
            y (np.ndarray): Outcome values.
            rng (Optional[np.random.Generator]): Optional random number generator.
        """
        if rng is None:
            rng = np.random.default_rng()

        n_samples = X.shape[0]
        self.trees = []

        for _ in range(self.n_estimators):
            # Formulate bootstrap sample indices
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            tree = CausalTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X[indices], treatment[indices], y[indices])
            self.trees.append(tree)

        return self

    def estimate_effect(self, X: np.ndarray) -> np.ndarray:
        """Estimates individual-level treatment effects (CATE) by averaging forest outcomes.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Vector of averaged causal treatment effect estimates.
        """
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(preds, axis=0)
