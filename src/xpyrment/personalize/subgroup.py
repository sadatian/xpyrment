"""Subgroup Heterogeneity Segment Discovery (Block 48).

Scans baseline covariates to partition sample units into segments maximizing CATE differences
between subsets.
"""

from typing import Dict, Any, List
import numpy as np


class SubgroupHeterogeneityDiscoverer:
    """Finds covariate segments maximizing treatment effect variance.

    TODO: Support multi-way split interactions using Causal Trees (honest splitting) to evaluate subgroup CATE bounds.
    TODO: Add false-discovery rate control corrections (e.g. Benjamini-Hochberg) across all evaluated segment candidates to prevent false subgroup discoveries.
    """

    def __init__(self, min_sample_size: int = 15, feature_names: List[str] = None) -> None:
        """Initializes the subgroup discoverer.

        Args:
            min_sample_size (int): Minimum required samples in each resulting partition.
                Defaults to 15.
            feature_names (List[str], optional): Custom list of labels for each covariate feature.
        """
        self.min_samples = min_sample_size
        self.feature_names = feature_names
        
        self.best_feature_idx_: int = -1
        self.best_feature_name_: str = ""
        self.best_threshold_: float = 0.0
        self.left_cate_: float = 0.0
        self.right_cate_: float = 0.0
        self.max_cate_gap_: float = 0.0

    def fit(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> "SubgroupHeterogeneityDiscoverer":
        """Greedily searches for a feature and threshold maximizing the CATE variance.

        CATE = Mean(Y_Treatment) - Mean(Y_Control) inside a partition.
        Maximizes: (CATE_left - CATE_right)^2
        """
        X = np.asarray(X, dtype=float)
        w = np.asarray(w, dtype=int)
        y = np.asarray(y, dtype=float)

        N, P = X.shape
        if len(w) != N or len(y) != N:
            raise ValueError("All input arrays (X, w, y) must match on row dimension.")

        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(P)]
        elif len(self.feature_names) != P:
            raise ValueError("Length of feature_names must match column dimension of X.")

        best_gap = -1.0
        best_f = -1
        best_thresh = 0.0
        best_left_cate = 0.0
        best_right_cate = 0.0

        for f_idx in range(P):
            feature_vals = X[:, f_idx]
            # Try unique midpoints as potential thresholds
            sorted_vals = np.sort(np.unique(feature_vals))
            if len(sorted_vals) < 2:
                continue

            thresholds = 0.5 * (sorted_vals[:-1] + sorted_vals[1:])

            for threshold in thresholds:
                left_mask = (feature_vals <= threshold)
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples or np.sum(right_mask) < self.min_samples:
                    continue

                # Left split indices
                w_left = w[left_mask]
                y_left = y[left_mask]

                # Right split indices
                w_right = w[right_mask]
                y_right = y[right_mask]

                # Must have both control and treatment in left and right partitions
                if (np.sum(w_left == 0) < 3 or np.sum(w_left == 1) < 3 or
                        np.sum(w_right == 0) < 3 or np.sum(w_right == 1) < 3):
                    continue

                # Calculate left CATE
                mean_y_t_left = np.mean(y_left[w_left == 1])
                mean_y_c_left = np.mean(y_left[w_left == 0])
                cate_left = mean_y_t_left - mean_y_c_left

                # Calculate right CATE
                mean_y_t_right = np.mean(y_right[w_right == 1])
                mean_y_c_right = np.mean(y_right[w_right == 0])
                cate_right = mean_y_t_right - mean_y_c_right

                # Calculate squared gap
                gap_sq = (cate_left - cate_right) ** 2

                if gap_sq > best_gap:
                    best_gap = gap_sq
                    best_f = f_idx
                    best_thresh = threshold
                    best_left_cate = cate_left
                    best_right_cate = cate_right

        if best_gap > 0.0:
            self.best_feature_idx_ = best_f
            self.best_feature_name_ = self.feature_names[best_f]
            self.best_threshold_ = float(best_thresh)
            self.left_cate_ = float(best_left_cate)
            self.right_cate_ = float(best_right_cate)
            self.max_cate_gap_ = float(np.sqrt(best_gap))

        return self

    @property
    def results(self) -> Dict[str, Any]:
        """Returns the segment partition parameters."""
        if self.best_feature_idx_ == -1:
            return {"heterogeneity_found": False}

        return {
            "heterogeneity_found": True,
            "best_feature_index": self.best_feature_idx_,
            "best_feature_name": self.best_feature_name_,
            "best_threshold": self.best_threshold_,
            "left_subgroup_cate": self.left_cate_,
            "right_subgroup_cate": self.right_cate_,
            "cate_gap": self.max_cate_gap_,
        }
