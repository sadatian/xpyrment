"""Winsorization & Outlier Stabilization (Block 43).

Capping metrics within percentile bounds to reduce treatment estimator variance on fat-tailed
distributions.
"""

from typing import Tuple, Union
import numpy as np


class WinsorizationEngine:
    """Provides symmetric or asymmetric percentile-based capping boundaries.

    TODO: Support adaptive Hampel filters for time-series rolling outlier identification and correction.
    TODO: Implement Huber-loss robust M-estimation scaling as an alternative to hard Winsorized trimming.
    """

    def __init__(self, bounds: Tuple[float, float] = (0.01, 0.99)) -> None:
        """Initializes the Winsorization Engine.

        Args:
            bounds (Tuple[float, float]): Percentile thresholds [lower, upper] in [0, 1].
                Defaults to (0.01, 0.99).
        """
        lower, upper = bounds
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError("Winsorization bounds must satisfy 0.0 <= lower < upper <= 1.0.")
        self.lower = lower
        self.upper = upper
        self.lower_val_: float = 0.0
        self.upper_val_: float = 0.0

    def fit(self, data: np.ndarray) -> "WinsorizationEngine":
        """Calculates value thresholds at the configured percentiles.

        Args:
            data (np.ndarray): Data array to extract percentiles from.
        """
        arr = np.asarray(data)
        if len(arr) == 0:
            self.lower_val_ = 0.0
            self.upper_val_ = 0.0
            return self

        self.lower_val_ = float(np.percentile(arr, self.lower * 100.0))
        self.upper_val_ = float(np.percentile(arr, self.upper * 100.0))
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Applies winsorization capping boundaries on the target data.

        Args:
            data (np.ndarray): Target data to transform.
        """
        arr = np.asarray(data).copy()
        if len(arr) == 0:
            return arr

        return np.clip(arr, self.lower_val_, self.upper_val_)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fits thresholds and applies winsorization transformation."""
        return self.fit(data).transform(data)
