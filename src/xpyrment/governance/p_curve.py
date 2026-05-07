"""P-Curve analysis for governance, power checks, and p-hacking detections.

This module provides the `PCurve` class to analyze distributions of significant
p-values (p < 0.05) for system-level reporting bias and true statistical power.
"""

from typing import Dict, Any, List
import numpy as np
from scipy import stats


class PCurve:
    """Analyzes significant p-value distributions to evaluate power and flag gaming/p-hacking."""

    def __init__(self, p_values: List[float]):
        """Initializes the PCurve analyzer.

        Args:
            p_values (List[float]): A list of all historical experiment p-values.
        """
        raw_p = np.array(p_values)
        # Select only statistically significant p-values (p < 0.05) as required by Simonsohn et al.
        self.significant_p = raw_p[(raw_p >= 0.0) & (raw_p < 0.05)]
        self.n_total = len(self.significant_p)

    def analyze(self) -> Dict[str, Any]:
        """Evaluates right-skewness (true power) and left-skewness (p-hacking / gaming).

        Returns:
            Dict[str, Any]: Counts, skewness indicators, test p-values, and warning alerts.
        """
        if self.n_total < 5:
            return {
                "n_significant": self.n_total,
                "status": "Inconclusive",
                "message": "Too few significant p-values to evaluate distribution skewness (minimum 5 required).",
                "p_right_skew": 1.0,
                "p_left_skew": 1.0,
            }

        # Count p-values in [0.0, 0.025]
        n_low = int(np.sum(self.significant_p <= 0.025))
        n_high = self.n_total - n_low

        # 1. Test for Right-Skewness (True Evidential Power)
        # H0: Significant p-values are uniformly distributed (p = 0.5 for low half).
        # H1: Distribution is right-skewed (proportion in low half > 0.5).
        p_right_skew = 1.0 - stats.binom.cdf(n_low - 1, self.n_total, 0.5)

        # 2. Test for Left-Skewness (p-hacking / selective stopping gaming)
        # H1: Distribution is left-skewed (proportion in high half > 0.5).
        p_left_skew = stats.binom.cdf(n_low, self.n_total, 0.5)

        # Determine status and alerts
        if p_left_skew < 0.05:
            status = "Reporting Bias / P-Hacking Detected"
            message = "The distribution of significant p-values is significantly left-skewed. This is a classic indicator of publication bias or selective peeking and stopping."
        elif p_right_skew < 0.05:
            status = "Strong Evidential Value"
            message = "The distribution of significant p-values is right-skewed, demonstrating high true statistical power and valid experimental effects."
        else:
            status = "Uniform / Flat (Lack of Power)"
            message = "The significant p-values are uniformly distributed, suggesting low evidential power or a high proportion of false positives."

        return {
            "n_significant": self.n_total,
            "n_low_half": n_low,
            "n_high_half": n_high,
            "ratio_low_to_high": float(n_low / n_high) if n_high > 0 else float("inf"),
            "p_right_skew": float(p_right_skew),
            "p_left_skew": float(p_left_skew),
            "status": status,
            "message": message,
        }
