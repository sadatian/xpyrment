"""Sample Ratio Mismatch (SRM) Detection & Sequential Guardrails (Block 41).

Implements retrospective Chi-Squared tests and sequential binomial likelihood ratio tests
(Wald's SPRT) to detect user assignment imbalances and selection biases.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
from scipy.stats import chi2


class SampleRatioMismatchDetector:
    """Detects Sample Ratio Mismatch (SRM) using frequentist retrospective and online sequential tests.

    TODO: Implement sequential binomial SPRT variance boundaries to handle dynamic sample-rate drift under multi-arm scenarios.
    TODO: Add Monte Carlo power estimation diagnostics for retrospective sample size mismatch sensitivity.
    """

    def __init__(self, target_treatment_ratio: float = 0.5) -> None:
        """Initializes the SRM detector.

        Args:
            target_treatment_ratio (float): Targeted assignment fraction for the treatment group. Defaults to 0.5.
        """
        if not (0.0 < target_treatment_ratio < 1.0):
            raise ValueError("Target treatment ratio must be strictly in the range (0, 1).")
        self.target_trt = target_treatment_ratio
        self.target_ctrl = 1.0 - target_treatment_ratio

    def test_retrospective(self, observed_counts: Tuple[int, int]) -> Dict[str, Union[float, bool]]:
        """Performs a retrospective Pearson Chi-Squared goodness-of-fit test.

        Args:
            observed_counts (Tuple[int, int]): A tuple of (observed_control, observed_treatment).

        Returns:
            Dict[str, Union[float, bool]]: SRM test statistics, p-value, and flag indicating mismatch.
        """
        obs_ctrl, obs_trt = observed_counts
        if obs_ctrl < 0 or obs_trt < 0:
            raise ValueError("Observed counts must be non-negative integers.")
        
        N = obs_ctrl + obs_trt
        if N == 0:
            return {
                "chi_squared_statistic": 0.0,
                "p_value": 1.0,
                "srm_detected": False,
            }

        # Expected counts
        exp_ctrl = N * self.target_ctrl
        exp_trt = N * self.target_trt

        # Pearson Chi-Squared formula
        chi_sq = ((obs_ctrl - exp_ctrl) ** 2) / exp_ctrl + ((obs_trt - exp_trt) ** 2) / exp_trt
        
        # 1 Degree of Freedom for a 2-class goodness-of-fit test
        p_val = float(1.0 - chi2.cdf(chi_sq, df=1))

        # SRM is conventionally flagged at highly conservative significance levels (e.g. alpha = 0.001)
        srm_detected = p_val < 0.001

        return {
            "chi_squared_statistic": float(chi_sq),
            "p_value": p_val,
            "srm_detected": bool(srm_detected),
        }

    def test_sequential(self, assignments: np.ndarray, delta: float = 0.02, alpha: float = 0.01) -> Dict[str, Union[np.ndarray, bool]]:
        """Performs Wald's Sequential Probability Ratio Test (SPRT) over binomial allocations.

        Determines if the running allocation ratio departs significantly from target ratio.
        Null Hypothesis (H_0): p = target_treatment_ratio
        Alternative Hypotheses (H_1): p = target_treatment_ratio + delta OR target_treatment_ratio - delta

        Returns:
            Dict[str, Union[np.ndarray, bool]]: Running likelihood ratios and stopping decisions.
        """
        if delta <= 0.0 or self.target_trt + delta >= 1.0 or self.target_trt - delta <= 0.0:
            raise ValueError("Delta must satisfy 0 < target_treatment_ratio +/- delta < 1.")

        # Convert assignments to binary array where 1 = treatment, 0 = control
        arr = np.array(assignments, dtype=int)
        N = len(arr)
        if N == 0:
            return {
                "running_likelihood_ratios": np.array([]),
                "srm_detected": False,
                "stopped_index": -1,
            }

        # Define bounds for Wald's test
        # Accept H_0 if LR <= beta / (1 - alpha) [not stopping for SRM acceptance]
        # Reject H_0 if LR >= (1 - beta) / alpha ~ 1 / alpha (frequentist Type I bound)
        upper_threshold = 1.0 / alpha

        p_0 = self.target_trt
        p_alt_high = p_0 + delta
        p_alt_low = p_0 - delta

        # Precompute log-likelihood ratios for an assignment:
        # log_lr = x * ln(p_alt / p_0) + (1 - x) * ln((1 - p_alt) / (1 - p_0))
        log_ratio_high_1 = np.log(p_alt_high / p_0)
        log_ratio_high_0 = np.log((1.0 - p_alt_high) / (1.0 - p_0))

        log_ratio_low_1 = np.log(p_alt_low / p_0)
        log_ratio_low_0 = np.log((1.0 - p_alt_low) / (1.0 - p_0))

        running_log_lr_high = 0.0
        running_log_lr_low = 0.0
        
        lr_series = []
        srm_detected = False
        stopped_idx = -1

        for idx, val in enumerate(arr):
            if val == 1:
                running_log_lr_high += log_ratio_high_1
                running_log_lr_low += log_ratio_low_1
            else:
                running_log_lr_high += log_ratio_high_0
                running_log_lr_low += log_ratio_low_0

            # Two-sided mixture of likelihood ratios
            lr_mixed = 0.5 * np.exp(running_log_lr_high) + 0.5 * np.exp(running_log_lr_low)
            lr_series.append(lr_mixed)

            if lr_mixed >= upper_threshold and not srm_detected:
                srm_detected = True
                stopped_idx = idx

        return {
            "running_likelihood_ratios": np.array(lr_series),
            "srm_detected": srm_detected,
            "stopped_index": stopped_idx,
        }
