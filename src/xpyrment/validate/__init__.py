"""Experiment diagnostics, sanity checks, and validation engines.

This package houses the diagnostic layer of `xpyrment`. It provides automated safeguards to
validate experiment execution, ensuring that results are not corrupted by assignment imbalances,
system bugs, or temporary behavioral anomalies.

Submodules:
- `srm`: Detects Sample Ratio Mismatch (SRM) using Pearson Chi-Square Goodness-of-Fit tests.
- `aa_test`: Simulates A/A tests and validates Type I error rate ($\alpha$) uniformity.
- `balance`: Computes Standardized Mean Differences (SMD) to evaluate pre-period covariate balance.
- `novelty`: Identifies novelty and primacy effects using temporal interaction models.
"""

from xpyrment.validate.aa_test import run_aa_test_validation
from xpyrment.validate.balance import check_covariate_balance
from xpyrment.validate.novelty import check_novelty_effects
from xpyrment.validate.srm import check_srm

__all__ = [
    "check_srm",
    "run_aa_test_validation",
    "check_covariate_balance",
    "check_novelty_effects",
]
