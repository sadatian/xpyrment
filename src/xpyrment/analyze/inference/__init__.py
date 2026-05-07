"""Statistical inference engines, frameworks, and decision-making systems.

This package provides standard and state-of-the-art inferential frameworks for computing treatment
effects, confidence bounds, and decision probabilities.

Submodules:
- `frequentist`: Standard hypothesis tests including Welch's t-test and Mann-Whitney U rank sums.
- `bayesian`: Conjugate posterior models (Beta-Binomial, Normal-Normal) and expected loss decisions.
- `sequential`: Always-Valid Confidence Intervals (AVCIs) derived from mSPRT boundaries.
- `bootstrap`: Non-parametric percentile and BCa resampling estimators.
- `router`: Automated matching between metric structures, designs, and calculation engines.
"""

from xpyrment.analyze.inference.bootstrap import run_bootstrap_ci
from xpyrment.analyze.inference.bayesian import BayesianInference
from xpyrment.analyze.inference.frequentist import run_welch_t_test, run_mann_whitney_u
from xpyrment.analyze.inference.router import route_inference_engine
from xpyrment.analyze.inference.sequential import SequentialInference

__all__ = [
    "run_bootstrap_ci",
    "BayesianInference",
    "run_welch_t_test",
    "run_mann_whitney_u",
    "route_inference_engine",
    "SequentialInference",
]
