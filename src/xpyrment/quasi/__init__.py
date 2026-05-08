"""Quasi-experiments and matching methods when randomized trials are impossible.

Submodules:
- `diff_in_diff`: Difference-in-Differences regressions and trend validations.
- `synthetic_control`: Synthesizes virtual controls via quadratic programming.
- `sdid`: Synthetic Difference-in-Differences.
"""

from xpyrment.quasi.diff_in_diff import DifferenceInDifferences
from xpyrment.quasi.synthetic_control import SyntheticControl
from xpyrment.quasi.matching import PropensityScoreMatcher, CoarsenedExactMatcher, mahalanobis_distance
from xpyrment.quasi.sdid import SyntheticDifferenceInDifferences
from xpyrment.quasi.snmm import StructuralNestedMeanModel

__all__ = [
    "DifferenceInDifferences",
    "SyntheticControl",
    "PropensityScoreMatcher",
    "CoarsenedExactMatcher",
    "mahalanobis_distance",
    "SyntheticDifferenceInDifferences",
    "StructuralNestedMeanModel",
]
