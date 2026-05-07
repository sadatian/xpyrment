"""Quasi-experiments and matching methods when randomized trials are impossible.

Submodules:
- `diff_in_diff`: Difference-in-Differences regressions and trend validations.
- `synthetic_control`: Synthesizes virtual controls via quadratic programming.
"""

from xpyrment.quasi.diff_in_diff import DifferenceInDifferences
from xpyrment.quasi.synthetic_control import SyntheticControl

__all__ = [
    "DifferenceInDifferences",
    "SyntheticControl",
]
