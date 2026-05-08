"""Quasi-experiments and matching methods when randomized trials are impossible.

Submodules:
- `diff_in_diff`: Difference-in-Differences regressions and trend validations.
- `synthetic_control`: Synthesizes virtual controls via quadratic programming.
- `sdid`: Synthetic Difference-in-Differences.
"""

from xpyrment.quasi.diff_in_diff import DifferenceInDifferences, ParallelTrendsPlaceboTest
from xpyrment.quasi.synthetic_control import SyntheticControl
from xpyrment.quasi.matching import PropensityScoreMatcher, CoarsenedExactMatcher, mahalanobis_distance
from xpyrment.quasi.sdid import SyntheticDifferenceInDifferences
from xpyrment.quasi.snmm import StructuralNestedMeanModel
from xpyrment.quasi.matrix_completion import PanelMatrixCompletion
from xpyrment.quasi.sensitivity import CausalSensitivityAnalyzer
from xpyrment.quasi.optimal_transport import QuantileOptimalTransport
from xpyrment.quasi.instrumental_variables import InstrumentalVariables2SLS
from xpyrment.quasi.rolling_synthetic_control import RollingSyntheticControl
from xpyrment.quasi.balance import CovariateBalanceChecker

__all__ = [
    "DifferenceInDifferences",
    "ParallelTrendsPlaceboTest",
    "SyntheticControl",
    "PropensityScoreMatcher",
    "CoarsenedExactMatcher",
    "mahalanobis_distance",
    "SyntheticDifferenceInDifferences",
    "StructuralNestedMeanModel",
    "PanelMatrixCompletion",
    "CausalSensitivityAnalyzer",
    "QuantileOptimalTransport",
    "InstrumentalVariables2SLS",
    "RollingSyntheticControl",
    "CovariateBalanceChecker",
]
