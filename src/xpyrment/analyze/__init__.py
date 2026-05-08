"""Experiment analysis, variance reduction, multiple testing corrections, and statistical inference.

This package houses the core statistical analysis engine of `xpyrment`. It coordinates the calculation of treatment
effects, computes variance-reduced adjusted statistics, corrects for multiple simultaneous comparisons, and
provides a modular statistical inference suite (frequentist, bayesian, sequential, and bootstrap).

Fluent API Integration:
    To preserve elegant, object-oriented fluent API chaining, this package dynamically registers the
    `run_analysis` method on the main `Experiment` orchestrator class when imported:
    ```python
    # Equivalent to:
    result = setup(data, "variant").run_analysis()
    ```
"""

from xpyrment.core.experiment import Experiment
from xpyrment.analyze.orchestrator import AnalysisResult, run_analysis, setup
from xpyrment.analyze.variance_reduction import apply_cuped
from xpyrment.analyze.corrections import apply_multiple_testing_correction
from xpyrment.analyze.streaming import StreamingOLS
from xpyrment.analyze.confounding import AliasResolver
from xpyrment.analyze.copula import CopulaMultiMetricInference
from xpyrment.analyze.markov import MarkovJourneyAnalyzer
from xpyrment.analyze.extreme import ExtremeValueTailEstimator
from xpyrment.analyze.its import InterruptedTimeSeries
from xpyrment.analyze.sequential import GroupSequentialMonitor
from xpyrment.analyze.meta_regression import MetaRegressor
from xpyrment.analyze.srm import SampleRatioMismatchDetector
from xpyrment.analyze.outliers import WinsorizationEngine
from xpyrment.analyze.ratio import RatioMetricDeltaMethod
from xpyrment.analyze.registry import MetricRegistry
from xpyrment.analyze import inference

# Dynamically inject run_analysis method to the Experiment container
# to preserve fluent method chaining without circular static imports.
if not hasattr(Experiment, "run_analysis"):
    def _run_analysis_method(self, *args, **kwargs):
        return run_analysis(self, *args, **kwargs)
    Experiment.run_analysis = _run_analysis_method

__all__ = [
    "AnalysisResult",
    "run_analysis",
    "setup",
    "apply_cuped",
    "apply_multiple_testing_correction",
    "StreamingOLS",
    "AliasResolver",
    "CopulaMultiMetricInference",
    "MarkovJourneyAnalyzer",
    "ExtremeValueTailEstimator",
    "InterruptedTimeSeries",
    "GroupSequentialMonitor",
    "MetaRegressor",
    "SampleRatioMismatchDetector",
    "WinsorizationEngine",
    "RatioMetricDeltaMethod",
    "MetricRegistry",
    "inference",
]
