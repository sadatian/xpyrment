from xpyrment.core.experiment import Experiment
from xpyrment.analyze.orchestrator import AnalysisResult, run_analysis, setup
from xpyrment.analyze.variance_reduction import apply_cuped
from xpyrment.analyze.corrections import apply_multiple_testing_correction
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
    "inference",
]
