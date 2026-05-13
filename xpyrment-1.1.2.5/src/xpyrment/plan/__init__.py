"""Experiment planning, power analysis, duration estimation, and preregistration.

This package houses components dedicated to the planning stage of the experimental lifecycle,
helping experimenters define scientific/business hypotheses, calculate necessary sample sizes
and run runtimes, and lock plans via cryptographic preregistration.

Submodules:
- `hypothesis`: Forms HypothesisSpec containers mapping outcomes to statistical directions.
- `power`: Handles a priori statistical power calculations and sample sizing.
- `duration`: Maps required sample sizes to calendar run durations.
- `preregistration`: Issues immutable PreregistrationCards to protect analysis integrity.
"""

from xpyrment.plan.duration import estimate_duration_days
from xpyrment.plan.hypothesis import HypothesisSpec
from xpyrment.plan.power import ExperimentDesignResult, design_experiment, generate_power_curve_data
from xpyrment.plan.preregistration import PreregistrationCard

__all__ = [
    "HypothesisSpec",
    "ExperimentDesignResult",
    "design_experiment",
    "generate_power_curve_data",
    "estimate_duration_days",
    "PreregistrationCard",
]
