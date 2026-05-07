"""Experimental result interpretation, practical significance, and product launch decision-making.

This package provides a suite of high-level diagnostic and decision tools to help experimenters
go beyond basic p-values, mapping raw statistical estimations directly to business utility and product decisions.

Submodules:
- `decision`: Integrates statistical and economic constraints to output structured launch recommendations.
- `effect_size`: Computes standardized scale-free effect sizes (e.g., Cohen's d).
- `hte`: Evaluates Heterogeneous Treatment Effects (HTE) and executes subgroup interaction screening.
- `significance`: Assesses whether observed lifts meet Minimum Valuable Effect (MVE) business targets.
"""

from xpyrment.interpret.decision import generate_launch_recommendation
from xpyrment.interpret.effect_size import compute_cohens_d
from xpyrment.interpret.hte import scan_subgroups_for_hte
from xpyrment.interpret.significance import check_practical_significance

__all__ = [
    "generate_launch_recommendation",
    "compute_cohens_d",
    "scan_subgroups_for_hte",
    "check_practical_significance",
]

