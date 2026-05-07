"""Governance, Meta-analysis, and P-curve auditing.

Submodules:
- `meta_analysis`: Fixed and Random effects meta-pooling.
- `p_curve`: Evidential value and publication bias auditing.
"""

from xpyrment.governance.meta_analysis import MetaAnalysis
from xpyrment.governance.p_curve import PCurve

__all__ = [
    "MetaAnalysis",
    "PCurve",
]
