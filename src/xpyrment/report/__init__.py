"""Experimental reporting, lifecycle tracking, and scientific visualization.

This package provides logging, recording, and charting tools to summarize experimental phases.
It ensures that setups, runtime quality checks, and analytical inferences are aggregated and
presented in standard-compliant, publication-ready formats.

Submodules:
- `card`: Compiles standard, machine-readable Experiment Cards for metadata cataloging.
- `audit`: Logs and chains immutable lifecycle events for audit and governance compliance.
- `export`: Generates high-quality statistical plots (Forest plots, Power curves).
"""

from xpyrment.report.card import ExperimentCard
from xpyrment.report.audit import AuditTrail
from xpyrment.report.export import plot_forest, plot_power_curve
from xpyrment.report.generator import ExperimentReportGenerator

__all__ = [
    "ExperimentCard",
    "AuditTrail",
    "plot_forest",
    "plot_power_curve",
    "ExperimentReportGenerator",
]
