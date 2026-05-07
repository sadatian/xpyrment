"""Live experiment runtime execution, data ingestion, monitoring, and sequential stopping.

This package manages the execution phase of `xpyrment`. It provides the operational scaffolding for
ingesting raw telemetry datasets, logging and deduplicating variant exposures, monitoring cumulative allocations
in real-time, and evaluating continuous early-stopping rules.

Submodules:
- `ingestion`: Connectors and schema gates to import SQL datasets or pandas DataFrames.
- `assignment`: Records variant exposures, enforcing first-touch attribution for causal safety.
- `monitor`: Aggregates active traffic into binned time-series metrics for real-time dashboards and audits.
- `stopping`: Implements mixture Sequential Probability Ratio Tests (mSPRTs) for continuous peeking and safe early-stopping.
"""

from xpyrment.run.assignment import AssignmentLogger
from xpyrment.run.ingestion import ingest_dataframe, load_from_sql
from xpyrment.run.monitor import LiveMonitor
from xpyrment.run.stopping import StoppingRules

__all__ = [
    "AssignmentLogger",
    "ingest_dataframe",
    "load_from_sql",
    "LiveMonitor",
    "StoppingRules",
]
