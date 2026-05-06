import pandas as pd


class AssignmentLogger:
    """Manages tracking live assignment exposures and deduplicating multiple logs."""

    def __init__(self):
        self._exposures = []

    def log_assignment(self, unit_id: str, variant: str, timestamp: str):
        """Logs an assignment exposure."""
        self._exposures.append({
            "unit_id": unit_id,
            "variant": variant,
            "timestamp": timestamp
        })

    def get_deduplicated_exposures(self) -> pd.DataFrame:
        """Deduplicates multiple logs of the same unit_id, taking the earliest exposure."""
        df = pd.DataFrame(self._exposures)
        if df.empty:
            return df
        return df.sort_values("timestamp").drop_duplicates(subset=["unit_id"], keep="first")
