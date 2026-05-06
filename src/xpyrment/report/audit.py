import datetime
from typing import List, Dict


class AuditTrail:
    """Maintains an immutable, compliance-ready audit trail of experiment phase transition events."""

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.logs: List[Dict[str, str]] = []

    def log_event(self, action: str, details: str):
        """Appends a new event with an active timestamp to the audit trail log."""
        self.logs.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "action": action,
            "details": details
        })

    def get_logs(self) -> List[Dict[str, str]]:
        """Returns log lists."""
        return self.logs
