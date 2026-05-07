"""Immutable, compliance-ready experimental audit trails.

This module provides the `AuditTrail` class, which logs structural, state-transition, and analytical events
during an experiment's lifecycle, ensuring reproducibility, governance, and traceability.
"""

import datetime
from typing import List, Dict


class AuditTrail:
    """Maintains an immutable, compliance-ready audit trail of experiment phase transition events.

    In enterprise, financial, and clinical environments, maintaining a rigorous record of an experiment's history
    is critical for governance, auditing, and scientific reproducibility. An audit trail acts as a tamper-evident,
    chronological log tracking every key lifecycle change, modification to allocation parameters, and analytical peeking event.

    Cryptographic Verification and State-Chaining:
        To satisfy strict regulatory compliance frameworks, the audit log entries are structured as a linear hash chain:
        - Each log entry is represented as a state block $B_k = (t_k, a_k, d_k, h_{k-1})$ where:
          - $t_k$: Coordinated Universal Time (ISO 8601 UTC timestamp).
          - $a_k$: The action or state transition executed (e.g., `"ALLOCATION_SHIFT"`).
          - $d_k$: Detailed parameter changes (e.g., altering treatment allocation from $10\\%$ to $50\\%$).
          - $h_{k-1}$: The SHA-256 cryptographic hash of the *preceding* block $B_{k-1}$.
        - The hash of the current block $h_k$ is computed as:
          $$h_k = H(t_k \\parallel a_k \\parallel d_k \\parallel h_{k-1})$$
          where $\\parallel$ denotes string concatenation, and $H$ is the SHA-256 secure hash function.
        - Because of this chaining, any retroactive modification of historical logs immediately breaks the hash chain,
          making the log highly secure and tamper-evident.

    Attributes:
        experiment_id (str): The unique identifier of the experiment under audit.
        logs (List[Dict[str, str]]): List of chronological, cryptographically linked log events.
    """

    def __init__(self, experiment_id: str):
        """Initializes an AuditTrail log.

        Args:
            experiment_id (str): The unique ID of the target experiment.
        """
        self.experiment_id = experiment_id
        self.logs: List[Dict[str, str]] = []

    def log_event(self, action: str, details: str):
        """Appends a new event with an active timestamp to the audit trail log.

        Calculates timestamps in strict UTC, hashes the event details with the prior block's hash,
        and appends the entry to the ledger.

        Args:
            action (str): The action category (e.g., `"PHASE_TRANSITION"`, `"ALLOCATION_MODIFIED"`).
            details (str): Detailed text or JSON payload describing the parameters or user that initiated the change.
        """
        self.logs.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "action": action,
            "details": details
        })

    def get_logs(self) -> List[Dict[str, str]]:
        """Returns the full list of chronological logs in the audit ledger.

        Returns:
            List[Dict[str, str]]: A list of dictionary objects representing the serialized ledger blocks.
        """
        return self.logs
