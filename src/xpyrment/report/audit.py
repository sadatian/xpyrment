"""Immutable, compliance-ready experimental audit trails.

This module provides the `AuditTrail` class, which logs structural, state-transition, and analytical events
during an experiment's lifecycle, ensuring reproducibility, governance, and traceability.
"""

import datetime
from typing import List, Dict


class AuditTrail:
    r"""Maintains an immutable, compliance-ready audit trail of experiment phase transition events.

    In enterprise, financial, and clinical environments, maintaining a rigorous record of an experiment's history
    is critical for governance, auditing, and scientific reproducibility. An audit trail acts as a tamper-evident,
    chronological log tracking every key lifecycle change, modification to allocation parameters, and analytical peeking event.

    ??? mathbox "Cryptographic Verification and State-Chaining"

        To satisfy strict regulatory compliance frameworks, the audit log entries are structured as a linear hash chain:
        - Each log entry is represented as a state block $B_k = (t_k, a_k, d_k, h_{k-1})$ where:
          - $t_k$: Coordinated Universal Time (ISO 8601 UTC timestamp).
          - $a_k$: The action or state transition executed (e.g., `"ALLOCATION_SHIFT"`).
          - $d_k$: Detailed parameter changes (e.g., altering treatment allocation from $10\\%$ to $50\\%$).
          - $h_{k-1}$: The SHA-256 cryptographic hash of the *preceding* block $B_{k-1}$.
        - The hash of the current block $h_k$ is computed as:
          $$
          h_k = H(t_k \\parallel a_k \\parallel d_k \\parallel h_{k-1})
          $$
          where $\\parallel$ denotes string concatenation, and $H$ is the SHA-256 secure hash function.
        - Because of this chaining, any retroactive modification of historical logs immediately breaks the hash chain,
          making the log highly secure and tamper-evident.

    Attributes:
        experiment_id (str): The unique identifier of the experiment under audit.
        logs (List[Dict[str, str]]): List of chronological, cryptographically linked log events.
    """

    def __init__(self, experiment_id: str, db_path: str = None) -> None:
        """Initializes an AuditTrail log.

        Args:
            experiment_id (str): The unique ID of the target experiment.
            db_path (str, optional): Optional SQLite database path for tamper-proof persistence.
        """
        self.experiment_id = experiment_id
        self.db_path = db_path
        self.logs: List[Dict[str, str]] = []

        if self.db_path:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''CREATE TABLE IF NOT EXISTS audit_logs (
                        experiment_id TEXT,
                        timestamp TEXT,
                        action TEXT,
                        details TEXT,
                        prev_hash TEXT,
                        hash TEXT,
                        signature TEXT,
                        public_key TEXT
                    )'''
                )
                conn.commit()

    def log_event(self, action: str, details: str, signature: str = None, public_key: str = None) -> None:
        """Appends a new event with an active timestamp to the audit trail log.

        Calculates timestamps in strict UTC, hashes the event details with the prior block's hash,
        and appends the entry to the ledger. Optionally attaches RSA/ECDSA digital signatures.

        Args:
            action (str): The action category (e.g., `"PHASE_TRANSITION"`, `"ALLOCATION_MODIFIED"`).
            details (str): Detailed text or JSON payload describing the parameters or user that initiated the change.
            signature (str, optional): Cryptographic signature of the event hash.
            public_key (str, optional): Public key string to verify the signature.
        """
        import hashlib

        timestamp = datetime.datetime.now(datetime.UTC).isoformat()
        prev_hash = "0" * 64 if len(self.logs) == 0 else self.logs[-1]["hash"]

        safe_signature = signature or ""
        safe_public_key = public_key or ""
        # Formulate canonical block string for SHA-256 hashing
        data_str = f"{timestamp}||{action}||{details}||{prev_hash}||{safe_signature}||{safe_public_key}"
        current_hash = hashlib.sha256(data_str.encode("utf-8")).hexdigest()

        log_entry = {
            "timestamp": timestamp,
            "action": action,
            "details": details,
            "prev_hash": prev_hash,
            "hash": current_hash,
            "signature": safe_signature,
            "public_key": safe_public_key
        }
        self.logs.append(log_entry)

        if self.db_path:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''INSERT INTO audit_logs (experiment_id, timestamp, action, details, prev_hash, hash, signature, public_key)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                    (self.experiment_id, timestamp, action, details, prev_hash, current_hash, log_entry["signature"], log_entry["public_key"])
                )
                conn.commit()

    def verify_integrity(self) -> bool:
        """Verifies the complete cryptographic chain of the audit trail ledger.

        Returns:
            bool: True if the hash chain is fully intact and unmodified, False otherwise.
        """
        import hashlib

        for i in range(len(self.logs)):
            block = self.logs[i]
            expected_prev = "0" * 64 if i == 0 else self.logs[i-1]["hash"]

            if block["prev_hash"] != expected_prev:
                return False

            # Recalculate block hash
            data_str = f"{block['timestamp']}||{block['action']}||{block['details']}||{block['prev_hash']}||{block.get('signature', '')}||{block.get('public_key', '')}"
            actual_hash = hashlib.sha256(data_str.encode("utf-8")).hexdigest()

            if block["hash"] != actual_hash:
                return False

        return True

    def get_logs(self) -> List[Dict[str, str]]:
        """Returns the full list of chronological logs in the audit ledger.

        Returns:
            List[Dict[str, str]]: A list of dictionary objects representing the serialized ledger blocks.
        """
        return self.logs
