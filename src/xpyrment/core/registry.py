"""Preregistration registry for locking and verifying experiment specifications.

This module provides the cryptographic registry used to record, lock, and verify the structural
specifications (hypotheses, primary metrics, target power, etc.) of an experiment before running it.
Pre-registration of experiment plans is a vital safeguard against p-hacking, hindsight bias, and
unauthorized post-hoc changes to experimental parameters.
"""

import hashlib
import json
from typing import Any, Dict


class ExperimentRegistry:
    """Manages immutable experiment specifications to prevent post-hoc changes (pre-registration).

    By calculating a cryptographic SHA-256 signature of serialized, key-sorted experiment
    specifications, the registry provides an audit trail. Analysts can verify that the running
    parameters (such as target sample sizes, significance levels, and chosen metrics) precisely
    match the registered plan, preventing retrospective optimization of analysis parameters.

    Attributes:
        _registry (Dict[str, Dict[str, Any]]): Internal store mapping experiment IDs to their
            registered specification dictionaries and pre-computed hashes.

    Examples:
        ??? example "Example"

            ```python
            >>> registry = ExperimentRegistry()
            >>> spec = {"primary_metric": "conversion_rate", "alpha": 0.05, "target_n": 10000}
            >>> spec_hash = registry.register_spec("EXP-101", spec)
            >>> len(spec_hash)
            64
            >>> registry.verify_spec("EXP-101", spec)
            True
            >>> modified_spec = {"primary_metric": "conversion_rate", "alpha": 0.10, "target_n": 10000}
            >>> registry.verify_spec("EXP-101", modified_spec)
            False
            ```
    """

    def __init__(self):
        """Initializes an empty registry store."""
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register_spec(self, experiment_id: str, spec_dict: Dict[str, Any]) -> str:
        r"""Serializes the experiment specification, hashes it, and stores it in the registry.

        Ensures that dictionaries are serialized with sorted keys to maintain deterministic
        hashing across systems, irrespective of key-insertion order.

        Mathematical Representation:
            Let $S$ be the key-sorted, compact JSON serialization of `spec_dict` encoded in UTF-8.
            The registered hash $H$ is:
            $$
            H = \text{SHA256}(S)
            $$
        Args:
            experiment_id (str): Unique identifier of the experiment.
            spec_dict (Dict[str, Any]): Structural parameters representing the experiment plan,
                including registered metrics, statistical thresholds ($\alpha, \beta$), and design configurations.

        Returns:
            str: The hexadecimal representation of the SHA-256 signature hash.
        """
        serialized = json.dumps(spec_dict, sort_keys=True)
        spec_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        self._registry[experiment_id] = {
            "spec": spec_dict,
            "hash": spec_hash,
        }
        return spec_hash

    def verify_spec(self, experiment_id: str, spec_dict: Dict[str, Any]) -> bool:
        """Verifies if the current spec_dict matches the registered hash to prevent p-hacking.

        Re-hashes the incoming specification dictionary using key-sorted serialization and performs
        a constant-time comparison against the stored hash for the given experiment ID.

        Args:
            experiment_id (str): Registered ID of the experiment to verify.
            spec_dict (Dict[str, Any]): The active specification dictionary to validate.

        Returns:
            bool: True if the current specification matches the pre-registered specification exactly,
                False if there is a mismatch or if the experiment ID was never registered.
        """
        if experiment_id not in self._registry:
            return False

        serialized = json.dumps(spec_dict, sort_keys=True)
        current_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        return current_hash == self._registry[experiment_id]["hash"]
