import hashlib
import json
from typing import Any, Dict


class ExperimentRegistry:
    """Manages immutable experiment specifications to prevent post-hoc changes (pre-registration)."""

    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register_spec(self, experiment_id: str, spec_dict: Dict[str, Any]) -> str:
        """Serializes the experiment specification, hashes it, and stores it in the registry.

        Args:
            experiment_id (str): Unique ID of the experiment.
            spec_dict (Dict[str, Any]): Spec configurations.

        Returns:
            str: SHA-256 signature hash of the pre-registration spec.
        """
        serialized = json.dumps(spec_dict, sort_keys=True)
        spec_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        self._registry[experiment_id] = {
            "spec": spec_dict,
            "hash": spec_hash,
        }
        return spec_hash

    def verify_spec(self, experiment_id: str, spec_dict: Dict[str, Any]) -> bool:
        """Verifies if the spec_dict matches the registered hash to prevent p-hacking.

        Args:
            experiment_id (str): Registered ID.
            spec_dict (Dict[str, Any]): Current Spec configuration.

        Returns:
            bool: True if verified, False if there is a mismatch.
        """
        if experiment_id not in self._registry:
            return False

        serialized = json.dumps(spec_dict, sort_keys=True)
        current_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        return current_hash == self._registry[experiment_id]["hash"]
