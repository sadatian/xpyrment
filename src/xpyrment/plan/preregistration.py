import json
from typing import Dict, Any
from xpyrment.core.registry import ExperimentRegistry


class PreregistrationCard:
    """Represents an immutable spec card registered prior to running the experiment."""

    def __init__(self, experiment_id: str, spec: Dict[str, Any]):
        self.experiment_id = experiment_id
        self.spec = spec
        self._registry = ExperimentRegistry()
        self.hash_signature = self._registry.register_spec(experiment_id, spec)

    def verify(self, current_spec: Dict[str, Any]) -> bool:
        """Verifies if the current spec matches the registered immutable signature."""
        return self._registry.verify_spec(self.experiment_id, current_spec)

    def to_json(self) -> str:
        """Returns registered spec and signature as JSON."""
        return json.dumps({
            "experiment_id": self.experiment_id,
            "spec": self.spec,
            "signature": self.hash_signature
        }, indent=2)
