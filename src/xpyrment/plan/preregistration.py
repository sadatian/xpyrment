"""Preregistration specifications to prevent retrospective study optimization.

This module provides the `PreregistrationCard` representation. This card acts as a formalized,
cryptographically signed contract of the experimental parameters, verifying that the actual analysis
adheres strictly to the pre-planned setup (metrics, splits, sample thresholds).
"""

import json
from typing import Dict, Any
from xpyrment.core.registry import ExperimentRegistry


class PreregistrationCard:
    """Represents an immutable spec card registered prior to running the experiment.

    A `PreregistrationCard` bundles the complete configurations of an experiment (such as primary and
    secondary metrics, target statistical power, significance thresholds, and sample sizes) and
    secures them against post-hoc tampering by computing a SHA-256 signature via the `ExperimentRegistry`.

    Why Pre-registration Matters:
        In both clinical trials and industrial A/B testing, a common failure mode is post-hoc
        tampering (p-hacking, retrospective metric selection, or adjusting significance thresholds mid-run).
        A pre-registered card acts as an immutable ledger. Before final reports are compiled,
        the system validates the active analysis configuration against this card's signature.

    Attributes:
        experiment_id (str): Unique tracking identifier for the experiment.
        spec (Dict[str, Any]): Structural dictionary outlining the planned experimental parameters.
        hash_signature (str): Cryptographic SHA-256 hash representing the serialized `spec` dictionary.

    Examples:
        ??? example "Examples"

            ```python
            >>> spec = {"metric": "conversion_rate", "alpha": 0.05, "target_n": 10000}
            >>> card = PreregistrationCard("EXP-999", spec)
            >>> card.hash_signature[:8]
            '88e6e885'
            >>> card.verify({"metric": "conversion_rate", "alpha": 0.05, "target_n": 10000})
            True
            >>> card.verify({"metric": "conversion_rate", "alpha": 0.10, "target_n": 10000}) # alpha altered!
            False
            ```
    """

    def __init__(self, experiment_id: str, spec: Dict[str, Any]):
        """Initializes a new PreregistrationCard and registers its specification.

        Args:
            experiment_id (str): Unique tracking identifier for the experiment.
            spec (Dict[str, Any]): Dictionary containing complete design parameters
                to be registered.
        """
        self.experiment_id = experiment_id
        self.spec = spec
        self._registry = ExperimentRegistry()
        self.hash_signature = self._registry.register_spec(experiment_id, spec)

    def verify(self, current_spec: Dict[str, Any]) -> bool:
        """Verifies if the current spec matches the registered immutable signature.

        Compares the provided specification dictionary against the original registered spec.

        Args:
            current_spec (Dict[str, Any]): The operational parameters dictionary currently under execution.

        Returns:
            bool: True if the current specification matches the pre-registered specification exactly,
                False if there is any mismatch or if the card was not properly recorded.
        """
        return self._registry.verify_spec(self.experiment_id, current_spec)

    def to_json(self) -> str:
        """Serializes the preregistration card parameters and cryptographic signature to JSON.

        Returns:
            str: Indented, human-readable JSON string representing the card metadata.
        """
        return json.dumps({
            "experiment_id": self.experiment_id,
            "spec": self.spec,
            "signature": self.hash_signature
        }, indent=2)
