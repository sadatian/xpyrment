import json


class ExperimentCard:
    """Consumes metadata, planning state, and calculations to compile a unified report card."""

    def __init__(self, experiment_id: str, plan_spec: dict, validation_spec: dict, analysis_summary: dict):
        self.experiment_id = experiment_id
        self.plan_spec = plan_spec
        self.validation_spec = validation_spec
        self.analysis_summary = analysis_summary

    def to_dict(self) -> dict:
        """Serializes the experiment card metadata."""
        return {
            "experiment_id": self.experiment_id,
            "plan_spec": self.plan_spec,
            "validation_spec": self.validation_spec,
            "analysis_summary": self.analysis_summary
        }

    def to_json(self) -> str:
        """Dumps the card as a formatted JSON document."""
        return json.dumps(self.to_dict(), indent=2)
