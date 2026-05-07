"""Unified, standard-compliant Experiment Cards and meta-ledgers.

This module provides the `ExperimentCard` class, which serves as a structured, human-and-machine-readable
"birth certificate" and cumulative registry file for an experiment, aggregating all planning, validation,
and final analytical outcomes.
"""

import json


class ExperimentCard:
    """Consumes metadata, planning state, and calculations to compile a unified report card.

    An Experiment Card (inspired by Model Cards, Mitchell et al. 2019) is the definitive, unified record
    and metadata registry of an experiment. It acts as a standardized document that records the design, execution,
    and results of an experiment in a machine-readable format. This makes it possible to search, catalog, and run
    large-scale meta-analyses across thousands of past experiments (e.g., tracking cumulative lift, estimating
    p-value distributions, or measuring historical power).

    The Experiment Card schema unifies three core lifecycle stages:
        1. **Planning & Setup Specification** (`plan_spec`):
           - `mde`: Minimum Detectable Effect (relative or absolute).
           - `alpha`: Nominal Type I error rate (e.g., $0.05$).
           - `power`: Target statistical power ($1 - \\beta$, e.g., $0.80$).
           - `target_sample_size`: Calculated sample size requirement.
           - `metric_registry`: Names and types of registered primary, secondary, and guardrail metrics.
        2. **Runtime Validation & Diagnostics** (`validation_spec`):
           - `srm_p_value`: Pearson Chi-Square goodness-of-fit p-value for sample allocation ratio mismatches.
           - `covariate_balance`: Standardized Mean Differences (SMDs) confirming unbiased random assignments.
        3. **Statistical Analysis Outcomes** (`analysis_summary`):
           - `treatment_effect`: Relative and absolute lifts, standard errors, and confidence intervals.
           - `p_values`: Observed p-values (with any multiple-testing adjustments applied).
           - `recommendation`: Automated decision outcome (e.g., `"SHIP"`, `"NO-SHIP"`, `"INCONCLUSIVE"`).

    Attributes:
        experiment_id (str): Unique tracking identifier for the experiment.
        plan_spec (dict): Setup configurations, expected metrics, and calculated power parameters.
        validation_spec (dict): Summary of SRM and covariate balance diagnostics.
        analysis_summary (dict): Calculated point estimates, confidence intervals, and launch recommendations.
    """

    def __init__(self, experiment_id: str, plan_spec: dict, validation_spec: dict, analysis_summary: dict):
        """Initializes a new ExperimentCard.

        Args:
            experiment_id (str): The unique ID of the experiment.
            plan_spec (dict): Setup and design characteristics dictionary.
            validation_spec (dict): Pre-analysis quality check outcomes.
            analysis_summary (dict): Post-analysis statistical summaries.
        """
        self.experiment_id = experiment_id
        self.plan_spec = plan_spec
        self.validation_spec = validation_spec
        self.analysis_summary = analysis_summary

    def to_dict(self) -> dict:
        """Serializes the experiment card metadata to a standard python dictionary.

        Returns:
            dict: The nested dictionary of card metadata.
        """
        return {
            "experiment_id": self.experiment_id,
            "plan_spec": self.plan_spec,
            "validation_spec": self.validation_spec,
            "analysis_summary": self.analysis_summary
        }

    def to_json(self) -> str:
        """Dumps the card as a formatted JSON document.

        Returns:
            str: Indented, pretty-printed JSON string of the complete experiment card ledger.
        """
        return json.dumps(self.to_dict(), indent=2)

