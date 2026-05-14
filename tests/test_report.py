import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from xpyrment.report.audit import AuditTrail
from xpyrment.report.card import ExperimentCard
from xpyrment.report.export import plot_forest, plot_power_curve


def test_experiment_card_serialization():
    """Validates metadata structure, dictionary serialization, and JSON outputs of ExperimentCard."""
    plan_spec = {
        "mde": 0.05,
        "alpha": 0.05,
        "power": 0.80,
        "target_sample_size": 1250,
        "metric_registry": ["revenue", "conversion"]
    }
    validation_spec = {
        "srm_p_value": 0.45,
        "covariate_balance": "PASS"
    }
    analysis_summary = {
        "treatment_effect": 0.042,
        "recommendation": "SHIP"
    }

    card = ExperimentCard(
        experiment_id="EXP-101",
        plan_spec=plan_spec,
        validation_spec=validation_spec,
        analysis_summary=analysis_summary
    )

    # Dictionary Serialization
    card_dict = card.to_dict()
    assert card_dict["experiment_id"] == "EXP-101"
    assert card_dict["plan_spec"]["mde"] == 0.05
    assert card_dict["validation_spec"]["srm_p_value"] == 0.45
    assert card_dict["analysis_summary"]["recommendation"] == "SHIP"

    # JSON Formatting & Schema
    card_json = card.to_json()
    parsed_json = json.loads(card_json)
    assert parsed_json["experiment_id"] == "EXP-101"
    assert "plan_spec" in parsed_json


def test_audit_trail_cryptographic_chain():
    """Validates that AuditTrail builds an immutable, secure, SHA-256 chain and detects tampering."""
    trail = AuditTrail("EXP-101")

    # 1. Log multiple chronological event states
    trail.log_event("SETUP", "Experiment created with 50/50 allocation.")
    trail.log_event("PHASE_TRANSITION", "Advanced state to RUNNING.")
    trail.log_event("METRIC_ADDITION", "Added Revenue primary metric.")

    logs = trail.get_logs()
    assert len(logs) == 3

    # 2. Assert SHA-256 Chaining structure
    assert logs[0]["prev_hash"] == "0" * 64
    assert logs[1]["prev_hash"] == logs[0]["hash"]
    assert logs[2]["prev_hash"] == logs[1]["hash"]

    # 3. Assert untampered chain is valid
    assert trail.verify_integrity() is True

    # 4. Perform tampering: modify historical log content
    logs[1]["details"] = "Advanced state to STOPPED."  # Secret modifications

    # 5. Assert tampering is successfully caught
    assert trail.verify_integrity() is False


def test_visualizations_plot_generation():
    """Asserts that forest plot and power curve generators return valid matplotlib canvases."""
    # 1. Test Forest Plot
    df_results = pd.DataFrame({
        "metric_name": ["Revenue", "Conversion", "Sessions"],
        "relative_lift": [0.042, -0.015, 0.005],
        "rel_ci_lower": [0.015, -0.035, -0.012],
        "rel_ci_upper": [0.069, 0.005, 0.022],
        "p_value": [0.003, 0.125, 0.650]
    })

    fig, ax = plot_forest(df_results, alpha=0.05)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

    # 2. Test Power Curve Plot
    power_data = {
        "mde_relative": np.array([0.01, 0.02, 0.05, 0.10]),
        "sample_size_per_variant": np.array([10000, 2500, 400, 100]),
        "cuped_sample_size_per_variant": np.array([5000, 1250, 200, 50])
    }

    fig_power, ax_power = plot_power_curve(power_data)
    assert isinstance(fig_power, plt.Figure)
    assert isinstance(ax_power, plt.Axes)
    plt.close(fig_power)


def test_experiment_report_generator(tmp_path):
    """Verifies standalone HTML dashboard and Github-compatible Markdown report generation and file writes."""
    from xpyrment.analyze.orchestrator import setup
    from xpyrment.report.generator import ExperimentReportGenerator

    # Mock experimental setup
    df = pd.DataFrame({
        "variant": ["control"] * 100 + ["treatment"] * 100,
        "revenue": np.random.default_rng(42).normal(10, 2, 200),
        "pre_revenue": np.random.default_rng(42).normal(10, 2, 200),
        "user_id": range(200),
    })

    exp = setup(df, treatment_col="variant", id_col="user_id", covariates=["pre_revenue"])
    exp.register_metric("revenue")

    res = exp.run_analysis(control="control", treatment="treatment")

    # Instantiate generator
    generator = ExperimentReportGenerator(res, experiment_name="Harden Growth Experiment")

    # Verify SRM computation
    assert generator.control_n == 100
    assert generator.treatment_n == 100
    assert generator.srm_passed is True
    assert generator.srm_p_value == 1.0  # Perfect 50/50 division

    # Test Markdown compilation
    md_report = generator.generate_markdown()
    assert "Harden Growth Experiment" in md_report
    assert "revenue" in md_report.lower()
    assert "CUPED" in md_report
    assert "Covariate Balance Love Plot" in md_report

    # Test HTML compilation
    html_report = generator.generate_html()
    assert "<!DOCTYPE html>" in html_report
    assert "Harden Growth Experiment" in html_report
    assert "revenue" in html_report.lower()
    assert "smd-success" in html_report.lower()  # balanced covariate styles
    assert "header-logo-container" in html_report
    assert "sizes=\"any\"" in html_report

    # Test saving capabilities
    html_file = tmp_path / "report.html"
    md_file = tmp_path / "report.md"

    generator.save_html(str(html_file))
    generator.save_markdown(str(md_file))

    assert html_file.exists()
    assert md_file.exists()
    assert len(html_file.read_text(encoding="utf-8")) > 1000
    assert len(md_file.read_text(encoding="utf-8")) > 100

