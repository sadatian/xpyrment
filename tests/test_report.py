import json
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def test_audit_trail_sqlite_persistence(tmp_path):
    """Verifies that AuditTrail cleanly logs events to SQLite, persists them, and loads safely."""
    db_file = tmp_path / "audit.db"
    trail = AuditTrail("EXP-202", db_path=str(db_file))

    # Log events with signatures and public keys
    trail.log_event("INIT", "Initialized experiment", signature="sig123", public_key="key456")
    trail.log_event("TRANSITION", "Moved to RUNNING")

    assert len(trail.get_logs()) == 2
    assert trail.verify_integrity() is True

    # Check database records directly
    import sqlite3
    with sqlite3.connect(str(db_file)) as conn:
        df = pd.read_sql_query("SELECT * FROM audit_logs WHERE experiment_id='EXP-202'", conn)
        assert len(df) == 2
        assert df.iloc[0]["action"] == "INIT"
        assert df.iloc[0]["signature"] == "sig123"
        assert df.iloc[0]["public_key"] == "key456"
        assert df.iloc[1]["action"] == "TRANSITION"


def test_report_generator_exceptions_and_empty():
    """Asserts proper exceptions are raised for invalid or empty inputs to report generator."""
    from xpyrment.report.generator import ExperimentReportGenerator

    # 1. Invalid AnalysisResult
    with pytest.raises(ValueError, match="Invalid AnalysisResult provided"):
        ExperimentReportGenerator(None)

    # 2. Empty df_raw
    from xpyrment.analyze.orchestrator import AnalysisResult
    res_empty = AnalysisResult(
        raw_results=[],
        alpha=0.05,
        balance_checker=None
    )
    generator = ExperimentReportGenerator(res_empty)
    assert generator.control_n == 0
    assert generator.treatment_n == 0
    assert generator.srm_passed is True
    assert generator.srm_p_value == 1.0


def test_report_generator_imbalanced_covariates(tmp_path):
    """Validates that imbalanced covariates correctly trigger warning badges in report layouts."""
    from xpyrment.analyze.orchestrator import AnalysisResult
    from xpyrment.report.generator import ExperimentReportGenerator

    # Setup dummy diagnostics showing a large SMD (imbalance)
    class MockBalanceChecker:
        def __init__(self):
            self.diagnostics_ = {
                "age": {
                    "smd": 0.25,  # > 0.1 threshold
                    "variance_ratio": 1.1,
                    "mean_control": 30.0,
                    "mean_treatment": 32.5
                },
                "income": {
                    "smd": 0.02,  # balanced
                    "variance_ratio": 1.0,
                    "mean_control": 50000.0,
                    "mean_treatment": 50100.0
                }
            }
        def generate_love_plot(self):
            return "Mock Love Plot"

    df_raw = pd.DataFrame({
        "metric_name": ["revenue"],
        "metric_type": ["mean"],
        "control_mean": [10.0],
        "treatment_mean": [10.5],
        "control_n": [500],
        "treatment_n": [500],
        "relative_lift": [0.05],
        "p_value": [0.04],
        "cuped_applied": [False],
        "rel_ci_lower": [0.01],
        "rel_ci_upper": [0.09]
    })

    res = AnalysisResult(
        raw_results=df_raw.to_dict(orient="records"),
        alpha=0.05,
        balance_checker=MockBalanceChecker()
    )

    generator = ExperimentReportGenerator(res, experiment_name="Imbalance Test")
    
    # 1. Verify markdown output shows warning emoji
    md = generator.generate_markdown()
    assert "IMBALANCE DETECTED" in md
    assert "age" in md
    assert "income" not in md.split("IMBALANCE DETECTED")[1].split("\n")[0] # Only age is flagged

    # 2. Verify HTML output has imbalanced class
    html = generator.generate_html()
    assert "smd-fail" in html
    assert "IMBALANCED" in html


