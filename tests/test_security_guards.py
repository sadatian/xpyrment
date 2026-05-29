import os
import tempfile
import pytest
from pathlib import Path

from xpyrment.core.validators import validate_secure_path
from xpyrment.run.ingestion import DuckDBIngester
from xpyrment.report.generator import ExperimentReportGenerator
from xpyrment.analyze.orchestrator import AnalysisResult


def test_validate_secure_path_workspace_and_temp():
    """Asserts that validate_secure_path allows paths inside the workspace and temporary directories."""
    # 1. Valid workspace path
    workspace_path = Path("c:/Users/Dan/projects/xpyrment/reports/my_report.html")
    resolved = validate_secure_path(workspace_path)
    assert resolved == workspace_path.resolve()

    # 2. Valid temp directory path
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / "xpyrment_temp_file.txt"
    resolved_temp = validate_secure_path(temp_path, allow_temp=True)
    assert resolved_temp == temp_path.resolve()


def test_validate_secure_path_traversal_escapes():
    """Asserts that validate_secure_path correctly blocks directory traversal escape attempts."""
    # 1. Traversal string escaping workspace
    escape_workspace = "c:/Users/Dan/projects/xpyrment/reports/../../../some_secret.txt"
    with pytest.raises(PermissionError, match="escapes secure boundaries"):
        validate_secure_path(escape_workspace)

    # 2. Path entirely outside allowed directories
    outside_path = "c:/Windows/System32/cmd.exe"
    with pytest.raises(PermissionError, match="escapes secure boundaries"):
        validate_secure_path(outside_path)


def test_duckdb_ingester_path_traversal_security():
    """Asserts that DuckDBIngester methods enforce path traversal protections."""
    with DuckDBIngester() as ingester:
        # Invalid path outside workspace
        outside_path = "c:/Windows/System32/cmd.exe"
        
        with pytest.raises(PermissionError, match="escapes secure boundaries"):
            ingester.compute_covariate_balance(
                parquet_path=outside_path,
                treatment_col="group",
                covariate_cols=["age"]
            )

        with pytest.raises(PermissionError, match="escapes secure boundaries"):
            ingester.compute_welch_statistics(
                parquet_path=outside_path,
                treatment_col="group",
                metric_cols=["revenue"]
            )


def test_report_generator_save_path_traversal_security():
    """Asserts that ExperimentReportGenerator enforces path traversal protections on save operations."""
    import pandas as pd
    
    # Setup dummy AnalysisResult
    dummy_results = [{"metric_name": "m1", "metric_type": "mean", "control_mean": 1.0, "treatment_mean": 1.1, "relative_lift": 0.1, "p_value": 0.04, "control_n": 100, "treatment_n": 100, "cuped_applied": False}]
    res = AnalysisResult(raw_results=dummy_results)
    
    generator = ExperimentReportGenerator(res, "Dummy Experiment")
    
    outside_path = "c:/Windows/System32/cmd.exe"
    
    with pytest.raises(PermissionError, match="escapes secure boundaries"):
        generator.save_html(outside_path)

    with pytest.raises(PermissionError, match="escapes secure boundaries"):
        generator.save_markdown(outside_path)
