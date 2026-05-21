"""Integration and unit tests for Block 65's ExperimentDashboardServer and Web-UI endpoints.

Verifies correct socket binding on port 0, endpoint response validation,
NumPy type-safe serialization, and clean thread teardowns across multiple scenarios.
"""

import json
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
import pytest
from xpyrment.run.monitor import LiveMonitor, WebhookAlertDispatcher
from xpyrment.run.webui import ExperimentDashboardServer, numpy_to_python


def test_numpy_to_python_serialization_safety() -> None:
    """Verifies that numpy_to_python converts complex NumPy and Pandas types safely to standard Python types."""
    test_dict = {
        "int_val": np.int64(42),
        "float_val": np.float64(3.14159),
        "bool_val": np.bool_(True),
        "arr_val": np.array([1, 2, np.int32(3)]),
        "nan_val": np.nan,
        "nat_val": pd.NaT,
        "timestamp": pd.Timestamp("2026-05-21 12:00:00"),
        "nested": {
            "nested_float": np.float32(1.23)
        }
    }

    cleaned = numpy_to_python(test_dict)

    # Assert types are fully converted to standard Python structures
    assert isinstance(cleaned["int_val"], int)
    assert cleaned["int_val"] == 42
    assert isinstance(cleaned["float_val"], float)
    assert abs(cleaned["float_val"] - 3.14159) < 1e-5
    assert isinstance(cleaned["bool_val"], bool)
    assert cleaned["bool_val"] is True
    assert isinstance(cleaned["arr_val"], list)
    assert cleaned["arr_val"] == [1, 2, 3]
    assert cleaned["nan_val"] is None
    assert cleaned["nat_val"] is None
    assert isinstance(cleaned["timestamp"], str)
    assert cleaned["timestamp"] == "2026-05-21T12:00:00"
    assert isinstance(cleaned["nested"]["nested_float"], float)
    assert abs(cleaned["nested"]["nested_float"] - 1.23) < 1e-2

    # Verify standard json serialization succeeds without error
    json_str = json.dumps(cleaned)
    assert "nested_float" in json_str


def test_dashboard_server_healthy_scenario() -> None:
    """Tests the dashboard server under a healthy, balanced scenario with no anomalies."""
    # 1. Create healthy balanced mock data
    df = pd.DataFrame({
        "unit_id": [f"user_{i}" for i in range(100)],
        "exposed_at": pd.date_range(start="2026-05-01", periods=100, freq="h"),
        "variant": ["control", "treatment"] * 50
    })

    monitor = LiveMonitor(df, time_col="exposed_at")
    # Setting freq="h" to avoid daily bin traffic dropouts
    server = ExperimentDashboardServer(monitor, expected_ratios=[0.5, 0.5], port=0, freq="h")

    try:
        # Start server - uses port 0 to bind to a free dynamic port
        server.start()
        assert server.port > 0
        assert server._is_running is True

        base_url = f"http://127.0.0.1:{server.port}"

        # Fetch HTML Client
        with urllib.request.urlopen(f"{base_url}/", timeout=5.0) as response:
            assert response.status == 200
            html = response.read().decode("utf-8")
            assert "xpyrment Live Monitor" in html
            assert "glass-panel" in html
            assert "SYSTEM ACTIVE" in html

        # Fetch JSON API
        with urllib.request.urlopen(f"{base_url}/api/data", timeout=5.0) as response:
            assert response.status == 200
            raw_body = response.read().decode("utf-8")
            data = json.loads(raw_body)
            
            assert data["status"] == "success"
            payload = data["data"]
            assert payload["shutoff_triggered"] is False
            assert len(payload["alerts_triggered"]) == 0
            assert payload["expected_ratios"] == [0.5, 0.5]
            assert payload["variants"] == ["control", "treatment"]
            assert payload["observed_counts"] == [50, 50]
            assert "trends" in payload
            assert "labels" in payload["trends"]
            assert len(payload["trends"]["ratios"]) > 0

        # Fetch invalid URL to verify 404 response
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(f"{base_url}/nonexistent", timeout=5.0)
        assert excinfo.value.code == 404

    finally:
        # Stop and join background server thread cleanly
        server.stop()
        assert server._is_running is False
        assert server.thread is None


def test_dashboard_server_srm_alert_scenario() -> None:
    """Verifies that the dashboard correctly flags a highly mismatched Sample Ratio Mismatch (SRM)."""
    # Create mismatched assignments (1000 control vs 1 treatment user)
    df = pd.DataFrame({
        "unit_id": [f"user_{i}" for i in range(1001)],
        "exposed_at": pd.date_range(start="2026-05-01", periods=1001, freq="min"),
        "variant": ["control"] * 1000 + ["treatment"] * 1
    })

    dispatcher = WebhookAlertDispatcher()
    alerts = []
    dispatcher.register_callback(lambda alert_type, message, payload: alerts.append(alert_type))

    monitor = LiveMonitor(df, time_col="exposed_at", dispatcher=dispatcher)
    server = ExperimentDashboardServer(monitor, expected_ratios=[0.5, 0.5], port=0, freq="min")

    try:
        server.start()
        base_url = f"http://127.0.0.1:{server.port}"

        # Fetch JSON API
        with urllib.request.urlopen(f"{base_url}/api/data", timeout=5.0) as response:
            assert response.status == 200
            data = json.loads(response.read().decode("utf-8"))
            
            payload = data["data"]
            assert payload["shutoff_triggered"] is True
            assert "SRM_SHUTOFF_TRIGGERED" in payload["alerts_triggered"]
            assert payload["cumulative_srm"]["srm_detected"] is True
            assert payload["cumulative_srm"]["p_value"] < 0.001

    finally:
        server.stop()


def test_dashboard_server_traffic_drop_scenario() -> None:
    """Verifies that a sudden traffic dropout triggers a volatility anomaly alert in the UI payload."""
    # Create traffic logs: 5 hours with baseline counts [49, 51, 49, 51, 50], then 1 log in the 6th hour
    time_series = []
    variants = []
    
    baseline_counts = [49, 51, 49, 51, 50]
    for hour, count in enumerate(baseline_counts):
        t_str = f"2026-05-01 {hour:02d}:00:00"
        time_series.extend([t_str] * count)
        for i in range(count):
            variants.append("control" if i % 2 == 0 else "treatment")

    # 6th hour of only 1 log (extreme drop)
    time_series.append("2026-05-01 05:00:00")
    variants.append("control")

    df = pd.DataFrame({
        "unit_id": [f"user_{i}" for i in range(len(time_series))],
        "exposed_at": time_series,
        "variant": variants
    })

    # Ensure temporal binning aligns on hour
    monitor = LiveMonitor(df, time_col="exposed_at")
    server = ExperimentDashboardServer(monitor, expected_ratios=[0.5, 0.5], port=0, freq="h")

    try:
        server.start()
        base_url = f"http://127.0.0.1:{server.port}"

        # Fetch JSON API
        with urllib.request.urlopen(f"{base_url}/api/data", timeout=5.0) as response:
            assert response.status == 200
            data = json.loads(response.read().decode("utf-8"))
            
            payload = data["data"]
            assert "TRAFFIC_DROP_ANOMALY" in payload["alerts_triggered"]
            assert payload["traffic_anomaly"]["anomaly_detected"] is True
            assert payload["traffic_anomaly"]["z_score"] < -3.0

    finally:
        server.stop()


def test_teardown_and_redundant_stop() -> None:
    """Verifies server lifecycle safety, confirming redundant stop() calls resolve silently."""
    df = pd.DataFrame({
        "unit_id": ["u1"],
        "exposed_at": ["2026-05-01 10:00:00"],
        "variant": ["control"]
    })
    monitor = LiveMonitor(df, time_col="exposed_at")
    server = ExperimentDashboardServer(monitor, expected_ratios=[0.5, 0.5], port=0)

    # Test redundant stops on non-started server do not raise errors
    server.stop()
    assert server._is_running is False

    # Start and stop standard cycle
    server.start()
    assert server._is_running is True
    server.stop()
    assert server._is_running is False

    # Redundant stop on stopped server
    server.stop()
    assert server._is_running is False
