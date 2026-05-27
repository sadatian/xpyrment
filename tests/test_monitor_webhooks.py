"""Unit tests for Dynamic SRM Shutoff Webhooks & Alert System (Block 61)."""

import json
import urllib.error
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from xpyrment.run.monitor import LiveMonitor, WebhookAlertDispatcher


# -------------------------------------------------------------------------
# 1. WebhookAlertDispatcher Tests
# -------------------------------------------------------------------------

def test_webhook_alert_dispatcher_callbacks():
    """Verifies that the dispatcher can register and invoke custom callback functions."""
    dispatcher = WebhookAlertDispatcher()
    calls: List[Dict[str, Any]] = []

    def test_cb(alert_type: str, message: str, payload: Dict[str, Any]) -> None:
        calls.append({"type": alert_type, "msg": message, "pay": payload})

    dispatcher.register_callback(test_cb)
    dispatcher.dispatch("TEST_ALERT", "Hello world", {"data": 123})

    assert len(calls) == 1
    assert calls[0]["type"] == "TEST_ALERT"
    assert calls[0]["msg"] == "Hello world"
    assert calls[0]["pay"] == {"data": 123}


def test_webhook_alert_dispatcher_failure_isolation():
    """Verifies that a failure in one webhook or callback does not block others from executing."""
    dispatcher = WebhookAlertDispatcher()
    calls: List[str] = []

    def failing_cb(alert_type: str, message: str, payload: Dict[str, Any]) -> None:
        raise ValueError("Simulated handler crash")

    def successful_cb(alert_type: str, message: str, payload: Dict[str, Any]) -> None:
        calls.append(alert_type)

    dispatcher.register_callback(failing_cb)
    dispatcher.register_callback(successful_cb)

    # Dispatch should not raise an error
    dispatcher.dispatch("ISOLATION_TEST", "Test", {})
    assert calls == ["ISOLATION_TEST"]


@patch("urllib.request.urlopen")
def test_webhook_alert_dispatcher_slack_and_custom(mock_urlopen):
    """Verifies that Slack and Custom webhooks format payloads and invoke urllib.request correctly."""
    mock_response = MagicMock()
    mock_response.read.return_value = b"OK"
    mock_urlopen.return_value = mock_response

    dispatcher = WebhookAlertDispatcher()
    dispatcher.register_slack("https://hooks.slack.com/services/test")
    dispatcher.register_custom("https://myapi.com/webhook")

    dispatcher.dispatch("TRAFFIC_DROP_ANOMALY", "Drop detected", {"metric": "exposures"})

    # Check urlopen calls
    assert mock_urlopen.call_count == 2
    
    first_call_args = mock_urlopen.call_args_list[0][0]
    req1 = first_call_args[0]
    assert req1.full_url == "https://hooks.slack.com/services/test"
    assert req1.headers.get("Content-type") == "application/json"
    
    # Verify Slack text contains required info
    body1 = json.loads(req1.data.decode("utf-8"))
    assert "text" in body1
    assert "Drop detected" in body1["text"]
    assert "TRAFFIC_DROP_ANOMALY" in body1["text"]

    second_call_args = mock_urlopen.call_args_list[1][0]
    req2 = second_call_args[0]
    assert req2.full_url == "https://myapi.com/webhook"
    assert req2.headers.get("Content-type") == "application/json"
    
    # Verify Custom format
    body2 = json.loads(req2.data.decode("utf-8"))
    assert body2["alert_type"] == "TRAFFIC_DROP_ANOMALY"
    assert body2["message"] == "Drop detected"
    assert body2["details"] == {"metric": "exposures"}


@patch("urllib.request.urlopen")
def test_webhook_alert_dispatcher_email(mock_urlopen):
    """Verifies email webhook registration and dispatching."""
    mock_response = MagicMock()
    mock_response.read.return_value = b"OK"
    mock_urlopen.return_value = mock_response

    dispatcher = WebhookAlertDispatcher()
    dispatcher.register_email("https://email-service.internal/webhook")

    dispatcher.dispatch("SRM_SHUTOFF_TRIGGERED", "SRM alarm", {"p_value": 0.0001})

    assert mock_urlopen.call_count == 1
    req = mock_urlopen.call_args[0][0]
    assert req.full_url == "https://email-service.internal/webhook"
    body = json.loads(req.data.decode("utf-8"))
    assert body["subject"] == "[xpyrment Alert] SRM_SHUTOFF_TRIGGERED"
    assert "SRM alarm" in body["body"]


@patch("urllib.request.urlopen")
def test_webhook_alert_dispatcher_http_error_handling(mock_urlopen):
    """Verifies that urllib connection timeouts or HTTP errors are caught and logged without raising exceptions."""
    # Simulate urllib timeout or connection failure
    mock_urlopen.side_effect = urllib.error.URLError("Connection timed out")

    dispatcher = WebhookAlertDispatcher()
    dispatcher.register_slack("https://hooks.slack.com/services/bad")

    # Dispatch should handle exception gracefully
    dispatcher.dispatch("TIMEOUT_TEST", "Test timeout", {})
    assert mock_urlopen.call_count == 1


# -------------------------------------------------------------------------
# 2. LiveMonitor Anomaly and SRM Verification
# -------------------------------------------------------------------------

def test_live_monitor_ideal_state():
    """Verifies that stable, balanced traffic generates no alerts and p_value > 0.05."""
    # Generate stable binned traffic for 10 days
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-05-01", periods=10, freq="D")
    
    data_list = []
    for d in dates:
        # 100 control and 100 treatment exposures per day
        for _ in range(100):
            data_list.append({"timestamp": d, "variant": "control", "unit_id": f"c_{rng.integers(0, 100000)}"})
            data_list.append({"timestamp": d, "variant": "treatment", "unit_id": f"t_{rng.integers(0, 100000)}"})

    df = pd.DataFrame(data_list)
    dispatcher = WebhookAlertDispatcher()
    alerts: List[Dict[str, Any]] = []
    dispatcher.register_callback(lambda t, m, p: alerts.append({"type": t, "msg": m}))

    monitor = LiveMonitor(df, time_col="timestamp", dispatcher=dispatcher)
    
    # Run checks
    res = monitor.run_telemetry_checks(expected_ratios=[0.5, 0.5])
    
    assert not res["shutoff_triggered"]
    assert len(res["alerts_triggered"]) == 0
    assert not res["traffic_anomaly"]["anomaly_detected"]
    assert not res["cumulative_srm"]["srm_detected"]
    assert res["cumulative_srm"]["p_value"] > 0.05
    assert len(alerts) == 0


def test_live_monitor_insufficient_history():
    """Verifies that traffic anomaly check is bypassed when there are less than 3 bins of history."""
    dates = pd.date_range("2026-05-01", periods=2, freq="D")
    data_list = [
        {"timestamp": dates[0], "variant": "control", "unit_id": "u1"},
        {"timestamp": dates[0], "variant": "treatment", "unit_id": "u2"},
        {"timestamp": dates[1], "variant": "control", "unit_id": "u3"},
        {"timestamp": dates[1], "variant": "treatment", "unit_id": "u4"},
    ]
    df = pd.DataFrame(data_list)
    monitor = LiveMonitor(df, time_col="timestamp")
    
    anomaly_res = monitor.check_traffic_anomaly(window=5)
    assert not anomaly_res["anomaly_detected"]
    assert "Insufficient history" in anomaly_res["message"]


def test_live_monitor_degenerate_variance():
    """Verifies degenerate variance fallback checks (when historical volume is perfectly constant)."""
    dates = pd.date_range("2026-05-01", periods=5, freq="D")
    data_list = []
    # Exactly 10 exposures per day for control and treatment (perfectly constant baseline)
    for i, d in enumerate(dates):
        # On day 5, drop exposures to 3 (which is a > 50% drop)
        count = 3 if i == 4 else 10
        for u in range(count):
            data_list.append({"timestamp": d, "variant": "control", "unit_id": f"u_{i}_{u}_c"})
            data_list.append({"timestamp": d, "variant": "treatment", "unit_id": f"u_{i}_{u}_t"})

    df = pd.DataFrame(data_list)
    
    # 1. Test drop that triggers fallback alert (drop to 3 unique units is < 5 unique units mean * (1 - 0.50))
    # Mean of preceding 4 days is 20 units total.
    # Latest day volume is 6 units total.
    # 6 < 20 * (1 - 0.5) = 10, so it should trigger anomaly!
    monitor1 = LiveMonitor(df, time_col="timestamp")
    res1 = monitor1.check_traffic_anomaly(window=3, drop_threshold=0.5)
    assert res1["anomaly_detected"]
    assert res1["historical_std"] == 0.0
    assert "Degenerate variance" in res1["message"]

    # 2. Test drop that does NOT trigger fallback alert (drop to 7 exposures = 14 total units)
    # 14 is not < 20 * (1 - 0.5) = 10, so no anomaly.
    data_list_stable = []
    for i, d in enumerate(dates):
        count = 7 if i == 4 else 10
        for u in range(count):
            data_list_stable.append({"timestamp": d, "variant": "control", "unit_id": f"u_{i}_{u}_c"})
            data_list_stable.append({"timestamp": d, "variant": "treatment", "unit_id": f"u_{i}_{u}_t"})
            
    df_stable = pd.DataFrame(data_list_stable)
    monitor2 = LiveMonitor(df_stable, time_col="timestamp")
    res2 = monitor2.check_traffic_anomaly(window=3, drop_threshold=0.5)
    assert not res2["anomaly_detected"]
    assert res2["historical_std"] == 0.0


def test_live_monitor_traffic_drop_anomaly():
    """Verifies that a statistically significant traffic drop triggers a Z-score anomaly alert."""
    dates = pd.date_range("2026-05-01", periods=6, freq="D")
    data_list = []
    # Historical volume fluctuates around 100 per variant (total ~200 per day)
    # Preceding days: ~200, ~210, ~190, ~205, ~195
    # Latest day: 50 total (major drop)
    rng = np.random.default_rng(42)
    daily_counts = [100, 105, 95, 102, 98, 25] # per variant
    
    for i, d in enumerate(dates):
        cnt = daily_counts[i]
        for u in range(cnt):
            data_list.append({"timestamp": d, "variant": "control", "unit_id": f"u_{i}_{u}_c"})
            data_list.append({"timestamp": d, "variant": "treatment", "unit_id": f"u_{i}_{u}_t"})

    df = pd.DataFrame(data_list)
    dispatcher = WebhookAlertDispatcher()
    alerts = []
    dispatcher.register_callback(lambda t, m, p: alerts.append(t))

    monitor = LiveMonitor(df, time_col="timestamp", dispatcher=dispatcher)
    res = monitor.run_telemetry_checks(expected_ratios=[0.5, 0.5], window=4, z_threshold=2.5)

    assert "TRAFFIC_DROP_ANOMALY" in res["alerts_triggered"]
    assert not res["shutoff_triggered"]  # Traffic drop doesn't trigger shutoff
    assert res["traffic_anomaly"]["anomaly_detected"]
    assert "TRAFFIC_DROP_ANOMALY" in alerts


def test_live_monitor_cumulative_srm():
    """Verifies that a severe cumulative sample ratio mismatch triggers a shutoff and Webhook alert."""
    dates = pd.date_range("2026-05-01", periods=5, freq="D")
    data_list = []
    
    # Generate heavy SRM: 300 control, 700 treatment total
    for d in dates:
        for _ in range(60):
            data_list.append({"timestamp": d, "variant": "control", "unit_id": f"c_{d.day}_{np.random.randint(0, 100000)}"})
        for _ in range(140):
            data_list.append({"timestamp": d, "variant": "treatment", "unit_id": f"t_{d.day}_{np.random.randint(0, 100000)}"})

    df = pd.DataFrame(data_list)
    dispatcher = WebhookAlertDispatcher()
    alerts = []
    dispatcher.register_callback(lambda t, m, p: alerts.append(t))

    monitor = LiveMonitor(df, time_col="timestamp", dispatcher=dispatcher)
    res = monitor.run_telemetry_checks(expected_ratios=[0.5, 0.5])

    assert res["shutoff_triggered"]
    assert "SRM_SHUTOFF_TRIGGERED" in res["alerts_triggered"]
    assert res["cumulative_srm"]["srm_detected"]
    assert res["cumulative_srm"]["p_value"] < 0.001
    assert "SRM_SHUTOFF_TRIGGERED" in alerts


def test_live_monitor_sequential_srm():
    """Verifies that a sequential SPRT mismatch triggers a shutoff alert."""
    # Generate 500 control and 800 treatment exposures over 5 days, sorted chronologically
    dates = pd.date_range("2026-05-01", periods=5, freq="D")
    data_list = []
    
    for d in dates:
        # Control exposures
        for _ in range(100):
            data_list.append({"timestamp": d, "variant": "control", "unit_id": f"c_{d.day}_{np.random.randint(0, 100000)}"})
        # Treatment exposures (major imbalance)
        for _ in range(160):
            data_list.append({"timestamp": d, "variant": "treatment", "unit_id": f"t_{d.day}_{np.random.randint(0, 100000)}"})

    df = pd.DataFrame(data_list)
    dispatcher = WebhookAlertDispatcher()
    alerts = []
    dispatcher.register_callback(lambda t, m, p: alerts.append(t))

    monitor = LiveMonitor(df, time_col="timestamp", dispatcher=dispatcher)
    
    # We use run_telemetry_checks with expected ratios [0.5, 0.5]
    # This should trigger both cumulative chi-square and sequential SPRT SRM alerts
    res = monitor.run_telemetry_checks(expected_ratios=[0.5, 0.5])

    assert res["shutoff_triggered"]
    assert "SEQUENTIAL_SRM_TRIGGERED" in res["alerts_triggered"]
    assert res["sequential_srm"]["srm_detected"]
    assert "SEQUENTIAL_SRM_TRIGGERED" in alerts
