"""Integration and unit tests for Xpyrment Hub Server (Block 53 / run/hub.py)."""

import json
import socket
import urllib.request
import urllib.error
import pytest
import pandas as pd
import numpy as np
from xpyrment.run.hub import XpyrmentHubServer, numpy_to_python


def test_numpy_to_python():
    """Asserts numpy_to_python converts complex scientific types to JSON-serializable types."""
    data = {
        "int": np.int64(42),
        "float": np.float64(3.14),
        "array": np.array([1, 2, 3]),
        "bool": np.bool_(True),
        "nan": np.nan,
        "nat": pd.NaT,
        "timestamp": pd.Timestamp("2026-05-27 12:00:00"),
        "list": [np.int32(10), {"nested_float": np.float32(1.23)}],
        "dict": {"a": np.int64(1)},
        "str": "normal_string"
    }

    converted = numpy_to_python(data)
    assert isinstance(converted["int"], int)
    assert isinstance(converted["float"], float)
    assert isinstance(converted["array"], list)
    assert isinstance(converted["bool"], bool)
    assert converted["nan"] is None
    assert converted["nat"] is None
    assert isinstance(converted["timestamp"], str)
    assert isinstance(converted["list"][0], int)
    assert isinstance(converted["list"][1]["nested_float"], float)
    assert isinstance(converted["dict"]["a"], int)
    assert converted["str"] == "normal_string"


def test_hub_server_integration():
    """Runs integration verification of the XpyrmentHubServer, hitting all REST routes and mock actions."""
    # Initialize hub server on ephemeral port (port=0) to isolate tests
    server = XpyrmentHubServer(host="127.0.0.1", port=0, log_usage=True)
    server.start()

    # Confirm it is running and captures real port
    assert server._is_running is True
    assert server.port > 0
    base_url = f"http://127.0.0.1:{server.port}"

    try:
        # 1. Test GET root page / index.html
        with urllib.request.urlopen(f"{base_url}/") as response:
            html = response.read().decode("utf-8")
            assert "xpyrment Primary Hub" in html

        with urllib.request.urlopen(f"{base_url}/index.html") as response:
            html = response.read().decode("utf-8")
            assert "xpyrment Primary Hub" in html

        # 2. Test GET data status (no data loaded initially)
        with urllib.request.urlopen(f"{base_url}/api/data/status") as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload["status"] == "success"
            assert payload["has_data"] is False
            assert payload["dataset_name"] == "None"

        # 3. Test POST generate simulated data
        simulate_req = urllib.request.Request(
            f"{base_url}/api/data/simulate",
            data=json.dumps({"n_samples": 200}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(simulate_req) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload["status"] == "success"

        # Check status after simulation
        with urllib.request.urlopen(f"{base_url}/api/data/status") as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload["has_data"] is True
            assert payload["rows"] == 200
            assert "revenue" in payload["columns"]

        # 4. Test GET 404 Route
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{base_url}/invalid-route")
        assert exc_info.value.code == 404

        # 5. Test POST Start Monitoring Module
        monitor_req = urllib.request.Request(
            f"{base_url}/api/module/monitoring/start",
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(monitor_req) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload["status"] == "success"
            assert payload["port"] > 0

        # 6. Test POST Design of Experiments List
        design_req = urllib.request.Request(
            f"{base_url}/api/module/design/generate",
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(design_req) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload["status"] == "success"
            assert len(payload["data"]) > 0

        # 7. Test POST Quasi-Experiments OLS/DiD Analysis
        quasi_req = urllib.request.Request(
            f"{base_url}/api/module/quasi/analyze",
            data=json.dumps({"y_col": "revenue", "x_cols": ["variant"]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(quasi_req) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload["status"] == "success"
            assert "intercept" in payload["data"]

        # 8. Test POST Covariate Balance Checks
        balance_req = urllib.request.Request(
            f"{base_url}/api/module/balance",
            data=json.dumps({"group_col": "variant", "covariates": ["pre_revenue"]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(balance_req) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload["status"] == "success"

        # 9. Test POST Personalization Meta-Learners
        personalize_req = urllib.request.Request(
            f"{base_url}/api/module/personalize/train",
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(personalize_req) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload["status"] == "success"
            assert "T-Learner trained successfully" in payload["message"]

        # 10. Test POST Graph Clustering
        network_req = urllib.request.Request(
            f"{base_url}/api/module/network/cluster",
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(network_req) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload["status"] == "success"
            assert "Graph partitioned successfully" in payload["message"]

        # 11. Test POST Interactions ANOVA
        interactions_req = urllib.request.Request(
            f"{base_url}/api/module/interactions/anova",
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(interactions_req) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload["status"] == "success"
            assert "ANOVA Calculated" in payload["message"]

        # 12. Test POST duplicate start guard
        server.start()

    finally:
        # Shutdown server cleanly
        server.stop()
        assert server._is_running is False


def test_hub_server_no_data_errors():
    """Asserts that hitting analytical endpoints before generating data returns clean 400 Bad Request statuses."""
    server = XpyrmentHubServer(host="127.0.0.1", port=0)
    server.start()
    base_url = f"http://127.0.0.1:{server.port}"

    try:
        # 1. Quasi without data
        req1 = urllib.request.Request(
            f"{base_url}/api/module/quasi/analyze",
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req1)
        assert exc.value.code == 400

        # 2. Balance without data
        req2 = urllib.request.Request(
            f"{base_url}/api/module/balance",
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req2)
        assert exc.value.code == 400

        # 3. Personalize without data
        req3 = urllib.request.Request(
            f"{base_url}/api/module/personalize/train",
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req3)
        assert exc.value.code == 400

        # 4. Network without data
        req4 = urllib.request.Request(
            f"{base_url}/api/module/network/cluster",
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req4)
        assert exc.value.code == 400

        # 5. Interactions without data
        req5 = urllib.request.Request(
            f"{base_url}/api/module/interactions/anova",
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req5)
        assert exc.value.code == 400

    finally:
        server.stop()
