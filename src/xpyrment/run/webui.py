"""Standalone HTTP Web-UI dashboard server for real-time digital experiment monitoring.

Provides a thread-safe HTTPServer that binds securely, handles dynamic port allocation,
and serves a premium dark-mode, glassmorphic client-side dashboard with live charts.
"""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from xpyrment.run.monitor import LiveMonitor

logger = logging.getLogger(__name__)


def numpy_to_python(obj: Any) -> Any:
    """Recursively converts NumPy and Pandas data types to native Python types for JSON safety.

    Args:
        obj (Any): Any input data structure.

    Returns:
        Any: Standard Python types suitable for json.dumps serialization.
    """
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [numpy_to_python(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return numpy_to_python(obj.tolist())
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


class ExperimentDashboardServer:
    """Hosts an interactive Web-UI dashboard to monitor running experiments.

    Spawns a thread-safe background server. Supports port 0 binding to avoid testing
     EADDRINUSE conflicts and provides clean stop/join teardown lifecycle hook.
    """

    def __init__(
        self,
        monitor: LiveMonitor,
        expected_ratios: List[float],
        host: str = "127.0.0.1",
        port: int = 0,
        variant_col: str = "variant",
        freq: str = "D"
    ) -> None:
        """Initializes the ExperimentDashboardServer.

        Args:
            monitor (LiveMonitor): The active LiveMonitor containing exposure logs.
            expected_ratios (List[float]): Expected target allocation ratios (e.g. [0.5, 0.5]).
            host (str): Host binding address. Defaults to "127.0.0.1".
            port (int): Port to bind to. Set to 0 to dynamically allocate a free port.
            variant_col (str): Variant column name. Defaults to "variant".
            freq (str): Binning frequency. Defaults to "D".
        """
        self.monitor = monitor
        self.expected_ratios = expected_ratios
        self.host = host
        self.port_requested = port
        self.variant_col = variant_col
        self.freq = freq
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.port: int = port
        self._is_running = False

    def start(self) -> None:
        """Binds the HTTP socket and starts the dashboard server loop in a background thread."""
        if self._is_running:
            return

        server_instance = self

        class DashboardHTTPRequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                # Direct server logs to python logging debug level to keep tests silent
                logger.debug(format, *args)

            def do_GET(self) -> None:
                if self.path == "/" or self.path == "/index.html":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(server_instance.get_html_content().encode("utf-8"))
                elif self.path == "/api/data":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    payload = server_instance.get_api_payload()
                    self.wfile.write(json.dumps(payload).encode("utf-8"))
                elif self.path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                else:
                    self.send_response(404)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"Not Found")

        # Bind immediately on the parent thread to capture port or EADDRINUSE exceptions
        self.server = HTTPServer((self.host, self.port_requested), DashboardHTTPRequestHandler)
        self.port = self.server.server_address[1]

        self._is_running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        logger.info(f"xpyrment Live Dashboard started at http://{self.host}:{self.port}")

    def _run_server(self) -> None:
        try:
            if self.server:
                self.server.serve_forever()
        except Exception as e:
            logger.error(f"Error in dashboard server loop: {e}", exc_info=True)

    def stop(self) -> None:
        """Gracefully halts the server, closes sockets, and joins the background thread."""
        if not self._is_running or not self.server:
            return

        self.server.shutdown()
        self.server.server_close()

        if self.thread:
            self.thread.join(timeout=5.0)

        self._is_running = False
        self.server = None
        self.thread = None
        logger.info("xpyrment Live Dashboard server stopped cleanly.")

    def get_api_payload(self) -> Dict[str, Any]:
        """Runs diagnostics checks and serializes stats safely to JSON-compatible data structures."""
        checks = self.monitor.run_telemetry_checks(
            expected_ratios=self.expected_ratios,
            variant_col=self.variant_col,
            freq=self.freq
        )

        cumulative = self.monitor.get_cumulative_traffic(variant_col=self.variant_col, freq=self.freq)

        labels = []
        columns = {}
        ratios = []

        if not cumulative.empty:
            cumulative = cumulative.reindex(columns=sorted(cumulative.columns))
            labels = [str(idx.date()) for idx in cumulative.index]
            for col in cumulative.columns:
                columns[str(col)] = cumulative[col].tolist()

            # Ratio tracking if exactly two arms exist
            if len(cumulative.columns) == 2:
                col1, col2 = cumulative.columns[0], cumulative.columns[1]
                for val1, val2 in zip(cumulative[col1], cumulative[col2]):
                    total = val1 + val2
                    ratios.append(float(val2 / total) if total > 0 else 0.0)

        observed_counts = []
        if not cumulative.empty:
            observed_counts = [int(x) for x in cumulative.iloc[-1].values]

        payload = {
            "status": "success",
            "data": {
                "shutoff_triggered": checks.get("shutoff_triggered", False),
                "alerts_triggered": checks.get("alerts_triggered", []),
                "expected_ratios": self.expected_ratios,
                "variants": sorted(list(cumulative.columns)) if not cumulative.empty else [],
                "observed_counts": observed_counts,
                "traffic_anomaly": checks.get("traffic_anomaly", {}),
                "cumulative_srm": checks.get("cumulative_srm", {}),
                "sequential_srm": checks.get("sequential_srm", {}),
                "trends": {
                    "labels": labels,
                    "columns": columns,
                    "ratios": ratios
                }
            }
        }
        return numpy_to_python(payload)

    def get_html_content(self) -> str:
        """Returns the self-contained premium glassmorphic HTML structure for the client dashboard."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>xpyrment Live Experiment Dashboard</title>
    <!-- Premium Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <!-- Chart.js CDN for live rendering -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-dark: #09090e;
            --panel-bg: rgba(18, 18, 28, 0.45);
            --border-glass: rgba(255, 255, 255, 0.07);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --color-control: #0ea5e9;    /* Glowing Electric Blue */
            --color-treatment: #a855f7;  /* Glowing Vibrant Violet */
            --color-healthy: #10b981;    /* Deep Emerald Green */
            --color-alert: #ef4444;      /* Glowing Crimson Red */
            --color-warn: #f59e0b;       /* Glowing Amber Gold */
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(at 0% 0%, rgba(14, 165, 233, 0.06) 0, transparent 50%),
                radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.06) 0, transparent 50%);
            font-family: 'Outfit', 'Inter', sans-serif;
            color: var(--text-primary);
            min-height: 100vh;
            padding: 32px 24px;
        }

        .dashboard-container {
            max-width: 1440px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 28px;
        }

        /* Glassmorphic Panel Base styling */
        .glass-panel {
            background: var(--panel-bg);
            border: 1px solid var(--border-glass);
            border-radius: 18px;
            box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            padding: 26px;
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.3s ease;
        }

        .glass-panel:hover {
            transform: translateY(-4px);
            border-color: rgba(255, 255, 255, 0.15);
        }

        /* Header Layout */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 8px;
        }

        .brand-title {
            display: flex;
            flex-direction: column;
        }

        .brand-title h1 {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #f8fafc 30%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-title span {
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-family: 'Inter', sans-serif;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-top: 4px;
        }

        /* Pulsing Status indicator Badge */
        .status-badge {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 16px;
            border-radius: 9999px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            color: var(--color-healthy);
            font-weight: 600;
            font-size: 0.85rem;
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
        }

        .status-badge.alert-active {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            color: var(--color-alert);
            animation: pulse-glow 2s infinite;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--color-healthy);
            box-shadow: 0 0 8px var(--color-healthy);
        }

        .status-badge.alert-active .status-dot {
            background-color: var(--color-alert);
            box-shadow: 0 0 8px var(--color-alert);
        }

        /* KPI Responsive Grid */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px;
        }

        .kpi-card {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 140px;
        }

        .kpi-card .kpi-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-secondary);
            font-family: 'Inter', sans-serif;
            font-weight: 500;
        }

        .kpi-card .kpi-value {
            font-size: 2.2rem;
            font-weight: 800;
            margin-top: 12px;
            letter-spacing: -1px;
        }

        .kpi-card .kpi-delta {
            font-size: 0.85rem;
            margin-top: 8px;
            color: var(--text-secondary);
        }

        /* KPI Custom Border highlights */
        .kpi-total { border-left: 4px solid var(--color-control); }
        .kpi-srm { border-left: 4px solid var(--color-healthy); }
        .kpi-sprt { border-left: 4px solid var(--color-treatment); }
        .kpi-volatility { border-left: 4px solid var(--color-warn); }

        .kpi-card.anomaly-highlight {
            border: 1px solid var(--color-alert);
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.15);
        }

        /* Charts Responsive Grid */
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(580px, 1fr));
            gap: 24px;
        }

        .chart-panel {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .chart-header h3 {
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .chart-container {
            position: relative;
            height: 280px;
            width: 100%;
        }

        /* SVG Offline Fallback Styling */
        .chart-fallback {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .no-data {
            font-size: 0.9rem;
            color: var(--text-secondary);
            font-family: 'Inter', sans-serif;
        }

        /* Alert and Diagnostics Panel */
        .alert-panel {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .alert-feed {
            max-height: 200px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 12px;
            padding-right: 6px;
        }

        .alert-item {
            display: flex;
            align-items: flex-start;
            gap: 16px;
            padding: 14px 18px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            animation: fadeIn 0.4s ease-out;
        }

        .alert-item.alert-danger {
            background: rgba(239, 68, 68, 0.06);
            border-color: rgba(239, 68, 68, 0.15);
        }

        .alert-item.alert-warn {
            background: rgba(245, 158, 11, 0.06);
            border-color: rgba(245, 158, 11, 0.15);
        }

        .alert-icon {
            font-size: 1.2rem;
            line-height: 1;
        }

        .alert-info {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .alert-title {
            font-weight: 600;
            color: var(--text-primary);
        }

        .alert-msg {
            color: var(--text-secondary);
            font-size: 0.85rem;
        }

        .alert-time {
            font-size: 0.75rem;
            color: rgba(255, 255, 255, 0.25);
            margin-top: 2px;
        }

        /* Controls / Footer */
        footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-family: 'Inter', sans-serif;
            padding-top: 8px;
        }

        .refresh-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-glass);
            border-radius: 8px;
            color: var(--text-primary);
            padding: 8px 16px;
            font-size: 0.8rem;
            font-family: 'Inter', sans-serif;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .refresh-btn:hover {
            background: rgba(255, 255, 255, 0.15);
            border-color: rgba(255, 255, 255, 0.3);
        }

        /* Alert flashing Animation */
        @keyframes pulse-glow {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
            70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Custom Scrollbar for Alert Feed */
        .alert-feed::-webkit-scrollbar {
            width: 6px;
        }
        .alert-feed::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 99px;
        }
        .alert-feed::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 99px;
        }
        .alert-feed::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Top Header Panel -->
        <header class="glass-panel">
            <div class="brand-title">
                <h1>xpyrment Live Monitor</h1>
                <span>Real-Time Telemetry & Safe Shutoff Panel</span>
            </div>
            <div class="status-badge" id="system-badge">
                <div class="status-dot"></div>
                <span id="badge-text">SYSTEM ACTIVE</span>
            </div>
        </header>

        <!-- KPI Grid -->
        <section class="kpi-grid">
            <!-- KPI: Total Volume -->
            <div class="glass-panel kpi-card kpi-total">
                <div class="kpi-label">Exposed Units</div>
                <div class="kpi-value" id="kpi-total-val">0</div>
                <div class="kpi-delta" id="kpi-total-breakdown">Control: 0 | Treatment: 0</div>
            </div>
            <!-- KPI: Chi-Square SRM -->
            <div class="glass-panel kpi-card kpi-srm" id="card-srm">
                <div class="kpi-label">Cumulative SRM P-Value</div>
                <div class="kpi-value" id="kpi-srm-val">1.0000</div>
                <div class="kpi-delta" id="kpi-srm-status">SRM: HEALTHY</div>
            </div>
            <!-- KPI: SPRT likelihood -->
            <div class="glass-panel kpi-card kpi-sprt" id="card-sprt">
                <div class="kpi-label">SPRT Likelihood Ratio</div>
                <div class="kpi-value" id="kpi-sprt-val">1.00</div>
                <div class="kpi-delta" id="kpi-sprt-status">Boundary: 100.00 (Healthy)</div>
            </div>
            <!-- KPI: Volatility Z-Score -->
            <div class="glass-panel kpi-card kpi-volatility" id="card-anomaly">
                <div class="kpi-label">Latest Traffic Z-Score</div>
                <div class="kpi-value" id="kpi-anomaly-val">0.00</div>
                <div class="kpi-delta" id="kpi-anomaly-status">Traffic stable</div>
            </div>
        </section>

        <!-- Charts Section -->
        <section class="charts-grid">
            <!-- Chart 1: Allocation Splits -->
            <div class="glass-panel chart-panel">
                <div class="chart-header">
                    <h3>Cumulative Assignment Split</h3>
                </div>
                <div class="chart-container" id="chart1-container">
                    <canvas id="assignmentChart"></canvas>
                    <div id="chart1-fallback" class="chart-fallback" style="display: none;"></div>
                </div>
            </div>
            <!-- Chart 2: Martingale Likelihood Ratio -->
            <div class="glass-panel chart-panel">
                <div class="chart-header">
                    <h3>Sequential SPRT Martingale Path</h3>
                </div>
                <div class="chart-container" id="chart2-container">
                    <canvas id="sprtChart"></canvas>
                    <div id="chart2-fallback" class="chart-fallback" style="display: none;"></div>
                </div>
            </div>
        </section>

        <!-- Alerts Panel -->
        <section class="glass-panel alert-panel">
            <h3>Diagnostic Activity Feed & Alerts</h3>
            <div class="alert-feed" id="alert-feed">
                <!-- Dynamic alerts populated here -->
            </div>
        </section>

        <!-- Footer section -->
        <footer>
            <span>Status: Polling Active (Every 3s)</span>
            <button class="refresh-btn" onclick="fetchUpdate()">Refresh Now</button>
        </footer>
    </div>

    <!-- Live Polling & Drawing Script -->
    <script>
        let assignmentChart = null;
        let sprtChart = null;
        let isChartJsLoaded = typeof Chart !== 'undefined';

        // Custom responsive SVG generators if Chart.js fails
        function generateAssignmentSvg(trends) {
            if (!trends || !trends.labels || trends.labels.length === 0) {
                return '<div class="no-data">No cumulative trends data</div>';
            }
            const labels = trends.labels;
            const keys = Object.keys(trends.columns || {});
            if (keys.length === 0) return '<div class="no-data">No trend variables</div>';

            const width = 600;
            const height = 280;
            const pad = 50;

            const allVals = keys.flatMap(k => trends.columns[k]);
            const maxVal = Math.max(...allVals, 100) * 1.1;
            const minVal = 0;

            const getX = (idx) => pad + (idx / Math.max(1, labels.length - 1)) * (width - 2 * pad);
            const getY = (val) => height - pad - ((val - minVal) / (maxVal - minVal)) * (height - 2 * pad);

            let svg = `<svg viewBox="0 0 ${width} ${height}" style="width: 100%; height: 100%;">`;
            
            // Neon glow Filters & gradients
            svg += `
              <defs>
                <linearGradient id="svg-grad-ctrl" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="#0ea5e9" stop-opacity="0.25"/>
                  <stop offset="100%" stop-color="#0ea5e9" stop-opacity="0"/>
                </linearGradient>
                <linearGradient id="svg-grad-treat" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="#a855f7" stop-opacity="0.25"/>
                  <stop offset="100%" stop-color="#a855f7" stop-opacity="0"/>
                </linearGradient>
              </defs>
            `;

            // Grid Lines
            for (let i = 0; i <= 4; i++) {
                const y = pad + (i / 4) * (height - 2 * pad);
                const val = maxVal - (i / 4) * (maxVal - minVal);
                svg += `
                  <line x1="${pad}" y1="${y}" x2="${width - pad}" y2="${y}" stroke="rgba(255,255,255,0.04)" stroke-width="1" />
                  <text x="${pad - 10}" y="${y + 4}" fill="#64748b" font-size="9" text-anchor="end">${Math.round(val)}</text>
                `;
            }

            const colors = ["#0ea5e9", "#a855f7", "#10b981", "#ef4444"];
            const grads = ["#svg-grad-ctrl", "#svg-grad-treat"];

            keys.forEach((key, kIdx) => {
                const vals = trends.columns[key];
                const color = colors[kIdx % colors.length];
                const grad = grads[kIdx % grads.length] || "none";

                let pathStr = "";
                let areaStr = `M ${getX(0)} ${height - pad} `;

                vals.forEach((v, idx) => {
                    const x = getX(idx);
                    const y = getY(v);
                    if (idx === 0) {
                        pathStr += `M ${x} ${y} `;
                    } else {
                        pathStr += `L ${x} ${y} `;
                    }
                    areaStr += `L ${x} ${y} `;
                });
                areaStr += `L ${getX(vals.length - 1)} ${height - pad} Z`;

                svg += `<path d="${areaStr}" fill="url(${grad})" />`;
                svg += `<path d="${pathStr}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" />`;

                // Dots
                vals.forEach((v, idx) => {
                    svg += `<circle cx="${getX(idx)}" cy="${getY(v)}" r="4" fill="#ffffff" stroke="${color}" stroke-width="2" />`;
                });
            });

            // X Labels
            labels.forEach((lbl, idx) => {
                svg += `<text x="${getX(idx)}" y="${height - pad + 18}" fill="#64748b" font-size="9" text-anchor="middle">${lbl}</text>`;
            });

            svg += "</svg>";
            return svg;
        }

        function generateSprtSvg(seqData, alpha) {
            const ratios = (seqData && seqData.running_likelihood_ratios) || [1.0];
            const targetAlpha = alpha || 0.01;
            const threshold = 1.0 / targetAlpha;

            const width = 600;
            const height = 280;
            const pad = 50;

            const maxVal = Math.max(threshold * 1.2, ...ratios) || 120;
            const minVal = 0;

            const getX = (idx) => pad + (idx / Math.max(1, ratios.length - 1)) * (width - 2 * pad);
            const getY = (val) => height - pad - ((val - minVal) / (maxVal - minVal)) * (height - 2 * pad);

            let svg = `<svg viewBox="0 0 ${width} ${height}" style="width: 100%; height: 100%;">`;

            // Grid Lines
            for (let i = 0; i <= 4; i++) {
                const y = pad + (i / 4) * (height - 2 * pad);
                const val = maxVal - (i / 4) * (maxVal - minVal);
                svg += `
                  <line x1="${pad}" y1="${y}" x2="${width - pad}" y2="${y}" stroke="rgba(255,255,255,0.04)" stroke-width="1" />
                  <text x="${pad - 10}" y="${y + 4}" fill="#64748b" font-size="9" text-anchor="end">${Math.round(val)}</text>
                `;
            }

            // Target Threshold Rejection Line
            const threshY = getY(threshold);
            svg += `
              <line x1="${pad}" y1="${threshY}" x2="${width - pad}" y2="${threshY}" stroke="#ef4444" stroke-width="2" stroke-dasharray="6,4" />
              <text x="${width - pad - 10}" y="${threshY - 6}" fill="#ef4444" font-size="9" text-anchor="end" font-weight="bold">Rejection Boundary (1/α = ${Math.round(threshold)})</text>
            `;

            // Martingale path
            let pathStr = "";
            ratios.forEach((r, idx) => {
                const x = getX(idx);
                const y = getY(r);
                if (idx === 0) {
                    pathStr += `M ${x} ${y} `;
                } else {
                    pathStr += `L ${x} ${y} `;
                }
            });

            svg += `<path d="${pathStr}" fill="none" stroke="#a855f7" stroke-width="3" stroke-linecap="round" />`;

            // Dots
            ratios.forEach((r, idx) => {
                svg += `<circle cx="${getX(idx)}" cy="${getY(r)}" r="3.5" fill="#ffffff" stroke="#a855f7" stroke-width="1.5" />`;
            });

            svg += "</svg>";
            return svg;
        }

        // Live stats parser and card updates
        function updateCards(data) {
            const totalVal = (data.observed_counts || []).reduce((a, b) => a + b, 0);
            document.getElementById('kpi-total-val').innerText = totalVal.toLocaleString();

            const variants = data.variants || [];
            const counts = data.observed_counts || [];
            if (variants.length > 0) {
                const parts = variants.map((v, i) => `${v}: ${counts[i].toLocaleString()}`);
                document.getElementById('kpi-total-breakdown').innerText = parts.join(' | ');
            }

            // SRM Chi-Square
            const srmVal = data.cumulative_srm ? data.cumulative_srm.p_value : 1.0;
            const srmCard = document.getElementById('card-srm');
            document.getElementById('kpi-srm-val').innerText = srmVal.toFixed(4);
            
            if (data.cumulative_srm && data.cumulative_srm.srm_detected) {
                srmCard.classList.add('anomaly-highlight');
                document.getElementById('kpi-srm-status').innerText = 'SRM DETECTED (SHUTOFF)';
                document.getElementById('kpi-srm-status').style.color = 'var(--color-alert)';
            } else {
                srmCard.classList.remove('anomaly-highlight');
                document.getElementById('kpi-srm-status').innerText = 'SRM: HEALTHY';
                document.getElementById('kpi-srm-status').style.color = 'var(--text-secondary)';
            }

            // Sequential SPRT
            const sprtRatio = (data.sequential_srm && data.sequential_srm.running_likelihood_ratios) 
                ? data.sequential_srm.running_likelihood_ratios[data.sequential_srm.running_likelihood_ratios.length - 1] 
                : 1.0;
            const sprtCard = document.getElementById('card-sprt');
            document.getElementById('kpi-sprt-val').innerText = sprtRatio.toFixed(2);

            if (data.sequential_srm && data.sequential_srm.srm_detected) {
                sprtCard.classList.add('anomaly-highlight');
                document.getElementById('kpi-sprt-status').innerText = `SRM DETECTED at index ${data.sequential_srm.stopped_index}`;
                document.getElementById('kpi-sprt-status').style.color = 'var(--color-alert)';
            } else {
                sprtCard.classList.remove('anomaly-highlight');
                document.getElementById('kpi-sprt-status').innerText = 'Boundary: 100.00 (Healthy)';
                document.getElementById('kpi-sprt-status').style.color = 'var(--text-secondary)';
            }

            // Volatility / Traffic drop anomaly
            const anomalyData = data.traffic_anomaly || {};
            const zScore = anomalyData.z_score || 0.0;
            const anomalyCard = document.getElementById('card-anomaly');
            document.getElementById('kpi-anomaly-val').innerText = zScore.toFixed(2);

            if (anomalyData.anomaly_detected) {
                anomalyCard.classList.add('anomaly-highlight');
                document.getElementById('kpi-anomaly-status').innerText = 'TRAFFIC DROP ANOMALY';
                document.getElementById('kpi-anomaly-status').style.color = 'var(--color-alert)';
            } else {
                anomalyCard.classList.remove('anomaly-highlight');
                document.getElementById('kpi-anomaly-status').innerText = 'Traffic stable';
                document.getElementById('kpi-anomaly-status').style.color = 'var(--text-secondary)';
            }

            // System Active Pulsing Badge
            const badge = document.getElementById('system-badge');
            const badgeText = document.getElementById('badge-text');
            if (data.shutoff_triggered) {
                badge.classList.add('alert-active');
                badgeText.innerText = 'SHUTOFF TRIGGERED';
            } else {
                badge.classList.remove('alert-active');
                badgeText.innerText = 'SYSTEM ACTIVE';
            }
        }

        // Live alert feed updates
        function updateAlertFeed(data) {
            const feed = document.getElementById('alert-feed');
            feed.innerHTML = '';

            const alarms = data.alerts_triggered || [];
            
            // Include baseline logs if no active alerts
            if (alarms.length === 0) {
                feed.innerHTML = `
                    <div class="alert-item">
                        <span class="alert-icon">⚡</span>
                        <div class="alert-info">
                            <span class="alert-title">Diagnostic Feed Initialized</span>
                            <span class="alert-msg">Running diagnostics. Standard traffic ratio balanced. No structural anomalies detected in active partitions.</span>
                            <span class="alert-time">${new Date().toLocaleTimeString()}</span>
                        </div>
                    </div>
                `;
                return;
            }

            alarms.forEach(type => {
                let cl = 'alert-danger';
                let icon = '🚨';
                let title = type;
                let desc = '';

                if (type === 'TRAFFIC_DROP_ANOMALY') {
                    cl = 'alert-warn';
                    icon = '⚠️';
                    title = 'Traffic Dropout Alert';
                    desc = data.traffic_anomaly.message || 'Latest binned exposure volume indicates severe traffic drop.';
                } else if (type === 'SRM_SHUTOFF_TRIGGERED') {
                    title = 'SRM Shutoff Safety Triggered';
                    desc = data.cumulative_srm.message || 'Sample ratios mismatched beyond Wald Chi-Square bounds. Assignment halt recommended.';
                } else if (type === 'SEQUENTIAL_SRM_TRIGGERED') {
                    title = 'Sequential SPRT SRM Boundary Crossed';
                    desc = data.sequential_srm.message || 'Sequential likelihood ratio crossed type-I error martingale boundary.';
                }

                feed.innerHTML += `
                    <div class="alert-item ${cl}">
                        <span class="alert-icon">${icon}</span>
                        <div class="alert-info">
                            <span class="alert-title">${title}</span>
                            <span class="alert-msg">${desc}</span>
                            <span class="alert-time">${new Date().toLocaleTimeString()}</span>
                        </div>
                    </div>
                `;
            });
        }

        // Draw dynamic ChartJS or trigger SVG fallback
        function drawCharts(data) {
            const trends = data.trends || {};
            const labels = trends.labels || [];
            const cols = trends.columns || {};
            const keys = Object.keys(cols);

            if (!isChartJsLoaded) {
                // Trigger dynamic premium SVG rendering offline fallbacks
                document.getElementById('chart1-fallback').innerHTML = generateAssignmentSvg(trends);
                document.getElementById('chart2-fallback').innerHTML = generateSprtSvg(data.sequential_srm, 0.01);
                document.getElementById('chart1-fallback').style.display = 'block';
                document.getElementById('chart2-fallback').style.display = 'block';
                document.getElementById('assignmentChart').style.display = 'none';
                document.getElementById('sprtChart').style.display = 'none';
                return;
            }

            // Chart 1: Cumulative Assignment Split
            const datasetList = [];
            const colors = ['#0ea5e9', '#a855f7', '#10b981', '#f59e0b'];
            
            keys.forEach((key, idx) => {
                const color = colors[idx % colors.length];
                datasetList.push({
                    label: `Cumulative ${key}`,
                    data: cols[key],
                    borderColor: color,
                    backgroundColor: `${color}15`,
                    borderWidth: 3,
                    fill: true,
                    tension: 0.35,
                    pointBackgroundColor: '#ffffff',
                    pointBorderColor: color,
                    pointBorderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6
                });
            });

            if (assignmentChart) {
                assignmentChart.data.labels = labels;
                assignmentChart.data.datasets = datasetList;
                assignmentChart.update();
            } else {
                const ctx = document.getElementById('assignmentChart').getContext('2d');
                assignmentChart = new Chart(ctx, {
                    type: 'line',
                    data: { labels, datasets: datasetList },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
                            }
                        },
                        scales: {
                            x: { grid: { color: 'rgba(255, 255, 255, 0.04)' }, ticks: { color: '#94a3b8', font: { family: 'Inter' } } },
                            y: { grid: { color: 'rgba(255, 255, 255, 0.04)' }, ticks: { color: '#94a3b8', font: { family: 'Inter' } } }
                        }
                    }
                });
            }

            // Chart 2: SPRT Martingale path
            const sprtRatios = (data.sequential_srm && data.sequential_srm.running_likelihood_ratios) || [];
            const boundaryVal = 100.0;
            const thresholdArray = new Array(sprtRatios.length).fill(boundaryVal);

            if (sprtChart) {
                sprtChart.data.labels = sprtRatios.map((_, i) => `Step ${i}`);
                sprtChart.data.datasets[0].data = sprtRatios;
                sprtChart.data.datasets[1].data = thresholdArray;
                sprtChart.update();
            } else {
                const ctx = document.getElementById('sprtChart').getContext('2d');
                sprtChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: sprtRatios.map((_, i) => `Step ${i}`),
                        datasets: [
                            {
                                label: 'Likelihood Ratio (LR)',
                                data: sprtRatios,
                                borderColor: '#a855f7',
                                backgroundColor: 'rgba(168, 85, 247, 0.05)',
                                borderWidth: 3,
                                fill: true,
                                tension: 0.2,
                                pointBackgroundColor: '#ffffff',
                                pointBorderColor: '#a855f7',
                                pointBorderWidth: 1.5,
                                pointRadius: 3
                            },
                            {
                                label: 'Rejection Boundary (1/α)',
                                data: thresholdArray,
                                borderColor: '#ef4444',
                                borderWidth: 2,
                                borderDash: [6, 4],
                                fill: false,
                                pointRadius: 0
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
                            }
                        },
                        scales: {
                            x: { grid: { color: 'rgba(255, 255, 255, 0.04)' }, ticks: { color: '#94a3b8', font: { family: 'Inter' } } },
                            y: { grid: { color: 'rgba(255, 255, 255, 0.04)' }, ticks: { color: '#94a3b8', font: { family: 'Inter' } } }
                        }
                    }
                });
            }
        }

        // Periodic API polling function
        function fetchUpdate() {
            fetch('/api/data')
                .then(res => res.json())
                .then(res => {
                    if (res.status === 'success') {
                        updateCards(res.data);
                        updateAlertFeed(res.data);
                        drawCharts(res.data);
                    }
                })
                .catch(err => {
                    console.error('Failed to poll dashboard update:', err);
                });
        }

        // Run immediate check and schedule loop
        fetchUpdate();
        setInterval(fetchUpdate, 3000);
    </script>
</body>
</html>
"""
