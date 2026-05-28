import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional
import numpy as np
import pandas as pd
from xpyrment.simulation import generate_ab_data

logger = logging.getLogger(__name__)

def numpy_to_python(obj: Any) -> Any:
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


def get_doe_design_summaries() -> list[dict]:
    """Retrieves summaries of all design of experiments classes dynamically."""
    import inspect
    try:
        import xpyrment.design.doe as doe_pkg
    except ImportError:
        return []
        
    designs = {}
    names = doe_pkg.__all__ if hasattr(doe_pkg, "__all__") else dir(doe_pkg)
    for name in names:
        if name.startswith("_"):
            continue
        if name in ("DesignMatrix", "CarryoverDecomposition"):
            continue
        cls = getattr(doe_pkg, name, None)
        if cls is not None and inspect.isclass(cls):
            # Filter out any classes imported from other modules
            if cls.__module__.startswith("xpyrment.design.doe"):
                doc = inspect.getdoc(cls) or ""
                first_line = doc.split("\n")[0] if doc else "No description available."
                first_line = first_line.replace("$", "").replace("|", "\\|")
                designs[name] = {"summary": first_line}
    return [{"name": k, "desc": v["summary"]} for k, v in sorted(designs.items())]


def run_network_partition(df: pd.DataFrame) -> int:
    """Helper to partition the graph based on network structure."""
    from xpyrment.network.partition import EntropyBalancedGraphPartitioner
    num_nodes = len(df)
    
    # Generate a mock adjacency dict for the loaded rows
    adj = {i: [] for i in range(num_nodes)}
    rng = np.random.default_rng(42)
    for i in range(num_nodes):
        # Connect each node to 2-4 random neighbors
        n_edges = int(rng.integers(2, 5))
        targets = rng.choice(num_nodes, size=min(n_edges, num_nodes), replace=False)
        for target in targets:
            if target != i:
                adj[i].append(int(target))
                adj[int(target)].append(i)
                
    # Deduplicate neighbor lists
    adj = {node: list(set(neighbors)) for node, neighbors in adj.items()}
    
    partitioner = EntropyBalancedGraphPartitioner(adjacency_dict=adj, gamma=0.1)
    clusters = partitioner.fit_predict()
    return len(set(clusters.values()))


def format_hte_results(res: dict) -> str:
    """Formats heterogeneous treatment effects results from ANOVA."""
    hte = res.get("heterogeneous_treatment_effects", [])
    if hte:
        return "\n".join(
            f"{item['metric']} x {item['covariate']}: p={item['p_value']:.4f}"
            for item in hte
        )
    return "No significant heterogeneous treatment effects detected (all p >= 0.05)."


def run_anova_interaction_detection(df: pd.DataFrame, y_col: str) -> str:
    """Helper to run ANOVA interaction detection on shared dataframe."""
    from xpyrment.analyze.orchestrator import setup
    from xpyrment.interactions.detector import InteractionDetector
    
    covariates = ["pre_revenue", "pre_impressions", "pre_clicks"]
    covariates = [c for c in covariates if c in df.columns]
    
    # Create standard experiment structure and register target metric
    exp = setup(df, treatment_col="variant", id_col="user_id", covariates=covariates)
    exp.register_metric(y_col)
    
    detector = InteractionDetector(exp)
    res = detector.detect_all()
    return format_hte_results(res)


class XpyrmentHubServer:
    """Primary Hub Server acting as a launcher for the suite of modules."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7000, log_usage: bool = False):
        self.host = host
        self.port_requested = port
        self.log_usage = log_usage

        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.port: int = port
        self._is_running = False

        self._lock = threading.Lock()
        self.shared_data: Optional[pd.DataFrame] = None
        self.dataset_name: str = "None"

        # Modules specific server references
        self.monitoring_server = None

    def start(self) -> None:
        if self._is_running:
            return

        server_instance = self

        class HubHTTPRequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                if server_instance.log_usage:
                    logger.info(format, *args)
                else:
                    logger.debug(format, *args)

            def do_GET(self) -> None:
                if self.path == "/" or self.path == "/index.html":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(server_instance.get_html_content().encode("utf-8"))
                elif self.path == "/api/data/status":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    with server_instance._lock:
                        payload = {
                            "status": "success",
                            "dataset_name": server_instance.dataset_name,
                            "has_data": server_instance.shared_data is not None,
                            "columns": list(server_instance.shared_data.columns) if server_instance.shared_data is not None else [],
                            "rows": len(server_instance.shared_data) if server_instance.shared_data is not None else 0
                        }
                    self.wfile.write(json.dumps(payload).encode("utf-8"))
                else:
                    self.send_response(404)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"Not Found")

            def do_POST(self) -> None:
                content_length = int(self.headers.get("Content-Length", 0))
                post_data = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else ""
                try:
                    params = json.loads(post_data) if post_data else {}
                except Exception:
                    params = {}

                if self.path == "/api/data/simulate":
                    try:
                        n_samples = int(params.get("n_samples", 1000))
                        df = generate_ab_data(n_samples=n_samples)
                        with server_instance._lock:
                            server_instance.shared_data = df
                            server_instance.dataset_name = f"Simulated ({n_samples} rows)"
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "success", "message": "Simulation successful"}).encode("utf-8"))
                    except Exception as e:
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))

                # Implement initiating various backend modules using the shared_data cache
                elif self.path == "/api/module/monitoring/start":
                    try:
                        port = server_instance.start_monitoring_server(server_instance.host, 0)
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "success", "port": port}).encode("utf-8"))
                    except Exception as e:
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
                elif self.path.startswith("/api/module/design/generate"):
                    try:
                        data = get_doe_design_summaries()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "success", "data": data}).encode("utf-8"))
                    except Exception as e:
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
                elif self.path == "/api/module/quasi/analyze":
                    if server_instance.shared_data is None:
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "error", "message": "No data loaded."}).encode("utf-8"))
                        return
                    try:
                        from xpyrment.quasi.diff_in_diff import fit_ols
                        df = server_instance.shared_data.copy()
                        y_col = params.get("y_col", "revenue" if "revenue" in df.columns else df.columns[-1])
                        if "converted" in df.columns and "revenue" not in df.columns:
                            y_col = "converted"
                        x_cols = params.get("x_cols", ["variant"] if "variant" in df.columns else [df.columns[0]])
                        if "variant" in df.columns:
                            df["variant"] = df["variant"].map({"treatment": 1, "control": 0}).fillna(0)
                        X = df[x_cols].to_numpy()
                        y = df[y_col].to_numpy()
                        res = fit_ols(X, y)
                        results = {"intercept": {"coef": res["beta"][0], "p_value": res["p_values"][0], "se": res["standard_errors"][0]}}
                        for idx, col in enumerate(x_cols):
                            results[col] = {"coef": res["beta"][idx+1], "p_value": res["p_values"][idx+1], "se": res["standard_errors"][idx+1]}
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "success", "data": results}).encode("utf-8"))
                    except Exception as e:
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
                elif self.path == "/api/module/balance":
                    from xpyrment.validate.balance import check_covariate_balance
                    if server_instance.shared_data is None:
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "error", "message": "No data loaded."}).encode("utf-8"))
                        return
                    try:
                        group_col = params.get("group_col", "variant")
                        covariates = params.get("covariates", [])
                        results = check_covariate_balance(server_instance.shared_data, group_col, covariates)
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "success", "data": results}).encode("utf-8"))
                    except Exception as e:
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
                elif self.path == "/api/module/personalize/train":
                    if server_instance.shared_data is None:
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "error", "message": "No data loaded."}).encode("utf-8"))
                        return
                    try:
                        from xpyrment.personalize.meta_learners import TLearner
                        df = server_instance.shared_data.copy()
                        y_col = "revenue" if "revenue" in df.columns else df.columns[-1]
                        X = df.drop(columns=[y_col, "variant", "user_id"], errors="ignore").to_numpy()
                        T = df["variant"].map({"treatment": 1, "control": 0}).fillna(0).to_numpy()
                        y = df[y_col].to_numpy()
                        learner = TLearner()
                        learner.fit(X, T, y)
                        cate = learner.estimate_effect(X)
                        avg_cate = float(cate.mean())
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "success", "message": f"T-Learner trained successfully. Average CATE: {avg_cate:.4f}"}).encode("utf-8"))
                    except Exception as e:
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
                elif self.path == "/api/module/network/cluster":
                    if server_instance.shared_data is None:
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "error", "message": "No data loaded."}).encode("utf-8"))
                        return
                    try:
                        num_clusters = run_network_partition(server_instance.shared_data)
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "success", "message": f"Graph partitioned successfully into {num_clusters} clusters."}).encode("utf-8"))
                    except Exception as e:
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
                elif self.path == "/api/module/interactions/anova":
                    if server_instance.shared_data is None:
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "error", "message": "No data loaded."}).encode("utf-8"))
                        return
                    try:
                        df = server_instance.shared_data.copy()
                        y_col = "revenue" if "revenue" in df.columns else df.columns[-1]
                        out = run_anova_interaction_detection(df, y_col)
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "success", "message": f"ANOVA Calculated:\n{out}"}).encode("utf-8"))
                    except Exception as e:
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
                else:
                    self.send_response(404)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"Not Found")

        self.server = HTTPServer((self.host, self.port_requested), HubHTTPRequestHandler)
        self.port = self.server.server_address[1]

        self._is_running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        logger.info(f"xpyrment Primary Hub started at http://{self.host}:{self.port}")

    def _run_server(self) -> None:
        try:
            if self.server:
                self.server.serve_forever()
        except Exception as e:
            logger.error(f"Error in hub server loop: {e}", exc_info=True)

    def stop(self) -> None:
        if not self._is_running or not self.server:
            return

        self.server.shutdown()
        self.server.server_close()

        if self.thread:
            self.thread.join(timeout=5.0)

        self._is_running = False
        self.server = None
        self.thread = None
        logger.info("xpyrment Primary Hub server stopped cleanly.")

    def get_html_content(self) -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>xpyrment Primary Hub</title>
    <!-- Premium Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
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
            --sidebar-width: 280px;
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
            height: 100vh;
            display: flex;
            overflow: hidden;
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

        /* Sidebar */
        .sidebar {
            width: var(--sidebar-width);
            height: 100vh;
            border-right: 1px solid var(--border-glass);
            background: rgba(10, 10, 18, 0.6);
            backdrop-filter: blur(20px);
            display: flex;
            flex-direction: column;
            z-index: 100;
        }

        .sidebar-header {
            padding: 24px;
            border-bottom: 1px solid var(--border-glass);
        }

        .sidebar-header h1 {
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #f8fafc 30%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .sidebar-nav {
            flex: 1;
            overflow-y: auto;
            padding: 16px 0;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .nav-item {
            padding: 12px 24px;
            cursor: pointer;
            color: var(--text-secondary);
            font-size: 0.95rem;
            font-weight: 500;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .nav-item:hover {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
        }

        .nav-item.active {
            background: linear-gradient(90deg, rgba(168, 85, 247, 0.15) 0%, transparent 100%);
            border-left: 3px solid var(--color-treatment);
            color: var(--text-primary);
            font-weight: 600;
        }

        .data-panel {
            padding: 20px 24px;
            border-top: 1px solid var(--border-glass);
            background: rgba(0, 0, 0, 0.2);
        }

        .data-panel-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
            margin-bottom: 12px;
            font-weight: 600;
        }

        .data-status {
            font-size: 0.85rem;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--color-warn);
        }
        .status-dot.active {
            background: var(--color-healthy);
            box-shadow: 0 0 8px var(--color-healthy);
        }

        .btn {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border-glass);
            border-radius: 8px;
            color: var(--text-primary);
            padding: 10px 16px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            width: 100%;
            text-align: center;
            font-family: 'Inter', sans-serif;
            margin-bottom: 8px;
        }

        .btn:hover {
            background: rgba(255, 255, 255, 0.15);
            border-color: rgba(255, 255, 255, 0.3);
        }

        .btn-primary {
            background: rgba(168, 85, 247, 0.15);
            border-color: rgba(168, 85, 247, 0.3);
            color: #d8b4fe;
        }

        /* Main Content */
        .main-content {
            flex: 1;
            padding: 32px;
            overflow-y: auto;
            position: relative;
        }

        .module-view {
            display: none;
            flex-direction: column;
            gap: 24px;
            height: 100%;
        }

        .module-view.active {
            display: flex;
            animation: fadeIn 0.3s ease-out;
        }

        .module-header h2 {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .module-header p {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }

        .results-box {
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 16px;
            font-family: monospace;
            font-size: 0.85rem;
            color: var(--text-secondary);
            min-height: 200px;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Continuous Monitoring Frame */
        #monitoring-frame {
            width: 100%;
            height: calc(100vh - 120px);
            border: none;
            border-radius: 18px;
            display: none;
        }

    </style>
</head>
<body>

    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-header">
            <h1>xpyrment Hub</h1>
        </div>
        <div class="sidebar-nav">
            <div class="nav-item active" onclick="switchView('overview')">📊 Overview</div>
            <div class="nav-item" onclick="switchView('monitoring')">📈 Continuous Monitoring</div>
            <div class="nav-item" onclick="switchView('design')">📐 Design of Experiments</div>
            <div class="nav-item" onclick="switchView('quasi')">🔬 Quasi-Experiments</div>
            <div class="nav-item" onclick="switchView('governance')">🛡️ Governance & Validate</div>
            <div class="nav-item" onclick="switchView('personalize')">🎯 Personalization & Bandits</div>
            <div class="nav-item" onclick="switchView('network')">🕸️ Network & Federated</div>
            <div class="nav-item" onclick="switchView('interactions')">🧩 Interactions & Interpret</div>
        </div>

        <div class="data-panel">
            <div class="data-panel-title">Active Data Source</div>
            <div class="data-status">
                <div class="status-dot" id="data-dot"></div>
                <span id="data-name">None</span>
            </div>
            <button class="btn" onclick="simulateData()">Generate Simulated Data</button>
            <button class="btn" onclick="alert('CSV Upload coming soon!')">Upload CSV</button>
        </div>
    </div>

    <!-- Main Content -->
    <div class="main-content">

        <!-- Overview View -->
        <div id="view-overview" class="module-view active">
            <div class="module-header">
                <h2>Welcome to Xpyrment Primary Hub</h2>
                <p>Enterprise-grade digital experimentation, causal inference, and classical DoE.</p>
            </div>
            <div class="glass-panel">
                <p style="color: var(--text-secondary); line-height: 1.6;">
                    Select a module from the sidebar to begin. <br><br>
                    <strong>Note:</strong> Most modules require an active data source. Use the data panel in the bottom left to generate synthetic data or upload a dataset.
                </p>
            </div>
        </div>

        <!-- Continuous Monitoring View -->
        <div id="view-monitoring" class="module-view">
            <div class="module-header">
                <h2>Continuous Monitoring</h2>
                <p>Real-time Sequential Testing, SPRT, and Live Telemetry.</p>
            </div>
            <div class="glass-panel" id="monitoring-launcher">
                <p style="margin-bottom: 20px; color: var(--text-secondary);">The live monitoring dashboard runs as a dedicated background server to handle high-frequency streams.</p>
                <button class="btn btn-primary" style="width: auto; padding: 12px 24px;" onclick="startMonitoring()">Launch Live Dashboard</button>
            </div>
            <iframe id="monitoring-frame"></iframe>
        </div>

        <!-- Design View -->
        <div id="view-design" class="module-view">
            <div class="module-header">
                <h2>Design of Experiments (DoE)</h2>
                <p>Factorial, DSD, Taguchi, and Sample Size calculations.</p>
            </div>
            <div class="grid-2">
                <div class="glass-panel">
                    <h3 style="margin-bottom: 16px;">Configure Design</h3>
                    <button class="btn btn-primary" onclick="fetchDesigns()">List Available Designs</button>
                </div>
                <div class="glass-panel">
                    <h3 style="margin-bottom: 16px;">Results</h3>
                    <div class="results-box" id="design-results">Results will appear here...</div>
                </div>
            </div>
        </div>

        <!-- Quasi View -->
        <div id="view-quasi" class="module-view">
            <div class="module-header">
                <h2>Quasi-Experiments</h2>
                <p>Difference-in-Differences, Synthetic Controls, and Propensity Score Matching.</p>
            </div>
            <div class="grid-2">
                <div class="glass-panel">
                    <h3 style="margin-bottom: 16px;">Configure Analysis</h3>
                    <div style="margin-bottom: 16px;">
                        <label style="display:block; margin-bottom:8px; font-size:0.85rem; color:var(--text-secondary);">Target Metric (Y)</label>
                        <input type="text" id="quasi-y" value="revenue" style="width:100%; padding:8px; background:rgba(0,0,0,0.3); border:1px solid var(--border-glass); color:white; border-radius:4px;">
                    </div>
                    <div style="margin-bottom: 16px;">
                        <label style="display:block; margin-bottom:8px; font-size:0.85rem; color:var(--text-secondary);">Treatment/Variant (X)</label>
                        <input type="text" id="quasi-x" value="variant" style="width:100%; padding:8px; background:rgba(0,0,0,0.3); border:1px solid var(--border-glass); color:white; border-radius:4px;">
                    </div>
                    <button class="btn btn-primary" onclick="runQuasi()">Run OLS / DiD Analysis</button>
                </div>
                <div class="glass-panel">
                    <h3 style="margin-bottom: 16px;">Results</h3>
                    <div class="results-box" id="quasi-results">Results will appear here...</div>
                </div>
            </div>
        </div>

        <!-- Governance View -->
        <div id="view-governance" class="module-view">
            <div class="module-header">
                <h2>Governance & Validation</h2>
                <p>SRM Diagnostics and Pre-experiment Covariate Balance.</p>
            </div>
            <div class="grid-2">
                <div class="glass-panel">
                    <h3 style="margin-bottom: 16px;">Covariate Balance Check</h3>
                    <div style="margin-bottom: 16px;">
                        <label style="display:block; margin-bottom:8px; font-size:0.85rem; color:var(--text-secondary);">Group Column</label>
                        <input type="text" id="gov-group" value="variant" style="width:100%; padding:8px; background:rgba(0,0,0,0.3); border:1px solid var(--border-glass); color:white; border-radius:4px;">
                    </div>
                    <div style="margin-bottom: 16px;">
                        <label style="display:block; margin-bottom:8px; font-size:0.85rem; color:var(--text-secondary);">Covariates (Comma separated)</label>
                        <input type="text" id="gov-covs" value="pre_revenue,pre_clicks" style="width:100%; padding:8px; background:rgba(0,0,0,0.3); border:1px solid var(--border-glass); color:white; border-radius:4px;">
                    </div>
                    <button class="btn btn-primary" onclick="runBalance()">Run SMD Balance Check</button>
                </div>
                <div class="glass-panel">
                    <h3 style="margin-bottom: 16px;">Results</h3>
                    <div class="results-box" id="gov-results">Results will appear here...</div>
                </div>
            </div>
        </div>

        <!-- Personalize View -->
        <div id="view-personalize" class="module-view">
            <div class="module-header">
                <h2>Personalization & Bandits</h2>
                <p>Heterogeneous Treatment Effects (HTE) and Multi-Armed Bandits.</p>
            </div>
            <div class="glass-panel">
                <p>Select variables and target estimators (T-Learner, DragonNet) for training.</p>
                <button class="btn btn-primary" style="width:auto; margin-top:20px;" onclick="runPersonalize()">Train Meta-Learner</button>
                <div class="results-box" id="pers-results" style="margin-top:20px;"></div>
            </div>
        </div>

        <!-- Network View -->
        <div id="view-network" class="module-view">
            <div class="module-header">
                <h2>Network & Federated</h2>
                <p>Cluster Randomization, Graph Partitioning, and Secure Multiparty Computation.</p>
            </div>
            <div class="glass-panel">
                <p>Configure graph edges and run Label Propagation for clustered assignments.</p>
                <button class="btn btn-primary" style="width:auto; margin-top:20px;" onclick="runNetwork()">Run Partitioning</button>
                <div class="results-box" id="net-results" style="margin-top:20px;"></div>
            </div>
        </div>

        <!-- Interactions View -->
        <div id="view-interactions" class="module-view">
            <div class="module-header">
                <h2>Interactions & Interpret</h2>
                <p>Factorial ANOVA, SHAP Values, and Business Decision Frameworks.</p>
            </div>
            <div class="glass-panel">
                <p>Calculate multi-factor ANOVA to resolve confounding and interpret interaction effects.</p>
                <button class="btn btn-primary" style="width:auto; margin-top:20px;" onclick="runInteractions()">Compute ANOVA</button>
                <div class="results-box" id="int-results" style="margin-top:20px;"></div>
            </div>
        </div>

    </div>

    <script>
        function switchView(viewId) {
            document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
            event.currentTarget.classList.add('active');

            document.querySelectorAll('.module-view').forEach(el => el.classList.remove('active'));
            document.getElementById('view-' + viewId).classList.add('active');
        }

        function checkDataStatus() {
            fetch('/api/data/status')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('data-name').innerText = data.dataset_name;
                    if (data.has_data) {
                        document.getElementById('data-dot').classList.add('active');
                    } else {
                        document.getElementById('data-dot').classList.remove('active');
                    }
                });
        }

        // Polling status
        setInterval(checkDataStatus, 5000);
        checkDataStatus();

        function simulateData() {
            document.getElementById('data-name').innerText = "Generating...";
            fetch('/api/data/simulate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({n_samples: 5000})
            })
            .then(r => r.json())
            .then(data => {
                if(data.status === 'success') {
                    checkDataStatus();
                    alert("Simulated dataset (5000 rows) generated and loaded into memory.");
                } else {
                    alert("Error: " + data.message);
                }
            });
        }

        function startMonitoring() {
            fetch('/api/module/monitoring/start', {method: 'POST'})
            .then(r => r.json())
            .then(data => {
                if(data.status === 'success') {
                    document.getElementById('monitoring-launcher').style.display = 'none';
                    const frame = document.getElementById('monitoring-frame');
                    frame.src = `http://${window.location.hostname}:${data.port}`;
                    frame.style.display = 'block';
                } else {
                    alert("Error starting monitoring: " + data.message);
                }
            });
        }

        function fetchDesigns() {
            document.getElementById('design-results').innerText = "Loading...";
            fetch('/api/module/design/generate', {method: 'POST'})
            .then(r => r.json())
            .then(data => {
                if(data.status === 'success') {
                    let text = "Available Experimental Designs:\n\n";
                    data.data.forEach(d => {
                        text += `[${d.name}]\n  ${d.desc}\n\n`;
                    });
                    document.getElementById('design-results').innerText = text;
                } else {
                    document.getElementById('design-results').innerText = "Error: " + data.message;
                }
            });
        }

        function runQuasi() {
            const y = document.getElementById('quasi-y').value;
            const x = document.getElementById('quasi-x').value;
            document.getElementById('quasi-results').innerText = "Running OLS...";
            fetch('/api/module/quasi/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({y_col: y, x_cols: x.split(',')})
            })
            .then(r => r.json())
            .then(data => {
                if(data.status === 'success') {
                    document.getElementById('quasi-results').innerText = JSON.stringify(data.data, null, 2);
                } else {
                    document.getElementById('quasi-results').innerText = "Error: " + data.message;
                }
            });
        }

        function runBalance() {
            const group = document.getElementById('gov-group').value;
            const covs = document.getElementById('gov-covs').value;
            document.getElementById('gov-results').innerText = "Running...";
            fetch('/api/module/balance', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({group_col: group, covariates: covs.split(',')})
            })
            .then(r => r.json())
            .then(data => {
                if(data.status === 'success') {
                    document.getElementById('gov-results').innerText = JSON.stringify(data.data, null, 2);
                } else {
                    document.getElementById('gov-results').innerText = "Error: " + data.message;
                }
            });
        }

        function runPersonalize() {
            document.getElementById('pers-results').innerText = "Running...";
            fetch('/api/module/personalize/train', {method: 'POST'})
            .then(r => r.json())
            .then(data => { document.getElementById('pers-results').innerText = data.message; });
        }

        function runNetwork() {
            document.getElementById('net-results').innerText = "Running...";
            fetch('/api/module/network/cluster', {method: 'POST'})
            .then(r => r.json())
            .then(data => { document.getElementById('net-results').innerText = data.message; });
        }

        function runInteractions() {
            document.getElementById('int-results').innerText = "Running...";
            fetch('/api/module/interactions/anova', {method: 'POST'})
            .then(r => r.json())
            .then(data => { document.getElementById('int-results').innerText = data.message; });
        }

    </script>
</body>
</html>"""

    def start_monitoring_server(self, host: str, port: int) -> int:
        """Starts the background continuous monitoring dashboard if not running."""
        from xpyrment.run.monitor import LiveMonitor
        from xpyrment.run.webui import ExperimentDashboardServer

        if self.monitoring_server is not None and self.monitoring_server._is_running:
             return self.monitoring_server.port

        df = self.shared_data.copy() if self.shared_data is not None else pd.DataFrame(columns=["unit_id", "exposed_at", "variant"])
        monitor = LiveMonitor(df=df, time_col="exposed_at")

        self.monitoring_server = ExperimentDashboardServer(
            monitor=monitor,
            expected_ratios=[0.5, 0.5],
            host=host,
            port=port
        )
        self.monitoring_server.start()
        return self.monitoring_server.port
