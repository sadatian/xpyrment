"""Standalone beautiful report generation engine (Block 53).

Consolidates metrics, SRM allocation diagnostics, and covariate balances into premium,
self-contained Markdown reports and dynamic HTML dashboards.
"""

import os
from scipy.stats import chi2

from xpyrment.analyze.orchestrator import AnalysisResult


class ExperimentReportGenerator:
    """Generates premium standalone Markdown and HTML reports from experiment AnalysisResult instances."""

    def __init__(self, result: AnalysisResult, experiment_name: str = "A/B Experiment Report"):
        """Initializes the report generator.

        Args:
            result (AnalysisResult): The completed analysis result object.
            experiment_name (str): The logical name of the experiment.
            
        Raises:
            ValueError: If the analysis results are empty or invalid.
        """
        if result is None or not hasattr(result, "df_raw") or result.df_raw is None:
            raise ValueError("Invalid AnalysisResult provided to report generator.")
        
        self.result = result
        self.df_raw = result.df_raw
        self.alpha = result.alpha
        self.balance_checker = result.balance_checker
        self.experiment_name = experiment_name

        # Compute retrospective Sample Ratio Mismatch (SRM) stats
        self.control_n = 0
        self.treatment_n = 0
        self.srm_p_value = 1.0
        self.srm_passed = True

        if len(self.df_raw) > 0:
            row = self.df_raw.iloc[0]
            self.control_n = int(row.get("control_n", 0))
            self.treatment_n = int(row.get("treatment_n", 0))
            total_n = self.control_n + self.treatment_n

            if total_n > 0:
                expected_n = total_n / 2.0
                chi_sq = ((self.control_n - expected_n) ** 2 / expected_n) + ((self.treatment_n - expected_n) ** 2 / expected_n)
                self.srm_p_value = float(1.0 - chi2.cdf(chi_sq, df=1))
                self.srm_passed = self.srm_p_value >= 0.01  # Standard 0.01 SRM critical alpha

    def generate_markdown(self) -> str:
        """Generates a complete, beautiful GitHub-compatible Markdown summary card.

        Returns:
            str: Markdown card representation of the experiment results.
        """
        lines = []
        lines.append(f"# 📊 {self.experiment_name}")
        lines.append("")
        lines.append("## 📌 Executive Summary")
        lines.append(f"- **Nominal Significance Level (Alpha)**: `{self.alpha}`")
        lines.append(f"- **Total Samples**: `{self.control_n + self.treatment_n:,}` (Control: `{self.control_n:,}`, Treatment: `{self.treatment_n:,}`)")
        
        # SRM Status
        srm_status = "🟢 **PASSED**" if self.srm_passed else "🔴 **FAILED (Potential Bias!)**"
        lines.append(f"- **Sample Ratio Mismatch (SRM)**: {srm_status} (p-value: `{self.srm_p_value:.6f}`)")
        
        # Covariate balance
        if self.balance_checker is not None:
            imbalanced = [name for name, diag in self.balance_checker.diagnostics_.items() if abs(diag["smd"]) > 0.1]
            if imbalanced:
                lines.append(f"- **Covariate Balance**: ⚠️ **IMBALANCE DETECTED** in: `{', '.join(imbalanced)}`")
            else:
                lines.append("- **Covariate Balance**: 🟢 **ALL COVARIATES BALANCED** (SMD <= 0.1)")
        else:
            lines.append("- **Covariate Balance**: `Not Evaluated`")

        lines.append("")
        lines.append("## 📈 Metric Performance")
        lines.append("| Metric | Type | Control Mean | Treatment Mean | Relative Lift | P-Value | Significance | CUPED |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        for row in self.df_raw.itertuples(index=False):
            m_name = getattr(row, "metric_name")
            m_type = getattr(row, "metric_type", "mean")
            c_mean = getattr(row, "control_mean", 0.0)
            t_mean = getattr(row, "treatment_mean", 0.0)
            lift = getattr(row, "relative_lift", 0.0)
            p_val = getattr(row, "p_value", 1.0)
            cuped = "✅" if getattr(row, "cuped_applied", False) else "❌"

            is_sig = p_val < self.alpha
            sig_badge = "🌟 **Significant**" if is_sig else "Neutral"

            lines.append(
                f"| **{m_name}** | `{m_type}` | {c_mean:.4f} | {t_mean:.4f} | **{lift:+.2%}** | `{p_val:.5f}` | {sig_badge} | {cuped} |"
            )

        if self.balance_checker is not None:
            lines.append("")
            lines.append("## ⚖️ Covariate Balance Love Plot")
            lines.append("```text")
            lines.append(self.result.love_plot())
            lines.append("```")

        return "\n".join(lines)

    def generate_html(self) -> str:
        """Generates a premium, self-contained interactive HTML dashboard of the results.

        Returns:
            str: Portable HTML report page content with embedded modern CSS and layouts.
        """
        # Load premium custom SVG logo which also serves as the favicon
        svg_logo_text = self._get_svg_logo()

        favicon_tags = []
        if svg_logo_text:
            import base64
            svg_b64 = base64.b64encode(svg_logo_text.encode("utf-8")).decode("utf-8")
            favicon_tags.append(f'<link rel="icon" href="data:image/svg+xml;base64,{svg_b64}" sizes="any" type="image/svg+xml">')

        favicons_html = "\n    ".join(favicon_tags)

        logo_html = ""
        if svg_logo_text:
            cleaned_svg = svg_logo_text
            if cleaned_svg.startswith("<?xml"):
                end_xml_idx = cleaned_svg.find("?>")
                if end_xml_idx != -1:
                    cleaned_svg = cleaned_svg[end_xml_idx + 2:].strip()
            if cleaned_svg.startswith("<!DOCTYPE"):
                end_doc_idx = cleaned_svg.find(">")
                if end_doc_idx != -1:
                    cleaned_svg = cleaned_svg[end_doc_idx + 1:].strip()
            logo_html = f'<div class="header-logo-container">{cleaned_svg}</div>'

        # Formulate HTML metric table rows
        table_rows = []
        for row in self.df_raw.itertuples(index=False):
            m_name = getattr(row, "metric_name")
            m_type = getattr(row, "metric_type", "mean")
            c_mean = getattr(row, "control_mean", 0.0)
            t_mean = getattr(row, "treatment_mean", 0.0)
            lift = getattr(row, "relative_lift", 0.0)
            p_val = getattr(row, "p_value", 1.0)
            cuped_applied = getattr(row, "cuped_applied", False)

            is_sig = p_val < self.alpha
            sig_class = "sig-badge" if is_sig else "neutral-badge"
            sig_text = "SIGNIFICANT" if is_sig else "NEUTRAL"
            
            lift_class = "positive-lift" if lift > 0 else "negative-lift"
            lift_str = f"{lift:+.2%}" if lift != 0.0 else "0.00%"
            
            cuped_badge = '<span class="badge badge-success">CUPED Applied</span>' if cuped_applied else '<span class="badge badge-gray">Standard</span>'

            table_rows.append(f"""
            <tr>
                <td><strong>{m_name}</strong></td>
                <td><span class="badge-type">{m_type}</span></td>
                <td>{c_mean:.4f}</td>
                <td>{t_mean:.4f}</td>
                <td><span class="{lift_class}">{lift_str}</span></td>
                <td><code>{p_val:.5f}</code></td>
                <td><span class="{sig_class}">{sig_text}</span></td>
                <td>{cuped_badge}</td>
            </tr>
            """)

        # Covariate balance diagnostics section
        cov_rows = []
        if self.balance_checker is not None:
            for c_name, diag in self.balance_checker.diagnostics_.items():
                smd = diag["smd"]
                vr = diag["variance_ratio"]
                balanced = abs(smd) <= 0.1
                balance_class = "badge-success" if balanced else "badge-danger"
                balance_text = "BALANCED" if balanced else "IMBALANCED"

                cov_rows.append(f"""
                <tr>
                    <td>{c_name}</td>
                    <td>{diag['mean_control']:.4f}</td>
                    <td>{diag['mean_treatment']:.4f}</td>
                    <td><code class="{"smd-success" if balanced else "smd-fail"}">{smd:+.4f}</code></td>
                    <td><code>{vr:.4f}</code></td>
                    <td><span class="badge {balance_class}">{balance_text}</span></td>
                </tr>
                """)
        else:
            cov_rows.append('<tr><td colspan="6" class="text-center">No baseline covariates were specified.</td></tr>')

        # SRM card rendering
        srm_class = "card-success-border" if self.srm_passed else "card-danger-border"
        srm_badge_text = "PASSED" if self.srm_passed else "ALERT"

        love_plot_content = self.result.love_plot() if self.balance_checker is not None else "No covariate balance available."

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.experiment_name}</title>
    {favicons_html}
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0f172a;
            --panel-bg: #1e293b;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --teal: #0ea5e9;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --gray: #64748b;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            padding: 2rem;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        /* Header block */
        header {{
            margin-bottom: 2rem;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        header h1 {{
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(to right, #38bdf8, #0ea5e9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        header .meta-tag {{
            font-size: 0.875rem;
            color: var(--text-secondary);
        }}

        .header-logo-container svg {{
            width: 80px;
            height: auto;
            display: block;
        }}

        /* KPI Cards Grid */
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .card {{
            background-color: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}

        .card:hover {{
            transform: translateY(-2px);
            border-color: var(--teal);
        }}

        .card h3 {{
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}

        .card .value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-primary);
        }}

        .card .sub {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }}

        .card-success-border {{
            border-left: 5px solid var(--success);
        }}

        .card-danger-border {{
            border-left: 5px solid var(--danger);
        }}

        /* Badges */
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .badge-success {{
            background-color: rgba(16, 185, 129, 0.2);
            color: #34d399;
        }}

        .badge-danger {{
            background-color: rgba(239, 68, 68, 0.2);
            color: #f87171;
        }}

        .badge-gray {{
            background-color: rgba(100, 116, 139, 0.2);
            color: #94a3b8;
        }}

        .badge-type {{
            background-color: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-family: monospace;
        }}

        /* Table */
        .section-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text-primary);
        }}

        .panel {{
            background-color: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            overflow: hidden;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}

        th, td {{
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            color: var(--text-secondary);
            font-size: 0.875rem;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.05em;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover td {{
            background-color: rgba(255, 255, 255, 0.02);
        }}

        /* Lift Colors */
        .positive-lift {{
            color: #34d399;
            font-weight: 600;
        }}

        .negative-lift {{
            color: #f87171;
            font-weight: 600;
        }}

        /* Signficance Badges */
        .sig-badge {{
            background-color: rgba(16, 185, 129, 0.25);
            color: #34d399;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: bold;
        }}

        .neutral-badge {{
            background-color: rgba(100, 116, 139, 0.2);
            color: #94a3b8;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
        }}

        .smd-success {{
            color: #34d399;
            font-weight: bold;
        }}

        .smd-fail {{
            color: #fbbf24;
            font-weight: bold;
        }}

        /* Preformatted containers */
        pre {{
            background-color: #0b0f19;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            overflow-x: auto;
            color: #38bdf8;
            font-family: monospace;
            font-size: 0.875rem;
        }}

        .row {{
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
        }}

        .col {{
            flex: 1;
            min-width: 300px;
        }}

        .text-center {{
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div style="display: flex; align-items: center; gap: 1.25rem;">
                {logo_html}
                <div>
                    <h1>{self.experiment_name}</h1>
                    <div class="meta-tag">Generated by xpyrment on standard execution pipeline</div>
                </div>
            </div>
            <div class="meta-tag">Nominal Alpha: <strong>{self.alpha}</strong></div>
        </header>

        <!-- KPI Summary row -->
        <div class="grid">
            <div class="card">
                <h3>Total Assigned Users</h3>
                <div class="value">{self.control_n + self.treatment_n:,}</div>
                <div class="sub">Control: {self.control_n:,} | Treatment: {self.treatment_n:,}</div>
            </div>
            
            <div class="card {srm_class}">
                <h3>SRM Safety Shield</h3>
                <div class="value" style="color: {'#34d399' if self.srm_passed else '#f87171'}">
                    {srm_badge_text}
                </div>
                <div class="sub">Pearson Chi-Square p-value: {self.srm_p_value:.6f}</div>
            </div>

            <div class="card">
                <h3>Baseline Covariates</h3>
                <div class="value">
                    {len(self.balance_checker.diagnostics_) if self.balance_checker is not None else 0}
                </div>
                <div class="sub">
                    { "Diagnostics fully evaluated" if self.balance_checker is not None else "No registered covariates" }
                </div>
            </div>
        </div>

        <!-- Main dashboard layouts -->
        <div class="panel">
            <div class="section-title">📊 Statistical Metric Performance Summary</div>
            <table>
                <thead>
                    <tr>
                        <th>Metric Name</th>
                        <th>Type</th>
                        <th>Control Mean</th>
                        <th>Treatment Mean</th>
                        <th>Relative Lift</th>
                        <th>P-Value</th>
                        <th>Significance</th>
                        <th>CUPED</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(table_rows)}
                </tbody>
            </table>
        </div>

        <div class="row">
            <div class="col">
                <div class="panel">
                    <div class="section-title">⚖️ Baseline Covariate Balance Detail</div>
                    <table>
                        <thead>
                            <tr>
                                <th>Covariate Name</th>
                                <th>Mean (Ctrl)</th>
                                <th>Mean (Trt)</th>
                                <th>SMD</th>
                                <th>Var Ratio</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(cov_rows)}
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="col">
                <div class="panel" style="height: 100%;">
                    <div class="section-title">📉 Covariate Balance ASCII Love Plot</div>
                    <pre>{love_plot_content}</pre>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        return html_template

    def save_html(self, filepath: str):
        """Saves the beautifully compiled HTML dashboard report to a local file.

        Args:
            filepath (str): Full target file path.
        """
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.generate_html())

    def save_markdown(self, filepath: str):
        """Saves the GitHub-compatible Markdown summary report to a local file.

        Args:
            filepath (str): Full target file path.
        """
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown())

    def _get_icon_base64(self, filename: str) -> str:
        """Removed as icons folder has been deleted."""
        return ""

    def _get_svg_logo(self) -> str:
        """Helper to load and return the raw SVG package logo."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            pkg_dir = os.path.dirname(current_dir)
            svg_path = os.path.join(pkg_dir, "assets", "images", "xpyrment_logo.svg")
            if os.path.exists(svg_path):
                with open(svg_path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception:
            pass
        return ""
