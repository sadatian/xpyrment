"""Live telemetry monitoring, cumulative traffic accumulation, and telemetry audit feeds.

This module provides the `LiveMonitor` class, which aggregates exposure logs into time-series
bins. This enables real-time diagnostic auditing to detect mid-experiment routing anomalies
and telemetry dropouts.
"""

import json
import logging
import urllib.request
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WebhookAlertDispatcher:
    """Manages pluggable webhook registrations and broadcasts alerts to external systems."""

    def __init__(self) -> None:
        # Categorized registry of handlers to enable independent failures
        self._handlers: Dict[str, List[Callable[[str, str, Dict[str, Any]], None]]] = {
            "slack": [],
            "email": [],
            "custom": []
        }
        self._custom_callables: List[Callable[[str, str, Dict[str, Any]], None]] = []

    def register_slack(self, url: str) -> None:
        """Registers a Slack webhook URL."""
        def slack_handler(alert_type: str, message: str, payload: Dict[str, Any]) -> None:
            slack_payload = {
                "text": f"🚨 *[xpyrment Alert]* {message}\n*Type*: `{alert_type}`\n*Payload*: {json.dumps(payload, indent=2)}"
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(slack_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5.0) as response:
                response.read()
        self._handlers["slack"].append(slack_handler)

    def register_email(self, url: str) -> None:
        """Registers an Email webhook URL (JSON POST to an email notification service)."""
        def email_handler(alert_type: str, message: str, payload: Dict[str, Any]) -> None:
            email_payload = {
                "subject": f"[xpyrment Alert] {alert_type}",
                "body": f"{message}\n\nDetails:\n{json.dumps(payload, indent=2)}"
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(email_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5.0) as response:
                response.read()
        self._handlers["email"].append(email_handler)

    def register_custom(self, url: str) -> None:
        """Registers a custom generic HTTP POST webhook URL."""
        def custom_handler(alert_type: str, message: str, payload: Dict[str, Any]) -> None:
            custom_payload = {
                "alert_type": alert_type,
                "message": message,
                "details": payload
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(custom_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5.0) as response:
                response.read()
        self._handlers["custom"].append(custom_handler)

    def register_callback(self, callback: Callable[[str, str, Dict[str, Any]], None]) -> None:
        """Registers a custom callback function for local testing or custom notification logic."""
        self._custom_callables.append(callback)

    def dispatch(self, alert_type: str, message: str, payload: Dict[str, Any]) -> None:
        """Dispatches an alert to all registered handlers and callbacks.

        Guarantees failure isolation: an exception in one handler does not halt execution of others.
        """
        for category, handlers in self._handlers.items():
            for handler in handlers:
                try:
                    handler(alert_type, message, payload)
                except Exception as e:
                    logger.error(f"Failed to dispatch to {category} handler: {e}", exc_info=True)

        for callback in self._custom_callables:
            try:
                callback(alert_type, message, payload)
            except Exception as e:
                logger.error(f"Failed to execute custom callback: {e}", exc_info=True)


class LiveMonitor:
    r"""Provides active monitoring of experimental groups to build diagnostics dashboards.

    Accumulates exposure logs chronologically to generate time-series metrics. By evaluating traffic
    trends in real time, experimenters can verify that the randomization splits remain stable and that
    no asymmetric telemetry dropouts or scheduling anomalies occur.

    ??? mathbox "Temporal Binning and Accumulation Theory"

        Let there be $k$ variants. Let the experimental logs be grouped into sequential, non-overlapping temporal
        intervals (bins) $t \in \{1, 2, \dots, H\}$ (such as hours or days).
        - Let $n_v(t)$ be the number of unique units newly exposed to variant $v$ during time bin $t$.
        - The cumulative traffic $C_v(t)$ for variant $v$ up to time bin $t$ is calculated as:
          $$
          C_v(t) = \sum_{\tau=1}^{t} n_v(\tau)
          $$
        The ratio of cumulative traffic across variants should remain statistically stable and proportional to the designed
        allocation ratios. A sudden shift or step-function deviation in:
        $$
        R(t) = \frac{C_{\text{treatment}}(t)}{C_{\text{control}}(t)}
        $$
        indicates a critical operational failure (e.g., treatment servers crashing, CDN configuration issues, or regional tracking bugs).

    Pseudocode for Binning and Accumulation:
        ```text
        function get_cumulative_traffic(df, time_col, variant_col, bin_frequency):
            1. Truncate timestamps in time_col to the specified bin_frequency (e.g., 'H' for Hour, 'D' for Day).
            2. Group by binned time and variant_col, calculating count of unique units.
            3. Pivot the grouped DataFrame to have binned time as index and variant names as columns.
            4. Fill missing values with 0.
            5. Compute cumulative sums along the rows (axis=0) for each column.
            6. Return the resulting cumulative DataFrame.
        ```

    Attributes:
        df (pd.DataFrame): Raw log DataFrame containing exposure details.
        time_col (str): Name of the column containing assignment timestamps.
        dispatcher (Optional[WebhookAlertDispatcher]): Alert dispatcher.
        shutoff_triggered (bool): Status flag indicating whether an active SRM or critical alert has suspended assignment.
    """

    def __init__(self, df: pd.DataFrame, time_col: str, dispatcher: Optional[WebhookAlertDispatcher] = None):
        """Initializes a LiveMonitor.

        Args:
            df (pd.DataFrame): The experimental dataset.
            time_col (str): The column representing assignment timestamps.
            dispatcher (Optional[WebhookAlertDispatcher]): Alert dispatcher. Defaults to None.
        """
        self.df = df
        self.time_col = time_col
        self.dispatcher = dispatcher
        self.shutoff_triggered = False

    def get_cumulative_traffic(self, variant_col: str = "variant", freq: str = "D") -> pd.DataFrame:
        """Calculates cumulative traffic counts over time for each variant.

        Processes and groups timestamps, returning a cumulative summation matrix suitable
        for charting and structural allocation audits.

        Args:
            variant_col (str): Column representing treatment assignment groups. Defaults to "variant".
            freq (str): Binning frequency (e.g., "h" for Hour, "D" for Day). Defaults to "D".

        Returns:
            pd.DataFrame: A pandas DataFrame indexed by time bins, with columns representing
                variants and cells containing cumulative exposure counts.
        """
        binned = self.get_binned_traffic(variant_col=variant_col, freq=freq)
        cumulative = binned.cumsum(axis=0)
        return cumulative

    def get_binned_traffic(self, variant_col: str = "variant", freq: str = "D") -> pd.DataFrame:
        """Calculates binned (non-cumulative) traffic counts over time for each variant.

        Processes and groups timestamps, returning a non-cumulative summation matrix
        indexed by time bins with variants as columns.

        Args:
            variant_col (str): Column representing treatment assignment groups. Defaults to "variant".
            freq (str): Binning frequency (e.g., "h" for Hour, "D" for Day). Defaults to "D".

        Returns:
            pd.DataFrame: A pandas DataFrame indexed by time bins, with columns representing
                variants and cells containing binned exposure counts.
        """
        binned_df = self.df.copy()
        binned_df["binned_time"] = pd.to_datetime(binned_df[self.time_col]).dt.floor(freq)

        # Group by binned time and variant, counting unique units
        unit_col = "unit_id" if "unit_id" in binned_df.columns else binned_df.columns[0]
        grouped = binned_df.groupby(["binned_time", variant_col])[unit_col].nunique().reset_index()

        # Pivot to place variants as columns
        pivoted = grouped.pivot(index="binned_time", columns=variant_col, values=unit_col)

        # Fill missing time slots with 0
        binned = pivoted.fillna(0.0)
        return binned

    def check_traffic_anomaly(
        self,
        variant_col: str = "variant",
        freq: str = "D",
        window: int = 7,
        z_threshold: float = 3.0,
        drop_threshold: float = 0.5
    ) -> Dict[str, Any]:
        r"""Performs a Z-score anomaly test on non-cumulative temporal traffic volumes.

        Calculates the unique units per time bin, and checks if the latest bin represents
        a significant traffic drop compared to the simple moving average and standard deviation
        of the preceding window.

        ??? mathbox "Z-Score Traffic Drop Check Theory"

            Let $x_t$ be the total unique units exposed across all variants in time bin $t$.
            For a history window of size $W$, the baseline mean $\mu_T$ and sample standard deviation $\sigma_T$
            of the preceding window are computed as:
            $$
            \mu_T = \frac{1}{K} \sum_{\tau=1}^{K} x_{T-\tau}
            $$
            $$
            \sigma_T = \sqrt{\frac{1}{K-1} \sum_{\tau=1}^{K} (x_{T-\tau} - \mu_T)^2}
            $$
            where $K = \min(T-1, W)$.
            The Z-score for the latest bin $x_T$ is defined as:
            $$
            Z = \frac{x_T - \mu_T}{\sigma_T}
            $$
            If $\sigma_T > 0$ and $Z < -z_{\text{threshold}}$, a sudden traffic drop anomaly is flagged.
            If $\sigma_T = 0$, a fallback check flags an anomaly if:
            $$
            x_T < \mu_T \cdot (1 - \text{drop\_threshold})
            $$

        Args:
            variant_col (str): Column representing treatment groups.
            freq (str): Binning frequency.
            window (int): Moving window size for baseline computation.
            z_threshold (float): Z-score threshold for drop detection (must be positive).
            drop_threshold (float): Percentage threshold drop fallback if variance is 0.

        Returns:
            Dict[str, Any]: Dict containing keys:
                - anomaly_detected (bool)
                - latest_volume (float)
                - historical_mean (float)
                - historical_std (float)
                - z_score (float)
                - message (str)
        """
        binned = self.get_binned_traffic(variant_col=variant_col, freq=freq)
        if binned.empty:
            return {
                "anomaly_detected": False,
                "latest_volume": 0.0,
                "historical_mean": 0.0,
                "historical_std": 0.0,
                "z_score": 0.0,
                "message": "Traffic DataFrame is empty."
            }

        # Sum unique units across all variants per time bin
        volumes = binned.sum(axis=1).values
        n_bins = len(volumes)

        if n_bins < 3:
            return {
                "anomaly_detected": False,
                "latest_volume": float(volumes[-1]),
                "historical_mean": float(np.mean(volumes[:-1])) if n_bins > 1 else 0.0,
                "historical_std": 0.0,
                "z_score": 0.0,
                "message": "Insufficient history to stably calculate baseline standard deviation (need at least 3 bins)."
            }

        latest_volume = float(volumes[-1])
        # Historical baseline: preceding window excluding the latest bin
        k = min(n_bins - 1, window)
        baseline = volumes[-(k+1):-1]

        mean = float(np.mean(baseline))
        std = float(np.std(baseline, ddof=1))

        if std == 0.0:
            anomaly = latest_volume < mean * (1.0 - drop_threshold)
            return {
                "anomaly_detected": anomaly,
                "latest_volume": latest_volume,
                "historical_mean": mean,
                "historical_std": std,
                "z_score": 0.0,
                "message": f"Degenerate variance (std=0). Drop check fallback applied: {anomaly}."
            }

        z_score = (latest_volume - mean) / std
        anomaly = z_score < -z_threshold

        return {
            "anomaly_detected": anomaly,
            "latest_volume": latest_volume,
            "historical_mean": mean,
            "historical_std": std,
            "z_score": z_score,
            "message": f"Z-score = {z_score:.4f}. Anomaly detected: {anomaly}."
        }

    def check_live_srm(self, expected_ratios: List[float], variant_col: str = "variant") -> Dict[str, Any]:
        """Calculates cumulative chi-square p-value and SRM flagging on the latest cumulative traffic.

        Args:
            expected_ratios (List[float]): Target allocation proportions/ratios.
            variant_col (str): Column representing treatment assignment groups.

        Returns:
            Dict[str, Any]: Dict containing keys:
                - srm_detected (bool)
                - p_value (float)
                - observed_counts (List[int])
                - expected_ratios (List[float])
                - message (str)
        """
        cumulative = self.get_cumulative_traffic(variant_col=variant_col)
        if cumulative.empty:
            return {
                "srm_detected": False,
                "p_value": 1.0,
                "observed_counts": [],
                "expected_ratios": expected_ratios,
                "message": "Cumulative traffic DataFrame is empty."
            }

        cumulative = cumulative.reindex(columns=sorted(cumulative.columns))
        observed_counts = [int(x) for x in cumulative.iloc[-1].values]

        if len(observed_counts) != len(expected_ratios):
            raise ValueError(
                f"Number of observed variants ({len(observed_counts)}) does not match "
                f"number of expected ratios ({len(expected_ratios)})."
            )

        from xpyrment.validate.srm import check_srm
        from xpyrment.core.exceptions import SRMError
        from scipy import stats

        total_observed = sum(observed_counts)
        sum_ratios = sum(expected_ratios)
        expected_counts = [ratio * total_observed / sum_ratios for ratio in expected_ratios]
        if total_observed > 0:
            _, p_val = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)
        else:
            p_val = 1.0

        try:
            check_srm(observed_counts, expected_ratios)
            srm_detected = False
            message = f"No SRM detected. p-value = {p_val:.4f}"
        except SRMError as e:
            srm_detected = True
            message = str(e)

        return {
            "srm_detected": srm_detected,
            "p_value": float(p_val),
            "observed_counts": observed_counts,
            "expected_ratios": expected_ratios,
            "message": message
        }

    def check_sequential_srm(
        self,
        variant_col: str = "variant",
        target_treatment_ratio: float = 0.5,
        delta: float = 0.02,
        alpha: float = 0.01
    ) -> Dict[str, Any]:
        r"""Runs Wald's SPRT on the assignment series.

        ??? mathbox "Sequential Sample Ratio Mismatch SPRT Theory"

            Wald's Sequential Probability Ratio Test (SPRT) is applied to the assignment sequence $y_1, y_2, \dots, y_N$
            where $y_i \in \{0, 1\}$ (0 = control, 1 = treatment).
            The likelihood ratio $LR_N$ at step $N$ is calculated under a mixture of alternative hypotheses:
            $$
            LR_N = 0.5 \cdot \prod_{i=1}^N \frac{f(y_i \mid p_0 + \delta)}{f(y_i \mid p_0)} + 0.5 \cdot \prod_{i=1}^N \frac{f(y_i \mid p_0 - \delta)}{f(y_i \mid p_0)}
            $$
            where $p_0$ is the target treatment allocation ratio.
            If $LR_N \ge \frac{1}{\alpha}$, the null hypothesis of balanced allocation is rejected, indicating a sample ratio mismatch.

        Args:
            variant_col (str): Column representing treatment assignment groups.
            target_treatment_ratio (float): Target allocation ratio for the treatment group.
            delta (float): SPRT mismatch shift boundary.
            alpha (float): SPRT Type I error threshold boundary.

        Returns:
            Dict[str, Any]: Dict containing keys:
                - srm_detected (bool)
                - running_likelihood_ratios (np.ndarray)
                - stopped_index (int)
                - message (str)
        """
        unique_variants = sorted(self.df[variant_col].dropna().unique())
        if len(unique_variants) != 2:
            logger.warning("Sequential SRM (SPRT) is only supported for two-arm (binary) experiments.")
            return {
                "srm_detected": False,
                "message": "Sequential SRM SPRT is only supported for exactly two variants."
            }

        control_val = unique_variants[0]
        treatment_val = unique_variants[1]
        for v in unique_variants:
            if "ctrl" in str(v).lower() or "control" in str(v).lower():
                control_val = v
                treatment_val = [x for x in unique_variants if x != v][0]
                break

        sorted_df = self.df.dropna(subset=[self.time_col, variant_col]).sort_values(by=self.time_col)
        if sorted_df.empty:
            return {
                "srm_detected": False,
                "message": "No assignment data available."
            }

        assignments = np.where(sorted_df[variant_col] == treatment_val, 1, 0)

        from xpyrment.analyze.srm import SampleRatioMismatchDetector
        detector = SampleRatioMismatchDetector(target_treatment_ratio=target_treatment_ratio)
        res = detector.test_sequential(assignments, delta=delta, alpha=alpha)

        return {
            "srm_detected": bool(res["srm_detected"]),
            "running_likelihood_ratios": res["running_likelihood_ratios"],
            "stopped_index": int(res["stopped_index"]),
            "message": f"Sequential SPRT finished. SRM detected: {res['srm_detected']} at index {res['stopped_index']}."
        }

    def run_telemetry_checks(
        self,
        expected_ratios: List[float],
        variant_col: str = "variant",
        freq: str = "D",
        window: int = 7,
        z_threshold: float = 3.0
    ) -> Dict[str, Any]:
        """Runs traffic anomaly, cumulative SRM, and sequential SRM checks, dispatching alerts if needed.

        Args:
            expected_ratios (List[float]): Target allocation proportions/ratios.
            variant_col (str): Column representing treatment assignment groups.
            freq (str): Binning frequency.
            window (int): Moving window size for traffic anomaly check.
            z_threshold (float): Z-score threshold for drop detection.

        Returns:
            Dict[str, Any]: Combined results of all checks.
        """
        anomaly_res = self.check_traffic_anomaly(
            variant_col=variant_col, freq=freq, window=window, z_threshold=z_threshold
        )

        live_srm_res = self.check_live_srm(expected_ratios=expected_ratios, variant_col=variant_col)

        seq_srm_res = {"srm_detected": False, "message": "SPRT bypassed (unsupported multi-arm)."}
        unique_variants = sorted(self.df[variant_col].dropna().unique())
        if len(unique_variants) == 2:
            target_treatment_ratio = expected_ratios[1] / sum(expected_ratios)
            seq_srm_res = self.check_sequential_srm(
                variant_col=variant_col,
                target_treatment_ratio=target_treatment_ratio
            )

        alerts_triggered = []

        if anomaly_res["anomaly_detected"]:
            alerts_triggered.append("TRAFFIC_DROP_ANOMALY")
            if self.dispatcher:
                self.dispatcher.dispatch(
                    alert_type="TRAFFIC_DROP_ANOMALY",
                    message=f"Traffic dropout anomaly detected in the latest bin! {anomaly_res['message']}",
                    payload=anomaly_res
                )

        if live_srm_res["srm_detected"]:
            alerts_triggered.append("SRM_SHUTOFF_TRIGGERED")
            self.shutoff_triggered = True
            if self.dispatcher:
                self.dispatcher.dispatch(
                    alert_type="SRM_SHUTOFF_TRIGGERED",
                    message=f"Critical Sample Ratio Mismatch (SRM) detected! Shutoff triggered. {live_srm_res['message']}",
                    payload=live_srm_res
                )

        if seq_srm_res.get("srm_detected", False):
            alerts_triggered.append("SEQUENTIAL_SRM_TRIGGERED")
            self.shutoff_triggered = True
            if self.dispatcher:
                self.dispatcher.dispatch(
                    alert_type="SEQUENTIAL_SRM_TRIGGERED",
                    message=f"Sequential SPRT Sample Ratio Mismatch detected! Shutoff triggered. {seq_srm_res['message']}",
                    payload=seq_srm_res
                )

        return {
            "shutoff_triggered": self.shutoff_triggered,
            "alerts_triggered": alerts_triggered,
            "traffic_anomaly": anomaly_res,
            "cumulative_srm": live_srm_res,
            "sequential_srm": seq_srm_res
        }
