import numpy as np
import pandas as pd
import pytest
from xpyrment.run.assignment import AssignmentLogger
from xpyrment.run.ingestion import ingest_dataframe, load_from_sql
from xpyrment.run.monitor import LiveMonitor
from xpyrment.run.stopping import StoppingRules
from xpyrment.plan.duration import estimate_duration_days


def test_assignment_logger_deduplication():
    """Verifies that AssignmentLogger correctly implements first-touch attribution."""
    logger = AssignmentLogger()

    # Log user exposure multiple times with varying timestamps and variants
    logger.log_assignment("user_1", "control", "2026-05-01T10:00:00")
    logger.log_assignment("user_1", "treatment", "2026-05-01T12:00:00")  # Later peek
    logger.log_assignment("user_2", "treatment", "2026-05-01T09:00:00")
    logger.log_assignment("user_2", "treatment", "2026-05-01T11:00:00")

    df = logger.get_deduplicated_exposures()

    # user_1 should have control (earliest), user_2 should have treatment
    assert len(df) == 2
    user1_row = df[df["unit_id"] == "user_1"].iloc[0]
    user2_row = df[df["unit_id"] == "user_2"].iloc[0]

    assert user1_row["variant"] == "control"
    assert user1_row["timestamp"] == "2026-05-01T10:00:00"

    assert user2_row["variant"] == "treatment"
    assert user2_row["timestamp"] == "2026-05-01T09:00:00"


def test_ingest_dataframe_auditing():
    """Tests the primary key, datetime conversion, and missing imputation gates."""
    raw_df = pd.DataFrame({
        "user_id": ["u1", None, "u3"],  # u2 is missing
        "joined_at": ["2026-05-01", "2026-05-02", "2026-05-03"],
        "revenue": [50.0, np.nan, 20.0],
        "browser": ["Chrome", "Safari", np.nan]
    })

    # Ingest with auditing rules
    clean_df = ingest_dataframe(
        raw_df,
        unit_id_col="user_id",
        time_col="joined_at",
        metric_cols=["revenue"],
        categorical_cols=["browser"]
    )

    # 1. Null unit_id row dropped -> length is 2
    assert len(clean_df) == 2
    assert "u1" in clean_df["user_id"].values
    assert "u3" in clean_df["user_id"].values

    # 2. Chronological Alignment -> datetime64 type
    assert pd.api.types.is_datetime64_any_dtype(clean_df["joined_at"])

    # 3. Continuous metric NaN imputed to 0.0, categorical imputed to UNKNOWN
    u3_row = clean_df[clean_df["user_id"] == "u3"].iloc[0]
    assert u3_row["revenue"] == 20.0
    assert u3_row["browser"] == "UNKNOWN"


def test_live_monitor_cumulative_traffic():
    """Tests temporal binning and cumulative sums generation for monitoring traffic shifts."""
    raw_df = pd.DataFrame({
        "unit_id": ["u1", "u2", "u3", "u4", "u5"],
        "exposed_at": [
            "2026-05-01 10:00:00",
            "2026-05-01 14:00:00",
            "2026-05-02 08:00:00",
            "2026-05-02 12:00:00",
            "2026-05-02 15:00:00"
        ],
        "group": ["A", "B", "A", "A", "B"]
    })

    monitor = LiveMonitor(raw_df, time_col="exposed_at")
    
    # Compute cumulative daily counts
    cum_df = monitor.get_cumulative_traffic(variant_col="group", freq="D")

    # Result index should be Days, columns should be A and B
    assert len(cum_df) == 2
    assert "A" in cum_df.columns
    assert "B" in cum_df.columns

    # Day 1: A=1, B=1. Cumulative: A=1, B=1
    # Day 2: A=2, B=1. Cumulative: A=3, B=2
    assert cum_df.loc["2026-05-01", "A"] == 1
    assert cum_df.loc["2026-05-01", "B"] == 1
    assert cum_df.loc["2026-05-02", "A"] == 3
    assert cum_df.loc["2026-05-02", "B"] == 2


def test_stopping_rules_msprt():
    """Tests mSPRT sequential testing likelihood calculation and stopping decisions."""
    rules = StoppingRules(alpha=0.01)

    # 1. Assert boundaries check: stopping threshold is 1 / alpha = 100.0
    assert rules.check_msprt_stop(150.0) is True
    assert rules.check_msprt_stop(50.0) is False

    # 2. Test mathematical calculation of the martingale likelihood ratio (Lambda_n)
    # Under H0 (mean_diff = 0.0), Lambda should be strictly less than 1.0 (actually term1 < 1.0)
    lambda_h0 = rules.calculate_msprt_lambda(n=1000, mean_diff=0.0, variance=10.0, tau=0.5)
    assert lambda_h0 < 1.0

    # With a significant mean difference, Lambda should be very large, crossing the boundary
    lambda_h1 = rules.calculate_msprt_lambda(n=1000, mean_diff=1.2, variance=10.0, tau=0.5)
    assert lambda_h1 > 100.0
    assert rules.check_msprt_stop(lambda_h1) is True


def test_estimate_duration_days():
    """Tests duration estimations based on traffic and sample size requirements."""
    assert estimate_duration_days(50000, 5000) == 10.0
    assert estimate_duration_days(50000.0, 5000.0) == 10.0

    with pytest.raises(ValueError):
        estimate_duration_days(0, 5000)

    with pytest.raises(ValueError):
        estimate_duration_days(50000, 0)

    with pytest.raises(TypeError):
        estimate_duration_days("50000", 5000)

    with pytest.raises(TypeError):
        estimate_duration_days(50000, "5000")


def test_load_from_sql_mock():
    """Tests load_from_sql integration (using sqlite3 memory mock)."""
    # Simply verify it returns a dataframe for a valid query
    df = load_from_sql('SELECT 1 as id', 'sqlite:///:memory:')
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]['id'] == 1
